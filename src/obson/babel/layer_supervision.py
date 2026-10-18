"""Causal endpoint encoder with a training-only second-block readout.

Both arms retain the exact observed-history control parameter layout and public
final-state interface. The auxiliary state uses the existing coordinates layer
and query head; it is not an additional output channel or persistent memory.
"""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from . import history_query as hq
from . import observed_history as original
from . import state_coverage as coverage
from . import uniform_context
from .history_autoencoder import positions as positional_encoding

SCHEMA = "babel-layer-supervision-model-v1"
ARMS = ("control", "layer2")
ATTENTION_MODES = dict.fromkeys(ARMS, "causal")
CONFIG = dict(original.CONFIG)
VAL_POSITIONS = original.VAL_POSITIONS
MIDDLE_BLOCK = 2
targets = original.targets
rolling_windows = original.rolling_windows


class Encoder(original.Encoder):
    """Same trainable tensors/order as the original causal control."""

    def __init__(self, arm, seed, config=None):
        if arm not in ARMS:
            raise ValueError(f"Expected a declared arm from {ARMS}")
        super().__init__("control", seed, config)
        self.arm = arm
        self.attention_mode = "causal"

    def attention_mask(self, length, window, batch, ref):
        # The arm selects a training objective, never a different attention graph.
        return uniform_context.Encoder.attention_mask(self, length, window, batch, ref)

    def forward_levels(self, x):
        """Return (final, block2) endpoint states without hooks or cached tensors.

        coordinates is applied to the complete hidden arrays before endpoint
        selection, preserving the original final forward's matrix layout. This
        explicit path also makes checkpoint recomputation independent of mutable
        hook state. All128 inputs end at the requested closed bar.
        """
        if x.ndim != 3 or x.shape[1:] != (128, 28) or not len(x):
            raise ValueError("Only nonempty B x128 x28 observed windows required")
        h = self.input(x)
        h = h + positional_encoding(128, h.shape[-1], h.device, h.dtype)
        middle = None
        for index, (layer, window) in enumerate(
            zip(self.layers, uniform_context.WINDOWS, strict=True), start=1
        ):
            mask = self.attention_mask(h.shape[1], window, len(h), h)
            if self.checkpointing and torch.is_grad_enabled():
                h = checkpoint(
                    lambda value, layer=layer, mask=mask: layer(value, src_mask=mask),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(h, src_mask=mask)
            if index == MIDDLE_BLOCK:
                middle = h
        final = self.coordinates(h)[:, -1]
        auxiliary = self.coordinates(middle)[:, -1]
        return final, auxiliary


def model_contract(encoder):
    if (
        not isinstance(encoder, Encoder)
        or encoder.arm not in ARMS
        or encoder.attention_mode != "causal"
        or encoder.mode != "rolling"
        or len(encoder.layers) != 4
    ):
        raise ValueError("Layer-supervision encoder and matching causal semantics required")
    return {
        "schema": SCHEMA,
        "arm": encoder.arm,
        "attention_mode": "causal",
        "scope": "closed_endpoint_observed_trailing128",
        "endpoint_only": True,
        "input_length": 128,
        "input_channels": 28,
        "state_width": encoder.config["width"],
        "encoder": dict(encoder.config),
        "internal_tokens_are_public_bar_states": False,
        "internal_attention_is_causal": True,
        "early_tokens_have_complete_own128": False,
        "public_output": "final_block_only",
        "training_auxiliary_block": MIDDLE_BLOCK,
        "shared_coordinates": True,
        "additional_parameters": 0,
        "checkpoint_semantics": "arm must be bound externally; tensor keys do not identify objective",
    }


def make_models(arm, seed, device, config=None):
    encoder = Encoder(arm, seed, config).to(device)
    width = encoder.config["width"]
    qc = {"slots": 4, "width": width // 4, "heads": 4, "layers": 2, "ff": width // 2}
    query = hq.HistoryQuery(
        width, {"mean": [0.0] * width, "scale": [1.0] * width}, seed, config=qc
    ).to(device)
    return encoder, query


def read_states(encoder, x, positions):
    """Public inference returns the final state only."""
    model_contract(encoder)
    windows = rolling_windows(x, positions)
    states = encoder(windows.flatten(0, 1))
    return states.reshape(len(x), positions.shape[1], encoder.config["width"])


def read_state_levels(encoder, x, positions):
    """Training-only final and middle states from identical own128 windows."""
    model_contract(encoder)
    windows = rolling_windows(x, positions)
    final, middle = encoder.forward_levels(windows.flatten(0, 1))
    shape = (len(x), positions.shape[1], encoder.config["width"])
    return final.reshape(shape), middle.reshape(shape)


def predict(encoder, query, x, positions):
    states = read_states(encoder, x, positions)
    return states, query(states.flatten(0, 1)).reshape(len(x), positions.shape[1], 127, 7)


@torch.no_grad()
def audit(encoder, query, x, statistics):
    """Check interface and prepared-row causality; no efficacy claim."""
    encoder.eval()
    query.eval()
    contract = model_contract(encoder)
    ps = torch.tensor([1, 2, 31, 32, 127, 128], device=x.device).expand(len(x), -1)
    y, mask = targets(x, ps, statistics)
    oracle, oracle_mask = coverage.oracle_targets(x.cpu().numpy(), ps.cpu().numpy(), statistics)
    checks = {
        "independent_labels": bool(np.allclose(y.cpu().numpy(), oracle, atol=1e-5, rtol=2e-5)),
        "independent_masks": bool(np.array_equal(mask.cpu().numpy(), oracle_mask)),
    }
    z, pred = predict(encoder, query, x, ps)
    final, middle = read_state_levels(encoder, x, ps)
    checks["public_final_state_parity"] = bool(torch.equal(z, final))
    checks["training_middle_shape"] = middle.shape == z.shape
    graph = encoder.attention_mask(128, uniform_context.WINDOWS[0], len(x), x)
    expected = torch.triu(torch.ones(128, 128, dtype=torch.bool, device=x.device), diagonal=1)
    checks["declared_attention_graph"] = bool(
        graph.shape == (128, 128)
        and graph.dtype == torch.bool
        and graph.device == x.device
        and torch.equal(graph, expected)
    )
    checks["endpoint_only"] = z.shape == (len(x), ps.shape[1], encoder.config["width"])
    for length in (127, 129, 255):
        rejected = False
        try:
            encoder(x[:, :length])
        except ValueError:
            rejected = True
        checks[f"reject_input_length_{length}"] = rejected
    for j, endpoint in enumerate(ps[0].tolist()):
        single = encoder(x[:, endpoint - 1 : endpoint + 127])
        checks[f"rolling_{endpoint}"] = bool(torch.allclose(z[:, j], single, atol=1e-4, rtol=2e-4))
        baseline = read_state_levels(encoder, x, ps[:, j : j + 1])
        changed = x.clone()
        changed[:, endpoint + 127 :] += 0.7
        future = read_state_levels(encoder, changed, ps[:, j : j + 1])
        cy, cm = targets(changed, ps[:, j : j + 1], statistics)
        checks[f"future_{endpoint}"] = bool(
            all(torch.equal(a, b) for a, b in zip(baseline, future, strict=True))
            and torch.equal(y[:, j : j + 1], cy)
            and torch.equal(mask[:, j : j + 1], cm)
        )
        changed = x.clone()
        changed[:, : endpoint - 1] += 0.7
        outside = read_state_levels(encoder, changed, ps[:, j : j + 1])
        checks[f"prepared_rows_before_window_{endpoint}"] = all(
            torch.equal(a, b) for a, b in zip(baseline, outside, strict=True)
        )
        checks[f"query_batch_{endpoint}"] = bool(
            torch.allclose(query(z[:, j]), pred[:, j], atol=1e-4, rtol=2e-4)
        )
    ages = torch.tensor([1, 20, 64, 127], device=x.device)
    checks["query_subset"] = bool(
        torch.allclose(query(z[:, -1], ages), pred[:, -1, ages - 1], atol=1e-4, rtol=2e-4)
    )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "label_max_abs": float(np.max(np.abs(y.cpu().numpy() - oracle))),
        "model_contract": contract,
        "attention_graph": {
            "mask_dtype": str(graph.dtype),
            "mask_shape": list(graph.shape),
            "mask_device": str(graph.device),
            "allowed_edges_per_layer": int((~graph).sum()),
            "layers": len(encoder.layers),
            "query_rows_can_read_later_observed_tokens": False,
        },
        "limits": [
            "The public interface uses only the final state; block2 is a training readout",
            "Prepared-row perturbations do not erase earlier EMA prehistory",
            "Feature-derived labels are separate from raw-close training/evaluation truth",
        ],
    }
