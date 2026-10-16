"""F01/F05: endpoint states with two graphs over the same observed128 bars.

Full attention is causal only with respect to the WINDOW ENDPOINT. Its internal
early tokens are not causal per-bar states and are deliberately not returned by
the public encoder. Every queried endpoint is encoded from its own complete
trailing window; a bidirectional255 pass is never substituted for that work.
"""

import numpy as np
import torch

from . import history_query as hq
from . import state_coverage as coverage
from . import uniform_context as original

SCHEMA = "babel-observed-history-model-v1"
ARMS = ("control", "full_history")
ATTENTION_MODES = {"control": "causal", "full_history": "full_observed"}
CONFIG = dict(original.CONFIG)
VAL_POSITIONS = coverage.VAL_POSITIONS
targets = coverage.targets


class Encoder(original.Encoder):
    """Same scratch tensors as rolling Encoder, but endpoint-only public output."""

    def __init__(self, arm, seed, config=None):
        if arm not in ARMS:
            raise ValueError(f"Expected a declared arm from {ARMS}")
        super().__init__("rolling", seed, config)
        self.arm = arm
        self.attention_mode = ATTENTION_MODES[arm]

    def attention_mask(self, length, window, batch, ref):
        if self.arm == "control":
            return super().attention_mask(length, window, batch, ref)
        # Explicit same-size/device/dtype mask in both arms. None would also
        # expose every observed token but could select a different fast path.
        return torch.zeros((length, length), device=ref.device, dtype=torch.bool)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (128, 28) or not len(x):
            raise ValueError(
                "Only nonempty B x128 x28 observed windows; no padding or extra history"
            )
        return super().forward(x)[:, -1]


def model_contract(encoder):
    if (
        not isinstance(encoder, Encoder)
        or encoder.arm not in ARMS
        or encoder.attention_mode != ATTENTION_MODES[encoder.arm]
        or encoder.mode != "rolling"
    ):
        raise ValueError("Observed-history encoder and matching attention semantics required")
    return {
        "schema": SCHEMA,
        "arm": encoder.arm,
        "attention_mode": encoder.attention_mode,
        "scope": "closed_endpoint_observed_trailing128",
        "endpoint_only": True,
        "input_length": 128,
        "input_channels": 28,
        "state_width": encoder.config["width"],
        "encoder": dict(encoder.config),
        "internal_tokens_are_causal_bar_states": encoder.arm == "control",
        "checkpoint_semantics": "arm must be bound externally; tensor keys do not identify attention",
    }


def make_models(arm, seed, device, config=None):
    encoder = Encoder(arm, seed, config).to(device)
    width = encoder.config["width"]
    qc = {"slots": 4, "width": width // 4, "heads": 4, "layers": 2, "ff": width // 2}
    query = hq.HistoryQuery(
        width, {"mean": [0.0] * width, "scale": [1.0] * width}, seed, config=qc
    ).to(device)
    return encoder, query


def rolling_windows(x, positions):
    if (
        x.ndim != 3
        or x.shape[1:] != (255, 28)
        or not len(x)
        or positions.ndim != 2
        or len(positions) != len(x)
        or positions.shape[1] == 0
        or positions.dtype != torch.long
        or positions.device != x.device
        or not ((positions >= 1) & (positions <= 128)).all()
    ):
        raise ValueError(
            "Nonempty255-bar inputs and integer endpoints1..128 on the same device required"
        )
    return original.rolling_windows(x, positions)


def read_states(encoder, x, positions):
    model_contract(encoder)
    windows = rolling_windows(x, positions)
    states = encoder(windows.flatten(0, 1))
    return states.reshape(len(x), positions.shape[1], encoder.config["width"])


def predict(encoder, query, x, positions):
    states = read_states(encoder, x, positions)
    return states, query(states.flatten(0, 1)).reshape(len(x), positions.shape[1], 127, 7)


@torch.no_grad()
def audit(encoder, query, x, statistics):
    """Synthetic/real input checks, not a claim of market efficacy or raw lineage.

    Left-outside perturbation concerns already prepared feature rows. The causal
    EMA features inside the window may still depend on earlier contract history.
    Independent label checks here retain the inherited feature-derived target;
    the training layer separately replaces price with original float64 closes.
    """
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
    graph = encoder.attention_mask(128, original.WINDOWS[0], len(x), x)
    expected = torch.triu(torch.ones(128, 128, dtype=torch.bool, device=x.device), diagonal=1)
    if encoder.arm == "full_history":
        expected.zero_()
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
        # Compare causal perturbations through the identical gather/layout and
        # batch shape. The strided single-window path above is a separate
        # numerical parity check, not an exact-bit dependency check.
        baseline = read_states(encoder, x, ps[:, j : j + 1])[:, 0]
        changed = x.clone()
        changed[:, endpoint + 127 :] += 0.7
        future = read_states(encoder, changed, ps[:, j : j + 1])[:, 0]
        cy, cm = targets(changed, ps[:, j : j + 1], statistics)
        checks[f"future_{endpoint}"] = bool(
            torch.equal(baseline, future)
            and torch.equal(y[:, j : j + 1], cy)
            and torch.equal(mask[:, j : j + 1], cm)
        )
        changed = x.clone()
        changed[:, : endpoint - 1] += 0.7
        outside = read_states(encoder, changed, ps[:, j : j + 1])[:, 0]
        checks[f"prepared_rows_before_window_{endpoint}"] = bool(torch.equal(baseline, outside))
        checks[f"query_batch_{endpoint}"] = bool(
            torch.allclose(query(z[:, j]), pred[:, j], atol=1e-4, rtol=2e-4)
        )
    ages = torch.tensor([1, 20, 64, 127], device=x.device)
    checks["query_subset"] = bool(
        torch.allclose(query(z[:, -1], ages), pred[:, -1, ages - 1], atol=1e-4, rtol=2e-4)
    )
    oldy, oldm = original.targets(x, ps[:, 3:], statistics)
    checks["old_targets"] = bool(torch.equal(y[:, 3:], oldy) and torch.equal(mask[:, 3:], oldm))
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
            "query_rows_can_read_later_observed_tokens": encoder.arm == "full_history",
        },
        "limits": [
            "No guarantee from intermediate full-attention tokens; only window endpoint is public",
            "Prepared-row perturbations do not erase earlier EMA prehistory",
            "Feature-derived label oracle is separate from raw-close training and evaluation truth",
        ],
    }
