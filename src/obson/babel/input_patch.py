"""Endpoint-only, ordered four-bar input grouping with a matched stride control.

The public input is always the endpoint's own observed128 rows. ``local4`` and
``patch4`` share the exact projection; they differ only in which endpoint tokens
enter the causal backbone. No intermediate token is a public full-context state.
"""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from . import history_query as hq
from . import observed_history as original
from . import state_coverage as coverage
from . import uniform_context
from .history_autoencoder import positions as positional_encoding

SCHEMA = "babel-input-patch-model-v1"
ARMS = ("bar", "local4", "patch4")
PATCH_LENGTH = 4
INPUT_SEED_OFFSET = 20261025
CONFIG = dict(original.CONFIG)
VAL_POSITIONS = original.VAL_POSITIONS
targets = original.targets
rolling_windows = original.rolling_windows


class Encoder(uniform_context.Encoder):
    def __init__(self, arm, seed, config=None):
        expected = contract(arm, config)
        # Construct the unchanged base first. Replacing the input afterwards
        # must not advance the RNG stream used for backbone/coordinates/query.
        super().__init__("rolling", seed, expected["encoder"])
        self.arm = arm
        self.attention_mode = "causal"
        self.patch_length = 1 if arm == "bar" else PATCH_LENGTH
        self.token_stride = PATCH_LENGTH if arm == "patch4" else 1
        if arm != "bar":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + INPUT_SEED_OFFSET)
                self.input = nn.Linear(28 * PATCH_LENGTH, self.config["width"])

    def token_endpoints(self, device=None):
        """Original zero-based bar positions, not compressed token ordinals."""
        return torch.arange(self.token_stride - 1, 128, self.token_stride, device=device)

    def input_rows(self, x):
        if x.ndim != 3 or x.shape[1:] != (128, 28) or not len(x):
            raise ValueError("Only nonempty B x128 x28 observed windows required")
        if self.arm == "bar":
            return x
        # Missing earlier rows mean standardized-feature zero, never extra
        # pre-window bars. Each flattened row is oldest bar first, feature last.
        padded = functional.pad(x, (0, 0, PATCH_LENGTH - 1, 0))
        packed = padded.unfold(1, PATCH_LENGTH, 1).permute(0, 1, 3, 2).flatten(2)
        return packed[:, self.token_stride - 1 :: self.token_stride]

    def input_tokens(self, x, *, include_positions=True):
        h = self.input(self.input_rows(x))
        if include_positions:
            pe = positional_encoding(128, h.shape[-1], h.device, h.dtype)
            h = h + pe[self.token_endpoints(h.device)]
        return h

    def forward(self, x):
        h = self.input_tokens(x)
        for layer, window in zip(self.layers, uniform_context.WINDOWS, strict=True):
            mask = self.attention_mask(h.shape[1], window, len(h), h)
            if self.checkpointing and torch.is_grad_enabled():
                h = checkpoint(
                    lambda value, layer=layer, mask=mask: layer(value, src_mask=mask),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(h, src_mask=mask)
        # Preserve original bar projection/GEMM layout before endpoint selection.
        return self.coordinates(h)[:, -1]


def contract(arm, config=None):
    """Canonical metadata without constructing parameters or executing a model."""
    if arm not in ARMS:
        raise ValueError(f"Expected a declared arm from {ARMS}")
    c = dict(CONFIG if config is None else config)
    if (
        set(c) != set(CONFIG)
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in c.values())
        or c["layers"] != 4
        or c["width"] % 16
        or c["width"] % c["heads"]
    ):
        raise ValueError(
            "Four layers, positive dimensions, width divisible by16 and heads required"
        )
    length = 1 if arm == "bar" else PATCH_LENGTH
    stride = PATCH_LENGTH if arm == "patch4" else 1
    return {
        "schema": SCHEMA,
        "arm": arm,
        "attention_mode": "causal",
        "scope": "closed_endpoint_observed_trailing128",
        "endpoint_only": True,
        "input_length": 128,
        "input_channels": 28,
        "state_width": c["width"],
        "encoder": c,
        "patch_length": length,
        "token_stride": stride,
        "token_count": 128 // stride,
        "token_endpoints_zero_based": list(range(stride - 1, 128, stride)),
        "position_coordinates": "original_zero_based_bar_endpoints",
        "input_order": "oldest_bar_first_then_all28_features",
        "padding": "three_standardized_zero_rows_inside_window_only" if arm == "local4" else "none",
        "internal_tokens_are_public_bar_states": False,
        "early_tokens_have_complete_own128": False,
        "state_scaler": "fixed_zero_mean_unit_scale_no_refit",
        "raw_or_target_decoder_bypass": False,
        "additional_parameters_vs_bar": 28 * (length - 1) * c["width"],
        "checkpoint_semantics": "arm_and_stride_must_be_bound_separately_from_identical_local4_patch4_tensor_keys",
    }


def model_contract(encoder):
    if not isinstance(encoder, Encoder):
        raise ValueError("Input-patch encoder required")
    expected = contract(encoder.arm, encoder.config)
    if (
        encoder.attention_mode != "causal"
        or encoder.mode != "rolling"
        or len(encoder.layers) != 4
        or encoder.patch_length != expected["patch_length"]
        or encoder.token_stride != expected["token_stride"]
        or encoder.input.in_features != 28 * encoder.patch_length
        or encoder.input.out_features != encoder.config["width"]
    ):
        raise ValueError("Input-patch encoder and matching input/stride semantics required")
    return expected


def make_models(arm, seed, device, config=None):
    encoder = Encoder(arm, seed, config).to(device)
    width = encoder.config["width"]
    qc = {"slots": 4, "width": width // 4, "heads": 4, "layers": 2, "ff": width // 2}
    query = hq.HistoryQuery(
        width, {"mean": [0.0] * width, "scale": [1.0] * width}, seed, config=qc
    ).to(device)
    return encoder, query


def read_states(encoder, x, positions):
    model_contract(encoder)
    windows = rolling_windows(x, positions)
    state = encoder(windows.flatten(0, 1))
    return state.reshape(len(x), positions.shape[1], encoder.config["width"])


def predict(encoder, query, x, positions):
    state = read_states(encoder, x, positions)
    return state, query(state.flatten(0, 1)).reshape(len(x), positions.shape[1], 127, 7)


def _tensor_signature(pair, *, exclude_input=False):
    digest = hashlib.sha256()
    for family, network in zip(("encoder", "query"), pair, strict=True):
        for name, tensor in network.state_dict().items():
            if exclude_input and family == "encoder" and name.startswith("input."):
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{family}/{name}:{array.dtype}:{array.shape}\n".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def compare_initialized_pairs(pairs):
    """Check supplied scratch pairs; the caller owns their provenance/locking.

    Equality alone does not prove scratch provenance. Use this before ANY update
    and bind it to the fresh-construction and empty-Adam receipts in the runner.
    """
    if set(pairs) != set(ARMS):
        raise ValueError("All three declared scratch pairs required")
    for arm, pair in pairs.items():
        if model_contract(pair[0])["arm"] != arm:
            raise ValueError("Scratch pair key and model arm differ")
    signatures = {arm: _tensor_signature(pair) for arm, pair in pairs.items()}
    common = {arm: _tensor_signature(pair, exclude_input=True) for arm, pair in pairs.items()}
    counts = {
        arm: {
            "encoder": sum(p.numel() for p in pair[0].parameters()),
            "query": sum(p.numel() for p in pair[1].parameters()),
            "input": sum(p.numel() for p in pair[0].input.parameters()),
        }
        for arm, pair in pairs.items()
    }
    checks = {
        "common_backbone_coordinates_query_equal": len(set(common.values())) == 1,
        "local4_patch4_full_tensors_equal": signatures["local4"] == signatures["patch4"],
        "query_scalers_zero_one": all(
            not pair[1].state_mean.count_nonzero().item() and bool((pair[1].state_scale == 1).all())
            for pair in pairs.values()
        ),
        "parameter_difference_only_input": all(
            counts[arm]["encoder"] - counts["bar"]["encoder"]
            == counts[arm]["input"] - counts["bar"]["input"]
            == model_contract(pairs[arm][0])["additional_parameters_vs_bar"]
            for arm in ARMS
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "signatures": signatures,
        "common_signatures": common,
        "parameter_counts": counts,
        "limits": "Shared tensors equal; different input functions are not function-preserving initialization",
    }


def initialization_receipt(seed, device, config=None):
    """Fresh models only, no checkpoint loading or train-statistics fitting."""
    pairs = {arm: make_models(arm, seed, device, config) for arm in ARMS}
    report = compare_initialized_pairs(pairs)
    report["seed"] = int(seed)
    report["arms"] = {
        arm: {
            "signature": report["signatures"][arm],
            "common_signature": report["common_signatures"][arm],
            "parameter_count": sum(p.numel() for net in pair for p in net.parameters()),
            "parameter_bytes": sum(
                p.numel() * p.element_size() for net in pair for p in net.parameters()
            ),
            "model_contract": model_contract(pair[0]),
        }
        for arm, pair in pairs.items()
    }
    return report


@torch.no_grad()
def audit(encoder, query, x, statistics):
    """Endpoint/context and packing checks; neither raw lineage nor efficacy."""
    encoder.eval()
    query.eval()
    contract = model_contract(encoder)
    ps = torch.tensor([1, 2, 3, 4, 31, 32, 127, 128], device=x.device).expand(len(x), -1)
    y, mask = targets(x, ps, statistics)
    oracle, oracle_mask = coverage.oracle_targets(x.cpu().numpy(), ps.cpu().numpy(), statistics)
    z, pred = predict(encoder, query, x, ps)
    count = contract["token_count"]
    graph = encoder.attention_mask(count, 33, len(x), x)
    expected = torch.triu(torch.ones(count, count, dtype=torch.bool, device=x.device), diagonal=1)
    checks = {
        "independent_labels": bool(np.allclose(y.cpu().numpy(), oracle, atol=1e-5, rtol=2e-5)),
        "independent_masks": bool(np.array_equal(mask.cpu().numpy(), oracle_mask)),
        "declared_attention_graph": bool(
            torch.equal(graph, expected) and graph.dtype == torch.bool
        ),
        "endpoint_only": z.shape == (len(x), ps.shape[1], encoder.config["width"]),
    }
    observed = x[:, :128]
    raw = observed.cpu().numpy()
    ends = encoder.token_endpoints().tolist()
    if encoder.arm == "bar":
        packed_oracle = raw
    else:
        packed_oracle = np.zeros((len(x), len(ends), 4 * 28), dtype=raw.dtype)
        for token, end in enumerate(ends):
            for slot, row in enumerate(range(end - 3, end + 1)):
                if row >= 0:
                    packed_oracle[:, token, slot * 28 : (slot + 1) * 28] = raw[:, row]
    checks["independent_ordered_packing"] = bool(
        np.array_equal(encoder.input_rows(observed).cpu().numpy(), packed_oracle)
    )
    h = encoder.input_tokens(observed, include_positions=False)
    expected_tokens = h + positional_encoding(128, h.shape[-1], h.device, h.dtype)[ends]
    checks["original_bar_positions"] = bool(
        torch.equal(encoder.input_tokens(observed), expected_tokens)
    )
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
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "model_contract": contract,
        "attention_graph": {
            "mask_shape": list(graph.shape),
            "allowed_edges_per_layer": int((~graph).sum()),
        },
        "limits": [
            "Only each full128 endpoint is public; no patch-interior causal state claim",
            "Prepared features still contain causal EMA prehistory",
            "Feature-label oracle is separate from original float64 close truth",
        ],
    }
