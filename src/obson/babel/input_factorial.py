"""Recent single-bar input and old ordered context: organization x initialization.

Every public state uses its own observed128 rows. All arms retain recent32
individual bars. Only old96 token density and context-projection initialization
vary. Endpoint initialization preserves the original bar function for dense,
not hybrid; neither arm preserves the original bar's later training dynamics.
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

SCHEMA = "babel-input-factorial-model-v1"
ARMS = ("dense_random", "hybrid_random", "dense_endpoint", "hybrid_endpoint")
SEEDS = (42, 43)
PATCH_LENGTH = 4
RECENT_LENGTH = 32
OLD_LENGTH = 96
INPUT_SEED_OFFSET = 20261026
CONFIG = dict(original.CONFIG)
VAL_POSITIONS = original.VAL_POSITIONS
targets = original.targets
rolling_windows = original.rolling_windows


class Encoder(uniform_context.Encoder):
    def __init__(self, arm, seed, config=None):
        expected = contract(arm, config)
        # The base factory preserves original single/bar, trunk and query RNG.
        super().__init__("rolling", seed, expected["encoder"])
        self.arm = arm
        self.attention_mode = "causal"
        self.structure, self.initialization = arm.split("_")
        self.patch_length = PATCH_LENGTH
        self.recent_length = RECENT_LENGTH
        self.old_length = OLD_LENGTH
        self.token_stride = expected["old_token_stride"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + INPUT_SEED_OFFSET)
            self.context_input = nn.Linear(28 * PATCH_LENGTH, self.config["width"])
        if self.initialization == "endpoint":
            with torch.no_grad():
                self.context_input.weight.zero_()
                self.context_input.weight[:, -28:].copy_(self.input.weight)
                self.context_input.bias.copy_(self.input.bias)

    def token_endpoints(self, device=None):
        return torch.cat(
            (
                torch.arange(self.token_stride - 1, OLD_LENGTH, self.token_stride, device=device),
                torch.arange(OLD_LENGTH, 128, device=device),
            )
        )

    def input_rows(self, x):
        if x.ndim != 3 or x.shape[1:] != (128, 28) or not len(x):
            raise ValueError("Only nonempty B x128 x28 observed windows required")
        # Build context only inside old96; no pack crosses into recent32.
        padded = functional.pad(x[:, :OLD_LENGTH], (0, 0, PATCH_LENGTH - 1, 0))
        packed = padded.unfold(1, PATCH_LENGTH, 1).permute(0, 1, 3, 2).flatten(2)
        return {
            "context": packed[:, self.token_stride - 1 :: self.token_stride],
            "single": x[:, OLD_LENGTH:],
        }

    def input_tokens(self, x, *, include_positions=True):
        rows = self.input_rows(x)
        h = torch.cat((self.context_input(rows["context"]), self.input(rows["single"])), 1)
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
        return self.coordinates(h)[:, -1]


def contract(arm, config=None):
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
    structure, initialization = arm.split("_")
    stride = 4 if structure == "hybrid" else 1
    return {
        "schema": SCHEMA,
        "arm": arm,
        "structure": structure,
        "initialization": initialization,
        "attention_mode": "causal",
        "scope": "closed_endpoint_observed_trailing128",
        "endpoint_only": True,
        "input_length": 128,
        "input_channels": 28,
        "state_width": c["width"],
        "encoder": c,
        "patch_length": PATCH_LENGTH,
        "recent_length": RECENT_LENGTH,
        "old_length": OLD_LENGTH,
        "old_token_stride": stride,
        "token_count": OLD_LENGTH // stride + RECENT_LENGTH,
        "token_endpoints_zero_based": list(range(stride - 1, OLD_LENGTH, stride))
        + list(range(OLD_LENGTH, 128)),
        "position_coordinates": "original_zero_based_bar_endpoints",
        "input_order": "oldest_bar_first_then_all28_features",
        "padding": "first_three_old_dense_positions_use_standardized_zero_rows"
        if stride == 1
        else "none",
        "recent_projection": "original_single28_to_width",
        "old_projection": "separate_ordered112_to_width",
        "initialization_scope": "dense_endpoint_only_matches_original_bar_at_step0; independent_projections_then_train_separately",
        "hybrid_endpoint_initial_information": "only_old_patch_endpoints_active_at_step0; zero_columns_are_trainable",
        "internal_tokens_are_public_bar_states": False,
        "early_tokens_have_complete_own128": False,
        "state_scaler": "fixed_zero_mean_unit_scale_no_refit",
        "raw_or_target_decoder_bypass": False,
        "additional_parameters_vs_bar": (28 * PATCH_LENGTH + 1) * c["width"],
        "checkpoint_semantics": "bind_arm_structure_and_initialization_separately_from_identical_tensor_keys",
    }


def model_contract(encoder):
    if not isinstance(encoder, Encoder):
        raise ValueError("Input-factorial encoder required")
    expected = contract(encoder.arm, encoder.config)
    if (
        encoder.mode != "rolling"
        or encoder.attention_mode != "causal"
        or len(encoder.layers) != 4
        or encoder.structure != expected["structure"]
        or encoder.initialization != expected["initialization"]
        or encoder.patch_length != PATCH_LENGTH
        or encoder.recent_length != RECENT_LENGTH
        or encoder.old_length != OLD_LENGTH
        or encoder.token_stride != expected["old_token_stride"]
        or (encoder.input.in_features, encoder.input.out_features) != (28, encoder.config["width"])
        or (encoder.context_input.in_features, encoder.context_input.out_features)
        != (112, encoder.config["width"])
    ):
        raise ValueError("Input-factorial encoder and declared projection/stride semantics differ")
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
    return encoder(windows.flatten(0, 1)).reshape(
        len(x), positions.shape[1], encoder.config["width"]
    )


def predict(encoder, query, x, positions):
    state = read_states(encoder, x, positions)
    return state, query(state.flatten(0, 1)).reshape(len(x), positions.shape[1], 127, 7)


def _tensor_signature(pair, *, exclude_context=False):
    digest = hashlib.sha256()
    for family, network in zip(("encoder", "query"), pair, strict=True):
        for name, tensor in network.state_dict().items():
            if exclude_context and family == "encoder" and name.startswith("context_input."):
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{family}/{name}:{array.dtype}:{array.shape}\n".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def compare_initialized_pairs(pairs):
    if set(pairs) != set(ARMS):
        raise ValueError("All four declared fresh scratch pairs required")
    for arm, pair in pairs.items():
        if model_contract(pair[0])["arm"] != arm:
            raise ValueError("Scratch pair key and arm differ")
    signatures = {arm: _tensor_signature(pair) for arm, pair in pairs.items()}
    common = {arm: _tensor_signature(pair, exclude_context=True) for arm, pair in pairs.items()}
    counts = {
        arm: {
            "encoder": sum(p.numel() for p in pair[0].parameters()),
            "query": sum(p.numel() for p in pair[1].parameters()),
            "input": sum(p.numel() for p in pair[0].input.parameters()),
            "context_input": sum(p.numel() for p in pair[0].context_input.parameters()),
        }
        for arm, pair in pairs.items()
    }
    checks = {
        "common_single_backbone_coordinates_query_equal": len(set(common.values())) == 1,
        "random_structure_tensors_equal": signatures["dense_random"] == signatures["hybrid_random"],
        "endpoint_structure_tensors_equal": signatures["dense_endpoint"]
        == signatures["hybrid_endpoint"],
        "parameter_counts_equal": len({tuple(v.values()) for v in counts.values()}) == 1,
        "query_scalers_zero_one": all(
            not p[1].state_mean.count_nonzero().item() and bool((p[1].state_scale == 1).all())
            for p in pairs.values()
        ),
        "endpoint_projection_copy": all(
            not pairs[a][0].context_input.weight[:, :84].count_nonzero().item()
            and torch.equal(pairs[a][0].context_input.weight[:, -28:], pairs[a][0].input.weight)
            and torch.equal(pairs[a][0].context_input.bias, pairs[a][0].input.bias)
            for a in ("dense_endpoint", "hybrid_endpoint")
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "signatures": signatures,
        "common_signatures": common,
        "parameter_counts": counts,
        "limits": "Fresh construction needed; tensor equality alone does not establish source provenance or efficacy",
    }


@torch.no_grad()
def initialization_receipt(seed, device, config=None):
    pairs = {arm: make_models(arm, seed, device, config) for arm in ARMS}
    report = compare_initialized_pairs(pairs)
    reference = uniform_context.make_models("rolling", seed, device, config)
    original_signature = _tensor_signature(reference)
    report["checks"]["original_bar_common_tensors_equal"] = all(
        s == original_signature for s in report["common_signatures"].values()
    )
    # Deterministic synthetic rows, not research data or restored weights.
    x = (
        torch.sin(
            torch.arange(2 * 128 * 28, device=device, dtype=torch.float32).reshape(2, 128, 28) / 17
        )
        * 0.02
    )
    for pair in (*pairs.values(), reference):
        for net in pair:
            net.eval()
    encoder, query = pairs["dense_endpoint"]
    state, ref_state = encoder(x), reference[0](x)[:, -1]
    checks = {
        "input_tokens": (encoder.input_tokens(x, include_positions=False), reference[0].input(x)),
        "endpoint_state": (state, ref_state),
        "query_output": (query(state), reference[1](ref_state)),
    }
    parity = {
        name: {
            "passed": bool(torch.allclose(a, b, atol=1e-5, rtol=2e-5)),
            "max_abs": float((a - b).abs().max()),
            "atol": 1e-5,
            "rtol": 2e-5,
        }
        for name, (a, b) in checks.items()
    }
    report["checks"]["dense_endpoint_step0_original_bar_function"] = all(
        v["passed"] for v in parity.values()
    )
    report["dense_endpoint_original_bar_parity"] = parity
    report["hybrid_endpoint_original_bar_equivalent"] = False
    report["passed"] = all(report["checks"].values())
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
    """Synthetic/feature dependency checks; not raw lineage or real-model acceptance."""
    encoder.eval()
    query.eval()
    declared = model_contract(encoder)
    ps = torch.tensor([1, 2, 3, 4, 31, 32, 127, 128], device=x.device).expand(len(x), -1)
    y, mask = targets(x, ps, statistics)
    oracle, oracle_mask = coverage.oracle_targets(x.cpu().numpy(), ps.cpu().numpy(), statistics)
    z, pred = predict(encoder, query, x, ps)
    count = declared["token_count"]
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
    raw, ends = observed.cpu().numpy(), encoder.token_endpoints().tolist()
    old_ends = [end for end in ends if end < OLD_LENGTH]
    packed = np.zeros((len(x), len(old_ends), 112), dtype=raw.dtype)
    for token, end in enumerate(old_ends):
        for slot, row in enumerate(range(end - 3, end + 1)):
            if row >= 0:
                packed[:, token, slot * 28 : (slot + 1) * 28] = raw[:, row]
    rows = encoder.input_rows(observed)
    checks["independent_ordered_old_packing"] = bool(
        np.array_equal(rows["context"].cpu().numpy(), packed)
    )
    checks["recent_original_single_rows"] = bool(
        np.array_equal(rows["single"].cpu().numpy(), raw[:, OLD_LENGTH:])
    )
    h = encoder.input_tokens(observed, include_positions=False)
    checks["original_bar_positions"] = bool(
        torch.equal(
            encoder.input_tokens(observed),
            h + positional_encoding(128, h.shape[-1], h.device, h.dtype)[ends],
        )
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
        "model_contract": declared,
        "attention_graph": {
            "mask_shape": list(graph.shape),
            "allowed_edges_per_layer": int((~graph).sum()),
        },
        "limits": [
            "Only each full128 endpoint is public",
            "Prepared features include causal EMA prehistory",
            "Raw float64 close truth is audited separately",
        ],
    }
