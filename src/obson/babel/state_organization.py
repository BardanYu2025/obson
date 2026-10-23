"""Endpoint-only cross-depth content aggregation with a private two-write memory.

The backbone never reads the memory. Both writes may run after the backbone has
finished: staged versus late isolates the first write's input depth, not online
execution timing. Only the final768 state is public; working memory is larger.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import history_query as hq
from . import observed_history as original
from . import state_coverage as coverage
from . import uniform_context
from .history_autoencoder import positions as positional_encoding

SCHEMA = "babel-state-organization-model-v1"
ARMS = ("plain", "late_memory", "staged_memory")
CONFIG = dict(original.CONFIG)
VAL_POSITIONS = original.VAL_POSITIONS
targets = original.targets
rolling_windows = original.rolling_windows
MEMORY_SLOTS = 4
MEMORY_HEADS = 4


class WriteBlock(nn.Module):
    """Read observed hidden rows, then mix the four private memory slots."""

    def __init__(self, width):
        super().__init__()
        lane = width // MEMORY_SLOTS
        self.query_norm = nn.LayerNorm(lane)
        self.history_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(
            lane, MEMORY_HEADS, kdim=width, vdim=width, dropout=0.0, batch_first=True
        )
        self.self_norm = nn.LayerNorm(lane)
        self.self_attention = nn.MultiheadAttention(
            lane, MEMORY_HEADS, dropout=0.0, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(lane)
        self.ff = nn.Sequential(nn.Linear(lane, 4 * lane), nn.GELU(), nn.Linear(4 * lane, lane))

    def forward(self, memory, history):
        h = self.history_norm(history)
        memory = memory + self.cross(self.query_norm(memory), h, h, need_weights=False)[0]
        u = self.self_norm(memory)
        memory = memory + self.self_attention(u, u, u, need_weights=False)[0]
        return memory + self.ff(self.ff_norm(memory))


class Memory(nn.Module):
    """Two independent write blocks; state resets for every independent128 window."""

    def __init__(self, width, seed):
        super().__init__()
        if width < MEMORY_SLOTS * MEMORY_HEADS or width % (MEMORY_SLOTS * MEMORY_HEADS):
            raise ValueError("Width must support four equal slots and four heads per slot")
        self.width = width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 20261024)
            self.initial = nn.Parameter(torch.randn(MEMORY_SLOTS, width // MEMORY_SLOTS) * 0.02)
            self.writes = nn.ModuleList([WriteBlock(width), WriteBlock(width)])
            self.output = nn.Linear(width, width)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward_parts(self, first_history, final_history, *, reset_first=False):
        if (
            first_history.ndim != 3
            or first_history.shape[1:] != (128, self.width)
            or first_history.shape != final_history.shape
            or not len(first_history)
            or first_history.device != final_history.device
            or first_history.dtype != final_history.dtype
        ):
            raise ValueError("Two matched nonempty B x128 xwidth hidden arrays required")
        initial = self.initial.expand(len(first_history), -1, -1)
        first = self.writes[0](initial, first_history)
        carried = initial if reset_first else first
        final = self.writes[1](carried, final_history)
        return first, final, self.output(final.flatten(1))

    def forward(self, first_history, final_history):
        return self.forward_parts(first_history, final_history)[-1]


class Encoder(original.Encoder):
    def __init__(self, arm, seed, config=None):
        if arm not in ARMS:
            raise ValueError(f"Expected a declared arm from {ARMS}")
        super().__init__("control", seed, config)
        self.arm = arm
        self.attention_mode = "causal"
        if self.config["width"] % (MEMORY_SLOTS * MEMORY_HEADS):
            raise ValueError("State width must support four slots with four memory heads")
        if arm != "plain":
            # Registered last so all inherited parameter names/order are stable.
            self.memory = Memory(self.config["width"], seed)

    def attention_mask(self, length, window, batch, ref):
        return uniform_context.Encoder.attention_mask(self, length, window, batch, ref)

    def depth_features(self, x):
        if x.ndim != 3 or x.shape[1:] != (128, 28) or not len(x):
            raise ValueError("Only nonempty B x128 x28 observed windows required")
        h = self.input(x)
        h = h + positional_encoding(128, h.shape[-1], h.device, h.dtype)
        middle = None
        for index, (layer, window) in enumerate(
            zip(self.layers, uniform_context.WINDOWS, strict=True), start=1
        ):
            mask = self.attention_mask(128, window, len(h), h)
            if self.checkpointing and torch.is_grad_enabled():
                h = checkpoint(
                    lambda value, layer=layer, mask=mask: layer(value, src_mask=mask),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(h, src_mask=mask)
            if index == 2:
                middle = h
        return middle, h

    def forward_parts(self, x, *, reset_first=False):
        """Diagnostics only: raw/private arrays never enter the public query API."""
        h2, h4 = self.depth_features(x)
        # Preserve source GEMM layout: project128 rows before selecting endpoint.
        base = self.coordinates(h4)[:, -1]
        first = final = None
        if self.arm == "plain":
            correction = torch.zeros_like(base)
            state = base
        else:
            first_history = h2 if self.arm == "staged_memory" else h4
            first, final, correction = self.memory.forward_parts(
                first_history, h4, reset_first=reset_first
            )
            state = base + correction
        return {
            "state": state,
            "base": base,
            "h2": h2,
            "h4": h4,
            "memory_first": first,
            "memory_final": final,
            "correction": correction,
        }

    def forward(self, x):
        return self.forward_parts(x)["state"]


def model_contract(encoder):
    has_memory = hasattr(encoder, "memory")
    if (
        not isinstance(encoder, Encoder)
        or encoder.arm not in ARMS
        or encoder.attention_mode != "causal"
        or encoder.mode != "rolling"
        or len(encoder.layers) != 4
        or has_memory != (encoder.arm != "plain")
        or (has_memory and encoder.memory.width != encoder.config["width"])
    ):
        raise ValueError("State-organization encoder and matching causal semantics required")
    width = encoder.config["width"]
    return {
        "schema": SCHEMA,
        "arm": encoder.arm,
        "attention_mode": "causal",
        "scope": "closed_endpoint_observed_trailing128",
        "endpoint_only": True,
        "input_length": 128,
        "input_channels": 28,
        "state_width": width,
        "encoder": dict(encoder.config),
        "internal_tokens_are_public_bar_states": False,
        "internal_attention_is_causal": True,
        "early_tokens_have_complete_own128": False,
        "public_output": "organized_final_state_only",
        "memory_write_sources": (
            [2, 4] if encoder.arm == "staged_memory" else [4, 4] if has_memory else []
        ),
        "private_memory_exposed": False,
        "memory": {
            "enabled": has_memory,
            "slots": MEMORY_SLOTS if has_memory else 0,
            "slot_width": width // MEMORY_SLOTS if has_memory else 0,
            "heads": MEMORY_HEADS if has_memory else 0,
            "independent_write_blocks": 2 if has_memory else 0,
            "source_depths": (
                [2, 4] if encoder.arm == "staged_memory" else [4, 4] if has_memory else []
            ),
            "backbone_reads_memory": False,
            "persistent_between_windows": False,
            "total_working_memory_is_not_public_state_size": True,
            "forced_temporal_bands": False,
        },
        "additional_parameters": (
            sum(p.numel() for p in encoder.memory.parameters()) if has_memory else 0
        ),
        "decoder_raw_or_memory_bypass": False,
        "auxiliary_loss": False,
        "checkpoint_semantics": "arm bound externally; candidate tensor keys do not identify first source depth",
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
    model_contract(encoder)
    windows = rolling_windows(x, positions)
    states = encoder(windows.flatten(0, 1))
    return states.reshape(len(x), positions.shape[1], encoder.config["width"])


def predict(encoder, query, x, positions):
    states = read_states(encoder, x, positions)
    return states, query(states.flatten(0, 1)).reshape(len(x), positions.shape[1], 127, 7)


@torch.no_grad()
def audit(encoder, query, x, statistics):
    """Prepared-row causality and endpoint interface, not efficacy or raw lineage."""
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
        before = read_states(encoder, changed, ps[:, j : j + 1])[:, 0]
        checks[f"prepared_rows_before_window_{endpoint}"] = bool(torch.equal(baseline, before))
        checks[f"query_batch_{endpoint}"] = bool(
            torch.allclose(query(z[:, j]), pred[:, j], atol=1e-4, rtol=2e-4)
        )
    ages = torch.tensor([1, 20, 64, 127], device=x.device)
    checks["query_subset"] = bool(
        torch.allclose(query(z[:, -1], ages), pred[:, -1, ages - 1], atol=1e-4, rtol=2e-4)
    )
    again = read_states(encoder, x, ps)
    checks["no_persistent_cross_window_memory"] = bool(torch.equal(z, again))
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
            "private_memory_can_read_all_observed_rows": encoder.arm != "plain",
        },
        "limits": [
            "Only public endpoints have their own complete128 history",
            "Prepared-row checks retain causal EMA prehistory",
            "Feature-derived label oracle is separate from raw-close truth",
            "Private memory increases internal work; public state remains768",
            "Source-depth comparison does not isolate readout adaptation or online write timing",
        ],
    }
