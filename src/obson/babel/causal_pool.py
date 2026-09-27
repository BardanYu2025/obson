"""Function-preserving, state-only causal history pooling before compression."""

import math

import torch
from torch import nn

from . import history_structure as structure

MODES = ("plain", "full_pool", "scale_pool")
RANGES = (16, 32, 64, 128)


class HistoryPool(nn.Module):
    def __init__(self, width, mode, seed):
        super().__init__()
        if mode not in MODES[1:] or width % 4:
            raise ValueError("Four equal pooling lanes and a declared pooling mode required")
        self.width, self.mode = width, mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 20260928)
            self.norm = nn.LayerNorm(width)
            self.q = nn.Linear(width, width)
            self.k = nn.Linear(width, width)
            self.v = nn.Linear(width, width)
            self.output = nn.Linear(width, width)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            self.age_bias = nn.Parameter(torch.zeros(4, 128))

    def mask(self, length, device):
        if length < 1:
            raise ValueError("A nonempty visible context is required")
        # Existing causality audit appends16 artificial future bars to128.
        # Support that check without exposing ages>=128 to any pooling lane.
        ages = (
            torch.arange(length, device=device)[:, None] - torch.arange(length, device=device)[None]
        )
        ranges = RANGES if self.mode == "scale_pool" else (128,) * 4
        limits = torch.tensor(ranges, device=device)[:, None, None]
        return ages, (ages[None] >= 0) & (ages[None] < limits)

    def pool(self, q, k, v):
        """Exposed for boundary tests using independent K/V, not mixed causal states."""
        if q.shape != k.shape or k.shape != v.shape or q.ndim != 3 or q.shape[-1] != self.width:
            raise ValueError("Matched full-width query/key/value sequences required")
        n, t, _ = q.shape
        age, allowed = self.mask(t, q.device)
        q, k, v = [x.reshape(n, t, 4, self.width // 4).transpose(1, 2) for x in (q, k, v)]
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.width // 4)
        logits = logits + self.age_bias[:, age.clamp(0, 127)]
        logits = logits.masked_fill(~allowed[None], -torch.inf)
        weights = logits.softmax(-1)
        return (weights @ v).transpose(1, 2).reshape(n, t, self.width)

    def forward(self, h):
        u = self.norm(h)
        return self.output(self.pool(self.q(u), self.k(u), self.v(u)))


class PooledEncoder(nn.Module):
    def __init__(self, original, mode, seed):
        super().__init__()
        # Preserve all inherited state_dict paths, including the PathInput wrapper.
        self.backbone = original.backbone
        self.coordinates = original.coordinates
        self.pool = HistoryPool(self.coordinates.out_features, mode, seed)

    def forward(self, x):
        h = self.backbone(x)
        return self.coordinates(h) + self.pool(h)


def install(model, mode, seed):
    if mode not in MODES:
        raise ValueError("Unknown pooling arm")
    if mode != "plain":
        encoder = model.core.encoder
        model.core.encoder = PooledEncoder(encoder, mode, seed).to(
            encoder.coordinates.weight.device
        )
    return model


def gradient_groups(model, query):
    encoder = model.core.encoder
    base = [p for name, p in encoder.named_parameters() if not name.startswith("pool.")]
    groups = [base, list(model.local_head.parameters()), list(query.parameters())]
    if hasattr(encoder, "pool"):
        groups.append(list(encoder.pool.parameters()))
    return groups


def losses(model, query, batch, prefixes, query_prefixes, shifts, statistics, local, mode):
    if mode not in MODES:
        raise ValueError("Unknown pooling arm")
    return structure.losses(
        model, query, batch, prefixes, query_prefixes, shifts, statistics, local, "structure"
    )


metrics = structure.metrics
WIDTHS = structure.WIDTHS
