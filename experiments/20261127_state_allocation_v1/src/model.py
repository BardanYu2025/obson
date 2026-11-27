"""Causal interval encoder, FP32 public state and addressed historical readout."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .views import U


def geometry(g, valid):
    # g: [B,T,5] trade start/end, wall start/end, coverage. Terminal is last.
    a, b, _, wall, cov = g.unbind(-1)
    terminal = torch.zeros_like(valid)
    terminal[:, -1] = True
    qi, kj = terminal[:, :, None], terminal[:, None, :]
    market = ~(qi | kj)

    def pair(x):
        return x[:, :, None], x[:, None, :]

    ai, aj = pair(a)
    bi, bj = pair(b)
    di, dj = pair(b - a)
    vi, vj = pair(cov)
    endi, endj = pair(wall)
    both = torch.ones_like(bi - bj)

    def signed(x):
        return x.sign() * x.abs().log1p()

    inter = (torch.minimum(bi, bj) - torch.maximum(ai, aj)).clamp_min(0)
    union = di + dj - inter
    phi = torch.stack(
        (
            signed(bi - bj),
            signed(endi - endj),
            di.log1p() * both,
            dj.log1p() * both,
            (di.log1p() - dj.log1p()) * market,
            inter / union.clamp_min(1) * market,
            ((ai <= aj) & (bj <= bi) & market).float(),
            ((aj <= ai) & (bi <= bj) & market).float(),
            vi * both,
            vj * both,
            qi * both,
            kj * both,
        ),
        dim=-1,
    )
    legal = valid[:, :, None] & valid[:, None, :] & (endi >= endj) & (~kj | qi)
    return phi, legal


class IntervalBlock(nn.Module):
    def __init__(self, width, heads, ffn, deep):
        super().__init__()
        self.heads = heads
        self.before = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
        self.relation = nn.Sequential(
            nn.Linear(12, 32), nn.GELU(), nn.Linear(32, heads)
        )
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)
        self.alpha = nn.Parameter(torch.zeros(heads)) if deep else None

    def forward(self, x, phi, allowed, first):
        batch, t, width = x.shape
        q, k, v = (
            self.qkv(self.before(x))
            .reshape(batch, t, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        if first is None:
            first = v
        else:
            alpha = self.alpha.sigmoid()[None, :, None, None]
            v = torch.cat(
                (torch.lerp(v[:, :, :-1], first[:, :, :-1], alpha), v[:, :, -1:]), dim=2
            )
        bias = 2 * self.relation(phi).tanh().permute(0, 3, 1, 2)
        bias = torch.cat((bias[..., :-1], torch.zeros_like(bias[..., -1:])), dim=-1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(k.shape[-1]) + bias
        safe = allowed | (
            ~allowed.any(-1, keepdim=True)
            & torch.eye(t, device=x.device, dtype=torch.bool)
        )
        weights = (
            scores.masked_fill(~safe[:, None], -torch.inf).softmax(-1)
            * allowed[:, None]
        )
        y = x + self.out((weights @ v).transpose(1, 2).reshape(batch, t, width))
        return y + self.ff(self.ff_norm(y)), first


class Encoder(nn.Module):
    def __init__(self, width=768, heads=12, ffn=3072, layers=6):
        super().__init__()
        self.stem = nn.Linear(28, width)
        self.context = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, width))
        self.period = nn.Embedding(4, width)
        self.terminal = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.terminal, std=0.02)
        self.layers = nn.ModuleList(
            IntervalBlock(width, heads, ffn, i > 0) for i in range(layers)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x, ctx, p, g, valid):
        emb = self.stem(x) + self.context(ctx) + self.period(p)
        seq = torch.cat((emb, self.terminal.expand(len(x), 1, -1)), 1)
        phi, allowed = geometry(
            F.pad(g, (0, 0, 0, 1)), F.pad(valid, (0, 1), value=True)
        )
        first = None
        for layer in self.layers:
            seq, first = layer(seq, phi, allowed, first)
        return self.norm(seq[:, -1])


class FilmBlock(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.cond = nn.Linear(dim, 2 * width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, u, z):
        gain, shift = self.cond(z)[:, None].chunk(2, -1)
        return u + self.ff((1 + gain) * self.norm(u) + shift)


class Decoder(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.query = nn.Linear(10, width)
        self.blocks = nn.ModuleList(FilmBlock(dim, width) for _ in range(4))
        self.out = nn.Linear(width, 7)

    def forward(self, z, query):
        u = self.query(query)
        for block in self.blocks:
            u = block(u, z)
        return self.out(u)


def physical_output(raw, period, current, mu, sd):
    # De-standardize in FP64. Absolute value is nonnegative and admits zero.
    cols = [0, 2, 3, 4, 5, 6, 7]
    y = raw.double() * sd[period][:, cols] + mu[period][:, cols]
    close = torch.where(current, 0.0, y[:, 0].clamp(-20, 20))
    return torch.cat(
        (
            close[:, None],
            y[:, 1:2].clamp(-20, 20),
            y[:, 2:4].abs().clamp_max(20),
            y[:, 4:].abs().clamp_max(32),
        ),
        dim=1,
    )


def calibrate(y, facts, units, basis):
    # y [G,16,7]. Every group uses canonical chronological order.
    p = y[:, :, 0]
    c = p + (facts[:, :2] - p @ basis) @ basis.T
    direct = y[:, :, 4].expm1()
    total = facts[:, 2].expm1().clamp_min(0.0)
    ordered = direct.sort(dim=1, descending=True).values
    div = torch.arange(1, 17, device=y.device)
    tau = (ordered.cumsum(1) - total[:, None]) / div
    n = (ordered > tau).sum(1).clamp_min(1)
    v = (direct - tau.gather(1, (n - 1)[:, None])).clamp_min(0.0)
    v = torch.where(total[:, None] > 0, v, 0.0)
    delta = facts[:, 3].sinh() * units
    x = torch.maximum(
        (-delta).clamp_min(0.0), (y[:, 0, 5].expm1() + y[:, -1, 6].expm1() - delta) / 2
    )
    oi0 = torch.cat((x.log1p()[:, None], y[:, 1:, 5]), dim=1)
    oi1 = torch.cat((y[:, :-1, 6], (x + delta).clamp_min(0.0).log1p()[:, None]), dim=1)
    return torch.cat(
        (
            c[:, :, None],
            y[:, :, 1:4],
            v.log1p()[:, :, None],
            oi0[:, :, None],
            oi1[:, :, None],
        ),
        dim=2,
    )


class StateModel(nn.Module):
    def __init__(
        self, arm, stats, width=768, heads=12, ffn=3072, layers=6, film_width=256
    ):
        super().__init__()
        if arm not in ("P2", "P3") or width <= 160 or width % heads:
            raise ValueError("Invalid arm or encoder width")
        self.arm = arm
        self.encoder = Encoder(width, heads, ffn, layers)
        self.history = Decoder(width, film_width)
        self.describe = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 18)
        )
        for key, value in stats.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float64))
        self.register_buffer("basis", torch.as_tensor(U, dtype=torch.float64))
        self.register_buffer(
            "block_period", torch.tensor([0] * 16 + [1] * 8 + [2] * 4 + [3] * 4)
        )

    def pack(self, h, facts):
        if self.arm == "P2":
            return h.float()
        return torch.cat((facts.detach().reshape(len(h), 160), h[:, :-160]), 1).float()

    def condition(self, z):
        if self.arm == "P2":
            return z
        facts = z[:, :160].reshape(-1, 32, 5)
        norm = (facts[:, :, :4] - self.f_mu[self.block_period]) / self.f_sd[
            self.block_period
        ]
        data = torch.where(facts[:, :, 4:] > 0, norm, 0.0).to(z.dtype)
        return torch.cat(
            (torch.cat((data, facts[:, :, 4:]), dim=2).flatten(1), z[:, 160:]), 1
        )

    def encode(self, sample):
        """A single sample or a batch with independently padded market histories."""
        if not isinstance(sample, (list, tuple)):
            device = self.encoder.terminal.device
            vals = [
                torch.as_tensor(a, device=device)[None]
                for a in (
                    sample.x,
                    sample.context,
                    sample.period,
                    sample.geom,
                    sample.valid,
                    sample.facts,
                )
            ]
            return self.pack(self.encoder(*vals[:5]), vals[5])
        if not sample:
            raise ValueError("Empty endpoint batch")
        device = self.encoder.terminal.device

        def padded(field):
            return nn.utils.rnn.pad_sequence(
                [torch.as_tensor(getattr(s, field), device=device) for s in sample],
                batch_first=True,
            )

        h = self.encoder(
            *(padded(k) for k in ("x", "context", "period", "geom", "valid"))
        )
        return self.pack(
            h, torch.as_tensor(np.stack([s.facts for s in sample]), device=device)
        )

    def read(self, z, query, period, current, block, slot, requested):
        """Only z and public calendar addresses. Batched tables return [B,K,7]."""
        if isinstance(query, np.ndarray) and query.ndim == 2:
            if len(z) != 1:
                raise ValueError(
                    "Supply one address table and requested-index array per endpoint"
                )
            return self._read_endpoint(
                z, query, period, current, block, slot, requested
            )
        parts = (query, period, current, block, slot, requested)
        if not len(z) or any(len(arg) != len(z) for arg in parts):
            raise ValueError("Address tables and states require the same batch size")
        result = [
            self._read_endpoint(z[i : i + 1], *(arg[i] for arg in parts))
            for i in range(len(z))
        ]
        post, direct, counts = zip(*result)
        return torch.stack(post), torch.stack(direct), sum(counts)

    def _read_endpoint(self, z, query, period, current, block, slot, requested):
        device = z.device
        requested = np.asarray(requested, dtype=int)
        if requested.ndim != 1 or np.any((requested < 0) | (requested >= len(query))):
            raise ValueError("Historical query index outside its address table")
        table, back = np.unique(requested, return_inverse=True)
        groups = np.unique(block[table])
        groups = groups[groups >= 0]
        if self.arm == "P3":
            valid = z[0, 4:160:5].detach().cpu().numpy() > 0
            groups = groups[valid[groups]]
        else:
            groups = groups[:0]
        rows = []
        for g in groups:
            loc = np.flatnonzero(block == g)
            loc = loc[np.argsort(slot[loc])]
            if not np.array_equal(slot[loc], np.arange(16)):
                raise ValueError("Public block calendar does not supply 16 addresses")
            rows.append(loc)
        expanded = np.unique(np.r_[table, np.ravel(rows).astype(int)])

        def as_tensor(a):
            return torch.as_tensor(a, device=device)

        raw = self.history(self.condition(z), as_tensor(query[expanded])[None])[0]
        direct = physical_output(
            raw,
            as_tensor(period[expanded]),
            as_tensor(current[expanded]),
            self.y_mu,
            self.y_sd,
        )
        post = direct
        if len(rows):
            pos = as_tensor(np.searchsorted(expanded, rows))
            facts = z[:, :160].reshape(32, 5)[as_tensor(groups)].double()
            fixed = calibrate(
                direct[pos],
                facts,
                self.block_delta[self.block_period[as_tensor(groups)]],
                self.basis,
            )
            post = torch.index_copy(direct, 0, pos.flatten(), fixed.flatten(0, 1))
        ix = as_tensor(np.searchsorted(expanded, table)[back])
        return post[ix], direct[ix], len(expanded)
