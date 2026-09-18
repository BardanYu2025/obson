"""Causal per-bar embedding with a joint mixture over next-bar coordinates."""

import math

import torch
from torch import nn

from .ar_codec import FEATURES, TARGETS


class ARModel(nn.Module):
    def __init__(self, hidden=128, layers=4, heads=4, dropout=.1, components=5):
        super().__init__()
        self.config = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout, components=components)
        self.input = nn.Linear(len(FEATURES), hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.distribution = nn.Linear(hidden, components * (1 + 2 * len(TARGETS)))
        self.reconstruct = nn.Linear(hidden, len(FEATURES))

    def forward(self, x):
        _, t, _ = x.shape
        hidden = self.config["hidden"]
        pos = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(torch.arange(0, hidden, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / hidden))
        pe = torch.zeros(t, hidden, device=x.device, dtype=x.dtype)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        h = self.norm(self.encoder(self.input(x) + pe, mask=mask))
        raw = self.distribution(h).reshape(*h.shape[:2], self.config["components"], -1)
        return {"h": h, "seq": h[:, -1], "logits": raw[..., 0],
                "mean": raw[..., 1:8], "log_std": raw[..., 8:].clamp(-5, 3),
                "reconstruction": self.reconstruct(h)}


def nll(out, y, available):
    """Joint density of transformed coordinates; folded normal for nonnegative channels.

    A shared mixture component models cross-channel dependence. OI is marginalized
    when absent. This score is not a density in raw price units.
    """
    mu, ls = out["mean"], out["log_std"]
    value = y.unsqueeze(-2)
    normal = -.5 * ((value - mu) * torch.exp(-ls)) ** 2 - ls - .5 * math.log(2 * math.pi)
    reflected = -.5 * ((-value - mu) * torch.exp(-ls)) ** 2 - ls - .5 * math.log(2 * math.pi)
    logp = torch.cat((normal[..., :2], torch.logaddexp(normal[..., 2:], reflected[..., 2:])), -1)
    logp = (logp * available.unsqueeze(-2)).sum(-1)
    return -torch.logsumexp(torch.log_softmax(out["logits"], -1) + logp, -1)


def sample(out, generator):
    weights = torch.softmax(out["logits"], -1)
    shape = weights.shape[:-1]
    component = torch.multinomial(weights.reshape(-1, weights.shape[-1]), 1, generator=generator).reshape(*shape, 1, 1)
    index = component.expand(*shape, 1, 7)
    mean = out["mean"].gather(-2, index).squeeze(-2)
    std = out["log_std"].gather(-2, index).squeeze(-2).exp()
    y = mean + std * torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return torch.cat((y[..., :2], y[..., 2:].abs()), -1)
