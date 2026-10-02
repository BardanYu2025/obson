"""Matched scratch encoders; bounded relative attention has exactly128 feature RF."""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import history_query as hq
from . import history_structure as hs
from .history_autoencoder import positions

MODES = ("prefix", "rolling", "relative")
WINDOWS = (33, 33, 33, 32)  # RF =1+sum(window-1)=128, NOT128 at each of4 layers.
CONFIG = {"width": 768, "heads": 8, "ff": 3072, "layers": 4}


class Encoder(nn.Module):
    def __init__(self, mode, seed, config=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown context mode")
        self.mode = mode
        c = dict(CONFIG if config is None else config)
        if c["layers"] != 4 or c["width"] % c["heads"]:
            raise ValueError("Four layers and divisible attention width required")
        self.config = c
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input = nn.Linear(28, c["width"])
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        c["width"],
                        c["heads"],
                        c["ff"],
                        dropout=0.0,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                    for _ in range(4)
                ]
            )
            self.coordinates = nn.Linear(c["width"], c["width"])
            nn.init.normal_(self.coordinates.weight, std=0.02)
            nn.init.zeros_(self.coordinates.bias)
        # Fixed per-head distance penalties, no extra learned parameters.
        self.register_buffer("slopes", torch.logspace(-2, 0, c["heads"], base=2.0))
        self.checkpointing = True

    def attention_mask(self, length, window, batch, ref):
        age = (
            torch.arange(length, device=ref.device)[:, None]
            - torch.arange(length, device=ref.device)[None]
        )
        if self.mode != "relative":
            return age < 0
        bias = -self.slopes[:, None, None].to(ref.dtype) * age.clamp_min(0).to(ref.dtype)[None] / 8
        bias = bias.masked_fill(((age < 0) | (age >= window))[None], -torch.inf)
        return bias.repeat(batch, 1, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 28:
            raise ValueError("Expected causal28 input features")
        h = self.input(x)
        if self.mode != "relative":
            if x.shape[1] > 128:
                raise ValueError("Global baseline accepts at most128; no hidden extra history")
            h = h + positions(len(x[0]), h.shape[-1], h.device, h.dtype)
        for layer, window in zip(self.layers, WINDOWS, strict=True):
            mask = self.attention_mask(h.shape[1], window, len(h), h)
            if self.checkpointing and torch.is_grad_enabled():
                h = checkpoint(
                    lambda value, layer=layer, mask=mask: layer(value, src_mask=mask),
                    h,
                    use_reentrant=False,
                )
            else:
                h = layer(h, src_mask=mask)
        return self.coordinates(h)


def make_models(mode, seed, device, config=None):
    encoder = Encoder(mode, seed, config).to(device)
    width = encoder.config["width"]
    qc = {"slots": 4, "width": width // 4, "heads": 4, "layers": 2, "ff": width // 2}
    query = hq.HistoryQuery(
        width, {"mean": [0.0] * width, "scale": [1.0] * width}, seed, config=qc
    ).to(device)
    return encoder, query


def prefixes(n, seed, epoch):
    # Same four interior prefixes and endpoint as the established sparse protocol.
    return hq.sample_prefixes(n, seed, epoch)


def rolling_windows(x, ps):
    """255-bar blocks; prefix p of final128 ends at126+p and has full128 past."""
    if (
        x.shape[1:] != (255, 28)
        or ps.ndim != 2
        or len(ps) != len(x)
        or not ((ps >= 1) & (ps <= 128)).all()
    ):
        raise ValueError("255-bar block and prefixes1..128 required")
    ix = ps[..., None] - 1 + torch.arange(128, device=x.device)
    return x[torch.arange(len(x), device=x.device)[:, None, None], ix]


def read_states(encoder, x, ps, native=False):
    if encoder.mode == "relative":
        h = encoder(x)
        return h[torch.arange(len(x), device=x.device)[:, None], ps + 126]
    if encoder.mode == "prefix" and native:
        h = encoder(x[:, -128:])
        return h[torch.arange(len(x), device=x.device)[:, None], ps - 1]
    windows = rolling_windows(x, ps)
    return encoder(windows.flatten(0, 1))[:, -1].reshape(len(x), ps.shape[1], -1)


def targets(x, ps, statistics, native=False):
    """Canonical reconstruction masks reuse observed features; no future/current targets."""
    if x.shape[1:] != (255, 28) or not ((ps >= 32) & (ps <= 128)).all():
        raise ValueError("255 observed features and prefixes32..128 required")
    raw = x.double() * x.new_tensor(statistics["x_scale"], dtype=torch.float64) + x.new_tensor(
        statistics["x_mean"], dtype=torch.float64
    )
    for i in (20, 22):
        zero = np.float32(-statistics["x_mean"][i] / statistics["x_scale"][i])
        raw[..., i] = torch.where(x[..., i] == float(zero), 0.0, raw[..., i])
    geometry = raw[..., :2].sinh()
    physical = torch.cat(
        (geometry.sum(-1).cumsum(-1)[..., None], geometry[..., 1:2], raw[..., 18:23]), -1
    )
    valid = torch.ones_like(physical, dtype=torch.bool)
    valid[..., 3] = raw[..., 27] > 0
    valid[..., 4] = raw[..., 23] > 0
    valid[..., 5] = raw[..., 24] > 0
    valid[..., 6] = raw[..., 25] > 0
    suspect = ((raw[..., 23] > 0) & (raw[..., 22] == 0) & (raw[..., 20] != 0)) | (
        (raw[..., 24] > 0) & (raw[..., 21].abs() > np.arcsinh(1.0) + 1e-6)
    )
    valid[..., 4:6] &= ~suspect[..., None]
    ages = torch.arange(1, 128, device=x.device)
    index = ps[..., None] + 126 - ages
    rows = torch.arange(len(x), device=x.device)[:, None, None]
    value = physical[rows, index].clone()
    anchor = physical[torch.arange(len(x), device=x.device)[:, None], ps + 126, 0]
    value[..., 0] = (value[..., 0] - anchor[..., None]) / hq.price_scales(statistics, value)
    value[..., 1:] = (
        value[..., 1:] - value.new_tensor(statistics["y_mean"][1:])
    ) / value.new_tensor(statistics["y_scale"][1:])
    mask = valid[rows, index]
    if native:
        mask &= (ages[None, None] < ps[..., None])[..., None]
    return torch.where(mask, value, 0.0).to(x.dtype), mask


def loss_rows(pred, y, mask, stats, smooth=False):
    q = hq.metrics(pred, y, mask, stats, smooth)
    s = hs.metrics(pred, y, mask, stats, smooth)
    return {
        "query": q["primary"],
        "path": q["path"],
        "change1": q["change1"],
        "body": q["body"],
        "activity": q["activity"],
        "structure": s["primary"],
        "objective": q["primary"] + s["primary"],
    }


def band_rows(pred, y, mask, stats):
    out = loss_rows(pred, y, mask, stats)
    for name, (lo, hi) in zip(("near", "mid", "far"), hq.BANDS, strict=True):
        keep = mask.clone()
        keep[..., : lo - 1, :] = False
        keep[..., hi:, :] = False
        q = hq.metrics(pred, y, keep, stats)
        out[name] = (0.4 * q["path"] + 0.2 * q["change1"] + 0.1 * q["body"]) / 0.7
        out[name + "_activity"] = q["activity"]
    out["price"] = (0.4 * out["path"] + 0.2 * out["change1"] + 0.1 * out["body"]) / 0.7
    return out


def learning_rate(epoch, total, peak):
    if epoch <= 10:
        return peak * epoch / 10
    return float(peak * (0.05 + 0.95 * (1 + np.cos(np.pi * (epoch - 10) / (total - 10))) / 2))


def validation_score(values):
    # All models selected on the SAME complete128-context four endpoints.
    return float(values["query"] + values["structure"])
