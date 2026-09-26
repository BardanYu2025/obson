"""A state-only, relative-age decoder and strictly past historical supervision."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from . import bar_alignment as ba
from .history_autoencoder import positions

CONFIG = {"slots": 4, "width": 192, "heads": 4, "layers": 2, "ff": 384}
AGES = tuple(range(1, 128))
BANDS = ((1, 16), (17, 64), (65, 127))
QUERY_PREFIXES = tuple(p for p in ba.TRAIN_PREFIXES if p != 128)


class QueryBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config["width"]
        self.norm = nn.LayerNorm(w)
        self.cross = nn.MultiheadAttention(w, config["heads"], dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(w), nn.Linear(w, config["ff"]), nn.GELU(), nn.Linear(config["ff"], w)
        )

    def forward(self, q, memory):
        q = q + self.cross(self.norm(q), memory, memory, need_weights=False)[0]
        return q + self.ff(q)


class HistoryQuery(nn.Module):
    """Queries never attend to other queries or to original bars/other states."""

    def __init__(self, latent, scaling, seed, config=None):
        super().__init__()
        self.config = dict(CONFIG if config is None else config)
        c = self.config
        if c["width"] * c["slots"] != latent or c["width"] % c["heads"]:
            raise ValueError("Memory token widths must sum to the fixed state width")
        self.register_buffer("state_mean", torch.tensor(scaling["mean"], dtype=torch.float32))
        self.register_buffer("state_scale", torch.tensor(scaling["scale"], dtype=torch.float32))
        if (
            self.state_mean.shape != (latent,)
            or self.state_scale.shape != (latent,)
            or not torch.isfinite(self.state_mean).all()
            or not torch.isfinite(self.state_scale).all()
            or (self.state_scale <= 0).any()
        ):
            raise ValueError("Finite frozen train-only state scaler required")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 20260927)
            self.memory = nn.Linear(latent, latent)
            self.query = nn.Linear(c["width"], c["width"])
            self.blocks = nn.ModuleList([QueryBlock(c) for _ in range(c["layers"])])
            self.output = nn.Linear(c["width"], 7)
        self.register_buffer("age_encoding", positions(128, c["width"], "cpu", torch.float32))

    def forward(self, z, ages=None):
        if z.ndim != 2 or z.shape[-1] != len(self.state_mean):
            raise ValueError("One vector per historical query context required")
        if ages is None:
            ages = torch.arange(1, 128, device=z.device)
        if ages.ndim != 1 or ages.dtype != torch.long or not ((ages >= 1) & (ages <= 127)).all():
            raise ValueError("Only historical ages1..127; current/future queries forbidden")
        c = self.config
        memory = self.memory((z - self.state_mean) / self.state_scale).reshape(
            len(z), c["slots"], c["width"]
        )
        q = self.query(self.age_encoding[ages]).expand(len(z), -1, -1)
        for block in self.blocks:
            q = block(q, memory)
        return self.output(q)


def price_scales(stats, ref):
    return torch.arange(1, 128, device=ref.device, dtype=ref.dtype).sqrt() * stats["delta_scale"][0]


def anchors(batch, prefixes, stats):
    """Observed current close coordinate is a label/display anchor, never decoder input.

    Source y's last row is masked to zero. Recover its close using the last
    observed gap/body plus the previous close, never the masked target row.
    """
    y = batch["y"].to(torch.float64)
    close = y[..., 0] * stats["y_scale"][0] + stats["y_mean"][0]
    raw = batch["x"][:, -1, :2].to(torch.float64) * y.new_tensor(
        stats["x_scale"][:2]
    ) + y.new_tensor(stats["x_mean"][:2])
    close = close.clone()
    close[:, -1] = close[:, -2] + torch.sinh(raw).sum(-1)
    return close.gather(1, prefixes - 1)


def targets(batch, prefixes, stats):
    if (
        prefixes.ndim != 2
        or len(prefixes) != len(batch["x"])
        or not ((prefixes >= 32) & (prefixes <= 128)).all()
    ):
        raise ValueError("Historical prefixes32..128 required")
    ages = torch.arange(1, 128, device=prefixes.device)
    index = prefixes[..., None] - 1 - ages
    observed = index >= 0
    index = index.clamp_min(0)
    rows = torch.arange(len(prefixes), device=prefixes.device)[:, None, None]
    physical = batch["y"].to(torch.float64) * batch["y"].new_tensor(
        stats["y_scale"], dtype=torch.float64
    ) + batch["y"].new_tensor(stats["y_mean"], dtype=torch.float64)
    values = physical[rows, index].clone()
    values[..., 0] -= anchors(batch, prefixes, stats)[..., None]
    mask = batch["mask"][rows, index] & observed[..., None]
    values[..., 0] /= price_scales(stats, values)
    values[..., 1:] = (
        values[..., 1:] - values.new_tensor(stats["y_mean"][1:])
    ) / values.new_tensor(stats["y_scale"][1:])
    return torch.where(mask, values, 0).to(batch["y"].dtype), mask


def physical(pred, stats):
    values = pred.to(torch.float64).clone()
    values[..., 0] *= price_scales(stats, values)
    values[..., 1:] = values[..., 1:] * values.new_tensor(stats["y_scale"][1:]) + values.new_tensor(
        stats["y_mean"][1:]
    )
    return values


def metrics(pred, target, mask, stats, smooth=False):
    """Equal available age bands per context, then equal query contexts per window."""
    if pred.shape != target.shape or pred.shape != mask.shape or pred.shape[-2:] != (127, 7):
        raise ValueError("Matched query predictions/targets/masks required")
    error = (
        functional.smooth_l1_loss(pred, target, reduction="none")
        if smooth
        else (pred - target).square()
    )
    p, y = physical(pred, stats), physical(target, stats)
    delta = ((p[..., 1:, 0] - p[..., :-1, 0]) - (y[..., 1:, 0] - y[..., :-1, 0])) / stats[
        "delta_scale"
    ][0]
    de = (
        functional.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
        if smooth
        else delta.square()
    )
    families = {k: [] for k in ("path", "body", "activity", "change1", "close_mae_bps")}
    supports = []
    for lo, hi in BANDS:
        a, b = lo - 1, hi
        valid = mask[..., a:b, :]
        support = valid[..., 0].any(-1)
        supports.append(support)
        counts = valid.sum(-2)
        channel = (error[..., a:b, :] * valid).sum(-2) / counts.clamp_min(1)
        ok = counts[..., 2:] > 0
        activity = (channel[..., 2:] * ok).sum(-1) / ok.sum(-1).clamp_min(1)
        families["path"].append(channel[..., 0])
        families["body"].append(channel[..., 1])
        families["activity"].append(activity)
        # Delta age i->i+1 belongs to the older age's band; preserve band boundaries.
        da, db = max(0, a - 1), b - 1
        dm = mask[..., da + 1 : db + 1, 0] & mask[..., da:db, 0]
        families["change1"].append((de[..., da:db] * dm).sum(-1) / dm.sum(-1).clamp_min(1))
        families["close_mae_bps"].append(
            ((p[..., a:b, 0] - y[..., a:b, 0]).abs() * valid[..., 0]).sum(-1)
            / counts[..., 0].clamp_min(1)
            * 100
        )
    support = torch.stack(supports, -1)
    result = {
        k: (torch.stack(v, -1) * support).sum(-1) / support.sum(-1).clamp_min(1)
        for k, v in families.items()
    }
    result["primary"] = (
        0.4 * result["path"]
        + 0.2 * result["change1"]
        + 0.1 * result["body"]
        + 0.3 * result["activity"]
    )
    return result


def sample_prefixes(n, seed, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260928]))
    chosen = np.argsort(rng.random((n, len(QUERY_PREFIXES))), axis=1)[:, :4]
    return np.c_[np.sort(np.asarray(QUERY_PREFIXES)[chosen], axis=1), np.full(n, 128)].astype(
        np.int64
    )


def predict_states(model, query, x, prefixes, detach=False):
    states = model.core.encoder(x)
    picked = states[torch.arange(len(x), device=x.device)[:, None], prefixes - 1]
    if detach:
        picked = picked.detach()
    return states, query(picked.flatten(0, 1)).reshape(len(x), prefixes.shape[1], 127, 7)


def recent_prediction(pred, stats, local):
    """Same original past16 target: relative to OWN predicted close at age17."""
    values = physical(pred, stats)
    recent = values[..., :16, :].flip(-2).clone()
    recent[..., 0] -= values[..., 16:17, 0]
    return ((recent - recent.new_tensor(local["mean"])) / recent.new_tensor(local["scale"])).to(
        pred.dtype
    )
