"""Frozen causal state dynamics. Prediction interfaces never receive future labels."""

import numpy as np
import torch
from torch import nn

ARMS = ("L1", "N1", "N8", "Z-direct", "Raw-direct")
LRS = (3e-4, 1e-3)
TRAIN_H = 8
EVAL_H = 32


def labels(close, end, same):
    close, same = np.asarray(close, float), np.asarray(same, bool)
    if close.ndim != 1 or same.shape != close.shape:
        raise ValueError("Contract/partition shape differs")
    if end < 127 or end + EVAL_H >= len(close):
        return None, "insufficient_history_or_future"
    if not same[end - 127 : end + EVAL_H + 1].all():
        return None, "input_or_target_crosses_partition"
    c = close[end - 127 : end + EVAL_H + 1]
    if not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("Invalid observed or label prices; cannot silently exclude failures")
    return 100 * np.log(close[end + 1 : end + EVAL_H + 1] / close[end]), None


def past(close, end):
    c = np.asarray(close, float)[end - 127 : end + 1]
    if len(c) != 128 or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("128 valid observed closes required")
    r = 100 * np.diff(np.log(c))
    return np.array([np.sqrt(np.mean(r[-64:] ** 2)), r[-8:].mean(), r[-32:].mean()])


def positive_floor(values, fraction):
    positive = np.asarray(values)[np.asarray(values) > 0]
    if not len(positive) or not np.isfinite(values).all():
        raise ValueError("Degenerate or nonfinite training scale")
    return max(float(np.median(positive)) * fraction, 1e-6)


def state_scaler(z, pairs):
    """Unique TRAIN endpoint states and unique directed adjacent endpoint pairs."""
    z = np.asarray(z, np.float64)
    pairs = np.unique(np.asarray(pairs, np.int64).reshape(-1, 2), axis=0)
    if z.ndim != 2 or len(z) < 2 or not len(pairs) or not np.isfinite(z).all():
        raise ValueError("Invalid training state bank")
    mean, scale = z.mean(0), z.std(0)
    scale_floor = positive_floor(scale, 0.01)
    scale = scale.clip(scale_floor)
    d = (z[pairs[:, 1]] - z[pairs[:, 0]]) / scale
    inc = np.sqrt(np.mean(d**2, axis=0))
    inc_floor = positive_floor(inc, 0.1)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "increment": inc.clip(inc_floor).tolist(),
        "scale_floor": scale_floor,
        "increment_floor": inc_floor,
        "unique_states": len(z),
        "unique_pairs": len(pairs),
    }


def normalize(z, scaling):
    result = ((np.asarray(z, float) - scaling["mean"]) / scaling["scale"]).astype("float32")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized states")
    return result


class Transition(nn.Module):
    def __init__(self, arm, width=768):
        super().__init__()
        if arm not in ARMS[:3]:
            raise ValueError("Invalid state arm")
        self.arm = arm
        self.delta = (
            nn.Linear(width, width)
            if arm == "L1"
            else nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, width))
        )
        last = self.delta if arm == "L1" else self.delta[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, u):
        return u + self.delta(u)

    def rollout(self, initial, steps):
        if steps < 1 or initial.ndim != 2:
            raise ValueError("One initial state and positive horizon required")
        result, u = [], initial
        for _ in range(steps):
            u = self(u)
            result.append(u)
        return torch.stack(result, 1)


class Direct(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, TRAIN_H))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def make_model(arm, width=768, raw_width=3584):
    return (
        Transition(arm, width)
        if arm in ARMS[:3]
        else Direct(raw_width if arm == "Raw-direct" else width)
    )


def objective(model, states, increment, raw=None, target=None, scale=None):
    if isinstance(model, Transition):
        truth = states[:, 1 : TRAIN_H + 1]
        pred = (
            model.rollout(states[:, 0], TRAIN_H)
            if model.arm == "N8"
            else model(states[:, :TRAIN_H])
        )
        return ((pred - truth) / increment).square().mean()
    pred = model(raw if raw is not None else states[:, 0])
    return ((pred - target[:, :TRAIN_H]) / scale[:, :TRAIN_H]).square().mean()


def read_prices(query, z, delta_scale, ages=None):
    """z[:,h-1] is predicted t+h state. No future close anchors are accepted.

    Returns cumulative 100log(C[t+h]/C[t]). Each query uses its actual age;
    the old physical() full127-age broadcaster is intentionally not used.
    """
    if z.ndim != 3 or delta_scale <= 0:
        raise ValueError("State trajectory and positive price scale required")
    ages = list(range(1, z.shape[1] + 1)) if ages is None else list(ages)
    if len(ages) != z.shape[1] or not all(1 <= h <= 127 for h in ages):
        raise ValueError("One historical age per predicted state required")
    return torch.stack(
        [
            -query(z[:, i], torch.tensor([h], device=z.device, dtype=torch.long))[:, 0, 0].double()
            * np.sqrt(h)
            * delta_scale
            for i, h in enumerate(ages)
        ],
        1,
    )


def price_fit(bank):
    fit = {}
    for p in (15, 30, 60):
        ix = bank["period"] == p
        if ix.sum() < 2:
            raise ValueError(f"Insufficient training origins for period {p}")
        fit[str(p)] = {
            "rms_floor": max(float(np.quantile(bank["past"][ix, 0], 0.05)), 1e-6),
            "mean": bank["targets"][ix].mean(0).tolist(),
            "rms_tertiles": np.quantile(bank["past"][ix, 0], [1 / 3, 2 / 3]).tolist(),
        }
    return fit


def price_scale(bank, fit):
    rms = np.maximum(bank["past"][:, 0], [fit[str(int(p))]["rms_floor"] for p in bank["period"]])
    return rms[:, None] * np.sqrt(np.arange(1, EVAL_H + 1))[None]


def baselines(bank, fit):
    h = np.arange(1, EVAL_H + 1)[None]
    return {
        "flat": np.zeros((len(bank["period"]), EVAL_H)),
        "trend8": bank["past"][:, 1, None] * h,
        "trend32": bank["past"][:, 2, None] * h,
        "train_mean": np.array([fit[str(int(p))]["mean"] for p in bank["period"]]),
    }


def errors(pred, truth, scale):
    pred, truth, scale = np.asarray(pred, float), np.asarray(truth, float), np.asarray(scale, float)
    h = pred.shape[1]
    if (
        pred.shape != truth[:, :h].shape
        or scale[:, :h].shape != pred.shape
        or not all(np.isfinite(v).all() for v in (pred, truth, scale))
        or (scale <= 0).any()
    ):
        raise ValueError("Nonfinite or mismatched forecasts; no dropping failed rows")
    e = pred - truth[:, :h]
    increments = np.diff(np.c_[np.zeros(len(e)), e], axis=1)
    result = {
        "price": (e / scale[:, :h]) ** 2,
        "absolute": np.abs(e) * 100,
        "incremental": (increments / scale[:, :1]) ** 2,
    }
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("Forecast error overflow; cannot discard failed samples")
    return result
