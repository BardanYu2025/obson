"""Frozen-state future-volatility protocol. Pure helpers; no encoder updates."""

import hashlib

import numpy as np
import torch
from torch import nn

HORIZONS = (16, 64)
QUANTILES = (0.1, 0.5, 0.9)
PERIODS = (15, 30, 60)
FRACTIONS = (0.1, 1.0)
READERS = ("linear", "mlp")
REPRESENTATIONS = ("simple", "raw", "pca", "embedding")
LRS = (1e-3, 3e-4)
SCORE_SCALE = (1.0, 1.0)  # Same log-RMS units across horizons and label budgets.


def future_target(close, end, same):
    """Exactly h NEW close-to-close returns, including C[t+1]/C[t]."""
    close = np.asarray(close, float)
    same = np.asarray(same, bool)
    if (
        close.ndim != 1
        or same.shape != close.shape
        or not np.isfinite(close).all()
        or (close <= 0).any()
    ):
        raise ValueError("Invalid contract prices/partition")
    if end < 127 or end + max(HORIZONS) >= len(close):
        return None, "insufficient_history_or_future"
    if not same[end - 127 : end + max(HORIZONS) + 1].all():
        return None, "input_or_target_crosses_partition"
    r = np.diff(np.log(close[end : end + max(HORIZONS) + 1]))
    return np.array([np.log(max(np.sqrt(np.mean(r[:h] ** 2)), 1e-8)) for h in HORIZONS]), None


def past_features(close, end):
    """No future access. Log RMS; 127-return longest baseline stays inside128 closes."""
    close = np.asarray(close, float)
    if end < 127 or end >= len(close):
        raise ValueError("Insufficient past")
    r = np.diff(np.log(close[end - 127 : end + 1]))
    return np.array([np.log(max(np.sqrt(np.mean(r[-h:] ** 2)), 1e-8)) for h in (8, 16, 64, 127)])


def period_context(rows):
    periods = np.array([int(r["period"]) for r in rows])
    if not np.isin(periods, PERIODS).all():
        raise ValueError("Unknown period")
    return (periods[:, None] == np.array(PERIODS)).astype(np.float32)


def subset(rows, fraction, seed):
    """Same stratified, nested label subset for every representation/reader."""
    if fraction not in FRACTIONS:
        raise ValueError("Unregistered label fraction")
    groups = {}
    for i, r in enumerate(rows):
        key = (r["symbol"], int(r["period"]))
        token = f"{seed}:{r['key']}:{r['row']}".encode()
        groups.setdefault(key, []).append((hashlib.sha256(token).hexdigest(), i))
    ids = []
    for values in groups.values():
        count = max(1, int(np.ceil(len(values) * fraction)))
        ids.extend(i for _, i in sorted(values)[:count])
    return np.array(sorted(ids), dtype=np.int64)


def scaler(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite training matrix required")
    return {"mean": x.mean(0).tolist(), "scale": x.std(0).clip(1e-6).tolist()}


def normalize(x, stats):
    result = ((np.asarray(x, np.float64) - stats["mean"]) / stats["scale"]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized values")
    return result


class Reader(nn.Module):
    def __init__(self, width, family):
        super().__init__()
        if family not in READERS or width < 1:
            raise ValueError("Invalid reader")
        self.net = (
            nn.Linear(width, 6)
            if family == "linear"
            else nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 6))
        )

    def forward(self, x):
        return self.net(x).reshape(-1, 2, 3)


def pinball(pred, truth):
    """Train independent quantiles; monotone rearrangement only at inference."""
    err = truth[..., None] - pred
    q = pred.new_tensor(QUANTILES)
    return torch.maximum(q * err, (q - 1) * err)


def baseline_fit(y, past, periods):
    result = {}
    for period in PERIODS:
        keep = np.asarray(periods) == period
        if keep.sum() < 2:
            raise ValueError(f"Insufficient training labels for period {period}")
        result[str(period)] = {
            "constant": np.quantile(y[keep], QUANTILES, axis=0).T.tolist(),
            "persistence": np.quantile(
                y[keep] - past[keep][:, [1, 2]], QUANTILES, axis=0
            ).T.tolist(),
        }
    return result


def baseline_predict(fit, past, periods, kind):
    if kind not in ("constant", "persistence"):
        raise ValueError("Invalid baseline")
    result = np.array([fit[str(int(p))][kind] for p in periods])
    if kind == "persistence":
        result += past[:, [1, 2], None]
    return result


def score(pred, y, scale):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    if pred.shape != (*y.shape, 3) or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("Prediction shape mismatch")
    if (
        not np.isfinite(pred).all()
        or not np.isfinite(y).all()
        or (np.diff(pred, axis=-1) < 0).any()
    ):
        raise ValueError("Invalid quantile predictions")
    err = y[..., None] - pred
    q = np.array(QUANTILES)
    loss = np.maximum(q * err, (q - 1) * err).mean(-1) / np.array(scale)
    return {
        "primary": float(loss.mean()),
        "horizons": {
            str(h): {
                "pinball": float(loss[:, i].mean()),
                "median_mae_log_rms": float(np.abs(pred[:, i, 1] - y[:, i]).mean()),
                "coverage80": float(
                    ((y[:, i] >= pred[:, i, 0]) & (y[:, i] <= pred[:, i, 2])).mean()
                ),
                "width80_log_rms": float((pred[:, i, 2] - pred[:, i, 0]).mean()),
            }
            for i, h in enumerate(HORIZONS)
        },
    }, loss


def interval(a, b, rows, draws=2000):
    """Paired month-cluster bootstrap; all contracts/periods share calendar blocks."""
    delta = np.asarray(a, float) - np.asarray(b, float)
    if delta.shape != (len(rows),) or not len(rows) or not np.isfinite(delta).all():
        raise ValueError("Invalid paired cohort")
    groups = np.array([str(r["month"]) for r in rows])
    months = sorted(set(groups))
    sums = np.array([delta[groups == g].sum() for g in months])
    counts = np.array([(groups == g).sum() for g in months])
    ids = np.random.default_rng(20261001).integers(0, len(months), (draws, len(months)))
    values = sums[ids].sum(1) / counts[ids].sum(1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "delta": float(delta.mean()),
        "low": float(lo),
        "high": float(hi),
        "months": len(months),
        "windows": len(rows),
        "supported": len(months) >= 6 and len(rows) >= 100,
        "scope": "Exploratory paired calendar-month interval; not multiplicity corrected or independent new-data confirmation.",
    }


def comparisons(errors, rows, baseline):
    a = errors["embedding"]
    checks = []
    for reference, factor in ((baseline, 0.95), ("pca", 0.95), ("raw", 1.05)):
        ci = interval(a.mean(1), factor * errors[reference].mean(1), rows)
        checks.append(
            {
                "target": "primary",
                "reference": reference,
                "factor": factor,
                "interval": ci,
                "passed": bool(ci["supported"] and ci["high"] <= 0),
            }
        )
    for i, h in enumerate(HORIZONS):
        ci = interval(a[:, i], 1.10 * errors["raw"][:, i], rows)
        checks.append(
            {
                "target": str(h),
                "reference": "raw",
                "factor": 1.10,
                "interval": ci,
                "passed": bool(ci["supported"] and ci["high"] <= 0),
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
