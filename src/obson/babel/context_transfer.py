"""Warm full-context supervision with unchanged Macro encoder and historical reader."""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from . import uniform_context as original
from . import uniform_context_run as previous

PREFIXES = (32, 64, 96, 128)
METRICS = ("price", "near", "mid", "far", "activity", "structure")
MODES = ("prefix", "rolling")


def states(encoder, x, ps, native=False, checkpointing=True):
    """Original encoder object, including trained input adapter and coordinates."""

    def call(value):
        return (
            checkpoint(encoder, value, use_reentrant=False)
            if checkpointing and torch.is_grad_enabled()
            else encoder(value)
        )

    if native:
        z = call(x[:, -128:])
        return z[torch.arange(len(x), device=x.device)[:, None], ps - 1]
    windows = original.rolling_windows(x, ps)
    return call(windows.flatten(0, 1))[:, -1].reshape(len(x), ps.shape[1], -1)


def predict(encoder, query, x, ps, native=False):
    z = states(encoder, x, ps, native)
    return z, query(z.flatten(0, 1)).reshape(len(x), ps.shape[1], 127, 7)


def score(values, initial):
    """Epoch0 eligible; preserve full and native interfaces before selecting price."""
    if values.keys() != initial.keys():
        raise ValueError("Validation field identities changed")
    ratios = {
        k: float(v) / max(float(initial[k]), 1e-12)
        for k, v in values.items()
        if k.endswith(tuple("/" + m for m in METRICS))
    }
    if not ratios or not all(np.isfinite(list(ratios.values()))):
        raise ValueError("Finite matched validation fields required")
    protected = all(v <= 1.10 for v in ratios.values())
    primary = max(ratios["full/endpoint/price"], ratios["full/interior/price"])
    return {"eligible": protected, "score": float(primary), "worst_retention": max(ratios.values())}


def stop(mode, epoch, seconds, target_seconds, base=60, cap=300):
    if mode not in MODES or not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Valid mode and accumulated training seconds required")
    if mode == "rolling":
        return epoch >= base
    if target_seconds is None or not np.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("Completed paired rolling training time required")
    return epoch >= cap or (epoch >= base and seconds >= target_seconds)


def time_match(seconds, target):
    ratio = float(seconds) / float(target)
    return {
        "ratio": ratio,
        "matched": bool(0.95 <= ratio <= 1.05),
        "scope": "Single-worker synchronized training-stage wall time; includes CPU feeding/targets, excludes validation/checkpoint I/O. Not exact FLOPs.",
    }


def schedule(n, seed, epoch):
    # Deliberately retain the same deterministic stream across arms and continuation.
    return previous.schedule(n, seed, epoch)
