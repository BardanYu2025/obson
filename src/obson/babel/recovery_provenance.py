"""V14 independent train-only scaler refits and exact checkpoint comparisons.

No real model execution here. Historical bound fitting functions are deliberately
not used for the arithmetic oracle. The CUDA entrypoint lives in the run module.
"""

import copy

import numpy as np
import torch


def state_moments(states):
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite original training states required")
    mean = np.sum(x, axis=0) / len(x)
    variance = np.sum(np.square(x - mean), axis=0) / len(x)
    return {"mean": mean.tolist(), "scale": np.maximum(np.sqrt(variance), 1e-6).tolist()}


def local_moments(data, stats):
    """Original Prefix32/64/96/128 past16, excluding current, with original rounding."""
    cached = np.asarray(data["y"])
    mask = np.asarray(data["mask"])
    if cached.ndim != 3 or cached.shape[1:] != (128, 7) or mask.shape != cached.shape:
        raise ValueError("Original128x7 target cache required")
    if mask.dtype != np.bool_:
        raise ValueError("Boolean source masks required")
    physical = cached.astype(np.float64) * np.asarray(stats["y_scale"]) + stats["y_mean"]
    blocks, supports = [], []
    for p in (32, 64, 96, 128):
        first, last = p - 17, p - 1
        valid = mask[:, first:last].copy()
        if not valid[..., 0].all() or not mask[:, first - 1, 0].all():
            raise ValueError("Local close or anchor unavailable")
        y = physical[:, first:last].copy()
        y[..., 0] -= physical[:, first - 1, 0, None]
        y = np.where(valid, y, 0).astype(np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite local targets")
        blocks.append(y)
        supports.append(valid)
    values, valid = np.concatenate(blocks), np.concatenate(supports)
    count = valid.sum(axis=(0, 1))
    if (count < 2).any():
        raise ValueError("Insufficient local training support")
    means, scales = [], []
    for channel in range(7):
        observed = values[..., channel][valid[..., channel]].astype(np.float64)
        mean = np.sum(observed) / len(observed)
        means.append(float(mean))
        scales.append(max(float(np.sqrt(np.sum((observed - mean) ** 2) / len(observed))), 1e-5))
    # Source computes differences in float32, then their RMS in float64.
    changes = (values[:, 1:, 0] - values[:, :-1, 0]).astype(np.float64)
    return {
        "mean": means,
        "scale": scales,
        "count": count.tolist(),
        "delta_scale": max(float(np.sqrt(np.sum(changes**2) / changes.size)), 0.01),
        "fitted_on": "original_fixed_train_all_prefixes",
    }


def compare_tree(actual, expected, path=""):
    """Exact comparison including optimizer moments, counters, dtype and RNG bytes."""
    if isinstance(expected, torch.Tensor):
        ok = (
            isinstance(actual, torch.Tensor)
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(expected).all())
            and torch.equal(actual.cpu(), expected.cpu())
        )
        if ok:
            return {}
        detail = {
            "reason": "tensor value/shape/dtype/nonfinite mismatch",
            "expected_shape": list(expected.shape),
            "expected_dtype": str(expected.dtype),
        }
        if isinstance(actual, torch.Tensor):
            detail.update(actual_shape=list(actual.shape), actual_dtype=str(actual.dtype))
            if actual.shape == expected.shape and actual.numel():
                error = (actual.cpu().double() - expected.cpu().double()).abs()
                detail["max_abs"] = float(error.max()) if torch.isfinite(error).all() else None
        return {path: detail}
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return {path: "dictionary keys differ"}
        bad = {}
        for key in expected:
            bad.update(compare_tree(actual[key], expected[key], f"{path}/{key}"))
        return bad
    if isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return {path: "sequence differs"}
        bad = {}
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            bad.update(compare_tree(a, b, f"{path}/{i}"))
        return bad
    return {} if type(actual) is type(expected) and actual == expected else {path: "value differs"}


def checkpoint_semantics(ck):
    """Only nondeterministic elapsed time and serialization-container hash excluded."""
    result = copy.deepcopy(ck)
    result.pop("best_sha256", None)  # best snapshot tensors compared separately.
    for row in result["history"]:
        row["train"].pop("seconds")
    return result
