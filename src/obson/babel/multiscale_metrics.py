"""Physical price reporting for the planned F01/F03 objective comparison.

CPU arithmetic only. These summaries are not the full acceptance evaluator:
body/activity protection, paired uncertainty and source binding belong to the
future runner. The observed current zero coordinate is never scored.
"""

import numpy as np

BANDS = {"near": (1, 16), "mid": (17, 64), "far": (65, 127)}


def raw_paths(closes, positions):
    """Independent chronological log-percent truth from source float64 prices."""
    closes = np.asarray(closes)
    positions = np.asarray(positions)
    if (
        closes.dtype != np.float64
        or closes.ndim != 2
        or closes.shape[1] != 255
        or not np.isfinite(closes).all()
        or (closes <= 0).any()
        or positions.ndim != 1
        or not np.issubdtype(positions.dtype, np.integer)
        or not len(positions)
        or (positions < 1).any()
        or (positions > 128).any()
    ):
        raise ValueError("Positive float64 close255 and integer positions1..128 required")
    windows = np.stack([closes[:, p - 1 : p + 127] for p in positions], axis=1)
    anchor = windows[..., -1:]
    # log1p retains genuinely tiny differences; no inverse of rounded features.
    relative = (windows - anchor) / anchor
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        small = np.log1p(np.clip(relative, -0.5, 0.5))
        wide = np.log(windows) - np.log(anchor)
    result = 100.0 * np.where(np.abs(relative) <= 0.5, small, wide)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite raw price coordinate")
    result[..., -1] = 0.0
    return result


def price_rows(predicted, truth):
    """Per-parent/per-position MSE; equal physical age bands, no age discount."""
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if (
        predicted.shape != truth.shape
        or predicted.ndim != 3
        or predicted.shape[-1] != 128
        or min(predicted.shape) == 0
        or not np.issubdtype(predicted.dtype, np.floating)
        or truth.dtype != np.float64
        or not np.isfinite(predicted).all()
        or not np.isfinite(truth).all()
        or np.any(predicted[..., -1] != 0)
        or np.any(truth[..., -1] != 0)
    ):
        raise ValueError(
            "Matched finite chronological paths and exact current-zero anchor required"
        )
    with np.errstate(over="ignore"):
        error = (predicted.astype(np.float64) - truth) ** 2
    if not np.isfinite(error).all():
        raise ValueError("Price squared error overflow")
    result = {
        name: error[..., 127 - hi : 128 - lo].mean(axis=-1) for name, (lo, hi) in BANDS.items()
    }
    result["composite"] = np.stack(list(result.values()), axis=-1).mean(axis=-1)
    return result


def full_position_rows(values, positions):
    """Keep full, interior and live endpoint separate; do not dilute p128 risk."""
    positions = np.asarray(positions)
    if positions.dtype.kind not in "iu" or not np.array_equal(positions, np.arange(1, 129)):
        raise ValueError("Primary report requires each position1..128 once, in order")
    if set(values) != {*BANDS, "composite"}:
        raise ValueError("All physical price bands and composite required")
    output = {}
    shape = None
    for name, array in values.items():
        array = np.asarray(array)
        if (
            array.ndim != 2
            or not len(array)
            or array.shape[1] != 128
            or not np.isfinite(array).all()
            or (array < 0).any()
            or (shape is not None and array.shape != shape)
        ):
            raise ValueError("Matched finite nonnegative per-position error rows required")
        shape = array.shape
        output[f"full/{name}"] = array.mean(axis=1)
        output[f"interior/{name}"] = array[:, :-1].mean(axis=1)
        output[f"endpoint/{name}"] = array[:, -1].copy()
    return output
