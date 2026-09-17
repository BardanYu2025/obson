"""Metrics with explicit event identity, one-to-one matches and block uncertainty."""

import numpy as np


def event_counts(predicted, truth, tolerance=0):
    """Sorted positions from ONE contract/scale/kind. Maximum ordered matching.

    Repeated predictions cost precision; callers must never flatten contracts.
    """
    p, g = sorted(predicted), sorted(truth)
    i = j = tp = 0
    while i < len(p) and j < len(g):
        if p[i] < g[j] - tolerance:
            i += 1
        elif p[i] > g[j] + tolerance:
            j += 1
        else:
            tp += 1
            i += 1
            j += 1
    return tp, len(p) - tp, len(g) - tp


def event_f1(groups, tolerance=0):
    counts = (
        np.sum([event_counts(p, g, tolerance) for p, g in groups], axis=0)
        if groups
        else np.zeros(3)
    )
    tp, fp, fn = (int(v) for v in counts)
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def balanced_accuracy(pred, truth, classes):
    pred, truth = np.asarray(pred).ravel(), np.asarray(truth).ravel()
    support = [int((truth == c).sum()) for c in range(classes)]
    recalls = [
        float((pred[truth == c] == c).mean()) if support[c] else None for c in range(classes)
    ]
    present = [r for r in recalls if r is not None]
    return {
        "ba": float(np.mean(present)) if present else None,
        "recall": recalls,
        "support": support,
    }


def block_interval(values, blocks, seed=42, repeats=1000):
    """Equal-weight block mean bootstrap; windows are not independent samples."""
    values, blocks = np.asarray(values), np.asarray(blocks)
    means = np.array([values[blocks == b].mean() for b in np.unique(blocks)])
    if len(means) < 5:
        return {
            "mean": float(means.mean()) if len(means) else None,
            "low": None,
            "high": None,
            "blocks": len(means),
            "status": "insufficient_blocks",
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, (repeats, len(means)), replace=True).mean(1)
    return {
        "mean": float(means.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "blocks": len(means),
        "status": "estimated",
    }
