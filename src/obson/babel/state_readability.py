"""F11: fixed transformations of existing historical decoder outputs; no fitting."""

import numpy as np

from . import state_coverage_evaluate as ev
from . import utility_probe as up


def close_path(prediction, statistics):
    """Age1..127 scaled log-percent offsets -> chronological relative log closes."""
    p = np.asarray(prediction, dtype=np.float64)
    scale = float(statistics["delta_scale"][0])
    if p.ndim != 3 or p.shape[1:] != (127, 7) or not np.isfinite(p).all() or not scale > 0:
        raise ValueError("Expected finite N x127x7 query outputs and positive price scale")
    offsets = p[:, :, 0] * (scale * np.sqrt(np.arange(1, 128)))
    return np.concatenate((offsets[:, ::-1], np.zeros((len(p), 1))), axis=1) / 100


def descriptors(path):
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 128 or not np.isfinite(path).all():
        raise ValueError("Expected finite chronological N x128 log closes")
    result = []
    for h in (16, 64):
        p = path[:, -h:]
        r = np.diff(p, axis=1)
        total = np.abs(r).sum(1)
        efficiency = np.divide(r.sum(1), total, out=np.zeros(len(p)), where=total > 1e-12)
        centered = p - p.mean(1, keepdims=True)
        t = np.arange(h, dtype=float) - (h - 1) / 2
        den = np.square(centered).sum(1) * np.square(t).sum()
        r2 = np.divide(np.square(centered @ t), den, out=np.zeros(len(p)), where=den > 1e-24)
        vol = np.log(np.maximum(np.sqrt(np.square(r).mean(1)), 1e-8))
        result.extend((efficiency, np.clip(r2, 0, 1), vol))
    return np.column_stack(result)


def true_path(raw):
    changes = np.sinh(np.asarray(raw, dtype=float)[..., :2]).sum(-1) / 100
    path = np.cumsum(changes, axis=1)
    return path - path[:, -1:]


def mismatch_indices(rows):
    """Half-rotation within contract/period, preserving inventory, never self-pairs."""
    by = {}
    for i, row in enumerate(rows):
        by.setdefault(row["key"], []).append(i)
    ids = np.full(len(rows), -1, dtype=np.int64)
    for group in by.values():
        group.sort(key=lambda i: (rows[i]["end"], rows[i]["row"]))
        if len(group) > 1:
            ids[group] = np.roll(group, len(group) // 2)
    return ids


def measure(pred, y, statistics):
    p, y = np.asarray(pred, float), np.asarray(y, float)
    scale = np.asarray(statistics["scale"], float)[:6]
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 6 or not len(p):
        raise ValueError("Matched nonempty six-target matrices required")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or not (scale > 0).all():
        raise ValueError("Nonfinite inputs or invalid train scales")
    error = np.square((p - y) / scale)
    fields = []
    for i, name in enumerate(up.NAMES[:6]):
        variance = float(y[:, i].var())
        mse = float(np.square(p[:, i] - y[:, i]).mean())
        fields.append(
            {
                "name": name,
                "nmse": float(error[:, i].mean()),
                "r2": 1 - mse / variance if variance > 1e-12 else None,
                "bias": float((p[:, i] - y[:, i]).mean()),
            }
        )
    errors = {k: error[:, idx].mean(1) for k, idx in up.GROUPS.items()}
    return {
        "windows": len(p),
        "targets": fields,
        "groups": {k: float(v.mean()) for k, v in errors.items()},
        "direction": {
            str(h): up.probe.classification(p[:, j], y[:, j]) for h, j in ((16, 0), (64, 3))
        },
    }, errors


def comparison(a, b, rows, factor):
    return {
        k: ev.gate(a[k], b[k], rows, factor if k == "utility" else 1.1)
        for k in ("utility", "direction", "volatility")
    }


def decide(scores, errors, rows, mismatched, stats, truth):
    result = {}
    valid = np.flatnonzero(mismatched >= 0)
    for seed in (42, 43):
        name = f"D_s{seed}"
        d = errors[name + "/decoded"]
        route = comparison(d, errors[name + "/direct"], rows, 0.95)
        absolute = all(
            scores[name + "/decoded"]["targets"][i]["r2"] is not None
            and scores[name + "/decoded"]["targets"][i]["r2"] >= 0.5
            for i in up.GROUPS["utility"]
        )
        mean = ev.gate(d["utility"], errors["train_mean"]["utility"], rows, 0.95)
        if len(valid):
            _, wrong = measure(
                stats["decoded"][name][mismatched[valid]], truth[valid], stats["target"]
            )
            wrong_gate = ev.gate(
                d["utility"][valid], wrong["utility"], [rows[i] for i in valid], 0.95
            )
        else:
            wrong_gate = {"passed": False, "reason": "no non-self matched-contract controls"}
        result[str(seed)] = {
            "route_checks": route,
            "decoder_route_advantage": all(x["passed"] for x in route.values()),
            "four_r2_at_least_half": absolute,
            "versus_train_mean": mean,
            "versus_wrong_state": wrong_gate,
            "recoverability_qualification": bool(
                absolute and mean["passed"] and wrong_gate["passed"]
            ),
            "decoded_versus_A": comparison(d, errors[f"A_s{seed}/decoded"], rows, 0.95),
            "decoded_versus_Macro": comparison(d, errors[f"macro_s{seed}/decoded"], rows, 1.05),
            "original_exit_qualification_changed": False,
            "promoted": False,
        }
    return {
        "seeds": result,
        "wrong_state_included": len(valid),
        "wrong_state_excluded": len(rows) - len(valid),
    }
