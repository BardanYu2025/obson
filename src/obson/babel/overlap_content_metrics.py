"""F03 paired content accounting; descriptive diagnostics, never model selection."""

import numpy as np

from . import overlap_content_audit as geometry
from .history_price_truth import from_raw_closes
from .state_readability import close_path

POSITIONS = (64, 112, 127, 128)
FIELDS = (
    "path",
    "body",
    "volume_vs_prior_ema32",
    "delta_log_volume",
    "oi_change_percent_asinh",
    "oi_per_volume_asinh",
    "turnover_asinh",
)


def masked_mean(values, mask):
    counts = mask.sum(1)
    return np.divide(
        np.where(mask, values, 0).sum(1),
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    ), counts


def paired_rows(prediction, target, mask, raw_close, statistics):
    """One row per original parent; keep masks/support, not imputed zero errors.

    Query arrays use age1..127; physical price arrays are chronological. All
    non-price channels share the original train coordinate across query ages.
    Current B is never predicted; A's current bar belongs to B's entered region.
    """
    p, y, m = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    n = len(p)
    if p.shape != (n, 4, 127, 7) or y.shape != p.shape or m.shape != p.shape:
        raise ValueError("Matched N x4x127x7 queries required")
    if m.dtype != bool or not n or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Finite queries and boolean masks required")
    if np.asarray(raw_close).shape != (n, 4, 128):
        raise ValueError("Matched raw close geometry required")
    paths = close_path(p.reshape(-1, 127, 7), statistics).reshape(n, 4, 128)
    truth = from_raw_closes(np.asarray(raw_close).reshape(-1, 128))["relative_log_close"]
    truth = truth.reshape(n, 4, 128)
    bank = {}
    for j, pos in enumerate(POSITIONS):
        for band, (lo, hi) in zip(
            ("near", "mid", "far"), ((1, 16), (17, 64), (65, 127)), strict=True
        ):
            err, count = masked_mean(
                (p[:, j, lo - 1 : hi] - y[:, j, lo - 1 : hi]) ** 2, m[:, j, lo - 1 : hi]
            )
            for field, name in enumerate(FIELDS):
                bank[f"source_scale/p{pos}/{band}/{name}"] = err[:, field]
                bank[f"support/p{pos}/{band}/{name}"] = count[:, field]
    # Chronological non-price fields: normalized by fixed source train scales.
    p, y, m = p[:, :, ::-1, 1:].astype(float), y[:, :, ::-1, 1:].astype(float), m[:, :, ::-1, 1:]
    for d in geometry.SHIFTS:
        a, b = POSITIONS.index(128 - d), 3
        s = geometry.segments(d)
        ca, cb = s["common_a"], s["common_b"]
        if not np.array_equal(m[:, a, ca], m[:, b, cb]) or not np.array_equal(
            y[:, a, ca], y[:, b, cb]
        ):
            raise ValueError("Common non-price targets/masks differ")
        price = geometry.compare(paths[:, a], paths[:, b], truth[:, a], truth[:, b], d)
        for name, value in price.items():
            if isinstance(value, np.ndarray):
                bank[f"shift{d}/price/{name}"] = value
                if name.endswith("_mse"):
                    bank[f"shift{d}/price/{name[:-4]}_rmse_bps"] = np.sqrt(value) * 10000
        for region, index, part in (
            ("common_a", a, ca),
            ("common_b", b, cb),
            ("entered_b", b, s["entered_b"]),
            ("leaving_a", a, s["leaving_a"]),
        ):
            delta = paths[:, index, part] - truth[:, index, part]
            bank[f"shift{d}/price/{region}_mae_bps"] = np.abs(delta).mean(1) * 10000
            bank[f"shift{d}/price/{region}_signed_offset_bps"] = delta.mean(1) * 10000
        for region, index, part in (
            ("common_a", a, ca),
            ("common_b", b, cb),
            ("entered_b", b, s["entered_b"]),
            ("leaving_a", a, s["leaving_a"]),
        ):
            err, counts = masked_mean(
                (p[:, index, part] - y[:, index, part]) ** 2, m[:, index, part]
            )
            baseline, _ = masked_mean(y[:, index, part] ** 2, m[:, index, part])
            for f, name in enumerate(FIELDS[1:]):
                key = f"shift{d}/{name}/{region}"
                bank[key + "_mse"] = err[:, f]
                bank[key + "_baseline_mse"] = baseline[:, f]
                bank[key + "_support"] = counts[:, f]
        difference, _ = masked_mean((p[:, a, ca] - p[:, b, cb]) ** 2, m[:, a, ca])
        for f, name in enumerate(FIELDS[1:]):
            bank[f"shift{d}/{name}/disagreement_mse"] = difference[:, f]
        prefix = f"shift{d}/price/"
        for left, right, name in (
            ("shape_b_mse", "shape_a_mse", "later_minus_earlier"),
            ("mean_shape_mse", "flat_shape_baseline_mse", "common_minus_flat"),
            ("entered_b_mse", "flat_entered_baseline_mse", "entered_minus_flat"),
            ("leaving_a_mse", "flat_leaving_baseline_mse", "leaving_minus_flat"),
        ):
            bank[prefix + name] = bank[prefix + left] - bank[prefix + right]
    return bank


def intervals(matrix, rows, draws=2000):
    """Paired parent means, resampling whole calendar clusters with replacement.

    NaN means no valid field in that parent. It is excluded from both sums and
    denominators; we never turn missing targets into zero-error observations.
    P95 below is descriptive; no P95 significance or model promotion is claimed.
    """
    x = np.asarray(matrix, float)
    if x.ndim != 2 or len(x) != len(rows) or not len(x) or np.isinf(x).any():
        raise ValueError("Finite-or-missing paired parent matrix required")
    result = {}
    for kind, labels in (
        ("month", [r["month"] for r in rows]),
        ("contract_period_month", [r["key"] + "/" + r["month"] for r in rows]),
    ):
        _, codes = np.unique(labels, return_inverse=True)
        groups = int(codes.max()) + 1
        valid = np.isfinite(x)
        sums = np.zeros((groups, x.shape[1]))
        counts = np.zeros_like(sums)
        np.add.at(sums, codes, np.where(valid, x, 0))
        np.add.at(counts, codes, valid)
        samples = np.random.default_rng(20261010).integers(0, groups, (draws, groups))
        weights = np.stack([np.bincount(sample, minlength=groups) for sample in samples]).astype(
            float
        )
        denominator = weights @ counts
        boot = np.divide(
            weights @ sums,
            denominator,
            out=np.full(denominator.shape, np.nan),
            where=denominator > 0,
        )
        pieces = []
        for j in range(x.shape[1]):
            values = x[valid[:, j], j]
            supported_groups = int((counts[:, j] > 0).sum())
            finite_draws = boot[np.isfinite(boot[:, j]), j]
            ci = np.quantile(finite_draws, [0.025, 0.975]) if len(finite_draws) else [None, None]
            pieces.append(
                {
                    "mean": float(values.mean()) if len(values) else None,
                    "low": None if ci[0] is None else float(ci[0]),
                    "high": None if ci[1] is None else float(ci[1]),
                    "parents": len(values),
                    "groups": supported_groups,
                    "valid_draws": len(finite_draws),
                    "supported": bool(
                        len(values) >= 100 and supported_groups >= 6 and len(finite_draws) == draws
                    ),
                }
            )
        result[kind] = pieces
    return result


def summarize(bank, rows):
    distribution = {}
    contrasts = {}
    for name, values in bank.items():
        values = np.asarray(values)
        valid = np.isfinite(values)
        v = values[valid]
        distribution[name] = {
            "parents": int(valid.sum()),
            "missing_parents": int((~valid).sum()),
            "mean": float(v.mean()) if len(v) else None,
            "p50": float(np.quantile(v, 0.5)) if len(v) else None,
            "p95": float(np.quantile(v, 0.95)) if len(v) else None,
        }
        if name.endswith(
            ("later_minus_earlier", "common_minus_flat", "entered_minus_flat", "leaving_minus_flat")
        ):
            contrasts[name] = values
    for d in geometry.SHIFTS:
        for field in FIELDS[1:]:
            base = f"shift{d}/{field}/"
            contrasts[base + "later_minus_earlier"] = (
                bank[base + "common_b_mse"] - bank[base + "common_a_mse"]
            )
            for region in ("entered_b", "leaving_a"):
                contrasts[base + region + "_minus_mean"] = (
                    bank[base + region + "_mse"] - bank[base + region + "_baseline_mse"]
                )
    names = list(contrasts)
    ci = intervals(np.column_stack(list(contrasts.values())), rows)
    accounting = {}
    for d in geometry.SHIFTS:
        base = f"shift{d}/price/"
        total = float(np.mean(bank[base + "mean_shape_mse"]))
        shared = float(np.mean(bank[base + "consensus_shape_mse"]))
        differing = float(np.mean(bank[base + "disagreement_mse"])) / 4
        accounting[str(d)] = {
            "total_mean_shape_mse": total,
            "consensus_error_component": shared,
            "disagreement_component": differing,
            "consensus_fraction": shared / total if total > 0 else None,
            "disagreement_fraction": differing / total if total > 0 else None,
            "interpretation": "arithmetic decomposition, not causal attribution or an online ensemble",
        }
    return {
        "error_accounting": accounting,
        "distributions": distribution,
        "paired_mean_intervals": {
            name: {kind: vals[j] for kind, vals in ci.items()} for j, name in enumerate(names)
        },
        "scope": "repeated research parents; P95 descriptive, no quality gates, no winner selection",
        "units": "physical price MSE in squared log units; source_scale and nonprice in squared training coordinates",
        "p95_unit": "distribution of parent-window mean errors, not pooled individual bars",
    }
