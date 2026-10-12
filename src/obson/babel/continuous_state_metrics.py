"""F11 descriptive continuity metrics, with no model execution or fitting.

A segment is one contract/period and consecutive physical bar rows. Direction
labels describe observed historical paths, not market regimes or future truth.
No runs or transitions may be joined across separately sampled segments.
"""

from datetime import datetime

import numpy as np

from . import history_price_truth as truth

SCHEMA = "babel-continuous-state-metrics-v1"
CLASS_NAMES = ("down", "mixed", "up")
PRICE_HORIZONS = (16, 64, 128)


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite derived metrics; input magnitude cannot be scored")
    if not values.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _classes(actual, predicted):
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    support = matrix.sum(axis=1)
    count = int(support.sum())
    recalls = [float(matrix[i, i] / support[i]) if support[i] else None for i in range(3)]
    present_recalls = [value for value in recalls if value is not None]
    majority = int(np.argmax(support)) if count else None
    return {
        "count": count,
        "class_order": list(CLASS_NAMES),
        "confusion_rows_raw_columns_model": matrix.tolist(),
        "raw_support": support.tolist(),
        "model_support": matrix.sum(axis=0).tolist(),
        "recall": recalls,
        "missing_raw_classes": [CLASS_NAMES[i] for i in range(3) if support[i] == 0],
        "accuracy": float(matrix.trace() / count) if count else None,
        "balanced_accuracy_present_classes": float(np.mean(present_recalls)) if count else None,
        "balanced_accuracy_all_three": float(np.mean(recalls)) if np.all(support) else None,
        "all_three_classes_supported": bool(np.all(support)),
        "majority_baseline": {
            "kind": "descriptive_majority_of_this_segment_not_a_fitted_deployment_rule",
            "class": CLASS_NAMES[majority] if majority is not None else None,
            "accuracy": float(support.max() / count) if count else None,
            "balanced_accuracy_present_classes": 1.0 / len(present_recalls) if count else None,
        },
    }


def _state_values(actual, predicted, raw_rms, model_rms, raw_efficiency, model_efficiency):
    rms_error = (model_rms - raw_rms) * 10000
    efficiency_error = model_efficiency - raw_efficiency
    return {
        "direction": _classes(actual, predicted),
        "volatility_rms_bps": {
            "mae": float(np.abs(rms_error).mean()) if len(actual) else None,
            "bias_model_minus_raw": float(rms_error.mean()) if len(actual) else None,
            "raw_mean": float(raw_rms.mean() * 10000) if len(actual) else None,
            "model_mean": float(model_rms.mean() * 10000) if len(actual) else None,
        },
        "efficiency_mae": float(np.abs(efficiency_error).mean()) if len(actual) else None,
    }


def _error_runs(wrong, times, period):
    changes = np.diff(np.concatenate(([False], wrong, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    result = []
    for start, stop in zip(starts, stops, strict=True):
        stop = int(stop)
        last = stop - 1
        start = int(start)
        bars = int(stop - start)
        nominal = bars * period
        wall = (times[last] - times[start]).total_seconds() / 60 + period
        result.append(
            {
                "start_index": start,
                "end_index_inclusive": last,
                "start_bar_time": times[start].isoformat(sep=" "),
                "end_bar_time": times[last].isoformat(sep=" "),
                "bars": bars,
                "nominal_trading_minutes": nominal,
                "wall_minutes": wall,
                "gap_minutes": max(0.0, wall - nominal),
                "left_censored": start == 0,
                "right_censored": stop == len(times),
            }
        )
    return result


def _continuity(actual, predicted, times, period):
    wrong = actual != predicted
    runs = _error_runs(wrong, times, period)
    raw_switch = np.flatnonzero(actual[1:] != actual[:-1]) + 1
    model_switch = np.flatnonzero(predicted[1:] != predicted[:-1]) + 1
    false_switch = model_switch[actual[model_switch] == actual[model_switch - 1]]
    transitions = []
    for i, start in enumerate(raw_switch):
        start = int(start)
        stop = int(raw_switch[i + 1]) if i + 1 < len(raw_switch) else len(actual)
        hits = np.flatnonzero(predicted[start:stop] == actual[start])
        hit = start + int(hits[0]) if len(hits) else None
        lag = hit - start if hit is not None else None
        transitions.append(
            {
                "switch_index": start,
                "from": CLASS_NAMES[int(actual[start - 1])],
                "to": CLASS_NAMES[int(actual[start])],
                "raw_regime_end_exclusive": stop,
                "observed_regime_bars": stop - start,
                "raw_regime_right_censored": stop == len(actual),
                "first_hit_index": hit,
                "hit": hit is not None,
                "lag_bars": lag,
                "lag_nominal_minutes": lag * period if lag is not None else None,
                "lag_wall_minutes": (times[hit] - times[start]).total_seconds() / 60
                if hit is not None
                else None,
                "miss_right_censored": hit is None and stop == len(actual),
            }
        )
    hits = [event for event in transitions if event["hit"]]
    misses = [event for event in transitions if not event["hit"]]
    return {
        "count": len(actual),
        "switch_opportunities": max(0, len(actual) - 1),
        "error_bars": int(wrong.sum()),
        "error_fraction": float(wrong.mean()),
        "error_runs": runs,
        "error_run_bars": _distribution([run["bars"] for run in runs]),
        "error_run_nominal_minutes": _distribution(
            [run["nominal_trading_minutes"] for run in runs]
        ),
        "error_run_wall_minutes": _distribution([run["wall_minutes"] for run in runs]),
        "raw_switches": int(len(raw_switch)),
        "model_switches": int(len(model_switch)),
        "false_switches": int(len(false_switch)),
        "false_switch_indices": false_switch.tolist(),
        "false_switches_per_100_bars": float(len(false_switch) * 100 / len(actual)),
        "raw_switches_per_100_bars": float(len(raw_switch) * 100 / len(actual)),
        "model_switches_per_100_bars": float(len(model_switch) * 100 / len(actual)),
        "transitions": transitions,
        "transition_summary": {
            "observed_raw_transitions": len(transitions),
            "hit_count": len(hits),
            "miss_count": len(misses),
            "miss_before_next_raw_switch_count": sum(
                not event["miss_right_censored"] for event in misses
            ),
            "right_censored_miss_count": sum(event["miss_right_censored"] for event in misses),
            "hit_rate_observed_transitions": len(hits) / len(transitions)
            if len(transitions)
            else None,
            "hit_lag_bars": _distribution([event["lag_bars"] for event in hits]),
            "hit_lag_nominal_minutes": _distribution(
                [event["lag_nominal_minutes"] for event in hits]
            ),
            "hit_lag_wall_minutes": _distribution([event["lag_wall_minutes"] for event in hits]),
            "initial_raw_regime_left_censored_excluded": True,
            "last_raw_regime_right_censored": True,
            "lag_is_conditional_on_hit_misses_not_zero": True,
        },
    }


def summarize_segment(predicted_path, raw_closes, timestamps, period):
    """Summarize one uninterrupted sequence of physical bar rows.

    Inputs contain consecutive rolling 128-close windows, oldest to newest, and
    prediction log paths relative to each window's known current close (zero).
    Timestamps are strictly increasing naive Asia/Shanghai endpoint bar starts;
    callers establish closure/as-of and single-contract identity. Session gaps
    remain allowed but are never counted as nominal trading minutes. A returned
    label disagreement is with a fixed causal historical descriptor, not proof
    of a wrong market-regime forecast. This function establishes no pass gate.
    """
    raw = truth.from_raw_closes(raw_closes)
    predicted = np.asarray(predicted_path, dtype=np.float64)
    model = truth.summarize_path(predicted)
    if predicted.shape != raw["relative_log_close"].shape:
        raise ValueError("Prediction and raw rows must match")
    if type(period) is not int or period not in (15, 30, 60):
        raise ValueError("Supported period is 15, 30 or 60 minutes")
    if len(timestamps) != len(predicted):
        raise ValueError("Exactly one timestamp is required per endpoint")
    times = [datetime.fromisoformat(value) for value in timestamps]
    if any(value.tzinfo is not None for value in times):
        raise ValueError("Use naive Asia/Shanghai bar-start times")
    intervals = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:], strict=False)]
    if any(interval < period for interval in intervals):
        raise ValueError("Bar starts must increase without overlapping the period")
    closes = np.asarray(raw_closes)
    if not np.array_equal(closes[1:, :-1], closes[:-1, 1:]):
        raise ValueError("Raw windows must be consecutive and share exactly 127 closes")
    errors = (predicted - raw["relative_log_close"]) * 10000
    price = {
        str(h): {
            "scored_historical_points_per_endpoint": h - 1,
            "endpoint_rmse_bps": _distribution(np.sqrt(np.mean(errors[:, -h:-1] ** 2, axis=1))),
        }
        for h in PRICE_HORIZONS
    }
    states = {}
    for j, horizon in enumerate(truth.HORIZONS):
        actual = raw["direction"][:, j]
        estimated = model["direction"][:, j]
        raw_rms, model_rms = raw["rms_log_return"][:, j], model["rms_log_return"][:, j]
        raw_eff = raw["descriptors"][:, j * 3]
        model_eff = model["descriptors"][:, j * 3]
        flat = raw_rms <= truth.FLOOR
        low = raw_rms <= truth.LOW_RMS
        groups = {
            "flat": flat,
            "low_including_flat": low,
            "low_nonflat": low & ~flat,
            "above_low": ~low,
        }
        states[str(horizon)] = {
            **_state_values(actual, estimated, raw_rms, model_rms, raw_eff, model_eff),
            "groups": {
                name: _state_values(
                    actual[mask],
                    estimated[mask],
                    raw_rms[mask],
                    model_rms[mask],
                    raw_eff[mask],
                    model_eff[mask],
                )
                for name, mask in groups.items()
            },
            "continuity": _continuity(actual, estimated, times, period),
        }
    return {
        "schema": SCHEMA,
        "endpoints": len(predicted),
        "period_minutes": period,
        "start_bar_time": times[0].isoformat(sep=" "),
        "end_bar_time": times[-1].isoformat(sep=" "),
        "time": {
            "nominal_bar_minutes": len(times) * period,
            "wall_coverage_minutes": (times[-1] - times[0]).total_seconds() / 60 + period,
            "gaps_between_observed_rows": sum(interval > period for interval in intervals),
            "gap_minutes": float(sum(max(0.0, interval - period) for interval in intervals)),
        },
        "definitions": {
            "direction": "raw historical signed efficiency at fixed -0.2/+0.2; down/mixed/up; mixed is not certified range",
            "volatility": "RMS of 15/63 historical log returns, expressed in basis points",
            "flat_rms_log_return_le": truth.FLOOR,
            "low_rms_log_return_le": truth.LOW_RMS,
            "anchor": "current known close is excluded from all price errors",
            "continuity": "within this segment only; no joining independent segments",
            "false_switch": "model class changes while raw descriptor class stays unchanged at that bar",
            "lag": "first new-class hit from a raw transition until the next raw transition; no smoothing; misses separate",
            "minutes": "nominal counts bars times period; wall time separately includes observed session/missing-time gaps",
            "interpretation": "descriptive research metrics; no scientific acceptance or trading gate",
        },
        "price": price,
        "state": states,
    }
