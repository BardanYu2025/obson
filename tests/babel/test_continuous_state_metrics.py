"""Synthetic physical-price and chronological boundaries for F11, no weights."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from obson.babel import continuous_state_metrics as metrics
from obson.babel import history_price_truth as truth


def windows(n=6, flat=False):
    values = np.full(n + 127, 100.0) if flat else 100 * np.exp(np.arange(n + 127) * 0.0003)
    return np.lib.stride_tricks.sliding_window_view(values, 128).copy()


def times(n=6):
    return [
        (datetime(2026, 1, 1, 9) + timedelta(minutes=15 * i)).isoformat(sep=" ") for i in range(n)
    ]


def test_perfect_reconstruction_is_not_claimed_three_class_success():
    raw = windows()
    path = truth.from_raw_closes(raw)["relative_log_close"]
    report = metrics.summarize_segment(path, raw, times(), 15)
    assert report["endpoints"] == 6
    for horizon in ("16", "64", "128"):
        assert report["price"][horizon]["endpoint_rmse_bps"]["max"] == 0
        assert report["price"][horizon]["scored_historical_points_per_endpoint"] == int(horizon) - 1
    state = report["state"]["16"]
    assert state["direction"]["accuracy"] == 1
    assert state["direction"]["recall"] == [None, None, 1.0]
    assert state["direction"]["missing_raw_classes"] == ["down", "mixed"]
    assert state["direction"]["balanced_accuracy_all_three"] is None
    assert state["continuity"]["transition_summary"]["hit_rate_observed_transitions"] is None
    assert state["continuity"]["error_run_bars"]["count"] == 0
    json.dumps(report, allow_nan=False)


def test_known_anchor_does_not_dilute_price_error_and_flat_failure_is_retained():
    raw = windows(flat=True)
    prediction = np.full_like(raw, 0.001)
    prediction[:, -1] = 0
    report = metrics.summarize_segment(prediction, raw, times(), 15)
    for horizon in ("16", "64", "128"):
        assert report["price"][horizon]["endpoint_rmse_bps"]["mean"] == pytest.approx(10)
    state = report["state"]["16"]
    assert state["groups"]["flat"]["direction"]["count"] == 6
    assert state["groups"]["low_including_flat"]["direction"]["count"] == 6
    assert state["groups"]["low_nonflat"]["direction"]["count"] == 0
    assert state["groups"]["above_low"]["volatility_rms_bps"]["mae"] is None
    assert state["direction"]["raw_support"] == [0, 6, 0]
    assert state["direction"]["model_support"] == [6, 0, 0]
    assert state["volatility_rms_bps"]["bias_model_minus_raw"] == pytest.approx(10 / np.sqrt(15))
    run = state["continuity"]["error_runs"][0]
    assert run["left_censored"] and run["right_censored"]
    assert run["bars"] == 6
    assert run["nominal_trading_minutes"] == 90
    json.dumps(report, allow_nan=False)


def test_low_nonflat_is_separate_and_threshold_not_changed():
    values = 100 * np.exp(np.arange(133) * 0.00005)
    raw = np.lib.stride_tricks.sliding_window_view(values, 128).copy()
    report = metrics.summarize_segment(
        truth.from_raw_closes(raw)["relative_log_close"], raw, times(), 15
    )
    for horizon in ("16", "64"):
        group = report["state"][horizon]["groups"]
        assert group["flat"]["direction"]["count"] == 0
        assert group["low_including_flat"]["direction"]["count"] == 6
        assert group["low_nonflat"]["direction"]["count"] == 6
    assert report["definitions"]["low_rms_log_return_le"] == 1e-4
    assert report["definitions"]["flat_rms_log_return_le"] == 1e-8


def test_class_support_confusion_and_majority_are_explicit():
    actual = np.array([0, 0, 1, 1, 1, 2])
    predicted = np.array([0, 1, 1, 1, 2, 2])
    result = metrics._classes(actual, predicted)
    assert result["confusion_rows_raw_columns_model"] == [[1, 1, 0], [0, 2, 1], [0, 0, 1]]
    assert result["recall"] == [0.5, 2 / 3, 1]
    assert result["balanced_accuracy_all_three"] == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert result["majority_baseline"]["class"] == "mixed"
    assert result["majority_baseline"]["accuracy"] == 0.5
    assert result["majority_baseline"]["balanced_accuracy_present_classes"] == pytest.approx(1 / 3)


def test_lag_false_switches_and_misses_not_zero_filled():
    actual = np.array([1, 1, 2, 2, 2, 0, 0, 1, 1])
    predicted = np.array([1, 0, 0, 2, 1, 1, 1, 0, 0])
    result = metrics._continuity(actual, predicted, list(map(datetime.fromisoformat, times(9))), 15)
    assert result["raw_switches"] == 3
    assert result["model_switches"] == 4
    assert result["false_switch_indices"] == [1, 3, 4]
    assert result["false_switches_per_100_bars"] == pytest.approx(100 / 3)
    summary = result["transition_summary"]
    assert summary["hit_count"] == 1
    assert summary["miss_count"] == 2
    assert summary["miss_before_next_raw_switch_count"] == 1
    assert summary["right_censored_miss_count"] == 1
    assert summary["hit_lag_bars"]["mean"] == 1
    assert result["transitions"][1]["lag_bars"] is None
    assert result["transitions"][2]["miss_right_censored"]
    assert [run["bars"] for run in result["error_runs"]] == [2, 5]
    assert result["error_runs"][-1]["right_censored"]


def test_first_hit_does_not_require_staying_in_the_new_class():
    actual = np.array([0, 2, 2, 2, 2])
    predicted = np.array([0, 2, 0, 2, 2])
    result = metrics._continuity(actual, predicted, list(map(datetime.fromisoformat, times(5))), 15)
    assert result["transitions"][0]["lag_bars"] == 0
    assert result["false_switch_indices"] == [2, 3]
    assert result["error_runs"][0]["bars"] == 1


def test_night_gap_is_not_counted_as_trading_minutes():
    timestamps = [
        "2026-01-01 14:45:00",
        "2026-01-01 15:00:00",
        "2026-01-02 09:00:00",
        "2026-01-02 09:15:00",
    ]
    actual = np.array([0, 2, 2, 2])
    predicted = np.array([2, 0, 0, 2])
    result = metrics._continuity(
        actual, predicted, list(map(datetime.fromisoformat, timestamps)), 15
    )
    event = result["transitions"][0]
    assert event["lag_bars"] == 2
    assert event["lag_nominal_minutes"] == 30
    assert event["lag_wall_minutes"] == 1095
    run = result["error_runs"][0]
    assert run["nominal_trading_minutes"] == 45
    assert run["wall_minutes"] == 1110
    assert run["gap_minutes"] == 1065
    raw = windows(4)
    report = metrics.summarize_segment(
        truth.from_raw_closes(raw)["relative_log_close"], raw, timestamps, 15
    )
    assert report["time"]["nominal_bar_minutes"] == 60
    assert report["time"]["gaps_between_observed_rows"] == 1
    assert report["time"]["gap_minutes"] == 1065


def test_single_endpoint_and_both_censored_error():
    raw = windows(1, flat=True)
    path = np.linspace(-0.1, 0, 128)[None]
    report = metrics.summarize_segment(path, raw, times(1), 15)
    continuity = report["state"]["16"]["continuity"]
    assert continuity["switch_opportunities"] == 0
    assert continuity["false_switches_per_100_bars"] == 0
    assert continuity["transitions"] == []
    run = continuity["error_runs"][0]
    assert run["left_censored"] and run["right_censored"]
    assert run["nominal_trading_minutes"] == 15


@pytest.mark.parametrize(
    "fault",
    [
        "precision",
        "order",
        "overlap_time",
        "timezone",
        "timestamp_count",
        "period",
        "missing_row",
        "nan",
        "anchor",
        "shape",
    ],
)
def test_rejects_wrong_physical_inputs(fault):
    raw = windows()
    path = truth.from_raw_closes(raw)["relative_log_close"]
    timestamps, period = times(), 15
    if fault == "precision":
        raw = raw.astype(np.float32)
    elif fault == "order":
        timestamps[2] = timestamps[1]
    elif fault == "overlap_time":
        timestamps[2] = "2026-01-01 09:16:00"
    elif fault == "timezone":
        timestamps[0] += "+08:00"
    elif fault == "timestamp_count":
        timestamps = timestamps[:-1]
    elif fault == "period":
        period = 5
    elif fault == "missing_row":
        raw[2, 10] += 1
    elif fault == "nan":
        path[2, 20] = np.nan
    elif fault == "anchor":
        path[:, -1] = 0.01
    elif fault == "shape":
        path = path[:-1]
    with pytest.raises(ValueError):
        metrics.summarize_segment(path, raw, timestamps, period)


def test_input_arrays_not_mutated():
    raw = windows()
    path = truth.from_raw_closes(raw)["relative_log_close"]
    raw_before, path_before = raw.copy(), path.copy()
    metrics.summarize_segment(path, raw, times(), 15)
    np.testing.assert_array_equal(raw, raw_before)
    np.testing.assert_array_equal(path, path_before)


def test_extreme_prediction_cannot_return_nonfinite_json():
    raw = windows()
    path = np.full_like(raw, 1e200)
    path[:, -1] = 0
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="Nonfinite derived"),
    ):
        metrics.summarize_segment(path, raw, times(), 15)
