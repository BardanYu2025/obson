"""Synthetic CPU parity with original batch formulas; no real-weight execution."""

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from obson.babel import closed_bar_state as stream
from obson.babel.activity_ablation import activity_features
from obson.babel.ae_context import encode_context
from obson.babel.data import validate_frame

KEY = "X/15/EX.X123"
ORIGIN = "synthetic-contract-file:start0"
STATS = {
    "x_mean": np.linspace(-0.3, 0.3, 28).tolist(),
    "x_scale": np.linspace(0.5, 1.5, 28).tolist(),
    "y_mean": [0.0] * 7,
}


def frame(n=421, oi_mode="open"):
    rng = np.random.default_rng(17)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.007, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0.0, 0.003, n))
    h = np.maximum(o, c) * 1.003
    lo = np.minimum(o, c) * 0.996
    times = pd.date_range("2026-01-02 09:00", periods=n, freq="15min").to_numpy(copy=True)
    times[60:] += np.timedelta64(2, "D")
    times[200:] += np.timedelta64(13, "h")
    volume = rng.integers(0, 200, n).astype(float)
    volume[::11] = 0
    values = {"datetime": times, "open": o, "high": h, "low": lo, "close": c, "volume": volume}
    if oi_mode != "missing":
        values["close_oi"] = rng.integers(900, 1200, n).astype(float)
        values["close_oi"][::19] = 0
        if oi_mode == "zero":
            values["close_oi"][:] = 0
    if oi_mode == "open":
        values["open_oi"] = rng.integers(800, 1100, n).astype(float)
        values["open_oi"][::7] = np.nan
        values["open_oi"][::13] = 0
        values["open_oi"][::23] = -1
    return validate_frame(pd.DataFrame(values))


def asof(bar):
    return pd.Timestamp(bar.datetime) + pd.Timedelta(minutes=bar.period)


def all_features(data):
    return np.column_stack(
        (encode_context(data, 15, "ema8_32")["x"], activity_features(data, 15)[0])
    )


def consume(data, obj=None):
    obj = obj or stream.PreparedBarStream(KEY, 15, STATS, ORIGIN)
    outputs = []
    features = []
    for b in stream.frame_to_bars(data, KEY, 15):
        output = obj.push(b, asof(b))
        features.append(obj.snapshot()["payload"]["features"][-1])
        outputs.append(output)
    return obj, np.asarray(features, np.float32), outputs


@pytest.mark.parametrize("oi_mode", ["open", "missing", "zero", "close"])
def test_all_channels_match_original_vectorized_features_across_gaps(oi_mode):
    data = frame(oi_mode=oi_mode)
    obj, actual, outputs = consume(data)
    expected = all_features(data)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=2e-6)
    assert all(x is None for x in outputs[:127])
    for end in (127, 199, 200, len(data) - 1):
        result = outputs[end]
        expected_window = (
            (expected[end - 127 : end + 1] - STATS["x_mean"]) / STATS["x_scale"]
        ).astype(np.float32)
        np.testing.assert_allclose(result["windows"][0], expected_window, atol=1e-7, rtol=2e-6)
        np.testing.assert_array_equal(result["raw_closes"][0], data.close.iloc[end - 127 : end + 1])
        assert result["windows"].dtype == np.float32
        assert result["raw_closes"].dtype == np.float64
        assert result["endpoints"][0]["row"] == end
        assert result["endpoints"][0]["end"] == str(data.datetime.iloc[end])
    assert obj.count == len(data)


def test_flat_prices_and_zero_activity_remain_physically_flat():
    data = frame(260, "zero")
    data[["open", "high", "low", "close"]] = 24425.0
    data.volume = 0.0
    _, actual, _ = consume(data)
    np.testing.assert_allclose(actual, all_features(data), atol=1e-7, rtol=2e-6)
    np.testing.assert_array_equal(actual[:, 14:18], 0)
    np.testing.assert_array_equal(actual[:, 18:27], 0)
    assert actual[-1, 27] == 1


@pytest.mark.parametrize("cut", [0, 1, 73, 128, 290])
def test_json_snapshot_restore_is_exact_for_long_prefix(cut):
    data = frame()
    original, _, _ = consume(data.iloc[:cut])
    snapshot = json.loads(json.dumps(original.snapshot(), allow_nan=False))
    restored = stream.PreparedBarStream.restore(snapshot, STATS, ORIGIN)
    for bar in stream.frame_to_bars(data.iloc[cut:], KEY, 15):
        a = original.push(bar, asof(bar))
        b = restored.push(bar, asof(bar))
        assert original.snapshot() == restored.snapshot()
        if a is not None:
            np.testing.assert_array_equal(a["windows"], b["windows"])
            np.testing.assert_array_equal(a["raw_closes"], b["raw_closes"])
    assert original.current()["row"] == len(data) - 1


def test_future_perturbation_does_not_change_saved_prefix_and_outputs_are_copies():
    data = frame()
    original, _, outputs = consume(data.iloc[:200])
    frozen = original.snapshot()
    saved = outputs[-1]["windows"].copy()
    altered = data.copy()
    altered.loc[200:, ["open", "high", "low", "close"]] *= 1.8
    other, _, other_outputs = consume(altered.iloc[:200])
    assert frozen == other.snapshot()
    np.testing.assert_array_equal(saved, other_outputs[-1]["windows"])
    consume(altered.iloc[200:], original)
    np.testing.assert_array_equal(saved, outputs[-1]["windows"])
    current = other.current()
    current["raw_bars"][-1]["close"] = -1
    current["raw_features"][:] = 0
    current["windows"][:] = 0
    assert frozen == other.snapshot()
    # EMA context uses the complete contract prefix, not a128-bar restart.
    reset_features = all_features(data.iloc[-128:].reset_index(drop=True))
    _, full, _ = consume(data)
    assert np.max(np.abs(full[-128:, 14:18] - reset_features[:, 14:18])) > 0.01


def test_all_rejected_bars_leave_state_unchanged():
    data = frame()
    obj, _, _ = consume(data.iloc[:150])
    bar = list(stream.frame_to_bars(data.iloc[150:151], KEY, 15))[0]
    baseline = obj.snapshot()
    invalid = [
        replace(bar, key="X/15/EX.OTHER"),
        replace(bar, period=30),
        replace(bar, closed=False),
        replace(bar, datetime=data.datetime.iloc[149]),
        replace(bar, datetime=data.datetime.iloc[149] + pd.Timedelta(minutes=5)),
        replace(bar, volume=-1),
        replace(bar, oi=-1),
        replace(bar, close=0),
        replace(bar, high=0),
        replace(bar, volume=np.inf),
        replace(bar, open=np.nan),
        replace(bar, datetime=pd.Timestamp(bar.datetime).tz_localize("Asia/Shanghai")),
        replace(bar, oi_available="false"),
    ]
    for bad in invalid:
        with pytest.raises(ValueError):
            obj.push(bad, asof(bar))
        assert baseline == obj.snapshot()
    for clock in (pd.Timestamp(bar.datetime), pd.NaT, pd.Timestamp(asof(bar)).tz_localize("UTC")):
        with pytest.raises(ValueError):
            obj.push(bar, clock)
        assert baseline == obj.snapshot()
    assert obj.push(bar, asof(bar)) is not None


def test_open_oi_and_consecutive_fallback_masks_match_original():
    data = frame(135, "missing")
    data["oi_available"] = True
    data["oi"] = 100.0
    data["open_oi"] = np.nan
    data.loc[0, "open_oi"] = 80.0
    data.loc[1, "oi"] = 0.0
    data.loc[60, "open_oi"] = 70.0
    _, features, _ = consume(data)
    expected = all_features(data)
    np.testing.assert_allclose(features, expected, atol=1e-7, rtol=2e-6)
    assert features[0, 26] == 1  # OpeningOI works on the very first bar.
    assert features[1, 23] == 0  # Zero closingOI is unavailable for a delta.
    assert features[2, 23] == 0  # Previouszero cannot be a denominator.
    assert features[3, 23] == 1  # Consecutive positive previousOI fallback.
    assert features[60, 26] == 1  # OpeningOI remains valid across a session gap.
    data.loc[60, "open_oi"] = np.nan
    _, changed, _ = consume(data)
    assert changed[60, 23] == 0  # A gap withoutopeningOI has no delta.


def test_snapshot_corruption_binding_and_invalid_structure_rejected():
    obj, _, _ = consume(frame())
    saved = obj.snapshot()
    changed_stats = copy.deepcopy(STATS)
    changed_stats["y_mean"][0] = 1
    for stats, origin in ((changed_stats, ORIGIN), (STATS, "different-origin")):
        with pytest.raises(ValueError, match="mismatch"):
            stream.PreparedBarStream.restore(saved, stats, origin)
    wrong = copy.deepcopy(saved)
    wrong["payload"]["features"][0][0] += 1
    with pytest.raises(ValueError, match="integrity"):
        stream.PreparedBarStream.restore(wrong, STATS, ORIGIN)
    for field, value in (
        ("count", 3),
        ("last_time", "2020-01-01"),
        ("codec", {"close": -1.0, "variance": 0.1}),
        ("previous_oi_valid", "yes"),
        ("features", [[0.0] * 28]),
    ):
        wrong = copy.deepcopy(saved)
        wrong["payload"][field] = value
        wrong["sha256"] = stream._token(wrong["payload"])
        with pytest.raises(ValueError):
            stream.PreparedBarStream.restore(wrong, STATS, ORIGIN)
    assert obj.snapshot() == saved


def test_statistics_and_key_validation_and_no_external_mutation():
    for key, period in (("continuous", 15), (KEY, 30), ("X/5/Y", 5), (KEY, True)):
        with pytest.raises(ValueError):
            stream.PreparedBarStream(key, period, STATS, ORIGIN)
    for bad in (
        {},
        {"x_mean": [0.0] * 28, "x_scale": [0.0] * 28},
        {"x_mean": [0.0] * 18, "x_scale": [1.0] * 18},
    ):
        with pytest.raises(ValueError):
            stream.PreparedBarStream(KEY, 15, bad, ORIGIN)
    stats = copy.deepcopy(STATS)
    obj = stream.PreparedBarStream(KEY, 15, stats, ORIGIN)
    signature = obj.statistics_signature
    stats["x_mean"][0] = 10
    obj.statistics["x_mean"][0] = 20
    assert obj.statistics_signature == signature
    assert obj.statistics == STATS
    with pytest.raises(ValueError, match="validated"):
        list(stream.frame_to_bars(pd.DataFrame({"close": [1.0]}), KEY, 15))


@pytest.mark.parametrize("error", [RuntimeError("synthetic inference failed"), KeyboardInterrupt()])
def test_reader_failure_rolls_back_cursor_then_same_bar_can_be_retried(error):
    class MockReader:
        statistics = STATS
        input_signature = stream.PreparedBarStream(KEY, 15, STATS, ORIGIN).input_signature
        fail = False
        calls = []

        def read_prepared(self, windows, endpoints, *, input_signature, audit_observed):
            assert input_signature == self.input_signature
            assert audit_observed is False
            self.calls.append(endpoints[0]["row"])
            if self.fail:
                raise error
            return {"row": endpoints[0]["row"], "state": windows[:, -1].copy()}

    reader = MockReader()
    wrapper = stream.ClosedBarHistoryReader(reader, KEY, 15, ORIGIN)
    bars = list(stream.frame_to_bars(frame(150), KEY, 15))
    for bar in bars[:127]:
        assert wrapper.push(bar, asof(bar)) is None
    assert reader.calls == []
    before = wrapper.stream.snapshot()
    reader.fail = True
    with pytest.raises(type(error)):
        wrapper.push(bars[127], asof(bars[127]))
    assert before == wrapper.stream.snapshot()
    reader.fail = False
    result = wrapper.push(bars[127], asof(bars[127]))
    assert result["estimate"]["row"] == 127
    assert result["prepared"]["count"] == 128
    assert reader.calls == [127, 127]
    reader.input_signature = "different"
    with pytest.raises(ValueError, match="normalization signatures"):
        stream.ClosedBarHistoryReader(reader, KEY, 15, ORIGIN)
