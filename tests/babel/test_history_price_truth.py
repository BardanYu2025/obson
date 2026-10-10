"""Physical-price regressions; no real-model execution."""

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pytest

from obson.babel import history_price_truth as truth


def test_exact_flat_is_neutral_even_when_legacy_roundtrip_looks_up():
    # Reproduces the measured scale of inverse-feature residual, not a new label floor.
    fake = np.arange(128, dtype=float)[None, :] * 1.4987721719088298e-12
    fake -= fake[:, -1:]
    assert truth.summarize_path(fake)["direction"].tolist() == [[2, 2]]
    actual = truth.from_raw_closes(np.full((1, 128), 24425.0))
    np.testing.assert_array_equal(actual["relative_log_close"], 0)
    np.testing.assert_array_equal(actual["descriptors"][:, [0, 1, 3, 4]], 0)
    np.testing.assert_array_equal(actual["rms_log_return"], 0)
    assert actual["direction"].tolist() == [[1, 1]]
    result = truth.audit_prediction(fake, np.full((1, 128), 24425.0))
    assert result["direction_correct"].tolist() == [[False, False]]
    assert (result["path_rmse"] > 0).all()


def test_known_log_ramp_and_scale_invariance():
    slope = np.array([0.0005, -0.002, 0.0])[:, None]
    closes = 24425 * np.exp(np.arange(128)[None, :] * slope)
    original = truth.from_raw_closes(closes)
    np.testing.assert_allclose(
        original["descriptors"][:, [0, 1, 3, 4]],
        [[1, 1, 1, 1], [-1, 1, -1, 1], [0, 0, 0, 0]],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        original["rms_log_return"], np.repeat(abs(slope), 2, axis=1), atol=1e-15
    )
    for scale in (1e-80, 0.125, 1e80):
        scaled = truth.from_raw_closes(closes * scale)
        np.testing.assert_allclose(
            scaled["relative_log_close"], original["relative_log_close"], atol=1e-14
        )
        np.testing.assert_allclose(scaled["descriptors"], original["descriptors"], atol=1e-12)
        np.testing.assert_array_equal(scaled["direction"], original["direction"])


def test_tiny_real_moves_are_not_artificially_flattened():
    close = 1 + np.arange(128, dtype=float)[None, :] * 1e-11
    out = truth.from_raw_closes(close)
    assert np.isfinite(out["descriptors"]).all()
    assert (out["rms_log_return"] > 0).all()
    assert out["direction"].tolist() == [[2, 2]]
    assert out["stratum"].tolist() == [[0, 0]]  # Low amplitude is separate from direction.


def test_extreme_positive_prices_are_finite():
    closes = np.geomspace(1e-300, 1e300, 128)[None, :]
    with np.errstate(all="raise"):
        out = truth.from_raw_closes(closes)
    assert np.isfinite(out["relative_log_close"]).all()
    assert np.isfinite(out["descriptors"]).all()


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((1, 127)),
        np.ones((0, 128)),
        np.ones((1, 128), dtype=np.float32),
        np.zeros((1, 128)),
        -np.ones((1, 128)),
        np.full((1, 128), np.nan),
        np.full((1, 128), np.inf),
        np.full((1, 128), "1"),
    ],
)
def test_reject_invalid_prices(bad):
    with pytest.raises(ValueError):
        truth.from_raw_closes(bad)


def test_audit_does_not_change_prediction_and_keeps_model_errors():
    close = np.exp(np.arange(128)[None, :] * 0.001)
    model = np.zeros((1, 128))
    before = model.copy()
    result = truth.audit_prediction(model, close)
    np.testing.assert_array_equal(model, before)
    assert (result["path_rmse"] > 0).all()
    assert not result["direction_correct"].any()
    with pytest.raises(ValueError, match="rows"):
        truth.audit_prediction(np.zeros((2, 128)), close)
    with pytest.raises(ValueError, match="anchored"):
        truth.audit_prediction(np.ones((1, 128)), close)


def fixture_csv(tmp_path, count=130):
    directory = tmp_path / "X"
    directory.mkdir(exist_ok=True)
    path = directory / "TEST_15m.csv"
    start = datetime(2026, 1, 1)
    times = [start + timedelta(minutes=15 * i) for i in range(count)]
    lines = ["datetime,close\n"] + [f"{t.isoformat(' ')},{100 + i}\n" for i, t in enumerate(times)]
    path.write_text("".join(lines))
    row = {"key": "X/15/TEST", "row": 127, "end": times[127].isoformat(" ")}
    return path, times, row


def loader(root, path):
    return truth.RawCloseWindows(root, {"X/15/TEST": hashlib.sha256(path.read_bytes()).hexdigest()})


def test_csv_identity_time_cutoff_and_future_suffix(tmp_path):
    path, times, row = fixture_csv(tmp_path)
    reader = loader(tmp_path, path)
    close_time = times[127] + timedelta(minutes=15)
    expected = reader.window(row, asof=close_time)
    np.testing.assert_array_equal(expected, np.arange(100, 228))
    # Changing prices strictly after the endpoint does not change its raw truth.
    lines = path.read_text().splitlines()
    lines[-1] = f"{times[-1].isoformat(' ')},99999999"
    path.write_text("\n".join(lines) + "\n")
    newer_snapshot = loader(tmp_path, path)
    np.testing.assert_array_equal(newer_snapshot.window(row, asof=close_time), expected)
    with pytest.raises(ValueError, match="fingerprint"):
        truth.RawCloseWindows(tmp_path, reader.inventory).window(row, asof=close_time)
    with pytest.raises(ValueError, match="not closed"):
        reader.window(row, asof=close_time - timedelta(microseconds=1))
    with pytest.raises(ValueError, match="timestamp differ"):
        reader.window({**row, "end": times[126].isoformat()}, asof=close_time)
    for index in (126, 130, True):
        with pytest.raises(ValueError, match="exact row"):
            reader.window({**row, "row": index}, asof=close_time)
    with pytest.raises(ValueError, match="naive"):
        reader.window(row, asof=f"{close_time.isoformat()}+08:00")


def test_duplicate_timestamp_and_bad_price_rejected(tmp_path):
    path, times, row = fixture_csv(tmp_path)
    lines = path.read_text().splitlines()
    lines[2] = lines[1]
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="strictly increasing"):
        loader(tmp_path, path).window(row, asof=times[-1])
    path, times, row = fixture_csv(tmp_path)
    lines = path.read_text().splitlines()
    lines[100] = lines[100].split(",")[0] + ",nan"
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="positive finite"):
        loader(tmp_path, path).window(row, asof=times[-1])
