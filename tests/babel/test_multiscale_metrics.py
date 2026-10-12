import numpy as np
import pytest

from obson.babel.multiscale_metrics import full_position_rows, price_rows, raw_paths


def test_raw_truth_future_and_small_variation():
    close = np.full((2, 255), 100_000.0, dtype=np.float64)
    close[0, 127] += 1e-8
    positions = np.array([1, 2, 128])
    truth = raw_paths(close, positions)
    assert truth.shape == (2, 3, 128)
    assert truth[0, 0, -1] == 0
    assert truth[0, 0, -2] < 0
    assert not truth[1].any()
    changed = close.copy()
    changed[:, 128:] *= 10
    np.testing.assert_array_equal(raw_paths(changed, positions)[:, 0], truth[:, 0])
    np.testing.assert_allclose(truth[0, 0, -2], -100 * np.log1p(1e-8 / 100_000), rtol=1e-3)


def test_physical_bands_have_explicit_unweighted_support():
    truth = np.zeros((1, 1, 128), dtype=np.float64)
    pred = truth.copy()
    pred[..., :63] = 3
    pred[..., 63:111] = 2
    pred[..., 111:127] = 1
    result = price_rows(pred, truth)
    assert result["near"].item() == 1
    assert result["mid"].item() == 4
    assert result["far"].item() == 9
    assert result["composite"].item() == pytest.approx(14 / 3)


def test_p128_cannot_be_hidden_in_full_average():
    truth = np.zeros((3, 128, 128), dtype=np.float64)
    pred = truth.copy()
    pred[:, -1, :-1] = 8
    report = full_position_rows(price_rows(pred, truth), np.arange(1, 129))
    np.testing.assert_array_equal(report["endpoint/composite"], [64] * 3)
    np.testing.assert_array_equal(report["interior/composite"], [0] * 3)
    np.testing.assert_array_equal(report["full/composite"], [0.5] * 3)


@pytest.mark.parametrize("positions", [np.arange(128), np.arange(1, 129)[::-1], [1, 128]])
def test_incomplete_or_misordered_inventory_not_full_position_report(positions):
    with pytest.raises(ValueError, match="each position"):
        full_position_rows({}, positions)


def test_raw_float32_and_nonfinite_are_rejected():
    for close in (np.ones((1, 255), np.float32), np.full((1, 255), np.nan)):
        with pytest.raises(ValueError, match="float64"):
            raw_paths(close, np.array([1]))


def test_anchor_is_not_a_scored_output():
    truth = np.zeros((1, 1, 128))
    pred = truth.copy()
    pred[..., -1] = 1
    with pytest.raises(ValueError, match="current-zero"):
        price_rows(pred, truth)


def test_price_metric_is_in_log_percent_squared_not_basis_point_rmse():
    truth = np.zeros((1, 1, 128))
    pred = truth.copy()
    pred[..., :-1] = 0.01  # one bp = .01 log-percent, approximately.
    assert price_rows(pred, truth)["composite"].item() == pytest.approx(1e-4)


def test_parent_shapes_must_match_across_metrics():
    values = {k: np.ones((2, 128)) for k in ("near", "mid", "far", "composite")}
    values["near"] = np.ones((3, 128))
    with pytest.raises(ValueError, match="Matched"):
        full_position_rows(values, np.arange(1, 129))
