"""Synthetic arithmetic only; no model inference or training."""

import numpy as np
import pytest

from obson.babel import overlap_content_audit as audit
from obson.babel.history_price_truth import from_raw_closes


def paired(shift):
    # Sharp movement is intentional: a flat consensus must not pass as accurate.
    log = np.sin(np.arange(128 + shift) * 0.21) * 0.015
    log[100:] += 0.02
    close = 100 * np.exp(log)
    return [from_raw_closes(x[None])["relative_log_close"] for x in (close[:128], close[shift:])]


@pytest.mark.parametrize("shift", audit.SHIFTS)
def test_physical_partition_and_exact_reconstruction(shift):
    ids_a, ids_b = np.arange(128), np.arange(shift, 128 + shift)
    s = audit.segments(shift)
    np.testing.assert_array_equal(ids_a[s["common_a"]], ids_b[s["common_b"]])
    np.testing.assert_array_equal(np.r_[ids_a[s["leaving_a"]], ids_a[s["common_a"]]], ids_a[:-1])
    np.testing.assert_array_equal(np.r_[ids_b[s["common_b"]], ids_b[s["entered_b"]]], ids_b[:-1])
    assert ids_a[-1] in ids_b[s["entered_b"]]
    a, b = paired(shift)
    output = audit.compare(a, b, a, b, shift)
    for name in (
        "shape_a_mse",
        "shape_b_mse",
        "entered_b_mse",
        "leaving_a_mse",
        "common_anchored_a_mse",
        "common_anchored_b_mse",
        "disagreement_mse",
    ):
        np.testing.assert_allclose(output[name], 0, atol=1e-28)


@pytest.mark.parametrize("shift", audit.SHIFTS)
def test_identically_wrong_is_not_success(shift):
    a, b = paired(shift)
    zero = np.zeros_like(a)
    out = audit.compare(zero, zero, a, b, shift)
    assert out["disagreement_mse"][0] == 0
    assert out["consensus_shape_mse"][0] > 0
    np.testing.assert_allclose(out["mean_shape_mse"], out["flat_shape_baseline_mse"])
    assert out["entered_b_mse"][0] > 0 and out["leaving_a_mse"][0] > 0


def test_error_identity_and_offset_protection():
    a, b = paired(16)
    rng = np.random.default_rng(8)
    pa, pb = a + rng.normal(0, 0.01, a.shape), b + rng.normal(0, 0.02, b.shape)
    pa[:, -1] = pb[:, -1] = 0
    out = audit.compare(pa, pb, a, b, 16)
    np.testing.assert_allclose(out["identity_residual"], 0, atol=1e-18)
    for arm in ("a", "b"):
        np.testing.assert_allclose(
            out[f"common_anchored_{arm}_mse"],
            out[f"shape_{arm}_mse"] + out[f"common_offset_{arm}_mse"],
            atol=1e-18,
        )
    # A constant wrong level remains a real anchored error even with perfect shape.
    pa, pb = a.copy(), b.copy()
    pa[:, :-1] += 0.02
    pb[:, :-1] -= 0.03
    out = audit.compare(pa, pb, a, b, 16)
    np.testing.assert_allclose(out["disagreement_mse"], 0, atol=1e-28)
    np.testing.assert_allclose(out["common_anchored_a_mse"], 0.02**2)
    np.testing.assert_allclose(out["common_anchored_b_mse"], 0.03**2)


def test_later_state_can_correct_error_without_forcing_old_answer():
    a, b = paired(64)
    pa = a.copy()
    pa[:, :-1] += np.linspace(0, 0.1, 127)
    out = audit.compare(pa, b, a, b, 64)
    assert out["disagreement_mse"][0] > 0
    assert out["shape_a_mse"][0] > 0
    np.testing.assert_allclose(out["shape_b_mse"], 0, atol=1e-28)


def test_reject_misaligned_truth_and_invalid_input():
    a, b = paired(16)
    wrong = b.copy()
    wrong[:, 10] += 0.001
    with pytest.raises(ValueError, match="physical prices"):
        audit.compare(a, b, a, wrong, 16)
    for shift in (0, 8, 127, True):
        with pytest.raises(ValueError, match="shifts"):
            audit.compare(a, b, a, b, shift)
    with pytest.raises(ValueError, match="zero"):
        audit.compare(a + 1, b, a, b, 16)
    with pytest.raises(ValueError, match="paired rows"):
        audit.compare(np.repeat(a, 2, axis=0), b, a, b, 16)
