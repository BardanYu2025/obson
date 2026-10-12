"""Independent arithmetic and synthetic-gradient checks, no neural weights."""

import json
import math
import warnings

import numpy as np
import pytest
import torch

from obson.babel import history_query as hq
from obson.babel import multiscale_content as mc


def scalar_haar(path):
    """Definition by scalar sums, not either implementation's operations."""
    result = [sum(float(v) for v in path) / math.sqrt(128)]
    for groups in (1, 2, 4, 8, 16, 32, 64):
        width = 128 // groups
        for group in range(groups):
            start = group * width
            result.append(
                (
                    sum(float(path[j]) for j in range(start, start + width // 2))
                    - sum(float(path[j]) for j in range(start + width // 2, start + width))
                )
                / math.sqrt(width)
            )
    return np.asarray(result)


def statistics():
    return {
        "delta_scale": [0.3, 0.4, 0.5],
        "y_mean": np.arange(7) / 10,
        "y_scale": np.arange(7) + 1.0,
    }


def example():
    rng = np.random.default_rng(51)
    target = torch.tensor(rng.normal(0, 0.3, (2, 3, 127, 7)), dtype=torch.float64)
    pred = (target + torch.tensor(rng.normal(0, 0.05, target.shape))).requires_grad_()
    mask = torch.ones_like(pred, dtype=torch.bool)
    return pred, target, mask, statistics(), np.linspace(0.2, 1.6, 8)


@pytest.mark.parametrize("kind", ["ramp", "impulse", "constant", "random"])
def test_numpy_and_torch_match_independent_scalar_definition(kind):
    data = {
        "ramp": np.arange(128, dtype=np.float64),
        "impulse": np.eye(1, 128, 95, dtype=np.float64)[0],
        "constant": np.full(128, 3.0),
        "random": np.random.default_rng(14).normal(size=128),
    }[kind]
    expected = scalar_haar(data)
    np.testing.assert_allclose(mc.haar_numpy(data), expected, atol=5e-13, rtol=1e-13)
    np.testing.assert_allclose(
        mc.haar_torch(torch.from_numpy(data)).numpy(), expected, atol=5e-13, rtol=1e-13
    )
    assert mc.BAND_SIZES == (1, 1, 2, 4, 8, 16, 32, 64)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-6), (torch.float64, 3e-14)])
def test_inverse_parseval_arbitrary_batch_and_gradients(dtype, tol):
    values = torch.tensor(
        np.random.default_rng(0).normal(size=(2, 3, 128)), dtype=dtype, requires_grad=True
    )
    coefficients = mc.haar_torch(values)
    torch.testing.assert_close(mc.inverse_haar_torch(coefficients), values, atol=tol, rtol=tol)
    torch.testing.assert_close(
        coefficients.square().sum(-1), values.square().sum(-1), atol=tol, rtol=tol
    )
    coefficients.square().sum().backward()
    torch.testing.assert_close(values.grad, 2 * values.detach(), atol=tol * 3, rtol=tol * 3)
    array = values.detach().double().numpy()
    np.testing.assert_allclose(mc.inverse_haar_numpy(mc.haar_numpy(array)), array, atol=2e-14)


def test_plain_mse_is_unchanged_not_new_information_and_scale_retains_amplitude():
    rng = np.random.default_rng(50)
    a, b = rng.normal(size=(3, 128)), rng.normal(size=(3, 128))
    transformed = mc.haar_numpy(a) - mc.haar_numpy(b)
    np.testing.assert_allclose(np.mean(transformed**2, axis=-1), np.mean((a - b) ** 2, axis=-1))
    np.testing.assert_allclose(mc.haar_numpy(7 * a), 7 * mc.haar_numpy(a), atol=2e-14)
    scales = np.arange(1, 9, dtype=np.float64)
    weighted = (
        sum(
            np.mean((transformed[:, lo:hi] / scale) ** 2, axis=-1)
            for (lo, hi), scale in zip(mc.BAND_SLICES, scales, strict=True)
        )
        / 8
    )
    assert not np.allclose(weighted, np.mean((a - b) ** 2, axis=-1))


def test_current_anchor_and_age1_latest_return_order():
    stats = statistics()
    pred = torch.zeros(1, 127, 7, dtype=torch.float64)
    physical = torch.arange(1, 128, dtype=torch.float64) * -0.2
    pred[0, :, 0] = physical / hq.price_scales(stats, pred)
    path = mc.query_path(pred, stats)
    torch.testing.assert_close(path[0, :-1], physical.flip(0), atol=1e-14, rtol=1e-14)
    assert path[0, -1].item() == 0
    assert (path[0, -1] - path[0, -2]).item() == pytest.approx(0.2)
    # Even a large nonprice current-like value cannot become the anchor.
    pred[..., 1:] = 12345
    torch.testing.assert_close(mc.query_path(pred, stats), path, atol=0, rtol=0)


def test_raw_targets_match_direct_log_ratios_at_boundaries_and_are_causal():
    stats = statistics()
    values = 100 * np.exp(np.random.default_rng(4).normal(0, 0.002, (2, 255)).cumsum(1))
    closes = torch.tensor(values, dtype=torch.float64)
    ps = torch.tensor([[1, 2, 31, 32, 127, 128]] * 2)
    target = mc.raw_price_targets(closes, ps, stats)
    for n in range(2):
        for k, position in enumerate(ps[n].tolist()):
            for age in (1, 16, 64, 127):
                end = position + 126
                answer = (
                    100
                    * math.log(values[n, end - age] / values[n, end])
                    / (math.sqrt(age) * stats["delta_scale"][0])
                )
                assert target[n, k, age - 1].item() == pytest.approx(answer, abs=1e-12)
    for position in (1, 31, 128):
        one = torch.full((2, 1), position)
        before = mc.raw_price_targets(closes, one, stats)
        changed = closes.clone()
        changed[:, position + 127 :] *= 3
        torch.testing.assert_close(
            mc.raw_price_targets(changed, one, stats), before, atol=0, rtol=0
        )


def test_raw_flat_nearflat_and_extreme_ratios_are_finite():
    closes = torch.full((1, 255), 100.0, dtype=torch.float64)
    ps = torch.tensor([[1, 128]])
    assert not mc.raw_price_targets(closes, ps, statistics()).any()
    closes[0, 127] += 1e-12
    value = mc.raw_price_targets(closes, ps[:, :1], statistics())[0, 0, 0]
    expected = math.log1p((100.0 - closes[0, 127].item()) / closes[0, 127].item()) * 100 / 0.3
    assert value.item() == expected
    closes[0, 0], closes[0, 127] = 1e-300, 1e300
    assert torch.isfinite(mc.raw_price_targets(closes, ps[:, :1], statistics())).all()


def test_price_replacement_preserves_nonprice_precision_and_original():
    target = torch.randn(2, 2, 127, 7)
    original = target.clone()
    closes = torch.full((2, 255), 200.0, dtype=torch.float64)
    result = mc.replace_price_targets(target, closes, torch.tensor([[1, 128]] * 2), statistics())
    assert result.dtype == torch.float64
    assert not result[..., 0].any()
    torch.testing.assert_close(result[..., 1:], original[..., 1:].double(), atol=0, rtol=0)
    torch.testing.assert_close(target, original, atol=0, rtol=0)


def test_streaming_train_rms_counts_floor_and_effective_huber_weights():
    paths = np.random.default_rng(9).normal(size=(5, 2, 128))
    paths[..., -1] = 0
    fit = mc.BandScaleAccumulator(0.3)
    fit.update(paths[:2], split="train")
    fit.update(paths[2:], split="train")
    result = fit.finalize()
    independent = np.stack([scalar_haar(row) for row in paths.reshape(-1, 128)])
    expected = [np.sqrt(np.mean(independent[:, lo:hi] ** 2)) for lo, hi in mc.BAND_SLICES]
    np.testing.assert_allclose(result["scales"], expected, atol=1e-14)
    assert result["counts"] == [10 * n for n in mc.BAND_SIZES]
    assert result["contexts"] == 10 and result["updates"] == 2
    weights = result["coordinate_weights"]
    for band, count in enumerate(mc.BAND_SIZES):
        assert weights["normalized_coefficient_weights"][band] == 0.6 / (8 * count)
        assert weights["physical_quadratic_curvatures"][band] == pytest.approx(
            0.6 / (8 * count * expected[band] ** 2)
        )
        assert weights["physical_linear_slopes"][band] == pytest.approx(
            0.6 / (8 * count * expected[band])
        )
    json.dumps(result, allow_nan=False)
    flat = mc.BandScaleAccumulator(0.3)
    flat.update(np.zeros((4, 128)), split="train")
    assert flat.finalize()["scales"] == [0.0003] * 8


def test_fitting_is_train_only_and_failed_update_does_not_mutate():
    fit = mc.BandScaleAccumulator(0.3)
    with pytest.raises(ValueError, match="without training"):
        fit.finalize()
    for split in ("val", "test", "cross_research", None):
        with pytest.raises(ValueError, match="training"):
            fit.update(np.zeros((2, 128)), split=split)
    with pytest.raises(TypeError):
        fit.update(np.zeros((2, 128)))
    for path in (np.zeros((2, 128), np.float32), np.ones((2, 128)), np.zeros((0, 128))):
        with pytest.raises(ValueError):
            fit.update(path, split="train")
    assert fit.contexts == 0 and fit.updates == 0 and not fit.sum_squares.any()


def test_both_arms_report_same_metrics_control_exactly_reuses_old_q():
    pred, target, mask, stats, scales = example()
    mask[0, :, 20:40, 4:] = False
    old = hq.metrics(pred, target, mask, stats, True)
    control = mc.loss_rows(pred, target, mask, stats, scales, "control")
    candidate = mc.loss_rows(pred, target, mask, stats, scales, "multiscale")
    torch.testing.assert_close(control["objective"], old["primary"], atol=0, rtol=0)
    for name in control:
        if name != "objective":
            torch.testing.assert_close(control[name], candidate[name], atol=0, rtol=0)
    torch.testing.assert_close(
        candidate["objective"],
        0.6 * candidate["multiscale"] + 0.1 * old["body"] + 0.3 * old["activity"],
        atol=0,
        rtol=0,
    )
    assert control["objective"].shape == (2, 3)
    assert control["multiscale_bands"].shape == (2, 3, 8)
    assert not control["query_mse"].requires_grad
    assert not control["multiscale_mse_bands"].requires_grad


@pytest.mark.parametrize("arm", mc.ARMS)
def test_float32_predictions_with_raw_float64_labels_backpropagate(arm):
    pred, old_target, mask, stats, scales = example()
    pred = pred.detach().float().requires_grad_()
    closes = torch.linspace(100.0, 102.0, 255, dtype=torch.float64).repeat(2, 1)
    positions = torch.tensor([[1, 64, 128]] * 2)
    target = mc.replace_price_targets(old_target.float(), closes, positions, stats)
    rows = mc.loss_rows(pred, target, mask, stats, scales, arm)
    assert rows["objective"].dtype == torch.float64
    rows["objective"].mean().backward()
    assert pred.grad.dtype == torch.float32
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    json.dumps(mc.detached_summary(rows), allow_nan=False)


def test_actual_multiscale_loss_matches_independent_scalar_huber():
    pred, target, mask, stats, scales = example()
    rows = mc.loss_rows(pred, target, mask, stats, scales, "multiscale")
    physical_scale = np.sqrt(np.arange(1, 128)) * stats["delta_scale"][0]
    delta = ((pred.detach().numpy() - target.numpy())[..., 0] * physical_scale)[..., ::-1]
    paths = np.concatenate((delta, np.zeros((2, 3, 1))), axis=-1)
    independent = np.empty((2, 3, 8))
    for n in range(2):
        for p in range(3):
            coefficients = scalar_haar(paths[n, p])
            for band, (lo, hi) in enumerate(mc.BAND_SLICES):
                errors = coefficients[lo:hi] / scales[band]
                independent[n, p, band] = np.mean(
                    [0.5 * e * e if abs(e) < 1 else abs(e) - 0.5 for e in errors]
                )
    np.testing.assert_allclose(rows["multiscale_bands"].detach().numpy(), independent, atol=2e-15)


def test_price_gradient_matches_independent_physical_haar_huber_derivative():
    stats = statistics()
    target = torch.zeros((1, 127, 7), dtype=torch.float64)
    pred = target.clone()
    pred[0, :, 0] = torch.linspace(-3.0, 4.0, 127)
    pred.requires_grad_()
    scales = np.linspace(0.2, 1.6, 8)
    rows = mc.loss_rows(
        pred, target, torch.ones_like(pred, dtype=torch.bool), stats, scales, "multiscale"
    )
    actual = torch.autograd.grad(rows["objective"].sum(), pred)[0][0, :, 0].numpy()
    basis = np.stack([scalar_haar(row) for row in np.eye(128)]).T
    age_scale = np.sqrt(np.arange(1, 128)) * stats["delta_scale"][0]
    path = np.r_[(pred.detach().numpy()[0, :, 0] * age_scale)[::-1], 0.0]
    coefficients = scalar_haar(path)
    coefficient_gradient = np.empty(128)
    for (lo, hi), scale in zip(mc.BAND_SLICES, scales, strict=True):
        coefficient_gradient[lo:hi] = (
            0.6 * np.clip(coefficients[lo:hi] / scale, -1, 1) / (8 * (hi - lo) * scale)
        )
    expected = (basis.T @ coefficient_gradient)[:-1][::-1] * age_scale
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("arm", mc.ARMS)
def test_finite_gradients_actual_component_sum_and_missing_activity(arm):
    pred, target, mask, stats, scales = example()
    mask[..., 2:] = False
    rows = mc.loss_rows(pred, target, mask, stats, scales, arm)
    assert torch.all(rows["activity"] == 0)
    if arm == "control":
        terms = [
            0.4 * rows["path"],
            0.2 * rows["change1"],
            0.1 * rows["body"],
            0.3 * rows["activity"],
        ]
    else:
        terms = [0.6 * rows["multiscale"], 0.1 * rows["body"], 0.3 * rows["activity"]]
    pieces = [torch.autograd.grad(term.mean(), pred, retain_graph=True)[0] for term in terms]
    actual = torch.autograd.grad(rows["objective"].mean(), pred)[0]
    torch.testing.assert_close(actual, sum(pieces), atol=3e-17, rtol=1e-12)
    assert torch.isfinite(actual).all() and actual[..., :2].abs().sum() > 0
    assert not actual[..., 2:].any()


def test_zero_loss_and_detach_safe_logging():
    pred, target, mask, stats, scales = example()
    rows = mc.loss_rows(target.clone().requires_grad_(), target, mask, stats, scales, "multiscale")
    assert all(torch.count_nonzero(value) == 0 for value in rows.values())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = mc.detached_summary(mc.loss_rows(pred, target, mask, stats, scales, "multiscale"))
    assert isinstance(summary["objective"], float) and len(summary["multiscale_bands"]) == 8
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(127),
        np.zeros((0, 128)),
        np.full(128, np.nan),
        np.zeros(128, dtype=int),
        np.zeros(128, dtype=np.float16),
    ],
)
def test_haar_rejects_malformed_inputs(bad):
    with pytest.raises(ValueError):
        mc.haar_numpy(bad)
    with pytest.raises(ValueError):
        mc.inverse_haar_numpy(bad)
    with pytest.raises(ValueError):
        mc.haar_torch(torch.as_tensor(bad))
    with pytest.raises(ValueError):
        mc.inverse_haar_torch(torch.as_tensor(bad))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, [1], "0.3"])
def test_scale_accumulator_rejects_invalid_delta(bad):
    with pytest.raises(ValueError):
        mc.BandScaleAccumulator(bad)


def test_loss_rejects_invalid_masks_scales_labels_or_arm():
    pred, target, mask, stats, scales = example()
    mask[..., 0, 0] = False
    with pytest.raises(ValueError, match="all127"):
        mc.loss_rows(pred, target, mask, stats, scales, "control")
    mask[..., 0, 0] = True
    for invalid in (
        np.zeros(8),
        np.ones(7),
        np.full(8, np.inf),
        np.ones(8, dtype=int),
        np.full(8, 1e-300),
    ):
        with pytest.raises(ValueError):
            mc.loss_rows(pred, target, mask, stats, invalid, "multiscale")
    with pytest.raises(ValueError):
        mc.loss_rows(pred, target, mask.float(), stats, scales, "control")
    with pytest.raises(ValueError):
        mc.loss_rows(pred, target.requires_grad_(), mask, stats, scales, "control")
    with pytest.raises(ValueError):
        mc.loss_rows(pred, target, mask, stats, scales, "unknown")


def test_raw_target_rejects_float32_invalid_positions_and_wrong_shapes():
    raw = torch.ones((2, 255), dtype=torch.float64)
    positions = torch.tensor([[1, 128]] * 2)
    for invalid in (raw.float(), raw[:, :128], raw * 0, raw.clone().requires_grad_()):
        with pytest.raises(ValueError):
            mc.raw_price_targets(invalid, positions, statistics())
    for invalid in (
        positions.float(),
        positions * 0,
        positions + 1,
        positions[:1],
        positions[:, :0],
    ):
        with pytest.raises(ValueError):
            mc.raw_price_targets(raw, invalid, statistics())
    with pytest.raises(ValueError):
        mc.replace_price_targets(torch.zeros(2, 3, 127, 7), raw, positions, statistics())


def test_finite_inputs_that_overflow_or_violate_frozen_floor_are_rejected():
    huge = np.full(128, 1e308)
    with pytest.raises(ValueError, match="Nonfinite"):
        mc.haar_numpy(huge)
    with pytest.raises(ValueError):
        mc.haar_torch(torch.from_numpy(huge))
    pred, target, mask, stats, scales = example()
    scales[0] = 0.000299
    with pytest.raises(ValueError, match="floor"):
        mc.loss_rows(pred, target, mask, stats, scales, "multiscale")
