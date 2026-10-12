"""F01/F03 components for a physical, multiscale price principal objective.

The encoder/query interface is unchanged. A query supplies ages 1..127; its
physical log-percent prices are reversed and the known age-zero coordinate is
appended as exactly zero. No raw price or target is supplied to a model here.

Haar coordinates are DC, then details of widths 1, 2, 4, 8, 16, 32, 64. Positive
detail means the earlier half exceeds the later half. Coarse/fine are temporal
scales, not assertions about market signal/noise. Equal squared-error weights
in an orthonormal basis do not create a new objective (Parseval).
"""

from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional

from . import history_query as hq

SCHEMA = "babel-multiscale-content-v1"
LENGTH = 128
BAND_SLICES = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128))
BAND_SIZES = tuple(hi - lo for lo, hi in BAND_SLICES)
BAND_NAMES = ("dc", "detail1", "detail2", "detail4", "detail8", "detail16", "detail32", "detail64")
ARMS = ("control", "multiscale")
PRICE_WEIGHT = 0.6
FLOOR_RATIO = 1e-3


def _numpy_vector(value, name):
    array = np.asarray(value)
    if (
        array.ndim < 1
        or array.shape[-1] != LENGTH
        or not array.size
        or array.dtype.kind != "f"
        or array.dtype.itemsize not in (4, 8)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name}: finite nonempty float32/64 array ending in128 required")
    return array.astype(np.float64, copy=False)


def _tensor_vector(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 1
        or value.shape[-1] != LENGTH
        or not value.numel()
        or value.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name}: finite nonempty float32/64 tensor ending in128 required")
    return value


def _numpy_basis():
    """Explicit scalar definitions, independent of the Torch pair recursion."""
    basis = np.zeros((LENGTH, LENGTH), dtype=np.float64)
    basis[0] = 1 / np.sqrt(LENGTH)
    for lo, hi in BAND_SLICES[1:]:
        width = LENGTH // (hi - lo)
        for group in range(hi - lo):
            start = group * width
            basis[lo + group, start : start + width // 2] = 1 / np.sqrt(width)
            basis[lo + group, start + width // 2 : start + width] = -1 / np.sqrt(width)
    basis.setflags(write=False)
    return basis


_BASIS = _numpy_basis()


def haar_numpy(path):
    """Independent float64 mathematical transform; does not require a zero anchor."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = _numpy_vector(path, "path") @ _BASIS.T
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transformed Haar coefficients")
    return result


def inverse_haar_numpy(coefficients):
    with np.errstate(over="ignore", invalid="ignore"):
        result = _numpy_vector(coefficients, "coefficients") @ _BASIS
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite inverse Haar path")
    return result


def haar_torch(path):
    """Differentiable orthonormal transform, preserving floating dtype/device."""
    averages = _tensor_vector(path, "path")
    details = []
    while averages.shape[-1] > 1:
        early, late = averages[..., ::2], averages[..., 1::2]
        details.append((early - late) / np.sqrt(2.0))
        averages = (early + late) / np.sqrt(2.0)
    return _tensor_vector(torch.cat([averages, *reversed(details)], dim=-1), "Haar coefficients")


def inverse_haar_torch(coefficients):
    coefficients = _tensor_vector(coefficients, "coefficients")
    averages = coefficients[..., :1]
    for lo, hi in BAND_SLICES[1:]:
        detail = coefficients[..., lo:hi]
        averages = torch.stack((averages + detail, averages - detail), dim=-1).flatten(-2)
        averages = averages / np.sqrt(2.0)
    return _tensor_vector(averages, "inverse Haar path")


def _positive_delta(value):
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ValueError("Fixed positive finite training delta scale required")
    delta = float(value)
    if not np.isfinite(delta) or delta <= 0 or not np.isfinite(delta * FLOOR_RATIO):
        raise ValueError("Fixed positive finite training delta scale required")
    if delta * FLOOR_RATIO == 0:
        raise ValueError("Training delta scale floor underflows")
    return delta


def _statistics(statistics, *, nonprice=False):
    if not isinstance(statistics, Mapping):
        raise ValueError("Fixed training statistics mapping required")
    delta = np.asarray(statistics.get("delta_scale", []))
    if delta.ndim != 1 or not len(delta):
        raise ValueError("Training delta_scale[0] required")
    result = _positive_delta(delta[0])
    if nonprice:
        for name in ("y_mean", "y_scale"):
            value = np.asarray(statistics.get(name, []))
            if value.shape != (7,) or value.dtype.kind not in "ifu" or not np.isfinite(value).all():
                raise ValueError(f"Finite seven-field training {name} required")
            if name == "y_scale" and not (value > 0).all():
                raise ValueError("Positive training y_scale required")
    return result


def _query(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 2
        or value.shape[-2:] != (127, 7)
        or not value.numel()
        or value.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name}: finite nonempty float32/64 canonical127x7 queries required")
    return value


def query_path(query, statistics):
    """Query coordinate -> chronological128 log-percent prices, exact age0 zero.

    Float64 arithmetic matches the existing physical query metric. The zero is
    a coordinate convention, not a reconstructed or externally supplied close.
    """
    _statistics(statistics)
    query = _query(query, "query").double()
    history = query[..., 0] * hq.price_scales(statistics, query)
    return _tensor_vector(
        torch.cat((history.flip(-1), torch.zeros_like(history[..., :1])), dim=-1),
        "physical query path",
    )


def raw_price_targets(closes, positions, statistics):
    """Original float64 N x255 closes -> normalized age1..127 targets only.

    Logical p1..128 ends at block index p+126. Stable log1p is used near the
    anchor; log differences handle extreme positive price ratios. These are
    labels, never an input to the encoder or query.
    """
    _statistics(statistics)
    if (
        not isinstance(closes, torch.Tensor)
        or closes.ndim != 2
        or closes.shape[1] != 255
        or not len(closes)
        or closes.dtype != torch.float64
        or closes.requires_grad
        or not torch.isfinite(closes).all()
        or not (closes > 0).all()
    ):
        raise ValueError("Positive original float64 N x255 closes without gradients required")
    if (
        not isinstance(positions, torch.Tensor)
        or positions.ndim != 2
        or positions.shape[0] != len(closes)
        or positions.shape[1] == 0
        or positions.dtype != torch.long
        or positions.device != closes.device
        or not ((positions >= 1) & (positions <= 128)).all()
    ):
        raise ValueError("Matching integer N xP positions1..128 on the closes device required")
    ages = torch.arange(1, 128, device=closes.device)
    rows = torch.arange(len(closes), device=closes.device)[:, None, None]
    history = closes[rows, positions[..., None] + 126 - ages]
    anchor = closes[rows[..., 0], positions + 126][..., None]
    difference = history - anchor
    near = difference.abs() <= anchor * 0.5
    ratio = torch.where(near, difference, 0.0) / anchor
    physical = torch.where(near, torch.log1p(ratio), history.log() - anchor.log()) * 100
    result = physical / hq.price_scales(statistics, physical)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite normalized raw price labels")
    return result


def replace_price_targets(target, closes, positions, statistics):
    """Replace price labels only, retaining raw precision and all other fields."""
    target = _query(target, "target")
    price = raw_price_targets(closes, positions, statistics)
    if target.requires_grad or target.device != closes.device:
        raise ValueError("Fixed labels and raw closes on the same device required")
    if target.shape != (*positions.shape, 127, 7):
        raise ValueError("Target/position identities must have matching dimensions")
    result = target.double().clone()
    result[..., 0] = price
    return result


def _band_scales(scales):
    values = np.asarray(scales)
    if (
        values.shape != (8,)
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
        or not (values > 0).all()
    ):
        raise ValueError("Eight finite positive frozen floating band scales required")
    return values.astype(np.float64, copy=False)


def huber_coordinate_weights(scales):
    """Effective per-coefficient weights, in physical Haar log-percent units.

    SmoothL1 beta=1 has curvature weight/scale**2 inside |error|<scale,
    and saturated derivative weight/scale outside. These are NOT independent
    chronological-bar weights: inverse Haar mixes the price coordinates.
    """
    scales = _band_scales(scales)
    weight = PRICE_WEIGHT / (8 * np.asarray(BAND_SIZES, dtype=np.float64))
    with np.errstate(over="ignore", divide="ignore"):
        curvature, slope = weight / scales**2, weight / scales
    if not np.isfinite(curvature).all() or not np.isfinite(slope).all():
        raise ValueError("Band scales produce nonfinite effective Huber weights")
    return {
        "coordinate": "physical Haar coefficient, log-percent",
        "price_weight": PRICE_WEIGHT,
        "band_weights": [PRICE_WEIGHT / 8] * 8,
        "coefficients_per_band": list(BAND_SIZES),
        "normalized_coefficient_weights": weight.tolist(),
        "physical_huber_thresholds": scales.tolist(),
        "physical_quadratic_curvatures": curvature.tolist(),
        "physical_linear_slopes": slope.tolist(),
    }


class BandScaleAccumulator:
    """Streaming CPU RMS fit; caller must bind the actual training row inventory.

    RMS pools every coefficient in a band, without centering or per-window
    normalization. The mandatory split marker rejects accidental val/test use;
    it cannot independently certify the caller's row provenance or exposure.
    """

    def __init__(self, delta_scale):
        self.delta_scale = _positive_delta(delta_scale)
        self.contexts = 0
        self.updates = 0
        self.sum_squares = np.zeros(8, dtype=np.float64)

    def update(self, paths, *, split):
        if split != "train":
            raise ValueError("Band scales may only be fitted on locked training rows")
        original = np.asarray(paths)
        if original.dtype != np.float64:
            raise ValueError("Original float64 physical paths required for fitting")
        paths = _numpy_vector(original, "training paths")
        if np.any(paths[..., -1] != 0):
            raise ValueError("Training paths require the exact current zero anchor")
        coefficients = haar_numpy(paths).reshape(-1, LENGTH)
        with np.errstate(over="ignore", invalid="ignore"):
            addition = np.asarray(
                [np.square(coefficients[:, lo:hi]).sum() for lo, hi in BAND_SLICES]
            )
            total = self.sum_squares + addition
        if not np.isfinite(total).all():
            raise ValueError("Training coefficient sum of squares overflow")
        self.sum_squares = total
        self.contexts += len(coefficients)
        self.updates += 1

    def finalize(self):
        if self.contexts == 0:
            raise ValueError("Cannot fit band scales without training contexts")
        counts = [self.contexts * width for width in BAND_SIZES]
        rms = np.sqrt(self.sum_squares / counts)
        floor = self.delta_scale * FLOOR_RATIO
        scales = np.maximum(rms, floor)
        return {
            "schema": SCHEMA,
            "fit_split": "train",
            "contexts": self.contexts,
            "updates": self.updates,
            "bands": list(BAND_NAMES),
            "coefficients_per_context": list(BAND_SIZES),
            "counts": counts,
            "sum_squares": self.sum_squares.tolist(),
            "unfloored_rms": rms.tolist(),
            "scales": scales.tolist(),
            "floor": floor,
            "floor_ratio": FLOOR_RATIO,
            "delta_scale": self.delta_scale,
            "centered": False,
            "coordinate_weights": huber_coordinate_weights(scales),
        }


def _matched(pred, target, mask, statistics):
    _query(pred, "prediction")
    _query(target, "target")
    _statistics(statistics, nonprice=True)
    if (
        pred.shape != target.shape
        or not isinstance(mask, torch.Tensor)
        or mask.shape != pred.shape
        or mask.dtype != torch.bool
        or mask.device != pred.device
        or target.device != pred.device
        or target.requires_grad
        or not mask[..., 0].all()
    ):
        raise ValueError("Matched fixed labels/boolean masks and all127 prices valid required")


def _multiscale_bands(pred, target, statistics, scales, smooth):
    # Transforming the residual equals subtracting transformed paths. Transform
    # first and divide second: per-age normalization is not Haar normalization.
    error = haar_torch(query_path(pred, statistics) - query_path(target, statistics))
    values = _band_scales(scales)
    results = []
    for (lo, hi), scale in zip(BAND_SLICES, values, strict=True):
        residual = error[..., lo:hi] / float(scale)
        point = (
            functional.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
            if smooth
            else residual.square()
        )
        results.append(point.mean(-1))
    return torch.stack(results, dim=-1)


def loss_rows(pred, target, mask, statistics, scales, arm):
    """One loss per query context, without changing parent-window reduction.

    Both arms report the same old Q and Haar diagnostics. Only ``objective`` is
    selected by arm. ``*_mse`` diagnostics are detached; SmoothL1 terms retain
    their actual gradient paths for decomposition checks. No optimizer/model
    is constructed or invoked. All labels must be fixed, finite values.
    """
    if arm not in ARMS:
        raise ValueError(f"Expected a declared arm from {ARMS}")
    _matched(pred, target, mask, statistics)
    values = _band_scales(scales)
    if np.any(values < _statistics(statistics) * FLOOR_RATIO):
        raise ValueError("Band scales must retain the fixed training delta floor")
    # Validate curvature as well as positivity before a scale reaches a loss.
    huber_coordinate_weights(scales)
    original = hq.metrics(pred, target, mask, statistics, smooth=True)
    bands = _multiscale_bands(pred, target, statistics, scales, True)
    multiscale = bands.mean(-1)
    candidate = PRICE_WEIGHT * multiscale + 0.1 * original["body"] + 0.3 * original["activity"]
    with torch.no_grad():
        original_mse = hq.metrics(pred, target, mask, statistics, smooth=False)
        mse_bands = _multiscale_bands(pred, target, statistics, scales, False)
    result = {
        "objective": original["primary"] if arm == "control" else candidate,
        "query": original["primary"],
        "candidate": candidate,
        "path": original["path"],
        "change1": original["change1"],
        "body": original["body"],
        "activity": original["activity"],
        "multiscale": multiscale,
        "multiscale_bands": bands,
        "query_mse": original_mse["primary"],
        "multiscale_mse": mse_bands.mean(-1),
        "multiscale_mse_bands": mse_bands,
        "close_mae_bps": original_mse["close_mae_bps"],
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite objective or metric rows")
    return result


def detached_summary(rows):
    """JSON-safe logging without converting a requires-grad tensor to float."""
    result = {}
    for name, value in rows.items():
        if not isinstance(value, torch.Tensor) or not value.numel():
            raise ValueError("Nonempty tensor loss rows required")
        value = value.detach().double()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite metric: {name}")
        if name.endswith("_bands"):
            if value.shape[-1] != 8:
                raise ValueError("Eight diagnostic bands required")
            result[name] = value.reshape(-1, 8).mean(0).cpu().tolist()
        else:
            result[name] = value.mean().cpu().item()
    return result
