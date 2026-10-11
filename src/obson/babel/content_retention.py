"""F03: matched physical-content supervision through the existing state query.

No new reader, latent alignment, future target, or raw-input shortcut. The extra
objective changes allocation of known-history supervision, not available facts.
"""

import numpy as np
import torch
from torch.nn import functional

from . import state_coverage as source

ARMS = ("control", "content", "aligned")
SEEDS = (42, 43)
TRAIN_SHIFTS = (1, 16)
POSITIONS = (1, 32, 64, 80, 96, 112, 127, 128)
CONTENT_WEIGHT, ALIGN_WEIGHT = 0.10, 0.01


def schedule(n, seed, epoch):
    if n < 1 or seed not in SEEDS or epoch < 1:
        raise ValueError("Positive inventory/epoch and a declared seed required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261012]))
    order = rng.permutation(n)
    shifts = np.asarray(TRAIN_SHIFTS)[rng.integers(0, len(TRAIN_SHIFTS), n)]
    later = rng.integers(shifts + 1, 129)
    return order, np.column_stack((later - shifts, later)).astype(np.int64)


def validate_pairs(ps):
    if ps.ndim != 2 or ps.shape[1] != 2 or ps.dtype != torch.long or not len(ps):
        raise ValueError("Nonempty integer N x2 endpoints required")
    shift = ps[:, 1] - ps[:, 0]
    if (
        not ((ps >= 1) & (ps <= 128)).all()
        or not torch.isin(shift, shift.new_tensor((1, 16, 64))).all()
    ):
        raise ValueError("Ordered complete-context endpoints and declared shifts required")
    return shift


def raw_targets(closes, ps, statistics):
    """Float64 CSV closes -> original age coordinate; labels only, no model call."""
    if closes.shape != (len(ps), 255) or closes.dtype != torch.float64:
        raise ValueError("Original float64 N x255 closes required")
    if not torch.isfinite(closes).all() or not (closes > 0).all():
        raise ValueError("Positive finite raw closes required")
    if ps.dtype != torch.long or ps.ndim != 2 or not ((ps >= 1) & (ps <= 128)).all():
        raise ValueError("Causal endpoint positions required")
    ages = torch.arange(1, 128, device=ps.device)
    ii = ps[..., None] + 126 - ages
    rows = torch.arange(len(ps), device=ps.device)[:, None, None]
    history = closes[rows, ii]
    anchor = closes[torch.arange(len(ps), device=ps.device)[:, None], ps + 126]
    offsets = torch.log1p((history - anchor[..., None]) / anchor[..., None]) * 100
    return offsets / (ages.double().sqrt() * statistics["delta_scale"][0])


def common_values(pred, target, mask, raw_close, ps, statistics):
    shifts = validate_pairs(ps)
    if pred.shape != (len(ps), 2, 127, 7) or target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError("Matched paired seven-field queries required")
    if mask.dtype != torch.bool:
        raise ValueError("Boolean target support required")
    price_scale = source.hq.price_scales(statistics, pred.double())
    # One training delta scale for BOTH views and ALL ages; nonprice fields retain
    # the original train coordinate. Never normalize by each evaluation outcome.
    factor = price_scale / statistics["delta_scale"][0]
    p = pred.double().clone()
    y = target.double().clone()
    p[..., 0] = p[..., 0] * factor
    y[..., 0] = raw_targets(raw_close, ps, statistics) * factor
    ii = torch.arange(127, device=ps.device)[None] + shifts[:, None]
    in_range = ii < 127
    index = ii.clamp_max(126)[..., None].expand(-1, -1, 7)
    a, b = p[:, 0], p[:, 1].gather(1, index)
    ya, yb = y[:, 0], y[:, 1].gather(1, index)
    ma, mb = mask[:, 0], mask[:, 1].gather(1, index)
    valid = in_range[..., None] & ma & mb
    if not torch.equal(ma[in_range], mb[in_range]) or not torch.equal(
        ya[..., 1:][in_range], yb[..., 1:][in_range]
    ):
        raise ValueError("Physical common nonprice answers/support differ")
    if not valid[..., 0].any(1).all():
        raise ValueError("Every pair requires common historical prices")

    def centered(v):
        v = v.clone()
        keep = valid[..., 0]
        mean = torch.where(keep, v[..., 0], 0).sum(1) / keep.sum(1)
        v[..., 0] = v[..., 0] - mean[:, None]
        return v

    a, b, ya, yb = map(centered, (a, b, ya, yb))
    if not torch.allclose(ya[..., 0][in_range], yb[..., 0][in_range], atol=1e-10, rtol=1e-10):
        raise ValueError("Raw common physical price shape differs")
    return a, b, ya, yb, valid


def errors(a, b, valid, smooth=True):
    delta = a - b
    point = (
        functional.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
        if smooth
        else delta.square()
    )
    count = valid.sum(1)
    field = torch.where(valid, point, 0).sum(1) / count.clamp_min(1)
    supported = count[:, 2:] > 0
    activity = (field[:, 2:] * supported).sum(1) / supported.sum(1).clamp_min(1)
    changes = delta[:, 1:, 0] - delta[:, :-1, 0]
    change_error = (
        functional.smooth_l1_loss(changes, torch.zeros_like(changes), reduction="none")
        if smooth
        else changes.square()
    )
    keep = valid[:, 1:, 0] & valid[:, :-1, 0]
    change = torch.where(keep, change_error, 0).sum(1) / keep.sum(1).clamp_min(1)
    return 0.4 * field[:, 0] + 0.2 * change + 0.1 * field[:, 1] + 0.3 * activity


def objectives(pred, target, mask, closes, ps, statistics, arm):
    if arm not in ARMS:
        raise ValueError("Declared arm required")
    a, b, ya, yb, valid = common_values(pred, target, mask, closes, ps, statistics)
    q = source.hq.metrics(pred, target, mask, statistics, True)["primary"].mean(1)
    content = (errors(a, ya, valid) + errors(b, yb, valid)) / 2
    alignment = errors(a, b, valid)
    total = q + int(arm != "control") * CONTENT_WEIGHT * content
    total = total + int(arm == "aligned") * ALIGN_WEIGHT * alignment
    return {"query": q, "content": content, "alignment": alignment, "objective": total}


def terms(encoder, query, x, closes, ps, statistics, arm):
    # Raw closes/labels are kept out of source.predict and every encoder/query call.
    _, pred = source.predict(encoder, query, x, ps)
    target, mask = source.targets(x, ps, statistics)
    return objectives(pred, target, mask, closes, ps, statistics, arm)
