"""Shared decoded history: retain remote price levels and multi-bar changes."""

import torch
from torch.nn import functional

from . import history_query as hq

MODES = ("control", "structure", "consistent")
WIDTHS = (4, 16)
STRUCTURE_WEIGHT = 1.0
CONSISTENCY_WEIGHT = 0.1


def pools(values, mask, width, lo=17):
    """Nonoverlapping age bins, including the final short bin; no padding labels."""
    if values.shape != mask.shape or values.shape[-1] != 127 or width not in WIDTHS:
        raise ValueError("Matched127 historical prices and declared bin width required")
    if lo not in (17, 65):
        raise ValueError("Only remote17+ or oldest65+ bands are declared")
    starts = torch.arange(lo - 1, 127, width, device=values.device)
    ends = (starts + width).clamp_max(127)
    indices = starts[:, None] + torch.arange(width, device=values.device)[None]
    inside = indices < 127
    indices = indices.clamp_max(126)
    available = mask[..., indices] & inside
    lengths = ends - starts
    means = torch.where(available, values[..., indices], 0).sum(-1) / lengths
    valid = available.sum(-1) == lengths
    centers = (starts + ends + 1).to(values.dtype) / 2
    return means, valid, centers


def masked_mean(values, valid):
    return torch.where(valid, values, 0).sum(-1) / valid.sum(-1).clamp_min(1)


def square_or_huber(value, smooth):
    return (
        functional.smooth_l1_loss(value, torch.zeros_like(value), reduction="none")
        if smooth
        else value.square()
    )


def metrics(pred, target, mask, stats, smooth=False, lo=17):
    """Physical close means preserve absolute relative levels and coarse changes."""
    if pred.shape != target.shape or pred.shape != mask.shape or pred.shape[-2:] != (127, 7):
        raise ValueError("Matched query tensors required")
    p, y = hq.physical(pred, stats)[..., 0], hq.physical(target, stats)[..., 0]
    values = {"level": [], "trend": []}
    supports = {"level": [], "trend": []}
    for width in WIDTHS:
        a, valid, centers = pools(p, mask[..., 0], width, lo)
        b, _, _ = pools(y, mask[..., 0], width, lo)
        scale = stats["delta_scale"][0]
        error = (a - b) / (scale * centers.sqrt())
        dm = valid[..., 1:] & valid[..., :-1]
        delta = ((a[..., 1:] - a[..., :-1]) - (b[..., 1:] - b[..., :-1])) / (
            scale * (centers[1:] - centers[:-1]).sqrt()
        )
        for name, err, ok in (("level", error, valid), ("trend", delta, dm)):
            values[name].append(masked_mean(square_or_huber(err, smooth), ok))
            supports[name].append(ok.any(-1))
    result = {}
    for k in values:
        ok = torch.stack(supports[k], -1)
        result[k] = masked_mean(torch.stack(values[k], -1), ok)
    result["primary"] = 0.5 * (result["level"] + result["trend"])
    return result


def overlap(pred_a, pred_b, mask_a, mask_b, shifts, stats, smooth=False):
    """Align the same physical past; remove each own predicted shared level only."""
    if pred_a.shape != pred_b.shape or pred_a.shape != mask_a.shape or pred_b.shape != mask_b.shape:
        raise ValueError("Matched historical tensors required")
    if (
        shifts.shape != (len(pred_a),)
        or shifts.dtype != torch.long
        or not ((shifts >= 1) & (shifts <= 64)).all()
    ):
        raise ValueError("Per-row shifts1..64 required")
    a = hq.physical(pred_a, stats)[..., 0]
    b = hq.physical(pred_b, stats)[..., 0]
    index = torch.arange(127, device=a.device)[None] + shifts[:, None]
    valid = (index < 127) & mask_a[..., 0] & mask_b[..., 0].gather(1, index.clamp_max(126))
    b = b.gather(1, index.clamp_max(126))
    result = []
    for width in WIDTHS:
        pa, ok, centers = pools(a, valid, width)
        pb, _, _ = pools(b, valid, width)
        # A pair's shared most recent complete block supplies its own origin.
        origin = ok.long().argmax(-1)[:, None]
        shape = (pa - pa.gather(1, origin)) - (pb - pb.gather(1, origin))
        distance = (centers[None] - centers[origin]).abs().clamp_min(1)
        shape_ok = ok & ok.gather(1, origin)
        shape_ok = shape_ok & (torch.arange(len(centers), device=a.device)[None] != origin)
        scale = stats["delta_scale"][0]
        shape = shape / (scale * distance.sqrt())
        dm = ok[:, 1:] & ok[:, :-1]
        delta = ((pa[:, 1:] - pa[:, :-1]) - (pb[:, 1:] - pb[:, :-1])) / (
            scale * (centers[1:] - centers[:-1]).sqrt()
        )
        result.append(
            0.5
            * (
                masked_mean(square_or_huber(shape, smooth), shape_ok)
                + masked_mean(square_or_huber(delta, smooth), dm)
            )
        )
    return torch.stack(result, -1).mean(-1)


def losses(model, query, batch, prefixes, query_prefixes, shifts, statistics, local, mode):
    if (
        mode not in MODES
        or len(batch["x"]) != 3 * len(shifts)
        or not (query_prefixes[:, -1] == 128).all()
    ):
        raise ValueError("Declared mode, three equal views and final endpoint128 required")
    z = model.core.encoder(batch["x"])
    rows = torch.arange(len(z), device=z.device)[:, None]
    n = len(shifts)
    detail = model.local_head(z.detach()[rows, prefixes - 1]).reshape(
        len(z), prefixes.shape[1], 16, 7
    )
    ly, lm = hq.ba.local_targets(batch["y"], batch["mask"], prefixes, statistics, local)
    local_loss = hq.ba.local_rows(detail, ly, lm, True)["primary"].reshape(3, n).mean(0)
    picked = z[rows, query_prefixes - 1]
    pred = query(picked.flatten(0, 1)).reshape(len(z), query_prefixes.shape[1], 127, 7)
    target, valid = hq.targets(batch, query_prefixes, statistics)
    qloss = (
        hq.metrics(pred, target, valid, statistics, True)["primary"].mean(1).reshape(3, n).mean(0)
    )
    coarse = metrics(pred, target, valid, statistics, True)["primary"].mean(1).reshape(3, n).mean(0)
    shared = overlap(
        pred[:n, -1],
        pred[n : 2 * n, -1],
        valid[:n, -1],
        valid[n : 2 * n, -1],
        shifts,
        statistics,
        True,
    )
    total = qloss + 0.25 * local_loss
    if mode in ("structure", "consistent"):
        total = total + STRUCTURE_WEIGHT * coarse
    if mode == "consistent":
        total = total + CONSISTENCY_WEIGHT * shared
    return {"structure": coarse, "consistency": shared, "query": qloss, "optimization_value": total}
