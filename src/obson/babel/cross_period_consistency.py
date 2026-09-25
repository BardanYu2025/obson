"""Closed-bar, common-interval price-change supervision; no vector matching."""

import numpy as np
import torch
from torch.nn import functional

from . import overlap_consistency as oc

MIN_LINKS = 8


def strict_map(fine, coarse, fine_period, coarse_period):
    """For every coarse bar, its last child index and strict raw eligibility."""
    if coarse_period <= fine_period or coarse_period % fine_period:
        raise ValueError("Invalid period ratio")
    ft = fine.datetime.to_numpy(dtype="datetime64[ns]")
    ct = coarse.datetime.to_numpy(dtype="datetime64[ns]")
    ratio = coarse_period // fine_period
    step = np.timedelta64(fine_period, "m")
    left = np.searchsorted(ft, ct)
    right = np.searchsorted(ft, ct + np.timedelta64(coarse_period, "m"))
    ix = np.minimum(left[:, None] + np.arange(ratio), len(ft) - 1)
    valid = (right - left == ratio) & (ft[ix] == ct[:, None] + np.arange(ratio) * step).all(1)
    fields = ("open", "high", "low", "close", "volume", "open_oi", "close_oi")
    if any(k not in fine or k not in coarse for k in fields):
        return right - 1, np.zeros(len(ct), bool)
    actual = np.column_stack(
        (
            fine.open.to_numpy()[ix[:, 0]],
            fine.high.to_numpy()[ix].max(1),
            fine.low.to_numpy()[ix].min(1),
            fine.close.to_numpy()[ix[:, -1]],
            fine.volume.to_numpy()[ix].sum(1),
            fine.open_oi.to_numpy()[ix[:, 0]],
            fine.close_oi.to_numpy()[ix[:, -1]],
        )
    )
    valid &= (np.abs(actual - coarse[list(fields)].to_numpy()) <= 1e-6).all(1)
    valid &= (actual[:, 5:] >= 0).all(1)
    return right - 1, valid


def links(fine_end, coarse_end, child_end, valid, coarse_times, coarse_period):
    """Exclude BOTH current bars. Changes use adjacent fully covered coarse bins."""
    result = []
    for j in range(coarse_end - 126, coarse_end):
        if j < 1 or not valid[j] or not valid[j - 1]:
            continue
        if coarse_times[j] - coarse_times[j - 1] != np.timedelta64(coarse_period, "m"):
            continue
        f0, f1 = int(child_end[j - 1] - fine_end + 127), int(child_end[j] - fine_end + 127)
        c0, c1 = j - 1 - coarse_end + 127, j - coarse_end + 127
        if 0 <= f0 < f1 < 127 and 0 <= c0 < c1 < 127:
            result.append([f0, f1, c0, c1])
    return result


def padded_links(rows):
    index = np.zeros((len(rows), 126, 4), np.int64)
    mask = np.zeros((len(rows), 126), bool)
    for i, row in enumerate(rows):
        values = row["links"]
        if not MIN_LINKS <= len(values) <= 126:
            raise ValueError("Insufficient or oversized common history")
        index[i, : len(values)] = values
        mask[i, : len(values)] = True
    return index, mask


def changes(a, b, index, valid, stats):
    if a.shape != b.shape or a.shape[1:] != (128, 7) or index.shape != (*valid.shape, 4):
        raise ValueError("Invalid common-history arrays")
    if index.device != a.device or valid.device != a.device or b.device != a.device:
        raise ValueError("Common-history tensors must share a device")
    if not valid.any(1).all() or ((index < 0) | (index >= 127)).any():
        raise ValueError("Missing support or current/future position in common history")
    if ((index[..., 0] >= index[..., 1]) & valid).any() or (
        (index[..., 2] >= index[..., 3]) & valid
    ).any():
        raise ValueError("Non-increasing common interval")

    def gather(p, i):
        return torch.gather(p[:, :, 0], 1, i)

    factor = stats["y_scale"][0] / stats["delta_scale"][0]
    return (
        (gather(a, index[..., 1]) - gather(a, index[..., 0])) * factor,
        (gather(b, index[..., 3]) - gather(b, index[..., 2])) * factor,
    )


def common_rows(a, b, index, valid, stats, truth_a=None, truth_b=None, smooth=True):
    x, y = changes(a, b, index, valid, stats)

    def reduce(v):
        return (v * valid).sum(1) / valid.sum(1)

    loss = functional.smooth_l1_loss(x, y, reduction="none") if smooth else (x - y).square()
    result = {"gap": reduce(loss)}
    if truth_a is not None:
        tx, ty = changes(truth_a, truth_b, index, valid, stats)
        # No truth correction in prediction: this is an input/target lineage check.
        if not torch.allclose(tx[valid], ty[valid], atol=2e-3, rtol=2e-4):
            raise ValueError("Common target price changes differ after feature encoding")
        result.update(fine_error=reduce((x - tx).square()), coarse_error=reduce((y - ty).square()))
    return result


def run_epoch(
    model,
    builder,
    ids,
    shifts,
    prefixes,
    stats,
    local,
    batch,
    micro,
    device,
    weight,
    encoder=None,
    head=None,
):
    training = encoder is not None
    if training != (head is not None) or not 0 < micro <= batch or batch % micro:
        raise ValueError("Both optimizers and valid gradient accumulation required")
    model.train(training)
    totals = np.zeros(4)
    steps = 0
    with torch.set_grad_enabled(training):
        for left in range(0, len(ids), batch):
            size = min(batch, len(ids) - left)
            if training:
                encoder.zero_grad(set_to_none=True)
                head.zero_grad(set_to_none=True)
            for start in range(left, left + size, micro):
                stop = min(start + micro, left + size)
                a, b, c, ix, mask = builder(ids[start:stop], shifts[start:stop])

                def convert(v):
                    return {k: torch.tensor(x, device=device) for k, x in v.items()}

                a, b, c = map(convert, (a, b, c))
                ix = torch.tensor(ix, device=device)
                mask = torch.tensor(mask, device=device)
                ps = torch.tensor(prefixes[start:stop], device=device)
                ds = torch.tensor(shifts[start:stop], device=device)
                pred, sup = oc.supervised(
                    model,
                    {k: torch.cat((a[k], b[k], c[k])) for k in a},
                    torch.cat((ps, ps, ps)),
                    stats,
                    local,
                )
                n = len(ps)
                pa, pb, pc = pred[:n], pred[n : 2 * n], pred[2 * n :]
                base = sup.reshape(3, n).mean(0)
                within = oc.consistency_rows(pa, pb, a["mask"], b["mask"], ds, stats)
                cross = common_rows(pb, pc, ix, mask, stats, b["y"], c["y"])["gap"]
                objective = base + 0.10 * within + weight * cross
                if not all(torch.isfinite(v).all() for v in (base, within, cross, objective)):
                    raise ValueError("Nonfinite cross-period objective")
                if training:
                    (objective.sum() / size).backward()
                totals += [float(v.detach().sum()) for v in (base, within, cross, objective)]
            if training:
                torch.nn.utils.clip_grad_norm_(
                    model.core.parameters(), 1.0, error_if_nonfinite=True
                )
                torch.nn.utils.clip_grad_norm_(
                    model.local_head.parameters(), 1.0, error_if_nonfinite=True
                )
                encoder.step()
                head.step()
                steps += 1
    return dict(
        zip(
            ("supervised", "within_period", "cross_period", "objective"),
            (totals / len(ids)).tolist(),
            strict=False,
        )
    ), steps
