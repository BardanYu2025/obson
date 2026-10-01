"""Position-density controls with exact chunked gradients and unchanged history heads."""

import numpy as np
import torch

from . import history_query as hq
from . import history_structure as structure
from . import linear_history as original

MODES = ("sampled", "dense_matched", "dense_all")
FAMILIES, price, configure = original.FAMILIES, original.price, original.configure
INTERIOR = tuple(hq.QUERY_PREFIXES)
ALL_PREFIXES = tuple(range(2, 129))


def position_weights(sampled, mode):
    """dense_matched integrates exactly the original random-prefix estimator."""
    if mode not in MODES or sampled.ndim != 2 or sampled.shape[1] != 5:
        raise ValueError("Declared mode and original four-plus-endpoint schedule required")
    if sampled.dtype != torch.long or not (sampled[:, -1] == 128).all():
        raise ValueError("Integer schedule ending at128 required")
    if (
        not torch.isin(sampled[:, :4], sampled.new_tensor(INTERIOR)).all()
        or (sampled[:, :4].sort(1).values.diff(dim=1) == 0).any()
    ):
        raise ValueError("Four distinct original interior prefixes required")
    if mode == "sampled":
        return sampled, torch.full(sampled.shape, 0.2, device=sampled.device)
    if mode == "dense_matched":
        positions = (*INTERIOR, 128)
        weights = [0.8 / len(INTERIOR)] * len(INTERIOR) + [0.2]
    else:
        positions = ALL_PREFIXES
        weights = [1 / len(positions)] * len(positions)
    return (
        sampled.new_tensor(positions)[None].expand(len(sampled), -1),
        torch.tensor(weights, device=sampled.device)[None].expand(len(sampled), -1),
    )


def targets(batch, prefixes, stats):
    """Original coordinates, extended to prefixes2..31; no current/future target."""
    if (
        prefixes.ndim != 2
        or len(prefixes) != len(batch["x"])
        or prefixes.dtype != torch.long
        or not ((prefixes >= 2) & (prefixes <= 128)).all()
    ):
        raise ValueError("Historical prefixes2..128 required; prefix1 has no past")
    ages = torch.arange(1, 128, device=prefixes.device)
    index = prefixes[..., None] - 1 - ages
    observed = index >= 0
    index = index.clamp_min(0)
    rows = torch.arange(len(prefixes), device=prefixes.device)[:, None, None]
    physical = batch["y"].double() * batch["y"].new_tensor(
        stats["y_scale"], dtype=torch.float64
    ) + batch["y"].new_tensor(stats["y_mean"], dtype=torch.float64)
    values = physical[rows, index].clone()
    values[..., 0] -= hq.anchors(batch, prefixes, stats)[..., None]
    mask = batch["mask"][rows, index] & observed[..., None]
    values[..., 0] /= hq.price_scales(stats, values)
    values[..., 1:] = (
        values[..., 1:] - values.new_tensor(stats["y_mean"][1:])
    ) / values.new_tensor(stats["y_scale"][1:])
    return torch.where(mask, values, 0).to(batch["y"].dtype), mask


def query_terms(query, states, batch, prefixes, statistics, smooth=True):
    rows = torch.arange(len(states), device=states.device)[:, None]
    # Queries are independent: skip ages unavailable to every context in this chunk.
    count = int(prefixes.max()) - 1
    ages = torch.arange(1, count + 1, device=states.device)
    decoded = query(states[rows, prefixes - 1].flatten(0, 1), ages).reshape(
        len(states), prefixes.shape[1], count, 7
    )
    pred = torch.nn.functional.pad(decoded, (0, 0, 0, 127 - count))
    y, valid = targets(batch, prefixes, statistics)
    terms = hq.metrics(pred, y, valid, statistics, smooth)
    terms["query_fixed"] = terms.pop("primary")
    terms["structure"] = structure.metrics(pred, y, valid, statistics, smooth)["primary"]
    return terms


def backward_batch(
    model,
    query,
    batch,
    local_prefixes,
    sampled,
    shifts,
    stats,
    local,
    mode,
    effective_triplets,
    chunk,
):
    """Accumulate reader/latent gradients per chunk; backpropagate encoder once.

    This is algebraically the dense objective (no truncation or optimizer step
    between chunks), but bounds decoder activation memory independently of count.
    """
    if chunk < 1 or effective_triplets < len(shifts) or len(batch["x"]) != 3 * len(shifts):
        raise ValueError("Positive chunk, full effective batch and three views required")
    states = model.core.encoder(batch["x"])
    proxy = states.detach().requires_grad_(True)
    ps, weights = position_weights(sampled, mode)
    n = len(shifts)
    keys = (*FAMILIES, "query_fixed", "structure", "local", "optimization_value")
    totals = {k: states.new_zeros(n) for k in keys}
    for left in range(0, ps.shape[1], chunk):
        terms = query_terms(query, proxy, batch, ps[:, left : left + chunk], stats)
        reduced = {
            k: (v * weights[:, left : left + chunk]).sum(1).reshape(3, n).mean(0)
            for k, v in terms.items()
            if k in totals
        }
        loss = reduced["query_fixed"] + reduced["structure"]
        if not torch.isfinite(loss).all():
            raise ValueError("Nonfinite historical objective")
        (loss.sum() / effective_triplets).backward()
        for k, v in reduced.items():
            totals[k] += v.detach()
    rows = torch.arange(len(states), device=states.device)[:, None]
    detail = model.local_head(states.detach()[rows, local_prefixes - 1]).reshape(
        len(states), -1, 16, 7
    )
    y, mask = hq.ba.local_targets(batch["y"], batch["mask"], local_prefixes, stats, local)
    local_loss = hq.ba.local_rows(detail, y, mask, True)["primary"].reshape(3, n).mean(0)
    if not torch.isfinite(local_loss).all():
        raise ValueError("Nonfinite diagnostic local loss")
    (0.25 * local_loss.sum() / effective_triplets).backward()
    if proxy.grad is None or not torch.isfinite(proxy.grad).all():
        raise ValueError("Missing/nonfinite accumulated state gradient")
    states.backward(proxy.grad)
    totals["local"] = local_loss.detach()
    totals["optimization_value"] = (
        totals["query_fixed"] + totals["structure"] + 0.25 * totals["local"]
    )
    return totals


def exposure(mode, views):
    counts = {"sampled": 5, "dense_matched": len(INTERIOR) + 1, "dense_all": 127}
    if mode not in counts or views < 1:
        raise ValueError("Declared positive exposure required")
    return {
        "query_contexts": int(views * counts[mode]),
        "positions_per_view": counts[mode],
        "supervised_position1": False,
        "legacy_held_trained": mode == "dense_all",
    }


def profile_summary(values):
    """Per-context arrays already contain masked per-age-band scores."""
    return {k: np.asarray(v, dtype=np.float64).mean(0).tolist() for k, v in values.items()}
