"""Isolate legacy fixed-slot supervision from a uniform bar-relative objective."""

import torch

from . import history_query as hq
from . import history_query_run as parent

MODES = ("control", "additive", "uniform")


def losses(model, query, batch, prefixes, query_prefixes, shifts, statistics, local, mode):
    if mode not in MODES or len(batch["x"]) != 3 * len(shifts):
        raise ValueError("A declared mode and three equal view groups are required")
    z = model.core.encoder(batch["x"])
    # Uniform encoders receive no gradients from either legacy decoder or its
    # consistency loss. The old local head still adapts on detached states so its
    # diagnostic readout has the same optimization budget in all three arms.
    old_z = z.detach() if mode == "uniform" else z
    g = model.core.decoder(old_z[:, -1])
    rows = torch.arange(len(z), device=z.device)[:, None]
    detail = model.local_head(old_z[rows, prefixes - 1]).reshape(len(z), prefixes.shape[1], 16, 7)
    y, mask = hq.ba.local_targets(batch["y"], batch["mask"], prefixes, statistics, local)
    n = len(shifts)
    global_loss = (
        hq.ba.ar.error_rows(g, batch["y"], batch["mask"], statistics, True)["primary"]
        .reshape(3, n)
        .mean(0)
    )
    local_loss = hq.ba.local_rows(detail, y, mask, True)["primary"].reshape(3, n).mean(0)
    within = parent.xr.xp.oc.consistency_rows(
        g[:n], g[n : 2 * n], batch["mask"][:n], batch["mask"][n : 2 * n], shifts, statistics
    )
    picked = z[rows, query_prefixes - 1]
    if mode == "control":
        picked = picked.detach()
    pred = query(picked.flatten(0, 1)).reshape(len(z), query_prefixes.shape[1], 127, 7)
    target, valid = hq.targets(batch, query_prefixes, statistics)
    query_loss = (
        hq.metrics(pred, target, valid, statistics, True)["primary"].mean(1).reshape(3, n).mean(0)
    )
    old = global_loss + 0.25 * local_loss
    total = query_loss + 0.25 * local_loss
    if mode != "uniform":
        total = total + global_loss + 0.1 * within
    return {
        "original": old,
        "consistency": within,
        "query": query_loss,
        "optimization_value": total,
    }
