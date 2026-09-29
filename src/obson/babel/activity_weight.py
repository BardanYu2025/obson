"""Fixed-input continuation: vary only historical activity supervision."""

import torch

from . import history_query as hq
from . import history_structure as structure

WEIGHTS = {"a000": 0.0, "a010": 0.1, "a030": 0.3}
FAMILIES = ("path", "change1", "body", "activity")


def price(parts):
    """Common fixed score, independent of the training activity coefficient."""
    return (0.4 * parts["path"] + 0.2 * parts["change1"] + 0.1 * parts["body"]) / 0.7


def losses(model, query, batch, prefixes, query_prefixes, shifts, statistics, local, weight):
    if weight not in WEIGHTS.values() or len(batch["x"]) != 3 * len(shifts):
        raise ValueError("Declared activity coefficient and three matched views required")
    if not (query_prefixes[:, -1] == 128).all():
        raise ValueError("Every query view must include its endpoint")
    z = model.core.encoder(batch["x"])
    rows = torch.arange(len(z), device=z.device)[:, None]
    n = len(shifts)
    detail = model.local_head(z.detach()[rows, prefixes - 1]).reshape(len(z), -1, 16, 7)
    ly, lm = hq.ba.local_targets(batch["y"], batch["mask"], prefixes, statistics, local)
    local_loss = hq.ba.local_rows(detail, ly, lm, True)["primary"].reshape(3, n).mean(0)
    picked = z[rows, query_prefixes - 1]
    pred = query(picked.flatten(0, 1)).reshape(len(z), -1, 127, 7)
    target, valid = hq.targets(batch, query_prefixes, statistics)
    raw = hq.metrics(pred, target, valid, statistics, True)
    result = {k: raw[k].mean(1).reshape(3, n).mean(0) for k in (*FAMILIES, "primary")}
    coarse = structure.metrics(pred, target, valid, statistics, True)["primary"]
    coarse = coarse.mean(1).reshape(3, n).mean(0)
    # Preserve the old control's arithmetic exactly; never renormalize other coefficients.
    weighted = (
        raw["primary"]
        if weight == 0.3
        else (
            0.4 * raw["path"] + 0.2 * raw["change1"] + 0.1 * raw["body"] + weight * raw["activity"]
        )
    )
    result.update(
        query_fixed=result.pop("primary"),
        query_weighted=weighted.mean(1).reshape(3, n).mean(0),
        structure=coarse,
        local=local_loss,
    )
    result["optimization_value"] = result["query_weighted"] + coarse + 0.25 * local_loss
    return result
