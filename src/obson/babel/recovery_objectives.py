"""Isolated native-prefix interventions; not a GPU training entry point.

Keep the historical modules immutable. The baseline is the Context Transfer
query + remote-structure objective. A frozen, source-trained local reader is
an optional extra gradient route, not evidence that its representation improves.
"""

from dataclasses import dataclass

import torch

from . import bar_alignment as ba
from . import history_query as hq
from . import history_structure as hs


@dataclass(frozen=True)
class Intervention:
    remote_structure: bool
    local_to_encoder: bool


ARMS = {
    "baseline": Intervention(True, False),
    "without_remote": Intervention(False, False),
    "with_local": Intervention(True, True),
    "without_remote_with_local": Intervention(False, True),
}


def prefix_loss_rows(model, query, batch, prefixes, statistics, local, arm):
    """Return losses per window, with identical targets/readers in all four arms.

    Callers must freeze and qualify the existing local head before using this
    function. No fitting, reader replacement, optimizer or sampling is hidden
    here. Local supervision covers price AND activity with the old 0.25 weight.
    Keep its diagnostic value separate from the actual optimization value.
    """
    if arm not in ARMS:
        raise ValueError("Unknown structural intervention")
    if (
        prefixes.dtype != torch.long
        or prefixes.ndim != 2
        or len(prefixes) != len(batch["x"])
        or prefixes.numel() == 0
        or batch["x"].shape[1] != 128
        or not ((prefixes >= 32) & (prefixes <= 128)).all()
    ):
        raise ValueError("Native 128-bar inputs and matched integer prefixes32..128 required")
    if any(module.training for module in model.local_head.modules()) or any(
        p.requires_grad for p in model.local_head.parameters()
    ):
        raise ValueError("Source local reader must be frozen and in evaluation mode")

    mode = ARMS[arm]
    states, prediction = hq.predict_states(model, query, batch["x"], prefixes)
    target, valid = hq.targets(batch, prefixes, statistics)
    query_rows = hq.metrics(prediction, target, valid, statistics, True)["primary"].mean(1)
    structure_rows = hs.metrics(prediction, target, valid, statistics, True)["primary"].mean(1)

    rows = torch.arange(len(states), device=states.device)[:, None]
    picked = states[rows, prefixes - 1]
    local_prediction = model.local_head(picked if mode.local_to_encoder else picked.detach())
    local_prediction = local_prediction.reshape(len(states), prefixes.shape[1], ba.HISTORY, 7)
    local_target, local_valid = ba.local_targets(
        batch["y"], batch["mask"], prefixes, statistics, local
    )
    local_rows = ba.local_rows(local_prediction, local_target, local_valid, True)["primary"]

    total = query_rows
    if mode.remote_structure:
        total = total + hs.STRUCTURE_WEIGHT * structure_rows
    if mode.local_to_encoder:
        total = total + ba.LOCAL_WEIGHT * local_rows
    return {
        "query": query_rows,
        "remote_structure": structure_rows,
        "local_diagnostic": local_rows,
        "optimization_value": total,
    }
