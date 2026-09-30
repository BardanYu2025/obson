"""Physical macro paths through a frozen, already-trained history reader.

No new learned readout. Gradients pass through the fixed reader into the encoder;
the ordinary trainable history reader retains its original objective.
"""

import torch

from . import shared_history as original

MODES = ("control", "content", "aligned")
TRAIN_BANDS = ((79, 95), (95, 111))
HELD_BANDS = ((75, 91), (91, 107))
CONTENT_WEIGHT = 0.10
ALIGN_WEIGHT = 0.01
FAMILIES, hq, structure, price = original.FAMILIES, original.hq, original.structure, original.price
losses = original.losses


def groups(model, query):
    return [
        list(model.core.encoder.parameters()),
        list(model.local_head.parameters()),
        list(query.parameters()),
    ]


def describe(pred, geometry, statistics):
    """Four ordered shared-node means and three differences, in shared units.

    Each view uses exactly the same physical nodes and weights. The oldest shared
    predicted node is the origin; no true price, current anchor or other state is
    an input to this map. The scalar is training delta scale times sqrt(fine-bar
    duration), identical across A/B/C and independent of outcomes.
    """
    ages, mask, bins, scale = (geometry[k] for k in ("ages", "mask", "bins", "duration"))
    if pred.shape != (3 * mask.shape[0], 127, 7):
        raise ValueError("Three equal batches of original history predictions required")
    if ages.dtype != torch.long or bins.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError("Integer coordinates and boolean validity required")
    if (
        ages.shape != (3, *mask.shape)
        or bins.shape != mask.shape
        or mask.ndim != 3
        or mask.shape[1] != 2
    ):
        raise ValueError("Two ordered macro queries with shared nodes required")
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Positive common physical duration required")
    if ((ages[:, mask] < 1) | (ages[:, mask] > 127)).any() or not mask[..., 0].all():
        raise ValueError("Only strictly historical nodes and an available origin allowed")
    values = hq.physical(pred, statistics)[..., 0].reshape(3, len(mask), 127)
    index = ages.clamp(1, 127) - 1
    gathered = values.gather(2, index.flatten(2)).reshape_as(ages)
    centered = gathered - gathered[..., :1]
    means = []
    for i in range(4):
        take = mask & (bins == i)
        if (take.sum(-1) < 2).any():
            raise ValueError("Every temporal group needs at least two verified physical nodes")
        means.append(torch.where(take[None], centered, 0).sum(-1) / take.sum(-1)[None])
    path = torch.stack(means, -1) / (
        float(statistics["delta_scale"][0]) * scale.sqrt()[None, ..., None]
    )
    changes = path[..., 1:] - path[..., :-1]
    return torch.cat([path, changes], -1)


def targets(batch, geometry, statistics):
    ps = torch.full((len(batch["x"]), 1), 128, dtype=torch.long, device=batch["x"].device)
    y, valid = hq.targets(batch, ps, statistics)
    ages, mask = geometry["ages"], geometry["mask"]
    available = valid[:, 0, :, 0].reshape(3, len(mask), 127)
    selected = available.gather(2, (ages.clamp(1, 127) - 1).flatten(2)).reshape_as(ages)
    if not (selected | ~mask[None]).all():
        raise ValueError("Missing close target at a required physical node")
    result = describe(y[:, 0], geometry, statistics)
    physical = result * (
        float(statistics["delta_scale"][0]) * geometry["duration"].sqrt()[None, ..., None]
    )
    if not torch.allclose(physical, physical[1:2].expand_as(physical), atol=2e-5, rtol=2e-5):
        raise ValueError("Common macro targets differ between physical views")
    return result


def per_query(error):
    # Equal weight for the path and its temporal changes, not seven anonymous slots.
    return 0.5 * (error[..., :4].square().mean(-1) + error[..., 4:].square().mean(-1))


def auxiliary(reader, endpoints, geometry, target, active, mode, statistics):
    if mode not in MODES or active.shape != (len(geometry["mask"]),) or active.dtype != torch.bool:
        raise ValueError("Declared mode and per-row cross eligibility required")
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("Auxiliary reader must be frozen")
    # All arms compute the same diagnostics. Only the declared encoder gradients differ.
    z = endpoints if mode != "control" else endpoints.detach()
    predicted = describe(reader(z), geometry, statistics)
    content = per_query(predicted - target).mean((0, 2))
    within = per_query(predicted[0] - predicted[1]).mean(-1)
    cross = per_query(predicted[1] - predicted[2]).mean(-1)
    alignment = (within + torch.where(active, cross, 0)) / (1 + active.to(within.dtype))
    zero = endpoints.sum(-1).reshape(3, -1).mean(0) * 0
    total = zero
    if mode != "control":
        total = total + CONTENT_WEIGHT * content
    if mode == "aligned":
        total = total + ALIGN_WEIGHT * alignment
    return total, {"content": content, "alignment": alignment}, predicted
