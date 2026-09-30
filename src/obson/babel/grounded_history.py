"""Query-grounded, ordered price content; the decoder receives only one latent."""

import torch
from torch import nn

from . import shared_history as original

SLOTS = 2
LATENT = 64
MODES = ("control", "content", "aligned")
CONTENT_WEIGHT = 0.1
ALIGN_WEIGHT = 0.01
FAMILIES, hq, structure, price = original.FAMILIES, original.hq, original.structure, original.price
losses, groups = original.losses, original.groups


class GroundedReader(nn.Module):
    def __init__(self, width, seed, latent=LATENT):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 20261001)
            self.state = nn.Linear(width, 192)
            self.geometry = nn.Sequential(nn.Linear(SLOTS * 2, 64), nn.GELU())
            self.combine = nn.Sequential(nn.Linear(256, 192), nn.GELU(), nn.Linear(192, latent))
            self.norm = nn.LayerNorm(latent, elementwise_affine=False)
            self.decoder = nn.Sequential(nn.Linear(latent, 128), nn.GELU(), nn.Linear(128, SLOTS))

    def forward(self, state, ages, ablation="none"):
        if ages.shape != (len(state), SLOTS, 2):
            raise ValueError("Two ordered historical intervals required")
        if ((ages < 1) | (ages > 127)).any() or not (ages[..., 0] > ages[..., 1]).all():
            raise ValueError("Only strictly past forward intervals are allowed")
        if ablation not in ("none", "state_only", "query_only"):
            raise ValueError("Unknown input ablation")
        s = self.state(torch.zeros_like(state) if ablation == "query_only" else state)
        q = ages.to(state.dtype).flatten(1) / 127
        if ablation == "state_only":
            q = torch.zeros_like(q)
        r = self.norm(self.combine(torch.cat([s, self.geometry(q)], -1)))
        return r, self.decoder(r)


def install(query, seed, latent=LATENT):
    # The predecessor's free reader is deliberately replaced, after strict source load.
    query.shared = GroundedReader(len(query.state_mean), seed, latent).to(query.state_mean.device)


def queries(ages, mask):
    """Two disjoint, ordered two-interval requests from the same endpoint state."""
    result = []
    for i in range(len(mask)):
        ids = mask[i].nonzero().flatten()
        if len(ids) < 2 * SLOTS:
            raise ValueError("At least four verified intervals required")
        result.append(torch.stack([ages[:, i, ids[:SLOTS]], ages[:, i, ids[-SLOTS:]]], 1))
    return torch.stack(result, 1)  # view, row, query, slot, endpoint


def targets(batch, coordinates, statistics):
    n = coordinates.shape[1]
    y = batch["y"].reshape(3, n, 128, -1)[..., 0]
    m = batch["mask"].reshape(3, n, 128, -1)[..., 0]
    ix = 127 - coordinates
    v = torch.arange(3, device=y.device)[:, None, None, None, None]
    r = torch.arange(n, device=y.device)[None, :, None, None, None]
    if not m[v, r, ix].all():
        raise ValueError("Missing common price targets")
    endpoints = y[v, r, ix]
    raw = (endpoints[..., 1] - endpoints[..., 0]) * float(statistics["y_scale"][0])
    if not torch.allclose(raw, raw[1:2].expand_as(raw), atol=2e-5, rtol=2e-5):
        raise ValueError("Common physical price targets disagree")
    return raw


def read_views(query, z, coordinates, gain=1):
    n = coordinates.shape[1]
    z = original.routed((z - query.state_mean) / query.state_scale, gain)
    z = z.reshape(3, n, -1)[:, :, None].expand(-1, -1, 2, -1)
    r, pred = query.shared(z.reshape(-1, z.shape[-1]), coordinates.reshape(-1, SLOTS, 2))
    return r.reshape(3, n, 2, -1), pred.reshape(3, n, 2, SLOTS)


def auxiliary(query, z, coordinates, target, active, mode):
    if mode not in MODES:
        raise ValueError("Invalid gradient route")
    _, pred = read_views(query, z, coordinates, mode != "control")
    content = (pred - target).square().mean(dim=(0, 2, 3))
    # Same head objective in all arms; only the declared encoder routes change.
    r, _ = read_views(query, z, coordinates, mode == "aligned")
    within = (r[0] - r[1]).square().mean((1, 2))
    cross = (r[1] - r[2]).square().mean((1, 2)) * active
    align = (within + cross) / 2
    return CONTENT_WEIGHT * content + ALIGN_WEIGHT * align, {"content": content, "alignment": align}
