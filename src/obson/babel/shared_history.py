"""Content-dependent historical interval reads and nuisance-centered VICReg.

The persistent representation remains one768-vector. Only validated common
close-to-close intervals enter this training-only64-dimensional reader.
"""

import torch
from torch import nn
from torch.nn import functional as functional

from . import activity_weight as reconstruction

MODES = {"control": (0.0, 0.0), "within": (1.0, 0.0), "cross": (1.0, 1.0)}
TRAIN_BAND = (103, 115)
HELD_BAND = (99, 111)
MIN_INTERVALS = 4
LATENT = 64
AUX_WEIGHT = 0.10


class IntervalReader(nn.Module):
    def __init__(self, width, seed, latent=LATENT):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 20260930)
            self.content = nn.Linear(width, 192)
            self.gate = nn.Sequential(
                nn.Linear(2, 192), nn.GELU(), nn.Linear(192, 192), nn.Sigmoid()
            )
            self.output = nn.Linear(192, latent, bias=False)

    def forward(self, state, ages, mask):
        if state.ndim != 2 or ages.shape != (*mask.shape, 2) or len(state) != len(ages):
            raise ValueError("One state and matched historical interval geometry per row required")
        if ((ages[mask] < 1) | (ages[mask] > 127)).any() or not (
            ages[mask][:, 0] > ages[mask][:, 1]
        ).all():
            raise ValueError("Strictly past, forward-in-time interval endpoints required")
        safe = torch.where(mask[..., None], ages, 1).to(state.dtype) / 127
        # Geometry can only gate state content; raw bars, targets and other states never enter.
        values = self.output(functional.gelu(self.content(state))[:, None] * self.gate(safe))
        return (values * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]


def install(query, seed, latent=LATENT):
    if hasattr(query, "shared"):
        raise ValueError("Interval reader already installed")
    query.shared = IntervalReader(len(query.state_mean), seed, latent).to(query.state_mean.device)
    return query


def routed(state, gain):
    return state.detach() + gain * (state - state.detach())


def reads(query, endpoints, ages, mask, mode):
    if mode not in MODES or endpoints.shape[0] != 3 * len(mask):
        raise ValueError("Three equal A/B/C endpoint batches required")
    n = len(mask)
    z = (endpoints - query.state_mean) / query.state_scale
    a, b, c = z[:n], z[n : 2 * n], z[2 * n :]
    within, cross = MODES[mode]
    return torch.stack(
        [
            query.shared(routed(a, within), ages[0], mask),
            query.shared(routed(b, within), ages[1], mask),
            query.shared(routed(b, cross), ages[1], mask),
            query.shared(routed(c, cross), ages[2], mask),
        ]
    )


def centered_statistics(z):
    """Every adjacent pair has identical contract/query geometry but distinct time.

    Centering within these pairs removes contract/geometry-only shortcuts. Use
    ALL effective-batch pairs, never independently regularize each microbatch.
    """
    if len(z) < 4 or len(z) % 2:
        raise ValueError("At least two complete nuisance-matched temporal pairs required")
    residual = z.reshape(-1, 2, z.shape[-1])
    residual = (residual - residual.mean(1, keepdim=True)).reshape_as(z)
    covariance = residual.T @ residual / (len(z) // 2)
    variance = covariance.diagonal()
    std = (variance + 1e-4).sqrt()
    offdiag = covariance - torch.diag(variance)
    return functional.relu(1 - std).mean(), offdiag.square().sum() / z.shape[1]


def pair_loss(a, b, active):
    if len(a) % 2 or not torch.equal(active[::2], active[1::2]):
        raise ValueError("Pair eligibility must be identical at both temporal endpoints")
    a, b = a[active], b[active]
    if len(a) < 4:
        zero = (a.sum() + b.sum()) * 0
        return {
            "invariance": zero,
            "variance": zero,
            "covariance": zero,
            "total": zero,
            "rows": len(a),
        }
    inv = functional.mse_loss(a, b)
    va, ca = centered_statistics(a)
    vb, cb = centered_statistics(b)
    var, cov = (va + vb) / 2, (ca + cb) / 2
    return {
        "invariance": inv,
        "variance": var,
        "covariance": cov,
        "total": inv + var + 0.04 * cov,
        "rows": len(a),
    }


def auxiliary(features, cross_active):
    within = pair_loss(
        features[0],
        features[1],
        torch.ones(len(cross_active), dtype=torch.bool, device=features.device),
    )
    cross = pair_loss(features[2], features[3], cross_active)
    return AUX_WEIGHT * (within["total"] + cross["total"]), {"within": within, "cross": cross}


def groups(model, query):
    return [
        list(model.core.encoder.parameters()),
        list(model.local_head.parameters()),
        [p for n, p in query.named_parameters() if not n.startswith("shared.")],
        list(query.shared.parameters()),
    ]


def losses(
    model, query, batch, prefixes, query_prefixes, shifts, statistics, local, endpoints=None
):
    # This copy uses the same fixed0.3 arithmetic; expose endpoints from the ONE encoder pass.
    hq = reconstruction.hq
    z = model.core.encoder(batch["x"]) if endpoints is None else endpoints
    n = len(shifts)
    rows = torch.arange(len(z), device=z.device)[:, None]
    detail = model.local_head(z.detach()[rows, prefixes - 1]).reshape(len(z), -1, 16, 7)
    ly, lm = hq.ba.local_targets(batch["y"], batch["mask"], prefixes, statistics, local)
    local_loss = hq.ba.local_rows(detail, ly, lm, True)["primary"].reshape(3, n).mean(0)
    pred = query(z[rows, query_prefixes - 1].flatten(0, 1)).reshape(len(z), -1, 127, 7)
    target, valid = hq.targets(batch, query_prefixes, statistics)
    raw = hq.metrics(pred, target, valid, statistics, True)
    result = {
        k: raw[k].mean(1).reshape(3, n).mean(0) for k in (*reconstruction.FAMILIES, "primary")
    }
    coarse = (
        reconstruction.structure.metrics(pred, target, valid, statistics, True)["primary"]
        .mean(1)
        .reshape(3, n)
        .mean(0)
    )
    result.update(query_fixed=result.pop("primary"), structure=coarse, local=local_loss)
    result["optimization_value"] = result["query_fixed"] + coarse + 0.25 * local_loss
    return result, z[:, -1]


FAMILIES = reconstruction.FAMILIES
hq = reconstruction.hq
structure = reconstruction.structure
price = reconstruction.price
