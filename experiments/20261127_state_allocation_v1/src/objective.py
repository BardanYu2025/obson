"""Per-endpoint importance normalization and one optimizer update for three tasks."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .common import COEFS, QUOTAS
from .views import sample, spans_for_mask

GROUPS = ((0, 1), (2, 3, 4), (5, 6, 7, 8))


def query_weights(s):
    valid = np.isfinite(s.target).any(1) & ~s.current
    age = np.select((s.age <= 16, s.age <= 64), (0, 1), default=2)
    w = np.zeros(len(age))
    mass = np.array([0.4, 0.35, 0.25])
    for p in np.unique(s.period[valid]):
        part = (s.period == p) & valid
        counts = np.bincount(age[part], minlength=3)
        w[part] = mass[age[part]] / counts[age[part]] / mass[counts > 0].sum()
    if w.sum():
        w /= w.sum()
    return w


def field_coeff(z, current=False):
    weights = np.array([0.0, 0.5, 0.5]) if current else np.array([0.6, 0.2, 0.2])
    coeff = np.zeros(9)
    supported = np.array([np.any(z[list(group)] > 0) for group in GROUPS])
    weights *= supported
    if weights.sum():
        weights /= weights.sum()
    for group, weight in zip(GROUPS, weights):
        cols = [i for i in group if z[i] > 0]
        if cols:
            coeff[cols] = weight / len(cols)
    return coeff


def loss_rows(s, rng, task):
    valid = np.isfinite(s.target)
    if task == 1:
        idx = np.flatnonzero(s.hidden)
        z = valid[idx].mean(0)
        coef = np.broadcast_to(
            field_coeff(z) / np.where(z > 0, z, 1) / len(idx), (len(idx), 9)
        ).copy()
        return idx, coef, z
    w = query_weights(s)
    zp = w @ valid
    past = rng.choice(len(w), 64, p=w) if w.any() else np.empty(0, dtype=int)
    curr = np.flatnonzero(s.current & valid.any(1))
    zc = valid[curr].mean(0) if len(curr) else np.zeros(9)
    pc, cc = field_coeff(zp), field_coeff(zc, True)
    fraction = np.array([0.9 * bool(pc.any()), 0.1 * bool(cc.any())])
    fraction /= max(fraction.sum(), 1e-16)
    a = np.broadcast_to(
        fraction[0] * pc / np.where(zp > 0, zp, 1) / max(1, len(past)), (len(past), 9)
    )
    b = np.broadcast_to(
        fraction[1] * cc / np.where(zc > 0, zc, 1) / max(1, len(curr)), (len(curr), 9)
    )
    return np.r_[past, curr], np.vstack((a, b)), zp


def readings(model, z, s, idx):
    # Deduplicate decoder work, then restore duplicate sample multiplicity and paired neighbours.
    prev = s.previous[idx]
    requested = np.r_[idx, prev[prev >= 0]]
    y, y0, expanded = model.read(
        z, s.query, s.period, s.current, s.block, s.slot, requested
    )
    device = z.device
    pp = torch.as_tensor(s.period[idx], device=device)

    def terms(raw):
        pred = raw[: len(idx)]
        changes = torch.zeros_like(pred[:, 0])
        supported = torch.as_tensor(prev >= 0, device=device)
        changes[supported] = pred[supported, 0] - raw[len(idx) :, 0]
        delta = torch.asinh((pred[:, 6].expm1() - pred[:, 5].expm1()) / model.delta[pp])
        return torch.cat(
            (pred[:, :1], changes[:, None], pred[:, 1:], delta[:, None]), 1
        )

    return terms(y), terms(y0), expanded


def example_loss(model, s, task, rng):
    z = model.encode(s)
    if task == 2:
        target = (torch.as_tensor(s.desc, device=z.device) - model.d_mu) / model.d_sd
        return F.smooth_l1_loss(
            model.describe(model.condition(z))[0].double(), target
        ), {}
    idx, coef, support = loss_rows(s, rng, task)
    pred, _, expanded = readings(model, z, s, idx)
    device = z.device
    truth = torch.as_tensor(s.target[idx], device=device)
    ok = torch.isfinite(truth)
    delta = (
        torch.where(ok, pred - torch.nan_to_num(truth), 0.0)
        / model.y_sd[torch.as_tensor(s.period[idx], device=device)]
    )
    elem = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
    loss = (elem * torch.as_tensor(coef, device=device)).sum()
    return loss, {
        "f": s.facts[:, 4].sum(),
        "queries": len(idx),
        "expanded": expanded,
        "hits": np.isfinite(s.target[idx]).sum(0),
        "support": support,
    }


def optimizer(model, lr=0.0002):
    decay, keep = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 else keep).append(p)
    # The 1-D terminal token is a learned embedding, not a bias.
    keep = [p for p in keep if p is not model.encoder.terminal]
    decay.append(model.encoder.terminal)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": keep, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        foreach=False,
    )


def rate(step, total, peak=0.0002):
    warm = max(1, round(0.05 * total))
    if step <= warm:
        return peak * step / warm
    phi = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * phi)) / 2)


def update(
    model, opt, corpus, stats, seed, step, total, quotas=QUOTAS, measure_gradients=False
):
    for group in opt.param_groups:
        group["lr"] = rate(step, total)
    opt.zero_grad(set_to_none=True)
    losses = np.zeros(3)
    hits = np.zeros((2, 9), int)
    miss = np.zeros(9)
    expansion = np.zeros((2, 3))
    exposure = []
    # The diagnostic changes only reduction order and also saves full CPU gradients.
    schedule = [(n, t) for n in range(max(quotas)) for t in range(3) if n < quotas[t]]
    parameters = list(model.parameters())
    task_grads = []
    if measure_gradients:
        schedule.sort(key=lambda item: item[1])
        before = [p.detach().cpu().clone() for p in parameters]
    for n, task in schedule:
        count = quotas[task]
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, task, n]))
        sid, row = corpus.draw(0, task, rng)
        shard = corpus.shard(sid)
        hidden = spans_for_mask(shard, row, rng) if task == 1 else ()
        s = sample(shard, row, stats, hidden)
        loss, info = example_loss(model, s, task, rng)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite task={task}, source={sid}, row={row}")
        (loss * (COEFS[task] / count)).backward()
        losses[task] += loss.detach().item() / count
        exposure.append([task, sid, row])
        if task < 2:
            hits[task] += info["hits"]
            expansion[task] += [info["f"], info["queries"], info["expanded"]]
            if task == 0:
                miss += (1 - info["support"]).clip(0, 1) ** 64 / count
        if measure_gradients and n + 1 == count:
            cumulative = [
                (
                    p.grad.detach().cpu().clone()
                    if p.grad is not None
                    else torch.zeros_like(p, device="cpu")
                )
                for p in parameters
            ]
            for prior in task_grads:
                for part, old in zip(cumulative, prior):
                    part.sub_(old)
            task_grads.append(cumulative)
    norms = [
        torch.nn.utils.clip_grad_norm_(
            m.parameters(), 1.0, error_if_nonfinite=True
        ).item()
        for m in (model.encoder, model.history, model.describe)
    ]
    opt.step()
    diagnostic = {}
    if measure_gradients:

        def dot(lhs, rhs):
            return sum(
                float(torch.dot(a.flatten().double(), b.flatten().double()))
                for a, b in zip(lhs, rhs)
            )

        delta = [p.detach().cpu() - old for p, old in zip(parameters, before)]
        diagnostic = {
            "weighted_gradient_gram": [
                [dot(a, b) for b in task_grads] for a in task_grads
            ],
            "weighted_gradient_dot_actual_delta": [dot(g, delta) for g in task_grads],
        }
    return {
        **diagnostic,
        "loss": losses.tolist(),
        "step": step,
        "lr": rate(step, total),
        "grad_norms_E_D_A": norms,
        "exposure": exposure,
        "field_hits": hits.tolist(),
        "past_field_miss_probability": miss.tolist(),
        "blocks_queries_expanded": expansion.tolist(),
    }
