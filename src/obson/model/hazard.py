"""Discrete competing-risk hazard utilities for teacher-level experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hazard_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    event_weight: float = 1.0,
) -> torch.Tensor:
    """NLL for conditional hazards with classes [survive, up, down].

    ``event_weight`` raises the contribution of first-touch bins. This is
    needed because most samples are censored and otherwise an all-survival
    predictor can obtain a deceptively low loss.
    """
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("hazard logits must have shape [batch, bins, 3]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("hazard targets must have shape [batch, bins]")
    logp = F.log_softmax(logits, dim=-1)
    safe = targets.clamp(min=0)
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    event = (targets == 1) | (targets == 2)
    before_event = torch.cumsum(event.to(torch.int64), dim=1) - event.to(torch.int64)
    active = (targets >= 0) & (before_event == 0)
    weights = torch.where(event, torch.as_tensor(event_weight, device=logits.device), torch.ones_like(nll))
    return (nll * active * weights).sum() / (active * weights).sum().clamp(min=1e-8)


def hazard_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Aggregate hazards to the public class order [down, none, up]."""
    p = logits.softmax(dim=-1)
    survival = torch.ones_like(p[:, 0, 0])
    up = torch.zeros_like(survival)
    down = torch.zeros_like(survival)
    for k in range(p.shape[1]):
        up = up + survival * p[:, k, 1]
        down = down + survival * p[:, k, 2]
        survival = survival * p[:, k, 0]
    return torch.stack([down, survival, up], dim=-1)


def hazard_to_class_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return hazard_to_probs(logits).clamp_min(eps).log()
