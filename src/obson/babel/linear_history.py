"""A linear historical readout: change the main decoding task, not an auxiliary loss."""

import torch
from torch import nn

from . import shared_history as original

MODES = ("native_joint", "linear_frozen", "linear_joint")
FAMILIES, hq, structure, price = original.FAMILIES, original.hq, original.structure, original.price
losses = original.losses


class LinearHistory(nn.Module):
    """One affine map per age, from the same single causal state; no history bypass."""

    def __init__(self, mean, scale):
        super().__init__()
        mean, scale = (
            torch.as_tensor(mean).detach().clone(),
            torch.as_tensor(scale).detach().clone(),
        )
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Finite frozen train-only state coordinates required")
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_scale", scale)
        # Zero starts have a well-defined constant prediction, without consuming global RNG.
        self.weight = nn.Parameter(mean.new_zeros(127, 7, len(mean)))
        self.bias = nn.Parameter(mean.new_zeros(127, 7))

    def forward(self, z, ages=None):
        if z.ndim != 2 or z.shape[1] != len(self.state_mean):
            raise ValueError("Exactly one fixed-width causal state per context required")
        if ages is None:
            ages = torch.arange(1, 128, device=z.device)
        if ages.ndim != 1 or ages.dtype != torch.long or not ((ages >= 1) & (ages <= 127)).all():
            raise ValueError("Only historical ages1..127 allowed")
        w, b = self.weight[ages - 1], self.bias[ages - 1]
        normalized = (z - self.state_mean) / self.state_scale
        return (normalized @ w.flatten(0, 1).T).reshape(len(z), len(ages), 7) + b


def configure(model, query, joint):
    model.eval().requires_grad_(False)
    model.core.encoder.requires_grad_(joint)
    # This diagnostic head never backpropagates into the encoder.
    model.local_head.requires_grad_(True)
    query.eval().requires_grad_(True)
    return [
        ("encoder_lr", list(model.core.encoder.parameters()), 0.01),
        ("head_lr", list(model.local_head.parameters()), 1e-4),
        ("query_lr", list(query.parameters()), 1e-4),
    ][0 if joint else 1 :]
