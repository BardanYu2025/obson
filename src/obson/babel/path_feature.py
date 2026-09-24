"""One deterministic causal price-path input, with an exactly unchanged start."""
import math

import numpy as np
import torch
from torch import nn


def causal_path(x, mean, scale, price_scale):
    """Normalized log-percent close relative to the close before this window.

    Each output uses only rows through itself. No endpoint centering, volatility
    fit, labels, decoder skip or new market observation is involved.
    """
    if x.ndim != 3 or x.shape[-1] != 28:
        raise ValueError('Expected normalized [B,T,28] input')
    geometry = x[..., :2].to(torch.float64)*scale.to(torch.float64)+mean.to(torch.float64)
    path = torch.sinh(geometry).sum(-1).cumsum(1)/price_scale.to(torch.float64)
    if not torch.isfinite(path).all():
        raise ValueError('Nonfinite causal path input')
    return path.unsqueeze(-1).to(x.dtype)


class PathInput(nn.Module):
    def __init__(self, original, stats, enabled):
        super().__init__()
        self.original = original
        self.register_buffer('mean', torch.tensor(stats['x_mean'][:2],dtype=torch.float64,device=original.weight.device))
        self.register_buffer('scale', torch.tensor(stats['x_scale'][:2],dtype=torch.float64,device=original.weight.device))
        self.register_buffer('price_scale', torch.tensor(stats['y_scale'][0],dtype=torch.float64,device=original.weight.device))
        self.register_buffer('enabled', torch.tensor(float(enabled),device=original.weight.device))
        if not torch.isfinite(self.price_scale) or self.price_scale <= 0:
            raise ValueError('Positive frozen train price scale required')
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(1,original.out_features,bias=False)
            nn.init.zeros_(self.projection.weight)
        self.projection.to(device=original.weight.device,dtype=original.weight.dtype)

    def forward(self, x):
        path = causal_path(x,self.mean,self.scale,self.price_scale)
        return self.original(x)+self.enabled*self.projection(path)


def install(model, stats, enabled):
    original = model.core.encoder.backbone.input
    if not isinstance(original,nn.Linear) or original.in_features != 28:
        raise ValueError('Expected the inherited 28-channel linear input')
    model.core.encoder.backbone.input = PathInput(original,stats,enabled)
    return model.core.encoder.backbone.input.projection.weight


def learning_rate(epoch, epochs, initial):
    """Continue from the parent's last LR, then cosine decay to one tenth."""
    if not 1 <= epoch <= epochs or epochs < 2 or initial <= 0:
        raise ValueError('Invalid continuation schedule')
    return initial*(.1+.9*(1+math.cos(math.pi*(epoch-1)/(epochs-1)))/2)


def decompose_path(prediction, target, mask, price_scale):
    """Exact masked squared-error decomposition: level, linear drift, remainder."""
    error=np.asarray(prediction,dtype=np.float64)[...,0]-np.asarray(target,dtype=np.float64)[...,0]
    valid=np.asarray(mask)[...,0];count=valid.sum(1)
    if (count<2).any():raise ValueError('Need at least two valid historical prices')
    mean=(error*valid).sum(1)/count
    t=np.broadcast_to(np.arange(error.shape[1],dtype=float),error.shape)
    t=t-(t*valid).sum(1)[:,None]/count[:,None]
    slope=(error*t*valid).sum(1)/(t*t*valid).sum(1)
    drift=slope[:,None]*t
    remainder=error-mean[:,None]-drift
    result=dict(path=(error*error*valid).sum(1)/count,offset=mean*mean,
        linear_drift=(drift*drift*valid).sum(1)/count,
        remainder=(remainder*remainder*valid).sum(1)/count,
        signed_offset_bps=mean*price_scale*100)
    if not np.allclose(result['path'],result['offset']+result['linear_drift']+result['remainder'],atol=1e-10,rtol=1e-10):
        raise ValueError('Path error decomposition mismatch')
    return result
