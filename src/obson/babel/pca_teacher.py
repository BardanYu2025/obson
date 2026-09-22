"""Causal student with an explicit PCA-coordinate state and a state-only decoder."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import capacity as ca
from .architecture import NAMES

CONFIG = dict(ca.CONFIG, residual_width=256)


def coefficients(pca, data):
    flat = np.asarray(data['y']).reshape(len(data['y']), -1)
    return ((flat-pca['mean']) @ pca['components'].T).astype(np.float32)


def fit_coordinate_scales(train_coefficients):
    c = np.asarray(train_coefficients, dtype=np.float64)
    scale = c.std(0).clip(1e-4)
    return dict(mean=c.mean(0).tolist(), scale=scale.tolist(), floor=1e-4,
                fitted_on='train_only', count=len(c), width=c.shape[1])


def normalize_coordinates(c, stats):
    return ((c-np.array(stats['mean']))/np.array(stats['scale'])).astype(np.float32)


class CoordinateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ca.AttentionEncoder(config)
        self.coordinates = nn.Linear(config['latent'], config['latent'])
        nn.init.normal_(self.coordinates.weight, std=.02)
        nn.init.zeros_(self.coordinates.bias)

    def forward(self, x):
        return self.coordinates(self.backbone(x))


class PCADecoder(nn.Module):
    """Only predicted coordinates enter this decoder; no true-history skip path."""
    def __init__(self, config, pca, scales, residual=False):
        super().__init__()
        self.window = config['window']
        width = config['latent']
        if pca['components'].shape != (width, self.window*len(NAMES)):
            raise ValueError('PCA basis shape differs from declared state/output')
        if (len(scales['scale']) != width or len(scales['mean']) != width or min(scales['scale']) <= 0
                or not all(np.isfinite(np.asarray(v)).all() for v in (scales['scale'],scales['mean'],pca['components'],pca['mean']))):
            raise ValueError('Invalid coordinate normalization')
        for name, value in [('basis', pca['components']), ('target_mean', pca['mean']),
                            ('coordinate_mean', scales['mean']), ('coordinate_scale', scales['scale'])]:
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=torch.float32))
        self.residual_enabled = residual
        self.residual = nn.Sequential(nn.Linear(width, config['residual_width']), nn.GELU(),
                                     nn.Linear(config['residual_width'], self.window*len(NAMES)))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.residual.requires_grad_(residual)

    def base(self, z):
        if z.ndim != 2 or z.shape[1] != len(self.coordinate_mean):
            raise ValueError('Expected one coordinate vector per endpoint')
        c = z*self.coordinate_scale + self.coordinate_mean
        return (c @ self.basis + self.target_mean).reshape(-1, self.window, len(NAMES))

    def forward(self, z):
        base = self.base(z)
        return base + self.residual(z).reshape_as(base) if self.residual_enabled else base


class Student(nn.Module):
    def __init__(self, config, seed, pca, scales, residual=False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = CoordinateEncoder(config)
            torch.manual_seed(seed+9109)
            self.decoder = PCADecoder(config, pca, scales, residual)

    def forward(self, x):
        return self.decoder(self.encoder(x)[:, -1])


def alignment_rows(z, q):
    if z.shape != q.shape:
        raise ValueError('Student and teacher coordinate shapes differ')
    return dict(coordinate_mse=(z-q).square().mean(1),
                coordinate_smooth=F.smooth_l1_loss(z, q, reduction='none').mean(1))


def teacher_weight(job, epoch, release_epochs):
    if job['schedule'] == 'constant':
        return job['teacher_weight']
    if job['schedule'] != 'release' or release_epochs < 2:
        raise ValueError('Invalid teacher schedule')
    return job['teacher_weight']*max(0., 1-max(epoch-1, 0)/(release_epochs-1))
