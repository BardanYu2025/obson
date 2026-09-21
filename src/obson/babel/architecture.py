"""Controlled causal encoders and ordered, observed price/activity reconstruction."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .history_autoencoder import positions

NAMES = ('close_log_percent', 'body_log_percent', 'volume_vs_prior_ema32',
         'delta_log_volume', 'oi_change_percent_asinh', 'oi_per_volume_asinh', 'turnover_asinh')
HORIZONS = (1, 4, 16, 32)
WEIGHTS = {'path': .4, 'changes': .2, 'body': .1, 'activity': .3}
CONFIG = dict(latent=512, gru_width=512, gru_layers=2, attention_width=256,
              attention_layers=4, heads=8, conv_width=256, conv_layers=8,
              decoder_width=256, decoder_layers=2, window=128)


def ordered_targets(x):
    """Observed log-price trajectory; no wicks or invented order-flow labels.

    x[...,0:2] are asinh(log-percent gap/body). The anchor is the close
    immediately before the first supplied bar. Every target is derived from
    supplied causal features. Final current-bar targets are excluded from loss.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 28 or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,T,28] causal features')
    geometry = np.sinh(x[..., :2])
    y = np.concatenate((geometry.sum(-1).cumsum(-1)[..., None], geometry[..., 1:2], x[..., 18:23]), -1)
    mask = np.ones_like(y, dtype=bool)
    mask[..., 3] = x[..., 27] > 0
    mask[..., 4] = x[..., 23] > 0
    mask[..., 5] = x[..., 24] > 0
    mask[..., 6] = x[..., 25] > 0
    suspect = ((x[..., 23] > 0) & (x[..., 22] == 0) & (x[..., 20] != 0)) | (
        (x[..., 24] > 0) & (np.abs(x[..., 21]) > np.arcsinh(1.) + 1e-6))
    mask[..., 4:6] &= ~suspect[..., None]
    mask[:, -1] = False  # Primary objective is previous observations, not current copying.
    y = np.where(mask, y, 0).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite reconstructed historical targets')
    return y, mask


def fit_scales(x, y, mask):
    """Training split only; caller never supplies validation/research rows."""
    count = mask.sum((0, 1))
    if (count < 2).any():
        raise ValueError('Insufficient training target support')
    ym = np.where(mask, y, 0).sum((0, 1), dtype=np.float64) / count
    ys = np.sqrt(np.where(mask, (y-ym)**2, 0).sum((0, 1)) / count).clip(1e-5)
    deltas = []
    for h in HORIZONS:
        valid = mask[:, h:, 0] & mask[:, :-h, 0]
        d = (y[:, h:, 0] - y[:, :-h, 0])[valid].astype(float)
        if not len(d):
            raise ValueError('Insufficient delta support')
        deltas.append(max(float(np.sqrt(np.mean(d*d))), .01))
    xm = x.mean((0, 1), dtype=np.float64)
    xs = x.std((0, 1), dtype=np.float64)
    xs[xs < 1e-6] = 1.
    # Availability flags stay binary, including flags constant in training but
    # different in evaluation. Do not amplify an unseen flag by100000.
    flags = [11, 23, 24, 25, 26, 27]
    xm[flags] = 0.; xs[flags] = 1.
    return dict(x_mean=xm.tolist(), x_scale=xs.tolist(),
                y_mean=ym.tolist(), y_scale=ys.tolist(), delta_scale=deltas,
                target_support=count.tolist(), fitted_on='train_only')


def normalize(x, y, mask, stats):
    xx = ((x-np.array(stats['x_mean']))/np.array(stats['x_scale'])).astype(np.float32)
    yy = np.where(mask, (y-np.array(stats['y_mean']))/np.array(stats['y_scale']), 0).astype(np.float32)
    return xx, yy, mask


def masked_rows(values, mask):
    """Equal available channels per window, then caller averages windows."""
    count = mask.sum(1)
    channel = (values * mask).sum(1) / count.clamp_min(1)
    valid = count > 0
    return (channel * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def error_rows(pred, target, mask, stats, smooth=False):
    error = lambda a, b: F.smooth_l1_loss(a, b, reduction='none') if smooth else (a-b).square()
    e = error(pred, target)
    rows = {name: masked_rows(e[..., ids], mask[..., ids]) for name, ids in
            [('path', slice(0, 1)), ('body', slice(1, 2)), ('activity', slice(2, 7))]}
    ds = []
    for h, scale in zip(HORIZONS, stats['delta_scale']):
        # Normalized price offsets cancel; convert to the train-fitted delta scale.
        p = (pred[:, h:, :1]-pred[:, :-h, :1]) * (stats['y_scale'][0]/scale)
        y = (target[:, h:, :1]-target[:, :-h, :1]) * (stats['y_scale'][0]/scale)
        ds.append(masked_rows(error(p, y), mask[:, h:, :1] & mask[:, :-h, :1]))
        rows[f'change{h}'] = ds[-1]
    rows['changes'] = torch.stack(ds).mean(0)
    rows['primary'] = sum(WEIGHTS[k]*rows[k] for k in WEIGHTS)
    return rows


class GRUEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c['gru_width']
        self.input = nn.Sequential(nn.Linear(28, w), nn.LayerNorm(w), nn.GELU())
        self.rnn = nn.GRU(w, w, c['gru_layers'], batch_first=True)
        self.output = nn.LayerNorm(w) if w == c['latent'] else nn.Sequential(nn.Linear(w, c['latent']), nn.LayerNorm(c['latent']))

    def forward(self, x):
        h, _ = self.rnn(self.input(x))  # Reset for every common-context window.
        return self.output(h)


class AttentionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c['attention_width']
        self.input = nn.Linear(28, w)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(w, c['heads'], 4*w, dropout=0.,
            batch_first=True, norm_first=True, activation='gelu') for _ in range(c['attention_layers'])])
        self.output = nn.Sequential(nn.Linear(w, c['latent']), nn.LayerNorm(c['latent']))

    def forward(self, x):
        h = self.input(x)
        h = h + positions(x.shape[1], h.shape[-1], h.device, h.dtype)
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.output(h)


class CausalScaleBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        inner = width*2
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2*inner)
        self.depthwise = nn.Conv1d(inner, inner, 3, dilation=dilation, groups=inner)
        self.down = nn.Linear(inner, width)
        self.left = 2*dilation

    def forward(self, x):
        v, gate = self.up(self.norm(x)).chunk(2, -1)
        v = self.depthwise(F.pad(v.transpose(1, 2), (self.left, 0))).transpose(1, 2)
        return x + self.down(F.gelu(v)*gate.sigmoid())


class MultiScaleEncoder(nn.Module):
    """Causal dilated convolution. Explicit scales, not a new persistent memory."""
    def __init__(self, c):
        super().__init__()
        w = c['conv_width']
        self.input = nn.Linear(28, w)
        self.blocks = nn.ModuleList([CausalScaleBlock(w, 2**i) for i in range(c['conv_layers'])])
        self.output = nn.Sequential(nn.Linear(w, c['latent']), nn.LayerNorm(c['latent']))

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class PositionDecoder(nn.Module):
    """Only one endpoint vector and fixed positions. No target/input skip path."""
    def __init__(self, c):
        super().__init__()
        w = c['decoder_width']
        self.window = c['window']
        self.input = nn.Sequential(nn.LayerNorm(c['latent']), nn.Linear(c['latent'], w))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(w, c['heads'], 4*w, dropout=0.,
            batch_first=True, norm_first=True, activation='gelu') for _ in range(c['decoder_layers'])])
        self.output = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, len(NAMES)))

    def forward(self, z):
        if z.ndim != 2:
            raise ValueError('Decoder accepts endpoint vectors only')
        h = self.input(z)[:, None]
        h = h + positions(self.window, h.shape[-1], h.device, h.dtype)
        for layer in self.layers:
            h = layer(h)
        return self.output(h)


ENCODERS = {'gru': GRUEncoder, 'attention': AttentionEncoder, 'multiscale': MultiScaleEncoder}


class OrderedModel(nn.Module):
    def __init__(self, architecture, config, seed):
        super().__init__()
        # Different encoder parameter counts must not change common decoder initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = ENCODERS[architecture](config)
            torch.manual_seed(seed + 9109)
            self.decoder = PositionDecoder(config)

    def forward(self, x):
        return self.decoder(self.encoder(x)[:, -1])


def learning_rate(epoch, total, maximum):
    warmup = min(5, total)
    if epoch <= warmup:
        return maximum * epoch/warmup
    phase = (epoch-warmup)/max(total-warmup, 1)
    return maximum*(.1 + .9*.5*(1+math.cos(math.pi*phase)))


def permute_old_blocks(x):
    """Synthetic sensitivity only: preserve current block and bar-feature multiset."""
    if x.shape[1] != 128:
        raise ValueError('Diagnostic requires128 rows')
    ids = np.r_[np.arange(112).reshape(7, 16)[::-1].ravel(), np.arange(112, 128)].copy()
    return x[:, ids]
