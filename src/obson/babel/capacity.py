"""Native-width causal states and an explicitly rank-controlled reconstruction path."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .architecture import NAMES
from .history_autoencoder import positions

CONFIG = dict(latent=512, gru_layers=2, attention_layers=2, attention_ff=512,
              conv_layers=6, conv_inner=320, heads=8, decoder_width=256,
              decoder_layers=2, window=128)


class GRUEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c['latent']
        self.input = nn.Sequential(nn.Linear(28, w), nn.LayerNorm(w), nn.GELU())
        self.rnn = nn.GRU(w, w, c['gru_layers'], batch_first=True)

    def forward(self, x):
        return self.rnn(self.input(x))[0]


class AttentionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        w = c['latent']
        self.input = nn.Linear(28, w)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            w, c['heads'], c['attention_ff'], dropout=0., batch_first=True,
            norm_first=True, activation='gelu') for _ in range(c['attention_layers'])])

    def forward(self, x):
        h = self.input(x)
        h = h + positions(x.shape[1], h.shape[-1], h.device, h.dtype)
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return h  # No terminal projection or normalization of the complete state.


class ScaleBlock(nn.Module):
    def __init__(self, width, inner, dilation, kernel):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2*inner)
        self.depthwise = nn.Conv1d(inner, inner, kernel, dilation=dilation, groups=inner)
        self.down = nn.Linear(inner, width)
        self.left = (kernel-1)*dilation

    def forward(self, x):
        v, gate = self.up(self.norm(x)).chunk(2, -1)
        v = self.depthwise(F.pad(v.transpose(1, 2), (self.left, 0))).transpose(1, 2)
        return x + self.down(F.gelu(v)*gate.sigmoid())


class MultiScaleEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input = nn.Linear(28, c['latent'])
        self.blocks = nn.ModuleList([ScaleBlock(c['latent'], c['conv_inner'], 2**i,
                                               4 if i == 0 else 3) for i in range(c['conv_layers'])])
        self.receptive_field = 1 + sum(b.left for b in self.blocks)

    def forward(self, x):
        h = self.input(x)
        for layer in self.blocks:
            h = layer(h)
        return h


def orthogonal_basis(width):
    """Data-independent rotation, generated without changing the model RNG."""
    gen = torch.Generator(device='cpu').manual_seed(7001)
    q, _ = torch.linalg.qr(torch.randn(width, width, generator=gen, dtype=torch.float64))
    return q.float()


class MemoryDecoder(nn.Module):
    """One endpoint → rank constraint → two memory tokens → fixed position queries.

    All sample-specific conditioning is in the two memory tokens collectively.
    Query width is not the aggregate memory capacity. No input/target skip path.
    """
    def __init__(self, c, rank):
        super().__init__()
        w, d = c['decoder_width'], c['latent']
        if d != 2*w:
            raise ValueError('The decoder requires exactly two half-width memory tokens')
        if d % 2 or rank not in (d//2, d):
            raise ValueError('Only half/full even latent widths are supported')
        self.window, self.width, self.rank = c['window'], w, rank
        self.register_buffer('basis', orthogonal_basis(d))
        self.register_buffer('coordinate_mask', (torch.arange(d) < rank).float())
        self.memory = nn.Linear(d, d)
        nn.init.eye_(self.memory.weight)
        nn.init.zeros_(self.memory.bias)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
            w, c['heads'], 4*w, dropout=0., batch_first=True, norm_first=True,
            activation='gelu') for _ in range(c['decoder_layers'])])
        self.output = nn.Linear(w, len(NAMES))

    def conditioning(self, z):
        if z.ndim != 2 or z.shape[-1] != 2*self.width:
            raise ValueError('Decoder accepts one native-width endpoint per sample')
        # Explicit zeros create a true coordinate bottleneck. A precomputed FP32
        # dense low-rank projector could leave tiny nonzero leakage directions.
        coordinates = (z @ self.basis) * self.coordinate_mask
        constrained = (coordinates @ self.basis.T)*math.sqrt(2*self.width/self.rank)
        return self.memory(constrained).reshape(len(z), 2, self.width)

    def forward(self, z):
        memory = self.conditioning(z)
        memory = memory + positions(2, self.width, memory.device, memory.dtype)
        h = positions(self.window, self.width, memory.device, memory.dtype).expand(len(z), -1, -1)
        for layer in self.layers:
            h = layer(h, memory)
        return self.output(h)

    @property
    def projector(self):
        return (self.basis*self.coordinate_mask) @ self.basis.T * math.sqrt(2*self.width/self.rank)

    @torch.no_grad()
    def conditioning_audit(self):
        basis = self.basis.double().cpu()
        constraint = (basis*self.coordinate_mask.double().cpu()) @ basis.T
        matrix = self.memory.weight.double().cpu() @ constraint.T * math.sqrt(2*self.width/self.rank)
        singular = torch.linalg.svdvals(matrix)
        threshold = max(float(singular[0])*1e-5, 1e-8)
        return dict(declared_rank=self.rank, numerical_rank=int((singular > threshold).sum()),
                    rank_threshold=threshold, singular_values=singular.tolist(),
                    interpretation='Rank of endpoint to flattened memory tokens, not proof of learned useful information or full output Jacobian rank.')


ENCODERS = dict(gru=GRUEncoder, attention=AttentionEncoder, multiscale=MultiScaleEncoder)


class CapacityModel(nn.Module):
    def __init__(self, architecture, config, seed, rank):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = ENCODERS[architecture](config)
            torch.manual_seed(seed + 9109)
            self.decoder = MemoryDecoder(config, rank)

    def forward(self, x):
        return self.decoder(self.encoder(x)[:, -1])
