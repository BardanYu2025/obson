"""Two-factor causal bar-state training: coordinate width and local encoder supervision."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import pca_teacher as pt, architecture as ar

HELD_PREFIXES = (48, 80, 112)
TRAIN_PREFIXES = tuple(p for p in range(32, 129) if p not in HELD_PREFIXES)
VAL_PREFIXES = (32, 64, 96, 128)
LOCAL_COUNT = 4
LOCAL_WEIGHT = .25
HISTORY = 16


def position_plan(n, seed, epoch):
    """Four distinct positions per sampled window, independent of model/batch RNG."""
    if n < 1 or epoch < 1:
        raise ValueError('Positive position-plan size and epoch required')
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260924]))
    ranks = rng.random((n, len(TRAIN_PREFIXES)))
    chosen = np.argpartition(ranks, LOCAL_COUNT-1, axis=1)[:, :LOCAL_COUNT]
    return np.sort(np.array(TRAIN_PREFIXES, dtype=np.int64)[chosen], axis=1)


def local_targets(y, mask, prefixes, original, local):
    """Convert normalized full paths into relative past16 targets, excluding current bar."""
    if prefixes.ndim != 2 or prefixes.shape[0] != len(y) or prefixes.min() < 18 or prefixes.max() > y.shape[1]:
        raise ValueError('Invalid past-history prefix indices')
    rows = torch.arange(len(y), device=y.device)[:, None, None]
    indices = prefixes[..., None]-HISTORY-1+torch.arange(HISTORY, device=y.device)
    work = y.to(torch.float64)
    scale = work.new_tensor(original['y_scale']); mean = work.new_tensor(original['y_mean'])
    values = work[rows, indices]*scale+mean
    valid = mask[rows, indices]
    anchor_idx = prefixes-HISTORY-2
    anchor = work[torch.arange(len(y), device=y.device)[:, None], anchor_idx, 0]*scale[0]+mean[0]
    if not valid[..., 0].all() or not mask[torch.arange(len(y), device=y.device)[:, None], anchor_idx, 0].all():
        raise ValueError('Local targets require observed price anchors')
    # Clone to avoid overwriting any source cache or shared view.
    values = values.clone(); values[..., 0] -= anchor[..., None]
    # Match the established two-stage float32 target cache exactly.
    values = values.to(y.dtype).to(torch.float64)
    normalized = (values-work.new_tensor(local['mean']))/work.new_tensor(local['scale'])
    return torch.where(valid, normalized, 0).to(y.dtype), valid


def local_rows(pred, y, mask, smooth=False):
    e = F.smooth_l1_loss(pred, y, reduction='none') if smooth else (pred-y).square()
    count = mask.sum(-2); channels = (e*mask).sum(-2)/count.clamp_min(1); ok = count > 0
    activity = (channels[..., 2:]*ok[..., 2:]).sum(-1)/ok[..., 2:].sum(-1).clamp_min(1)
    return dict(path=channels[..., 0].mean(1), body=channels[..., 1].mean(1), activity=activity.mean(1),
                primary=((channels[..., 0]+channels[..., 1]+activity)/3).mean(1))


class AlignedStudent(nn.Module):
    def __init__(self, config, seed, pca, scales, hidden=256):
        super().__init__()
        self.core = pt.Student(config, seed, pca, scales, residual=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+13001)
            self.local_head = nn.Sequential(nn.Linear(config['latent'], hidden), nn.GELU(), nn.Linear(hidden, HISTORY*7))

    def forward(self, x, prefixes, joint):
        states = self.core.encoder(x)
        chosen = states[torch.arange(len(x), device=x.device)[:, None], prefixes-1]
        if not joint:
            chosen = chosen.detach()  # Control reader learns; its loss cannot modify the encoder.
        local = self.local_head(chosen).reshape(len(x), prefixes.shape[1], HISTORY, 7)
        return self.core.decoder(states[:, -1]), local


def run_epoch(model, data, original, local, batch, micro, device, joint, prefixes, encoder_opt=None, head_opt=None, order_seed=0):
    training = encoder_opt is not None
    if training != (head_opt is not None): raise ValueError('Both optimizers required for training')
    n = len(data['x']); model.train(training)
    if prefixes.shape != (n, LOCAL_COUNT): raise ValueError('Exactly four positions per window required')
    if training and (not np.isin(prefixes, TRAIN_PREFIXES).all() or np.any(np.diff(np.sort(prefixes, axis=1), axis=1)==0)):
        raise ValueError('Training cannot include held or repeated positions')
    order = np.random.default_rng(order_seed).permutation(n) if training else np.arange(n)
    totals = {}; steps = 0
    with torch.set_grad_enabled(training):
        for left in range(0, n, batch):
            ids = order[left:left+batch]
            if training: encoder_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
            for start in range(0, len(ids), micro):
                use = ids[start:start+micro]
                b = {k: torch.tensor(np.asarray(v[use]), device=device) for k, v in data.items()}
                ps = torch.tensor(prefixes[use], dtype=torch.long, device=device)
                pred, detail = model(b['x'], ps, joint)
                target, mask = local_targets(b['y'], b['mask'], ps, original, local)
                global_rows = ar.error_rows(pred, b['y'], b['mask'], original)
                g_smooth = ar.error_rows(pred, b['y'], b['mask'], original, True)['primary']
                l_rows = local_rows(detail, target, mask); l_smooth = local_rows(detail, target, mask, True)['primary']
                loss = g_smooth+LOCAL_WEIGHT*l_smooth
                rows = dict(global_rows, **{'local_'+k: v for k, v in l_rows.items()},
                            selection=global_rows['primary']+LOCAL_WEIGHT*l_rows['primary'],
                            global_smooth=g_smooth, local_smooth=l_smooth, total_objective=loss)
                if any(not torch.isfinite(v).all() for v in rows.values()): raise ValueError('Nonfinite bar-alignment objective')
                if training: (loss.sum()/len(ids)).backward()
                for k, v in rows.items(): totals[k] = totals.get(k, 0.)+float(v.detach().sum())
            if training:
                # Reader gradients must not alter the endpoint control's clipping factor.
                nn.utils.clip_grad_norm_(model.core.parameters(), 1., error_if_nonfinite=True)
                nn.utils.clip_grad_norm_(model.local_head.parameters(), 1., error_if_nonfinite=True)
                encoder_opt.step(); head_opt.step(); steps += 1
    return {k: v/n for k, v in totals.items()}, steps
