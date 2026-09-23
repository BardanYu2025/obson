"""Position-shared frozen-state readouts and train-only linear capacity references."""
import numpy as np

from . import prefix_readout as pr

TRAIN_PREFIXES = (32, 64, 96, 128)
HELD_PREFIXES = (48, 80, 112)
PREFIXES = tuple(sorted(TRAIN_PREFIXES + HELD_PREFIXES))
RANKS = (32, 64, 128, 256, 512, 768, 896)


def targets(data, stats, prefix):
    """Identical past16 target at both trained and held positions, no future rows."""
    if prefix not in PREFIXES:
        raise ValueError('Unregistered prefix')
    start, stop = prefix-pr.HISTORY-1, prefix-1
    scale, mean = np.asarray(stats['y_scale']), np.asarray(stats['y_mean'])
    y = np.asarray(data['y'][:, start:stop], dtype=np.float64)*scale+mean
    anchor = np.asarray(data['y'][:, start-1, 0], dtype=np.float64)*scale[0]+mean[0]
    mask = np.asarray(data['mask'][:, start:stop], dtype=bool).copy()
    if not mask[..., 0].all() or not np.asarray(data['mask'][:, start-1, 0]).all():
        raise ValueError('Local price targets require observed anchors and closes')
    y[..., 0] -= anchor[:, None]
    return np.where(mask, y, 0).astype(np.float32), mask


def pool(rows):
    """Only registered training positions enter fitting/validation selection."""
    if set(rows) != set(TRAIN_PREFIXES):
        raise ValueError('Shared fitting requires exactly the four training prefixes')
    if len({len(v['x']) for v in rows.values()}) != 1:
        raise ValueError('All positions require the same source window cohort')
    return {k: np.concatenate([rows[p][k] for p in TRAIN_PREFIXES]) for k in ('x', 'y', 'mask')}


def pca_spectrum(pca, ranks):
    energy = np.asarray(pca['singular'], dtype=np.float64)**2
    if not energy.sum() > 0 or max(ranks) > len(energy):
        raise ValueError('Invalid PCA spectrum or rank')
    return {str(rank): float(energy[:rank].sum()/energy.sum()) for rank in ranks}
