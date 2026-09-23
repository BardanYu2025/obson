"""Fixed, past-only market-state diagnostics and closed-form linear readouts."""
import numpy as np
import pandas as pd
import torch

HORIZONS = (16, 64)
FAMILIES = ('directional_efficiency', 'trend_r2', 'log_realized_rms')
NAMES = tuple(f'{family}/{h}' for h in HORIZONS for family in FAMILIES)
ALPHAS = (1., .1, .01, .001, .0001)
DIRECTION_THRESHOLD = .2


def descriptors(raw):
    """Each h-close path has h-1 internal returns, including the completed current bar."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (128, 28) or not np.isfinite(raw).all():
        raise ValueError('Expected finite [N,128,28] raw causal features')
    changes = np.sinh(raw[..., 0]) + np.sinh(raw[..., 1])  # log-percent, not sigma units
    targets = []
    for h in HORIZONS:
        r = changes[:, -h+1:] / 100
        path = np.c_[np.zeros(len(r)), np.cumsum(r, axis=1)]
        centered = path - path.mean(1, keepdims=True)
        t = np.arange(h, dtype=float); t -= t.mean()
        total = np.abs(r).sum(1)
        efficiency = np.divide(r.sum(1), total, out=np.zeros(len(r)), where=total > 1e-12)
        denom = np.square(centered).sum(1) * np.square(t).sum()
        strength = np.divide(np.square(centered @ t), denom, out=np.zeros(len(r)), where=denom > 1e-24)
        volatility = np.log(np.maximum(np.sqrt(np.square(r).mean(1)), 1e-8))
        targets.extend((efficiency, strength.clip(0, 1), volatility))
    y = np.column_stack(targets)
    if not np.isfinite(y).all(): raise ValueError('Nonfinite diagnostic target')
    return y


def scales(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all(): raise ValueError('Finite training matrix required')
    return dict(mean=x.mean(0).tolist(), scale=x.std(0).clip(1e-6).tolist())


def normalize(x, stats):
    return (np.asarray(x, dtype=np.float64)-stats['mean'])/stats['scale']


def eigensystem(x, device):
    a = torch.as_tensor(x, dtype=torch.float64, device=device)
    if len(a) < a.shape[1]:
        # The right-hand side of ridge is in the training row space; omitted
        # null directions have zero coefficient. Useful for small test fixtures.
        _, singular, vh = torch.linalg.svd(a, full_matrices=False)
        return (singular.square()/len(a)).flip(0), vh.T.flip(1)
    values, vectors = torch.linalg.eigh(a.T @ a / len(a))
    return values.clamp_min(0), vectors


def ridge_select(train, target, validation, val_target, device, eigen=None):
    """MSE/n + alpha*||W||², unpenalized intercept. Same grid for every representation."""
    stats = scales(train); x = normalize(train, stats); v = normalize(validation, stats)
    eigen = eigensystem(x, device) if eigen is None else eigen
    values, vectors = eigen
    xt = torch.as_tensor(x, dtype=torch.float64, device=device)
    yt = torch.as_tensor(target, dtype=torch.float64, device=device)
    intercept = yt.mean(0); rhs = vectors.T @ (xt.T @ (yt-intercept) / len(xt))
    vt = torch.as_tensor(v, dtype=torch.float64, device=device)
    truth = torch.as_tensor(val_target, dtype=torch.float64, device=device)
    candidates = []; best = None
    for alpha in ALPHAS:
        weights = vectors @ (rhs / (values[:, None]+alpha))
        loss = float(((vt @ weights+intercept-truth)**2).mean().item())
        candidates.append(dict(alpha=alpha, validation_nmse=loss))
        if best is None or loss < best['validation_nmse']:
            best = dict(alpha=alpha, validation_nmse=loss, weights=weights.cpu().numpy(),
                        intercept=intercept.cpu().numpy(), statistics=stats)
    if not all(np.isfinite(c['validation_nmse']) for c in candidates): raise ValueError('Invalid ridge selection')
    return best, candidates


def predict(head, x):
    return normalize(x, head['statistics']) @ head['weights'] + head['intercept']


def direction(values):
    return np.where(values < -DIRECTION_THRESHOLD, 0, np.where(values > DIRECTION_THRESHOLD, 2, 1))


def classification(predicted, actual):
    p, y = direction(predicted), direction(actual)
    matrix = np.zeros((3, 3), dtype=np.int64); np.add.at(matrix, (y, p), 1)
    support = matrix.sum(1); present = support > 0
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(3), where=present)
    denom = support+matrix.sum(0)
    f1 = np.divide(2*matrix.diagonal(), denom, out=np.zeros(3), where=denom > 0)
    return dict(classes=['down','mixed','up'], confusion=matrix.tolist(), support=support.tolist(),
        balanced_accuracy_present=float(recall[present].mean()), macro_f1_present=float(f1[present].mean()),
        accuracy=float((p == y).mean()), all_classes_present=bool(present.all()))


def measure(pred, truth, target_stats):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != 6 or not len(truth):
        raise ValueError('Matched nonempty six-target arrays required')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all(): raise ValueError('Nonfinite predictions')
    errors = np.square((pred-truth)/target_stats['scale'])
    targets = []
    for i, name in enumerate(NAMES):
        variance = float(np.var(truth[:, i]))
        targets.append(dict(name=name, nmse=float(errors[:, i].mean()), mae=float(np.abs(pred[:, i]-truth[:, i]).mean()),
            r2=float(1-np.mean((pred[:, i]-truth[:, i])**2)/variance) if variance > 1e-12 else None))
    per_window = dict(primary=errors.mean(1))
    for i, name in enumerate(FAMILIES): per_window[name] = errors[:, i::3].mean(1)
    report = dict(windows=len(truth), primary=float(errors.mean()), targets=targets,
        families={name:float(per_window[name].mean()) for name in FAMILIES},
        direction={str(h):classification(pred[:, 3*i], truth[:, 3*i]) for i, h in enumerate(HORIZONS)})
    return report, per_window


def inventory_audit(inventories):
    """Verify pinned row identities and disjoint128-bar input ranges across time splits."""
    grouped = {}; report = {}
    for split, rows in inventories.items():
        seen = set(); grouped[split] = {}
        for r in rows:
            key = r['key']; parts = key.split('/'); end = int(r['row'])
            if len(parts) != 3 or parts[0] != r['symbol'] or int(parts[1]) != int(r['period']) or end < 511:
                raise ValueError('Invalid endpoint or insufficient source warmup')
            if (key, end) in seen: raise ValueError('Duplicate endpoint')
            seen.add((key, end)); pd.Timestamp(r['end'])
            grouped[split].setdefault(key, []).append(end)
        report[split] = dict(windows=len(rows), symbols=sorted({r['symbol'] for r in rows}),
            contract_periods=len(grouped[split]), first_end=min(r['end'] for r in rows), last_end=max(r['end'] for r in rows))
    for a,b in (('train','val'),('train','test'),('val','test')):
        if report[a]['last_end'] >= report[b]['first_end']: raise ValueError('Chronological endpoint order violated')
        for key in set(grouped[a]) & set(grouped[b]):
            if max(grouped[a][key]) >= min(grouped[b][key])-127: raise ValueError('Cross-split128-bar input overlap')
    original = set().union(*(set(report[s]['symbols']) for s in ('train','val','test')))
    if original & set(report['cross_research']['symbols']): raise ValueError('Cross-symbol set overlaps original symbols')
    report['scope'] = 'Chronological original splits may share contracts; cross set has disjoint symbols. Reused research sets, not a new holdout. Causal pre-window EMA history may cross a split boundary.'
    return report
