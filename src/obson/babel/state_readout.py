"""Frozen-state usefulness and paired long-refresh experiment; no encoder updates."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .ae_extend import atomic_json
from .data import split_mask
from .dual_state import sha256
from .fusion_readout_audit import load_cache
from .metrics import balanced_accuracy
from .progress import progress
from .representation import time_mask
from .residual_fusion import extract_short
from .stage_audit import load_data
from .streaming import load_bundle
from .structure import SCALES, STATE_NAMES

SCHEMA = 'babel-state-readout-v1'
SPLITS = ('train', 'val', 'test')
AGES = (0, 16, 48, 96, 127)
ALPHAS = (1., 10., 100., 1000.)
VARIANTS = ('price_ema', 'short', 'long_held', 'dual_held', 'price_dual_held',
            'long_fresh', 'dual_fresh', 'price_dual_fresh')


def sample_rows(series, bounds, split, cached, ages=AGES):
    """Current eligible bars only; never extend a label or neural context past a split.

    Each row points at a previously validated absolute-grid endpoint. Incomplete
    tail groups are retained rather than conditioning inclusion on future eligibility.
    """
    rows, source_ids = [], []
    masks = {i: split_mask(s, bounds, split) for i, s in enumerate(series)}
    starts = {}
    for i, s in enumerate(series):
        valid = np.flatnonzero(time_mask(s, bounds, split))
        if len(valid) and np.any(np.diff(valid) != 1):
            raise ValueError('Noncontiguous neural partition')
        starts[i] = int(valid[0]) if len(valid) else len(s.frame)
    for idx, ((i, end), blocks) in enumerate(zip(cached['keys'], cached['blocks'])):
        i, end, blocks = int(i), int(end), int(blocks)
        if end % 128 != 127 or not 4 <= blocks <= 16 or end + 1 - blocks * 128 < starts[i]:
            raise ValueError('Invalid source endpoint/context')
        for age in ages:
            row = end + age
            if row < len(masks[i]) and masks[i][row]:
                rows.append((i, row, end, age, blocks))
                source_ids.append(idx)
    result = np.asarray(rows, dtype=np.int64).reshape(-1, 5)
    if not len(result):
        raise ValueError(f'No eligible {split} samples')
    if len(np.unique(result[:, :2], axis=0)) != len(result):
        raise ValueError('Duplicate contract/row samples')
    return result, np.asarray(source_ids, dtype=np.int64)


def price_features(frame, encoded):
    """Past/current price geometry, EMA, returns and range; no pivot labels as inputs."""
    logc = np.log(frame.close)
    sigma = np.exp(encoded[:, 8]).clip(1e-4)
    parts = [encoded[:, list(range(9)) + [14, 15, 16, 17]]]
    for span in (4, 16, 32, 64, 128, 256):
        parts.append(np.arcsinh(logc.diff(span).fillna(0).to_numpy() / (sigma * np.sqrt(span)))[:, None])
    for span in (16, 64, 256):
        path = logc.diff().abs().rolling(span, min_periods=1).sum().clip(lower=1e-12)
        efficiency = (logc.diff(span).fillna(0) / path).fillna(0).to_numpy()
        hi = np.log(frame.high).rolling(span, min_periods=1).max()
        lo = np.log(frame.low).rolling(span, min_periods=1).min()
        position = ((logc - lo) / (hi - lo).clip(lower=1e-12)).to_numpy()
        parts.extend((efficiency[:, None], position[:, None]))
    return np.column_stack(parts).astype(np.float32)


@torch.no_grad()
def rolling_long(model, series, encoded, rows, held, batch, device):
    """Re-tile n complete128-bar blocks ending NOW, with the same n as held state.

    This is an alternative refresh policy, not a partial-block input to the old
    encoder. All local blocks are deduplicated per contract and end position.
    """
    result = held.copy()
    for i in np.unique(rows[:, 0]):
        ids = np.flatnonzero((rows[:, 0] == i) & (rows[:, 3] != 0))
        if not len(ids):
            continue
        ends = sorted({int(end) for j in ids for end in
                       range(int(rows[j, 1] - (rows[j, 4] - 1) * 128), int(rows[j, 1]) + 1, 128)})
        bank = {}
        for start in range(0, len(ends), batch):
            selected = ends[start:start + batch]
            x = np.stack([encoded[i]['x'][end - 127:end + 1] for end in selected])
            z = model.local.encode(torch.tensor(x, device=device))[:, -1].cpu().numpy()
            bank.update(zip(selected, z))
            if start % (batch * 20) == 0:
                progress(f'Rolling long {series[i].key}: local blocks {min(start+batch,len(ends))}/{len(ends)}')
        close = series[i].frame.close.to_numpy()
        for start in range(0, len(ids), batch):
            selected = ids[start:start + batch]
            z = np.zeros((len(selected), model.blocks, model.config['latent']), np.float32)
            offsets = np.zeros((len(selected), model.blocks), np.float32)
            valid = np.zeros((len(selected), model.blocks), bool)
            for lane, j in enumerate(selected):
                _, end, _, _, n = map(int, rows[j])
                local_ends = list(range(end - (n - 1) * 128, end + 1, 128))
                anchors = np.array([close[e - 128] if e >= 128 else series[i].frame.open.iloc[0]
                                    for e in local_ends])
                z[lane, -n:] = np.stack([bank[e] for e in local_ends])
                offsets[lane, -n:] = (np.log(anchors) - np.log(close[end - 128])) * 100
                valid[lane, -n:] = True
            result[selected] = model.summarize(torch.tensor(z, device=device),
                torch.tensor(offsets, device=device), torch.tensor(valid, device=device)).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite rolling long state')
    return result


def write_npz(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


@torch.no_grad()
def prepare_split(engine, base, bounds, split, old, out, identity, batch, stream_batch, device):
    started = time.perf_counter()
    cache = out / 'cache'; cache.mkdir(exist_ok=True)
    path = cache / f'{split}.npz'; index = path.with_suffix('.json')
    rows, source_ids = sample_rows(base.series, bounds, split, old)
    if path.exists() and index.exists():
        info = json.loads(index.read_text())
        if info['identity'] != identity or info['sha256'] != sha256(path):
            raise ValueError('Readout cache identity/hash mismatch; use a new run directory')
        with np.load(path, allow_pickle=False) as f:
            values = {k: f[k] for k in f.files}
        if not np.array_equal(values['rows'], rows):
            raise ValueError('Readout sample inventory changed')
        progress(f'Reusing complete {split} cache ({len(rows)} samples)')
        return values
    series = base.series
    labels = np.stack([series[i].labels['state'][row] for i, row, *_ in rows])
    previous_labels = np.stack([series[i].labels['state'][end] for i, _, end, *_ in rows])
    baseline = np.empty((len(rows), 25), np.float32)
    for i in np.unique(rows[:, 0]):
        ids = np.flatnonzero(rows[:, 0] == i)
        baseline[ids] = price_features(series[i].frame, base.encoded[i]['x'])[rows[ids, 1]]
    short = extract_short(engine.short_model, base, bounds, split, rows[:, :2], stream_batch, device)
    held = old['long'][source_ids]
    fresh = rolling_long(engine.long_model, series, base.encoded, rows, held, batch, device)
    values = dict(rows=rows, labels=labels, previous_labels=previous_labels, price_ema=baseline,
                  short=short, long_held=held, long_fresh=fresh,
                  symbol=np.array([series[i].code for i in rows[:, 0]]),
                  period=np.array([series[i].period for i in rows[:, 0]]),
                  # Session week blocks keep simultaneous symbols/periods together in uncertainty estimates.
                  week=np.array([str(pd.Timestamp(series[i].sessions[row]).to_period('W-SUN'))
                                 for i, row, *_ in rows]))
    for key in ('price_ema', 'short', 'long_held', 'long_fresh'):
        if not np.isfinite(values[key]).all():
            raise ValueError(f'Nonfinite {key}')
    write_npz(path, **values)
    atomic_json(dict(identity=identity, sha256=sha256(path), samples=len(rows),
                     extraction_seconds=time.perf_counter()-started), index)
    return values


def design(data, variant):
    parts = []
    if variant.startswith('price'):
        parts.append(data['price_ema'])
    if variant == 'short' or 'dual' in variant:
        parts.append(data['short'])
    if 'held' in variant:
        parts.append(data['long_held'])
    if 'fresh' in variant:
        parts.append(data['long_fresh'])
    # Same observable update-age metadata for every learned baseline.
    parts.append(data['rows'][:, 3:4] / 127.)
    return np.column_stack(parts).astype(np.float64)


def fit_ridge(train, labels, val, val_labels, alphas=ALPHAS):
    """Train-only normalization/class weights, validation-only alpha; no test input."""
    mean, scale = train.mean(0), train.std(0).clip(.01)
    a = np.column_stack(((train - mean) / scale, np.ones(len(train))))
    b = np.column_stack(((val - mean) / scale, np.ones(len(val))))
    counts = np.bincount(labels, minlength=4)
    weights = len(labels) / (max(1, (counts > 0).sum()) * counts[labels].clip(1))
    lhs = a.T @ (a * weights[:, None])
    rhs = a.T @ (np.eye(4)[labels] * weights[:, None])
    best, candidates = None, []
    for alpha in alphas:
        penalty = np.eye(a.shape[1]) * alpha; penalty[-1, -1] = 0
        coef = np.linalg.solve(lhs + penalty, rhs)
        ba = balanced_accuracy((b @ coef).argmax(1), val_labels, 4)['ba']
        candidates.append(dict(alpha=alpha, validation_ba=ba))
        if best is None or ba > best['validation_ba']:
            best = dict(mean=mean, scale=scale, coef=coef, alpha=alpha, validation_ba=ba)
    best['candidates'] = candidates
    return best


def predict(fit, x):
    return (np.column_stack(((x - fit['mean']) / fit['scale'], np.ones(len(x)))) @ fit['coef']).argmax(1)


def score(pred, truth):
    value = balanced_accuracy(pred, truth, 4)
    confusion = np.bincount(truth * 4 + pred, minlength=16).reshape(4, 4)
    precision = np.diag(confusion) / confusion.sum(0).clip(1)
    value.update(samples=len(truth), accuracy=float((pred == truth).mean()) if len(truth) else None,
                 confusion=confusion.tolist(), precision=precision.tolist(),
                 present_classes=int((np.bincount(truth, minlength=4) > 0).sum()))
    return value


def groups(data):
    age = data['rows'][:, 3]
    yield 'all', np.ones(len(age), bool)
    for name, mask in (('0', age == 0), ('1-31', (age >= 1) & (age <= 31)),
                       ('32-63', (age >= 32) & (age <= 63)), ('64-127', age >= 64)):
        yield 'age/' + name, mask
    for a in AGES:
        yield f'exact_age/{a}', age == a
    for field in ('symbol', 'period'):
        for value in np.unique(data[field]):
            yield f'{field}/{value}', data[field] == value


def grouped_scores(pred, truth, data):
    return {name: score(pred[mask], truth[mask]) for name, mask in groups(data)}


def paired_interval(a, b, truth, weeks, repeats=1000):
    """Paired BA delta, resampling whole calendar weeks across all contracts/periods."""
    unique, ids = np.unique(weeks, return_inverse=True)
    n = np.zeros((len(unique), 4)); delta = np.zeros_like(n)
    np.add.at(n, (ids, truth), 1)
    np.add.at(delta, (ids, truth), (a == truth).astype(float) - (b == truth))
    support = n.sum(0); present = support > 0
    result = dict(delta_ba=float((delta.sum(0)[present] / support[present]).mean()) if present.any() else None,
                  weeks=len(unique), low=None, high=None, valid_draws=0,
                  scope='Paired calendar-week block bootstrap; descriptive research uncertainty, not a fresh holdout or IID bars.')
    if len(unique) < 5:
        return result
    rng = np.random.default_rng(42); draws = []
    for _ in range(repeats):
        chosen = rng.integers(len(unique), size=len(unique))
        total = n[chosen].sum(0)
        if (total[present] == 0).any():
            continue
        draws.append(float((delta[chosen].sum(0)[present] / total[present]).mean()))
    if draws:
        result.update(low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)), valid_draws=len(draws))
    return result


def evaluate(data, out):
    train, val, test = (data[s] for s in SPLITS)
    report = dict(schema=SCHEMA, encoder_training=False, learned_readout='Class-balanced linear ridge',
        scope='Past/current causal rule-state recovery, not future prediction or trading profitability. Existing research test period reused.',
        label_names=list(STATE_NAMES), scales=list(SCALES), sampled_ages=list(AGES), variants={}, comparisons={},
        refresh_scope='Fresh long re-tiles the same number of complete128-bar blocks ending at current bar. Held long stays on the absolute128-bar grid. This changes refresh/window alignment, not weights.',
        coverage={s: dict(samples=len(d['rows']), source_endpoints=len(np.unique(d['rows'][:, [0, 2]], axis=0)),
                         per_age={str(a): int((d['rows'][:, 3] == a).sum()) for a in AGES}) for s, d in data.items()})
    preds = {}; weights_dir = out / 'readouts'; weights_dir.mkdir(exist_ok=True)
    for variant in VARIANTS:
        progress(f'Fitting frozen readout {variant} (3 scales; validation-only selection)')
        x = {s: design(d, variant) for s, d in data.items()}
        pred = np.zeros_like(test['labels']); item = {}
        for k, scale in enumerate(SCALES):
            fit = fit_ridge(x['train'], train['labels'][:, k], x['val'], val['labels'][:, k])
            pred[:, k] = predict(fit, x['test'])
            write_npz(weights_dir / f'{variant}_scale{k}.npz', mean=fit['mean'], scale=fit['scale'], coef=fit['coef'])
            item[str(scale)] = dict(alpha=fit['alpha'], validation_ba=fit['validation_ba'], candidates=fit['candidates'],
                                   dimensions=x['train'].shape[1], test=grouped_scores(pred[:, k], test['labels'][:, k], test))
        preds[variant] = pred; report['variants'][variant] = item
        atomic_json(report, out / 'state_metrics.json')
    # Persistence of the exact observable rule at the old endpoint is a cheap,
    # strong comparator. It is not supplied to any learned representation probe.
    preds['rule_hold'] = test['previous_labels']
    majority = np.stack([np.full(len(test['rows']), np.bincount(train['labels'][:, k], minlength=4).argmax())
                         for k in range(3)], axis=1)
    preds['train_majority'] = majority
    for name in ('rule_hold', 'train_majority'):
        report['variants'][name] = {str(scale): dict(test=grouped_scores(preds[name][:, k], test['labels'][:, k], test))
                                    for k, scale in enumerate(SCALES)}
    for a, b in (('dual_held', 'price_ema'), ('price_dual_held', 'price_ema'), ('dual_held', 'short'),
                 ('long_fresh', 'long_held'), ('dual_fresh', 'dual_held'),
                 ('price_dual_fresh', 'price_dual_held'), ('dual_held', 'rule_hold')):
        report['comparisons'][a + '_minus_' + b] = {str(scale): {
            name: paired_interval(preds[a][mask, k], preds[b][mask, k], test['labels'][mask, k], test['week'][mask])
            for name, mask in groups(test) if name == 'all' or name.startswith('age/')}
            for k, scale in enumerate(SCALES)}
    # Hold the fitted readout fixed as well: isolates representation refresh from refitting.
    for variant in ('long_held', 'dual_held', 'price_dual_held'):
        fresh_variant = variant.replace('held', 'fresh'); pred = np.zeros_like(test['labels'])
        x = design(test, fresh_variant)
        for k, scale in enumerate(SCALES):
            with np.load(weights_dir / f'{variant}_scale{k}.npz', allow_pickle=False) as f:
                pred[:, k] = predict({key: f[key] for key in f.files}, x)
        name = variant + '_refresh_same_head'; preds[name] = pred
        report['variants'][name] = {str(scale): dict(test=grouped_scores(pred[:, k], test['labels'][:, k], test))
                                    for k, scale in enumerate(SCALES)}
        report['comparisons'][name + '_minus_' + variant] = {str(scale): {
            group: paired_interval(pred[mask, k], preds[variant][mask, k], test['labels'][mask, k], test['week'][mask])
            for group, mask in groups(test) if group == 'all' or group.startswith('age/')}
            for k, scale in enumerate(SCALES)}
    # Small, reviewable raw predictions, excluding high-dimensional cached vectors.
    with (out / 'test_predictions.jsonl').open('w') as f:
        for j, row in enumerate(test['rows']):
            record = dict(series=int(row[0]), row=int(row[1]), long_row=int(row[2]), age=int(row[3]), blocks=int(row[4]),
                          symbol=str(test['symbol'][j]), period=int(test['period'][j]), week=str(test['week'][j]),
                          truth=test['labels'][j].tolist(), predictions={n: p[j].tolist() for n, p in preds.items()})
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    atomic_json(report, out / 'state_metrics.json')
    lines = ['# 冻结状态读出与长状态刷新对照', '',
             '标签是截至当前的规则结构状态；测试期曾被研发使用，不代表新样本预测或交易收益。',
             '采样年龄0/16/48/96/127；非全量逐bar。新鲜长状态使用截至当前、重新对齐的完整历史块。', '',
             '| 表示 | 短尺度 BA | 中尺度 BA | 长尺度 BA |', '|---|---:|---:|---:|']
    for name, value in report['variants'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{value[str(s)]["test"]["all"]["ba"]:.2%}' for s in SCALES) + ' |')
    lines += ['', '中尺度分组（BA；各组标签分布见state_metrics.json，不将组间差异直接解释为刷新因果效应）：', '',
              '| 表示 | age=0 | 1–31 | 32–63 | 64–127 |', '|---|---:|---:|---:|---:|']
    for name, value in report['variants'].items():
        vals = [value[str(SCALES[1])]['test']['age/' + age]['ba'] for age in ('0', '1-31', '32-63', '64-127')]
        lines.append('| ' + name + ' | ' + ' | '.join('N/A' if v is None else f'{v:.2%}' for v in vals) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True); p.add_argument('--fusion-run', required=True)
    p.add_argument('--long-run', required=True); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=128); p.add_argument('--stream-batch', type=int, default=128)
    a = p.parse_args()
    if min(a.batch, a.stream_batch) < 1 or not torch.cuda.is_available():
        raise ValueError('CUDA and positive extraction batches required; no local training fallback')
    torch.set_num_threads(4)
    started = time.perf_counter(); out = Path(a.out); bundle = Path(a.bundle); fusion = Path(a.fusion_run)
    old_meta = json.loads((fusion / 'manifest.json').read_text())['sources']
    bundle_meta = torch.load(bundle, map_location='cpu', weights_only=True)['metadata']
    if any(bundle_meta['sources'][k] != v for k, v in old_meta.items()):
        raise ValueError('Bundle and fusion encoders differ')
    if sha256(Path(a.long_run) / 'manifest.json') != old_meta['long_manifest']:
        raise ValueError('Long data manifest differs')
    old = {}; hashes = {}
    for split in SPLITS:
        old[split], hashes[split] = load_cache(fusion, split, old_meta)
    if hashes['train'] != bundle_meta['sources']['train_cache']:
        raise ValueError('Bundle normalization used a different training cache')
    meta = dict(schema=SCHEMA, bundle=str(bundle.resolve()), bundle_sha256=sha256(bundle),
                fusion=str(fusion.resolve()), source_hashes=hashes, root=str(Path(a.root).resolve()),
                ages=list(AGES), alphas=list(ALPHAS), variants=list(VARIANTS), batch=a.batch, stream_batch=a.stream_batch,
                rule_scales=list(SCALES), encoder_training=False,
                selection='Train-only normalization/class weights; alpha selected on validation BA separately for each variant and scale.')
    if (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text()) != meta:
            raise ValueError('Settings changed; choose a new run directory')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Nonempty run directory lacks manifest')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('completion.json', 'summary.md', 'state_metrics.json'):
        (out / name).unlink(missing_ok=True)
    atomic_json(meta, out / 'manifest.json')
    progress('Loading unchanged data/encoders; only linear readouts will be fitted')
    ref, series, encoded, bases = load_data(a.root, a.long_run)
    atomic_json(dict(boundaries=ref['boundaries'], series=[dict(index=i, key=s.key) for i, s in enumerate(series)],
                     label_definition='structure.annotate state at current row; confirmed pivots only, never backfilled'), out / 'data_inventory.json')
    engine = load_bundle(bundle, 'readout', 60, 'cuda')
    data = {}
    for split, base in zip(SPLITS, bases):
        progress(f'Preparing {split}: held cache + chronological short state + rolling long control')
        data[split] = prepare_split(engine, base, ref['boundaries'], split, old[split], out, meta, a.batch, a.stream_batch, 'cuda')
    del engine, bases, series, encoded, old
    torch.cuda.empty_cache()
    evaluate(data, out)
    if sha256(bundle) != meta['bundle_sha256']:
        raise ValueError('Frozen inference bundle changed during experiment')
    atomic_json(dict(status='complete', encoder_training=False, seconds=time.perf_counter()-started), out / 'completion.json')
    progress(f'State readout experiment complete: {out}')


if __name__ == '__main__':
    main()
