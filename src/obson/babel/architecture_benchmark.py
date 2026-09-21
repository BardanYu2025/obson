"""From-scratch matched-context architecture benchmark; formal training on AutoDL."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import architecture as ar, state_combination as sc, state_holdout as sh
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import run_jobs, paired_error_interval, derangement
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-ordered-architecture-v1'
SPLITS = ('train', 'val', 'test', 'cross_research')
GOAL = dict(stage='Compare causal architectures on ordered observed price/activity history, not descriptor imitation.',
    context='All encoders reset for each128-row window. Identical28 features include prior causal EMA context.',
    exclusions='Not a streaming/2048-history replacement, full OHLC reconstruction, future prediction or new holdout.',
    automatic_promotion=False, independent_holdout=False)


def code_identity():
    from . import ae_extend, ar_reconstruction, history_autoencoder
    return {Path(m.__file__).name: sha256(m.__file__) for m in
            (ar, sc, sh, sc.st, ae_extend, ar_reconstruction, history_autoencoder)} | {Path(__file__).name: sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256'] != code_identity():
        raise ValueError('Benchmark code changed; use a new run')


def source_identity(transfer, cross):
    tm, identity = sc.source_identity(transfer, cross)
    bank = Path(tm['encoder_run']) / 'cache'
    for split in ('train', 'val', 'test'):
        for suffix in ('x.npy', 'sequences.json'):
            path = bank/f'{split}_{suffix}'
            identity[str(path.resolve())] = sha256(path)
    return tm, identity


def make_manifest(transfer, cross, tm, identity, epochs=200, batch=128, micro=64):
    if min(epochs, batch, micro) < 1 or micro > batch or batch % micro:
        raise ValueError('Positive epochs; micro must divide effective batch')
    experiments = [dict(name=f'{mode}_s{seed}_lr{lr:g}', architecture=mode, seed=seed, lr=lr)
                   for mode in ar.ENCODERS for seed in (42, 43) for lr in (1e-4, 3e-4)]
    return dict(schema=SCHEMA, transfer=str(transfer.resolve()), cross_run=str(cross.resolve()),
        bank=str((Path(tm['encoder_run'])/'cache').resolve()), source_identity=identity,
        config=ar.CONFIG, epochs=epochs, batch=batch, micro=micro, precision='fp32_tf32_disabled',
        seeds=[42, 43], lrs=[1e-4, 3e-4], experiments=experiments,
        objective=dict(weights=ar.WEIGHTS, delta_horizons=ar.HORIZONS, train='SmoothL1', selection='normalized_MSE',
                       current_bar_scored=False, targets=ar.NAMES),
        goal=GOAL, code_sha256=code_identity(),
        selection='Per architecture/seed select epoch and LR only on original validation. Report all12 trials and fixed last epochs. No automatic winner.',
        initialization='All encoders random; common decoder weights exactly shared across architectures within seed. No old pretrained checkpoint.',
        budget='Equal windows, effective batches, optimizer updates and200 epochs; approximate parameter match, not equal compute or guaranteed convergence.')


def extract_windows(bank, specs, n, window=128):
    x = np.empty((n, window, 28), np.float32)
    seen = np.zeros(n, bool)
    for s in specs:
        offset, length = s['offset'], s['length']
        if offset < 0 or length < 0 or offset+length > len(bank):
            raise ValueError('Packed sequence outside bank')
        for end, idx in s['endpoints']:
            if not 0 <= idx < n or seen[idx] or end < window-1 or end >= length:
                raise ValueError('Endpoint crosses contract/partition, duplicated or invalid')
            x[idx] = bank[offset+end-window+1:offset+end+1]
            seen[idx] = True
    if not seen.all() or not np.isfinite(x).all():
        raise ValueError('Incomplete/nonfinite common-context input')
    return x


def prepare(meta, out):
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    if (cache/'index.json').exists():
        index = read_json(cache/'index.json')
        if index['manifest'] != meta:
            raise ValueError('Cache metadata mismatch')
        verify_files(cache, index['files'])
        return
    files, coverage = {}, {}
    def save(name, value):
        np.save(cache/name, value, allow_pickle=False)
        files[name] = sha256(cache/name)
    for split in SPLITS:
        cross = split == 'cross_research'
        bank = Path(meta['cross_run'])/'cache' if cross else Path(meta['bank'])
        prefix = 'test' if cross else split
        counts = Path(meta['cross_run'])/'cache' if cross else Path(meta['transfer'])/'cache'
        n = len(np.load(counts/f'{prefix}_y.npy', mmap_mode='r'))
        progress(f'Common-context cache {split}: extracting {n} windows on CPU')
        x = extract_windows(np.load(bank/f'{prefix}_x.npy', mmap_mode='r'), read_json(bank/f'{prefix}_sequences.json'), n)
        y, mask = ar.ordered_targets(x)
        if split == 'train':
            stats = ar.fit_scales(x, y, mask)
            atomic_json(stats, cache/'statistics.json')
            files['statistics.json'] = sha256(cache/'statistics.json')
        x, y, mask = ar.normalize(x, y, mask, stats)
        for suffix, value in [('x', x), ('y', y), ('mask', mask)]:
            save(f'{split}_{suffix}.npy', value)
        coverage[split] = dict(windows=n, target_support=mask.sum((0, 1)).tolist(),
                               target_bars_per_window=127, input_bars=128, input_channels=28)
        if split in ('test', 'cross_research'):
            inv = read_json(counts/'test_inventory.json')
            if len(inv) != n:
                raise ValueError('Inventory size mismatch')
            for row in inv:
                row['month'] = row['end'][:7]
            atomic_json(inv, cache/f'{split}_inventory.json')
            files[f'{split}_inventory.json'] = sha256(cache/f'{split}_inventory.json')
    atomic_json(dict(coverage=coverage, goal=GOAL, target_definition='128 observed rows; final row not scored. Close path anchored before first row; five causal transformed activity quantities.',
                     split_policy='Original audited chronological partitions; windows never cross packed contract/partition bounds.',
                     inherited_context='Input EMA/prior-price features can summarize pre-window history equally for all candidates.'), out/'data_audit.json')
    atomic_json(dict(manifest=meta, files=files), cache/'index.json')


def load_arrays(out, split):
    return {k: np.load(out/f'cache/{split}_{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('x', 'y', 'mask')}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_epoch(model, data, stats, batch, micro, device, opt=None, order_seed=0):
    training = opt is not None
    model.train(training)
    n = len(data['x'])
    order = np.random.default_rng(order_seed).permutation(n) if training else np.arange(n)
    totals, updates = {}, 0
    with torch.set_grad_enabled(training):
        for left in range(0, n, batch):
            ids = order[left:left+batch]
            if training:
                opt.zero_grad(set_to_none=True)
            for start in range(0, len(ids), micro):
                use = ids[start:start+micro]
                b = {k: torch.tensor(np.asarray(v[use]), device=device) for k, v in data.items()}
                pred = model(b['x'])
                rows = ar.error_rows(pred, b['y'], b['mask'], stats)
                loss = ar.error_rows(pred, b['y'], b['mask'], stats, smooth=True)['primary']
                if not torch.isfinite(loss).all() or any(not torch.isfinite(v).all() for v in rows.values()):
                    raise ValueError('Nonfinite architecture objective')
                if training:
                    (loss.sum()/len(ids)).backward()  # Correct weight for a final partial microbatch.
                for k, v in dict(rows, smooth_objective=loss).items():
                    totals[k] = totals.get(k, 0.) + float(v.detach().sum())
            if training:
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                updates += 1
    return {k: v/n for k, v in totals.items()}, updates


def publish(state, path, write_best=True):
    atomic_save(state, path/'last.pt')
    if write_best or not (path/'best.pt').exists():
        atomic_save(dict(metadata=state['metadata'], epoch=state['best_epoch'], validation=state['best_validation'],
                         model=state['best_model']), path/'best.pt')
    atomic_json(state['history'], path/'history.json')
    row = state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'], epochs=state['metadata']['manifest']['epochs'], best_epoch=state['best_epoch'],
                     train=row.get('train', {}).get('primary'), validation=row.get('validation', {}).get('primary'),
                     seconds=row.get('seconds'), remaining_minutes=row.get('remaining_minutes')), path/'progress.json')


def worker(out, name, device='cuda'):
    meta = read_json(out/'manifest.json')
    verify_code(meta)
    job = next(j for j in meta['experiments'] if j['name'] == name)
    path = out/name
    path.mkdir(exist_ok=True)
    metadata = dict(manifest=meta, job=job)
    if (path/'completion.json').exists():
        done = read_json(path/'completion.json')
        if done['metadata'] != metadata or done['status'] != 'complete':
            raise ValueError('Completed worker identity mismatch')
        verify_files(path, done['files'])
        return
    index = read_json(out/'cache/index.json')
    if index['manifest'] != meta:
        raise ValueError('Cache metadata mismatch')
    verify_files(out/'cache', index['files'])
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = ar.OrderedModel(job['architecture'], meta['config'], job['seed']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=job['lr'], weight_decay=.01)
    stats = read_json(out/'cache/statistics.json')
    data = {s: load_arrays(out, s) for s in ('train', 'val')}
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata or state['epoch'] > meta['epochs'] or [r['epoch'] for r in state['history']] != list(range(1, state['epoch']+1)):
            raise ValueError('Resume identity/history mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        replay, _ = run_epoch(model, data['val'], stats, meta['batch'], meta['micro'], device)
        expected = state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k], expected[k], atol=1e-6, rtol=2e-5) for k in replay):
            raise ValueError('Resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'], max_abs=max(abs(replay[k]-expected[k]) for k in replay)), path/'resume_validation.json')
    else:
        initial, _ = run_epoch(model, data['val'], stats, meta['batch'], meta['micro'], device)
        state = dict(metadata=metadata, epoch=0, history=[], initial_validation=initial,
                     best_epoch=0, best_validation=initial, best_model=cpu_state(model))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1, meta['epochs']+1):
        started = time.monotonic()
        lr = ar.learning_rate(epoch, meta['epochs'], job['lr'])
        for group in opt.param_groups:
            group['lr'] = lr
        train, updates = run_epoch(model, data['train'], stats, meta['batch'], meta['micro'], device, opt,
                                   order_seed=job['seed']+epoch*1009)
        val, _ = run_epoch(model, data['val'], stats, meta['batch'], meta['micro'], device)
        improved = val['primary'] < state['best_validation']['primary']
        if improved:
            state.update(best_epoch=epoch, best_validation=val, best_model=cpu_state(model))
        elapsed = time.monotonic()-started
        state['history'].append(dict(epoch=epoch, train=train, validation=val, lr=lr, optimizer_steps=updates,
            windows=len(data['train']['x']), seconds=elapsed, remaining_minutes=elapsed*(meta['epochs']-epoch)/60))
        state.update(epoch=epoch, model=cpu_state(model), optimizer=opt.state_dict(), rng=rng_state())
        publish(state, path, write_best=improved)
        progress(f'{name} epoch={epoch}/{meta["epochs"]} val={val["primary"]:.5f} best={state["best_epoch"]} seconds={elapsed:.1f}')
    # Repair derivatives if interrupted after authoritative last.pt write.
    publish(state, path)
    parameters = dict(encoder=sum(p.numel() for p in model.encoder.parameters()), decoder=sum(p.numel() for p in model.decoder.parameters()))
    atomic_json(dict(parameters=parameters, peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                     selected_epoch=state['best_epoch'], trained_selection=state['best_epoch']>0,
                     selected_at_budget_end=state['best_epoch']==meta['epochs'], validation=state['best_validation'],
                     training_seconds=sum(r['seconds'] for r in state['history']), optimizer_steps=sum(r['optimizer_steps'] for r in state['history'])), path/'training_summary.json')
    files = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json', 'progress.json', 'run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete', metadata=metadata, files=files), path/'completion.json')


@torch.no_grad()
def preflight(meta, out, device='cuda'):
    """Synthetic forward/backward and prefix causality; no optimizer update."""
    results = {}
    for mode in ar.ENCODERS:
        model = ar.OrderedModel(mode, meta['config'], 999).to(device)
        x = torch.randn(meta['micro'], 128, 28, device=device)
        stats = dict(y_scale=[1.]*7, delta_scale=[1.]*4)
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            p = model(x)
            ar.error_rows(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool), stats, True)['primary'].mean().backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite preflight gradient')
        model.eval()
        original = model.encoder(x[:2])
        changed = x[:2].clone(); changed[:, 65:] += 5
        delta = float((original[:, :65]-model.encoder(changed)[:, :65]).abs().max())
        if delta > 2e-5:
            raise ValueError(f'Causal prefix check failed: {mode} {delta}')
        results[mode] = dict(shape=list(p.shape), causal_prefix_maxdiff=delta,
            encoder_parameters=sum(v.numel() for v in model.encoder.parameters()),
            decoder_parameters=sum(v.numel() for v in model.decoder.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
        del model, x, p, original, changed
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    atomic_json(dict(results=results, optimizer_updates=0, scope='Single worker synthetic memory; optimizer state/concurrent jobs add memory.'), out/'preflight.json')


@torch.no_grad()
def predictions(model, x, micro, device):
    model.eval()
    outputs, zs = [], []
    for left in range(0, len(x), micro):
        b = torch.tensor(np.asarray(x[left:left+micro]), device=device)
        z = model.encoder(b)[:, -1]
        outputs.append(model.decoder(z).cpu().numpy()); zs.append(z.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(zs)


def measure(pred, data, stats):
    p = torch.as_tensor(pred, dtype=torch.float64)
    y = torch.tensor(np.asarray(data['y']), dtype=torch.float64)
    m = torch.tensor(np.asarray(data['mask']), dtype=torch.bool)
    values = {k: v.numpy() for k, v in ar.error_rows(p, y, m, stats).items()}
    yscale = np.array(stats['y_scale'])
    values['close_mae_bps'] = ((np.abs(pred[..., 0]-data['y'][..., 0])*data['mask'][..., 0]).sum(1) /
                                data['mask'][..., 0].sum(1))*yscale[0]*100
    report = dict(metrics={k: float(v.mean()) for k, v in values.items()}, targets=[], age_bands={}, price_changes={})
    for j, name in enumerate(ar.NAMES):
        ok = data['mask'][..., j]; actual = data['y'][..., j][ok].astype(float); output = pred[..., j][ok].astype(float)
        mse = float(np.mean((output-actual)**2)) if len(actual) else None
        variance = float(np.var(actual)) if len(actual) else 0.
        report['targets'].append(dict(name=name, support=int(ok.sum()), mse=mse, r2=1-mse/variance if variance>1e-12 else None))
    for label, start, end in [('oldest_64', 0, 64), ('middle_48', 64, 112), ('recent_15', 112, 127)]:
        valid = data['mask'][:, start:end]; e = (pred[:, start:end]-data['y'][:, start:end])**2
        report['age_bands'][label] = dict(path_mse=float(e[..., 0][valid[..., 0]].mean()),
            activity_mse=float(np.mean([e[..., j][valid[..., j]].mean() for j in range(2, 7) if valid[..., j].any()])))
    for h in ar.HORIZONS:
        valid = data['mask'][:, h:, 0] & data['mask'][:, :-h, 0]
        a = (data['y'][:, h:, 0]-data['y'][:, :-h, 0])[valid].astype(float)
        b = (pred[:, h:, 0]-pred[:, :-h, 0])[valid].astype(float)
        sa, sb = float(np.std(a)), float(np.std(b))
        report['price_changes'][str(h)] = dict(correlation=float(np.corrcoef(a, b)[0, 1]) if min(sa, sb)>1e-12 else None,
                                              std_ratio=sb/sa if sa>1e-12 else None)
    return report, values


def fit_pca(train, width):
    """Transparent observed-history compression, training-only basis; no target mask side channel."""
    flat = np.asarray(train['y'], dtype=np.float64).reshape(len(train['y']), -1)
    if width > min(flat.shape):
        raise ValueError('Insufficient training rows for declared PCA dimension')
    mean = flat.mean(0)
    _, singular, vh = np.linalg.svd(flat-mean, full_matrices=False)
    return dict(mean=mean.astype(np.float32), components=vh[:width].astype(np.float32), singular=singular[:width].astype(np.float32))


def pca_predict(model, data):
    x = np.asarray(data['y']).reshape(len(data['y']), -1)
    z = ((x-model['mean'])@model['components'].T).astype(np.float32)
    return (z@model['components']+model['mean']).reshape(data['y'].shape).astype(np.float32)


def baseline_predictions(data):
    # These use observed history, just as an encoder does; no future availability.
    y, mask = np.asarray(data['y']), np.asarray(data['mask'])
    mean = np.where(mask, y, 0).sum(1)/mask.sum(1).clip(1)
    #16 fixed chronological bins x7 quantities =112 floats, no per-sample mask sent to decoder.
    a = y.reshape(len(y), 16, 8, 7); m = mask.reshape(len(y), 16, 8, 7)
    coarse = np.where(m, a, 0).sum(2)/m.sum(2).clip(1)
    return {'train_mean': np.zeros_like(y), 'window_mean7': np.broadcast_to(mean[:, None], y.shape).copy(),
            'coarse112': np.repeat(coarse, 8, axis=1)}


def lock_selection(meta, out):
    rows, chosen = {}, {}
    for job in meta['experiments']:
        path = out/job['name']; done = read_json(path/'completion.json')
        if done['status'] != 'complete' or done['metadata'] != dict(manifest=meta, job=job):
            raise ValueError('Incomplete architecture trial')
        verify_files(path, done['files'])
        rows[job['name']] = read_json(path/'training_summary.json')
    for mode in ar.ENCODERS:
        for seed in meta['seeds']:
            candidates = [j['name'] for j in meta['experiments'] if j['architecture']==mode and j['seed']==seed]
            name = min(candidates, key=lambda n: rows[n]['validation']['primary'])
            chosen[f'{mode}_s{seed}'] = name
    lock = dict(manifest=meta, chosen=chosen, all_trials=rows,
                weights={j['name']: {kind: sha256(out/j['name']/f'{kind}.pt') for kind in ('best', 'last')} for j in meta['experiments']})
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json') != lock:
        raise ValueError('Selection changed after evaluation')
    atomic_json(lock, out/'selection_lock.json')
    return lock


def pair_groups(a, b, inventory):
    result = {}
    groups = {'all': np.arange(len(inventory))}
    for field in ('symbol', 'period', 'month'):
        for value in sorted({str(r[field]) for r in inventory}):
            groups[f'{field}/{value}'] = np.array([i for i, r in enumerate(inventory) if str(r[field])==value])
    for name, ids in groups.items():
        row = paired_error_interval(a[ids], b[ids], np.array([inventory[i]['week'] for i in ids]))
        row['delta'] = row.pop('delta_close_mae_bps')
        row.update(support=len(ids), supported=len(ids)>=50 and row['weeks']>=5,
                   interpretation='Candidate minus reference; negative favors candidate. Research data, uncorrected weekly interval.')
        result[name] = row
    return result


def save_examples(out, records):
    """Self-contained SVG: six preselected examples, all selected architectures and PCA."""
    import html
    lines = ['<!doctype html><meta charset="utf-8"><title>Ordered history reconstruction</title>',
             '<style>body{font:15px sans-serif;margin:24px}svg{border:1px solid #ddd}section{margin-bottom:30px}</style>',
             '<h1>Observed historical trajectories — not forecasts</h1>',
             '<p>Fixed evenly spaced examples. Black=truth; red=GRU42; blue=attention42; green=multiscale42; purple=PCA512. All seeds are in JSON/metrics.</p>']
    colors = {'truth':'#222', 'gru_s42':'#df4d3e', 'attention_s42':'#287ad5', 'multiscale_s42':'#1b985a', 'pca512':'#9040b0'}
    for row in records:
        lines.append('<section><h3>'+html.escape(row['dataset']+' '+row['key']+' '+row['end'])+'</h3>')
        for channel, label in [(0, 'Close, log-percent relative to pre-window anchor'), (2, 'Volume relative to prior EMA'), (4, 'Observed OI change (asinh)')]:
            vals = [np.array(v)[:127, channel] for k, v in row['series'].items() if k in colors]
            valid = np.array(row['mask'])[:127, channel]
            finite = np.concatenate([v[valid] for v in vals]); lo = float(finite.min()) if len(finite) else 0.; hi = float(finite.max()) if len(finite) else 1.
            span = max(hi-lo, 1e-5)
            lines.append('<p>'+label+f' [{lo:.3g}, {hi:.3g}]</p><svg viewBox="0 0 900 180" width="900" height="180">')
            for key, color in colors.items():
                if key not in row['series']: continue
                v = np.array(row['series'][key])[:127, channel]; parts = []
                for i in range(len(v)):
                    if valid[i]: parts.append(('M' if i==0 or not valid[i-1] else 'L')+f'{10+i*880/126:.2f},{170-(v[i]-lo)/span*160:.2f}')
                lines.append(f'<path d="{" ".join(parts)}" fill="none" stroke="{color}" stroke-width="1.3"/>')
            lines.append('</svg>')
        lines.append('</section>')
    (out/'examples.html').write_text('\n'.join(lines))


def evaluate(meta, out, device='cuda'):
    verify_code(meta)
    lock = lock_selection(meta, out)  # Must precede any research array read.
    stats = read_json(out/'cache/statistics.json')
    train, val = load_arrays(out, 'train'), load_arrays(out, 'val')
    pca_path = out/'pca.npz'
    if not pca_path.exists():
        progress('Fitting512-dimensional PCA on training observations only (CPU)')
        pca = fit_pca(train, meta['config']['latent'])
        np.savez(out/'pca.tmp.npz', **pca); (out/'pca.tmp.npz').replace(pca_path)
    with np.load(pca_path) as f: pca = {k: f[k] for k in f.files}
    pca_lock = dict(manifest=meta, sha256=sha256(pca_path), cache_sha256=sha256(out/'cache/index.json'))
    if (out/'baseline_lock.json').exists() and read_json(out/'baseline_lock.json') != pca_lock:
        raise ValueError('PCA source changed')
    atomic_json(pca_lock, out/'baseline_lock.json')
    report = dict(schema=SCHEMA, goal=GOAL, selection=lock['chosen'], trials=lock['all_trials'], datasets={},
        baseline='PCA uses512 sample coordinates and train-only shared basis; coarse uses112 coordinates. Inputs derived from same observed window. Missing values filled with train mean, no target-mask decoder side channel.',
        limits='Task differs from old64-bar full-price decoder. No direct old-checkpoint ranking, no architecture-wide superiority claim.')
    cards = []
    for split in ('test', 'cross_research'):
        data = load_arrays(out, split); inventory = read_json(out/f'cache/{split}_inventory.json')
        chosen_ids = list(np.linspace(0, len(data['y'])-1, min(6, len(data['y'])), dtype=int))
        examples = {i: dict(dataset=split, key=inventory[i]['key'], end=inventory[i]['end'], mask=data['mask'][i].tolist(),
            series={'truth': (data['y'][i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()}) for i in chosen_ids}
        scores, errors = {}, {}
        def record(name, pred):
            scores[name], errors[name] = measure(pred, data, stats)
            for i in chosen_ids:
                examples[i]['series'][name] = (pred[i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()
        for name, pred in baseline_predictions(data).items(): record(name, pred)
        record('pca512', pca_predict(pca, data))
        trials = {}
        for job in meta['experiments']:
            path = out/job['name']; model = ar.OrderedModel(job['architecture'], meta['config'], job['seed']).to(device)
            ck = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            if ck['metadata'] != dict(manifest=meta, job=job): raise ValueError('Best checkpoint mismatch')
            model.load_state_dict(ck['model'])
            replay, _ = run_epoch(model, val, stats, meta['batch'], meta['micro'], device)
            if any(not np.isclose(replay[k], ck['validation'][k], atol=1e-6, rtol=2e-5) for k in replay):
                raise ValueError('Selected validation did not reproduce')
            pred, z = predictions(model, data['x'], meta['micro'], device)
            trials[job['name']] = dict(best=measure(pred, data, stats)[0], selected_epoch=ck['epoch'])
            alias = f'{job["architecture"]}_s{job["seed"]}'
            if lock['chosen'][alias] == job['name']:
                record(alias, pred)
                shuffled_pred = []
                with torch.no_grad():
                    for left in range(0, len(z), meta['micro']):
                        permuted_z = z[derangement(len(z))[left:left+meta['micro']]]
                        shuffled_pred.append(model.decoder(torch.tensor(permuted_z, device=device)).cpu().numpy())
                swapped = np.concatenate(shuffled_pred)
                order_pred, order_z = predictions(model, ar.permute_old_blocks(data['x']), meta['micro'], device)
                scores[alias]['sensitivity'] = dict(shuffled_state=measure(swapped, data, stats)[0]['metrics'],
                    permuted_old_blocks=measure(order_pred, data, stats)[0]['metrics'],
                    state_rms_change=float(np.sqrt(np.mean((order_z-z)**2))),
                    interpretation='Synthetic corruption against unchanged targets; sensitivity alone is not correct order understanding. EMA channels not recomputed.')
            last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(last['model'])
            last_pred, _ = predictions(model, data['x'], meta['micro'], device)
            trials[job['name']]['last'] = measure(last_pred, data, stats)[0]
            del model, ck, last, pred, z, last_pred
            progress(f'Evaluated {split} {job["name"]}, best and fixed last')
        contrasts = {}
        for seed in meta['seeds']:
            for a, b in [(f'attention_s{seed}', f'gru_s{seed}'), (f'multiscale_s{seed}', f'gru_s{seed}'), (f'attention_s{seed}', f'multiscale_s{seed}')]+[(f'{mode}_s{seed}', base) for mode in ar.ENCODERS for base in ('pca512', 'coarse112')]:
                contrasts[a+'_minus_'+b] = {k: pair_groups(errors[a][k], errors[b][k], inventory) for k in ('primary', 'path', 'changes', 'body', 'activity', 'close_mae_bps')}
        report['datasets'][split] = dict(variants=scores, all_trials=trials, paired=contrasts)
        atomic_json({name: {k: v.tolist() for k, v in rows.items()} for name, rows in errors.items()}, out/f'{split}_per_window_errors.json')
        cards.extend(examples.values())
    report['exploratory_screen'] = {}
    for mode in ('attention', 'multiscale'):
        checks = []
        regressions = []
        for split, rows in report['datasets'].items():
            for seed in meta['seeds']:
                key = f'{mode}_s{seed}_minus_gru_s{seed}'
                primary = rows['paired'][key]['primary']['all']
                trained = all(lock['all_trials'][lock['chosen'][f'{k}_s{seed}']]['trained_selection'] for k in (mode, 'gru'))
                checks.append(trained and primary['supported'] and primary['high'] is not None and primary['high'] < 0)
                for family in ('path', 'changes', 'body', 'activity'):
                    row = rows['paired'][key][family]['all']
                    if row['supported'] and row['low'] is not None and row['low'] > 0:
                        regressions.append(dict(dataset=split, seed=seed, family=family, interval=row))
        report['exploratory_screen'][mode] = dict(primary_improvement_all_seeds_and_datasets=all(checks), family_regressions=regressions,
            interpretation='Exploratory common-context signal only. No regression detected does not establish equivalence/retention.')
    atomic_json(report, out/'architecture_metrics.json'); atomic_json(cards, out/'examples.json'); save_examples(out, cards)
    lines = ['# Ordered observed-history architecture benchmark', '', 'Lower MSE is better. Research data only; no automatic promotion.', '']
    for split, rows in report['datasets'].items():
        lines += [f'## {split}', '', '| representation | primary | path | changes | activity | close bp |', '|---|---:|---:|---:|---:|---:|']
        for name, row in rows['variants'].items():
            m = row['metrics']; lines.append(f'| {name} | '+ ' | '.join(f'{m[k]:.5f}' for k in ('primary', 'path', 'changes', 'activity', 'close_mae_bps'))+' |')
        lines.append('')
    (out/'summary.md').write_text('\n'.join(lines))
    return report


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('all', 'worker', 'evaluate', 'preflight'))
    p.add_argument('--out', required=True); p.add_argument('--transfer'); p.add_argument('--cross-run'); p.add_argument('--name')
    p.add_argument('--epochs', type=int, default=200); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--micro', type=int, default=64); p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args(); out = Path(a.out).resolve()
    if not torch.cuda.is_available(): raise ValueError('Formal neural training/evaluation runs on AutoDL CUDA')
    configure_runtime()
    if a.action == 'worker': worker(out, a.name); return
    if not a.transfer or not a.cross_run: p.error('--transfer and --cross-run required')
    if not 1 <= a.jobs <= 4: raise ValueError('jobs must be1..4')
    transfer, cross = Path(a.transfer).resolve(), Path(a.cross_run).resolve()
    if any(out==src or out in src.parents or src in out.parents for src in (transfer, cross)):
        raise ValueError('Separate sibling output required')
    tm, identity = source_identity(transfer, cross)
    meta = make_manifest(transfer, cross, tm, identity, a.epochs, a.batch, a.micro)
    # JSON roundtrip canonicalizes tuple-valued constants before equality/resume checks.
    meta = json.loads(json.dumps(meta))
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json') != meta: raise ValueError('Configuration changed; use a new run')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty output without manifest')
    out.mkdir(parents=True, exist_ok=True); atomic_json(meta, out/'manifest.json')
    if a.action != 'preflight': (out/'completion.json').unlink(missing_ok=True)
    runtime = dict(torch=str(torch.__version__), numpy=np.__version__, gpu=torch.cuda.get_device_name(),
                   jobs=a.jobs, precision=meta['precision'], started_unix=time.time(), action=a.action)
    atomic_json(runtime, out/'runtime.json')
    with (out/'runtime_history.jsonl').open('a') as history:
        history.write(json.dumps(runtime)+'\n')
    if a.action in ('all', 'preflight'):
        preflight(meta, out)
        if a.action == 'preflight': return
    prepare(meta, out)
    if a.action == 'all': run_jobs(out, a.jobs, 'obson.babel.architecture_benchmark')
    evaluate(meta, out)
    verify_files(out/'cache', read_json(out/'cache/index.json')['files'])
    if source_identity(transfer, cross)[1] != identity: raise ValueError('Upstream sources changed')
    files = {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl', '.md', '.html') and p.name != 'completion.json'}
    files['cache/index.json'] = sha256(out/'cache/index.json')
    atomic_json(dict(status='complete', trials=len(meta['experiments']), epochs=meta['epochs'], source_unchanged=True,
                     automatic_promotion=False, independent_holdout=False, files=files), out/'completion.json')
    progress('Architecture benchmark complete; reports compare common128-bar context only')


if __name__ == '__main__':
    # A process-scoped lock blocks accidental duplicate trainers for the same output.
    # Workers use separate lock paths; locks release automatically on crash.
    import fcntl
    import sys
    if '--out' not in sys.argv:
        main()
    else:
        root = Path(sys.argv[sys.argv.index('--out')+1]).resolve()
        lock_name = (sys.argv[sys.argv.index('--name')+1] if len(sys.argv)>1 and sys.argv[1]=='worker' and '--name' in sys.argv else 'controller')
        root.parent.mkdir(parents=True, exist_ok=True)
        with (root.parent/('.'+root.name+'.'+lock_name+'.lock')).open('a') as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('Another process is already running this output/job')
            main()
