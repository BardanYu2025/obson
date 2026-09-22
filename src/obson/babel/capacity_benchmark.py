"""Controlled half/full conditioning capacity; preserve the completed architecture run."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import architecture as ar, architecture_benchmark as ab, capacity as ca
from .ae_extend import atomic_json, rng_state, restore_rng
from .ar_reconstruction import run_jobs
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-native-capacity-v1'
METRICS = ('primary', 'path', 'changes', 'body', 'activity', 'close_mae_bps')
GOAL = dict(stage='Retain ordered observed price/activity history in a causal state; isolate rank-limited decoder conditioning.',
            context='Same reset128-row windows and original train-only normalization.',
            automatic_promotion=False, independent_holdout=False,
            exclusions='Not forecasting, long-history streaming replacement, or proof of architecture-wide superiority.')


def code_identity():
    return ab.code_identity() | {Path(ca.__file__).name: sha256(ca.__file__), Path(__file__).name: sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256'] != code_identity():
        raise ValueError('Capacity code changed; use a new run')


def source_identity(source):
    """Read-only verification, including original validation-only LR choices and weights."""
    meta = read_json(source/'manifest.json')
    if meta['schema'] != ab.SCHEMA:
        raise ValueError('Expected completed ordered architecture source')
    ab.verify_code(meta)
    done = read_json(source/'completion.json')
    if done['status'] != 'complete' or not done['source_unchanged']:
        raise ValueError('Source architecture run is incomplete')
    ab.sc.verify_worker_files(source, done['files'])
    index = read_json(source/'cache/index.json')
    if index['manifest'] != meta:
        raise ValueError('Source cache metadata mismatch')
    verify_files(source/'cache', index['files'])
    baseline = read_json(source/'baseline_lock.json')
    if baseline != dict(manifest=meta, sha256=sha256(source/'pca.npz'), cache_sha256=sha256(source/'cache/index.json')):
        raise ValueError('Original PCA identity changed')
    lock = read_json(source/'selection_lock.json')
    summaries, weights = {}, {}
    for job in meta['experiments']:
        path = source/job['name']; row = read_json(path/'completion.json')
        if row['status'] != 'complete' or row['metadata'] != dict(manifest=meta, job=job):
            raise ValueError('Source worker identity mismatch')
        verify_files(path, row['files'])
        summaries[job['name']] = read_json(path/'training_summary.json')
        history = read_json(path/'history.json')
        if [r['epoch'] for r in history] != list(range(1, meta['epochs']+1)):
            raise ValueError('Source worker budget/history incomplete')
        weights[job['name']] = {kind: sha256(path/f'{kind}.pt') for kind in ('best', 'last')}
    chosen = {}
    for mode in ar.ENCODERS:
        for seed in meta['seeds']:
            names = [j['name'] for j in meta['experiments'] if j['architecture']==mode and j['seed']==seed]
            chosen[f'{mode}_s{seed}'] = min(names, key=lambda n: summaries[n]['validation']['primary'])
    if lock != dict(manifest=meta, chosen=chosen, all_trials=summaries, weights=weights):
        raise ValueError('Original selection lock does not reproduce')
    files = dict(done['files'])
    for name in ('completion.json', 'pca.npz'):
        files[name] = sha256(source/name)
    for job in meta['experiments']:
        files[job['name']+'/completion.json'] = sha256(source/job['name']/'completion.json')
    return dict(files=files, cache_files=index['files'], source_manifest=meta)


def make_manifest(source, identity, epochs=200, batch=128, micro=64):
    if min(epochs, batch, micro) < 1 or micro > batch or batch % micro:
        raise ValueError('Positive epochs; micro must divide batch')
    width = ca.CONFIG['latent']
    experiments = [dict(name=f'{mode}_r{rank}_s{seed}', architecture=mode, rank=rank, seed=seed, lr=3e-4)
                   for mode in ca.ENCODERS for seed in (42, 43) for rank in (width//2, width)]
    return json.loads(json.dumps(dict(schema=SCHEMA, source=str(source.resolve()), source_identity=identity,
        config=ca.CONFIG, epochs=epochs, batch=batch, micro=micro, seeds=[42, 43], experiments=experiments,
        ranks=[width//2, width], precision='fp32_tf32_disabled', code_sha256=code_identity(), goal=GOAL,
        objective=dict(weights=ar.WEIGHTS, delta_horizons=ar.HORIZONS, targets=ar.NAMES,
                       train='SmoothL1', selection='normalized_MSE', current_bar_scored=False),
        selection='Each predeclared cell selects epoch on original validation only. All cells and fixed last epochs reported.',
        learning_rate='Fixed3e-4, chosen by all six original validation LR comparisons; no architecture-specific tuning claim.',
        initialization='From scratch. Same encoder/decoder initial parameters across ranks within seed; same decoder across architectures within seed.',
        constraint='Fixed orthogonal rotation, exact zeroing of excluded coordinates, inverse rotation, scaled sqrt(width/rank); identical parameter count/matmul shapes, different functional capacity.',
        budget='12 trials, equal windows/updates/epochs. Approximate encoder parameter match; compute measured, convergence not assumed.',
        old_comparison='Old and new differ in encoder depth/normalization/decoder. Only within-new rank pairs isolate the conditioning constraint.')))


def verify_cache(meta):
    source = Path(meta['source'])
    index = read_json(source/'cache/index.json')
    identity = meta['source_identity']
    if sha256(source/'cache/index.json') != identity['files']['cache/index.json'] or index['files'] != identity['cache_files']:
        raise ValueError('Pinned source cache index changed')
    verify_files(source/'cache', index['files'])


def source_pca(meta):
    path = Path(meta['source'])/'pca.npz'
    if sha256(path) != meta['source_identity']['files']['pca.npz']:
        raise ValueError('Pinned PCA changed')
    with np.load(path, allow_pickle=False) as f:
        pca = {k: f[k] for k in f.files}
    if len(pca['components']) != meta['config']['latent']:
        raise ValueError('Source PCA width differs from declared full capacity')
    return pca


def prefix_pca(pca, width):
    return dict(mean=pca['mean'], components=pca['components'][:width], singular=pca['singular'][:width])


def audit(meta, out):
    verify_code(meta); verify_cache(meta)
    source = Path(meta['source']); stats = read_json(source/'cache/statistics.json'); pca = source_pca(meta)
    width = meta['config']['latent']
    original_config = meta['source_identity']['source_manifest']['config']
    report = dict(schema=SCHEMA, goal=GOAL, original_structural_audit=dict(
        gru_endpoint=original_config['gru_width'], attention_endpoint=original_config['attention_width'],
        multiscale_endpoint=original_config['conv_width'], nominal_output=original_config['latent'],
        decoder_pre_position_width=original_config['decoder_width'], pca_coordinates=original_config['latent'],
        interpretation='Original neural conditioning has an upper bound of256; nominal512 PCA was not capacity matched. This does not imply a GRU512 state contains only256 dimensions.'),
        new_structural_audit=dict(native_endpoint=width, ranks=meta['ranks'], memory_tokens=2,
            token_width=meta['config']['decoder_width'], terminal_state_normalization=False,
            claim='No structural width below declared rank before memory expansion; useful learned dimension is not guaranteed.'),
        pca='Reuse first256/512 components of the same train-fitted basis; no refit or parameter selection on research data.', datasets={})
    old_report = read_json(source/'architecture_metrics.json')
    for split in ab.SPLITS:
        data = ab.load_arrays(source, split); scores, errors = {}, {}
        for rank in meta['ranks']:
            name = f'pca{rank}'
            scores[name], errors[name] = ab.measure(ab.pca_predict(prefix_pca(pca, rank), data), data, stats)
        contrasts = {}
        if split in ('test', 'cross_research'):
            inv = read_json(source/f'cache/{split}_inventory.json')
            old = read_json(source/f'{split}_per_window_errors.json')
            full = f'pca{width}'
            for k, values in errors[full].items():
                if not np.allclose(values, old[full][k], atol=1e-6, rtol=2e-5):
                    raise ValueError(f'Original PCA per-window replay failed: {split}/{k}')
                if not np.isclose(values.mean(), old_report['datasets'][split]['variants'][full]['metrics'][k], atol=1e-6, rtol=2e-5):
                    raise ValueError('Original PCA aggregate replay failed')
            half = f'pca{width//2}'
            for name, rows in old.items():
                if name.startswith(('gru_', 'attention_', 'multiscale_', 'pca')):
                    contrasts[name+'_minus_'+half] = {k: ab.pair_groups(np.asarray(rows[k]), errors[half][k], inv) for k in METRICS}
            atomic_json({name: {k: v.tolist() for k, v in rows.items()} for name, rows in errors.items()}, out/f'{split}_pca_errors.json')
        report['datasets'][split] = dict(variants=scores, old_minus_pca_half=contrasts)
    atomic_json(report, out/'bottleneck_audit.json')
    atomic_json(stats, out/'statistics.json')
    progress('Read-only source audit and PCA half/full replay passed; fixed neural matrix unchanged')
    return report


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
    verify_cache(meta)
    source = Path(meta['source'])
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = ca.CapacityModel(job['architecture'], meta['config'], job['seed'], job['rank']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=job['lr'], weight_decay=.01)
    stats = read_json(source/'cache/statistics.json')
    data = {s: ab.load_arrays(source, s) for s in ('train', 'val')}
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata or state['epoch'] > meta['epochs'] or [r['epoch'] for r in state['history']] != list(range(1, state['epoch']+1)):
            raise ValueError('Resume identity/history mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        replay, _ = ab.run_epoch(model, data['val'], stats, meta['batch'], meta['micro'], device)
        expected = state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k], expected[k], atol=1e-6, rtol=2e-5) for k in replay):
            raise ValueError('Resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'], max_abs=max(abs(replay[k]-expected[k]) for k in replay)), path/'resume_validation.json')
    else:
        initial, _ = ab.run_epoch(model, data['val'], stats, meta['batch'], meta['micro'], device)
        state = dict(metadata=metadata, epoch=0, history=[], initial_validation=initial,
                     best_epoch=0, best_validation=initial, best_model=ab.cpu_state(model))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1, meta['epochs']+1):
        started = time.monotonic()
        lr = ar.learning_rate(epoch, meta['epochs'], job['lr'])
        for group in opt.param_groups:
            group['lr'] = lr
        train, updates = ab.run_epoch(model, data['train'], stats, meta['batch'], meta['micro'], device, opt,
                                   order_seed=job['seed']+epoch*1009)
        val, _ = ab.run_epoch(model, data['val'], stats, meta['batch'], meta['micro'], device)
        improved = val['primary'] < state['best_validation']['primary']
        if improved:
            state.update(best_epoch=epoch, best_validation=val, best_model=ab.cpu_state(model))
        elapsed = time.monotonic()-started
        state['history'].append(dict(epoch=epoch, train=train, validation=val, lr=lr, optimizer_steps=updates,
            windows=len(data['train']['x']), seconds=elapsed, remaining_minutes=elapsed*(meta['epochs']-epoch)/60))
        state.update(epoch=epoch, model=ab.cpu_state(model), optimizer=opt.state_dict(), rng=rng_state())
        ab.publish(state, path, write_best=improved)
        progress(f'{name} epoch={epoch}/{meta["epochs"]} val={val["primary"]:.5f} best={state["best_epoch"]} seconds={elapsed:.1f}')
    # Repair derivatives if interrupted after authoritative last.pt write.
    ab.publish(state, path)
    parameters = dict(encoder=sum(p.numel() for p in model.encoder.parameters()), decoder=sum(p.numel() for p in model.decoder.parameters()))
    atomic_json(dict(parameters=parameters, peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
                     selected_epoch=state['best_epoch'], trained_selection=state['best_epoch']>0,
                     selected_at_budget_end=state['best_epoch']==meta['epochs'], validation=state['best_validation'],
                     training_seconds=sum(r['seconds'] for r in state['history']), optimizer_steps=sum(r['optimizer_steps'] for r in state['history'])), path/'training_summary.json')
    files = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json', 'progress.json', 'run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete', metadata=metadata, files=files), path/'completion.json')



@torch.no_grad()
def preflight(meta, out, device='cuda'):
    results = {}
    for mode in ca.ENCODERS:
        for rank in meta['ranks']:
            model = ca.CapacityModel(mode, meta['config'], 999, rank).to(device)
            x = torch.randn(meta['micro'], 128, 28, device=device)
            if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            with torch.enable_grad():
                p = model(x)
                ar.error_rows(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool),
                              dict(y_scale=[1.]*7, delta_scale=[1.]*4), True)['primary'].mean().backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite preflight gradient')
            model.eval()
            original = model.encoder(x[:2]); changed = x[:2].clone(); changed[:, 65:] += 5
            delta = float((original[:, :65]-model.encoder(changed)[:, :65]).abs().max())
            if delta > 2e-5: raise ValueError(f'Prefix causality failed: {mode}')
            conditioning = model.decoder.conditioning_audit()
            if conditioning['numerical_rank'] != rank:
                raise ValueError('Initial aggregate memory rank differs from declared capacity')
            results[f'{mode}_r{rank}'] = dict(shape=list(p.shape), causal_prefix_maxdiff=delta,
                conditioning=conditioning, encoder_parameters=sum(v.numel() for v in model.encoder.parameters()),
                decoder_parameters=sum(v.numel() for v in model.decoder.parameters()),
                receptive_field=getattr(model.encoder, 'receptive_field', None),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
            del model, x, p, original, changed
            if str(device).startswith('cuda'): torch.cuda.empty_cache()
    atomic_json(dict(results=results, optimizer_updates=0,
                     scope='Synthetic single-worker forward/backward; optimizer/concurrent workers add memory.'), out/'preflight.json')


def lock_selection(meta, out):
    rows, weights = {}, {}
    for job in meta['experiments']:
        path = out/job['name']; done = read_json(path/'completion.json')
        if done['status'] != 'complete' or done['metadata'] != dict(manifest=meta, job=job):
            raise ValueError('Incomplete capacity trial')
        verify_files(path, done['files'])
        last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
        if (last['metadata'] != done['metadata'] or best['metadata'] != done['metadata'] or
            last['epoch'] != meta['epochs'] or [r['epoch'] for r in last['history']] != list(range(1, meta['epochs']+1))):
            raise ValueError('Budget/checkpoint metadata mismatch')
        candidates = [(0, last['initial_validation'])] + [(r['epoch'], r['validation']) for r in last['history']]
        epoch, val = min(candidates, key=lambda row: row[1]['primary'])
        if best['epoch'] != epoch or last['best_epoch'] != epoch or best['validation'] != val:
            raise ValueError('Best epoch is not the first validation minimum')
        summary = read_json(path/'training_summary.json')
        if summary['selected_epoch'] != epoch or summary['validation'] != val:
            raise ValueError('Selection summary mismatch')
        rows[job['name']] = summary
        weights[job['name']] = {kind: sha256(path/f'{kind}.pt') for kind in ('best', 'last')}
        del last, best
    lock = dict(manifest=meta, all_trials=rows, weights=weights)
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json') != lock:
        raise ValueError('Selection changed after research evaluation')
    atomic_json(lock, out/'selection_lock.json')
    return lock


def save_examples(out, cards, width):
    """Fixed examples, explicitly labelled both ranks for all three families."""
    import html
    palette = ['#df4d3e', '#287ad5', '#1b985a']
    colors = {'truth': '#222', f'pca{width//2}': '#bc8c20', f'pca{width}': '#9040b0'}
    for mode, color in zip(ca.ENCODERS, palette):
        for rank in (width//2, width): colors[f'{mode}_r{rank}_s42'] = color
    lines = ['<!doctype html><meta charset="utf-8"><title>Capacity reconstruction</title>',
        '<style>body{font:14px sans-serif;margin:24px}svg{border:1px solid #ddd;max-width:100%}</style>',
        '<h1>Observed history reconstruction, not forecasts</h1>',
        '<p>Fixed evenly spaced examples; seed42 displayed, both seeds in JSON. Dashed = constrained neural rank; solid = full rank. Toggles affect display only.</p>']
    for name, color in colors.items():
        lines.append(f'<label style="color:{color}"><input type="checkbox" checked onchange="document.querySelectorAll(\'[data-series={name}]\').forEach(p=>p.style.display=this.checked?\'\':\'none\')">{name}</label> ')
    for row in cards:
        lines.append('<h3>'+html.escape(row['dataset']+' '+row['key']+' '+row['end'])+'</h3>')
        for channel, label in [(0, 'Close, log-percent from pre-window anchor'), (2, 'Volume relative to prior EMA'), (4, 'Observed OI change, asinh')]:
            valid = np.array(row['mask'])[:127, channel]
            values = [np.array(v)[:127, channel][valid] for k, v in row['series'].items() if k in colors]
            flat = np.concatenate(values); lo = float(flat.min()) if len(flat) else 0.; hi = float(flat.max()) if len(flat) else 1.
            span = max(hi-lo, 1e-5)
            lines.append(f'<p>{label} [{lo:.3g}, {hi:.3g}]</p><svg viewBox="0 0 900 180" width="900" height="180">')
            for name, color in colors.items():
                if name not in row['series']: continue
                v = np.array(row['series'][name])[:127, channel]; parts = []
                for i in range(len(v)):
                    if valid[i]: parts.append(('M' if i==0 or not valid[i-1] else 'L')+f'{10+i*880/126:.2f},{170-(v[i]-lo)/span*160:.2f}')
                dash = '5 3' if f'_r{width//2}_' in name else 'none'
                lines.append(f'<path data-series="{name}" d="{" ".join(parts)}" fill="none" stroke="{color}" stroke-dasharray="{dash}" stroke-width="1.2"/>')
            lines.append('</svg>')
    (out/'examples.html').write_text('\n'.join(lines))


def evaluate(meta, out, device='cuda'):
    verify_code(meta); verify_cache(meta)
    lock = lock_selection(meta, out)  # All cells complete and selected before neural research evaluation.
    source = Path(meta['source']); stats = read_json(source/'cache/statistics.json')
    val = ab.load_arrays(source, 'val'); pca = source_pca(meta); width = meta['config']['latent']
    report = dict(schema=SCHEMA, goal=GOAL, trials=lock['all_trials'], datasets={},
        limitations='Research sets already opened. Same stored parameter count does not imply same functional capacity. Original models are context only; new rank pairs isolate the conditioning constraint.')
    cards = []
    for split in ('test', 'cross_research'):
        data = ab.load_arrays(source, split); inv = read_json(source/f'cache/{split}_inventory.json')
        ids = list(np.linspace(0, len(data['y'])-1, min(6, len(data['y'])), dtype=int))
        examples = {i: dict(dataset=split, key=inv[i]['key'], end=inv[i]['end'], mask=data['mask'][i].tolist(),
            series={'truth': (data['y'][i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()}) for i in ids}
        scores, errors = {}, {}
        def record(name, pred):
            scores[name], errors[name] = ab.measure(pred, data, stats)
            for i in ids:
                examples[i]['series'][name] = (pred[i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()
        for name, pred in ab.baseline_predictions(data).items(): record(name, pred)
        for rank in meta['ranks']: record(f'pca{rank}', ab.pca_predict(prefix_pca(pca, rank), data))
        last_scores = {}
        for job in meta['experiments']:
            path = out/job['name']
            model = ca.CapacityModel(job['architecture'], meta['config'], job['seed'], job['rank']).to(device)
            ck = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(ck['model'])
            replay, _ = ab.run_epoch(model, val, stats, meta['batch'], meta['micro'], device)
            if any(not np.isclose(replay[k], ck['validation'][k], atol=1e-6, rtol=2e-5) for k in replay):
                raise ValueError('Selected validation did not reproduce')
            pred, _ = ab.predictions(model, data['x'], meta['micro'], device)
            record(job['name'], pred)
            scores[job['name']]['conditioning'] = model.decoder.conditioning_audit()
            last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(last['model'])
            pred, _ = ab.predictions(model, data['x'], meta['micro'], device)
            last_scores[job['name']] = ab.measure(pred, data, stats)[0]
            del model, ck, last, pred
            progress(f'Evaluated {split} {job["name"]}: selected and fixed last')
        pairs = []
        for seed in meta['seeds']:
            for mode in ca.ENCODERS:
                full, half = f'{mode}_r{width}_s{seed}', f'{mode}_r{width//2}_s{seed}'
                pairs.append((full, half))
                pairs.extend([(full, f'pca{width}'), (half, f'pca{width//2}'), (full, 'coarse112')])
            pairs.extend([(f'attention_r{width}_s{seed}', f'gru_r{width}_s{seed}'),
                          (f'multiscale_r{width}_s{seed}', f'gru_r{width}_s{seed}'),
                          (f'attention_r{width}_s{seed}', f'multiscale_r{width}_s{seed}')])
        paired = {a+'_minus_'+b: {k: ab.pair_groups(errors[a][k], errors[b][k], inv) for k in METRICS} for a, b in pairs}
        report['datasets'][split] = dict(variants=scores, fixed_last=last_scores, paired=paired)
        atomic_json({name: {k: v.tolist() for k, v in rows.items()} for name, rows in errors.items()}, out/f'{split}_per_window_errors.json')
        cards.extend(examples.values())
    report['capacity_screen'] = {}
    for mode in ca.ENCODERS:
        checks, regressions = [], []
        for split, rows in report['datasets'].items():
            for seed in meta['seeds']:
                full, half = f'{mode}_r{width}_s{seed}', f'{mode}_r{width//2}_s{seed}'
                contrast = rows['paired'][full+'_minus_'+half]
                row = contrast['primary']['all']
                trained = all(lock['all_trials'][name]['trained_selection'] for name in (full, half))
                checks.append(trained and row['supported'] and row['high'] is not None and row['high'] < 0)
                for family in ('path', 'changes', 'body', 'activity'):
                    row = contrast[family]['all']
                    if row['supported'] and row['low'] is not None and row['low'] > 0:
                        regressions.append(dict(dataset=split, seed=seed, family=family, interval=row))
        report['capacity_screen'][mode] = dict(primary_improvement_all_seeds_and_datasets=all(checks), family_regressions=regressions,
            interpretation='Exploratory weekly paired intervals, no multiplicity correction; absence of detected regression is not equivalence.')
    atomic_json(report, out/'capacity_metrics.json'); atomic_json(cards, out/'examples.json'); save_examples(out, cards, width)
    lines = ['# Native capacity benchmark', '', 'Observed history, already-open research sets; no automatic promotion.', '']
    for split, rows in report['datasets'].items():
        lines += [f'## {split}', '', '| variant | primary | path | changes | body | activity | close bp |', '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in rows['variants'].items():
            lines.append(f'| {name} | '+' | '.join(f'{row["metrics"][k]:.5f}' for k in METRICS)+' |')
        lines.append('')
    (out/'summary.md').write_text('\n'.join(lines))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('all', 'audit', 'preflight', 'worker', 'evaluate'))
    p.add_argument('--source'); p.add_argument('--out', required=True); p.add_argument('--name')
    p.add_argument('--epochs', type=int, default=200); p.add_argument('--batch', type=int, default=128)
    p.add_argument('--micro', type=int, default=64); p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args(); out = Path(a.out).resolve(); ab.configure_runtime()
    if a.action != 'audit' and not torch.cuda.is_available():
        raise ValueError('Formal neural training/evaluation runs on AutoDL CUDA')
    if a.action == 'worker': worker(out, a.name); return
    if not a.source: p.error('--source required')
    if not 1 <= a.jobs <= 4: raise ValueError('jobs must be1..4')
    source = Path(a.source).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Separate sibling output required')
    progress('Verifying completed architecture artifacts and cache, read-only')
    identity = source_identity(source)
    meta = make_manifest(source, identity, a.epochs, a.batch, a.micro)
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json') != meta: raise ValueError('Configuration changed; use a new run')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty output without manifest')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out/'manifest.json')  # Freeze matrix before even the PCA research replay.
    (out/'completion.json').unlink(missing_ok=True)
    runtime = dict(torch=str(torch.__version__), numpy=np.__version__, jobs=a.jobs, action=a.action,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, started_unix=time.time())
    atomic_json(runtime, out/'runtime.json')
    with (out/'runtime_history.jsonl').open('a') as f: f.write(json.dumps(runtime)+'\n')
    audit(meta, out)
    if a.action == 'audit': return
    if a.action in ('all', 'preflight'):
        preflight(meta, out)
        if a.action == 'preflight': return
    if a.action == 'all': run_jobs(out, a.jobs, 'obson.babel.capacity_benchmark')
    evaluate(meta, out)
    if source_identity(source) != identity: raise ValueError('Source changed during experiment')
    files = {f.name: sha256(f) for f in out.iterdir() if f.is_file() and f.suffix in ('.json', '.jsonl', '.md', '.html') and f.name != 'completion.json'}
    atomic_json(dict(status='complete', trials=len(meta['experiments']), epochs=meta['epochs'], source_unchanged=True,
                     independent_holdout=False, automatic_promotion=False, files=files), out/'completion.json')
    progress('Capacity benchmark complete; source model remains unchanged')


if __name__ == '__main__':
    import fcntl
    import sys
    if '--out' not in sys.argv:
        main()
    else:
        root = Path(sys.argv[sys.argv.index('--out')+1]).resolve()
        name = sys.argv[sys.argv.index('--name')+1] if len(sys.argv)>1 and sys.argv[1]=='worker' and '--name' in sys.argv else 'controller'
        root.parent.mkdir(parents=True, exist_ok=True)
        with (root.parent/('.'+root.name+'.'+name+'.lock')).open('a') as lockfile:
            try: fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise SystemExit('Another process is already running this output/job')
            main()
