"""Frozen multi-position readout; unseen-position interpolation and PCA capacity curve."""
import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import shared_readout as sr, prefix_readout as pr, prefix_readout_benchmark as pb
from . import architecture as ar, architecture_benchmark as ab, capacity_benchmark as cb
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-shared-readout-v1'
GOAL = 'Test a shared reader of frozen causal historical states before changing encoder capacity or supervision.'


def code_identity():
    return pb.code_identity() | {Path(sr.__file__).name: sha256(sr.__file__), Path(__file__).name: sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256'] != code_identity():
        raise ValueError('Shared readout implementation changed; use a new run')


def source_identity(root):
    """Read-only: never call the previous controller's selection publisher."""
    meta = read_json(root/'manifest.json'); done = read_json(root/'completion.json'); pb.verify_code(meta)
    if meta['schema'] != pb.SCHEMA or done['status'] != 'complete' or not done['source_unchanged'] or done['encoder_updates'] != 0:
        raise ValueError('Completed frozen prefix experiment required')
    ab.sc.verify_worker_files(root, done['files'])
    if pb.source_identity(Path(meta['source'])) != meta['identity']:
        raise ValueError('Upstream frozen encoder lineage changed')
    for split in ('train', 'val', 'test', 'cross_research'):
        pb.verify_cache(meta, root, split)
    lock = read_json(root/'selection_lock.json')
    if lock['manifest'] != meta:
        raise ValueError('Prefix source selection identity changed')
    files = dict(done['files']); files['completion.json'] = sha256(root/'completion.json')
    for job in meta['experiments']:
        path = root/job['name']; row = read_json(path/'completion.json')
        expected = dict(manifest=meta, job=job, cache_sha256=pb.cache_identity(root))
        if row['status'] != 'complete' or row['metadata'] != expected:
            raise ValueError('Incomplete source readout')
        verify_files(path, row['files'])
        summary = read_json(path/'training_summary.json'); ridge = read_json(path/'ridge.json')
        if lock['selections'][job['name']] != dict(mlp=summary, ridge=ridge['selected']):
            raise ValueError('Source readout selection changed')
        for name, digest in lock['weights'][job['name']].items():
            if sha256(path/name) != digest:
                raise ValueError('Source readout weight changed')
        files[f'{job["name"]}/completion.json'] = sha256(path/'completion.json')
    return dict(manifest=meta, files=files)


def make_manifest(source, identity, epochs=100, batch=256, hidden=256, ranks=sr.RANKS):
    if min(epochs, batch, hidden) < 1:
        raise ValueError('Positive budgets required')
    old = identity['manifest']
    width = old['identity']['manifest']['config']['latent']
    if tuple(sorted(set(ranks))) != tuple(ranks) or min(ranks) < 1 or width not in ranks:
        raise ValueError('Increasing unique PCA ranks including original control required')
    jobs = [dict(name=name, variant=name, seed=v['seed']+91000, epochs=epochs, lr=1e-3,
                 width=28 if name == 'current28' else old['identity']['manifest']['config']['latent'])
            for name, v in old['variants'].items()]
    return dict(schema=SCHEMA, source=str(source.resolve()), identity=identity, experiments=jobs,
        epochs=epochs, batch=batch, hidden=hidden, ranks=list(ranks), alphas=list(pr.ALPHAS),
        train_prefixes=list(sr.TRAIN_PREFIXES), held_prefixes=list(sr.HELD_PREFIXES),
        code_sha256=code_identity(), goal=GOAL,
        data='Same fixed4789 source train windows. Four training prefixes yield19156 rows, not independent windows; validation uses only those same four positions.',
        selection='Pooled train-only input statistics; inherited train-only target scales; pooled anchor validation selects alpha/epoch including0. Held positions never select shared heads.',
        controls='Reuse independent source heads at32/64/96/128. Separate held-position ridge reference fits only train, selects its own alpha on held-position validation; never changes the shared reader.',
        budget='100epochs x19156 rows, batch256=7500 MLP updates/head. Four old independent heads total7600; equal aggregate row exposure but NOT identical updates/capacity.',
        capacity='One nested train-only PCA basis on original normalized seven-channel128-bar targets; masked labels imputed as source0. All ranks fixed, no research selection.896 is a numerical full-space control.',
        limits='Research sets reused; uncorrected original-endpoint weekly intervals. Held positions test interpolation within the same window, not unseen market/time. Position changes context and target segment. No encoder updates or automatic promotion.')


def original_root(meta):
    return Path(meta['identity']['manifest']['identity']['manifest']['source'])


def verify_cache(meta, out, split):
    idx = read_json(out/f'cache/{split}_index.json')
    if idx['manifest'] != meta or idx['target_scales_sha256'] != sha256(out/'cache/target_scales.json'):
        raise ValueError('Shared cache identity changed')
    verify_files(out/'cache', idx['files'])


@torch.no_grad()
def prepare_split(meta, out, split, device='cuda'):
    if split not in ('train', 'val', 'test', 'cross_research'):
        raise ValueError('Unknown split')
    if split in ('test', 'cross_research') and not (out/'selection_lock.json').exists():
        raise ValueError('Shared selections must be locked before research extraction')
    cache = out/'cache'; cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists():
        verify_cache(meta, out, split); return
    source = Path(meta['source']); old = meta['identity']['manifest']; files = {}
    scale_path = cache/'target_scales.json'
    if scale_path.exists() and sha256(scale_path) != sha256(source/'cache/target_scales.json'):
        raise ValueError('Inherited target scales changed')
    shutil.copyfile(source/'cache/target_scales.json', scale_path)
    stats = read_json(scale_path)
    data = ab.load_arrays(original_root(meta), split)
    original_stats = read_json(original_root(meta)/'cache/statistics.json')
    for p in sr.PREFIXES:
        y, mask = sr.targets(data, original_stats, p); y = pr.normalize(y, mask, stats)
        for key, value in [('y', y), ('mask', mask), ('current28', np.asarray(data['x'][:, p-1]))]:
            file = cache/f'{split}_p{p}_{key}.npy'
            if p in sr.TRAIN_PREFIXES:
                prior = np.load(source/'cache'/file.name, allow_pickle=False)
                if not np.array_equal(value, prior):
                    raise ValueError('Source anchor target/input replay mismatch')
            np.save(file, value, allow_pickle=False); files[file.name] = sha256(file)
        if split == 'train':
            file = cache/f'mean_p{p}.npy'
            np.save(file, (y*mask).sum(0, dtype=np.float64)/mask.sum(0).clip(1))
            files[file.name] = sha256(file)
    for name in old['identity']['variants']:
        model = pb.encoder(old, name, device)
        for p in sr.PREFIXES:
            file = cache/f'{split}_p{p}_{name}.npy'
            if p in sr.TRAIN_PREFIXES:
                shutil.copyfile(source/'cache'/file.name, file)
            else:
                parts = []
                for start in range(0, len(data['x']), old['extract_batch']):
                    x = torch.tensor(np.asarray(data['x'][start:start+old['extract_batch'], :p]), device=device)
                    parts.append(model.encoder(x)[:, -1].cpu().numpy())
                z = np.concatenate(parts)
                if not np.isfinite(z).all():
                    raise ValueError('Nonfinite frozen state')
                np.save(file, z, allow_pickle=False)
            files[file.name] = sha256(file)
        if any(param.requires_grad for param in model.parameters()):
            raise ValueError('Encoder not frozen')
        del model
        progress(f'Shared states cached: {split}/{name}, held positions={sr.HELD_PREFIXES}')
    files['target_scales.json'] = sha256(scale_path)
    atomic_json(dict(manifest=meta, files=files, windows=len(data['x']), encoder_updates=0,
        strict_prefix_inputs=True, target_scales_sha256=sha256(scale_path)), cache/f'{split}_index.json')
    verify_cache(meta, out, split)


def raw_data(out, split, variant, p):
    cache = out/'cache'
    return dict(x=np.load(cache/f'{split}_p{p}_{variant}.npy', allow_pickle=False),
                y=np.load(cache/f'{split}_p{p}_y.npy', allow_pickle=False),
                mask=np.load(cache/f'{split}_p{p}_mask.npy', allow_pickle=False))


def pooled_data(out, split, variant, scales=None):
    data = sr.pool({p: raw_data(out, split, variant, p) for p in sr.TRAIN_PREFIXES})
    if scales is not None:
        data['x'] = pr.design(data['x'], scales)
    return data


def fit_ridge(path, metadata, data, stats, alphas, device):
    if not (path/'ridge.json').exists():
        fits = pr.ridge_candidates(data['train']['x'], data['train']['y'], data['train']['mask'], alphas, device)
        choices = []
        for alpha, fit in fits.items():
            score, _ = pr.measure(pr.ridge_predict(fit, data['val']['x']), data['val']['y'], data['val']['mask'], stats)
            choices.append(dict(alpha=alpha, validation=score['metrics']))
        best = min(choices, key=lambda v: v['validation']['primary'])
        np.savez(path/'ridge.npz', **fits[best['alpha']])
        atomic_json(dict(metadata=metadata, candidates=choices, selected=best,
                         weights_sha256=sha256(path/'ridge.npz')), path/'ridge.json')
    ridge = read_json(path/'ridge.json')
    if (ridge['metadata'] != metadata or ridge['weights_sha256'] != sha256(path/'ridge.npz')
            or [v['alpha'] for v in ridge['candidates']] != list(alphas)
            or ridge['selected'] != min(ridge['candidates'], key=lambda v: v['validation']['primary'])):
        raise ValueError('Ridge selection/weights changed')
    return ridge


def prepare_references(meta, out, device='cuda'):
    """Unseen-position reference heads are separate; never enter pooled fitting."""
    stats = read_json(out/'cache/target_scales.json')
    for job in meta['experiments']:
        for p in sr.HELD_PREFIXES:
            path = out/'references'/f'{job["name"]}_p{p}'; path.mkdir(parents=True, exist_ok=True)
            metadata = dict(manifest=meta, variant=job['variant'], prefix=p, cache_sha256=pb.cache_identity(out))
            data = {s: raw_data(out, s, job['variant'], p) for s in ('train', 'val')}
            scales = pr.feature_scales(data['train']['x'])
            if (path/'feature_scales.json').exists() and read_json(path/'feature_scales.json') != scales:
                raise ValueError('Reference scales changed')
            atomic_json(scales, path/'feature_scales.json')
            for d in data.values(): d['x'] = pr.design(d['x'], scales)
            fit_ridge(path, metadata, data, stats, meta['alphas'], device)
    pca_path = out/'capacity_pca.npz'; original = original_root(meta)
    metadata = dict(manifest=meta, train_y_sha256=sha256(original/'cache/train_y.npy'),
                    source_pca_sha256=sha256(original/'pca.npz'))
    if not (out/'capacity_fit.json').exists():
        data = ab.load_arrays(original, 'train')
        progress('Fitting nested PCA on original training targets only')
        pca = ab.fit_pca(data, max(meta['ranks']))
        np.savez(pca_path, **pca)
        atomic_json(dict(metadata=metadata, pca_sha256=sha256(pca_path),
            train_energy_fraction=sr.pca_spectrum(pca, meta['ranks'])), out/'capacity_fit.json')
    fit = read_json(out/'capacity_fit.json')
    if fit['metadata'] != metadata or fit['pca_sha256'] != sha256(pca_path):
        raise ValueError('Capacity basis or source changed')


def worker(out, name, device='cuda'):
    meta = read_json(out/'manifest.json'); verify_code(meta)
    job = next(j for j in meta['experiments'] if j['name'] == name)
    path = out/name; path.mkdir(exist_ok=True)
    for split in ('train', 'val'): verify_cache(meta, out, split)
    metadata = dict(manifest=meta, job=job, cache_sha256=pb.cache_identity(out))
    if (path/'completion.json').exists():
        done = read_json(path/'completion.json')
        if done['metadata'] != metadata or done['status'] != 'complete':
            raise ValueError('Completed shared head identity changed')
        verify_files(path, done['files']); return
    raw = pooled_data(out, 'train', name); scales = pr.feature_scales(raw['x']); n = len(raw['x'])
    if (path/'feature_scales.json').exists() and read_json(path/'feature_scales.json') != scales:
        raise ValueError('Shared training scales changed')
    atomic_json(scales, path/'feature_scales.json')
    data = dict(train=dict(raw, x=pr.design(raw['x'], scales)), val=pooled_data(out, 'val', name, scales))
    fit_ridge(path, metadata, data, read_json(out/'cache/target_scales.json'), meta['alphas'], device)
    torch.manual_seed(job['seed']); model = pr.Probe(job['width'], job['seed'], meta['hidden']).to(device)
    initial = dict(seed=job['seed'], width=job['width'], hidden=meta['hidden'], sha256=pb.head_signature(model))
    if (path/'head_initialization.json').exists() and read_json(path/'head_initialization.json') != initial:
        raise ValueError('Shared initialization changed')
    atomic_json(initial, path/'head_initialization.json')
    opt = torch.optim.AdamW(model.parameters(), lr=job['lr'], weight_decay=1e-4)
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata or state['epoch'] != len(state['history']) or state['epoch'] > job['epochs']:
            raise ValueError('Resume identity or budget mismatch')
        pb.verify_history(state['history'], job, n, meta['batch'])
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        replay, _ = pr.run_epoch(model, data['val'], meta['batch'], device)
        expected = state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k], v, atol=1e-6, rtol=2e-5) for k, v in expected.items()):
            raise ValueError('Resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'], matched=True), path/'resume_validation.json')
    else:
        val, _ = pr.run_epoch(model, data['val'], meta['batch'], device)
        state = dict(metadata=metadata, epoch=0, history=[], initial_validation=val,
                     best_epoch=0, best_validation=val, best_model=ab.cpu_state(model))
        state.update(model=ab.cpu_state(model), optimizer=opt.state_dict(), rng=rng_state()); pb.publish(state, path)
    # Export epoch0 so report-only reviews can independently replay selection.
    atomic_json(dict(metadata=metadata, validation=state['initial_validation']), path/'initial_validation.json')
    for epoch in range(state['epoch']+1, job['epochs']+1):
        started = time.monotonic(); lr = ar.learning_rate(epoch, job['epochs'], job['lr'])
        for group in opt.param_groups: group['lr'] = lr
        train, steps = pr.run_epoch(model, data['train'], meta['batch'], device, opt, job['seed']+epoch*1009)
        if steps != (n+meta['batch']-1)//meta['batch']:
            raise ValueError('Shared update budget changed')
        val, _ = pr.run_epoch(model, data['val'], meta['batch'], device)
        if val['primary'] < state['best_validation']['primary']:
            state.update(best_epoch=epoch, best_validation=val, best_model=ab.cpu_state(model))
        state['history'].append(dict(epoch=epoch, lr=lr, train=train, validation=val,
            optimizer_steps=steps, windows=n, seconds=time.monotonic()-started))
        state.update(epoch=epoch, model=ab.cpu_state(model), optimizer=opt.state_dict(), rng=rng_state()); pb.publish(state, path)
        if epoch % 10 == 0 or epoch == 1:
            progress(f'{name}: epoch={epoch}/{job["epochs"]} val={val["primary"]:.5f} best={state["best_epoch"]}')
    atomic_json(dict(selected_epoch=state['best_epoch'], validation=state['best_validation'], allocated_epochs=job['epochs'],
        optimizer_steps=sum(r['optimizer_steps'] for r in state['history']), parameters=sum(p.numel() for p in model.parameters()),
        encoder_updates=0, rows_per_epoch=n, source_windows=n//len(sr.TRAIN_PREFIXES)), path/'training_summary.json')
    files = {p.name: sha256(p) for p in path.iterdir() if p.is_file()
             and p.name not in ('completion.json', 'progress.json', 'run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete', metadata=metadata, files=files), path/'completion.json')


def lock_selection(meta, out):
    """Reuse the strict checkpoint/epoch0 budget verifier for the pooled heads."""
    for split in ('train', 'val'):
        verify_cache(meta, out, split)
    rows = {}; weights = {}
    n = read_json(out/'cache/train_index.json')['windows']*len(sr.TRAIN_PREFIXES)
    for job in meta['experiments']:
        path = out/job['name']; done = read_json(path/'completion.json')
        expected = dict(manifest=meta, job=job, cache_sha256=pb.cache_identity(out))
        if done['status'] != 'complete' or done['metadata'] != expected:
            raise ValueError('Incomplete shared head')
        verify_files(path, done['files'])
        last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
        history = read_json(path/'history.json'); pb.verify_history(history, job, n, meta['batch'])
        if last['metadata'] != expected or best['metadata'] != expected or last['epoch'] != job['epochs'] or last['history'] != history:
            raise ValueError('Shared checkpoint budget or identity mismatch')
        if read_json(path/'initial_validation.json') != dict(metadata=expected, validation=last['initial_validation']):
            raise ValueError('Exported initial validation mismatch')
        epoch, val = min([(0, last['initial_validation'])]+[(r['epoch'], r['validation']) for r in history], key=lambda r: r[1]['primary'])
        summary = read_json(path/'training_summary.json'); ridge = read_json(path/'ridge.json')
        if (best['epoch'] != epoch or summary['selected_epoch'] != epoch or best['validation'] != val or summary['validation'] != val
                or last['best_epoch'] != epoch or last['best_validation'] != val or summary['encoder_updates'] != 0
                or summary['optimizer_steps'] != sum(r['optimizer_steps'] for r in history)):
            raise ValueError('Shared selection mismatch')
        if (ridge['metadata'] != expected or [r['alpha'] for r in ridge['candidates']] != meta['alphas']
                or ridge['selected'] != min(ridge['candidates'], key=lambda r: r['validation']['primary'])
                or ridge['weights_sha256'] != sha256(path/'ridge.npz')):
            raise ValueError('Shared ridge selection mismatch')
        rows[job['name']] = dict(mlp=summary, ridge=ridge['selected'])
        weights[job['name']] = {k: sha256(path/k) for k in ('best.pt', 'last.pt', 'ridge.npz', 'feature_scales.json')}
    for seed in (42, 43):
        if read_json(out/f'plain_s{seed}/head_initialization.json') != read_json(out/f'sampled_s{seed}/head_initialization.json'):
            raise ValueError('Paired shared heads have different initialization')
    refs = {}
    for path in sorted((out/'references').iterdir()):
        ridge = read_json(path/'ridge.json')
        ref = ridge['metadata']
        if (ridge['metadata']['manifest'] != meta or ridge['metadata']['cache_sha256'] != pb.cache_identity(out)
                or path.name != f'{ref["variant"]}_p{ref["prefix"]}'
                or ridge['weights_sha256'] != sha256(path/'ridge.npz')
                or ridge['selected'] != min(ridge['candidates'], key=lambda v: v['validation']['primary'])
                or [v['alpha'] for v in ridge['candidates']] != meta['alphas']):
            raise ValueError('Held-position reference changed')
        refs[path.name] = dict(selection=ridge['selected'], files={p.name: sha256(p) for p in path.iterdir() if p.is_file()})
    expected_refs = {f'{j["name"]}_p{p}' for j in meta['experiments'] for p in sr.HELD_PREFIXES}
    if set(refs) != expected_refs:
        raise ValueError('Missing held-position reference')
    fit = read_json(out/'capacity_fit.json')
    if fit['metadata']['manifest'] != meta or fit['pca_sha256'] != sha256(out/'capacity_pca.npz'):
        raise ValueError('Capacity basis changed')
    result = dict(manifest=meta, selections=rows, weights=weights, references=refs, capacity_fit=fit)
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json') != result:
        raise ValueError('Shared selections changed after locking')
    atomic_json(result, out/'selection_lock.json'); return result


def predict(meta, out, job, split, kind, prefix=None, device='cuda'):
    path = out/job['name']; scales = read_json(path/'feature_scales.json')
    if prefix is None:
        data = pooled_data(out, split, job['name'], scales)
    else:
        data = raw_data(out, split, job['name'], prefix); data['x'] = pr.design(data['x'], scales)
    if kind == 'ridge':
        with np.load(path/'ridge.npz', allow_pickle=False) as fit: pred = pr.ridge_predict(fit, data['x'])
    else:
        model = pr.Probe(job['width'], job['seed'], meta['hidden']).to(device)
        ck = torch.load(path/('last.pt' if kind == 'mlp_last' else 'best.pt'), map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model']); pred = pr.predictions(model, data['x'], meta['batch'], device)
    return pred, data


def reference_prediction(meta, out, job, split, p, device):
    if p in sr.TRAIN_PREFIXES:
        old = meta['identity']['manifest']
        old_job = next(j for j in old['experiments'] if j['name'] == f'{job["name"]}_p{p}')
        return pb.predict_probe(old, Path(meta['source']), old_job, split, 'ridge', device=device)
    path = out/'references'/f'{job["name"]}_p{p}'
    data = raw_data(out, split, job['name'], p)
    data['x'] = pr.design(data['x'], read_json(path/'feature_scales.json'))
    with np.load(path/'ridge.npz', allow_pickle=False) as fit: pred = pr.ridge_predict(fit, data['x'])
    return pred, data


def evaluate_capacity(meta, out):
    if not (out/'selection_lock.json').exists():
        raise ValueError('Capacity research evaluation requires selection lock')
    original = original_root(meta); stats = read_json(original/'cache/statistics.json')
    with np.load(out/'capacity_pca.npz', allow_pickle=False) as f: pca = dict(f)
    with np.load(original/'pca.npz', allow_pickle=False) as f: prior = dict(f)
    control_rank = len(prior['components'])
    report = dict(fit=read_json(out/'capacity_fit.json'), reference_rank=control_rank, datasets={},
        limits='PCA directly compresses observed targets, not raw28 inputs. Not a future forecast or proof of neural trainability. Full-space rank is a numerical control; masked MSE need not be monotonic with rank.')
    for split in ('val', 'test', 'cross_research'):
        data = ab.load_arrays(original, split); scores = {}; errors = {}
        for rank in meta['ranks']:
            prediction = ab.pca_predict(cb.prefix_pca(pca, rank), data)
            scores[str(rank)], errors[str(rank)] = ab.measure(prediction, data, stats)
        old_pred = ab.pca_predict(prior, data)
        new_pred = ab.pca_predict(cb.prefix_pca(pca, control_rank), data)
        matched = bool(np.allclose(new_pred, old_pred, atol=1e-4, rtol=2e-5))
        if not matched:
            raise ValueError('Nested PCA failed to reproduce existing PCA control')
        full_diff = None
        if max(meta['ranks']) == np.prod(data['y'].shape[1:]):
            full = ab.pca_predict(cb.prefix_pca(pca, max(meta['ranks'])), data)
            full_diff = float(np.max(np.abs(full-data['y'])))
            if not np.allclose(full, data['y'], atol=1e-4, rtol=2e-5):
                raise ValueError('Full-space PCA roundtrip failed')
        paired = {}
        # The inherited cache has endpoint inventories only for the research splits.
        if split != 'val':
            inv = read_json(original/f'cache/{split}_inventory.json')
            paired = {f'{rank}_minus_{control_rank}': {k: ab.pair_groups(errors[str(rank)][k], errors[str(control_rank)][k], inv)
                      for k in ('primary', 'path', 'changes', 'body', 'activity', 'close_mae_bps')}
                      for rank in meta['ranks'] if rank != control_rank}
        report['datasets'][split] = dict(scores=scores, paired=paired,
            original_control_replay=dict(matched=matched, max_abs=float(np.abs(new_pred-old_pred).max())), full_roundtrip_max_abs=full_diff)
        atomic_json({n: {k: v.tolist() for k, v in row.items()} for n, row in errors.items()}, out/f'capacity_{split}_errors.json')
    atomic_json(report, out/'capacity_metrics.json')
    return report


def evaluate(meta, out, device='cuda'):
    lock = lock_selection(meta, out); stats = read_json(out/'cache/target_scales.json'); replay = {}
    for job in meta['experiments']:
        for kind in ('ridge', 'mlp', 'mlp_last'):
            pred, data = predict(meta, out, job, 'val', kind, device=device)
            actual = pr.measure(pred, data['y'], data['mask'], stats)[0]['metrics']
            expected = (read_json(out/job['name']/'history.json')[-1]['validation'] if kind == 'mlp_last'
                        else lock['selections'][job['name']][kind]['validation'])
            if any(not np.isclose(actual[k], v, atol=1e-6, rtol=2e-5) for k, v in expected.items()):
                raise ValueError('Shared validation replay mismatch')
            replay[job['name']+'/'+kind] = dict(matched=True, actual=actual, expected=expected)
        for p in sr.PREFIXES:
            pred, data = reference_prediction(meta, out, job, 'val', p, device)
            actual = pr.measure(pred, data['y'], data['mask'], stats)[0]['metrics']
            if p in sr.TRAIN_PREFIXES:
                expected = read_json(Path(meta['source'])/f'{job["name"]}_p{p}/ridge.json')['selected']['validation']
            else:
                expected = lock['references'][f'{job["name"]}_p{p}']['selection']['validation']
            if any(not np.isclose(actual[k], v, atol=1e-6, rtol=2e-5) for k, v in expected.items()):
                raise ValueError('Independent reference replay mismatch')
            replay[f'{job["name"]}_p{p}/independent_ridge'] = dict(matched=True, actual=actual, expected=expected)
    atomic_json(replay, out/'validation_replay.json')
    report = dict(schema=SCHEMA, selections=lock['selections'], goal=GOAL, datasets={}, limits=meta['limits'])
    old_errors = {s: read_json(Path(meta['source'])/f'{s}_errors.json') for s in ('test', 'cross_research')}
    for split in ('test', 'cross_research'):
        prepare_split(meta, out, split, device)
        inv = read_json(original_root(meta)/f'cache/{split}_inventory.json'); atomic_json(inv, out/f'{split}_inventory.json')
        scores = {}; errors = {}; pairs = []
        def record(name, prediction, data):
            scores[name], errors[name] = pr.measure(prediction, data['y'], data['mask'], stats)
        for job in meta['experiments']:
            for p in sr.PREFIXES:
                label = f'{job["name"]}_p{p}'
                for kind in ('ridge', 'mlp', 'mlp_last'):
                    pred, data = predict(meta, out, job, split, kind, p, device); record(label+'/'+kind, pred, data)
                pred, data = reference_prediction(meta, out, job, split, p, device)
                record(label+'/independent_ridge', pred, data)
                if p in sr.TRAIN_PREFIXES:
                    for k, arr in errors[label+'/independent_ridge'].items():
                        if not np.allclose(arr, old_errors[split][label+'/ridge'][k], atol=1e-5, rtol=2e-5):
                            raise ValueError('Original independent readout errors did not reproduce')
                pairs.extend((label+'/'+kind, label+'/independent_ridge') for kind in ('ridge', 'mlp'))
            for group, prefixes in [('trained', sr.TRAIN_PREFIXES), ('held', sr.HELD_PREFIXES)]:
                for kind in ('ridge', 'mlp', 'mlp_last', 'independent_ridge'):
                    key = f'{job["name"]}/{group}/{kind}'
                    # Average within original window FIRST, preserving dependence in intervals.
                    errors[key] = {metric: np.mean([errors[f'{job["name"]}_p{p}/{kind}'][metric] for p in prefixes], axis=0)
                                   for metric in pr.METRICS}
                    scores[key] = dict(metrics={k: float(v.mean()) for k, v in errors[key].items()}, positions=list(prefixes),
                        aggregation='equal positions within each original window; then windows')
                pairs.extend((f'{job["name"]}/{group}/{kind}', f'{job["name"]}/{group}/independent_ridge') for kind in ('ridge', 'mlp'))
        for p in sr.PREFIXES:
            data = raw_data(out, split, 'current28', p); mean = np.load(out/f'cache/mean_p{p}.npy')
            record(f'mean_p{p}', np.broadcast_to(mean, data['y'].shape).astype(np.float32), data)
            for kind in ('ridge', 'mlp', 'mlp_last'):
                for seed in (42, 43): pairs.append((f'sampled_s{seed}_p{p}/{kind}', f'plain_s{seed}_p{p}/{kind}'))
                for name in meta['identity']['manifest']['identity']['variants']:
                    pairs.append((f'{name}_p{p}/{kind}', f'current28_p{p}/{kind}'))
        paired = {a+'_minus_'+b: {k: ab.pair_groups(errors[a][k], errors[b][k], inv) for k in pr.METRICS} for a, b in pairs}
        report['datasets'][split] = dict(scores=scores, paired=paired, old_independent_replay=True)
        atomic_json({n: {k: v.tolist() for k, v in row.items()} for n, row in errors.items()}, out/f'{split}_errors.json')
        progress(f'Shared readouts evaluated: {split}; trained and held positions kept separate')
    atomic_json(report, out/'shared_readout_metrics.json')
    capacity = evaluate_capacity(meta, out)
    lines = ['# Shared frozen readout and linear capacity reference', '',
             'No encoder updates. Held-position reference heads are separate diagnostics, not shared-reader fits.', '']
    for split, dataset in report['datasets'].items():
        lines += [f'## {split}', '', '| reader | primary | path | body | activity | change1 |', '|---|---:|---:|---:|---:|---:|']
        for name, row in dataset['scores'].items():
            lines.append('| '+name+' | '+' | '.join(f'{row["metrics"][k]:.5f}' for k in pr.METRICS[:-1])+' |')
    lines += ['', '## PCA: all prespecified ranks, original full-window objective', '', '| split | rank | primary | changes |', '|---|---:|---:|---:|']
    for split, dataset in capacity['datasets'].items():
        for rank, row in dataset['scores'].items(): lines.append(f'| {split} | {rank} | {row["metrics"]["primary"]:.6f} | {row["metrics"]["changes"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(lines))
    return report


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);path=out/job['name'];path.mkdir(exist_ok=True);log=(path/'run.log').open('a')
                try:proc=subprocess.Popen([sys.executable,'-m','obson.babel.shared_readout_benchmark','worker','--out',str(out),'--name',job['name']],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[job['name']]=(proc,log);progress(f'Started {job["name"]}, pid={proc.pid}')
            for name,(proc,log) in list(active.items()):
                path=out/name/'progress.json'
                if path.exists():
                    try:row=read_json(path)
                    except (ValueError,OSError):row=None
                    if row and seen.get(name)!=row:progress(f'{name}: {row}');seen[name]=row
                code=proc.poll()
                if code is not None:
                    log.close();del active[name]
                    if code:raise RuntimeError(f'{name} exited{code}; see worker run.log')
            if active:time.sleep(1)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        signal.signal(signal.SIGTERM,handler)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('all', 'preflight', 'evaluate', 'worker'))
    parser.add_argument('--source'); parser.add_argument('--out', required=True)
    parser.add_argument('--name'); parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args(); out = Path(args.out).resolve(); ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError('Formal extraction and readout fitting run on AutoDL CUDA')
    if args.action == 'worker': worker(out, args.name); return
    if not args.source: parser.error('--source required')
    if not 1 <= args.jobs <= 2: raise ValueError('Use one or two concurrent workers')
    source = Path(args.source).resolve(); progress('Verifying immutable prefix/sampling encoder sources')
    identity = source_identity(source); old = identity['manifest']; sm = old['identity']['manifest']
    if any(v['job']['epochs'] != 200 for v in old['identity']['variants'].values()):
        raise ValueError('Frozen epoch200 encoders required')
    if (sm['windows_per_epoch'] != 4789 or sm['config']['latent'] != 512
            or old['batch'] != 256 or old['hidden'] != 256 or any(j['epochs'] != 100 for j in old['experiments'])):
        raise ValueError('Expected completed512 prefix experiment')
    for dep in [source, Path(old['source']), Path(sm['source']), Path(sm['teacher_source'])]:
        if out == dep or out in dep.parents or dep in out.parents: raise ValueError('Separate output required')
    runtime = read_json(source/'runtime.json')
    if runtime['torch'] != str(torch.__version__) or runtime['numpy'] != np.__version__:
        raise ValueError('Restore original Torch/NumPy environment')
    meta = make_manifest(source, identity)
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json') != meta: raise ValueError('Shared configuration changed')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty output without matching manifest')
    out.mkdir(parents=True, exist_ok=True); atomic_json(meta, out/'manifest.json'); (out/'completion.json').unlink(missing_ok=True)
    atomic_json(dict(torch=str(torch.__version__), numpy=np.__version__, gpu=torch.cuda.get_device_name(), jobs=args.jobs,
        source_runtime=runtime, started_unix=time.time()), out/'runtime.json')
    for split in ('train', 'val'): prepare_split(meta, out, split)
    if read_json(out/'cache/train_index.json')['windows'] != 4789 or read_json(out/'cache/val_index.json')['windows'] != 978:
        raise ValueError('Source window counts changed')
    prepare_references(meta, out)
    atomic_json(dict(status='passed', encoder_updates=0, train_windows=4789, pooled_train_rows=19156,
        train_prefixes=list(sr.TRAIN_PREFIXES), held_prefixes=list(sr.HELD_PREFIXES), strict_prefix_inputs=True), out/'preflight.json')
    if args.action == 'preflight': return
    if args.action == 'all': run_jobs(out, args.jobs)
    evaluate(meta, out)
    if source_identity(source) != identity: raise ValueError('Frozen source changed during shared experiment')
    for split in ('train', 'val', 'test', 'cross_research'): verify_cache(meta, out, split)
    files = {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.md', '.npz') and p.name != 'completion.json'}
    files.update({f'cache/{s}_index.json': sha256(out/f'cache/{s}_index.json') for s in ('train', 'val', 'test', 'cross_research')})
    atomic_json(dict(status='complete', encoder_updates=0, source_unchanged=True, automatic_promotion=False,
        mlp_optimizer_steps=sum(read_json(out/j['name']/'training_summary.json')['optimizer_steps'] for j in meta['experiments']),
        files=files), out/'completion.json')
    progress('Shared readout and capacity diagnostics complete; encoder weights unchanged')


if __name__ == '__main__':
    import fcntl
    if '--out' not in sys.argv: main()
    else:
        root = Path(sys.argv[sys.argv.index('--out')+1]).resolve(); root.parent.mkdir(parents=True, exist_ok=True)
        name = sys.argv[sys.argv.index('--name')+1] if len(sys.argv)>1 and sys.argv[1]=='worker' and '--name' in sys.argv else 'controller'
        with (root.parent/('.'+root.name+'.'+name+'.lock')).open('a') as lockfile:
            try: fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise SystemExit('Output/job already running')
            main()
