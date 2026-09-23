"""Convergent 2x2 encoder experiment: 512/768 coordinates and local causal supervision."""
import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import bar_alignment as ba, shared_readout_benchmark as sh, prefix_readout as pr
from . import sampling_benchmark as sb, window_sampling as ws, coverage_audit as cov
from . import architecture as ar, architecture_benchmark as ab, pca_teacher as pt, capacity_benchmark as cb
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-bar-alignment-v1'
EVAL_PREFIXES = (32, 48, 64, 80, 96, 112, 128)
GLOBAL_METRICS = ('primary', 'path', 'changes', 'body', 'activity', 'close_mae_bps')


def code_identity():
    return sh.code_identity() | {Path(ba.__file__).name: sha256(ba.__file__), Path(__file__).name: sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256'] != code_identity(): raise ValueError('Alignment implementation changed; use a new run')


def source_identity(root):
    meta = read_json(root/'manifest.json'); sh.verify_code(meta); done = read_json(root/'completion.json')
    if meta['schema'] != sh.SCHEMA or done['status'] != 'complete' or not done['source_unchanged'] or done['encoder_updates'] != 0:
        raise ValueError('Completed immutable shared-readout source required')
    ab.sc.verify_worker_files(root, done['files'])
    if sh.source_identity(Path(meta['source'])) != meta['identity']: raise ValueError('Upstream source changed')
    for split in ('train', 'val', 'test', 'cross_research'): sh.verify_cache(meta, root, split)
    lock = read_json(root/'selection_lock.json')
    if lock['manifest'] != meta or lock['capacity_fit'] != read_json(root/'capacity_fit.json'):
        raise ValueError('Shared source selection changed')
    files = dict(done['files']); files['completion.json'] = sha256(root/'completion.json')
    for job in meta['experiments']:
        path = root/job['name']; row = read_json(path/'completion.json')
        if row['status'] != 'complete' or row['metadata'] != dict(manifest=meta, job=job, cache_sha256=sh.pb.cache_identity(root)):
            raise ValueError('Incomplete source head')
        verify_files(path, row['files'])
        for name, digest in lock['weights'][job['name']].items():
            if sha256(path/name) != digest: raise ValueError('Shared source weights changed')
        files[f'{job["name"]}/completion.json'] = sha256(path/'completion.json')
    for name, row in lock['references'].items(): verify_files(root/'references'/name, row['files'])
    return dict(manifest=meta, files=files)


def make_manifest(source, identity, widths=(512, 768), epochs=200, batch=128, micro=64, hidden=256):
    sm = identity['manifest']['identity']['manifest']['identity']['manifest']
    if len(widths) != 2 or widths[0] != sm['config']['latent'] or widths[1] <= widths[0]:
        raise ValueError('Original and one larger width required')
    if min(epochs, batch, micro) < 1 or micro > batch or batch % micro: raise ValueError('Invalid training budget')
    jobs = [dict(name=f'w{w}_{mode}_s{s}', width=w, joint=mode=='joint', mode=mode, seed=s,
                 epochs=epochs, lr=3e-4, head_lr=1e-3) for w in widths for mode in ('endpoint', 'joint') for s in (42, 43)]
    return dict(schema=SCHEMA, source=str(source.resolve()), identity=identity,
        sampling_source=identity['manifest']['identity']['manifest']['source'], original_source=sm['source'],
        config=sm['config'], widths=list(widths), epochs=epochs, batch=batch, micro=micro,
        windows_per_epoch=sm['windows_per_epoch'], experiments=jobs, local_hidden=hidden,
        local_weight=ba.LOCAL_WEIGHT, local_count=ba.LOCAL_COUNT,
        train_prefixes=list(ba.TRAIN_PREFIXES), validation_prefixes=list(ba.VAL_PREFIXES), held_prefixes=list(ba.HELD_PREFIXES),
        code_sha256=code_identity(), goal='Converge on a causal per-bar historical state while retaining full-window reconstruction.',
        sampling='Reuse exact original dense-pool membership and minibatch order; same random four-prefix plan across all cells of a seed.',
        objective='Original global SmoothL1 plus0.25 local SmoothL1. Endpoint controls detach local states; separate AdamW/clipping prevents reader gradients changing global updates.',
        selection='Each cell uses global validation MSE +0.25 local validation MSE at32/64/96/128, including epoch0. No held-position or research selection of epochs.',
        budget='All8 cells from scratch200 epochs;7600 encoder and7600 reader steps each. Same957800 window and3831200 local-target exposures per cell; different width/extra encoder gradients mean different compute.',
        decision=dict(local_improvement=.10, global_retention=.05, family_retention=.10, upgrade_gain=.05,
                      priority='Prefer512 joint if eligible; use768 only for supported extra held-local gain with retention. Among768 prefer endpoint unless joint adds supported gain. Otherwise retain512 endpoint and stop.',
                      scope='Exploratory route screen on reused research data, both seeds AND best/last. No automatic deployment or new experiment generation.'),
        stop_rule='One matrix, fixed budgets and weight. No automatic layer/head/feature/weight sweep or training extension.',
        limits='Widths also change fixed PCA decoder rank; not a pure embedding-width effect. Per-bar evaluation is32..128 within endpoint-selected reset windows, not all initial bars or a streaming replacement.')


def load_data(meta, split):
    if split == 'pool':
        root = Path(meta['sampling_source'])/'candidates'; prefix = 'train'
    else:
        root = Path(meta['original_source'])/'cache'; prefix = split
    return {k: np.load(root/f'{prefix}_{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('x', 'y', 'mask')}


def verify_cache(meta, out):
    idx = read_json(out/'cache/index.json')
    if idx['manifest'] != meta: raise ValueError('Alignment cache identity changed')
    verify_files(out/'cache', idx['files'])
    for path, digest in idx['source_files'].items():
        if sha256(path) != digest: raise ValueError('Pinned data or sampling schedule changed')


def prepare(meta, out):
    cache = out/'cache'; cache.mkdir(exist_ok=True)
    if (cache/'index.json').exists(): verify_cache(meta, out); return
    shared = Path(meta['source']); sample = Path(meta['sampling_source']); original = Path(meta['original_source'])
    shutil.copyfile(original/'cache/statistics.json', cache/'statistics.json')
    shutil.copyfile(shared/'cache/target_scales.json', cache/'local_scales.json')
    with np.load(shared/'capacity_pca.npz', allow_pickle=False) as f: full = dict(f)
    with np.load(original/'pca.npz', allow_pickle=False) as f: old = dict(f)
    train = load_data(meta, 'train')
    for w in meta['widths']:
        pca = old if w == meta['widths'][0] else cb.prefix_pca(full, w)
        if len(pca['components']) != w: raise ValueError('Insufficient pinned PCA rank')
        np.savez(cache/f'pca{w}.npz', **pca)
        if w == meta['widths'][0]:
            scales = read_json(sample/'cache/coordinate_scales.json')
        else:
            scales = pt.fit_coordinate_scales(pt.coefficients(pca, train))
        atomic_json(scales, cache/f'coordinates{w}.json')
    source_files = {}
    for split in ('train', 'val', 'test', 'cross_research'):
        for key in ('x', 'y', 'mask'):
            path = original/f'cache/{split}_{key}.npy'; source_files[str(path)] = sha256(path)
    for key in ('x', 'y', 'mask'):
        path = sample/f'candidates/train_{key}.npy'; source_files[str(path)] = sha256(path)
    for s in (42, 43):
        path = sample/f'candidates/sampling_s{s}.json'; source_files[str(path)] = sha256(path)
        plan = read_json(path)
        if plan['budget'] != meta['windows_per_epoch'] or len(plan['epochs']) < meta['epochs']:
            raise ValueError('Sampling budget exceeds fixed source plan')
    for path in (shared/'capacity_pca.npz', sample/'candidates/index.json', sample/'cache/coordinate_scales.json', shared/'cache/target_scales.json'):
        source_files[str(path)] = sha256(path)
    files = {p.name: sha256(p) for p in cache.iterdir() if p.is_file() and p.name != 'index.json'}
    atomic_json(dict(manifest=meta, files=files, source_files=source_files, fitted_on='original_fixed_train_only'), cache/'index.json')
    verify_cache(meta, out)


def model_for(meta, out, job, device):
    config = dict(meta['config'], latent=job['width'])
    with np.load(out/f'cache/pca{job["width"]}.npz', allow_pickle=False) as f: pca = dict(f)
    scales = read_json(out/f'cache/coordinates{job["width"]}.json')
    return ba.AlignedStudent(config, job['seed'], pca, scales, meta['local_hidden']).to(device)


def state_signature(model):
    return sh.pb.head_signature(model)


def validation(model, data, stats, local, meta, job, device):
    ps = np.tile(ba.VAL_PREFIXES, (len(data['x']), 1)).astype(np.int64)
    return ba.run_epoch(model, data, stats, local, meta['batch'], meta['micro'], device, job['joint'], ps)[0]


def publish(state, path, improved=True):
    atomic_save(state, path/'last.pt')
    if improved or not (path/'best.pt').exists():
        atomic_save(dict(metadata=state['metadata'], epoch=state['best_epoch'], validation=state['best_validation'], model=state['best_model']), path/'best.pt')
    atomic_json(state['history'], path/'history.json')
    row = state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'], epochs=state['metadata']['job']['epochs'], best_epoch=state['best_epoch'],
        validation=row.get('validation'), seconds=row.get('seconds'), remaining_minutes=row.get('remaining_minutes')), path/'progress.json')


def plan_row(meta, n, job, epoch):
    ids = ws.sample_ids(n, meta['windows_per_epoch'], job['seed'], epoch)
    ps = ba.position_plan(len(ids), job['seed'], epoch)
    values, counts = np.unique(ps, return_counts=True)
    return ids, ps, dict(window_ids_sha256=cov.ndarray_hash(ids), prefixes_sha256=cov.ndarray_hash(ps),
                        prefix_exposures={str(k): int(v) for k, v in zip(values, counts)})


def verify_history(meta, out, job, history):
    plan = read_json(Path(meta['sampling_source'])/f'candidates/sampling_s{job["seed"]}.json')
    if [r['epoch'] for r in history] != list(range(1, len(history)+1)) or len(history) > job['epochs']:
        raise ValueError('Noncontiguous or excessive alignment history')
    for row in history:
        epoch = row['epoch']; _, _, expected = plan_row(meta, plan['pool_size'], job, epoch)
        if (row['sampling'] != expected or expected['window_ids_sha256'] != plan['epochs'][epoch-1]['ids_sha256']
                or row['windows'] != meta['windows_per_epoch'] or row['local_exposures'] != meta['windows_per_epoch']*ba.LOCAL_COUNT
                or row['encoder_steps'] != (meta['windows_per_epoch']+meta['batch']-1)//meta['batch'] or row['head_steps'] != row['encoder_steps']
                or row['lr'] != ar.learning_rate(epoch, job['epochs'], job['lr'])
                or row['head_lr'] != ar.learning_rate(epoch, job['epochs'], job['head_lr'])):
            raise ValueError('Sampling, positions, schedule or update budget changed')


def worker(out, name, device='cuda'):
    meta = read_json(out/'manifest.json'); verify_code(meta); verify_cache(meta, out)
    job = next(j for j in meta['experiments'] if j['name'] == name); path = out/name; path.mkdir(exist_ok=True)
    metadata = dict(manifest=meta, job=job, cache_sha256=sha256(out/'cache/index.json'))
    if (path/'completion.json').exists():
        row = read_json(path/'completion.json')
        if row['status'] != 'complete' or row['metadata'] != metadata: raise ValueError('Completed alignment identity mismatch')
        verify_files(path, row['files']); return
    torch.manual_seed(job['seed']); model = model_for(meta, out, job, device)
    signatures = dict(core=state_signature(model.core), reader=state_signature(model.local_head))
    encoder_opt = torch.optim.AdamW([p for p in model.core.parameters() if p.requires_grad], lr=job['lr'], weight_decay=.01)
    head_opt = torch.optim.AdamW(model.local_head.parameters(), lr=job['head_lr'], weight_decay=1e-4)
    stats = read_json(out/'cache/statistics.json'); local = read_json(out/'cache/local_scales.json')
    data = dict(pool=load_data(meta, 'pool'), val=load_data(meta, 'val'))
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata or state['epoch'] != len(state['history']) or state['epoch'] > job['epochs']:
            raise ValueError('Resume identity/budget mismatch')
        verify_history(meta, out, job, state['history'])
        model.load_state_dict(state['model']); encoder_opt.load_state_dict(state['encoder_optimizer']); head_opt.load_state_dict(state['head_optimizer']); restore_rng(state['rng'])
        replay = validation(model, data['val'], stats, local, meta, job, device)
        expected = state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k], v, atol=1e-6, rtol=2e-5) for k, v in expected.items()): raise ValueError('Resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'], matched=True), path/'resume_validation.json')
    else:
        initial = validation(model, data['val'], stats, local, meta, job, device)
        state = dict(metadata=metadata, epoch=0, history=[], initial_validation=initial,
                     best_epoch=0, best_validation=initial, best_model=ab.cpu_state(model))
    initial_record = dict(metadata=metadata, validation=state['initial_validation'], signatures=signatures)
    if (path/'initial_validation.json').exists() and read_json(path/'initial_validation.json') != initial_record:
        raise ValueError('Initialization changed')
    atomic_json(initial_record, path/'initial_validation.json')
    if job['width'] == meta['widths'][0]:
        source = torch.load(Path(meta['sampling_source'])/f'sampled_s{job["seed"]}/last.pt', map_location='cpu', weights_only=True)
        reference = source['initial_validation']
        if any(not np.isclose(state['initial_validation'][k], reference[k], atol=1e-6, rtol=2e-5) for k in ('primary','path','changes','body','activity','change1','change4','change16','change32')):
            raise ValueError('Original512 initialization failed to reproduce')
        atomic_json(dict(matched=True, source_validation=reference, new_validation=state['initial_validation']), path/'source_initialization.json'); del source
    if not (path/'last.pt').exists():
        state.update(model=ab.cpu_state(model), encoder_optimizer=encoder_opt.state_dict(), head_optimizer=head_opt.state_dict(), rng=rng_state())
        publish(state, path)
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1, job['epochs']+1):
        started = time.monotonic(); lr = ar.learning_rate(epoch, job['epochs'], job['lr']); hlr = ar.learning_rate(epoch, job['epochs'], job['head_lr'])
        for group in encoder_opt.param_groups: group['lr'] = lr
        for group in head_opt.param_groups: group['lr'] = hlr
        ids, ps, exposure = plan_row(meta, len(data['pool']['x']), job, epoch)
        train, steps = ba.run_epoch(model, ws.subset(data['pool'], ids), stats, local, meta['batch'], meta['micro'], device,
            job['joint'], ps, encoder_opt, head_opt, job['seed']+epoch*1009)
        if steps != (len(ids)+meta['batch']-1)//meta['batch']: raise ValueError('Encoder update budget changed')
        val = validation(model, data['val'], stats, local, meta, job, device)
        improved = val['selection'] < state['best_validation']['selection']
        if improved: state.update(best_epoch=epoch, best_validation=val, best_model=ab.cpu_state(model))
        elapsed = time.monotonic()-started
        state['history'].append(dict(epoch=epoch, train=train, validation=val, lr=lr, head_lr=hlr,
            encoder_steps=steps, head_steps=steps, windows=len(ids), local_exposures=int(ps.size), sampling=exposure,
            seconds=elapsed, remaining_minutes=elapsed*(job['epochs']-epoch)/60))
        state.update(epoch=epoch, model=ab.cpu_state(model), encoder_optimizer=encoder_opt.state_dict(), head_optimizer=head_opt.state_dict(), rng=rng_state()); publish(state, path, improved)
        progress(f'{name}: epoch={epoch}/{job["epochs"]} global={val["primary"]:.5f} local={val["local_primary"]:.5f} selected={state["best_epoch"]} seconds={elapsed:.1f}')
    verify_history(meta, out, job, state['history'])
    publish(state, path)
    atomic_json(dict(selected_epoch=state['best_epoch'], validation=state['best_validation'], allocated_epochs=job['epochs'],
        encoder_steps=sum(r['encoder_steps'] for r in state['history']), head_steps=sum(r['head_steps'] for r in state['history']),
        window_exposures=sum(r['windows'] for r in state['history']), local_exposures=sum(r['local_exposures'] for r in state['history']),
        parameters=dict(core=sum(p.numel() for p in model.core.parameters() if p.requires_grad), reader=sum(p.numel() for p in model.local_head.parameters())),
        training_seconds=sum(r['seconds'] for r in state['history']), peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None), path/'training_summary.json')
    files = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json', 'progress.json', 'run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete', metadata=metadata, files=files), path/'completion.json')


def lock_selection(meta, out):
    verify_cache(meta, out); rows = {}; weights = {}; initials = {}
    for job in meta['experiments']:
        path = out/job['name']; done = read_json(path/'completion.json')
        expected = dict(manifest=meta, job=job, cache_sha256=sha256(out/'cache/index.json'))
        if done['status'] != 'complete' or done['metadata'] != expected: raise ValueError('Incomplete alignment cell')
        verify_files(path, done['files'])
        last = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
        history = read_json(path/'history.json'); verify_history(meta, out, job, history)
        initial = read_json(path/'initial_validation.json'); summary = read_json(path/'training_summary.json')
        if (last['metadata'] != expected or best['metadata'] != expected or initial['metadata'] != expected
                or last['epoch'] != job['epochs'] or len(history) != job['epochs'] or last['history'] != history
                or initial['validation'] != last['initial_validation']):
            raise ValueError('Alignment checkpoint/initial validation mismatch')
        epoch, val = min([(0, initial['validation'])]+[(r['epoch'], r['validation']) for r in history], key=lambda v: v[1]['selection'])
        if (best['epoch'] != epoch or best['validation'] != val or summary['selected_epoch'] != epoch or summary['validation'] != val
                or last['best_epoch'] != epoch or last['best_validation'] != val or summary['allocated_epochs'] != job['epochs']
                or summary['encoder_steps'] != sum(r['encoder_steps'] for r in history) or summary['head_steps'] != sum(r['head_steps'] for r in history)
                or summary['window_exposures'] != sum(r['windows'] for r in history) or summary['local_exposures'] != sum(r['local_exposures'] for r in history)):
            raise ValueError('Alignment selection or budget mismatch')
        rows[job['name']] = summary; initials[job['name']] = initial['signatures']
        weights[job['name']] = {kind: sha256(path/f'{kind}.pt') for kind in ('best', 'last')}
    for width in meta['widths']:
        for seed in (42, 43):
            if initials[f'w{width}_endpoint_s{seed}'] != initials[f'w{width}_joint_s{seed}']:
                raise ValueError('Within-width paired initialization differs')
    result = dict(manifest=meta, trials=rows, weights=weights, initial_signatures=initials)
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json') != result:
        raise ValueError('Selection changed after research evaluation')
    atomic_json(result, out/'selection_lock.json'); return result


@torch.no_grad()
def predict(model, data, original, local, batch, device):
    model.eval(); globals_ = []; predictions = {p: [] for p in EVAL_PREFIXES}; targets = {p: [] for p in EVAL_PREFIXES}; masks = {p: [] for p in EVAL_PREFIXES}
    for start in range(0, len(data['x']), batch):
        b = {k: torch.tensor(np.asarray(v[start:start+batch]), device=device) for k, v in data.items()}
        ps = torch.tensor(EVAL_PREFIXES, device=device)[None].expand(len(b['x']), -1)
        pred, detail = model(b['x'], ps, False)
        target, mask = ba.local_targets(b['y'], b['mask'], ps, original, local)
        globals_.append(pred.cpu().numpy())
        for i, p in enumerate(EVAL_PREFIXES):
            predictions[p].append(detail[:, i].cpu().numpy()); targets[p].append(target[:, i].cpu().numpy()); masks[p].append(mask[:, i].cpu().numpy())
    return np.concatenate(globals_), {p: dict(pred=np.concatenate(predictions[p]), y=np.concatenate(targets[p]), mask=np.concatenate(masks[p])) for p in EVAL_PREFIXES}


def compare_groups(errors, inventory, a, b):
    result = {}
    for task in ('global', 'trained', 'held'):
        keys = GLOBAL_METRICS if task == 'global' else pr.METRICS
        result[task] = {k: ab.pair_groups(errors[a+'/'+task][k], errors[b+'/'+task][k], inventory) for k in keys}
    return result


def screen_contrast(all_errors, inventories, candidate, reference, gain, config):
    """Prespecified engineering margins, using paired differences against scaled reference."""
    records = []; passed = True
    for split in ('test', 'cross_research'):
        for kind in ('best', 'last'):
            e = all_errors[split][kind]
            for seed in (42, 43):
                a, b = candidate+f'_s{seed}', reference+f'_s{seed}'
                # Global retention applies to all families, not just a weighted aggregate.
                checks = [('held', 'primary', 1-gain)] + [('global', k, 1+(config['global_retention'] if k=='primary' else config['family_retention'])) for k in GLOBAL_METRICS]
                for task, metric, factor in checks:
                    av = np.asarray(e[a+'/'+task][metric], dtype=float); bv = np.asarray(e[b+'/'+task][metric], dtype=float)
                    ci = ab.pair_groups(av, factor*bv, inventories[split])['all']
                    ok = ci['supported'] and ci['high'] is not None and ci['high'] <= 0
                    passed &= ok
                    records.append(dict(dataset=split, checkpoint=kind, seed=seed, task=task, metric=metric,
                        allowed_reference_factor=factor, candidate_mean=float(av.mean()), reference_mean=float(bv.mean()), margin_interval=ci, passed=ok))
    return dict(candidate=candidate, reference=reference, required_held_gain=gain, passed=bool(passed), checks=records)


def route_decision(meta, report, all_errors, inventories):
    low, high = meta['widths']; base = f'w{low}_endpoint'; small = f'w{low}_joint'
    wide = f'w{high}_endpoint'; wide_joint = f'w{high}_joint'; cfg = meta['decision']; screens = {}
    for candidate in (small, wide, wide_joint):
        screens[candidate] = screen_contrast(all_errors, inventories, candidate, base, cfg['local_improvement'], cfg)
    eligible = {k for k, row in screens.items() if row['passed']}; upgrades = {}; chosen = base
    if small in eligible: chosen = small
    elif wide in eligible: chosen = wide
    elif wide_joint in eligible: chosen = wide_joint
    # Prefer lower complexity unless additional held-local benefit is supported.
    if chosen == small:
        for candidate in (wide, wide_joint):
            if candidate not in eligible: continue
            key = candidate+'_minus_'+chosen
            upgrades[key] = screen_contrast(all_errors, inventories, candidate, chosen, cfg['upgrade_gain'], cfg)
            if upgrades[key]['passed']: chosen = candidate
    elif chosen == wide and wide_joint in eligible:
        key = wide_joint+'_minus_'+wide
        upgrades[key] = screen_contrast(all_errors, inventories, wide_joint, wide, cfg['upgrade_gain'], cfg)
        if upgrades[key]['passed']: chosen = wide_joint
    return dict(recommended_route=chosen, retained_seeds=[42, 43], screens=screens, upgrades=upgrades,
        status='retain_baseline_and_stop' if chosen == base else 'candidate_for_usage_validation',
        automatic_promotion=False, next_stage='Stop this architecture/supervision search. If a candidate passes, next validate its use as an observed-state interface; do not automatically add variants.',
        limits='Exploratory screen on reused research sets, uncorrected intervals. Margins are declared engineering tradeoffs, not scientific optimality or trading validation.')


def evaluate(meta, out, device='cuda'):
    verify_code(meta); lock = lock_selection(meta, out)
    stats = read_json(out/'cache/statistics.json'); local = read_json(out/'cache/local_scales.json'); val = load_data(meta, 'val')
    replay = {}
    # Every epoch is selected before any held-position or research outcome is read.
    for job in meta['experiments']:
        model = model_for(meta, out, job, device)
        for kind in ('best', 'last'):
            ck = torch.load(out/job['name']/f'{kind}.pt', map_location='cpu', weights_only=True); model.load_state_dict(ck['model'])
            actual = validation(model, val, stats, local, meta, job, device)
            expected = ck['validation'] if kind == 'best' else ck['history'][-1]['validation']
            if any(not np.isclose(actual[k], v, atol=1e-6, rtol=2e-5) for k, v in expected.items()): raise ValueError('Selected validation replay mismatch')
            replay[job['name']+'/'+kind] = dict(matched=True, actual=actual, expected=expected)
        del model, ck
    atomic_json(replay, out/'validation_replay.json')
    report = dict(schema=SCHEMA, manifest=meta, trials=lock['trials'], datasets={}); all_errors = {}; inventories = {}; examples = []
    for split in ('test', 'cross_research'):
        data = load_data(meta, split); inventory = read_json(Path(meta['original_source'])/f'cache/{split}_inventory.json')
        inventories[split] = inventory; atomic_json(inventory, out/f'{split}_inventory.json')
        all_errors[split] = {}; report['datasets'][split] = {}
        ids = list(np.linspace(0, len(data['x'])-1, min(4, len(data['x'])), dtype=int))
        cards = {i: dict(dataset=split, key=inventory[i]['key'], end=inventory[i]['end'], series={}) for i in ids}
        for kind in ('best', 'last'):
            scores = {}; errors = {}
            for job in meta['experiments']:
                name = job['name']; model = model_for(meta, out, job, device)
                ck = torch.load(out/name/f'{kind}.pt', map_location='cpu', weights_only=True); model.load_state_dict(ck['model'])
                pred, detail = predict(model, data, stats, local, meta['micro'], device)
                scores[name+'/global'], errors[name+'/global'] = ab.measure(pred, data, stats)
                for p in EVAL_PREFIXES:
                    d = detail[p]; scores[f'{name}/p{p}'], errors[f'{name}/p{p}'] = pr.measure(d['pred'], d['y'], d['mask'], local)
                for group, positions in [('trained', ba.VAL_PREFIXES), ('held', ba.HELD_PREFIXES)]:
                    errors[name+'/'+group] = {k: np.mean([errors[f'{name}/p{p}'][k] for p in positions], axis=0) for k in pr.METRICS}
                    scores[name+'/'+group] = dict(metrics={k: float(v.mean()) for k, v in errors[name+'/'+group].items()},
                        positions=list(positions), aggregation='positions within each original window first')
                if kind == 'best':
                    for i in ids:
                        cards[i]['series'][name] = dict(global_prediction=(pred[i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist(),
                            global_truth=(data['y'][i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist(), global_mask=data['mask'][i].tolist(),
                            held={str(p):dict(prediction=(detail[p]['pred'][i]*np.array(local['scale'])+np.array(local['mean'])).tolist(),
                                truth=(detail[p]['y'][i]*np.array(local['scale'])+np.array(local['mean'])).tolist(), mask=detail[p]['mask'][i].tolist()) for p in ba.HELD_PREFIXES})
                del model, ck, pred, detail
                progress(f'Evaluated {split}/{kind}/{name}')
            low, high = meta['widths']; pairs = []
            for seed in (42, 43):
                a, b, c, d = [f'w{w}_{mode}_s{seed}' for w, mode in ((low,'endpoint'),(low,'joint'),(high,'endpoint'),(high,'joint'))]
                pairs.extend(((b,a),(c,a),(d,c),(d,b),(d,a)))
            paired = {a+'_minus_'+b: compare_groups(errors, inventory, a, b) for a, b in pairs}
            report['datasets'][split][kind] = dict(scores=scores, paired=paired)
            all_errors[split][kind] = errors
            atomic_json({n: {k: v.tolist() for k, v in row.items()} for n, row in errors.items()}, out/f'{split}_{kind}_errors.json')
        examples.extend(cards.values())
    atomic_json(report, out/'alignment_metrics.json'); atomic_json(examples, out/'examples.json')
    decision = route_decision(meta, report, all_errors, inventories); atomic_json(decision, out/'decision.json')
    lines = ['# Causal per-bar alignment: fixed2x2 matrix', '', f'Recommended research route: {decision["recommended_route"]}; status: {decision["status"]}',
             'Both seeds and best/last retained; no automatic promotion.', '']
    for split, kinds in report['datasets'].items():
        for kind, rows in kinds.items():
            lines += [f'## {split}/{kind}', '', '| variant | global primary | trained local | held local |', '|---|---:|---:|---:|']
            for job in meta['experiments']:
                name=job['name'];lines.append('| '+name+' | '+' | '.join(f'{rows["scores"][name+"/"+task]["metrics"]["primary"]:.6f}' for task in ('global','trained','held'))+' |')
    (out/'summary.md').write_text('\n'.join(lines))
    return report


def preflight(meta, out, device='cuda'):
    verify_cache(meta, out); data = load_data(meta, 'val')
    b = {k: torch.tensor(np.asarray(v[:2]), device=device) for k, v in data.items()}
    ps = torch.tensor(ba.VAL_PREFIXES, device=device)[None].expand(len(b['x']), -1)
    stats = read_json(out/'cache/statistics.json'); local = read_json(out/'cache/local_scales.json')
    target, mask = ba.local_targets(b['y'], b['mask'], ps, stats, local); records = []
    for width in meta['widths']:
        job = next(j for j in meta['experiments'] if j['width']==width and not j['joint'] and j['seed']==42)
        model = model_for(meta, out, job, device).eval()
        with torch.no_grad():
            full = model.core.encoder(b['x']); diffs = {}
            for p in EVAL_PREFIXES:
                truncated = model.core.encoder(b['x'][:, :p])[:, -1]
                diff = float((truncated-full[:, p-1]).abs().max()); diffs[str(p)] = diff
                if not torch.allclose(truncated, full[:, p-1], atol=1e-5, rtol=5e-5):
                    raise ValueError('Encoder failed strict causal-prefix preflight')
        model.zero_grad(set_to_none=True)
        pred, detail = model(b['x'], ps, False)
        ba.local_rows(detail, target, mask, True)['primary'].mean().backward()
        if any(p.grad is not None for p in model.core.parameters()): raise ValueError('Detached local head modified encoder gradient')
        model.zero_grad(set_to_none=True)
        old = ar.error_rows(model.core(b['x']), b['y'], b['mask'], stats, True)['primary'].mean(); old.backward()
        grads = {k: p.grad.clone() for k, p in model.core.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True); pred, detail = model(b['x'], ps, False)
        loss = ar.error_rows(pred, b['y'], b['mask'], stats, True)['primary'].mean()+ba.LOCAL_WEIGHT*ba.local_rows(detail, target, mask, True)['primary'].mean()
        loss.backward()
        if any(not torch.equal(p.grad, grads[k]) for k, p in model.core.named_parameters() if k in grads):
            raise ValueError('Endpoint control core gradients differ from original objective')
        model.zero_grad(set_to_none=True); _, detail = model(b['x'], ps, True)
        ba.local_rows(detail, target, mask, True)['primary'].mean().backward()
        if not any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.core.encoder.parameters()):
            raise ValueError('Joint local objective did not reach encoder')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite preflight gradients')
        records.append(dict(width=width, causal_max_abs=diffs, detached_gradient_zero=True, original_core_gradient_exact=True, joint_gradient_reaches_encoder=True))
        del model, grads
    result = dict(status='passed', optimizer_updates=0, checks=records, held_prefixes=list(ba.HELD_PREFIXES))
    atomic_json(result, out/'preflight.json'); return result


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);path=out/job['name'];path.mkdir(exist_ok=True);log=(path/'run.log').open('a')
                try:proc=subprocess.Popen([sys.executable,'-m','obson.babel.bar_alignment_benchmark','worker','--out',str(out),'--name',job['name']],stdout=log,stderr=subprocess.STDOUT)
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
    if not torch.cuda.is_available(): raise ValueError('Formal alignment training runs on AutoDL CUDA')
    if args.action == 'worker': worker(out, args.name); return
    if not args.source: parser.error('--source required')
    if not 1 <= args.jobs <= 2: raise ValueError('Use one or two concurrent training workers')
    source = Path(args.source).resolve(); progress('Verifying immutable shared/sampling lineage')
    identity = source_identity(source); meta = make_manifest(source, identity)
    if meta['windows_per_epoch'] != 4789 or meta['config'] != pt.CONFIG:
        raise ValueError('Expected original512 configuration and fixed sampling budget')
    for dep in (source, Path(meta['sampling_source']), Path(meta['original_source']), Path(identity['manifest']['source'])):
        if out == dep or out in dep.parents or dep in out.parents: raise ValueError('Use a separate output directory')
    runtime = read_json(source/'runtime.json')
    if runtime['torch'] != str(torch.__version__) or runtime['numpy'] != np.__version__: raise ValueError('Restore source Torch/NumPy environment')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json') != meta: raise ValueError('Alignment configuration changed')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty output without matching manifest')
    out.mkdir(parents=True, exist_ok=True); atomic_json(meta, out/'manifest.json'); (out/'completion.json').unlink(missing_ok=True)
    atomic_json(dict(torch=str(torch.__version__), numpy=np.__version__, gpu=torch.cuda.get_device_name(), jobs=args.jobs,
        source_runtime=runtime, started_unix=time.time()), out/'runtime.json')
    prepare(meta, out); preflight(meta, out)
    if args.action == 'preflight': return
    if args.action == 'all': run_jobs(out, args.jobs)
    evaluate(meta, out)
    if source_identity(source) != identity: raise ValueError('Source modified during alignment experiment')
    verify_cache(meta, out)
    files = {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.md') and p.name != 'completion.json'}
    files['cache/index.json'] = sha256(out/'cache/index.json')
    summaries = [read_json(out/j['name']/'training_summary.json') for j in meta['experiments']]
    atomic_json(dict(status='complete', cells=len(meta['experiments']), source_unchanged=True, automatic_promotion=False,
        encoder_steps=sum(s['encoder_steps'] for s in summaries), head_steps=sum(s['head_steps'] for s in summaries), files=files), out/'completion.json')
    progress('Fixed2x2 alignment complete; see decision.json for the retain/stop result')


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
