"""Frozen dual-state representation benchmark with independent, matched readout heads."""
import argparse
import copy
import hashlib
import json
import math
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .ae_diagnostics import metrics, summarize
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .autoregressive import probe
from .history_autoencoder import positions, to_ohlc
from .large_history import HierWindows, move, write_global_review
from .memory_benchmark import regression_metrics, fit_readout
from .progress import progress
from .reconstruction_fusion import (ReconstructionFusion, history_values, recent_loss,
                                   load_long, CachedWindows, validate as old_validate)
from .stage_audit import load_data

MODES = ('concat', 'compress', 'long', 'short')
SPLITS = ('train', 'val', 'test')
TASKS = ('recent', 'history_loss', 'history_close_bps', 'structure_0', 'structure_1', 'structure_2')


def sha256(path):
    """Stream large cache files instead of allocating their entire contents."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(directory, files):
    if not files:
        raise ValueError('Empty cache index')
    for name, digest in files.items():
        path = (directory / name).resolve()
        if path.parent != directory.resolve() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Cache fingerprint mismatch: {name}')


class PositionalDecoder(nn.Module):
    def __init__(self, width, length, outputs, layers=2):
        super().__init__()
        self.width, self.length = width, length
        layer = nn.TransformerEncoderLayer(width, 8, width * 4, dropout=.1,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, outputs))

    def forward(self, h):
        # Queries are fixed positions, with no teacher-forced bars, masks or raw-state bypass.
        return self.output(self.layers(h[:, None] + positions(self.length, self.width, h.device, h.dtype)))


class DualState(nn.Module):
    def __init__(self, mode, stats, width=512, decoder_width=256, blocks=16, window=128):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown dual-state mode')
        self.mode, self.width, self.blocks, self.window = mode, width, blocks, window
        self.dimensions = 2 * width if mode == 'concat' else width
        for key in ('long_mean', 'long_scale', 'short_mean', 'short_scale', 'structure_mean', 'structure_scale'):
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float32))
        # Construct common heads first, so their Transformer/MLP weights match for a given seed.
        self.recent_decoder = PositionalDecoder(decoder_width, 64, 7)
        # One query per historical block; predict its 128 bars jointly. No 2048^2 attention.
        self.history_decoder = PositionalDecoder(decoder_width, blocks, window * 7)
        self.structure_output = nn.Sequential(nn.LayerNorm(decoder_width), nn.GELU(), nn.Linear(decoder_width, 3))
        self.recent_input = nn.Linear(self.dimensions, decoder_width)
        self.history_input = nn.Linear(self.dimensions, decoder_width)
        self.structure_input = nn.Linear(self.dimensions, decoder_width)
        if mode == 'compress':
            self.project = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                         nn.Linear(width, width), nn.LayerNorm(width))

    def encode(self, long, short):
        if long.shape != short.shape or long.ndim != 2 or long.shape[1] != self.width:
            raise ValueError('Aligned frozen long/short vectors required')
        m, s = (long - self.long_mean) / self.long_scale, (short - self.short_mean) / self.short_scale
        if self.mode == 'long':
            return m
        if self.mode == 'short':
            return s
        pair = torch.cat((m, s), dim=-1)
        return self.project(pair) if self.mode == 'compress' else pair

    def decode(self, z):
        recent = self.recent_decoder(self.recent_input(z))
        history = self.history_decoder(self.history_input(z)).reshape(-1, self.blocks, self.window, 7)
        positive = lambda x: torch.cat((x[..., :2], nn.functional.softplus(x[..., 2:])), -1)
        return dict(recent=positive(recent), history=positive(history),
                    structure=self.structure_output(self.structure_input(z)))

    def forward(self, long, short):
        z = self.encode(long, short)
        return dict(z=z, **self.decode(z))


class DualWindows(Dataset):
    def __init__(self, cache, supplemental, split):
        self.arrays = {k: np.load(cache / f'{split}_{k}.npy', mmap_mode='r', allow_pickle=False)
                       for k in ('long', 'short', 'y', 'mask', 'valid', 'recent_y', 'recent_mask')}
        self.arrays['structure'] = np.load(supplemental / f'{split}_structure.npy', mmap_mode='r', allow_pickle=False)
        n = len(self.arrays['long'])
        if any(len(v) != n for v in self.arrays.values()):
            raise ValueError('Unaligned dual-state targets')

    def __len__(self):
        return len(self.arrays['long'])

    def __getitem__(self, i):
        return {k: np.array(v[i], copy=True) for k, v in self.arrays.items()}


def parts_from_output(model, r, b, normalizers):
    recent = recent_loss(r['recent'], b['recent_y'], b['recent_mask'])
    history, close = history_values(r['history'], b['y'], b['mask'], b['valid'])
    target = (b['structure'] - model.structure_mean) / model.structure_scale
    structure = (r['structure'] - target).square()
    # Equal weight for three task families. Denominators come from train-only constant predictors.
    total = recent / normalizers['recent'] + history / normalizers['history'] + structure.mean(1)
    return dict(total=total, recent=recent, history_loss=history, history_close_bps=close,
                **{f'structure_{j}': structure[:, j] for j in range(3)})


@torch.no_grad()
def validate(model, dataset, batch, device, normalizers):
    model.eval()
    sums, count = {}, 0
    for data in DataLoader(dataset, batch_size=batch):
        b = move(data, device)
        parts = parts_from_output(model, model(b['long'], b['short']), b, normalizers)
        count += len(b['long'])
        for key, value in parts.items():
            sums[key] = sums.get(key, 0.) + float(value.sum())
    result = {k: v / count for k, v in sums.items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite dual-state validation')
    return result


def selection_key(score, reference, tolerance=.02, recent_tolerance=.05):
    """Qualified candidates first; otherwise explicitly retain the best worst-case compromise."""
    ratios = {k: score[k] / max(reference[k], 1e-8) for k in TASKS}
    if not all(math.isfinite(v) and v >= 0 for v in ratios.values()):
        raise ValueError('Invalid selection metrics')
    allowed = {k: 1 + (recent_tolerance if k == 'recent' else tolerance) for k in TASKS}
    passed = all(ratios[k] <= allowed[k] for k in TASKS)
    worst = max(ratios[k] / allowed[k] for k in TASKS)
    return (0 if passed else 1, worst, sum(ratios.values()) / len(ratios)), ratios


def experiment_jobs(seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('At least one unique seed required')
    # Launch paired candidates first; single-branch controls use the first seed only.
    return [dict(name=f'{m}_s{s}', mode=m, seed=s) for s in seeds for m in ('concat', 'compress')] + [
        dict(name=f'{m}_s{seeds[0]}', mode=m, seed=seeds[0]) for m in ('long', 'short')]


def source_identity(source, long_run):
    meta = json.loads((source / 'manifest.json').read_text())
    index = json.loads((source / 'target_cache/index.json').read_text())
    if index['identity'] != meta['sources'] or meta['sources']['long_best'] != sha256(long_run / 'joint/best.pt'):
        raise ValueError('Reconstruction cache and long checkpoint identities differ')
    verify_files(source / 'target_cache', index['files'])
    artifacts = json.loads((source / 'artifacts.json').read_text())
    if artifacts['short'] != sha256(source / 'short/best.pt'):
        raise ValueError('Reference short decoder checkpoint changed')
    for split in SPLITS:
        for key in ('long', 'short', 'y', 'mask', 'valid', 'teacher', 'recent_y', 'recent_mask', 'labels', 'keys'):
            if f'{split}_{key}.npy' not in index['files']:
                raise ValueError(f'Missing source cache index entry: {split}/{key}')
    return dict(reconstruction_manifest=sha256(source / 'manifest.json'),
                cache_index=sha256(source / 'target_cache/index.json'),
                long_best=sha256(long_run / 'joint/best.pt'), short_decoder=artifacts['short'],
                long_manifest=sha256(long_run / 'manifest.json'))


@torch.no_grad()
def prepare(source, long_run, root, out, identity, batch, device):
    extra = out / 'targets'
    extra.mkdir(exist_ok=True)
    index_path = extra / 'index.json'
    # Verify the raw data fingerprint on every parent run, including cache reuse.
    ref, series, encoded, sets = load_data(root, long_run)
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index['identity'] != identity:
            raise ValueError('Supplemental targets source changed')
        verify_files(extra, index['files'])
        progress('Verified shared dual-state targets and source data')
        return
    cache = source / 'target_cache'
    dummy = [np.zeros((len(range(127, len(s.frame), 128)), 1), np.float32) for s in series]
    coverage, files = {}, {}
    for split, base in zip(SPLITS, sets):
        ds = HierWindows(base, dummy, ref['boundaries'], split)
        keys = np.load(cache / f'{split}_keys.npy', allow_pickle=False)
        if not np.array_equal(keys, np.array([(i, end) for i, end, _ in ds.items])):
            raise ValueError('Structure targets and frozen endpoints differ')
        targets, inventory = [], []
        for data in DataLoader(ds, batch_size=batch):
            targets.extend(data['targets'].numpy())
            for i, end, anchor, blocks in zip(data['series'].tolist(), data['row'].tolist(), data['anchor'].tolist(), data['blocks'].tolist()):
                s = series[i]
                inventory.append(dict(source=s.key, row=end, end=str(s.frame.datetime.iloc[end]), blocks=blocks,
                                      anchor=anchor, recent_anchor=float(s.frame.close.iloc[end - 64])))
        with (extra / f'{split}_structure.npy').open('wb') as f:
            np.save(f, np.asarray(targets, np.float32), allow_pickle=False)
        atomic_json(inventory, extra / f'{split}_inventory.json')
        coverage[split] = dict(windows=len(ds), blocks={str(k): sum(x['blocks'] == k for x in inventory) for k in range(4, 17)})
    tr = DualWindows(cache, extra, 'train')
    stats = {}
    for name in ('long', 'short', 'structure'):
        x = tr.arrays[name]
        stats[f'{name}_mean'] = x.mean(0).tolist()
        stats[f'{name}_scale'] = x.std(0).clip(.01).tolist()
    # Constant predictors are fitted exclusively on training targets; no test/validation fitting.
    constant_recent = np.asarray(tr.arrays['recent_y']).mean(0)
    valid = tr.arrays['valid']
    constant_history = np.asarray(tr.arrays['y'])[valid].mean(axis=(0, 1))
    sums = dict(recent=0., history=0.)
    for data in DataLoader(tr, batch_size=batch):
        b = move(data, device)
        recent = torch.as_tensor(constant_recent, device=device).expand_as(b['recent_y'])
        hist = torch.as_tensor(constant_history, device=device).expand_as(b['y'])
        sums['recent'] += float(recent_loss(recent, b['recent_y'], b['recent_mask']).sum())
        sums['history'] += float(next(history_values(hist, b['y'], b['mask'], b['valid'])).sum())
    normalizers = {k: max(v / len(tr), 1e-6) for k, v in sums.items()}
    long_model, ck = load_long(long_run / 'joint/best.pt', device)
    old_mean = torch.tensor(ck['metadata']['target_mean'], device=device)
    old_scale = torch.tensor(ck['metadata']['target_scale'], device=device)
    va = DualWindows(cache, extra, 'val')
    reference = dict(history_loss=0., history_close_bps=0., structure_0=0., structure_1=0., structure_2=0.)
    teacher = np.load(cache / 'val_teacher.npy', mmap_mode='r', allow_pickle=False)
    offset = 0
    target_scale = torch.tensor(stats['structure_scale'], device=device)
    for data in DataLoader(va, batch_size=batch):
        b = move(data, device); n = len(b['long'])
        pred = torch.tensor(np.array(teacher[offset:offset+n]), device=device)
        hist, close = history_values(pred, b['y'], b['mask'], b['valid'])
        reference['history_loss'] += float(hist.sum()); reference['history_close_bps'] += float(close.sum())
        prediction = long_model.structure(b['long']) * old_scale + old_mean
        errors = ((prediction - b['structure']) / target_scale).square().sum(0)
        for j in range(3): reference[f'structure_{j}'] += float(errors[j])
        offset += n
    reference = {k: v / len(va) for k, v in reference.items()}
    previous = ReconstructionFusion('short').to(device)
    previous.load_state_dict(torch.load(source / 'short/best.pt', map_location='cpu', weights_only=True)['model'])
    reference['recent'] = old_validate(previous, CachedWindows(cache, 'val'), batch, device)['recent']
    atomic_json(dict(stats=stats, normalizers=normalizers, reference=reference,
                     reference_definition='Validation: previous selected short recent decoder and original long history/structure heads. Structure errors use new train-only target scales.'), extra / 'statistics.json')
    atomic_json(coverage, out / 'coverage.json')
    for path in extra.iterdir():
        if path.name != 'index.json': files[path.name] = sha256(path)
    atomic_json(dict(identity=identity, files=files), index_path)
    progress('Shared targets ready; encoders will not run in training workers')


def publish(state, path):
    atomic_save(state, path / 'last.pt')
    atomic_save(dict(metadata=state['metadata'], epoch=state['best_epoch'], model=state['best_model'],
                     validation=state['best_validation'], selection_key=state['best_key']), path / 'best.pt')
    tmp = path / 'history.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in state['history']))
    tmp.replace(path / 'history.jsonl')
    atomic_json(dict(epoch=state['epoch'], best_epoch=state['best_epoch'],
                     qualified=state['best_key'][0] == 0), path / 'progress.json')


def worker(out, name, device='cuda'):
    meta = json.loads((out / 'manifest.json').read_text())
    job = next(j for j in meta['experiments'] if j['name'] == name)
    stats = json.loads((out / 'targets/statistics.json').read_text())
    seed = job['seed']; torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = DualState(job['mode'], stats['stats']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=meta['lr'], weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, meta['epochs'], eta_min=meta['lr'] * .1)
    cache = Path(meta['source']) / 'target_cache'
    tr, va = [DualWindows(cache, out / 'targets', s) for s in ('train', 'val')]
    effective, micro = meta['batch'], meta['micro']
    path = out / name; path.mkdir(exist_ok=True)
    metadata = dict(manifest=meta, job=job, target_index=sha256(out / 'targets/index.json'))
    if (path / 'last.pt').exists():
        state = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata: raise ValueError('Resume metadata mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['schedule']); restore_rng(state['rng'])
    else:
        score = validate(model, va, micro, device, stats['normalizers'])
        key, _ = selection_key(score, stats['reference'])
        state = dict(metadata=metadata, epoch=0, best_epoch=0, best_key=key, best_validation=score,
                     best_model=copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), history=[])
    for epoch in range(state['epoch'] + 1, meta['epochs'] + 1):
        start = time.perf_counter(); model.train(); opt.zero_grad(set_to_none=True)
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        totals, seen = {}, 0
        loader = DataLoader(tr, batch_size=micro, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))
        learning_rate = opt.param_groups[0]['lr']
        for step, data in enumerate(loader):
            b = move(data, device); n = len(b['long'])
            values = parts_from_output(model, model(b['long'], b['short']), b, stats['normalizers'])
            if not torch.isfinite(values['total']).all(): raise ValueError('Nonfinite training loss')
            group_start = (step // (effective // micro)) * effective
            (values['total'].sum() / min(effective, len(tr) - group_start)).backward()
            if (step + 1) % (effective // micro) == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True)
            for k, v in values.items(): totals[k] = totals.get(k, 0.) + float(v.detach().sum())
            seen += n
        score = validate(model, va, micro, device, stats['normalizers'])
        key, ratios = selection_key(score, stats['reference'])
        if key < tuple(state['best_key']):
            state.update(best_key=key, best_epoch=epoch, best_validation=score,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        schedule.step()
        state['history'].append(dict(epoch=epoch, train={k: v / seen for k, v in totals.items()}, validation=score,
                                     ratios=ratios, qualified=key[0] == 0, learning_rate=learning_rate,
                                     seconds=time.perf_counter() - start,
                                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else 0.))
        state.update(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(), schedule=schedule.state_dict(), rng=rng_state())
        publish(state, path)
        progress(f'{name} epoch={epoch}/{meta["epochs"]} recent={score["recent"]:.4f} history={score["history_loss"]:.4f} structure={[round(score[f"structure_{j}"],4) for j in range(3)]} qualified={key[0]==0} best={state["best_epoch"]}')
    # Completed resumes repair derivative reports without taking an optimizer step.
    publish(state, path)


def run_jobs(out, jobs):
    pending = list(json.loads((out / 'manifest.json').read_text())['experiments'])
    active, previous_report = {}, {}
    def stop(signum, frame): raise SystemExit(128 + signum)
    old_handler = signal.signal(signal.SIGTERM, stop)
    try:
        while pending or active:
            while pending and len(active) < jobs:
                job = pending.pop(0); name = job['name']; (out / name).mkdir(exist_ok=True)
                log = (out / name / 'run.log').open('a')
                try:
                    process = subprocess.Popen([sys.executable, '-m', 'obson.babel.dual_state', 'worker',
                                                '--out', str(out), '--name', name], stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close(); raise
                active[name] = (process, log); progress(f'Started {name}, pid={process.pid}')
            for name, (process, log) in list(active.items()):
                code = process.poll(); path = out / name / 'progress.json'
                if path.exists():
                    try: status = json.loads(path.read_text())
                    except (ValueError, OSError): status = None
                    if status and status != previous_report.get(name):
                        progress(f'{name}: {status}'); previous_report[name] = status
                if code is not None:
                    log.close(); del active[name]
                    if code: raise RuntimeError(f'{name} failed (exit {code}); see {out/name/"run.log"}')
                    progress(f'Completed {name}')
            if active: time.sleep(1)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            log.close()
        signal.signal(signal.SIGTERM, old_handler)


def fit_latent_readout(features, longs):
    """Train-only affine ridge recovery; choose alpha on validation latent MSE, never test."""
    mean, scale = features[0].mean(0), features[0].std(0).clip(.01)
    ym, ys = longs[0].mean(0), longs[0].std(0).clip(.01)
    design = [np.column_stack(((x - mean) / scale, np.ones(len(x)))).astype(np.float64) for x in features]
    target = (longs[0] - ym) / ys
    lhs, rhs = design[0].T @ design[0], design[0].T @ target
    best = None
    for alpha in (1., 10., 100.):
        penalty = np.eye(lhs.shape[0]) * alpha; penalty[-1, -1] = 0
        w = np.linalg.solve(lhs + penalty, rhs)
        score = float(np.square(design[1] @ w - (longs[1] - ym) / ys).mean())
        if best is None or score < best[0]: best = score, alpha, w
    return [(x @ best[2] * ys + ym).astype(np.float32) for x in design], dict(alpha=best[1], validation_standardized_mse=best[0])


@torch.no_grad()
def evaluate(out, long_run, batch, device):
    meta = json.loads((out / 'manifest.json').read_text()); cache = Path(meta['source']) / 'target_cache'
    stats = json.loads((out / 'targets/statistics.json').read_text())
    ds = {s: DualWindows(cache, out / 'targets', s) for s in SPLITS}
    labels = {s: np.load(cache / f'{s}_labels.npy', allow_pickle=False) for s in SPLITS}
    inventory = json.loads((out / 'targets/test_inventory.json').read_text())
    long_model, ck = load_long(long_run / 'joint/best.pt', device)
    old_mean = torch.tensor(ck['metadata']['target_mean'], device=device)
    old_scale = torch.tensor(ck['metadata']['target_scale'], device=device)
    report = dict(schema='babel-dual-state-evaluation-v1', variants={},
                  scope='Frozen encoder states at complete 128-bar endpoints; reconstruction of observed history, not forecasting. Reused research test period.',
                  selection_reference=stats['reference'],
                  historical_decoder='New matched 16-query decoder emits all 128 bars per block; results are not interchangeable with the original frozen decoder.',
                  compatibility='Original frozen decoder/head evaluated after a train-fitted, validation-selected affine recovery of original long coordinates; this is adapted compatibility, not a direct raw-vector swap.')
    # Independent frozen references make every report self-contained.
    previous = ReconstructionFusion('short').to(device)
    previous.load_state_dict(torch.load(Path(meta['source'])/'short/best.pt', map_location='cpu', weights_only=True)['model'])
    reference_records, reference_structure, short_records = {}, [], {}
    total_loss = total_close = 0.
    previous.eval()
    teacher = np.load(cache/'test_teacher.npy', mmap_mode='r', allow_pickle=False)
    offset = 0
    for data in DataLoader(ds['test'], batch_size=batch):
        b = move(data,device); n = len(b['long'])
        pred = torch.tensor(np.array(teacher[offset:offset+n]),device=device)
        hist,close = history_values(pred,b['y'],b['mask'],b['valid'])
        total_loss += float(hist.sum()); total_close += float(close.sum())
        reference_structure.extend((long_model.structure(b['long'])*old_scale+old_mean).cpu().numpy())
        recent = previous(b['long'],b['short'])['recent'].cpu().numpy()
        for j in range(n):
            valid = data['valid'][j].numpy()
            reference_records.setdefault('history',[]).append(metrics(data['y'][j].numpy()[valid].reshape(-1,7),pred[j].cpu().numpy()[valid].reshape(-1,7)))
            for length in (16,32,64):
                short_records.setdefault(f'recent/{length}',[]).append(metrics(data['recent_y'][j].numpy()[-length:],recent[j][-length:]))
        offset += n
    report['frozen_references'] = dict(original_long=dict(history_loss=total_loss/offset, history_close_bps=total_close/offset,
        reconstruction=summarize(reference_records), structure=regression_metrics(np.asarray(ds['test'].arrays['structure']),np.asarray(reference_structure))),
        previous_short=dict(reconstruction=summarize(short_records)))
    del previous
    for job in meta['experiments']:
        name = job['name']; path = out / name; progress(f'Evaluating {name}')
        best = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
        if best['metadata']['manifest'] != meta: raise ValueError('Selected checkpoint metadata mismatch')
        model = DualState(job['mode'], stats['stats']).to(device); model.load_state_dict(best['model']); model.eval()
        features = []
        for split in SPLITS:
            zs = []
            for data in DataLoader(ds[split], batch_size=batch):
                zs.append(model.encode(data['long'].to(device), data['short'].to(device)).cpu().numpy())
            features.append(np.concatenate(zs))
        score = validate(model, ds['val'], batch, device, stats['normalizers'])
        key, ratios = selection_key(score, stats['reference'])
        if not np.allclose([score[k] for k in TASKS], [best['validation'][k] for k in TASKS], rtol=1e-4, atol=1e-6):
            raise ValueError('Selected checkpoint validation does not reproduce')
        records, structures, recent_cards, history_cards, per_window = {}, [], [], [], []
        chosen = set(np.linspace(0, len(ds['test'])-1, 6, dtype=int)); offset = 0
        for data in DataLoader(ds['test'], batch_size=batch):
            b = move(data, device); result = model(b['long'], b['short'])
            structures.extend((result['structure'] * model.structure_scale + model.structure_mean).cpu().numpy())
            recent, historical = result['recent'].cpu().numpy(), result['history'].cpu().numpy()
            for j in range(len(recent)):
                item = inventory[offset+j]; valid = data['valid'][j].numpy()
                true_recent, true_history = data['recent_y'][j].numpy(), data['y'][j].numpy()[valid]
                pred_history = historical[j][valid]
                row = dict(source=item['source'], row=item['row'], blocks=item['blocks'],
                           structure_truth=data['structure'][j].tolist(), structure_prediction=structures[-len(recent)+j].tolist())
                for n in (16, 32, 64):
                    value = metrics(true_recent[-n:], recent[j][-n:]); records.setdefault(f'recent/{n}', []).append(value)
                    if n == 64: row['recent64'] = value
                value = metrics(true_history.reshape(-1, 7), pred_history.reshape(-1, 7))
                row['history'] = value
                anchor_shift = 100 * np.log(item['recent_anchor'] / item['anchor'])
                consistency = float(np.abs(pred_history[-1, -64:, 1] - recent[j, :, 1] - anchor_shift).mean() * 100)
                row['head_consistency_close_mae_bps'] = consistency
                records.setdefault('head_consistency', []).append(dict(close_mae_bps=consistency))
                for group in ('history', f'blocks/{item["blocks"]}', 'period/'+item['source'].split('/')[1]):
                    records.setdefault(group, []).append(value)
                per_window.append(row)
                if offset+j in chosen:
                    recent_cards.append(dict(source=item['source'], end=item['end'], blocks=1,
                        truth_ohlc=to_ohlc(true_recent, item['recent_anchor']).tolist(),
                        reconstructed_ohlc=to_ohlc(recent[j], item['recent_anchor']).tolist()))
                    history_cards.append(dict(source=item['source'], end=item['end'], blocks=item['blocks'],
                        truth_ohlc=to_ohlc(true_history.reshape(-1,7), item['anchor']).tolist(),
                        reconstructed_ohlc=to_ohlc(pred_history.reshape(-1,7), item['anchor']).tolist()))
            offset += len(recent)
        truth = np.asarray(ds['test'].arrays['structure'])
        result = dict(mode=job['mode'], seed=job['seed'], dimensions=model.dimensions,
                      trainable_parameters=sum(p.numel() for p in model.parameters()),
                      selected_epoch=best['epoch'], checkpoint_sha256=sha256(path/'best.pt'),
                      validation=score, validation_ratios=ratios, qualified=key[0] == 0,
                      test_objectives=validate(model, ds['test'], batch, device, stats['normalizers']),
                      reconstruction=summarize(records), structure=regression_metrics(truth, np.asarray(structures)),
                      state_probe=probe(*(v for x,s in zip(features,SPLITS) for v in (x,labels[s])),4),
                      structure_ridge=fit_readout(*(v for x,s in zip(features,SPLITS) for v in (x,np.asarray(ds[s].arrays['structure'])))))
        # Old-head degradation and information loss are reported separately from new-head scores.
        longs = [np.asarray(ds[s].arrays['long']) for s in SPLITS]
        recovered, recovery_info = fit_latent_readout(features, longs)
        predicted_structure, hist_loss, hist_close = [], 0., 0.
        for start in range(0, len(truth), batch):
            z = torch.tensor(recovered[2][start:start+batch], device=device)
            predicted_structure.extend((long_model.structure(z)*old_scale+old_mean).cpu().numpy())
            y = torch.tensor(np.array(ds['test'].arrays['y'][start:start+batch]), device=device)
            mask = torch.tensor(np.array(ds['test'].arrays['mask'][start:start+batch]), device=device)
            valid = torch.tensor(np.array(ds['test'].arrays['valid'][start:start+batch]), device=device)
            a,c = history_values(long_model.decode_history(z)['reconstruction'],y,mask,valid)
            hist_loss += float(a.sum()); hist_close += float(c.sum())
        result['adapted_legacy'] = dict(**recovery_info, structure=regression_metrics(truth,np.asarray(predicted_structure)),
            history_loss=hist_loss/len(truth), history_close_bps=hist_close/len(truth),
            test_latent_standardized_mse=float(np.square((recovered[2]-longs[2])/np.asarray(stats['stats']['long_scale'])).mean()))
        atomic_json(result, path/'metrics.json'); atomic_json(per_window,path/'per_window_metrics.json')
        for label,cards in (('recent',recent_cards),('history',history_cards)):
            atomic_json(cards,path/f'{label}_examples.json'); write_global_review(path/f'{label}_examples.html',cards)
            page=path/f'{label}_examples.html'
            page.write_text(page.read_text().replace('单个512维综合向量重建历史',f'{model.dimensions}维表示：{label}历史重建').replace('全局向量解码','新读出头重建'))
        report['variants'][name]=result; atomic_json(report,out/'dual_state_metrics.json')
        progress(f'{name}: selected={best["epoch"]}, qualified={result["qualified"]}, state_BA={result["state_probe"]["test"]["ba"]:.4f}')
    report['paired_comparison'] = {}
    for seed in meta['seeds']:
        reference = report['variants'][f'concat_s{seed}']; candidate = report['variants'][f'compress_s{seed}']
        ratios = {k:candidate['validation'][k]/max(reference['validation'][k],1e-8) for k in TASKS}
        report['paired_comparison'][str(seed)] = dict(validation_compress_over_concat=ratios,
            within_five_percent_on_every_validation_metric=all(v<=1.05 for v in ratios.values()),
            warning='Two decoder/representation seeds do not measure encoder or market-regime uncertainty. Dimensionality and adapter capacity differ; not a pure equal-parameter comparison.')
    atomic_json(report,out/'dual_state_metrics.json')
    lines=['# 长短双状态实验结果','', '按验证集选择权重；测试期已被研发反复使用。qualified=false 表示未同时满足全部保留门槛。','',
           '| 实验 | 维度 | 轮次 | 合格 | 近期64 MAE bp | 历史 MAE bp | 高点 R² | 低点 R² | 高点年龄 R² | 趋势 BA |',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for name,r in report['variants'].items():
        vals=[r['structure'][k]['r2'] for k in ('distance_from_prior_high','distance_from_prior_low','prior_high_age')]
        fmt=lambda x: f'{x:.3f}' if x is not None else 'N/A'
        lines.append(f'| {name} | {r["dimensions"]} | {r["selected_epoch"]} | {r["qualified"]} | {r["reconstruction"]["recent/64"]["metrics"]["close_mae_bps"]["mean"]:.2f} | {r["reconstruction"]["history"]["metrics"]["close_mae_bps"]["mean"]:.2f} | '+ ' | '.join(map(fmt,vals))+f' | {r["state_probe"]["test"]["ba"]:.2%} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('all','worker','evaluate')); p.add_argument('--out',required=True)
    p.add_argument('--source'); p.add_argument('--long-run'); p.add_argument('--root'); p.add_argument('--name')
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch',type=int,default=128)
    p.add_argument('--micro',type=int,default=128); p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--eval-batch',type=int,default=16); p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    a=p.parse_args()
    if not torch.cuda.is_available(): raise ValueError('CUDA required; no local neural training fallback')
    torch.set_num_threads(4)
    out=Path(a.out).resolve()
    if a.action=='worker':
        if not a.name: p.error('--name required')
        worker(out,a.name); return
    if not all((a.source,a.long_run,a.root)): p.error('--source, --long-run, --root required')
    if min(a.epochs,a.batch,a.micro,a.jobs,a.eval_batch)<1 or a.batch%a.micro or a.jobs>4:
        raise ValueError('Invalid epoch/batch/micro/job settings')
    source=Path(a.source).resolve(); long_run=Path(a.long_run).resolve()
    progress('Verifying source reconstruction cache and checkpoints')
    identity=source_identity(source,long_run)
    meta=dict(schema='babel-dual-state-v1',source=str(source),long_run=str(long_run),root=str(Path(a.root).resolve()),
              sources=identity,seeds=a.seeds,experiments=experiment_jobs(a.seeds),epochs=a.epochs,batch=a.batch,micro=a.micro,lr=3e-4,
              normalization='Train-only mean/std, floor .01. Recent/history losses normalized by train-only constant predictors; structure standardized MSE.',
              selection='All six validation metrics: recent <= 1.05 previous short baseline; history loss/MAE and each structure MSE <= 1.02 original long baseline. Prefer qualified; minimize worst threshold ratio, then mean ratio. If none qualify, report best compromise explicitly.',
              scope='Frozen causal long/short states at existing complete-block endpoints. No forecasting, no noise generation, no online per-bar deployment claim.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta: raise ValueError('Settings changed; use a new output directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Output directory lacks resumable manifest')
    out.mkdir(parents=True,exist_ok=True); atomic_json(meta,out/'manifest.json')
    # A previous successful completion marker must never describe a new partial attempt.
    (out/'completion.json').unlink(missing_ok=True)
    prepare(source,long_run,a.root,out,identity,a.eval_batch,'cuda'); torch.cuda.empty_cache()
    atomic_json(dict(jobs=a.jobs,eval_batch=a.eval_batch),out/'runtime.json')
    if a.action=='all': run_jobs(out,a.jobs)
    evaluate(out,long_run,a.eval_batch,'cuda')
    if sha256(long_run/'joint/best.pt')!=identity['long_best']: raise ValueError('Frozen source changed')
    atomic_json(dict(status='complete',experiments=[j['name'] for j in meta['experiments']]),out/'completion.json')
    progress(f'Dual-state matrix complete: {out}')


if __name__=='__main__': main()
