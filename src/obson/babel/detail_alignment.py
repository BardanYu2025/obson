"""Factorial encoder/loss experiment for preserving observed local price changes."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import (read_arrays, device_arrays, describe_predictions, paired_error_interval,
                                 run_jobs, derangement)
from .autoregressive import probe
from .dual_state import source_identity, sha256, verify_files
from .history_autoencoder import to_ohlc
from .large_history import write_global_review
from .progress import progress
from .reconstruction_fusion import ReconstructionFusion, recent_loss
from .representation import time_mask
from .short_state import ShortState
from .stage_audit import load_data
from .stream_audit import bounded_difference

SCHEMA = 'babel-detail-alignment-v1'
SPLITS = ('train', 'val', 'test')
MODES = ('frozen_base', 'frozen_detail', 'joint_base', 'joint_detail')
HORIZONS = (1, 4, 16)


def detail_scales(y):
    """One RMS scale per horizon, fitted only on training targets (log-percent)."""
    close = np.asarray(y, np.float64)[..., 1]
    return [float(max(np.sqrt(np.mean((close[:, h:]-close[:, :-h])**2)), .01)) for h in HORIZONS]


def detail_values(pred, target, scales):
    terms = []
    for h, scale in zip(HORIZONS, scales):
        a = (pred[:, h:, 1]-pred[:, :-h, 1])/scale
        b = (target[:, h:, 1]-target[:, :-h, 1])/scale
        terms.append(nn.functional.smooth_l1_loss(a, b, reduction='none').mean(1))
    return torch.stack(terms).mean(0)


def values(pred, b, scales):
    return dict(base=recent_loss(pred, b['y'], b['mask']), detail=detail_values(pred, b['y'], scales),
                close_bps=(pred[..., 1]-b['y'][..., 1]).abs().mean(1)*100,
                change16_mse=((pred[:, 16:, 1]-pred[:, :-16, 1])-(b['y'][:, 16:, 1]-b['y'][:, :-16, 1])).square().mean(1))


def eligible(score, reference):
    return (all(np.isfinite(v) for v in score.values()) and
            score['close_bps'] <= reference['close_bps']*1.02 + 1e-8 and
            score['base'] <= reference['base']*1.05 + 1e-8 and
            score['change16_mse'] <= reference['change16_mse']*1.05 + 1e-8)


def improves(score, best, reference):
    return eligible(score, reference) and (not eligible(best, reference) or
            (score['detail'], score['base']) < (best['detail'], best['base']))


class DetailModel(nn.Module):
    def __init__(self, width=512, decoder_width=256, input_dim=18):
        super().__init__()
        self.encoder = ShortState(width, 2, input_dim)
        # Its original small decoder is kept in the state dict but never trained/used.
        self.head = ReconstructionFusion('short', width, decoder_width)

    def configure(self, joint):
        self.encoder.requires_grad_(joint)
        self.encoder.decoder.requires_grad_(False)
        self.head.requires_grad_(True)


def initial_model(meta, device):
    model = DetailModel(**meta['config']).to(device)
    model.encoder.load_state_dict(torch.load(Path(meta['short_run'])/'best.pt', map_location='cpu', weights_only=True)['model'])
    model.head.load_state_dict(torch.load(Path(meta['source'])/'short/best.pt', map_location='cpu', weights_only=True)['model'])
    return model


def sequence_specs(series, encoded, bounds, split, keys):
    """Packed chronological prefixes from partition start through last scored endpoint."""
    specs = []; chunks = []; offset = 0
    for i in np.unique(keys[:, 0]):
        ids = np.flatnonzero(keys[:, 0] == i)
        valid = np.flatnonzero(time_mask(series[i], bounds, split))
        if not len(valid) or np.any(np.diff(valid) != 1): raise ValueError('Noncontiguous neural partition')
        lo = int(valid[0]); hi = int(keys[ids, 1].max())+1
        if (keys[ids, 1] < lo+127).any() or hi-1 > valid[-1]: raise ValueError('Endpoint crosses partition')
        order = ids[np.argsort(keys[ids, 1])]
        seq = encoded[i]['x'][lo:hi]
        chunks.append(seq)
        specs.append(dict(series=int(i), lo=lo, length=len(seq), offset=offset,
                          endpoints=[[int(keys[j, 1]-lo), int(j)] for j in order]))
        offset += len(seq)
    return np.concatenate(chunks), specs


class PackedStreams:
    def __init__(self, path, split):
        self.x = np.load(path/f'{split}_x.npy', mmap_mode='r', allow_pickle=False)
        self.specs = json.loads((path/f'{split}_sequences.json').read_text())

    def batches(self, batch, seed=None, inputs=True):
        # Group similarly sized sequences to limit padding. Shuffle whole groups;
        # never shuffle bars or recycle a finished lane into another contract.
        order = sorted(range(len(self.specs)), key=lambda i: self.specs[i]['length'])
        groups = [order[j:j+batch] for j in range(0, len(order), batch)]
        if seed is not None: random.Random(seed).shuffle(groups)
        for group in groups:
            seqs = [self.specs[i] for i in group]
            for left in range(0, max(s['length'] for s in seqs), 128):
                x = np.zeros((len(group), 128, self.x.shape[-1]), np.float32) if inputs else None
                pick = []
                for lane, s in enumerate(seqs):
                    right = min(left+128, s['length'])
                    if right <= left: continue
                    if inputs: x[lane, :right-left] = self.x[s['offset']+left:s['offset']+right]
                    pick.extend((lane, end-left, j) for end, j in s['endpoints'] if left <= end < right)
                yield left == 0, x, pick


def publish(state, path):
    atomic_save(state, path/'last.pt')
    atomic_save(dict(metadata=state['metadata'], epoch=state['best_epoch'], model=state['best_model'],
                     validation=state['best_validation']), path/'best.pt')
    temp = path/'history.jsonl.tmp'
    temp.write_text(''.join(json.dumps(x, allow_nan=False)+'\n' for x in state['history'])); temp.replace(path/'history.jsonl')
    row = state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'], best_epoch=state['best_epoch'], validation=row.get('validation'),
                     seconds=row.get('seconds'), remaining_minutes=row.get('remaining_minutes')), path/'progress.json')


def run_epoch(model, data, streams, joint, detail, scales, batch, opt=None, seed=None, collect=False, detail_weight=.25, objective_fn=None):
    """Same scored endpoints and optimizer grouping in all four cells."""
    model.train(opt is not None)
    if not joint: model.encoder.eval()
    model.encoder.decoder.eval()
    device = data['z'].device; hidden = None; sums = {}; seen = np.zeros(len(data['z']), bool)
    zs = np.empty_like(data['z'].cpu().numpy()) if collect else None
    ps = np.empty_like(data['y'].cpu().numpy()) if collect else None
    with torch.set_grad_enabled(opt is not None):
        for reset, x, pick in streams.batches(batch, seed, inputs=joint):
            if reset: hidden = None
            if joint:
                h, hidden = model.encoder(torch.tensor(x, device=device), hidden)
                hidden = hidden.detach()  # Values persist; backward graph spans one128-bar chunk.
            if not pick: continue
            lanes, steps, ids_tuple = zip(*pick); ids = list(ids_tuple)
            if seen[ids].any(): raise ValueError('Duplicate scored endpoints')
            seen[ids] = True
            z = h[list(lanes), list(steps)] if joint else data['z'][ids]
            b = {k: v[ids] for k, v in data.items()}
            pred = model.head.decode_recent(z); parts = values(pred, b, scales)
            loss = (objective_fn(pred, b, parts) if objective_fn else
                    parts['base'] + (detail_weight*parts['detail'] if detail else 0))
            if not torch.isfinite(loss).all(): raise ValueError('Nonfinite detail objective')
            if objective_fn: parts['optimized_objective'] = loss
            if opt is not None:
                opt.zero_grad(set_to_none=True); loss.mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
            for k, v in parts.items(): sums[k] = sums.get(k, 0.)+float(v.detach().sum())
            if collect:
                zs[ids] = z.detach().cpu().numpy(); ps[ids] = pred.detach().cpu().numpy()
    if not seen.all(): raise ValueError('Incomplete scored endpoint coverage')
    score = {k: v/len(seen) for k, v in sums.items()}
    if not all(np.isfinite(v) for v in score.values()): raise ValueError('Nonfinite detail metrics')
    return score, zs, ps


def worker(out, name, device='cuda', *, model_factory=None, streams_factory=None, allow_initial_regression=False):
    meta = json.loads((out/'manifest.json').read_text()); aux = json.loads((out/'cache/statistics.json').read_text())
    job = next(j for j in meta['experiments'] if j['name'] == name); joint = job['mode'].startswith('joint')
    detail = job['mode'].endswith('detail'); path = out/name; path.mkdir(exist_ok=True)
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = model_factory(meta, device, job) if model_factory else initial_model(meta, device)
    model.configure(joint)
    groups = [dict(params=model.head.parameters(), lr=meta['decoder_lr'])]
    if joint: groups.append(dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=meta['encoder_lr']))
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    arrays = {s: device_arrays(read_arrays(meta['source'], s), device) for s in ('train', 'val')}
    streams = {s: (streams_factory(out/'cache', s, job) if streams_factory else PackedStreams(out/'cache', s))
               for s in ('train', 'val')}
    metadata = dict(manifest=meta, job=job)
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata: raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    else:
        initial = run_epoch(model, arrays['val'], streams['val'], joint, detail, aux['scales'], meta['streams'])[0]
        if not eligible(initial, aux['reference']) and not allow_initial_regression:
            raise ValueError('Warm start fails baseline retention')
        state = dict(metadata=metadata, epoch=0, best_epoch=0, history=[], best_validation=initial,
                     best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    for epoch in range(state['epoch']+1, meta['epochs']+1):
        started = time.perf_counter()
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        train = run_epoch(model, arrays['train'], streams['train'], joint, detail, aux['scales'], meta['streams'],
                          opt, job['seed']+epoch, detail_weight=meta['detail_weight'])[0]
        score = run_epoch(model, arrays['val'], streams['val'], joint, detail, aux['scales'], meta['streams'])[0]
        if improves(score, state['best_validation'], aux['reference']):
            state.update(best_epoch=epoch, best_validation=score,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        seconds = time.perf_counter()-started
        state['history'].append(dict(epoch=epoch, train=train, validation=score, eligible=eligible(score, aux['reference']),
                                    seconds=seconds, remaining_minutes=seconds*(meta['epochs']-epoch)/60,
                                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else None))
        state.update(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state()); publish(state, path)
        progress(f'{name} epoch={epoch}/{meta["epochs"]} detail={score["detail"]:.5f} close={score["close_bps"]:.2f}bp eligible={eligible(score,aux["reference"])} best={state["best_epoch"]} seconds={seconds:.1f}')
    publish(state, path)


def prepare(meta, out, device, *, extra_builder=None):
    cache = out/'cache'; cache.mkdir(exist_ok=True); index = cache/'index.json'
    if index.exists():
        info = json.loads(index.read_text())
        if info['manifest'] != meta: raise ValueError('Prepared data identity changed')
        verify_files(cache, info['files']); progress('Verified prepared chronological streams'); return
    ref, series, encoded, bases = load_data(meta['root'], meta['long_run'])
    source = Path(meta['source']); files = {}; coverage = {}
    for split in SPLITS:
        keys = np.load(source/f'target_cache/{split}_keys.npy', allow_pickle=False)
        if extra_builder: files.update(extra_builder(cache, split, keys, encoded))
        x, specs = sequence_specs(series, encoded, ref['boundaries'], split, keys)
        with (cache/f'{split}_x.tmp').open('wb') as f: np.save(f, x, allow_pickle=False)
        (cache/f'{split}_x.tmp').replace(cache/f'{split}_x.npy')
        atomic_json(specs, cache/f'{split}_sequences.json')
        for name in (f'{split}_x.npy', f'{split}_sequences.json'): files[name] = sha256(cache/name)
        coverage[split] = dict(windows=len(keys), streams=len(specs), replayed_bars=len(x))
        if split == 'test':
            inventory = [dict(key=series[i].key, symbol=series[i].code, period=series[i].period, row=int(end),
                              end=str(series[i].frame.datetime.iloc[end]), anchor=float(series[i].frame.close.iloc[end-64]),
                              week=str(pd.Timestamp(series[i].sessions[end]).to_period('W-SUN'))) for i, end in keys]
            atomic_json(inventory, cache/'test_inventory.json'); files['test_inventory.json'] = sha256(cache/'test_inventory.json')
    scales = detail_scales(read_arrays(source, 'train')['y'])
    model = initial_model(meta, device).eval()
    va = device_arrays(read_arrays(source, 'val'), device); streams = PackedStreams(cache, 'val')
    reference = run_epoch(model, va, streams, False, False, scales, meta['streams'])[0]
    joint_score, reproduced, _ = run_epoch(model, va, streams, True, False, scales, meta['streams'], collect=True)
    numeric = bounded_difference(reproduced, va['z'].cpu().numpy())
    atomic_json(dict(comparison=numeric, cached=reference, replayed=joint_score), out/'warm_start_validation.json')
    if not numeric['passed'] or not eligible(joint_score, reference):
        raise ValueError('Original encoder replay differs from frozen cache; see warm_start_validation.json')
    atomic_json(dict(scales=scales, reference=reference, scale_units='training-only close-change RMS in log-percent'), cache/'statistics.json')
    files['statistics.json'] = sha256(cache/'statistics.json')
    atomic_json(dict(boundaries=ref['boundaries'], splits=coverage), out/'coverage.json')
    atomic_json(dict(manifest=meta, files=files), index)


def preflight(meta, out, device='cuda'):
    model = initial_model(meta, device); model.configure(True); model.train()
    scales = json.loads((out/'cache/statistics.json').read_text())['scales']; batch = meta['streams']
    x = torch.randn(batch, 128, 18, device=device)
    y = torch.randn(batch, 64, 7, device=device)*.1; y[..., 2:] = y[..., 2:].abs()
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    h, _ = model.encoder(x)
    pred = model.head.decode_recent(h[:, -1])
    parts = values(pred, dict(y=y, mask=torch.ones_like(y, dtype=torch.bool)), scales)
    loss = (parts['base']+meta['detail_weight']*parts['detail']).mean(); loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(loss): raise ValueError('Nonfinite synthetic preflight')
    return dict(streams=batch, parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                gradient_norm=float(norm), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else None,
                scope='Disposable full-chunk joint backward; no optimizer step, excludes AdamW states and other workers.')


@torch.no_grad()
def evaluate(out, device, *, model_factory=None, streams_factory=None, contrasts=None, schema=SCHEMA,
             report_name='detail_metrics.json', title='局部细节：编码器与损失2×2实验', scope=None):
    meta = json.loads((out/'manifest.json').read_text()); source = Path(meta['source'])
    aux = json.loads((out/'cache/statistics.json').read_text()); inventory = json.loads((out/'cache/test_inventory.json').read_text())
    arrays = {s: device_arrays(read_arrays(source, s), device) for s in SPLITS}
    streams = {s: PackedStreams(out/'cache', s) for s in SPLITS}
    labels = {s: np.load(source/f'target_cache/{s}_labels.npy', allow_pickle=False) for s in SPLITS}
    old = initial_model(meta, device).eval(); frozen_zs = [arrays[s]['z'].cpu().numpy() for s in SPLITS]
    frozen_probe = probe(*(v for z, s in zip(frozen_zs, SPLITS) for v in (z, labels[s])), 4)
    report = dict(schema=schema, variants={}, paired={}, reference=aux['reference'], scales=aux['scales'],
                  scope=scope or 'Two-by-two warm-start encoder/loss experiment; same architecture and endpoints. Reused research test period. Selected by validation detail with base/price/change16 retention gates.',
                  limits='Failure of a frozen head does not prove information absent from z. Joint improvement indicates trainable information preservation, not an information-theoretic capacity bound.')
    per_windows = {}; chosen = set(np.linspace(0, len(inventory)-1, min(6, len(inventory)), dtype=int))
    for job in [dict(name='original', mode='frozen_base')] + meta['experiments']:
        name = job['name']; path = out/name; path.mkdir(exist_ok=True); joint = job['mode'].startswith('joint')
        model = (model_factory(meta, device, job) if model_factory else initial_model(meta, device)).eval(); epoch = 0
        if streams_factory: streams = {s: streams_factory(out/'cache', s, job) for s in SPLITS}
        if name != 'original':
            ck = torch.load(path/'best.pt', map_location='cpu', weights_only=True)
            if ck['metadata'] != dict(manifest=meta, job=job): raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(ck['model']); epoch = ck['epoch']
        progress(f'Evaluating {name}: reconstruction, frozen-head compatibility and state readout')
        val_score = run_epoch(model, arrays['val'], streams['val'], joint, False, aux['scales'], meta['streams'])[0]
        if name != 'original' and not np.allclose([val_score[k] for k in val_score], [ck['validation'][k] for k in val_score], rtol=1e-4, atol=1e-5):
            raise ValueError('Selected validation failed to reproduce')
        test_score, z, pred = run_epoch(model, arrays['test'], streams['test'], joint, False, aux['scales'], meta['streams'], collect=True)
        result, rows = describe_predictions(arrays['test']['y'].cpu().numpy(), pred, inventory)
        details = detail_values(torch.tensor(pred, device=device), arrays['test']['y'], aux['scales']).cpu().tolist()
        for row, detail in zip(rows, details): row['normalized_detail'] = detail
        result.update(selected_epoch=epoch, validation=val_score, validation_retained=eligible(val_score, aux['reference']), test_objectives=test_score)
        if joint:
            zs = [run_epoch(model, arrays[s], streams[s], True, False, aux['scales'], meta['streams'], collect=True)[1] for s in ('train', 'val')]+[z]
            result['state_probe'] = probe(*(v for x, s in zip(zs, SPLITS) for v in (x, labels[s])), 4)
        else: result['state_probe'] = frozen_probe
        # Direct compatibility with the OLD head is distinct from adapted new-head performance.
        legacy = []; shuffled = []; perm = derangement(len(z))
        for start in range(0, len(z), 256):
            legacy.extend(old.head.decode_recent(torch.tensor(z[start:start+256], device=device)).cpu().numpy())
            shuffled.extend(model.head.decode_recent(torch.tensor(z[perm[start:start+256]], device=device)).cpu().numpy())
        result['legacy_head'], _ = describe_predictions(arrays['test']['y'].cpu().numpy(), np.asarray(legacy), inventory)
        result['shuffled'], _ = describe_predictions(arrays['test']['y'].cpu().numpy(), np.asarray(shuffled), inventory)
        result['relative_latent_mse'] = float(np.square(z-frozen_zs[2]).mean()/max(np.square(frozen_zs[2]).mean(), 1e-12))
        cards = [dict(source=inventory[j]['key'], end=inventory[j]['end'], blocks=1,
                      truth_ohlc=to_ohlc(arrays['test']['y'][j].cpu().numpy(), inventory[j]['anchor']).tolist(),
                      reconstructed_ohlc=to_ohlc(pred[j], inventory[j]['anchor']).tolist()) for j in sorted(chosen)]
        atomic_json(cards, path/'examples.json'); write_global_review(path/'examples.html', cards)
        page = path/'examples.html'; page.write_text(page.read_text().replace('单个512维综合向量重建历史', name+'：局部细节重建对照').replace('全局向量解码', '短状态解码'))
        atomic_json(result, path/'metrics.json'); atomic_json(rows, path/'per_window_metrics.json')
        per_windows[name] = rows; report['variants'][name] = result
        atomic_json(report, out/report_name)
    weeks = np.array([r['week'] for r in inventory])
    for seed in meta['seeds']:
        for a, b in (contrasts if contrasts is not None else (('frozen_detail', 'frozen_base'), ('joint_base', 'frozen_base'), ('joint_detail', 'joint_base'), ('joint_detail', 'frozen_detail'))):
            an, bn = f'{a}_s{seed}', f'{b}_s{seed}'
            compared = {}
            for metric in ('normalized_detail', 'close_mae_bps', 'change_1_mae_bps', 'change_1_correlation', 'change_1_std_ratio'):
                av = np.array([r.get(metric) if r.get(metric) is not None else np.nan for r in per_windows[an]])
                bv = np.array([r.get(metric) if r.get(metric) is not None else np.nan for r in per_windows[bn]])
                valid = np.isfinite(av) & np.isfinite(bv)
                if not valid.any():
                    compared[metric] = dict(supported_windows=0); continue
                item = paired_error_interval(av[valid], bv[valid], weeks[valid])
                item['delta_mean'] = item.pop('delta_close_mae_bps')
                item.update(supported_windows=int(valid.sum()), interpretation='Candidate minus reference; lower error or higher correlation is better. Std-ratio must be interpreted relative to1, not blindly maximized. Paired weekly research bootstrap.')
                compared[metric] = item
            report['paired'][an+'_minus_'+bn] = compared
    atomic_json(report, out/report_name)
    lines = ['# '+title, '', report['scope'], '',
             '| 组 | 轮次 | 验证保留门槛 | 收盘MAE bp | 1根相关性 | 1根标准差比 | 16根相关性 | 状态BA |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for name, r in report['variants'].items():
        m = r['groups']['all']['metrics']
        fmt = lambda v: '—' if v is None else f'{v:.3f}'
        lines.append(f'| {name} | {r["selected_epoch"]} | {r["validation_retained"]} | '+ ' | '.join(fmt(m[k]['mean']) for k in ('close_mae_bps','change_1_correlation','change_1_std_ratio','change_16_correlation'))+f' | {r["state_probe"]["test"]["ba"]:.2%} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('all', 'worker', 'evaluate')); p.add_argument('--out', required=True)
    p.add_argument('--source'); p.add_argument('--short-run'); p.add_argument('--long-run'); p.add_argument('--fusion-run'); p.add_argument('--root'); p.add_argument('--name')
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--streams', type=int, default=64); p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise ValueError('Formal detail training requires CUDA')
    torch.set_num_threads(4); out = Path(a.out).resolve()
    if a.action == 'worker':
        if not a.name: p.error('--name required')
        worker(out, a.name); return
    if not all((a.source,a.short_run,a.long_run,a.fusion_run,a.root)): p.error('All source paths required')
    if min(a.epochs,a.streams,a.jobs)<1 or a.jobs>4: raise ValueError('Invalid runtime settings')
    source, short, long, fusion = map(lambda s: Path(s).resolve(), (a.source,a.short_run,a.long_run,a.fusion_run))
    identity = source_identity(source, long)
    fm = json.loads((fusion/'manifest.json').read_text()); rm = json.loads((source/'manifest.json').read_text())
    if rm['sources']['fusion_manifest'] != sha256(fusion/'manifest.json') or fm['sources']['short_best'] != sha256(short/'best.pt'):
        raise ValueError('Short encoder is not the source of cached states')
    identity.update(short_best=sha256(short/'best.pt'), fusion_manifest=sha256(fusion/'manifest.json'))
    meta = dict(schema=SCHEMA, source=str(source), short_run=str(short), long_run=str(long), root=str(Path(a.root).resolve()), sources=identity,
                config=dict(width=512, decoder_width=256), seeds=[42,43], epochs=a.epochs, streams=a.streams,
                encoder_lr=3e-5, decoder_lr=1e-4, detail_weight=.25,
                experiments=[dict(name=f'{mode}_s{seed}',mode=mode,seed=seed) for seed in (42,43) for mode in MODES],
                selection='Minimize validation normalized1/4/16 change loss, while close<=1.02 original, base<=1.05 original, change16 MSE<=1.05 original; include epoch0.',
                protocol='Same chronological endpoint groups for frozen and joint; TBPTT128 with persistent hidden values, no sequence-internal shuffle; warm start all cells, no model expansion.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != meta: raise ValueError('Settings changed; choose new output directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty directory lacks manifest')
    out.mkdir(parents=True, exist_ok=True); atomic_json(meta, out/'manifest.json')
    for name in ('completion.json','detail_metrics.json','summary.md'): (out/name).unlink(missing_ok=True)
    progress('Preparing shared chronological streams and train-only detail scales')
    prepare(meta, out, 'cuda'); torch.cuda.empty_cache()
    atomic_json(dict(jobs=a.jobs,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)), out/'runtime.json')
    if a.action == 'all':
        progress('Synthetic GPU joint forward/backward preflight; original files remain unchanged')
        atomic_json(preflight(meta, out), out/'preflight.json'); torch.cuda.empty_cache()
        run_jobs(out, a.jobs, 'obson.babel.detail_alignment')
    evaluate(out, 'cuda')
    after = source_identity(source, long); after.update(short_best=sha256(short/'best.pt'),fusion_manifest=sha256(fusion/'manifest.json'))
    if after != identity: raise ValueError('Source artifacts changed during experiment')
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),source_weights_unchanged=True),out/'completion.json')
    progress(f'Detail alignment matrix complete: {out}')


if __name__ == '__main__': main()
