"""Frozen short embeddings: parallel versus conditional autoregressive reconstruction."""
import argparse
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .ae_diagnostics import metrics, summarize
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .dual_state import sha256, source_identity
from .history_autoencoder import positions, to_ohlc
from .large_history import write_global_review
from .progress import progress
from .reconstruction_fusion import ReconstructionFusion, recent_loss
from .stage_audit import load_data

SCHEMA = 'babel-ar-reconstruction-v1'
MODES = ('parallel', 'ar_tf', 'ar_mix', 'ar_noz_mix')
SPLITS = ('train', 'val', 'test')


class ARDecoder(nn.Module):
    """Same endpoint embedding at every step; teacher inputs are strictly shifted.

    rollout accepts no target tensor, mask, true initial bar or encoder sequence.
    Coordinates and explicit external price anchor follow the original recent head.
    """
    def __init__(self, latent=512, width=352, layers=2, length=64, no_z=False):
        super().__init__()
        self.length, self.no_z = length, no_z
        self.condition = nn.Linear(latent, width)
        self.previous = nn.Linear(7, width)
        self.bos = nn.Parameter(torch.zeros(width))
        self.input_norm = nn.LayerNorm(width)
        self.rnn = nn.GRU(width, width, layers, batch_first=True, dropout=.1 if layers > 1 else 0.)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 7))
        self.register_buffer('position', positions(length, width, 'cpu', torch.float32))

    def context(self, z):
        return self.condition(torch.zeros_like(z) if self.no_z else z)

    def project(self, h):
        raw = self.output(h)
        return torch.cat((raw[..., :2], nn.functional.softplus(raw[..., 2:])), -1)

    def teacher(self, z, y):
        if y.shape[1:] != (self.length, 7) or len(y) != len(z):
            raise ValueError('Expected aligned full reconstruction targets')
        # Position0 is a learned BOS, not the first true bar.
        tokens = torch.cat((self.bos.expand(len(z), 1, -1), self.previous(torch.asinh(y[:, :-1]))), 1)
        h, _ = self.rnn(self.input_norm(tokens + self.context(z)[:, None] + self.position))
        return self.project(h)

    def rollout(self, z):
        context = self.context(z); hidden = None; output = []
        for step in range(self.length):
            token = self.bos.expand(len(z), -1) if step == 0 else self.previous(torch.asinh(output[-1]))
            h, hidden = self.rnn(self.input_norm(token + context + self.position[step])[:, None], hidden)
            output.append(self.project(h[:, 0]))
        return torch.stack(output, 1)


def build(mode, config):
    if mode not in MODES:
        raise ValueError('Unknown reconstruction mode')
    if mode == 'parallel':
        return ReconstructionFusion('short', config['latent'], config['parallel_width'])
    return ARDecoder(config['latent'], config['ar_width'], no_z=mode == 'ar_noz_mix')


def free(model, z):
    return model.decode_recent(z) if isinstance(model, ReconstructionFusion) else model.rollout(z)


def rollout_weight(mode, epoch):
    # Pure teacher warmup5 epochs, then fixed10-epoch ramp to equal TF/free loss.
    return min(1., max(0., (epoch - 5) / 10.)) if mode in ('ar_mix', 'ar_noz_mix') else 0.


def objective(model, mode, b, epoch):
    if mode == 'parallel':
        return recent_loss(free(model, b['z']), b['y'], b['mask'])
    tf = recent_loss(model.teacher(b['z'], b['y']), b['y'], b['mask'])
    weight = rollout_weight(mode, epoch)
    if not weight:
        return tf
    # Full unrolled graph: neither generated bars nor hidden states are detached.
    roll = recent_loss(model.rollout(b['z']), b['y'], b['mask'])
    return (tf + weight * roll) / (1 + weight)


def read_arrays(source, split):
    cache = Path(source) / 'target_cache'
    arrays = {key: np.load(cache / f'{split}_{name}.npy', allow_pickle=False)
              for key, name in (('z', 'short'), ('y', 'recent_y'), ('mask', 'recent_mask'))}
    n = len(arrays['z'])
    if not n or arrays['y'].shape != (n, 64, 7) or arrays['mask'].shape != (n, 64, 7):
        raise ValueError('Invalid cached reconstruction shapes')
    if arrays['mask'].dtype != bool or not arrays['mask'][..., :5].all() or not arrays['mask'][..., 6].all():
        raise ValueError('Only unavailable open interest may be masked')
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Nonfinite cached reconstruction inputs')
    return arrays


def device_arrays(arrays, device):
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


@torch.no_grad()
def validate(model, data, batch):
    model.eval(); total = tf_total = close = 0.; n = len(data['z'])
    for start in range(0, n, batch):
        b = {k: v[start:start+batch] for k, v in data.items()}
        pred = free(model, b['z'])
        loss = recent_loss(pred, b['y'], b['mask'])
        if not torch.isfinite(pred).all() or not torch.isfinite(loss).all():
            raise ValueError('Nonfinite free reconstruction')
        total += float(loss.sum()); close += float((pred[..., 1]-b['y'][..., 1]).abs().mean(1).sum()) * 100
        tf_total += float(recent_loss(model.teacher(b['z'], b['y']), b['y'], b['mask']).sum()) if isinstance(model, ARDecoder) else float(loss.sum())
    result = dict(free_loss=total/n, free_close_mae_bps=close/n, teacher_loss=tf_total/n)
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite validation statistics')
    return result


def publish(state, path):
    atomic_save(state, path/'last.pt')
    atomic_save(dict(metadata=state['metadata'], model=state['best_model'], epoch=state['best_epoch'],
                     validation=state['best_validation']), path/'best.pt')
    temp = path/'history.jsonl.tmp'
    temp.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in state['history']))
    temp.replace(path/'history.jsonl')
    last = state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'], best_epoch=state['best_epoch'],
                     free_validation=last.get('validation'), seconds=last.get('seconds'),
                     remaining_minutes=last.get('remaining_minutes')), path/'progress.json')


def worker(out, name, device='cuda'):
    meta = json.loads((out/'manifest.json').read_text())
    job = next(j for j in meta['experiments'] if j['name'] == name)
    path = out/name; path.mkdir(exist_ok=True)
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = build(job['mode'], meta['config']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=meta['lr'], weight_decay=.01)
    tr, va = (device_arrays(read_arrays(meta['source'], s), device) for s in ('train', 'val'))
    metadata = dict(manifest=meta, job=job)
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata:
            raise ValueError('Resume settings differ; use a new run directory')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    else:
        initial = validate(model, va, meta['batch'])
        state = dict(metadata=metadata, epoch=0, best_epoch=0, best_validation=initial, history=[],
                     best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    for epoch in range(state['epoch']+1, meta['epochs']+1):
        started = time.perf_counter(); model.train(); total = 0.; n = len(tr['z'])
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        # Paired modes have the same epoch sample order for a given seed.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(job['seed']+epoch)).to(device)
        for start in range(0, n, meta['batch']):
            ids = order[start:start+meta['batch']]; b = {k: v[ids] for k, v in tr.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, job['mode'], b, epoch)
            if not torch.isfinite(loss).all():
                raise ValueError('Nonfinite training objective')
            loss.mean().backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach().sum())
        score = validate(model, va, meta['batch'])
        if score['free_loss'] < state['best_validation']['free_loss']:
            state.update(best_epoch=epoch, best_validation=score,
                         best_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        seconds = time.perf_counter()-started
        state['history'].append(dict(epoch=epoch, train_loss=total/n, validation=score,
             rollout_weight=rollout_weight(job['mode'], epoch), seconds=seconds,
             remaining_minutes=seconds*(meta['epochs']-epoch)/60,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else None))
        state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
        publish(state, path)
        progress(f'{name} epoch={epoch}/{meta["epochs"]} free={score["free_loss"]:.5f} teacher={score["teacher_loss"]:.5f} close={score["free_close_mae_bps"]:.2f}bp best={state["best_epoch"]} seconds={seconds:.1f}')
    # Repair derivative files on a completed resume; do not take another optimizer step.
    publish(state, path)


def run_jobs(out, jobs):
    pending = list(json.loads((out/'manifest.json').read_text())['experiments']); active = {}; previous = {}
    def stop(signum, frame): raise SystemExit(128+signum)
    handler = signal.signal(signal.SIGTERM, stop)
    try:
        while pending or active:
            while pending and len(active) < jobs:
                job = pending.pop(0); name = job['name']; (out/name).mkdir(exist_ok=True)
                log = (out/name/'run.log').open('a')
                try:
                    proc = subprocess.Popen([sys.executable, '-m', 'obson.babel.ar_reconstruction', 'worker',
                                             '--out', str(out), '--name', name], stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close(); raise
                active[name] = (proc, log); progress(f'Started {name} pid={proc.pid}')
            for name, (proc, log) in list(active.items()):
                path = out/name/'progress.json'; code = proc.poll()
                if path.exists():
                    try: status = json.loads(path.read_text())
                    except (ValueError, OSError): status = None
                    if status and status != previous.get(name):
                        progress(f'{name}: {status}'); previous[name] = status
                if code is not None:
                    log.close(); del active[name]
                    if code: raise RuntimeError(f'{name} exited {code}; see {out/name/"run.log"}')
                    progress(f'Completed {name}')
            if active: time.sleep(1)
    finally:
        for proc, log in active.values():
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            log.close()
        signal.signal(signal.SIGTERM, handler)


def derangement(n, seed=1729):
    if n < 2: raise ValueError('At least two samples required for shuffled-z control')
    order = np.random.default_rng(seed).permutation(n); result = np.empty(n, np.int64)
    result[order] = np.roll(order, 1)
    return result


def preflight(config, batch, device='cuda'):
    """Discarded synthetic forward/backward only; never update source weights."""
    result = {}
    for mode in ('parallel', 'ar_mix'):
        started = time.perf_counter()
        model = build(mode, config).to(device).train()
        b = dict(z=torch.randn(batch, config['latent'], device=device),
                 y=torch.randn(batch, 64, 7, device=device)*.1,
                 mask=torch.ones(batch, 64, 7, device=device, dtype=torch.bool))
        b['y'][..., 2:] = b['y'][..., 2:].abs()
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        loss = objective(model, mode, b, 15).mean()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss): raise ValueError('Synthetic preflight loss is nonfinite')
        result[mode] = dict(parameters=sum(p.numel() for p in model.parameters()), loss=float(loss.detach()),
                            gradient_norm=float(norm), seconds=time.perf_counter()-started,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else None)
        del model, b, loss
        if device == 'cuda': torch.cuda.empty_cache()
    return dict(batch=batch, stages=result, scope='Synthetic discarded forward/backward; excludes AdamW states and concurrent-worker memory. No optimizer step.')


@torch.no_grad()
def predict_all(model, data, batch, z=None, teacher=False):
    model.eval(); result = []
    for start in range(0, len(data['z']), batch):
        latent = (data['z'] if z is None else z)[start:start+batch]
        pred = model.teacher(latent, data['y'][start:start+batch]) if teacher else free(model, latent)
        result.append(pred.cpu().numpy())
    result = np.concatenate(result)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite reconstruction predictions')
    return result


def describe_predictions(y, pred, inventory):
    records = {}; windows = []
    for j, (truth, output) in enumerate(zip(y, pred)):
        value = metrics(truth, output); windows.append(value)
        for group in ('all', 'symbol/'+inventory[j]['symbol'], 'period/'+str(inventory[j]['period'])):
            records.setdefault(group, []).append(value)
        for span in (16, 32):
            records.setdefault(f'recent/{span}', []).append(metrics(truth[-span:], output[-span:]))
        for quarter, start in enumerate(range(0, 64, 16), 1):
            records.setdefault(f'position/Q{quarter}', []).append(metrics(truth[start:start+16], output[start:start+16]))
    return dict(groups=summarize(records), close_mae_by_step_bps=(np.abs(pred[..., 1]-y[..., 1]).mean(0)*100).tolist()), windows


def paired_error_interval(candidate, reference, weeks, repeats=1000):
    delta = np.asarray(candidate)-reference
    unique, ids = np.unique(weeks, return_inverse=True)
    sums = np.bincount(ids, weights=delta); counts = np.bincount(ids)
    result = dict(delta_close_mae_bps=float(delta.mean()), weeks=len(unique), low=None, high=None,
                  interpretation='Negative favors candidate; paired calendar-week bootstrap, reused research test period.')
    if len(unique) >= 5:
        picks = np.random.default_rng(42).integers(len(unique), size=(repeats, len(unique)))
        draws = sums[picks].sum(1)/counts[picks].sum(1)
        result.update(low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)))
    return result


@torch.no_grad()
def evaluate(out, batch, device):
    meta = json.loads((out/'manifest.json').read_text()); source = Path(meta['source'])
    inventory = json.loads((out/'test_inventory.json').read_text())
    arrays = read_arrays(source, 'test'); data = device_arrays(arrays, device)
    val = device_arrays(read_arrays(source, 'val'), device)
    permutation = derangement(len(arrays['z'])); shuffled = data['z'][torch.tensor(permutation, device=device)]
    report = dict(schema=SCHEMA, encoder_training=False, variants={}, paired={},
        scope='Frozen identical short embeddings; selected only by full free64 validation reconstruction. Teacher-forced scores are diagnostic, not standalone decoding.',
        controls='Fixed deranged test vectors paired with unchanged targets. This may be out of distribution; separately trained no-z model is the stronger control.',
        budget='Same60-epoch default and sample order per seed; architecture and compute differ. Report parameters, wall time and memory. Two decoder seeds do not quantify encoder uncertainty.')
    windows = {}; chosen = set(np.linspace(0, len(inventory)-1, min(6, len(inventory)), dtype=int))
    jobs = [dict(name='original_parallel', mode='parallel', reference=True)] + meta['experiments']
    for job in jobs:
        name = job['name']; progress(f'Evaluating {name}: free / teacher / shuffled embedding')
        path = out/name; path.mkdir(exist_ok=True)
        checkpoint_path = source/'short/best.pt' if job.get('reference') else path/'best.pt'
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model = build(job['mode'], meta['config']).to(device); model.load_state_dict(ck['model']); model.eval()
        reproduced = validate(model, val, batch)
        if not job.get('reference'):
            if ck['metadata'] != dict(manifest=meta, job=job): raise ValueError('Selected checkpoint identity mismatch')
            if not np.isclose(reproduced['free_loss'], ck['validation']['free_loss'], atol=1e-5, rtol=1e-4):
                raise ValueError('Selected free validation score did not reproduce')
        pred = predict_all(model, data, batch)
        shuffled_pred = predict_all(model, data, batch, z=shuffled)
        result, per_window = describe_predictions(arrays['y'], pred, inventory)
        shuffled_result, _ = describe_predictions(arrays['y'], shuffled_pred, inventory)
        result.update(mode=job['mode'], selected_epoch=ck['epoch'], parameters=sum(p.numel() for p in model.parameters()),
                      checkpoint_sha256=sha256(checkpoint_path), validation=reproduced, shuffled=shuffled_result,
                      z_control_max_output_change=float(np.abs(shuffled_pred-pred).max()))
        if isinstance(model, ARDecoder):
            tf = predict_all(model, data, batch, teacher=True)
            result['teacher_forced'], _ = describe_predictions(arrays['y'], tf, inventory)
        if job['mode'] == 'ar_noz_mix' and not np.allclose(pred, shuffled_pred, atol=2e-5, rtol=2e-5):
            raise ValueError('No-z control unexpectedly depends on embedding')
        cards = []
        for j in sorted(chosen):
            row = inventory[j]
            cards.append(dict(source=row['key'], end=row['end'], blocks=1,
                              truth_ohlc=to_ohlc(arrays['y'][j], row['anchor']).tolist(),
                              reconstructed_ohlc=to_ohlc(pred[j], row['anchor']).tolist()))
        atomic_json(cards, path/'examples.json'); write_global_review(path/'examples.html', cards)
        page = path/'examples.html'
        page.write_text(page.read_text().replace('单个512维综合向量重建历史', name+'：完整自由重建64根').replace('全局向量解码', '自由解码'))
        atomic_json(result, path/'metrics.json'); atomic_json(per_window, path/'per_window_metrics.json')
        windows[name] = np.array([v['close_mae_bps'] for v in per_window])
        report['variants'][name] = result; atomic_json(report, out/'ar_reconstruction_metrics.json')
        del model
    weeks = np.array([r['week'] for r in inventory])
    for seed in meta['seeds']:
        for a, b in ((f'ar_tf_s{seed}', f'parallel_s{seed}'), (f'ar_mix_s{seed}', f'parallel_s{seed}'),
                     (f'ar_mix_s{seed}', f'ar_tf_s{seed}'), (f'ar_mix_s{seed}', f'ar_noz_mix_s{seed}'),
                     (f'ar_mix_s{seed}', 'original_parallel')):
            report['paired'][a+'_minus_'+b] = paired_error_interval(windows[a], windows[b], weeks)
    atomic_json(report, out/'ar_reconstruction_metrics.json')
    lines = ['# 冻结短向量：并行与自回归解码', '', '所有新模型按验证集完整自由重建loss选轮；两种随机种子。', '',
             '| 模型 | 最佳轮次 | 自由重建MAE bp | teacher MAE bp | 打乱向量MAE bp | 1根变化相关性 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in report['variants'].items():
        values = result['groups']['all']['metrics']; corr = values['change_1_correlation']['mean']
        tf = result.get('teacher_forced', {}).get('groups', {}).get('all', {}).get('metrics', {}).get('close_mae_bps', {}).get('mean')
        lines.append(f'| {name} | {result["selected_epoch"]} | {values["close_mae_bps"]["mean"]:.3f} | '+
                     ('—' if tf is None else f'{tf:.3f}')+f' | {result["shuffled"]["groups"]["all"]["metrics"]["close_mae_bps"]["mean"]:.3f} | '+
                     ('—' if corr is None else f'{corr:.3f}')+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def prepare_inventory(root, long_run, source, out):
    ref, series, encoded, bases = load_data(root, long_run)
    rows = np.load(source/'target_cache/test_keys.npy', allow_pickle=False)
    inventory = []
    for i, end in rows:
        s = series[i]
        inventory.append(dict(key=s.key, symbol=s.code, period=s.period, row=int(end),
                              end=str(s.frame.datetime.iloc[end]), anchor=float(s.frame.close.iloc[end-64]),
                              week=str(pd.Timestamp(s.sessions[end]).to_period('W-SUN'))))
    atomic_json(inventory, out/'test_inventory.json')
    atomic_json(dict(boundaries=ref['boundaries'], samples={s: len(read_arrays(source, s)['z']) for s in SPLITS}), out/'coverage.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('all', 'worker', 'evaluate')); p.add_argument('--out', required=True)
    p.add_argument('--source'); p.add_argument('--long-run'); p.add_argument('--root'); p.add_argument('--name')
    p.add_argument('--epochs', type=int, default=60); p.add_argument('--batch', type=int, default=256)
    p.add_argument('--eval-batch', type=int, default=256); p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    a = p.parse_args()
    if not torch.cuda.is_available(): raise ValueError('CUDA required; no local training fallback')
    torch.set_num_threads(4); out = Path(a.out).resolve()
    if a.action == 'worker':
        if not a.name: p.error('--name required')
        worker(out, a.name); return
    if not all((a.source, a.long_run, a.root)): p.error('--source, --long-run, --root required')
    if min(a.epochs, a.batch, a.eval_batch, a.jobs) < 1 or a.jobs > 4 or not a.seeds or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Invalid experiment settings')
    source, long_run = Path(a.source).resolve(), Path(a.long_run).resolve()
    progress('Verifying frozen source cache; no encoder is loaded in training workers')
    identity = source_identity(source, long_run)
    meta = dict(schema=SCHEMA, source=str(source), long_run=str(long_run), root=str(Path(a.root).resolve()), sources=identity,
                config=dict(latent=512, parallel_width=256, ar_width=352), seeds=a.seeds,
                experiments=[dict(name=f'{m}_s{s}', mode=m, seed=s) for s in a.seeds for m in MODES],
                epochs=a.epochs, batch=a.batch, lr=3e-4, selection='Minimum validation full free64 recent_loss; include epoch0.',
                objective='Unchanged seven-channel 16/32/64 multiscale reconstruction plus original close differences. Mix:5 TF epochs then10-epoch ramp to equal TF/free loss. No encoder updates.',
                coordinates='Same previous-window anchor coordinates as original cache; BOS learned, no true bar seeded into free rollout.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != meta: raise ValueError('Settings changed; use a new output directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty directory has no resumable manifest')
    out.mkdir(parents=True, exist_ok=True); atomic_json(meta, out/'manifest.json')
    for name in ('completion.json', 'summary.md', 'ar_reconstruction_metrics.json'): (out/name).unlink(missing_ok=True)
    prepare_inventory(a.root, long_run, source, out)
    atomic_json(dict(jobs=a.jobs, eval_batch=a.eval_batch, torch=str(torch.__version__), cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name()), out/'runtime.json')
    if a.action == 'all':
        progress('Synthetic full-batch GPU forward/backward preflight; disposable decoder weights')
        atomic_json(preflight(meta['config'], a.batch), out/'preflight.json')
        run_jobs(out, a.jobs)
    evaluate(out, a.eval_batch, 'cuda')
    if source_identity(source, long_run) != identity: raise ValueError('Frozen sources changed during experiment')
    atomic_json(dict(status='complete', encoder_training=False, experiments=len(meta['experiments'])), out/'completion.json')
    progress(f'AR reconstruction matrix complete: {out}')


if __name__ == '__main__': main()
