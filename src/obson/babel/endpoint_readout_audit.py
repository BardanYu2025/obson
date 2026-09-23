"""Zero-update, immutable audit of two existing readout routes from the same h128."""
import argparse
import html
import time
from pathlib import Path

import numpy as np
import torch

from . import endpoint_readout as er, bar_alignment_benchmark as bb
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-endpoint-readout-v1'
SPLITS = ('test', 'cross_research')


def code_identity():
    return bb.code_identity() | {Path(er.__file__).name: sha256(er.__file__), Path(__file__).name: sha256(__file__)}


def source_identity(source):
    """Verify the prior run without invoking its mutating lock/evaluation functions."""
    meta = read_json(source/'manifest.json'); bb.verify_code(meta)
    done = read_json(source/'completion.json'); lock = read_json(source/'selection_lock.json')
    if meta['schema'] != bb.SCHEMA or done['status'] != 'complete' or not done['source_unchanged']:
        raise ValueError('Completed alignment source required')
    bb.ab.sc.verify_worker_files(source, done['files']); bb.verify_cache(meta, source)
    if bb.source_identity(Path(meta['source'])) != meta['identity']:
        raise ValueError('Upstream alignment lineage changed')
    if lock['manifest'] != meta or done['cells'] != len(meta['experiments']):
        raise ValueError('Alignment selection identity changed')
    files = dict(done['files']) | {'completion.json': sha256(source/'completion.json')}
    summaries = []
    for job in meta['experiments']:
        name = job['name']; path = source/name; row = read_json(path/'completion.json')
        expected = dict(manifest=meta, job=job, cache_sha256=sha256(source/'cache/index.json'))
        if row['status'] != 'complete' or row['metadata'] != expected:
            raise ValueError('Incomplete alignment checkpoint')
        verify_files(path, row['files'])
        history = read_json(path/'history.json'); bb.verify_history(meta, source, job, history)
        initial = read_json(path/'initial_validation.json'); summary = read_json(path/'training_summary.json')
        epoch, val = min([(0, initial['validation'])]+[(r['epoch'], r['validation']) for r in history], key=lambda x: x[1]['selection'])
        if (len(history) != job['epochs'] or summary != lock['trials'][name]
                or summary['selected_epoch'] != epoch or summary['validation'] != val
                or summary['encoder_steps'] != sum(r['encoder_steps'] for r in history)
                or summary['head_steps'] != sum(r['head_steps'] for r in history)
                or initial['signatures'] != lock['initial_signatures'][name]):
            raise ValueError('Source selected epoch or budget changed')
        for kind, digest in lock['weights'][name].items():
            if sha256(path/f'{kind}.pt') != digest: raise ValueError('Locked source weight changed')
        files.update({f'{name}/{n}': digest for n, digest in row['files'].items()})
        files[f'{name}/completion.json'] = sha256(path/'completion.json')
        summaries.append(summary)
    for key in ('encoder_steps', 'head_steps'):
        if done[key] != sum(s[key] for s in summaries): raise ValueError('Incomplete source total budget')
    return dict(manifest=meta, files=files)


def make_manifest(source, identity, batch=64, causal_samples=8):
    if batch < 1 or causal_samples < 1: raise ValueError('Positive inference sizes required')
    return dict(schema=SCHEMA, source=str(source.resolve()), identity=identity, batch=batch, causal_samples=causal_samples,
        code_sha256=code_identity(), encoder_updates=0, head_updates=0, statistics_fits=0,
        routes=['local_head', 'global_crop'], checkpoints=['best', 'last'],
        positions=list(er.PREFIXES), tolerances=er.TOLERANCES,
        protocol='Same h128, previous16 excluding current. Global crop uses its own predicted row110 anchor; never the observed anchor. Existing scales, masks and source-selected weights only.',
        selection='All8 source cells, both seeds and best/last. No new selection, fitting, weight update, or route promotion.',
        causal_protocol='Fixed evenly spaced validation windows: strict prefixes, perturbed/appended future, independent calls and batch independence. FP32 failures receive same-weights CPU FP64 replay, not a relaxed FP32 threshold.',
        reset_protocol='A fresh64-bar suffix is a different context and positional origin; report differences without demanding equality or claiming streaming equivalence.',
        stop_rule='One frozen diagnostic. Do not launch further architecture/supervision sweeps. Original failed promotion decision remains unchanged.',
        limits='Reused research data. Better existing decoder access establishes accessible information, not universal representation quality; failure of both heads cannot prove information absence.')


def verify_manifest(meta):
    if meta['code_sha256'] != code_identity(): raise ValueError('Audit code changed; use a new run')


def require_replay(actual, expected, name, atol=1e-6, rtol=2e-5):
    if set(actual) != set(expected): raise ValueError(f'{name}: changed metric keys')
    for key in expected:
        if not np.allclose(actual[key], expected[key], atol=atol, rtol=rtol):
            raise ValueError(f'{name}/{key}: source replay mismatch')


def load_model(meta, job, kind, device):
    source = Path(meta['source']); sm = meta['identity']['manifest']
    ck = torch.load(source/job['name']/f'{kind}.pt', map_location='cpu', weights_only=True)
    expected = dict(manifest=sm, job=job, cache_sha256=sha256(source/'cache/index.json'))
    lock = read_json(source/'selection_lock.json')['trials'][job['name']]
    if ck['metadata'] != expected: raise ValueError('Checkpoint metadata changed')
    if kind == 'best':
        if ck['epoch'] != lock['selected_epoch'] or ck['validation'] != lock['validation']:
            raise ValueError('Best checkpoint differs from locked selection')
        validation = ck['validation']
    else:
        if ck['epoch'] != job['epochs'] or ck['history'] != read_json(source/job['name']/'history.json'):
            raise ValueError('Last checkpoint differs from full source history')
        validation = ck['history'][-1]['validation']
    model = bb.model_for(sm, source, job, device)
    model.load_state_dict(ck['model']); model.requires_grad_(False); model.eval()
    return model, validation


def error_json(rows):
    return {route: {key: value.tolist() for key, value in metrics.items()} for route, metrics in rows.items()}


def example_rows(arrays, inventory, local):
    ids = np.linspace(0, len(inventory)-1, min(4, len(inventory)), dtype=int)
    return [dict(index=int(i), key=inventory[i]['key'], end=inventory[i]['end'], mask=arrays['mask'][i].tolist(),
        **{key: (arrays[key][i]*np.asarray(local['scale'])+np.asarray(local['mean'])).tolist()
           for key in ('target', 'local_head', 'global_crop')}) for i in ids]


@torch.inference_mode()
def trial(meta, out, job, kind, device='cuda'):
    verify_manifest(meta); source = Path(meta['source']); sm = meta['identity']['manifest']
    name = job['name']+'_'+kind; path = out/'trials'/name; path.mkdir(parents=True, exist_ok=True)
    metadata = dict(manifest=meta, job=job, checkpoint=kind)
    if (path/'completion.json').exists():
        done = read_json(path/'completion.json')
        if done['metadata'] != metadata or done['status'] != 'complete': raise ValueError('Audit resume identity changed')
        verify_files(path, done['files']); progress(f'Already complete: {name}'); return
    started = time.monotonic(); model, expected = load_model(meta, job, kind, device)
    signature = bb.state_signature(model)
    original = read_json(source/'cache/statistics.json'); local = read_json(source/'cache/local_scales.json')
    val = bb.load_data(sm, 'val')
    actual = bb.validation(model, val, original, local, sm, job, device)
    require_replay(actual, expected, 'validation')
    atomic_json(dict(matched=True, actual=actual, expected=expected), path/'validation_replay.json')
    ids = np.linspace(0, len(val['x'])-1, min(meta['causal_samples'], len(val['x'])), dtype=int)
    causal = er.trained_causality(model, torch.tensor(np.asarray(val['x'][ids]), device=device))
    causal['validation_indices'] = ids.tolist(); atomic_json(causal, path/'causality.json')
    if causal['status'] == 'failed': raise ValueError(f'Trained causal/reset audit failed: {name}')
    results = {}; examples = {}
    for split in SPLITS:
        data = bb.load_data(sm, split); inventory = read_json(source/f'{split}_inventory.json')
        arrays = er.predict(model, data, original, local, meta['batch'], device)
        global_report, global_errors = bb.ab.measure(arrays['global'], data, original)
        scores = {}; errors = {}
        for route in meta['routes']:
            scores[route], errors[route] = bb.pr.measure(arrays[route], arrays['target'], arrays['mask'], local)
        previous = read_json(source/f'{split}_{kind}_errors.json')
        require_replay(global_errors, previous[job['name']+'/global'], split+'/global')
        require_replay(errors['local_head'], previous[job['name']+'/p128'], split+'/p128')
        paired = {k: bb.ab.pair_groups(errors['global_crop'][k], errors['local_head'][k], inventory) for k in bb.pr.METRICS}
        arrays['anchor_error_bps'] = arrays['anchor_error_bps'].astype(float)
        results[split] = dict(scores=scores, paired_global_crop_minus_local_head=paired,
            global_source_report=global_report, global_and_local_source_replay=True,
            anchor_error=dict(mean_signed_bps=float(arrays['anchor_error_bps'].mean()),
                              mae_bps=float(np.abs(arrays['anchor_error_bps']).mean())),
            windows=len(inventory))
        atomic_json(error_json(errors), path/f'{split}_errors.json')
        atomic_json(arrays['anchor_error_bps'].tolist(), path/f'{split}_anchor_error_bps.json')
        examples[split] = example_rows(arrays, inventory, local)
        progress(f'{name}/{split}: local={scores["local_head"]["metrics"]["primary"]:.5f} crop={scores["global_crop"]["metrics"]["primary"]:.5f}')
    if bb.state_signature(model) != signature or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen model state changed')
    atomic_json(dict(metadata=metadata, datasets=results, encoder_updates=0, head_updates=0,
                     weights_unchanged=True, seconds=time.monotonic()-started), path/'metrics.json')
    atomic_json(examples, path/'examples.json')
    files = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name != 'completion.json'}
    atomic_json(dict(status='complete', metadata=metadata, files=files), path/'completion.json')


def render_examples(records, out):
    parts = ['<!doctype html><meta charset="utf-8"><title>Frozen endpoint readouts</title>',
        '<style>body{font:15px sans-serif;margin:24px}svg{border:1px solid #ddd}summary{margin:18px 0;cursor:pointer}</style>',
        '<h1>Same endpoint state, two existing decoders</h1>',
        '<p>Previous16 observed closes, excluding current. Black=truth; blue=local head; green=global crop with predicted anchor. Not forecasts. Fixed examples, all variants retained.</p>']
    for name, datasets in records.items():
        parts.append('<details><summary>'+html.escape(name)+'</summary>')
        for split, rows in datasets.items():
            for row in rows:
                values = {key: np.asarray(row[key])[:, 0] for key in ('target', 'local_head', 'global_crop')}
                low = min(v.min() for v in values.values()); high = max(v.max() for v in values.values()); span = max(high-low, 1e-6)
                parts.append('<p>'+html.escape(split+' '+row['key']+' '+row['end'])+'</p><svg width="660" height="180" viewBox="0 0 660 180">')
                for key, color in [('target','#222'),('local_head','#287ad5'),('global_crop','#16816d')]:
                    points = ' '.join(f'{20+i*40:.2f},{160-(v-low)/span*140:.2f}' for i,v in enumerate(values[key]))
                    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
                parts.append('</svg>')
        parts.append('</details>')
    (out/'examples.html').write_text('\n'.join(parts))


def summarize(meta, out):
    reports = {}; examples = {}; lines = ['# Frozen endpoint readout diagnostic', '',
        'No fitting or updates. Original promotion decision is unchanged. A better readout establishes accessibility only.', '',
        '| trial | dataset | local primary | global-crop primary | relative change | paired95% interval |',
        '|---|---|---:|---:|---:|---|']
    for job in meta['identity']['manifest']['experiments']:
        for kind in meta['checkpoints']:
            name = job['name']+'_'+kind; path = out/'trials'/name; done = read_json(path/'completion.json')
            if done['metadata'] != dict(manifest=meta, job=job, checkpoint=kind) or done['status'] != 'complete':
                raise ValueError('Missing complete frozen trial')
            verify_files(path, done['files']); reports[name] = read_json(path/'metrics.json'); examples[name] = read_json(path/'examples.json')
            for split, row in reports[name]['datasets'].items():
                a = row['scores']['local_head']['metrics']['primary']; b = row['scores']['global_crop']['metrics']['primary']
                ci = row['paired_global_crop_minus_local_head']['primary']['all']
                interval = f'[{ci["low"]:.6f}, {ci["high"]:.6f}]' if ci['low'] is not None else 'insufficient weeks'
                change = f'{(b/a-1)*100:+.2f}%' if a > 0 else 'undefined (zero reference)'
                lines.append(f'| {name} | {split} | {a:.6f} | {b:.6f} | {change} | {interval} |')
    report = dict(schema=SCHEMA, manifest=meta, trials=reports, encoder_updates=0, head_updates=0,
                  automatic_promotion=False, source_decision=read_json(Path(meta['source'])/'decision.json'))
    atomic_json(report, out/'endpoint_metrics.json'); (out/'summary.md').write_text('\n'.join(lines)); render_examples(examples, out)
    return report


def check_output(source, out):
    sm = read_json(source/'manifest.json')
    dependencies = (source, Path(sm['source']), Path(sm['original_source']), Path(sm['sampling_source']))
    for dep in dependencies:
        dep = dep.resolve()
        if out == dep or out in dep.parents or dep in out.parents: raise ValueError('Use a separate audit output')


def run(source, out, device='cuda', batch=64):
    source, out = source.resolve(), out.resolve()
    check_output(source, out)
    progress('Checking immutable alignment checkpoints and lineage')
    identity = source_identity(source); meta = make_manifest(source, identity, batch)
    sm = identity['manifest']
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json') != meta: raise ValueError('Audit configuration changed')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty output without audit manifest')
    runtime = read_json(source/'runtime.json')
    if runtime['torch'] != str(torch.__version__) or runtime['numpy'] != np.__version__:
        raise ValueError('Restore source Torch/NumPy environment')
    out.mkdir(parents=True, exist_ok=True); atomic_json(meta, out/'manifest.json')
    # All source selections are pinned before any new research inference.
    atomic_json(dict(source_selection_sha256=sha256(source/'selection_lock.json'), identity=identity), out/'source_lock.json')
    for split in SPLITS:
        atomic_json(read_json(source/f'{split}_inventory.json'), out/f'{split}_inventory.json')
    atomic_json(dict(torch=str(torch.__version__), numpy=np.__version__, device=str(device), started_unix=time.time()), out/'runtime.json')
    (out/'completion.json').unlink(missing_ok=True)
    for job in sm['experiments']:
        for kind in meta['checkpoints']: trial(meta, out, job, kind, device)
    summarize(meta, out)
    if source_identity(source) != identity: raise ValueError('Source changed during frozen audit')
    files = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.html') and p != out/'completion.json'}
    atomic_json(dict(status='complete', source_unchanged=True, encoder_updates=0, head_updates=0,
                     statistics_fits=0, automatic_promotion=False, files=files), out/'completion.json')
    progress('Frozen endpoint audit complete; no model or readout was trained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); bb.ab.configure_runtime()
    if not torch.cuda.is_available(): raise ValueError('Formal audit runs on AutoDL CUDA')
    source = Path(args.source).resolve(); out = Path(args.out).resolve()
    import fcntl
    check_output(source, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent/('.'+out.name+'.lock')).open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Audit output already running')
        run(source, out)


if __name__ == '__main__': main()
