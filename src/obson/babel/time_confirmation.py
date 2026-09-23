"""Frozen 512/768 reconstruction confirmation on provenance-gated later history.

No optimizer, fitting, new checkpoint selection or trend probe. Old source code
and binaries stay immutable. Raw data must be an independent extended snapshot.
"""
import argparse
import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import endpoint_readout_audit as ea, endpoint_readout as er
from . import ae_context, activity_ablation as aa, architecture as ar, data as raw
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json, strict_frame
from . import time_lineage
from .progress import progress

SCHEMA = 'babel-time-confirmation-v1'
JOBS = tuple(f'w{w}_{mode}_s{s}' for w, mode in ((512, 'endpoint'), (768, 'joint')) for s in (42, 43))
KINDS = ('best', 'last')


def timestamp(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError('Missing timestamp')
    return ts.tz_convert('Asia/Shanghai').tz_localize(None) if ts.tzinfo else ts


def registry_snapshot(registry, out):
    return {str(p):sha256(p) for p in time_lineage.manifest_paths(registry,out)}


def past_records(lineage):
    """Include previously inspected cross-symbol raw audits, not just training sources."""
    found = {(r['key'], r['start'], r['end'], r['sha256']): r for r in lineage['source_records']}
    extra = {}
    for name in lineage['manifests']:
        audit = Path(name).parent/'raw_audit.json'
        if not audit.is_file():
            continue
        extra[str(audit)] = sha256(audit)
        for row in read_json(audit).get('files', []):
            if {'symbol', 'period', 'contract', 'start', 'end', 'source_hash'} <= row.keys():
                r = dict(key=f'{row["symbol"]}/{row["period"]}/{row["contract"]}',
                         start=row['start'], end=row['end'], sha256=row['source_hash'])
                found[r['key'], r['start'], r['end'], r['sha256']] = r
    if not found:
        raise ValueError('No registered raw source history')
    return sorted(found.values(), key=lambda r: (r['key'], r['start'], r['end'], r['sha256'])), extra


def cutoff_from_records(rows):
    # Source timestamps denote bar starts; require inputs after the last close.
    return max(timestamp(r['end'])+pd.Timedelta(minutes=int(r['key'].split('/')[1])) for r in rows)


def raw_inventory(root, symbols):
    files = []
    for symbol in sorted(symbols):
        for period in (15, 30, 60):
            for path in sorted((root/symbol).glob(f'*_{period}m.csv')):
                files.append((symbol, period, path.stem.rsplit('_', 1)[0], path))
    return files


def audit_snapshot(root, history, cutoff):
    """Validate all files and historical prefixes before any model evaluation."""
    by_key = {}
    for row in history:
        by_key.setdefault(row['key'], []).append(row)
    files = []; issues = []; seen = set(); hashes = {}; later = 0
    for symbol, period, contract, path in raw_inventory(root, {k.split('/')[0] for k in by_key}):
        key = f'{symbol}/{period}/{contract}'; seen.add(key)
        digest = sha256(path); hashes[str(path.resolve())] = digest
        frame = strict_frame(path, period)
        for old in by_key.get(key, []):
            prefix = frame[(frame.datetime >= timestamp(old['start'])) & (frame.datetime <= timestamp(old['end']))].reset_index(drop=True)
            actual = hashlib.sha256(pd.util.hash_pandas_object(prefix, index=False).values.tobytes()).hexdigest()
            if actual != old['sha256']:
                issues.append(dict(key=key, reason='registered_raw_prefix_changed', start=old['start'], end=old['end']))
        n = int((frame.datetime > cutoff).sum()); later += n
        files.append(dict(key=key, path=str(path.resolve()), file_sha256=digest, rows=len(frame), later_rows=n,
                          start=str(frame.datetime.iloc[0]), end=str(frame.datetime.iloc[-1]),
                          sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()))
        if len(files) % 200 == 0:
            progress(f'raw eligibility: {len(files)} files, {later} rows beyond old history')
    issues.extend(dict(key=k, reason='registered_contract_missing') for k in sorted(set(by_key)-seen))
    return dict(files=files, sources=[{k:r[k] for k in ('key','start','end','sha256')} for r in files],
                hashes=hashes, later_rows=later, issues=issues, cutoff=str(cutoff),
                status='blocked' if issues or later == 0 else 'eligible',
                reason='raw_provenance_failed' if issues else ('no_later_raw_history' if later == 0 else None))


def eligible_ends(series, cutoff, asof):
    # 512 bars of causal warmup; all128 scored-window inputs after global cutoff.
    ends = np.arange(127, len(series.frame), 128)
    ends = ends[ends >= 511]
    starts = series.frame.datetime.to_numpy()
    closed = series.ends
    return ends[(starts[ends-127] > cutoff.to_datetime64()) & (closed[ends] <= asof.to_datetime64()) &
                series.main[ends] & ~np.isnat(series.sessions[ends])]


def prepare_data(root, audit, cutoff, asof, stats, out):
    symbols = sorted({r['key'].split('/')[0] for r in audit['sources']})
    series, warnings = raw.load_series(root, symbols, (15,30,60), asof=str(asof))
    xs = []; inventory = []
    for s in series:
        ends = eligible_ends(s, cutoff, asof)
        if not len(ends):
            continue
        x = np.column_stack((ae_context.encode_context(s.frame, s.period, 'ema8_32')['x'],
                             aa.activity_features(s.frame, s.period)[0]))
        for end in ends:
            xs.append(x[end-127:end+1])
            inventory.append(dict(key=s.key, symbol=s.code, period=s.period, row=int(end),
                start=str(s.frame.datetime.iloc[end-127]), end=str(s.frame.datetime.iloc[end]),
                available_at=str(pd.Timestamp(s.ends[end])), month=str(s.frame.datetime.iloc[end].to_period('M')),
                week=str(pd.Timestamp(s.sessions[end]).to_period('W-SUN'))))
    atomic_json(dict(warnings=warnings, windows=len(xs), weeks=len({r['week'] for r in inventory}),
                     cutoff=str(cutoff), asof=str(asof)), out/'coverage.json')
    atomic_json(inventory, out/'inventory.json')
    if not xs:
        return None, inventory
    x = np.stack(xs); y, mask = ar.ordered_targets(x)
    x, y, mask = ar.normalize(x, y, mask, stats)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite normalized confirmation data')
    arrays = dict(x=x, y=y, mask=mask); cache = out/'cache'; cache.mkdir(exist_ok=True)
    for name, value in arrays.items():
        np.save(cache/f'{name}.npy', value)
    atomic_json({p.name:sha256(p) for p in cache.iterdir()}, out/'cache_files.json')
    return arrays, inventory


def guard_inputs(meta, out):
    for p, digest in meta['raw_hashes'].items():
        if not Path(p).is_file() or sha256(p) != digest:
            raise ValueError('Raw snapshot changed')
    if registry_snapshot(Path(meta['registry']), out) != meta['registry_files']:
        raise ValueError('Research registry changed during confirmation')
    for p, h in meta['extra_audits'].items():
        if sha256(p) != h:
            raise ValueError('Prior raw audit changed')
    actual = {str(p.resolve()) for _,_,_,p in raw_inventory(Path(meta['root']), meta['symbols'])}
    if actual != set(meta['raw_hashes']):
        raise ValueError('Raw snapshot membership changed')
    if meta['code_sha256'] != code_identity():
        raise ValueError('Confirmation code changed')


def code_identity():
    return ea.code_identity() | {Path(__file__).name:sha256(__file__),
                                 Path(time_lineage.__file__).name:sha256(time_lineage.__file__)}


def comparison_checks(errors, inventory):
    pairs = {}; checks = []
    for kind in KINDS:
        for seed in (42,43):
            a = f'w768_joint_s{seed}/{kind}'; b = f'w512_endpoint_s{seed}/{kind}'
            for task in ('global', 'recent'):
                for metric in errors[a][task]:
                    pairs[f'{a}_minus_{b}/{task}/{metric}'] = ea.bb.ab.pair_groups(
                        errors[a][task][metric], errors[b][task][metric], inventory)
            # Confirm overall gain separately from the original retention bounds.
            rules = [('global','primary',1.,'gain'), ('recent','primary',1.,'gain')]
            rules += [('global','primary',1.05,'retention')]
            rules += [('global',m,1.10,'retention') for m in ('path','changes','body','activity','close_mae_bps')]
            for task, metric, factor, role in rules:
                row = ea.bb.ab.pair_groups(errors[a][task][metric], factor*errors[b][task][metric], inventory)['all']
                passed = row['supported'] and row['high'] is not None and (row['high'] < 0 if role == 'gain' else row['high'] <= 0)
                checks.append(dict(candidate=a, reference=b, task=task, metric=metric, factor=factor, role=role,
                                   evidence=row, passed=bool(passed)))
    supported = all(r['evidence']['supported'] for r in checks)
    return pairs, dict(status='insufficient_support' if not supported else
        ('gain_and_retention_confirmed' if all(r['passed'] for r in checks) else 'mixed_or_unconfirmed'),
        checks=checks, automatic_promotion=False,
        scope='New-time reconstruction confirmation only; not the old full multi-position promotion test.')


@torch.inference_mode()
def evaluate(meta, out, arrays, inventory, device):
    source = Path(meta['source']); sm = meta['identity']['manifest']
    runtime=read_json(source/'runtime.json')
    if runtime['torch'] != str(torch.__version__) or runtime['numpy'] != np.__version__:
        raise ValueError('Restore source Torch/NumPy environment before frozen inference')
    stats = read_json(source/'cache/statistics.json'); local = read_json(source/'cache/local_scales.json')
    jobs = [j for j in sm['experiments'] if j['name'] in JOBS]
    if {j['name'] for j in jobs} != set(JOBS):
        raise ValueError('Both locked widths and seeds are required')
    scores = {}; errors = {}; replay = {}; examples = {}
    ids = np.linspace(0,len(inventory)-1,min(4,len(inventory)),dtype=int)
    val = ea.bb.load_data(sm, 'val')
    for job in jobs:
        for kind in KINDS:
            name = job['name']+'/'+kind; progress(f'Frozen evaluation: {name}')
            model, expected = ea.load_model(meta, job, kind, device)
            before = ea.bb.state_signature(model)
            actual = ea.bb.validation(model, val, stats, local, sm, job, device)
            ea.require_replay(actual, expected, name+'/validation'); replay[name] = actual
            result = er.predict(model, arrays, stats, local, meta['batch'], device)
            g, ge = ea.bb.ab.measure(result['global'], arrays, stats)
            r, re = ea.bb.pr.measure(result['global_crop'], result['target'], result['mask'], local)
            scores[name] = dict(global_=g, recent=r); scores[name]['global'] = scores[name].pop('global_')
            errors[name] = dict(global_=ge, recent=re); errors[name]['global'] = errors[name].pop('global_')
            examples[name] = [dict(index=int(i), global_prediction=result['global'][i].tolist(),
                                  recent_prediction=result['global_crop'][i].tolist()) for i in ids]
            if before != ea.bb.state_signature(model) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError('Frozen model changed')
            del model
    # Same frozen train-only PCA bases, at both ranks; no new SVD or scales.
    for width in (512,768):
        with np.load(source/f'cache/pca{width}.npz',allow_pickle=False) as f:
            prediction = ea.bb.ab.pca_predict(dict(f), arrays)
        tensor = torch.tensor(prediction); y = torch.tensor(arrays['y']); mask = torch.tensor(arrays['mask'])
        target, valid = ea.bb.ba.local_targets(y, mask, torch.full((len(y),1),128), stats, local)
        crop = er.crop_prediction(tensor, stats, local).numpy()
        g, ge = ea.bb.ab.measure(prediction, arrays, stats)
        r, re = ea.bb.pr.measure(crop, target[:,0].numpy(), valid[:,0].numpy(), local)
        scores[f'pca{width}'] = {'global':g,'recent':r}; errors[f'pca{width}'] = {'global':ge,'recent':re}
    pairs, decision = comparison_checks(errors, inventory)
    for name in JOBS:
        for kind in KINDS:
            ref = 'pca'+name.split('_')[0][1:]; candidate = name+'/'+kind
            for task in ('global','recent'):
                for metric in errors[candidate][task]:
                    pairs[f'{candidate}_minus_{ref}/{task}/{metric}'] = ea.bb.ab.pair_groups(
                        errors[candidate][task][metric], errors[ref][task][metric], inventory)
    atomic_json(replay, out/'validation_replay.json')
    atomic_json({n:{t:{k:v.tolist() for k,v in e.items()} for t,e in rows.items()} for n,rows in errors.items()},out/'errors.json')
    atomic_json(dict(scores=scores, paired=pairs, encoder_updates=0, statistics_fits=0),out/'confirmation_metrics.json')
    decision['prior_decision'] = read_json(source/'decision.json')
    atomic_json(decision,out/'decision.json')
    atomic_json(dict(indices=ids.tolist(), normalized=True, statistics=stats, local_scales=local,
        truth=[dict(index=int(i),global_truth=arrays['y'][i].tolist(),global_mask=arrays['mask'][i].tolist()) for i in ids],
        predictions=examples),out/'examples.json')
    lines = ['# Frozen later-time reconstruction', '', 'No training, fitting or automatic promotion.', '',
             f'Decision: {decision["status"]}', '', '| model | global primary | recent primary |', '|---|---:|---:|']
    for name,row in scores.items():
        lines.append(f'| {name} | {row["global"]["metrics"]["primary"]:.6f} | {row["recent"]["metrics"]["primary"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(lines))


def execute_locked(meta, out, device):
    """Replay the same attempt after interruption, never move its data boundary."""
    started=time.monotonic(); source=Path(meta['source']); guard_inputs(meta,out)
    if ea.source_identity(source) != meta['identity']:
        raise ValueError('Source changed')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json')
        for name,digest in done['files'].items():
            if sha256(out/name) != digest:
                raise ValueError('Completed report changed')
        progress('Already complete; use export for the same report'); return 0
    if (out/'evaluation_lock.json').exists():
        lock=read_json(out/'evaluation_lock.json')
        for name in ('manifest','inventory','cache_files'):
            if sha256(out/f'{name}.json') != lock[f'{name}_sha256']:
                raise ValueError('Locked evaluation identity changed')
        for name,digest in read_json(out/'cache_files.json').items():
            if sha256(out/'cache'/name) != digest:
                raise ValueError('Locked data cache changed')
        arrays={k:np.load(out/f'cache/{k}.npy',allow_pickle=False) for k in ('x','y','mask')}
        inventory=read_json(out/'inventory.json')
    else:
        stats=read_json(source/'cache/statistics.json')
        audit=read_json(out/'raw_audit.json')
        arrays,inventory=prepare_data(Path(meta['root']),audit,timestamp(meta['cutoff']),timestamp(meta['asof']),stats,out)
        if arrays is None:
            atomic_json(dict(status='blocked',reason='no_eligible_later_windows'),out/'run_state.json'); return 3
        guard_inputs(meta,out)
        atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),inventory_sha256=sha256(out/'inventory.json'),
                         cache_files_sha256=sha256(out/'cache_files.json')),out/'evaluation_lock.json')
    atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        started_unix=time.time()),out/'runtime.json')
    evaluate(meta,out,arrays,inventory,device)
    guard_inputs(meta,out)
    if ea.source_identity(source) != meta['identity']:
        raise ValueError('Source changed during confirmation')
    atomic_json(dict(status='complete'),out/'run_state.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'}
    atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,statistics_fits=0,
                     files=files,seconds_this_attempt=time.monotonic()-started),out/'completion.json')
    return 0


def check_inputs(source, registry, root, out, batch):
    """Report actual resolved paths before expensive provenance or GPU work."""
    progress(f'Input paths: raw={root}; source={source}; registry={registry}; output={out}; batch={batch}')
    for dep in (source,root,registry):
        if out == dep or out in dep.parents or (dep != registry and dep in out.parents):
            raise ValueError(f'Output directory overlaps an input: output={out}, input={dep}; set BABEL_TIME_RUN to a separate directory')
    if batch < 1:
        raise ValueError(f'BABEL_TIME_BATCH must be positive; received {batch}')
    if not root.is_dir():
        raise ValueError(f'Raw data directory does not exist: {root}; set BABEL_TIME_ROOT to the actual contract-data directory (usual AutoDL path: /root/autodl-tmp/data/contracts)')
    if not registry.is_dir():
        raise ValueError(f'Research registry directory does not exist: {registry}; check BABEL_TIME_REGISTRY')
    if source == registry or registry not in source.parents:
        raise ValueError(f'Model source must be inside the research registry after resolving symlinks: source={source}, registry={registry}; check BABEL_TIME_SOURCE and BABEL_TIME_REGISTRY')
    if not source.is_dir():
        raise ValueError(f'Model source directory does not exist: {source}; check BABEL_TIME_SOURCE (expected completed babel_bar_alignment run)')
    for name in ('manifest.json','completion.json','selection_lock.json'):
        if not (source/name).is_file():
            raise ValueError(f'Incomplete model source: missing {source/name}; restore the completed alignment run')


def run(source, registry, root, out, device='cuda', batch=128, asof=None):
    source, registry, root, out = (p.resolve() for p in (source,registry,root,out))
    check_inputs(source,registry,root,out,batch)
    out.mkdir(parents=True,exist_ok=True)
    # Resume only the exact frozen attempt; a new data snapshot needs a new path.
    if any(out.iterdir()):
        if not (out/'manifest.json').exists():
            if (out/'run_state.json').exists() and read_json(out/'run_state.json')['status']=='blocked':
                progress('Existing blocked attempt; new data requires a new output path'); return 3
            raise ValueError('Incomplete source audit; export it or use a new output path')
        meta=read_json(out/'manifest.json')
        if any(meta[k]!=str(v) for k,v in (('source',source),('registry',registry),('root',root))) or meta['batch']!=batch:
            raise ValueError('Resume configuration changed')
        if asof is not None and timestamp(asof)!=timestamp(meta['asof']):
            raise ValueError('Resume asof changed')
        return execute_locked(meta,out,device)
    try:
        atomic_json(dict(status='running'),out/'run_state.json')
        progress('Checking immutable model sources and registered data history; no GPU work yet')
        identity = ea.source_identity(source)
        lineage = time_lineage.scan_lineage(registry, source, out)
        atomic_json(lineage, out/'lineage.json')
        if not lineage['verified']:
            progress(f'Source lineage blocked: {len(lineage["issues"])} issues; see lineage.json')
            atomic_json(dict(status='blocked',reason='unresolved_registry_lineage'),out/'run_state.json'); return 3
        history, extra = past_records(lineage); cutoff = cutoff_from_records(history)
        now = pd.Timestamp.now(tz='Asia/Shanghai').tz_localize(None)
        asof = timestamp(asof) if asof else now
        if asof > now or asof <= cutoff:
            raise ValueError('asof must be after old history and not in the future')
        snapshots = registry_snapshot(registry,out)
        audit = audit_snapshot(root, history, cutoff); atomic_json(audit,out/'raw_audit.json')
        if audit['status'] == 'blocked':
            atomic_json(dict(status='blocked',reason=audit['reason'],cutoff=str(cutoff),later_rows=audit['later_rows']),out/'run_state.json'); return 3
        meta = dict(schema=SCHEMA,source=str(source),registry=str(registry),root=str(root),identity=identity,
            code_sha256=code_identity(),raw_hashes=audit['hashes'],sources=audit['sources'],
            symbols=sorted({r['key'].split('/')[0] for r in audit['sources']}),registry_files=snapshots,
            extra_audits=extra,cutoff=str(cutoff),asof=str(asof),batch=batch,jobs=list(JOBS),kinds=list(KINDS),
            encoder_updates=0,statistics_fits=0,minimum_history=512,stride=128,
            selection='Source-locked best and fixed last, both seeds; no new selection.',
            protocol='All128 inputs start strictly after maximum registered raw bar close; past causal warmup permitted. Same-contract windows, lagged main selection. No end-cap or score-dependent sampling.',
            decision='Weekly paired support>=50 windows and>=5 weeks. Confirm global/recent gain separately; retain original global primary5% and family10% margins. No automatic promotion.',
            limits='Registry-scoped provenance, not certification of deleted/external human research. Frozen reconstruction only; no768 trend-readout claim.')
        atomic_json(meta,out/'manifest.json')
        return execute_locked(meta,out,device)
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'run_state.json')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('checkpoints/babel_bar_alignment'))
    p.add_argument('--registry',type=Path,default=Path('checkpoints'))
    p.add_argument('--root',type=Path,default=Path('data/contracts'))
    p.add_argument('--out',type=Path,default=Path('checkpoints/babel_time_confirmation'))
    p.add_argument('--batch',type=int,default=128);p.add_argument('--asof');p.add_argument('--device',default='cuda')
    a=p.parse_args();ea.bb.ab.configure_runtime();a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another confirmation process owns this output')
        raise SystemExit(run(a.source,a.registry,a.root,a.out,a.device,a.batch,a.asof))


if __name__ == '__main__':
    main()
