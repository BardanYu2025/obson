"""One frozen-state use-case audit: train-only PCA and matched ridge readouts."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import window_state as ws, window_state_delivery as wd, window_state_probe as probe
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-window-state-probe-v1'
SPLITS = ('train', 'val', 'test', 'cross_research')


def code_identity():
    return wd.code_identity() | {Path(probe.__file__).name:sha256(probe.__file__), Path(__file__).name:sha256(__file__)}


def source_identity(source):
    meta = read_json(source/'manifest.json'); done = read_json(source/'completion.json')
    if meta['schema'] != wd.SCHEMA or meta['code_sha256'] != wd.code_identity() or done['status'] != 'complete' or not done['source_unchanged']:
        raise ValueError('Completed frozen window-state delivery required')
    if any(meta[k] != 0 or done[k] != 0 for k in ('encoder_updates','head_updates','statistics_fits')):
        raise ValueError('Window-state delivery must not update source models or statistics')
    ws.ea.bb.ab.sc.verify_worker_files(source, done['files'])
    if wd.source_identity(Path(meta['source'])) != meta['identity']: raise ValueError('Endpoint lineage changed')
    index = read_json(source/'bundle/index.json'); certificate = read_json(source/'bundle/validation.json')
    if (index['delivery_manifest'] != meta or index['code_sha256'] != ws.code_identity()
            or certificate['status'] != 'passed' or certificate['index_sha256'] != sha256(source/'bundle/index.json')):
        raise ValueError('Invalid portable model certificate')
    alignment = meta['identity']['manifest']['identity']['manifest']
    sampling = read_json(Path(alignment['sampling_source'])/'manifest.json')
    coverage = Path(sampling['coverage_source'])
    actual = ws.ea.bb.ws.coverage_identity(coverage, Path(alignment['original_source']))
    if actual != sampling['coverage_identity']: raise ValueError('Coverage lineage changed')
    return dict(manifest=meta, files=done['files'] | {'completion.json':sha256(source/'completion.json')},
        coverage=str(coverage), coverage_identity=actual)


def alignment(meta): return meta['identity']['manifest']['identity']['manifest']['identity']['manifest']


def inventory(meta):
    result = {}
    for split in SPLITS:
        rows = read_json(Path(meta['identity']['coverage'])/f'{split}_windows.json')
        result[split] = [{k:r[k] for k in ('key','symbol','period','row','end','month')} |
            dict(week=str(pd.Timestamp(r['end']).to_period('W'))) for r in rows]
    return result


def make_manifest(source, identity, batch=256):
    if batch < 1: raise ValueError('Positive extraction batch required')
    sm = identity['manifest']['identity']['manifest']['identity']['manifest']
    width = sm['widths'][0]
    return dict(schema=SCHEMA, source=str(source.resolve()), identity=identity, code_sha256=code_identity(),
        batch=batch, pca_rank=width, variants=['embedding42','embedding43',f'pca{width}','raw3584','current28'],
        horizons=list(probe.HORIZONS), targets=list(probe.NAMES), alphas=list(probe.ALPHAS), threshold=probe.DIRECTION_THRESHOLD,
        encoder_updates=0, neural_reader_updates=0, precision='fp32 frozen extraction, float64 PCA/ridge',
        selection='Same five alphas, one alpha per representation by validation equal-six-target standardized MSE; exact ties prefer larger alpha. All selections locked before research extraction.',
        labels='Observed16/64 close paths, h-1 internal returns including completed current bar. Signed efficiency, OLS path R2, log realized return RMS. Exact formulas are reference values, not a learnable market truth.',
        data='Original fixed train/val/test and cross-symbol research windows from verified coverage; no sampling-pool expansion, no random split, no independent-holdout claim.',
        decision=dict(primary_improvement=.05, family_retention=.05, automatic_promotion=False,
            rule='Both seeds and both research sets: all six R2 positive, primary >=5% improvement versus PCA/raw/current with upper weekly CI <0 for candidate-0.95*reference; each family point loss <=1.05*reference.'),
        stop='One linear diagnostic. Do not expand models, add MLP sweeps, change cutoffs, or train encoders to fit these rules if evidence fails.')


def verify_cache(meta, out, split):
    index = read_json(out/f'cache/{split}_index.json')
    if index['manifest_sha256'] != sha256(out/'manifest.json'): raise ValueError('Probe cache manifest changed')
    verify_files(out/'cache', index['files'])
    return index


def check_selection(out):
    lock = read_json(out/'selection_lock.json')
    if lock['manifest_sha256'] != sha256(out/'manifest.json'): raise ValueError('Readout selection manifest changed')
    verify_files(out, lock['files'])
    for split,digest in lock['cache_indexes'].items():
        if sha256(out/f'cache/{split}_index.json') != digest: raise ValueError('Selection input cache changed')
    return lock


@torch.inference_mode()
def prepare(meta, out, split, rows, device):
    if split not in SPLITS: raise ValueError('Unknown split')
    if split in SPLITS[2:]: check_selection(out)
    cache = out/'cache'; cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists(): verify_cache(meta,out,split); return
    x = np.asarray(ws.ea.bb.load_data(alignment(meta), split)['x'])
    if x.shape != (len(rows),128,28): raise ValueError('Coverage/window count mismatch')
    arrays = dict(raw=x.reshape(len(x),-1), current=x[:,-1])
    stats = read_json(Path(alignment(meta)['original_source'])/'cache/statistics.json')
    arrays['targets'] = probe.descriptors(x.astype(np.float64)*stats['x_scale']+stats['x_mean'])
    for seed in (42,43):
        model = ws.load_bundle(Path(meta['source'])/'bundle', seed, 'AUDIT/15/contract', 15, device).model
        before = ws.ea.bb.state_signature(model); values = []
        for start in range(0,len(x),meta['batch']):
            tensor = torch.tensor(x[start:start+meta['batch']], device=device)
            values.append(model.encoder(tensor)[:,-1].cpu().numpy())
        if before != ws.ea.bb.state_signature(model) or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Encoder changed or unfrozen')
        arrays[f'embedding{seed}'] = np.concatenate(values)
        del model
    if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite cache')
    np.savez(cache/f'{split}.npz', **arrays)
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'), windows=len(x),
        files={f'{split}.npz':sha256(cache/f'{split}.npz')}, encoder_updates=0),cache/f'{split}_index.json')
    progress(f'Frozen extraction complete: {split}, {len(x)} windows, both seeds')


def arrays(out, split):
    with np.load(out/f'cache/{split}.npz',allow_pickle=False) as f: return dict(f)


def representation(name, bank, pca, raw_stats):
    if name.startswith('pca'): return probe.normalize(bank['raw'],raw_stats) @ pca
    return bank['raw' if name=='raw3584' else 'current' if name=='current28' else name]


def fit(meta, out, device):
    if (out/'selection_lock.json').exists(): check_selection(out); return
    for split in ('train','val'): verify_cache(meta,out,split)
    train, val = arrays(out,'train'), arrays(out,'val')
    target_stats = probe.scales(train['targets']); raw_stats = probe.scales(train['raw'])
    y, vy = (probe.normalize(b['targets'],target_stats) for b in (train,val))
    x = probe.normalize(train['raw'],raw_stats)
    rank = meta['pca_rank']
    if rank >= min(x.shape): raise ValueError('Insufficient fixed training rows for declared PCA rank')
    progress('Fitting one train-only input PCA covariance and closed-form ridge readouts')
    eigen = probe.eigensystem(x,device); pca = eigen[1][:,-rank:].flip(1).cpu().numpy()
    np.savez(out/'pca.npz', components=pca)
    fits = {}; tensors = {}
    for name in meta['variants']:
        tr = representation(name,train,pca,raw_stats); va = representation(name,val,pca,raw_stats)
        head, grid = probe.ridge_select(tr,y,va,vy,device,eigen if name=='raw3584' else None)
        tensors[name+'_weights'] = head.pop('weights'); tensors[name+'_intercept'] = head.pop('intercept')
        fits[name] = dict(**head, candidates=grid, input_width=tr.shape[1], parameters=(tr.shape[1]+1)*6)
        progress(f'{name}: locked alpha={head["alpha"]:g}, validation NMSE={head["validation_nmse"]:.5f}')
    np.savez(out/'heads.npz', **tensors)
    atomic_json(dict(target_stats=target_stats, raw_stats=raw_stats, readouts=fits,
        pca=dict(rank=rank, input_width=x.shape[1], training_windows=len(x), method='Exact FP64 eigendecomposition of train-standardized input covariance',
                 explained_variance_ratio=float(eigen[0][-rank:].sum()/eigen[0].sum().clamp_min(1e-15))),
        train_mean=train['targets'].mean(0).tolist(), fits_use_train_only=True),out/'fit.json')
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),
        files={n:sha256(out/n) for n in ('pca.npz','heads.npz','fit.json')},
        cache_indexes={s:sha256(out/f'cache/{s}_index.json') for s in ('train','val')}),out/'selection_lock.json')


def group_scores(pred, truth, stats, rows):
    result = {}
    for field in ('symbol','period','key'):
        for value in sorted({str(r[field]) for r in rows}):
            ids = np.array([str(r[field])==value for r in rows])
            result[field+'/'+value] = probe.measure(pred[ids],truth[ids],stats)[0]
    return result


def evaluate(meta, out, inventories):
    check_selection(out); fitted = read_json(out/'fit.json')
    with np.load(out/'heads.npz',allow_pickle=False) as f: heads = dict(f)
    with np.load(out/'pca.npz',allow_pickle=False) as f: pca = f['components']
    stats = fitted['target_stats']; report = {}; decisions = []
    for split in SPLITS[2:]:
        verify_cache(meta,out,split); bank = arrays(out,split); truth = bank['targets']; rows = inventories[split]
        predictions = {}; scores = {}; errors = {}; groups = {}; paired = {}
        for name in meta['variants']+['train_mean']:
            if name == 'train_mean': pred = np.tile(fitted['train_mean'],(len(truth),1))
            else:
                features = representation(name,bank,pca,fitted['raw_stats'])
                head = dict(fitted['readouts'][name], weights=heads[name+'_weights'], intercept=heads[name+'_intercept'])
                pred = probe.predict(head,features)*stats['scale']+stats['mean']
            predictions[name] = pred.tolist(); scores[name], errors[name] = probe.measure(pred,truth,stats)
            groups[name] = group_scores(pred,truth,stats,rows)
        for seed in (42,43):
            candidate = f'embedding{seed}'
            for reference in (f'pca{meta["pca_rank"]}','raw3584','current28'):
                key = candidate+'_minus_'+reference
                paired[key] = {k:ws.ea.bb.ab.pair_groups(errors[candidate][k],errors[reference][k],rows) for k in errors[candidate]}
                gate = ws.ea.bb.ab.pair_groups(errors[candidate]['primary'],.95*errors[reference]['primary'],rows)['all']
                readable = all(t['r2'] is not None and t['r2']>0 for t in scores[candidate]['targets'])
                retention = all(scores[candidate]['families'][f] <= 1.05*scores[reference]['families'][f] for f in probe.FAMILIES)
                passed = readable and retention and gate['supported'] and gate['high'] is not None and gate['high']<0
                decisions.append(dict(split=split, candidate=candidate, reference=reference, all_targets_positive_r2=readable,
                    family_retention=retention, improvement_margin_interval=gate, passed=bool(passed)))
        report[split] = dict(scores=scores, groups=groups, paired=paired)
        atomic_json(dict(targets=truth.tolist(), predictions=predictions,
            per_window_errors={n:{k:v.tolist() for k,v in e.items()} for n,e in errors.items()}),out/f'{split}_predictions.json')
    decision = dict(status='consistent_incremental_readout_evidence' if all(r['passed'] for r in decisions) else 'no_uniform_incremental_evidence',
        checks=decisions, automatic_promotion=False, encoder_updates=0,
        next='Review this fixed diagnostic, retain original baseline. No automatic MLP, threshold, architecture or training expansion.')
    atomic_json(dict(datasets=report, decision=decision, scope='Historical formula readability on reused research sets; not forecasting or execution signals'),out/'state_probe_metrics.json')
    atomic_json(decision,out/'decision.json')
    (out/'summary.md').write_text('# Frozen state readout diagnostic\n\n'+decision['status']+'\n\nZero encoder updates; fixed PCA and linear ridge only. See per-target and direction-class support before interpreting aggregate scores.\n')


def check_output(source, out):
    source,out=source.resolve(),out.resolve()
    if out==source or source in out.parents or out in source.parents: raise ValueError('Separate readout output required')
    meta=read_json(source/'manifest.json')
    endpoint=Path(meta['source']).resolve(); al=Path(meta['identity']['manifest']['source'])
    if out==endpoint or endpoint in out.parents or out in endpoint.parents: raise ValueError('Output overlaps endpoint source')
    ws.ea.check_output(al,out)


def run(source, out, batch=256, device='cuda'):
    source,out=source.resolve(),out.resolve();check_output(source,out)
    progress('Verifying completed baseline and immutable train/validation provenance')
    identity=source_identity(source);meta=make_manifest(source,identity,batch)
    old_runtime=read_json(Path(identity['manifest']['source'])/'runtime.json')
    if old_runtime['torch']!=str(torch.__version__) or old_runtime['numpy']!=np.__version__:
        raise ValueError('Restore source Torch/NumPy versions')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta: raise ValueError('Probe configuration/source changed; use a new directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty output without matching manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json')
        if done['status']!='complete':raise ValueError('Invalid completion')
        ws.ea.bb.ab.sc.verify_worker_files(out,done['files']);check_selection(out)
        progress('Already complete; no inference or readout refit');return
    started=time.time();inventories=inventory(meta);audit=probe.inventory_audit(inventories)
    audit['source_boundaries']=identity['coverage_identity'].get('boundaries')
    atomic_json(audit,out/'data_audit.json')
    for split,rows in inventories.items(): atomic_json(rows,out/f'{split}_inventory.json')
    for split in SPLITS[:2]:prepare(meta,out,split,inventories[split],device)
    fit(meta,out,device)
    for split in SPLITS[2:]:prepare(meta,out,split,inventories[split],device)
    evaluate(meta,out,inventories)
    if source_identity(source)!=identity:raise ValueError('Source changed during readout audit')
    atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__,device=device,seconds=time.time()-started,
        encoder_updates=0,neural_reader_updates=0,readout_method='closed_form_ridge',pca_fits=1,ridge_candidates=5*len(meta['variants'])),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.md') and p!=out/'completion.json'}
    atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,files=files),out/'completion.json')
    progress('Frozen state readout audit complete; reports ready')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=256)
    a=p.parse_args();ws.ea.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal model extraction and readout fit run on AutoDL CUDA')
    source,out=Path(a.source).resolve(),Path(a.out).resolve();check_output(source,out)
    if read_json(source/'bundle/index.json')['delivery_manifest']['identity']['manifest']['identity']['manifest']['widths'][0]!=512:
        raise ValueError('Formal protocol requires frozen512 baseline')
    import fcntl
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/('.'+out.name+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Readout output already running')
        run(source,out,a.batch)


if __name__=='__main__':main()
