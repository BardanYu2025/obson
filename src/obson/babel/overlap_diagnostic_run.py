"""AutoDL-only, zero-fit overlapping-window diagnostic of the two primary768 states."""
import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from . import utility_probe_run as ur, overlap_diagnostic as od
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA='babel-frozen-overlap768-v1'
MODELS=('control_s42','control_s43')
SPLITS=('test','cross_research')


def code_identity():
    return ur.code_identity()|{Path(od.__file__).name:sha256(od.__file__),Path(__file__).name:sha256(__file__)}


def banks(identity):
    original=Path(identity['manifest']['identity']['manifest']['original_source'])
    am=read_json(original/'manifest.json');result={}
    for split in SPLITS:
        cross=split=='cross_research';bank=Path(am['cross_run'])/'cache' if cross else Path(am['bank'])
        if cross:
            index_path=bank/'index.json';ur.bb.cov.file_check(index_path,am['source_identity'][str(index_path.resolve())])
            pinned=read_json(index_path)['files']
        files={}
        for suffix in ('x.npy','sequences.json'):
            p=bank/f'test_{suffix}'
            expected=pinned[p.name] if cross else am['source_identity'][str(p.resolve())]
            ur.bb.cov.file_check(p,expected);files[str(p)]=expected
        result[split]=dict(directory=str(bank),files=files)
    return result


def make_manifest(source,identity,packed,batch):
    context=ur.make_manifest(source,identity,batch);context['models']=list(MODELS)
    return dict(schema=SCHEMA,source=str(source),identity=identity,packed=packed,batch=batch,code_sha256=code_identity(),
        loader=context,models=list(MODELS),shifts=list(od.SHIFTS),updates=0,fits=0,
        cohort='Original research endpoint B; earlier128 A ends B-shift. Same contract/packed partition; earlier global row>=511. Intersection cohort shared by all three shifts; retain all exclusions.',
        coordinates='Both current bars excluded: shared A[shift:127] vs B[:127-shift]. Shape paths subtract each own first shared predicted close; native errors preserve original anchor. No truth correction of predictions.',
        selection='Two primary control source-selected best, no path branch, seed/weight/sample selection or fitting. All source endpoints replay before shifts.',
        assessment='Descriptive diagnostic, no model promotion or training trigger. Weekly paired B-minus-A true-error intervals; >=50 pairs and5 session weeks for supported comparisons. No arbitrary similarity threshold.',
        limitation='Window movement jointly changes context and positions; does not isolate either cause. B knows later observed bars than A. Shifted endpoints need not be main-contract sampled. Reused research sets; no future prediction.',
        state_distance='Cosine/RMS descriptive only: coordinates describe relative window locations; no required temporal monotonicity or semantic-distance interpretation.')


def normalized(raw,stats):
    y,mask=ur.bb.ar.ordered_targets(raw)
    return ur.bb.ar.normalize(raw,y,mask,stats)


def prepare(meta,out):
    context=meta['loader'];inventories=ur.inventories(context);ur.up.probe.inventory_audit(inventories)
    stats=read_json(Path(meta['identity']['manifest']['source'])/'cache/statistics.json');result={}
    for split in SPLITS:
        root=Path(meta['packed'][split]['directory']);bank=np.load(root/'test_x.npy',mmap_mode='r',allow_pickle=False)
        specs=read_json(root/'test_sequences.json');rows=inventories[split]
        if bank.ndim!=2 or bank.shape[1]!=28:raise ValueError('Expected chronological raw28 bank')
        raw=ur.bb.ab.extract_windows(bank,specs,len(rows));x,y,mask=normalized(raw,stats)
        cached=ur.bb.load_data(ur.alignment(context),split);replay={}
        for name,value in [('x',x),('y',y),('mask',mask)]:
            ref=np.asarray(cached[name])
            if value.shape!=ref.shape:raise ValueError(f'Original endpoint cache shape differs: {split}/{name}')
            diff=float(np.abs(value.astype(float)-ref.astype(float)).max())
            if (name=='mask' and not np.array_equal(value,ref)) or not np.allclose(value,ref,atol=1e-6,rtol=2e-5):
                raise ValueError(f'Original endpoint cache replay failed: {split}/{name}: {diff}')
            replay[name]=diff
        plan,excluded=od.pair_plan(specs,rows,len(bank),tuple(meta['shifts']))
        if not plan:raise ValueError(f'No eligible overlap pairs: {split}')
        atomic_json(dict(rows=plan,excluded=excluded,source_windows=len(rows),cache_replay_max_abs=replay),out/f'{split}_plan.json')
        ids=np.array([p['index'] for p in plan]);views={0:dict(x=x[ids],y=y[ids],mask=mask[ids])}
        for d in meta['shifts']:
            ax,ay,am=normalized(od.windows(bank,plan,d),stats);views[d]=dict(x=ax,y=ay,mask=am)
        result[split]=dict(rows=plan,views=views)
        progress(f'{split}: {len(rows)} originals replayed; {len(plan)} common pairs per shift; {len(excluded)} boundary exclusions')
    return result,stats


@torch.inference_mode()
def predict(model,x,stats,batch,device):
    zs=[];ys=[]
    for start in range(0,len(x),batch):
        z=model.core.encoder(torch.tensor(x[start:start+batch],device=device))[:,-1]
        y=model.core.decoder(z)
        if not torch.isfinite(z).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite model output')
        zs.append(z.cpu().numpy());ys.append(y.cpu().numpy())
    z=np.concatenate(zs);y=np.concatenate(ys).astype(float)*stats['y_scale']+stats['y_mean']
    return z,y


def physical(view,stats):return view['y'].astype(float)*stats['y_scale']+stats['y_mean']


def summaries(values,valid,rows):
    result={'all':{k:od.summarize(v,valid[k]) for k,v in values.items()}}
    for field in ('symbol','period'):
        for name in sorted({str(r[field]) for r in rows}):
            ids=np.array([str(r[field])==name for r in rows]);result[field+'/'+name]={k:od.summarize(v,valid[k]&ids) for k,v in values.items()}
    return result


def paired(values,valid,rows):
    result={}
    for family,suffix in [('path_shape','bps'),('path_native_mae','bps'),('change1','nmse'),('body','nmse'),('activity','nmse')]:
        a=f'{family}_a_{suffix}' if family=='path_native_mae' else f'{family}_error_a_{suffix}'
        b=a.replace('_a_','_b_');ids=np.flatnonzero(valid[a]&valid[b])
        if not len(ids):result[family]=dict(supported=False,support=0,weeks=0,delta=None,low=None,high=None);continue
        result[family]=ur.bb.ab.pair_groups(values[b][ids],values[a][ids],[rows[i] for i in ids])
    return result


def evaluate_cell(meta,out,name,split,data,stats,device):
    filename=f'{split}_{name}.json';p=out/filename;receipt=out/(filename+'.receipt.json')
    if receipt.exists():
        saved=read_json(receipt)
        if saved!=dict(manifest_sha256=sha256(out/'manifest.json'),file_sha256=sha256(p)):raise ValueError('Completed overlap cell changed')
        progress(f'Reusing verified cell: {split}/{name}');return read_json(p)
    model,_,_=ur.load_model(meta['loader'],name,device)
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval model required')
    before=ur.bb.state_signature(model);rows=data['rows'];views=data['views'];z,b=predict(model,views[0]['x'],stats,meta['batch'],device)
    # Numerical reference: SAME fixed windows independently batched, not another context.
    ids=np.linspace(0,len(rows)-1,min(8,len(rows)),dtype=int)
    rz,rb=predict(model,views[0]['x'][ids],stats,1,device)
    control=dict(state_max_abs=float(abs(rz-z[ids]).max()),prediction_max_abs=float(abs(rb-b[ids]).max()))
    # Match the existing trained-causality FP32 tolerance, preserving true numeric failures.
    ok=np.allclose(rz,z[ids],atol=5e-5,rtol=2e-4)
    yn=(b[ids]-stats['y_mean'])/stats['y_scale'];rn=(rb-stats['y_mean'])/stats['y_scale']
    ok=ok and np.allclose(yn,rn,atol=5e-5,rtol=2e-4);control['passed']=bool(ok)
    if not ok:
        atomic_json(control,out/f'{split}_{name}_numeric_failure.json');raise ValueError('Same-window batch discrepancy; preserve failure for numeric diagnosis')
    results={};truth_b=physical(views[0],stats)
    examples_ids=np.linspace(0,len(rows)-1,min(6,len(rows)),dtype=int)
    for d in meta['shifts']:
        az,a=predict(model,views[d]['x'],stats,meta['batch'],device);truth_a=physical(views[d],stats)
        values,valid=od.pair_scores(a,b,truth_a,truth_b,views[d]['mask'],views[0]['mask'],az,z,d,stats)
        oracle,oracle_valid=od.pair_scores(truth_a,truth_b,truth_a,truth_b,views[d]['mask'],views[0]['mask'],az,z,d,stats)
        examples=[dict(index=int(i),key=rows[i]['key'],end=rows[i]['end'],prediction_a=a[i].tolist(),prediction_b=b[i].tolist(),
            target_a=truth_a[i].tolist(),target_b=truth_b[i].tolist(),mask_a=views[d]['mask'][i].tolist(),mask_b=views[0]['mask'][i].tolist()) for i in examples_ids]
        results[str(d)]=dict(shared_bars=127-d,shared_target_support=views[d]['mask'][:,d:127].sum(1).tolist(),summary=summaries(values,valid,rows),paired_b_minus_a=paired(values,valid,rows),
            per_pair={k:v.tolist() for k,v in values.items()},valid={k:v.tolist() for k,v in valid.items()},examples=examples,
            target_identity_reference={k:od.summarize(oracle[k],oracle_valid[k]) for k in ('path_shape_gap_bps','change1_gap_nmse','body_gap_nmse','activity_gap_nmse')})
        progress(f'{split}/{name}/shift{d}: {len(rows)} shared-history pairs evaluated')
    if before!=ur.bb.state_signature(model):raise ValueError('Frozen model mutated')
    report=dict(model=name,split=split,pairs=len(rows),numeric_control=control,weights_unchanged=True,shifts=results)
    atomic_json(report,p);atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),file_sha256=sha256(p)),receipt)
    return report


def run(source,out,batch=128,device='cuda'):
    source,out=source.resolve(),out.resolve();ur.check_output(source,out)
    if batch<1:raise ValueError('Positive extraction batch required')
    progress('Verifying frozen768 source and immutable chronological banks')
    identity=ur.source_identity(source);packed=banks(identity);meta=make_manifest(source,identity,packed,batch)
    runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source Torch/NumPy versions')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Overlap source/config changed; use a new output')
    elif out.exists() and any(out.iterdir()):raise ValueError('Empty output required')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json')
        if done['status']!='complete' or done['updates']!=0 or done['fits']!=0:raise ValueError('Invalid completion')
        ur.bb.ab.sc.verify_worker_files(out,done['files']);progress('Already complete; no repeated inference');return
    started=time.time()
    try:
        ur.preflight(meta['loader'],out,device)
        data,stats=prepare(meta,out);report={}
        for split in SPLITS:
            report[split]={name:evaluate_cell(meta,out,name,split,data[split],stats,device) for name in MODELS}
        if ur.source_identity(source)!=identity or banks(identity)!=packed:raise ValueError('Source changed during diagnostic')
        summary={s:{n:{d:v['summary']['all'] for d,v in c['shifts'].items()} for n,c in models.items()} for s,models in report.items()}
        atomic_json(dict(schema=SCHEMA,summary=summary,decision='diagnostic_only_no_promotion_or_automatic_training',updates=0,fits=0),out/'overlap_metrics.json')
        lines=['# Frozen overlapping-window diagnostic','','No model updates or fits. A ends before B; both decode observed history.','Native path error retains offsets; shape compares each prediction relative to its own first shared close.','No pass/fail threshold on embedding similarity; no causal attribution to position alone.','', '| split/model/shift | pairs | path gap bp | A shape error bp | B shape error bp | A native MAE bp | B native MAE bp |','|---|---:|---:|---:|---:|---:|---:|']
        for s,models in report.items():
            for n,c in models.items():
                for d,v in c['shifts'].items():
                    a=v['summary']['all'];keys=('path_shape_gap_bps','path_shape_error_a_bps','path_shape_error_b_bps','path_native_mae_a_bps','path_native_mae_b_bps')
                    lines.append(f'| {s}/{n}/{d} | {c["pairs"]} | '+' | '.join(f'{a[k]["mean"]:.6f}' for k in keys)+' |')
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json');raise
    atomic_json(dict(seconds=time.time()-started,torch=str(torch.__version__),numpy=np.__version__,device=str(device),updates=0,fits=0),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md') and p.name!='completion.json'}
    atomic_json(dict(status='complete',source_unchanged=True,updates=0,fits=0,files=files),out/'completion.json')
    progress('Overlap diagnostic complete; reports ready')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('checkpoints/babel_path768'));p.add_argument('--out',type=Path,default=Path('checkpoints/babel_overlap768'));p.add_argument('--batch',type=int,default=128)
    a=p.parse_args();ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal model inference runs on AutoDL CUDA')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Overlap output already running')
        run(a.source,a.out,a.batch)


if __name__=='__main__':main()
