"""One bounded utility check of all four frozen path768 continuation checkpoints."""
import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import path_feature_benchmark as pf, utility_probe as up, window_state_probe_run as old
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

bb=pf.bb
SCHEMA='babel-frozen-utility768-v1'
SPLITS=old.SPLITS


def code_identity():
    return pf.code_identity()|old.code_identity()|{Path(up.__file__).name:sha256(up.__file__),Path(__file__).name:sha256(__file__)}


def source_identity(source):
    """Read-only: never call the source's mutating lock/publish functions."""
    meta=read_json(source/'manifest.json');pf.verify_run(meta)
    done=read_json(source/'completion.json');lock=read_json(source/'selection_lock.json')
    if done['status']!='complete' or not done['source_unchanged'] or done['cells']!=4 or lock['manifest']!=meta:
        raise ValueError('Completed four-arm path768 source required')
    bb.ab.sc.verify_worker_files(source,done['files'])
    if pf.ea.source_identity(Path(meta['source']))!=meta['identity']:raise ValueError('Alignment lineage changed')
    files=dict(done['files'])|{'completion.json':sha256(source/'completion.json')}
    total=0
    for job in meta['experiments']:
        name=job['name'];path=source/name;worker=read_json(path/'completion.json')
        if worker['status']!='complete' or worker['metadata']!=dict(manifest=meta,job=job):raise ValueError('Incomplete source worker')
        verify_files(path,worker['files']);history=read_json(path/'history.json');initial=read_json(path/'initial_validation.json');summary=read_json(path/'training_summary.json')
        selected=min([(0,initial['validation'])]+[(r['epoch'],r['validation']) for r in history],key=lambda v:v[1]['selection'])
        if ([r['epoch'] for r in history]!=list(range(1,meta['epochs']+1)) or summary!=lock['trials'][name]
                or (summary['selected_epoch'],summary['validation'])!=selected):raise ValueError('Source budget/selection changed')
        for key in ('windows','encoder_steps','head_steps'):
            if summary[key]!=sum(r[key] for r in history):raise ValueError('Source update summary changed')
        for kind,digest in lock['weights'][name].items():
            if sha256(path/f'{kind}.pt')!=digest:raise ValueError('Locked source weights changed')
        total+=summary['encoder_steps']
        files.update({f'{name}/{f}':digest for f,digest in worker['files'].items()})
        files[f'{name}/completion.json']=sha256(path/'completion.json')
    if total!=done['encoder_steps']:raise ValueError('Total source budget changed')
    alignment=meta['identity']['manifest'];sampling=read_json(Path(alignment['sampling_source'])/'manifest.json')
    coverage=Path(sampling['coverage_source']);actual=bb.ws.coverage_identity(coverage,Path(alignment['original_source']))
    if actual!=sampling['coverage_identity']:raise ValueError('Coverage lineage changed')
    return dict(manifest=meta,files=files,coverage=str(coverage),coverage_identity=actual)


def alignment(meta):return meta['identity']['manifest']['identity']['manifest']


def make_manifest(source,identity,batch=128):
    if batch<1:raise ValueError('Positive extraction batch required')
    sm=identity['manifest']
    if {j['name'] for j in sm['experiments']}!={'control_s42','control_s43','path_s42','path_s43'} or any(j['width']!=768 for j in sm['parents'].values()):
        raise ValueError('Expected paired768 checkpoints')
    return dict(schema=SCHEMA,source=str(source.resolve()),identity=identity,code_sha256=code_identity(),batch=batch,pca_rank=768,
        variants=['control_s42','control_s43','path_s42','path_s43','pca768','raw3584','current28'],
        models=[j['name'] for j in sm['experiments']],targets=list(up.NAMES),alphas=list(up.probe.ALPHAS),
        primary_candidate='control',secondary_candidate='path',checkpoints='All four source-selected best; no last/seed/model reselection',
        selection='Five ridge alphas independently per target per representation on masked validation NMSE; ties prefer stronger regularization. 7 representations x13 targets x5 alphas =455 candidates. Freeze all before research extraction.',
        utility='Equal-weight16/64 directional efficiency and log realized RMS:4 targets. Trend R2 at both scales is reported separately because the prior linear baseline could not read it reliably.',
        current='7 current observed input features: asinh log-percent gap/body plus5 activity features. Preserve source availability/suspect masks. Independent target scalers/heads; excluded from primary utility.',
        decision=dict(raw_retention=.05,family_retention=.10,baseline_gain=.05,minimum_utility_r2=.50,current_nmse=.10,current_r2=.80,min_windows=50,min_weeks=5,
            rule='Both seeds AND both research sets. Utility: candidate/raw R2>=0.50 on all4 targets; upper paired weekly interval for primary <=1.05 raw, each direction/volatility family <=1.10 raw, primary <=0.95 PCA/current. Current: all7 targets >=50 windows/5 weeks, positive raw R2, candidate R2>=0.80 and upper weekly NMSE bound<=0.10. No automatic replacement or general representation claim.'),
        encoder_updates=0,neural_reader_updates=0,scope='Reused research data; exploratory intervals uncorrected for multiplicity. Current-feature copy is an information-retention test, not novel utility. No decoder/encoder training, grid expansion or new-data prerequisite.')


def inventories(meta):
    result={}
    for split in SPLITS:
        rows=read_json(Path(meta['identity']['coverage'])/f'{split}_windows.json')
        result[split]=[{k:r[k] for k in ('key','symbol','period','row','end','month')}|dict(week=str(pd.Timestamp(r['end']).to_period('W'))) for r in rows]
    source=Path(meta['identity']['manifest']['source'])
    for split in SPLITS[2:]:
        previous=read_json(source/f'{split}_inventory.json')
        keys=('key','symbol','period','row','end','month','week')
        if result[split]!=[{k:r[k] for k in keys} for r in previous]:raise ValueError('Research row identities differ from alignment')
    return result


def load_model(meta,name,device):
    sm=meta['identity']['manifest'];job=next(j for j in sm['experiments'] if j['name']==name);source=Path(meta['source'])
    ck=torch.load(source/name/'best.pt',map_location='cpu',weights_only=True);selected=read_json(source/'selection_lock.json')['trials'][name]
    if ck['metadata']!=dict(manifest=sm,job=job) or ck['epoch']!=selected['selected_epoch'] or ck['validation']!=selected['validation']:
        raise ValueError('Selected source checkpoint changed')
    model=pf.construct(sm,job,device);model.load_state_dict(ck['model']);model.requires_grad_(False);model.eval()
    return model,job,ck['validation']


@torch.inference_mode()
def preflight(meta,out,device):
    sm=meta['identity']['manifest'];source=Path(sm['source']);data=bb.load_data(alignment(meta),'val')
    stats=read_json(source/'cache/statistics.json');local=read_json(source/'cache/local_scales.json');report={}
    ids=np.linspace(0,len(data['x'])-1,min(8,len(data['x'])),dtype=int)
    for name in meta['models']:
        model,job,expected=load_model(meta,name,device)
        actual=pf.validation(sm,job,model,data,stats,local,device);pf.ea.require_replay(actual,expected,name+'/selected validation')
        before=bb.state_signature(model);causal=pf.er.trained_causality(model,torch.tensor(np.asarray(data['x'][ids]),device=device))
        report[name]=dict(validation=actual,causality=causal,unchanged=before==bb.state_signature(model))
        atomic_json(report,out/'trained_validation_causality.json')
        if causal['status']=='failed' or not report[name]['unchanged']:raise ValueError('Trained causality/frozen-state failure')
        del model
    progress('All four selected checkpoints: validation replay and trained causality passed')


@torch.inference_mode()
def prepare(meta,out,split,rows,device):
    if split not in SPLITS:raise ValueError('Unknown split')
    if split in SPLITS[2:]:old.check_selection(out)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists():old.verify_cache(meta,out,split);return
    x=np.asarray(bb.load_data(alignment(meta),split)['x'])
    if x.shape!=(len(rows),128,28):raise ValueError('Input row count/shape differs')
    stats=read_json(Path(meta['identity']['manifest']['source'])/'cache/statistics.json')
    target,mask=up.targets(up.restore_raw(x,stats))
    bank=dict(raw=x.reshape(len(x),-1),current=x[:,-1],targets=target,mask=mask)
    for name in meta['models']:
        model,_,_=load_model(meta,name,device);before=bb.state_signature(model);values=[]
        for start in range(0,len(x),meta['batch']):
            z=model.core.encoder(torch.tensor(x[start:start+meta['batch']],device=device))[:,-1]
            values.append(z.cpu().numpy())
        if before!=bb.state_signature(model) or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen encoder changed')
        bank[name]=np.concatenate(values)
        if bank[name].shape!=(len(x),meta['pca_rank']):raise ValueError('State dimension differs from declared width')
        del model
    if not all(np.isfinite(v).all() for v in bank.values()):raise ValueError('Nonfinite cached values')
    np.savez(cache/f'{split}.npz',**bank)
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),windows=len(x),files={f'{split}.npz':sha256(cache/f'{split}.npz')},encoder_updates=0),cache/f'{split}_index.json')
    progress(f'Frozen extraction complete: {split}, {len(x)} windows')


def fit(meta,out,device):
    if (out/'selection_lock.json').exists():old.check_selection(out);return
    for split in SPLITS[:2]:old.verify_cache(meta,out,split)
    train,val=old.arrays(out,'train'),old.arrays(out,'val');stats=up.target_scales(train['targets'],train['mask'])
    raw_stats=up.probe.scales(train['raw']);x=up.probe.normalize(train['raw'],raw_stats);rank=meta['pca_rank']
    if rank>=min(x.shape):raise ValueError('Insufficient fixed training rows for input PCA rank')
    progress('Fitting train-only input PCA and masked closed-form readouts; encoder remains frozen')
    eigen=up.probe.eigensystem(x,device);pca=eigen[1][:,-rank:].flip(1).cpu().numpy()
    ratio=float(eigen[0][-rank:].sum()/eigen[0].sum().clamp_min(1e-15));del eigen
    np.savez(out/'pca.npz',components=pca);heads={};tensors={}
    for name in meta['variants']:
        tr=old.representation(name,train,pca,raw_stats);va=old.representation(name,val,pca,raw_stats)
        head,weights,intercepts=up.fit_heads(tr,train['targets'],train['mask'],va,val['targets'],val['mask'],stats,device)
        heads[name]=dict(targets=head,input_width=tr.shape[1],parameters=(tr.shape[1]+1)*len(up.NAMES))
        tensors[name+'_weights']=weights;tensors[name+'_intercepts']=intercepts
        progress(f'{name}: all13 target readouts selected on validation')
    np.savez(out/'heads.npz',**tensors)
    atomic_json(dict(target_stats=stats,raw_stats=raw_stats,heads=heads,pca=dict(rank=rank,explained_variance_ratio=ratio),
                     train_mean=stats['mean'],fits_use_train_only=True),out/'fit.json')
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),files={n:sha256(out/n) for n in ('pca.npz','heads.npz','fit.json')},
        cache_indexes={s:sha256(out/f'cache/{s}_index.json') for s in SPLITS[:2]}),out/'selection_lock.json')


def interval(a,b,rows):
    if not rows:return dict(support=0,weeks=0,supported=False,delta=None,low=None,high=None,interpretation='No observed targets; cannot assess')
    return bb.ab.pair_groups(np.asarray(a),np.asarray(b),rows)['all']


def decide(meta,scores,errors,mask,rows):
    config=meta['decision'];result={}
    for name in meta['models']:
        score=scores[name];utility=[];current=[]
        readable=all(scores[n]['targets'][i]['r2'] is not None and scores[n]['targets'][i]['r2']>=config['minimum_utility_r2'] for n in (name,'raw3584') for i in up.GROUPS['utility'])
        for task,reference,factor in [('utility','raw3584',1+config['raw_retention']),('direction','raw3584',1+config['family_retention']),
                ('volatility','raw3584',1+config['family_retention']),('utility',f'pca{meta["pca_rank"]}',1-config['baseline_gain']),('utility','current28',1-config['baseline_gain'])]:
            ci=interval(errors[name][task],factor*errors[reference][task],rows)
            utility.append(dict(task=task,reference=reference,factor=factor,interval=ci,passed=bool(readable and ci['supported'] and ci['high'] is not None and ci['high']<=0)))
        for i in range(6,len(up.NAMES)):
            ids=np.flatnonzero(mask[:,i]);valid_rows=[rows[k] for k in ids];ci=interval(errors[name][up.NAMES[i]][ids],np.full(len(ids),config['current_nmse']),valid_rows)
            r2=score['targets'][i]['r2'];raw_r2=scores['raw3584']['targets'][i]['r2']
            supported=bool(len(ids)>=config['min_windows'] and len({r['week'] for r in valid_rows})>=config['min_weeks'])
            current.append(dict(target=up.NAMES[i],support=len(ids),interval=ci,r2=r2,raw_r2=raw_r2,
                passed=bool(supported and ci['supported'] and ci['high'] is not None and ci['high']<=0 and r2 is not None and r2>=config['current_r2'] and raw_r2 is not None and raw_r2>0)))
        result[name]=dict(utility_targets_readable=readable,utility_passed=all(c['passed'] for c in utility),utility_checks=utility,
                          current_passed=all(c['passed'] for c in current),current_checks=current)
    return result


def evaluate(meta,out,inventories):
    old.check_selection(out);fitted=read_json(out/'fit.json');stats=fitted['target_stats']
    with np.load(out/'heads.npz',allow_pickle=False) as f:heads=dict(f)
    with np.load(out/'pca.npz',allow_pickle=False) as f:pca=f['components']
    report={};checks={}
    for split in SPLITS[2:]:
        old.verify_cache(meta,out,split);bank=old.arrays(out,split);truth=bank['targets'];mask=bank['mask'];rows=inventories[split]
        scores={};errors={};predictions={};groups={};paired={}
        for name in meta['variants']+['train_mean']:
            if name=='train_mean':pred=np.tile(fitted['train_mean'],(len(truth),1))
            else:
                x=old.representation(name,bank,pca,fitted['raw_stats'])
                pred=up.predict(fitted['heads'][name]['targets'],heads[name+'_weights'],heads[name+'_intercepts'],x,stats)
            predictions[name]=pred.tolist();scores[name],errors[name]=up.measure(pred,truth,mask,stats);groups[name]={}
            for field in ('symbol','period'):
                for value in sorted({str(r[field]) for r in rows}):
                    ids=np.array([str(r[field])==value for r in rows]);groups[name][field+'/'+value]=up.measure(pred[ids],truth[ids],mask[ids],stats)[0]
        for name in meta['models']:
            refs=[f'pca{meta["pca_rank"]}','raw3584','current28']
            if name.startswith('path'):refs.append(name.replace('path','control'))
            for ref in refs:paired[name+'_minus_'+ref]={k:bb.ab.pair_groups(errors[name][k],errors[ref][k],rows) for k in up.GROUPS}
        checks[split]=decide(meta,scores,errors,mask,rows);report[split]=dict(scores=scores,groups=groups,paired=paired)
        atomic_json(dict(target_names=list(up.NAMES),targets=truth.tolist(),mask=mask.tolist(),predictions=predictions,
            per_window_errors={n:{k:v.tolist() for k,v in e.items()} for n,e in errors.items()},missing_error_fill='0 only with false target mask; never score as observed'),out/f'{split}_predictions.json')
    modes={}
    for mode in ('control','path'):
        rows=[checks[s][f'{mode}_s{seed}'] for s in SPLITS[2:] for seed in (42,43)]
        utility=all(r['utility_passed'] for r in rows);current=all(r['current_passed'] for r in rows)
        modes[mode]=dict(utility_passed=utility,current_passed=current,status='limited_scope_qualified' if utility and current else 'utility_only' if utility else 'not_qualified')
    decision=dict(primary_candidate='control',candidates=modes,checks=checks,automatic_promotion=False,
        scope='Only declared linear readability; no forecasting, trading or universal-state claim. Path is secondary; never select a seed/model based on research scores.')
    atomic_json(dict(manifest=meta,datasets=report,decision=decision),out/'utility_metrics.json');atomic_json(decision,out/'decision.json')
    lines=['# Frozen768 utility and current-bar retention','',f'Primary control: {modes["control"]["status"]}. Secondary path: {modes["path"]["status"]}.','',
        'Utility =16/64 direction efficiency and volatility. Current feature recovery and nonlinear trend diagnostics are separate. No encoder updates.']
    for split in SPLITS[2:]:
        lines += ['',f'## {split}','','| representation | utility NMSE | direction | volatility | trend diagnostic |','|---|---:|---:|---:|---:|']
        for name,s in report[split]['scores'].items():lines.append('| '+name+' | '+' | '.join(f'{s["groups"][k]:.6f}' for k in up.GROUPS)+' |')
    (out/'summary.md').write_text('\n'.join(lines))


def check_output(source,out):
    if out==source or source in out.parents or out in source.parents:raise ValueError('Separate utility output required')
    pf.ea.check_output(Path(read_json(source/'manifest.json')['source']),out)


def run(source,out,batch=128,device='cuda'):
    source,out=source.resolve(),out.resolve();check_output(source,out)
    progress('Checking immutable four-arm source, selected weights and data lineage')
    identity=source_identity(source);meta=make_manifest(source,identity,batch);runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source Torch/NumPy environment')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Utility configuration/source changed')
    elif out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json')
        if done['status']!='complete' or done['encoder_updates']!=0:raise ValueError('Invalid completion')
        bb.ab.sc.verify_worker_files(out,done['files']);old.check_selection(out);progress('Already complete; no refit or extraction');return
    started=time.time()
    try:
        rows=inventories(meta);audit=up.probe.inventory_audit(rows);atomic_json(audit,out/'data_audit.json')
        for split,values in rows.items():atomic_json(values,out/f'{split}_inventory.json')
        preflight(meta,out,device)
        for split in SPLITS[:2]:prepare(meta,out,split,rows[split],device)
        fit(meta,out,device)
        for split in SPLITS[2:]:prepare(meta,out,split,rows[split],device)
        evaluate(meta,out,rows)
        if source_identity(source)!=identity:raise ValueError('Source changed during utility audit')
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json');raise
    atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device),seconds=time.time()-started,
        encoder_updates=0,neural_reader_updates=0,pca_fits=1,ridge_candidates=len(meta['variants'])*len(up.NAMES)*len(up.probe.ALPHAS)),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.md') and p.name!='completion.json'}
    atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,files=files),out/'completion.json')
    progress('Frozen utility audit complete; all reports ready')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('checkpoints/babel_path768'))
    p.add_argument('--out',type=Path,default=Path('checkpoints/babel_utility768'));p.add_argument('--batch',type=int,default=128)
    a=p.parse_args();bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal frozen extraction/readout fits run on AutoDL CUDA')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Utility output already running')
        run(a.source,a.out,a.batch)


if __name__=='__main__':main()
