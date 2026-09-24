"""Frozen, matched linear recalibration after the single consistency experiment."""
import argparse
import fcntl
import time
from pathlib import Path
import numpy as np
import torch
from . import overlap_consistency_run as cr, overlap_consistency_evaluate as ce
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

ur=cr.ur
up=ur.up
old=ur.old
MODELS=tuple(f'{mode}_s{s}' for mode in ('frozen','control','consistent') for s in (42,43))
BASELINES=('pca768','raw3584','current28')
SCHEMA='babel-consistency-readout768-v1'


def code_identity():
    return cr.code_identity() | {Path(__file__).name:sha256(__file__)}


def source_identity(source):
    """Validate the completed source without republishing its selection lock."""
    sm=read_json(source/'manifest.json');done=read_json(source/'completion.json');lock=read_json(source/'selection_lock.json')
    cr.verify_run(sm)
    if done['status']!='complete' or not done['source_unchanged'] or done['cells']!=4 or lock['manifest']!=sm:
        raise ValueError('Completed four-arm consistency source required')
    if {j['name'] for j in sm['experiments']}!=set(MODELS[2:]):raise ValueError('Unexpected source arms')
    ur.bb.ab.sc.verify_worker_files(source,done['files'])
    if ur.source_identity(Path(sm['source']))!=sm['identity']:raise ValueError('Parent lineage changed')
    if cr.reader_identity(Path(sm['reader']['root']),sm['identity'])!=sm['reader']:raise ValueError('Original readers changed')
    files=done['files']|{'completion.json':sha256(source/'completion.json')}
    n=read_json(source/'train_plan.json')['eligible']
    for job in sm['experiments']:
        name=job['name'];root=source/name;wd=read_json(root/'completion.json')
        if wd['status']!='complete' or wd['metadata']!=dict(manifest=sm,job=job):raise ValueError('Incomplete source worker')
        cr.verify_files(root,wd['files'])
        h=read_json(root/'history.json');summary=read_json(root/'training_summary.json');initial=read_json(root/'initial_validation.json')
        if summary!=lock['trials'][name] or summary['selected_epoch']!=sm['epochs'] or summary['epochs']!=sm['epochs']:
            raise ValueError('This bounded protocol requires source best=last epoch; do not reselect')
        cr.verify_history(sm,job,dict(epoch=sm['epochs'],history=h,initial_validation=initial['validation'],best_epoch=summary['selected_epoch'],best_validation=summary['validation']),n)
        for k in ('pairs','views','encoder_steps','head_steps'):
            if summary[k]!=sum(v[k] for v in h):raise ValueError('Source budget summary changed')
        for k,digest in lock['weights'][name].items():
            if sha256(root/f'{k}.pt')!=digest:raise ValueError('Selected source weights changed')
        files.update({name+'/'+k:v for k,v in wd['files'].items()});files[name+'/completion.json']=sha256(root/'completion.json')
    decision=read_json(source/'decision.json')
    if not all(c['passed'] for c in decision['checks'] if not c['metric'].startswith('utility vs ')):
        raise ValueError('Recalibration cannot rescue failed source stability/reconstruction guards')
    return dict(manifest=sm,files=files)


def make_manifest(source,identity,batch):
    if batch<1:raise ValueError('Positive extraction batch required')
    return dict(schema=SCHEMA,source=str(source),identity=identity,code_sha256=code_identity(),batch=batch,
        models=list(MODELS),baselines=list(BASELINES),targets=list(up.NAMES),alphas=list(up.probe.ALPHAS),
        encoder_updates=0,neural_reader_updates=0,pca_fits=0,ridge_candidates=390,
        selection='Six frozen states x13 targets x5 original alphas; train-only masked fitting, original validation selection. All78 heads locked before research extraction. Selected source best equals last; no epoch/seed reselection.',
        data='Original4789 train/978 val,984 test/453 cross_research, original masks and trading-session weeks. No dense-pool expansion. Reused research sets.',
        references='Refit both old best and both paired control/consistent states equally. Reuse the exact original PCA/raw/current heads and PCA: unchanged inputs, targets, train/val, algorithm and alpha budget. Replay old-state refits and all legacy reader scores.',
        decision=dict(utility_retention=.05,family_retention=.10,minimum_r2=.5,current_r2=.8,current_nmse=.1),
        stop='Both seeds and both research sets must retain main utility versus refitted same-budget control AND old best. Failure ends weight0.10 configuration, no MLP/alpha/weight sweep. Recovery means linear readout recovery only; never reverse old frozen-interface failure or claim full representation promotion.')


def load_selected(meta,name,device):
    sm=meta['identity']['manifest'];job=next(j for j in sm['experiments'] if j['name']==name)
    ck=torch.load(Path(meta['source'])/name/'best.pt',map_location='cpu',weights_only=True)
    selected=read_json(Path(meta['source'])/'selection_lock.json')['trials'][name]
    if ck['metadata']!=dict(manifest=sm,job=job) or ck['epoch']!=selected['selected_epoch'] or ck['validation']!=selected['validation']:
        raise ValueError('Selected checkpoint metadata differs')
    model=cr.construct(sm,job,device);model.load_state_dict(ck['model']);model.eval().requires_grad_(False)
    return model,job,ck['validation']


@torch.inference_mode()
def preflight(meta,out,device):
    sm=meta['identity']['manifest'];data=ur.bb.load_data(cr.alignment(sm),'val');results={}
    for name in MODELS[2:]:
        model,job,expected=load_selected(meta,name,device);before=ur.bb.state_signature(model)
        actual=cr.validation(sm,job,model,device);ur.pf.ea.require_replay(actual,expected,name+'/selected validation')
        causal=ur.pf.er.trained_causality(model,torch.tensor(np.asarray(data['x'][:4]),device=device))
        if causal['status']=='failed' or before!=ur.bb.state_signature(model):raise ValueError('Frozen validation/causality failure')
        results[name]=dict(validation=actual,causality=causal,unchanged=True)
        atomic_json(results,out/'trained_validation_causality.json');del model


def check_inputs(bank,x,stats,rows):
    if x.shape!=(len(rows),128,28) or not np.array_equal(bank['raw'],x.reshape(len(x),-1)) or not np.array_equal(bank['current'],x[:,-1]):
        raise ValueError('Original input identity/order differs')
    y,mask=up.targets(up.restore_raw(x,stats))
    if not np.array_equal(mask,bank['mask']) or not np.allclose(y,bank['targets'],atol=1e-10,rtol=1e-10):
        raise ValueError('Original target/mask identity differs')


@torch.inference_mode()
def prepare(meta,out,split,rows,device):
    if split not in old.SPLITS:raise ValueError('Unknown split')
    if split in old.SPLITS[2:]:check_selection(out)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists():old.verify_cache(meta,out,split);return
    sm=meta['identity']['manifest'];reader=Path(sm['reader']['root'])
    old.verify_cache(sm['reader']['manifest'],reader,split);original=old.arrays(reader,split)
    if rows!=read_json(reader/f'{split}_inventory.json'):raise ValueError('Original session-week inventory differs')
    x=np.asarray(ur.bb.load_data(cr.alignment(sm),split)['x']);stats,_=cr.statistics(sm);check_inputs(original,x,stats,rows)
    bank={k:original[k] for k in ('raw','current','targets','mask')}
    for seed in (42,43):bank[f'frozen_s{seed}']=original[f'control_s{seed}']
    for name in MODELS[2:]:
        model,_,_=load_selected(meta,name,device);before=ur.bb.state_signature(model);values=[]
        for start in range(0,len(x),meta['batch']):
            values.append(model.core.encoder(torch.tensor(x[start:start+meta['batch']],device=device))[:,-1].cpu().numpy())
        if before!=ur.bb.state_signature(model) or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen model changed')
        bank[name]=np.concatenate(values);del model
    if any(bank[n].shape!=(len(rows),768) for n in MODELS) or not all(np.isfinite(v).all() for v in bank.values()):raise ValueError('Invalid frozen bank')
    np.savez(cache/f'{split}.npz',**bank)
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),windows=len(rows),files={f'{split}.npz':sha256(cache/f'{split}.npz')},encoder_updates=0),cache/f'{split}_index.json')
    progress(f'Frozen extraction complete: {split}, {len(rows)} windows')


def check_selection(out):
    lock=old.check_selection(out)
    for split in old.SPLITS[:2]:old.verify_cache(read_json(out/'manifest.json'),out,split)
    return lock


def fit(meta,out,device):
    if (out/'selection_lock.json').exists():check_selection(out);return
    for split in old.SPLITS[:2]:old.verify_cache(meta,out,split)
    tr,va=(old.arrays(out,s) for s in old.SPLITS[:2]);reader=Path(meta['identity']['manifest']['reader']['root']);prior=read_json(reader/'fit.json')
    stats=up.target_scales(tr['targets'],tr['mask']);ce.require_nested(stats,prior['target_stats'],'train target scaling')
    heads={};tensors={};replay={}
    with np.load(reader/'heads.npz',allow_pickle=False) as f:original=dict(f)
    for name in MODELS:
        selected,w,b=up.fit_heads(tr[name],tr['targets'],tr['mask'],va[name],va['targets'],va['mask'],stats,device)
        heads[name]=dict(targets=selected,input_width=tr[name].shape[1]);tensors[name+'_weights']=w;tensors[name+'_intercepts']=b
        if name.startswith('frozen'):
            oldname=name.replace('frozen','control');expected=prior['heads'][oldname]['targets']
            ce.require_nested(selected,expected,name+'/same-budget refit')
            pred=up.predict(selected,w,b,va[name],stats);ref=up.predict(expected,original[oldname+'_weights'],original[oldname+'_intercepts'],va[name],stats)
            if not np.allclose(pred,ref,atol=1e-6,rtol=2e-5):raise ValueError('Old validation refit did not replay')
            replay[name]=dict(max_abs=float(abs(pred-ref).max()),passed=True)
        progress(f'{name}: selected13 heads from original five alphas')
    np.savez(out/'heads.npz',**tensors);atomic_json(dict(target_stats=stats,heads=heads,fits_use_train_only=True,old_validation_replay=replay),out/'fit.json')
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),files={n:sha256(out/n) for n in ('heads.npz','fit.json')},cache_indexes={s:sha256(out/f'cache/{s}_index.json') for s in old.SPLITS[:2]}),out/'selection_lock.json')


def retention(scores,errors,mask,rows):
    checks=[]
    def add(seed,metric,x,y,factor=1.,extra=True,cohort=None):
        ci=ur.interval(x,factor*np.asarray(y),rows if cohort is None else cohort)
        checks.append(dict(seed=seed,metric=metric,factor=factor,interval=ci,extra_condition=bool(extra),passed=bool(extra and ci['supported'] and ci['high'] is not None and ci['high']<=0)))
    for seed in (42,43):
        name=f'consistent_s{seed}'
        for ref in (f'control_s{seed}',f'frozen_s{seed}'):
            readable=all(scores[n]['targets'][i]['r2'] is not None and scores[n]['targets'][i]['r2']>=.5 for n in (name,ref) for i in up.GROUPS['utility'])
            for task,factor in [('utility',1.05),('direction',1.10),('volatility',1.10)]:
                add(seed,task+' vs '+ref,errors[name][task],errors[ref][task],factor,readable)
        for i,target in enumerate(up.NAMES[6:],6):
            ids=np.flatnonzero(mask[:,i]);r2=scores[name]['targets'][i]['r2']
            add(seed,target,np.asarray(errors[name][target])[ids],np.full(len(ids),.1),extra=r2 is not None and r2>=.8,cohort=[rows[j] for j in ids])
    return checks


def evaluate(meta,out,rows):
    check_selection(out);reader=Path(meta['identity']['manifest']['reader']['root']);prior=read_json(reader/'fit.json');fitted=read_json(out/'fit.json');source=Path(meta['source'])
    with np.load(reader/'heads.npz',allow_pickle=False) as f:legacy=dict(f)
    with np.load(reader/'pca.npz',allow_pickle=False) as f:pca=f['components']
    with np.load(out/'heads.npz',allow_pickle=False) as f:heads=dict(f)
    report={};all_checks={};absolute={};replays={}
    for split in old.SPLITS[2:]:
        old.verify_cache(meta,out,split);bank=old.arrays(out,split);y,mask=bank['targets'],bank['mask'];scores={};errors={};predictions={};groups={};replays[split]={}
        def score(name,pred):
            scores[name],errors[name]=up.measure(pred,y,mask,fitted['target_stats']);predictions[name]=pred.tolist();groups[name]={}
            for field in ('symbol','period'):
                for value in sorted({str(r[field]) for r in rows[split]}):
                    ids=np.array([str(r[field])==value for r in rows[split]])
                    groups[name][field+'/'+value]=up.measure(pred[ids],y[ids],mask[ids],fitted['target_stats'])[0]
        for name in MODELS:
            score(name,up.predict(fitted['heads'][name]['targets'],heads[name+'_weights'],heads[name+'_intercepts'],bank[name],fitted['target_stats']))
            refname='control_s'+name.rsplit('s',1)[1]
            score('legacy_'+name,up.predict(prior['heads'][refname]['targets'],legacy[refname+'_weights'],legacy[refname+'_intercepts'],bank[name],prior['target_stats']))
            original=read_json(source/f'{split}_{name}{"" if name.startswith("frozen") else "_best"}.json')['utility']
            ce.require_nested(scores['legacy_'+name],original['scores'],split+'/'+name+'/legacy scores')
            if not np.allclose(predictions['legacy_'+name],original['predictions'],atol=1e-6,rtol=2e-5):raise ValueError('Legacy prediction replay differs')
            if name.startswith('frozen'):ce.require_nested(scores[name],scores['legacy_'+name],split+'/'+name+'/old refit')
            replays[split][name]=dict(passed=True)
        previous=read_json(reader/'utility_metrics.json')['datasets'][split]['scores']
        for name in BASELINES:
            x=old.representation(name,bank,pca,prior['raw_stats'])
            score(name,up.predict(prior['heads'][name]['targets'],legacy[name+'_weights'],legacy[name+'_intercepts'],x,prior['target_stats']))
            ce.require_nested(scores[name],previous[name],split+'/'+name+'/unchanged baseline')
        score('train_mean',np.tile(fitted['target_stats']['mean'],(len(y),1)))
        all_checks[split]=retention(scores,errors,mask,rows[split])
        absolute[split]=ur.decide(dict(models=[f'consistent_s{s}' for s in (42,43)],pca_rank=768,decision=meta['identity']['manifest']['reader']['manifest']['decision']),scores,errors,mask,rows[split])
        paired={}
        for name in MODELS:
            refs=['legacy_'+name]+([f'control_s{name[-2:]}',f'frozen_s{name[-2:]}'] if name.startswith('consistent') else [])
            for ref in refs:paired[name+'_minus_'+ref]={k:ur.bb.ab.pair_groups(errors[name][k],errors[ref][k],rows[split]) for k in up.GROUPS}
        report[split]=dict(scores=scores,groups=groups,paired=paired)
        atomic_json(dict(target_names=list(up.NAMES),targets=y.tolist(),mask=mask.tolist(),predictions=predictions,per_window_errors={n:{k:v.tolist() for k,v in e.items()} for n,e in errors.items()}),out/f'{split}_predictions.json')
        progress(f'Evaluated {split}: recalibrated, legacy and original input baselines')
    recovered=all(c['passed'] for cs in all_checks.values() for c in cs)
    qualified=all(v['utility_passed'] and v['current_passed'] for cs in absolute.values() for v in cs.values())
    decision=dict(status='readout_recovered_stability_candidate' if recovered else 'stop_consistency010',retention=all_checks,original_utility_protocol=absolute,original_utility_qualified=qualified,automatic_promotion=False,
        old_frozen_interface_result='no_full_upgrade remains unchanged',scope='Recovery means comparable linear readability after recalibration, not lossless information, independent generalization or future prediction. No further grid or encoder training.')
    atomic_json(replays,out/'reader_replay.json');atomic_json(decision,out/'decision.json');atomic_json(dict(datasets=report,decision=decision),out/'readout_metrics.json')
    lines=['# Frozen consistency readout recalibration','',f'Decision: {decision["status"]}; original utility qualified: {qualified}.','', '| dataset / state | legacy utility NMSE | recalibrated NMSE |','|---|---:|---:|']
    for split,v in report.items():
        for name in MODELS:lines.append(f'| {split}/{name} | {v["scores"]["legacy_"+name]["groups"]["utility"]:.6f} | {v["scores"][name]["groups"]["utility"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def check_output(source,out,identity=None):
    # Each source schema has a different ancestor depth: never pass a newer
    # manifest into the old bar-alignment-only directory guard.
    dependencies=[source]
    if identity is not None:
        sm=identity['manifest'];a=cr.alignment(sm)
        dependencies += [Path(sm['source']),Path(sm['reader']['root'])]
        dependencies += [Path(a[k]) for k in ('source','original_source','sampling_source')]
        dependencies += [Path(v['directory']) for v in sm['packed'].values()]
    out=out.resolve()
    for dependency in dependencies:
        dep=dependency.resolve()
        if out==dep or out in dep.parents or dep in out.parents:raise ValueError('Separate recalibration output required')


def run(source,out,batch=128,device='cuda'):
    source,out=source.resolve(),out.resolve();check_output(source,out);progress('Verifying immutable consistency source and reader lineage')
    identity=source_identity(source);reader=Path(identity['manifest']['reader']['root']);check_output(source,out,identity)
    meta=make_manifest(source,identity,batch);runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source Torch/NumPy runtime')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Source/configuration changed; new output required')
    elif out.exists() and any(out.iterdir()):raise ValueError('Empty output required')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json');ur.bb.ab.sc.verify_worker_files(out,done['files']);check_selection(out)
        if done['status']!='complete' or done['encoder_updates']!=0:raise ValueError('Invalid completed run')
        progress('Already complete; no repeat fitting');return
    started=time.time()
    try:
        rows=ur.inventories(cr.context(identity['manifest']));atomic_json(up.probe.inventory_audit(rows),out/'data_audit.json')
        for split,values in rows.items():
            if values!=read_json(reader/f'{split}_inventory.json'):raise ValueError('Source reader inventory changed')
            atomic_json(values,out/f'{split}_inventory.json')
        preflight(meta,out,device)
        for split in old.SPLITS[:2]:prepare(meta,out,split,rows[split],device)
        fit(meta,out,device)
        for split in old.SPLITS[2:]:prepare(meta,out,split,rows[split],device)
        evaluate(meta,out,rows)
        if source_identity(source)!=identity:raise ValueError('Source changed during recalibration')
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json');raise
    atomic_json(dict(seconds=time.time()-started,torch=str(torch.__version__),numpy=np.__version__,device=str(device),encoder_updates=0,neural_reader_updates=0,pca_fits=0,ridge_candidates=390),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.md') and p.name not in ('completion.json','failure.json')}
    atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,files=files),out/'completion.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('checkpoints/babel_consistency768'));p.add_argument('--out',type=Path,default=Path('checkpoints/babel_consistency_readout768'));p.add_argument('--batch',type=int,default=128)
    a=p.parse_args();ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal extraction and readout fitting only on AutoDL CUDA')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Output already running')
        run(a.source,a.out,a.batch)


if __name__=='__main__':main()
