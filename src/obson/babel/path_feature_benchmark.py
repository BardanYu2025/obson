"""Matched 768-joint continuation: unchanged input versus one causal path feature."""
import argparse
import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import path_feature as pf, endpoint_readout_audit as ea, endpoint_readout as er
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

bb=ea.bb
SCHEMA='babel-causal-path768-v1'
SPLITS=('test','cross_research')


def code_identity():
    return ea.code_identity()|{Path(pf.__file__).name:sha256(pf.__file__),Path(__file__).name:sha256(__file__)}


def make_manifest(source,identity,epochs=100,micro=64):
    sm=identity['manifest']
    if epochs<2 or micro<1 or micro>sm['batch'] or sm['batch']%micro:
        raise ValueError('At least two epochs; micro must divide inherited effective batch')
    parents={str(s):next(j for j in sm['experiments'] if j['name']==f'w768_joint_s{s}') for s in (42,43)}
    if any(j['epochs']!=200 or j['width']!=768 or not j['joint'] for j in parents.values()):
        raise ValueError('Completed200-epoch 768 joint parents required')
    lock=read_json(source/'selection_lock.json')
    parent_lrs={}
    for seed,parent in parents.items():
        row=read_json(source/parent['name']/'history.json')[-1]
        parent_lrs[seed]=dict(encoder=row['lr'],head=row['head_lr'])
    return dict(schema=SCHEMA,source=str(source.resolve()),identity=identity,epochs=epochs,
        batch=sm['batch'],micro=micro,evaluation_batch=sm['micro'],parents=parents,parent_lrs=parent_lrs,code_sha256=code_identity(),
        parent_weights={s:lock['weights'][j['name']]['last'] for s,j in parents.items()},
        experiments=[dict(name=f'{mode}_s{s}',seed=s,enabled=mode=='path') for s in (42,43) for mode in ('control','path')],
        feature='Cumulative sinh(gap)+sinh(body), divided by frozen train path scale. Prefix-only; no window-end statistics, decoder bypass or new observation.',
        initialization='Both arms restore identical768-joint last200 weights, AdamW moments and RNG. Added768-weight projection is zero; control gates it off. Inherited parameter groups unchanged; new projection has a separate matching AdamW group.',
        budget='Default100 extra epochs each; same4789 windows,38 updates,4 local positions per window. Absolute sampling epochs201..300, no optimizer reset. LR starts at parent final LR and decays to0.1 times that value.',
        selection='Original validation global primary +0.25 local primary at32/64/96/128, includes inherited epoch0. Lock all four arms before research evaluation; fixed last reported too.',
        decision=dict(path_gain=.05,primary_retention=.05,family_retention=.10,local_retention=.05),
        scope='Exploratory reused research sets. Compare against matched continuation AND frozen parent; preserve old512 full retention screen. No automatic promotion or further sweep.')


def verify_run(meta):
    if meta['schema']!=SCHEMA or meta['code_sha256']!=code_identity():
        raise ValueError('Continuation implementation changed; use a new run')
    source=Path(meta['source'])
    for seed,job in meta['parents'].items():
        if sha256(source/job['name']/'last.pt')!=meta['parent_weights'][seed]:
            raise ValueError('Frozen parent checkpoint changed')


def parent_checkpoint(meta,job):
    sm=meta['identity']['manifest'];source=Path(meta['source']);parent=meta['parents'][str(job['seed'])]
    ck=torch.load(source/parent['name']/'last.pt',map_location='cpu',weights_only=True)
    expected=dict(manifest=sm,job=parent,cache_sha256=sha256(source/'cache/index.json'))
    if ck['metadata']!=expected or ck['epoch']!=parent['epochs']:
        raise ValueError('Parent checkpoint identity differs')
    return ck,parent


def construct(meta,job,device,optimizers=False):
    source=Path(meta['source']);sm=meta['identity']['manifest'];ck,parent=parent_checkpoint(meta,job)
    model=bb.model_for(sm,source,parent,device);model.load_state_dict(ck['model'])
    inherited=[p for p in model.core.parameters() if p.requires_grad]
    stats=read_json(source/'cache/statistics.json')
    added=pf.install(model,stats,job['enabled'])
    if not optimizers:return model
    enc=torch.optim.AdamW(inherited,lr=parent['lr'],weight_decay=.01)
    head=torch.optim.AdamW(model.local_head.parameters(),lr=parent['head_lr'],weight_decay=1e-4)
    enc.load_state_dict(ck['encoder_optimizer']);head.load_state_dict(ck['head_optimizer'])
    group={k:v for k,v in enc.param_groups[0].items() if k!='params'}
    enc.add_param_group(dict(group,params=[added]))
    restore_rng(ck['rng'])
    return model,enc,head,ck


def plan(meta,job,epoch,n):
    absolute=meta['parents'][str(job['seed'])]['epochs']+epoch
    ids=bb.ws.sample_ids(n,meta['identity']['manifest']['windows_per_epoch'],job['seed'],absolute)
    ps=bb.ba.position_plan(len(ids),job['seed'],absolute)
    return ids,ps,dict(absolute_epoch=absolute,window_ids_sha256=bb.cov.ndarray_hash(ids),
                      prefixes_sha256=bb.cov.ndarray_hash(ps))


def validation(meta,job,model,data,stats,local,device):
    return bb.validation(model,data,stats,local,meta['identity']['manifest'],meta['parents'][str(job['seed'])],device)


def verify_history(meta,job,state,n):
    history=state['history'];base=state['parent_lrs']
    if base!=meta['parent_lrs'][str(job['seed'])]:raise ValueError('Parent continuation learning rates changed')
    if [r['epoch'] for r in history]!=list(range(1,state['epoch']+1)) or state['epoch']>meta['epochs']:
        raise ValueError('Continuation epoch history changed')
    count=meta['identity']['manifest']['windows_per_epoch'];steps=(count+meta['batch']-1)//meta['batch']
    for row in history:
        _,_,expected=plan(meta,job,row['epoch'],n)
        if row['sampling']!=expected or row['encoder_steps']!=steps or row['head_steps']!=steps or row['windows']!=count:
            raise ValueError('Continuation exposure budget changed')
        if row['lr']!=pf.learning_rate(row['epoch'],meta['epochs'],base['encoder']) or row['head_lr']!=pf.learning_rate(row['epoch'],meta['epochs'],base['head']):
            raise ValueError('Continuation learning-rate history changed')
    choices=[(0,state['initial_validation'])]+[(r['epoch'],r['validation']) for r in history]
    epoch,val=min(choices,key=lambda item:item[1]['selection'])
    if state['best_epoch']!=epoch or state['best_validation']!=val:
        raise ValueError('Continuation validation selection changed')


def publish(state,path):
    # last.pt is authoritative if interrupted while writing derived files.
    atomic_save(state,path/'last.pt')
    atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],validation=state['best_validation'],model=state['best_model']),path/'best.pt')
    atomic_json(state['history'],path/'history.json')
    row=state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'],epochs=state['metadata']['manifest']['epochs'],best_epoch=state['best_epoch'],
                    validation_path=row.get('validation',{}).get('path'),seconds=row.get('seconds')),path/'progress.json')


def worker(out,name,device='cuda'):
    meta=read_json(out/'manifest.json');verify_run(meta);job=next(j for j in meta['experiments'] if j['name']==name)
    path=out/name;path.mkdir(exist_ok=True);metadata=dict(manifest=meta,job=job)
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['metadata']!=metadata:raise ValueError('Worker identity changed')
        verify_files(path,done['files']);return
    sm=meta['identity']['manifest'];source=Path(meta['source'])
    stats=read_json(source/'cache/statistics.json');local=read_json(source/'cache/local_scales.json')
    pool=bb.load_data(sm,'pool');val=bb.load_data(sm,'val')
    model,enc,head,parent=construct(meta,job,device,True)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Resume manifest changed')
        verify_history(meta,job,state,len(pool['x']))
        model.load_state_dict(state['model']);enc.load_state_dict(state['encoder_optimizer']);head.load_state_dict(state['head_optimizer'])
        restore_rng(state['rng'])
        actual=validation(meta,job,model,val,stats,local,device)
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        ea.require_replay(actual,expected,'continuation resume')
        atomic_json(dict(epoch=state['epoch'],matched=True),path/'resume_validation.json')
    else:
        initial=validation(meta,job,model,val,stats,local,device)
        ea.require_replay(initial,parent['history'][-1]['validation'],'zero-projection parent replay')
        state=dict(metadata=metadata,epoch=0,history=[],initial_validation=initial,best_epoch=0,best_validation=initial,
            best_model=bb.ab.cpu_state(model),parent_lrs=dict(encoder=enc.param_groups[0]['lr'],head=head.param_groups[0]['lr']))
        if state['parent_lrs']!=meta['parent_lrs'][str(job['seed'])]:raise ValueError('Parent optimizer LR differs from source history')
        atomic_json(dict(metadata=metadata,validation=initial,parent_validation=parent['history'][-1]['validation'],
                         parent_weights=meta['parent_weights'][str(job['seed'])],inherited_optimizer_restored=True,
                         added_parameters=model.core.encoder.backbone.input.projection.weight.numel()),path/'initial_validation.json')
        state.update(model=bb.ab.cpu_state(model),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state())
        publish(state,path)
    del parent
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started=time.monotonic();ids,ps,sampling=plan(meta,job,epoch,len(pool['x']))
        lr=pf.learning_rate(epoch,meta['epochs'],state['parent_lrs']['encoder']);hlr=pf.learning_rate(epoch,meta['epochs'],state['parent_lrs']['head'])
        for group in enc.param_groups:group['lr']=lr
        for group in head.param_groups:group['lr']=hlr
        train,steps=bb.ba.run_epoch(model,bb.ws.subset(pool,ids),stats,local,meta['batch'],meta['micro'],device,True,ps,enc,head,
                                   job['seed']+sampling['absolute_epoch']*1009)
        result=validation(meta,job,model,val,stats,local,device)
        if result['selection']<state['best_validation']['selection']:
            state.update(best_epoch=epoch,best_validation=result,best_model=bb.ab.cpu_state(model))
        state['history'].append(dict(epoch=epoch,sampling=sampling,train=train,validation=result,lr=lr,head_lr=hlr,
            windows=len(ids),encoder_steps=steps,head_steps=steps,seconds=time.monotonic()-started))
        state.update(epoch=epoch,model=bb.ab.cpu_state(model),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state());publish(state,path)
        progress(f'{name}: epoch={epoch}/{meta["epochs"]} global={result["primary"]:.6f} path={result["path"]:.6f} selected={state["best_epoch"]}')
    verify_history(meta,job,state,len(pool['x']));publish(state,path)
    atomic_json(dict(selected_epoch=state['best_epoch'],validation=state['best_validation'],epochs=meta['epochs'],
        encoder_steps=sum(r['encoder_steps'] for r in state['history']),head_steps=sum(r['head_steps'] for r in state['history']),
        windows=sum(r['windows'] for r in state['history']),training_seconds=sum(r['seconds'] for r in state['history']),
        core_parameters=sum(p.numel() for p in model.core.parameters() if p.requires_grad),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None),path/'training_summary.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


@torch.no_grad()
def preflight(meta,out,device):
    source=Path(meta['source']);sm=meta['identity']['manifest'];stats=read_json(source/'cache/statistics.json')
    x=torch.tensor(np.asarray(bb.load_data(sm,'val')['x'][:4]),device=device)
    rows=[]
    for seed in (42,43):
        jobs=[j for j in meta['experiments'] if j['seed']==seed]
        models=[construct(meta,j,device).eval() for j in jobs]
        a,b=[m.core(x) for m in models]
        if not torch.equal(a,b):raise ValueError('Paired zero-projection outputs differ')
        checks=er.trained_causality(models[1],x)
        if checks['status']=='failed':raise ValueError('Causal preflight failed')
        # Exercise the newly active feature, not only its zero initialization.
        models[1].core.encoder.backbone.input.projection.weight.fill_(.01)
        active=er.trained_causality(models[1],x)
        if active['status']=='failed':raise ValueError('Active path feature is noncausal')
        rows.append(dict(seed=seed,paired_initial_exact=True,zero_projection_causality=checks,active_feature_causality=active))
    atomic_json(dict(checks=rows,optimizer_updates=0),out/'preflight.json')


def lock_selection(meta,out):
    verify_run(meta);rows={};weights={};n=len(bb.load_data(meta['identity']['manifest'],'pool')['x'])
    for job in meta['experiments']:
        path=out/job['name'];done=read_json(path/'completion.json')
        if done['status']!='complete' or done['metadata']!=dict(manifest=meta,job=job):raise ValueError('Incomplete continuation')
        verify_files(path,done['files']);last=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        verify_history(meta,job,last,n)
        if last['epoch']!=meta['epochs'] or last['history']!=read_json(path/'history.json'):raise ValueError('Incomplete budget')
        best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        if best['metadata']!=last['metadata'] or best['epoch']!=last['best_epoch'] or best['validation']!=last['best_validation']:
            raise ValueError('Best selection mismatch')
        if any(not torch.equal(v,best['model'][k]) for k,v in last['best_model'].items()):raise ValueError('Best model payload differs')
        summary=read_json(path/'training_summary.json')
        if summary['selected_epoch']!=best['epoch'] or summary['validation']!=best['validation']:raise ValueError('Summary selection mismatch')
        if summary['epochs']!=meta['epochs'] or any(summary[k]!=sum(r[k] for r in last['history']) for k in ('windows','encoder_steps','head_steps')):
            raise ValueError('Summary exposure budget mismatch')
        rows[job['name']]=summary;weights[job['name']]={k:sha256(path/f'{k}.pt') for k in ('best','last')}
    lock=dict(manifest=meta,trials=rows,weights=weights)
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json')!=lock:raise ValueError('Selection changed after evaluation')
    atomic_json(lock,out/'selection_lock.json');return lock


@torch.no_grad()
def score(model,data,stats,local,batch,device):
    pred,detail=bb.predict(model,data,stats,local,batch,device)
    scores={};errors={};scores['global'],errors['global']=bb.ab.measure(pred,data,stats)
    for p,row in detail.items():scores[f'p{p}'],errors[f'p{p}']=bb.pr.measure(row['pred'],row['y'],row['mask'],local)
    for group,positions in (('trained',bb.ba.VAL_PREFIXES),('held',bb.ba.HELD_PREFIXES)):
        errors[group]={k:np.mean([errors[f'p{p}'][k] for p in positions],axis=0) for k in bb.pr.METRICS}
        scores[group]=dict(metrics={k:float(v.mean()) for k,v in errors[group].items()})
    crop=er.crop_prediction(torch.tensor(pred),stats,local).numpy();d=detail[128]
    scores['recent'],errors['recent']=bb.pr.measure(crop,d['y'],d['mask'],local)
    decomposition=pf.decompose_path(pred,data['y'],data['mask'],stats['y_scale'][0])
    return scores,errors,decomposition,pred


def decide(meta,all_errors,inventories):
    config=meta['decision'];checks=[]
    for split in SPLITS:
        for kind in ('best','last'):
            for seed in (42,43):
                a=f'path_s{seed}';b=f'control_s{seed}';err=all_errors[split][kind]
                rules=[('global','path',1-config['path_gain'])]
                rules += [('global',k,1+(config['primary_retention'] if k=='primary' else config['family_retention'])) for k in bb.GLOBAL_METRICS if k!='path']
                rules += [(task,'primary',1+config['local_retention']) for task in ('held','recent')]
                for task,metric,factor in rules:
                    row=bb.ab.pair_groups(err[f'{a}/{task}'][metric],factor*err[f'{b}/{task}'][metric],inventories[split])['all']
                    checks.append(dict(dataset=split,checkpoint=kind,seed=seed,task=task,metric=metric,factor=factor,interval=row,
                        passed=bool(row['supported'] and row['high'] is not None and row['high']<=0)))
    old=meta['identity']['manifest']['decision']
    screens={mode:bb.screen_contrast(all_errors,inventories,mode,'w512_endpoint',old['local_improvement'],old) for mode in ('control','path')}
    return dict(status='candidate_for_review' if all(r['passed'] for r in checks) and screens['path']['passed'] else 'no_full_upgrade',
                matched_control_checks=checks,original512_screens=screens,automatic_promotion=False,
                scope='Existing research samples; no deletions or source-seed reselection. More training is isolated by the matched control.')


@torch.no_grad()
def evaluate(meta,out,device='cuda'):
    lock=lock_selection(meta,out);source=Path(meta['source']);sm=meta['identity']['manifest']
    stats=read_json(source/'cache/statistics.json');local=read_json(source/'cache/local_scales.json')
    val=bb.load_data(sm,'val');replays={};report={};all_errors={};inventories={};examples={}
    for job in meta['experiments']:
        model=construct(meta,job,device)
        for kind in ('best','last'):
            ck=torch.load(out/job['name']/f'{kind}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
            actual=validation(meta,job,model,val,stats,local,device)
            expected=ck['validation'] if kind=='best' else ck['history'][-1]['validation']
            ea.require_replay(actual,expected,'selected continuation');replays[job['name']+'/'+kind]=actual
        del model
    atomic_json(replays,out/'validation_replay.json')
    for split in SPLITS:
        data=bb.load_data(sm,split);inventory=read_json(source/f'{split}_inventory.json');inventories[split]=inventory
        atomic_json(inventory,out/f'{split}_inventory.json');report[split]={};all_errors[split]={}
        # Fixed examples are diagnostics only; all per-window errors are retained.
        examples[split]={}
        for kind in ('best','last'):
            scores={};errors={};decompositions={}
            previous=read_json(source/f'{split}_{kind}_errors.json')
            for original_job in sm['experiments']:
                if original_job['name'] not in {f'w{w}_{m}_s{s}' for w,m in ((512,'endpoint'),(768,'joint')) for s in (42,43)}:continue
                name=original_job['name']
                model,_=ea.load_model(dict(source=str(source),identity=meta['identity']),original_job,kind,device)
                s,e,d,p=score(model,data,stats,local,meta['evaluation_batch'],device)
                ea.require_replay(e['global'],previous[name+'/global'],name+'/global')
                ea.require_replay(e['held'],previous[name+'/held'],name+'/held')
                scores[name]=s;decompositions[name]={k:v.tolist() for k,v in d.items()}
                errors.update({name+'/'+task:row for task,row in e.items()});del model,p
            for job in meta['experiments']:
                name=job['name'];model=construct(meta,job,device)
                ck=torch.load(out/name/f'{kind}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);model.requires_grad_(False)
                s,e,d,p=score(model,data,stats,local,meta['evaluation_batch'],device)
                scores[name]=s;decompositions[name]={k:v.tolist() for k,v in d.items()};errors.update({name+'/'+task:row for task,row in e.items()})
                ids=np.linspace(0,len(inventory)-1,min(4,len(inventory)),dtype=int)
                examples[split][name+'/'+kind]=[dict(index=int(i),prediction=p[i].tolist(),truth=data['y'][i].tolist(),mask=data['mask'][i].tolist()) for i in ids]
                del model,ck,p
            paired={}
            for seed in (42,43):
                for a,b in ((f'path_s{seed}',f'control_s{seed}'),(f'control_s{seed}',f'w768_joint_s{seed}'),(f'path_s{seed}',f'w768_joint_s{seed}')):
                    paired[a+'_minus_'+b]={task:{k:bb.ab.pair_groups(errors[a+'/'+task][k],errors[b+'/'+task][k],inventory) for k in errors[a+'/'+task]} for task in ('global','held','recent')}
            report[split][kind]=dict(scores=scores,paired=paired);all_errors[split][kind]=errors
            atomic_json({n:{k:v.tolist() for k,v in e.items()} for n,e in errors.items()},out/f'{split}_{kind}_errors.json')
            atomic_json(decompositions,out/f'{split}_{kind}_path_decomposition.json')
            progress(f'Evaluation complete: {split}/{kind}')
    decision=decide(meta,all_errors,inventories)
    atomic_json(dict(manifest=meta,trials=lock['trials'],datasets=report),out/'path_feature_metrics.json')
    atomic_json(decision,out/'decision.json');atomic_json(dict(normalized=True,statistics=stats,datasets=examples),out/'examples.json')
    lines=['# Causal path input: matched768 continuation','',f'Decision: {decision["status"]}; reused research sets, no automatic promotion.','']
    for split,kinds in report.items():
        for kind,row in kinds.items():
            lines += [f'## {split}/{kind}','','| model | global | path | held local | recent |','|---|---:|---:|---:|---:|']
            for name,s in row['scores'].items():lines.append(f'| {name} | {s["global"]["metrics"]["primary"]:.6f} | {s["global"]["metrics"]["path"]:.6f} | {s["held"]["metrics"]["primary"]:.6f} | {s["recent"]["metrics"]["primary"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(lines))


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);path=out/job['name'];path.mkdir(exist_ok=True);log=(path/'run.log').open('a')
                try:proc=subprocess.Popen([sys.executable,'-m','obson.babel.path_feature_benchmark','worker','--out',str(out),'--name',job['name']],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[job['name']]=(proc,log);progress(f'Started {job["name"]}, pid={proc.pid}')
            for name,(proc,log) in list(active.items()):
                path=out/name/'progress.json'
                if path.exists():
                    try:row=read_json(path)
                    except (OSError,ValueError):row=None
                    if row and seen.get(name)!=row:progress(f'{name}: {row}');seen[name]=row
                code=proc.poll()
                if code is not None:
                    log.close();del active[name]
                    if code:raise RuntimeError(f'{name} exited{code}; see its run.log')
            if active:time.sleep(1)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        signal.signal(signal.SIGTERM,handler)


def run(source,out,epochs=100,micro=64,jobs=2,device='cuda'):
    source,out=source.resolve(),out.resolve();ea.check_output(source,out)
    if jobs<1:raise ValueError('Positive worker count required')
    progress('Verifying complete alignment source and immutable paired parents')
    identity=ea.source_identity(source);meta=make_manifest(source,identity,epochs,micro)
    runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source Torch/NumPy environment')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Run configuration changed')
    elif out.exists() and any(out.iterdir()):raise ValueError('Use an empty output')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json')
        if done['status']!='complete' or done['cells']!=len(meta['experiments']):raise ValueError('Invalid completion status')
        for name,digest in done['files'].items():
            if sha256(out/name)!=digest:raise ValueError('Completed report changed')
        lock=lock_selection(meta,out)
        if done['encoder_steps']!=sum(r['encoder_steps'] for r in lock['trials'].values()):raise ValueError('Completed update budget changed')
        progress('Already complete');return
    atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device),started_unix=time.time()),out/'runtime.json')
    try:
        preflight(meta,out,device);run_jobs(out,jobs);evaluate(meta,out,device)
        if ea.source_identity(source)!=identity:raise ValueError('Source changed during training')
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json')
        raise
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md') and p.name not in ('progress.json','completion.json')}
    atomic_json(dict(status='complete',source_unchanged=True,files=files,cells=len(meta['experiments']),
        encoder_steps=sum(read_json(out/j['name']/'training_summary.json')['encoder_steps'] for j in meta['experiments'])),out/'completion.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker'))
    p.add_argument('--source',type=Path,default=Path('checkpoints/babel_bar_alignment'))
    p.add_argument('--out',type=Path,default=Path('checkpoints/babel_path768'));p.add_argument('--name')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--micro',type=int,default=64);p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal training runs on AutoDL CUDA')
    if a.action=='worker':worker(a.out,a.name);return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Output already running')
        run(a.source,a.out,a.epochs,a.micro,a.jobs)


if __name__=='__main__':main()
