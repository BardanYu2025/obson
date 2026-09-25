"""Bounded six-arm FFN/depth continuation with matched paired exposures."""
import argparse
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from . import capacity_growth as growth, consistency_readout as rr
from .ae_extend import atomic_json,atomic_save,rng_state,restore_rng
from .dual_state import sha256,verify_files
from .holdout_audit import read_json
from .progress import progress

cr=rr.cr;ur=rr.ur;ce=rr.ce
SCHEMA='babel-capacity-growth768-v1'


def code_identity():
    from . import capacity_growth_evaluate as ev
    return rr.code_identity()|{Path(m.__file__).name:sha256(m.__file__) for m in (growth,ev)}|{Path(__file__).name:sha256(__file__)}


def source_identity(source):
    rm=read_json(source/'manifest.json');done=read_json(source/'completion.json')
    if rm['schema']!=rr.SCHEMA or rm['code_sha256']!=rr.code_identity() or done['status']!='complete' or not done['source_unchanged'] or done['encoder_updates']!=0:
        raise ValueError('Completed immutable consistency recalibration required')
    ur.bb.ab.sc.verify_worker_files(source,done['files']);rr.check_selection(source)
    if rr.source_identity(Path(rm['source']))!=rm['identity']:raise ValueError('Consistency checkpoint lineage changed')
    if read_json(source/'decision.json')['status']!='readout_recovered_stability_candidate':raise ValueError('Recovered parent required')
    cm=rm['identity']['manifest']
    if cr.bank_identity(cm['identity'])!=cm['packed']:raise ValueError('Paired original banks changed')
    return dict(manifest=rm,files=done['files']|{'completion.json':sha256(source/'completion.json')})


def parent_meta(meta):return meta['identity']['manifest']['identity']['manifest']
def parent_root(meta):return Path(meta['identity']['manifest']['source'])
def statistics(meta):return cr.statistics(parent_meta(meta))
def train_plan(meta):return read_json(parent_root(meta)/'train_plan.json')


def make_manifest(source,identity,epochs=100,micro=64):
    if epochs!=100 or micro not in (16,32,64):raise ValueError('Fixed100 epochs; micro16/32/64 pairs')
    cm=identity['manifest']['identity']['manifest']
    if cm['epochs']!=100 or cm['batch']!=128 or cm['budget']!=4789:raise ValueError('Expected completed400-epoch paired parent')
    return dict(schema=SCHEMA,source=str(source),identity=identity,code_sha256=code_identity(),epochs=epochs,micro=micro,batch=128,budget=4789,
        evaluation_batch=128,state_width=768,encoder_lr=1e-5,head_lr=3e-5,warmup=5,weight=.1,
        experiments=[dict(name=f'{name}_s{s}',variant=name,seed=s,layers=layers,ff=ff) for s in (42,43) for name,layers,ff in growth.VARIANTS],
        initialization='All arms start from same-seed consistent best400 (also last400). Widened FFN retains old units, new output columns zero; appended prenorm blocks have zero residual output projections. Numeric initial-function replay mandatory.',
        optimizer='All arms deliberately reset BOTH AdamW optimizers, no inherited moments mixed with new dimensions. Fixed5-epoch warmup to encoder1e-5/head3e-5, cosine to0.1 peak. Reset affects all arms; old frozen parent also evaluated.',
        sampling='Same source36106 eligible paired pool.4789 pairs/epoch,9578 views,38 encoder/head updates. Source schedule at absolute401..500; shifts1/16 and four local prefixes identical across arms. Shift64 held.',
        objective='Unchanged mean global+0.25local SmoothL1 across two views, plus0.10 shared-history change1/body/activity consistency; fixed decoder and target statistics.',
        selection='Original full validation global MSE+0.25local only; epoch0 eligible. All6 full budgets locked before12 best/last readouts fit or research scoring. Best primary AND last confirmation, never select by research.',
        readouts='12 checkpoint states x13 targets x5 original Ridge alphas=780 candidates. Train4789/val978, fixed parent calibrated heads and exact PCA/raw/current baselines. All156 heads locked before research extraction.',
        decision=dict(gain=.05,retention=.05,family_retention=.10,current_r2=.8,current_nmse=.1,
            rule='wide vs base; deep vs wide AND base: >=5% global-primary gain with paired weekly upper CI<=0, both seeds/sets/best+last. Retain reconstruction, overlap, and freshly calibrated utility versus comparison AND frozen400 parent. Prefer eligible deep, otherwise eligible wide; otherwise no_capacity_upgrade.'),
        stop='Fixed budget, no automatic extension/head sweep. Record validation tail slope, elapsed time/peak memory/parameter counts. Equal exposures/updates are not equal compute. Tests reused for research, not final independent generalization. No automatic deployment.')


def construct(meta,job,device,optimizers=False):
    model,_,expected=rr.load_selected(meta['identity']['manifest'],f'consistent_s{job["seed"]}',device)
    info=growth.install(model,job['layers'],job['ff'],job['seed'])
    if not optimizers:return model
    enc=torch.optim.AdamW(model.core.encoder.parameters(),lr=meta['encoder_lr'],weight_decay=.01)
    head=torch.optim.AdamW(model.local_head.parameters(),lr=meta['head_lr'],weight_decay=1e-4)
    if enc.state or head.state:raise ValueError('All optimizers must start fresh in this phase')
    return model,enc,head,expected,info


def validation(meta,job,model,device):
    cm=parent_meta(meta);pj=next(j for j in cm['experiments'] if j['name']==f'consistent_s{job["seed"]}')
    return cr.validation(cm,pj,model,device)


def plan(meta,job,epoch,n):
    ids,ds,ps=cr.oc.schedule(n,meta['budget'],job['seed'],100+epoch)
    return ids,ds,ps,dict(absolute_epoch=400+epoch,ids=ur.bb.cov.ndarray_hash(ids),shifts=ur.bb.cov.ndarray_hash(ds),prefixes=ur.bb.cov.ndarray_hash(ps))


def verify_history(meta,job,state,n):
    if [v['epoch'] for v in state['history']]!=list(range(1,state['epoch']+1)) or state['epoch']>meta['epochs']:raise ValueError('Incomplete epoch history')
    steps=(meta['budget']+meta['batch']-1)//meta['batch']
    for v in state['history']:
        if v['sampling']!=plan(meta,job,v['epoch'],n)[3] or v['pairs']!=meta['budget'] or v['views']!=2*meta['budget'] or v['encoder_steps']!=steps or v['head_steps']!=steps:raise ValueError('Exposure/position schedule changed')
        for key in ('encoder_lr','head_lr'):
            if not np.isclose(v[key],growth.learning_rate(v['epoch'],meta['epochs'],meta[key],meta['warmup']),rtol=1e-14,atol=0):raise ValueError('Learning rate changed')
    best=min([(0,state['initial_validation'])]+[(v['epoch'],v['validation']) for v in state['history']],key=lambda v:v[1]['selection'])
    if best!=(state['best_epoch'],state['best_validation']):raise ValueError('Validation selection changed')


@torch.no_grad()
def preflight(meta,out,device):
    cm=parent_meta(meta);data=ur.bb.load_data(cr.alignment(cm),'val');x=torch.tensor(np.asarray(data['x'][:4]),device=device);ps=torch.tensor([[32,64,96,128]]*len(x),device=device);report={}
    for seed in (42,43):
        parent,_,_=rr.load_selected(meta['identity']['manifest'],f'consistent_s{seed}',device);parent.eval();z=parent.core.encoder(x);glob,loc=parent(x,ps,True)
        for job in [j for j in meta['experiments'] if j['seed']==seed]:
            model,enc,head,expected,info=construct(meta,job,device,True);model.eval()
            g,l=model(x,ps,True);checks=dict(state=growth.compare(model.core.encoder(x),z),global_output=growth.compare(g,glob),local_output=growth.compare(l,loc))
            if not all(v['passed'] for v in checks.values()):raise ValueError(f'Expanded initial function mismatch: {job["name"]}: {checks}')
            causal=ur.pf.er.trained_causality(model,x)
            if causal['status']=='failed':raise ValueError('Expanded initialization noncausal')
            report[job['name']]=dict(initial_function=checks,causality=causal,structure=info,optimizer_reset=True)
            atomic_json(report,out/'preflight.json');del model,enc,head
        del parent


def worker(out,name,device='cuda'):
    meta=read_json(out/'manifest.json')
    if meta['schema']!=SCHEMA or meta['code_sha256']!=code_identity():raise ValueError('Worker code/configuration changed')
    job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True);metadata=dict(manifest=meta,job=job)
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['status']!='complete' or done['metadata']!=metadata:raise ValueError('Worker metadata changed')
        verify_files(path,done['files']);return
    model,enc,head,expected,info=construct(meta,job,device,True);builder=cr.Builder(parent_meta(meta),parent_root(meta));stats,local=statistics(meta)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Resume changed')
        verify_history(meta,job,state,len(builder.plan));model.load_state_dict(state['model']);enc.load_state_dict(state['encoder_optimizer']);head.load_state_dict(state['head_optimizer']);restore_rng(state['rng'])
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        ur.pf.ea.require_replay(validation(meta,job,model,device),expected,'growth resume')
    else:
        initial=validation(meta,job,model,device);ur.pf.ea.require_replay(initial,expected,'expanded parent400 replay')
        torch.manual_seed(job['seed']+20261001)
        state=dict(metadata=metadata,epoch=0,history=[],initial_validation=initial,best_epoch=0,best_validation=initial,best_model=ur.bb.ab.cpu_state(model),structure=info)
        atomic_json(dict(validation=initial,parent_epoch=400,optimizer_reset=True,structure=info),path/'initial_validation.json')
    state.update(model=ur.bb.ab.cpu_state(model),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state());ur.pf.publish(state,path)
    decoder_signature={k:v.detach().cpu().clone() for k,v in model.core.decoder.state_dict().items()}
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started=time.monotonic();ids,ds,ps,sampling=plan(meta,job,epoch,len(builder.plan));rates={k:growth.learning_rate(epoch,meta['epochs'],meta[k],meta['warmup']) for k in ('encoder_lr','head_lr')}
        for g in enc.param_groups:g['lr']=rates['encoder_lr']
        for g in head.param_groups:g['lr']=rates['head_lr']
        train,steps=cr.oc.run_epoch(model,builder,ids,ds,ps,stats,local,meta['batch'],meta['micro'],device,meta['weight'],enc,head)
        result=validation(meta,job,model,device)
        if result['selection']<state['best_validation']['selection']:state.update(best_epoch=epoch,best_validation=result,best_model=ur.bb.ab.cpu_state(model))
        state['history'].append(dict(epoch=epoch,sampling=sampling,train=train,validation=result,**rates,pairs=len(ids),views=2*len(ids),encoder_steps=steps,head_steps=steps,seconds=time.monotonic()-started))
        state.update(epoch=epoch,model=ur.bb.ab.cpu_state(model),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state());ur.pf.publish(state,path)
        progress(f'{name}: epoch={epoch}/{meta["epochs"]}, val={result["selection"]:.6f}, best={state["best_epoch"]}, seconds={state["history"][-1]["seconds"]:.1f}')
    verify_history(meta,job,state,len(builder.plan))
    if any(not torch.equal(v,model.core.decoder.state_dict()[k].cpu()) for k,v in decoder_signature.items()):raise ValueError('Fixed decoder changed')
    summary=dict(selected_epoch=state['best_epoch'],epochs=meta['epochs'],validation=state['best_validation'],structure=info,**{k:sum(v[k] for v in state['history']) for k in ('pairs','views','encoder_steps','head_steps')},training_seconds=sum(v['seconds'] for v in state['history']),peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,decoder_unchanged=True)
    tail=state['history'][-min(20,len(state['history'])):]
    summary['validation_tail']=dict(epochs=[v['epoch'] for v in tail],selection=[v['validation']['selection'] for v in tail],slope_per_epoch=float(np.polyfit([v['epoch'] for v in tail],[v['validation']['selection'] for v in tail],1)[0]))
    atomic_json(summary,path/'training_summary.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


def lock_selection(meta,out):
    trials={};weights={};n=train_plan(meta)['eligible']
    for job in meta['experiments']:
        root=out/job['name'];done=read_json(root/'completion.json')
        if done['status']!='complete' or done['metadata']!=dict(manifest=meta,job=job):raise ValueError('All six complete budgets required')
        verify_files(root,done['files']);last=torch.load(root/'last.pt',map_location='cpu',weights_only=True);best=torch.load(root/'best.pt',map_location='cpu',weights_only=True);verify_history(meta,job,last,n)
        summary=read_json(root/'training_summary.json')
        if last['metadata']!=done['metadata'] or best['metadata']!=done['metadata'] or last['epoch']!=meta['epochs'] or best['epoch']!=last['best_epoch'] or best['validation']!=last['best_validation'] or read_json(root/'history.json')!=last['history']:raise ValueError('Checkpoint/selection mismatch')
        if summary['selected_epoch']!=best['epoch'] or summary['validation']!=best['validation'] or not summary['decoder_unchanged']:raise ValueError('Summary mismatch')
        if any(not torch.equal(v,best['model'][k]) for k,v in last['best_model'].items()):raise ValueError('Best model differs from selected state')
        for k in ('pairs','views','encoder_steps','head_steps'):
            if summary[k]!=sum(v[k] for v in last['history']):raise ValueError('Budget mismatch')
        trials[job['name']]=summary;weights[job['name']]={k:sha256(root/f'{k}.pt') for k in ('best','last')}
    result=dict(manifest=meta,trials=trials,weights=weights)
    if (out/'model_selection_lock.json').exists() and read_json(out/'model_selection_lock.json')!=result:raise ValueError('Model selection changed')
    atomic_json(result,out/'model_selection_lock.json');return result


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    try:
        while pending or active:
            while pending and len(active)<jobs:
                j=pending.pop(0);p=out/j['name'];p.mkdir(exist_ok=True);log=(p/'run.log').open('a')
                try:child=subprocess.Popen([sys.executable,'-m','obson.babel.capacity_growth_run','worker','--out',str(out),'--name',j['name']],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[j['name']]=(child,log);progress(f'Started {j["name"]}, pid={child.pid}')
            for name,(p,log) in list(active.items()):
                progress_file=out/name/'progress.json'
                if progress_file.exists():
                    row=read_json(progress_file);snapshot=(row['epoch'],row['best_epoch'])
                    if seen.get(name)!=snapshot:
                        seen[name]=snapshot;progress(f'{name}: epoch={row["epoch"]}/{row["epochs"]}, best={row["best_epoch"]}, seconds={row.get("seconds")}')
                if p.poll() is not None:
                    log.close();del active[name]
                    if p.returncode:raise RuntimeError(f'{name} failed with exit{p.returncode}; see worker log')
                    progress(f'Completed {name}')
            if active:time.sleep(1)
    finally:
        for p,log in active.values():
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()


def check_output(source,out,identity=None):
    rr.check_output(source,out)
    if identity is not None:
        rm=identity['manifest'];rr.check_output(Path(rm['source']),out,rm['identity'])


def run(source,out,jobs=2,micro=64,device='cuda'):
    from . import capacity_growth_evaluate as ev
    source,out=source.resolve(),out.resolve();check_output(source,out)
    if jobs not in (1,2):raise ValueError('One or two GPU workers supported')
    progress('Checking recovered parent, all checkpoints/readers and paired data banks')
    identity=source_identity(source);check_output(source,out,identity);meta=make_manifest(source,identity,micro=micro);runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source runtime')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Configuration/source changed; new output required')
    elif out.exists() and any(out.iterdir()):raise ValueError('Empty output required')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json');ur.bb.ab.sc.verify_worker_files(out,done['files']);lock_selection(meta,out);ev.check_readouts(out)
        if done['status']!='complete':raise ValueError('Invalid completion')
        progress('Already complete; no retraining');return
    started=time.time()
    try:
        for split in rr.old.SPLITS:shutil.copyfile(source/f'{split}_inventory.json',out/f'{split}_inventory.json')
        atomic_json(train_plan(meta),out/'train_plan.json');atomic_json(rr.up.probe.inventory_audit({s:read_json(out/f'{s}_inventory.json') for s in rr.old.SPLITS}),out/'data_audit.json')
        preflight(meta,out,device);run_jobs(out,jobs);lock_selection(meta,out)
        for split in ('train','val'):ev.prepare(meta,out,split,device)
        ev.fit(meta,out,device)
        for split in ('test','cross_research'):ev.prepare(meta,out,split,device)
        ev.evaluate(meta,out,device)
        if source_identity(source)!=identity:raise ValueError('Frozen source changed')
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json');raise
    atomic_json(dict(seconds=time.time()-started,torch=str(torch.__version__),numpy=np.__version__,jobs=jobs,device=str(device),ridge_candidates=780),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.npz') and p.name not in ('completion.json','failure.json','progress.json')}
    atomic_json(dict(status='complete',source_unchanged=True,cells=6,files=files),out/'completion.json')


def main():
    import signal
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker'));p.add_argument('--source',type=Path,default=Path('checkpoints/babel_consistency_readout768'));p.add_argument('--out',type=Path,default=Path('checkpoints/babel_growth768'));p.add_argument('--name');p.add_argument('--jobs',type=int,default=2);p.add_argument('--micro',type=int,default=64);a=p.parse_args();ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal training/inference/reader fitting only on AutoDL CUDA')
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if a.action=='worker':worker(a.out,a.name);return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Output already running')
        run(a.source,a.out,a.jobs,a.micro)


if __name__=='__main__':main()
