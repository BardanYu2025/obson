"""Four frozen encoders, matched fresh local readers, locked held-position audit."""
import argparse
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from . import local_readapt as lr,capacity_growth_run as gr,capacity_growth_evaluate as ge
from .ae_extend import atomic_json,atomic_save,rng_state,restore_rng
from .dual_state import sha256,verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA='babel-local-readapt768-v1'
ur=gr.ur;ba=ur.bb.ba


def code_identity():return gr.code_identity()|{Path(lr.__file__).name:sha256(lr.__file__),Path(__file__).name:sha256(__file__)}


def source_identity(source):
    m=read_json(source/'manifest.json');d=read_json(source/'completion.json')
    if m['schema']!=gr.SCHEMA or m['code_sha256']!=gr.code_identity() or d['status']!='complete' or not d['source_unchanged'] or d['cells']!=6:raise ValueError('Completed immutable capacity study required')
    ur.bb.ab.sc.verify_worker_files(source,d['files']);ge.check_models(source);ge.check_readouts(source)
    if gr.source_identity(Path(m['source']))!=m['identity']:raise ValueError('Parent/data lineage changed')
    lock=read_json(source/'model_selection_lock.json');files=d['files']|{'completion.json':sha256(source/'completion.json')};n=read_json(source/'train_plan.json')['eligible']
    for j in m['experiments']:
        root=source/j['name'];done=read_json(root/'completion.json');s=read_json(root/'training_summary.json');h=read_json(root/'history.json')
        if done['status']!='complete' or done['metadata']!=dict(manifest=m,job=j) or lock['trials'][j['name']]!=s:raise ValueError('Source training metadata changed')
        verify_files(root,done['files']);gr.verify_history(m,j,dict(epoch=m['epochs'],history=h,initial_validation=read_json(root/'initial_validation.json')['validation'],best_epoch=s['selected_epoch'],best_validation=s['validation']),n)
        for k in ('pairs','views','encoder_steps','head_steps'):
            if s[k]!=sum(v[k] for v in h):raise ValueError('Source exposure budget changed')
        files.update({j['name']+'/'+k:v for k,v in done['files'].items()});files[j['name']+'/completion.json']=sha256(root/'completion.json')
    return dict(manifest=m,files=files)


def check_output(source,out,identity=None):
    gr.rr.check_output(source,out)
    if identity is not None:
        m=identity['manifest'];gr.check_output(Path(m['source']),out,m['identity'])


def make_manifest(source,identity):
    gm=identity['manifest']
    if gm['state_width']!=768 or gm['epochs']!=100:raise ValueError('Expected fixed768 growth source')
    return dict(schema=SCHEMA,source=str(source),identity=identity,code_sha256=code_identity(),epochs=100,batch=128,extract_batch=128,hidden=256,lr=1e-3,width=768,
        experiments=[dict(name=f'{v}_s{s}',variant=v,seed=s) for s in (42,43) for v in ('base','deep')],
        train_prefixes=list(ba.TRAIN_PREFIXES),val_prefixes=list(ba.VAL_PREFIXES),held_prefixes=list(ba.HELD_PREFIXES),
        source_checkpoint='Predeclared growth validation-best for base/deep only. No research re-selection; no new width/head sweep.',
        initialization='Same-seed identical fresh768->256->112 GELU reader. Per-encoder train-only pooled nonheld state scaling; no learned extra capacity. Original paired head retained as frozen reference, never selects new head.',
        objective='Local past16 historical SmoothL1 only, same target masks/scales. Four distinct nonheld prefixes/window/epoch; no encoder/decoder update.',
        budget='Original4789 train windows, not prior36106 paired pool. Each arm100 epochs x4789 windows x4 prefixes=1915600 local exposures,3800 AdamW steps. Same order/positions across base/deep. Validation978 windows only32/64/96/128.',
        selection='Validation local MSE selects epoch incl0; all4 complete budgets and best/last locked before research extraction/scoring. No held position selects epoch or scaling.',
        decision='Both seeds/sets/best+last: deep held primary <=1.05 recalibrated base AND old deep AND old base; held path/body/activity <=1.10 references; trained-prefix primary <=1.05 references. Weekly upper CI<=0, original support rules. Passing supports bounded decoder adaptability only, does not erase prior capacity/utility failure.',
        stop='One fixed100-epoch reader budget; no head/LR grid, no automatic extension. Failure is inconclusive about absolute information loss. No original utility refit; frozen endpoint/global/utility certificates preserved, not claimed newly improved.')


def cache_positions(meta,split):
    if split=='train':return meta['train_prefixes']
    if split=='val':return meta['val_prefixes']
    if split in ('test','cross_research'):return sorted(meta['val_prefixes']+meta['held_prefixes'])
    raise ValueError('Unknown split')


def verify_cache(meta,out,split):
    idx=read_json(out/f'cache/{split}_index.json')
    if idx['manifest_sha256']!=sha256(out/'manifest.json') or idx['positions']!=cache_positions(meta,split):raise ValueError('Cache configuration changed')
    verify_files(out/'cache',idx['files']);return idx


def original_model(meta,job,device):
    gm=meta['identity']['manifest'];original=next(j for j in gm['experiments'] if j['name']==job['name'])
    return ge.load(gm,Path(meta['source']),original,'best',device)


@torch.no_grad()
def prepare(meta,out,split,device):
    if split in ('test','cross_research'):check_selection(meta,out)
    positions=cache_positions(meta,split);cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists():verify_cache(meta,out,split);return
    gm=meta['identity']['manifest'];cm=gr.parent_meta(gm);data=ur.bb.load_data(gr.cr.alignment(cm),split);stats,local=gr.statistics(gm);n=len(data['x']);batch=meta['extract_batch'];files={}
    rows=read_json(out/f'{split}_inventory.json')
    if len(rows)!=n or rows!=read_json(Path(meta['source'])/f'{split}_inventory.json'):raise ValueError('Source rows changed')
    def mmap(name,shape,dtype):return np.lib.format.open_memmap(cache/name,mode='w+',dtype=dtype,shape=shape)
    y=mmap(f'{split}_y.npy',(n,len(positions),16,7),np.float32);mask=mmap(f'{split}_mask.npy',y.shape,bool)
    for start in range(0,n,batch):
        end=min(start+batch,n);ps=torch.tensor([positions]*(end-start));target,valid=ba.local_targets(torch.tensor(np.asarray(data['y'][start:end])),torch.tensor(np.asarray(data['mask'][start:end])),ps,stats,local);y[start:end]=target.numpy();mask[start:end]=valid.numpy()
    y.flush();mask.flush();del y,mask
    for suffix in ('y','mask'):files[f'{split}_{suffix}.npy']=sha256(cache/f'{split}_{suffix}.npy')
    source=Path(meta['source']);gr.rr.old.verify_cache(gm,source,split);endpoint=gr.rr.old.arrays(source,split)
    for job in meta['experiments']:
        model,expected=original_model(meta,job,device);model.eval().requires_grad_(False);before=ur.bb.state_signature(model)
        if split=='train':
            original=next(j for j in gm['experiments'] if j['name']==job['name']);actual=gr.validation(gm,original,model,device);ur.pf.ea.require_replay(actual,expected,'frozen growth validation')
            val=ur.bb.load_data(gr.cr.alignment(cm),'val');causal=ur.pf.er.trained_causality(model,torch.tensor(np.asarray(val['x'][:4]),device=device))
            if causal['status']=='failed':raise ValueError('Frozen model noncausal')
            atomic_json(dict(validation=actual,causality=causal),out/f'{job["name"]}_frozen_audit.json')
        name=f'{split}_{job["name"]}_x.npy';x=mmap(name,(n,len(positions),meta['width']),np.float32)
        for start in range(0,n,batch):
            z=model.core.encoder(torch.tensor(np.asarray(data['x'][start:start+batch]),device=device));reference=torch.tensor(endpoint[job['name']+'_best'][start:start+batch],device=device)
            if not gr.growth.compare(z[:,-1],reference)['passed']:raise ValueError('Endpoint state differs from growth cache')
            x[start:start+batch]=z[:,np.array(positions)-1].cpu().numpy()
            if start and start%(batch*8)==0:progress(f'Caching {split}/{job["name"]}: {min(start+batch,n)}/{n}')
        if not np.isfinite(x).all() or before!=ur.bb.state_signature(model):raise ValueError('Nonfinite cache or mutated frozen model')
        x.flush();del x;files[name]=sha256(cache/name);del model
        progress(f'Cached {split}/{job["name"]}, {n} windows x{len(positions)} positions')
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),positions=positions,windows=n,files=files),cache/f'{split}_index.json')


def arrays(out,split,name):
    return {k:np.load(out/f'cache/{split}_{name+"_" if k=="x" else ""}{k}.npy',mmap_mode='r',allow_pickle=False) for k in ('x','y','mask')}


def device_arrays(data,device):return {k:torch.tensor(np.asarray(v),device=device) for k,v in data.items()}


def publish(state,path):
    atomic_save(state,path/'last.pt');atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],validation=state['best_validation'],model=state['best_model']),path/'best.pt');atomic_json(state['history'],path/'history.json')
    row=state['history'][-1] if state['history'] else {};atomic_json(dict(epoch=state['epoch'],epochs=state['metadata']['manifest']['epochs'],best_epoch=state['best_epoch'],seconds=row.get('seconds')),path/'progress.json')


def plan(n,seed,epoch):
    order,ps=lr.schedule(n,seed,epoch);return order,ps,dict(order=ur.bb.cov.ndarray_hash(order),prefixes=ur.bb.cov.ndarray_hash(ps))


def verify_history(meta,job,state,n):
    if [v['epoch'] for v in state['history']]!=list(range(1,state['epoch']+1)) or state['epoch']>meta['epochs']:raise ValueError('Incomplete reader history')
    steps=(n+meta['batch']-1)//meta['batch']
    for v in state['history']:
        if v['sampling']!=plan(n,job['seed'],v['epoch'])[2] or v['windows']!=n or v['prefix_exposures']!=4*n or v['steps']!=steps:raise ValueError('Reader exposure mismatch')
        if v['lr']!=gr.growth.learning_rate(v['epoch'],meta['epochs'],meta['lr']):raise ValueError('Reader LR changed')
    best=min([(0,state['initial_validation'])]+[(v['epoch'],v['validation']) for v in state['history']],key=lambda v:v[1]['primary'])
    if best!=(state['best_epoch'],state['best_validation']):raise ValueError('Reader selection changed')


def worker(out,name,device='cuda'):
    meta=read_json(out/'manifest.json')
    if meta['code_sha256']!=code_identity() or meta['schema']!=SCHEMA:raise ValueError('Worker code changed')
    job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True)
    indexes={s:verify_cache(meta,out,s) for s in ('train','val')};metadata=dict(manifest=meta,job=job,cache={s:sha256(out/f'cache/{s}_index.json') for s in indexes})
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['metadata']!=metadata or done['status']!='complete':raise ValueError('Reader completion changed')
        verify_files(path,done['files']);return
    host=arrays(out,'train',name);stats=lr.scales(host['x']);n=len(host['x']);data=device_arrays(host,device);val=device_arrays(arrays(out,'val',name),device)
    head=lr.reader(meta['width'],meta['hidden'],job['seed'],device);optimizer=torch.optim.AdamW(head.parameters(),lr=meta['lr'],weight_decay=1e-4)
    initial_signature=ur.bb.state_signature(head)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata or state['scales']!=stats or state['initial_signature']!=initial_signature:raise ValueError('Resume configuration changed')
        verify_history(meta,job,state,n);head.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation'];ur.pf.ea.require_replay(lr.validate(head,val,stats,meta['batch']),expected,'reader resume')
    else:
        initial=lr.validate(head,val,stats,meta['batch']);state=dict(metadata=metadata,scales=stats,initial_signature=initial_signature,epoch=0,history=[],initial_validation=initial,best_epoch=0,best_validation=initial,best_model=ur.bb.ab.cpu_state(head))
        atomic_json(dict(validation=initial,initial_signature=initial_signature,reader_parameters=sum(p.numel() for p in head.parameters())),path/'initial_validation.json')
    def save():
        state.update(model=ur.bb.ab.cpu_state(head),optimizer=optimizer.state_dict(),rng=rng_state());publish(state,path)
    save()
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started=time.monotonic();order,ps,sampling=plan(n,job['seed'],epoch);rate=gr.growth.learning_rate(epoch,meta['epochs'],meta['lr'])
        for g in optimizer.param_groups:g['lr']=rate
        trained=lr.epoch(head,data,stats,meta['train_prefixes'],order,ps,meta['batch'],optimizer);score=lr.validate(head,val,stats,meta['batch'])
        if score['primary']<state['best_validation']['primary']:state.update(best_epoch=epoch,best_validation=score,best_model=ur.bb.ab.cpu_state(head))
        state['history'].append(dict(epoch=epoch,lr=rate,sampling=sampling,validation=score,**trained,seconds=time.monotonic()-started));state['epoch']=epoch;save();progress(f'{name}: {epoch}/{meta["epochs"]}, validation={score["primary"]:.6f}, best={state["best_epoch"]}')
    verify_history(meta,job,state,n)
    atomic_json(dict(selected_epoch=state['best_epoch'],epochs=state['epoch'],validation=state['best_validation'],encoder_updates=0,**{k:sum(v[k] for v in state['history']) for k in ('windows','prefix_exposures','steps')},seconds=sum(v['seconds'] for v in state['history']),tail_validation=[v['validation']['primary'] for v in state['history'][-20:]]),path/'training_summary.json')
    files={n:sha256(path/n) for n in ('best.pt','last.pt','history.json','initial_validation.json','training_summary.json')};atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


def lock_selection(meta,out):
    trials={};weights={};n=verify_cache(meta,out,'train')['windows']
    for j in meta['experiments']:
        path=out/j['name'];d=read_json(path/'completion.json');last=torch.load(path/'last.pt',map_location='cpu',weights_only=True);best=torch.load(path/'best.pt',map_location='cpu',weights_only=True);summary=read_json(path/'training_summary.json')
        expected=dict(manifest=meta,job=j,cache={s:sha256(out/f'cache/{s}_index.json') for s in ('train','val')})
        if d['status']!='complete' or d['metadata']!=expected or last['metadata']!=expected or best['metadata']!=expected:raise ValueError('All reader completions required')
        verify_files(path,d['files']);verify_history(meta,j,last,n)
        if last['epoch']!=meta['epochs'] or best['epoch']!=last['best_epoch'] or best['validation']!=last['best_validation'] or summary['selected_epoch']!=best['epoch'] or summary['validation']!=best['validation'] or read_json(path/'history.json')!=last['history']:raise ValueError('Reader checkpoint selection mismatch')
        if any(not torch.equal(v,best['model'][k]) for k,v in last['best_model'].items()):raise ValueError('Selected reader payload mismatch')
        for k in ('windows','prefix_exposures','steps'):
            if summary[k]!=sum(v[k] for v in last['history']):raise ValueError('Reader budget mismatch')
        trials[j['name']]=summary;weights[j['name']]={k:sha256(path/f'{k}.pt') for k in ('best','last')}
    lock=dict(manifest_sha256=sha256(out/'manifest.json'),trials=trials,weights=weights,cache_indexes={s:sha256(out/f'cache/{s}_index.json') for s in ('train','val')})
    path=out/'selection_lock.json'
    if path.exists() and read_json(path)!=lock:raise ValueError('Selections changed')
    atomic_json(lock,path);return lock


def check_selection(meta,out):
    lock=read_json(out/'selection_lock.json');names={j['name'] for j in meta['experiments']}
    if lock['manifest_sha256']!=sha256(out/'manifest.json') or set(lock['trials'])!=names or set(lock['weights'])!=names or set(lock['cache_indexes'])!={'train','val'}:raise ValueError('All four readers must be locked')
    for n,w in lock['weights'].items():
        if set(w)!={'best','last'}:raise ValueError('Both reader checkpoints required')
        for k,h in w.items():
            if sha256(out/n/f'{k}.pt')!=h:raise ValueError('Locked reader changed')
    for s,h in lock['cache_indexes'].items():
        if sha256(out/f'cache/{s}_index.json')!=h:raise ValueError('Fitting cache changed')
        verify_cache(meta,out,s)
    return lock


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    try:
        while pending or active:
            while pending and len(active)<jobs:
                j=pending.pop(0);path=out/j['name'];path.mkdir(exist_ok=True);log=(path/'run.log').open('a')
                try:p=subprocess.Popen([sys.executable,'-m','obson.babel.local_readapt_run','worker','--out',str(out),'--name',j['name']],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[j['name']]=(p,log);progress(f'Started {j["name"]}, pid={p.pid}')
            for name,(p,log) in list(active.items()):
                f=out/name/'progress.json'
                if f.exists():
                    row=read_json(f)
                    if seen.get(name)!=row['epoch']:seen[name]=row['epoch'];progress(f'{name}: {row}')
                if p.poll() is not None:
                    log.close();del active[name]
                    if p.returncode:raise RuntimeError(f'{name} failed ({p.returncode}); see worker run.log')
                    progress(f'Completed {name}')
            if active:time.sleep(1)
    finally:
        for p,log in active.values():
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()


def decide(records,rows):
    checks=[]
    for split,bank in records.items():
        for seed in (42,43):
            for kind in ('best','last'):
                candidate=f'deep_s{seed}_{kind}'
                for reference in (f'base_s{seed}_{kind}',f'deep_s{seed}_original',f'base_s{seed}_original'):
                    for task,metric,factor in [('held','primary',1.05),('trained','primary',1.05),('held','path',1.10),('held','body',1.10),('held','activity',1.10)]:
                        a=np.array(bank[candidate]['errors'][task][metric]);b=np.array(bank[reference]['errors'][task][metric]);ci=ur.interval(a,factor*b,rows[split])
                        checks.append(dict(dataset=split,seed=seed,checkpoint=kind,reference=reference,task=task,metric=metric,factor=factor,interval=ci,passed=bool(ci['supported'] and ci['high'] is not None and ci['high']<=0)))
    passed=bool(checks) and all(v['passed'] for v in checks)
    return dict(status='local_readout_retention_recovered' if passed else 'local_readout_retention_unresolved',checks=checks,encoder_updates=0,automatic_promotion=False,
        scope='Matched fresh local readers of frozen validation-best base/deep states. Both seeds/sets and selected+last readers required. Passing supports same-capacity readout adaptability, not universal encoding or full capacity upgrade. Failure under100 epochs cannot prove information absent. Original utility/capacity verdict remains unchanged.')


@torch.no_grad()
def evaluate(meta,out,device):
    lock=check_selection(meta,out);_,local=gr.statistics(meta['identity']['manifest']);source=Path(meta['source']);records={};inventories={};replay={}
    for split in ('test','cross_research'):
        index=verify_cache(meta,out,split);positions=index['positions'];rows=read_json(out/f'{split}_inventory.json');inventories[split]=rows;records[split]={}
        for job in meta['experiments']:
            name=job['name'];data=arrays(out,split,name);original,_=original_model(meta,job,device);original.eval().requires_grad_(False);before=ur.bb.state_signature(original);old=read_json(source/f'{split}_{name}_best.json');last=torch.load(out/name/'last.pt',map_location='cpu',weights_only=True)
            for kind in ('original','best','last'):
                if kind=='original':head=original.local_head;stats=None
                else:
                    ck=torch.load(out/name/f'{kind}.pt',map_location='cpu',weights_only=True);head=lr.reader(meta['width'],meta['hidden'],job['seed'],device);head.load_state_dict(ck['model']);head.eval().requires_grad_(False);stats=last['scales']
                    expected=ck['validation'] if kind=='best' else ck['history'][-1]['validation'];actual=lr.validate(head,device_arrays(arrays(out,'val',name),device),stats,meta['batch']);ur.pf.ea.require_replay(actual,expected,'selected reader validation')
                pred=[]
                for start in range(0,len(data['x']),meta['batch']):
                    x=torch.tensor(np.asarray(data['x'][start:start+meta['batch']]),device=device)
                    p=head(x).reshape(*x.shape[:-1],16,7) if stats is None else lr.predict(head,x,stats)
                    pred.append(p.cpu().numpy())
                pred=np.concatenate(pred);scores={};errors={}
                for i,prefix in enumerate(positions):scores[f'p{prefix}'],errors[f'p{prefix}']=ur.bb.pr.measure(pred[:,i],np.asarray(data['y'][:,i]),np.asarray(data['mask'][:,i]),local)
                for group,ps in [('trained',meta['val_prefixes']),('held',meta['held_prefixes'])]:
                    errors[group]={k:np.mean([errors[f'p{p}'][k] for p in ps],axis=0) for k in ur.bb.pr.METRICS};scores[group]=dict(metrics={k:float(v.mean()) for k,v in errors[group].items()})
                if kind=='original':
                    for key in scores:gr.ce.require_nested(scores[key],old['reconstruction']['scores'][key],split+'/'+name+'/'+key+'/original local replay')
                    replay[split+'/'+name]=dict(original_head_replayed=True,frozen_model_signature=before,endpoint_certificate_sha256=sha256(source/f'{split}_{name}_best.json'))
                value=dict(scores=scores,errors={p:{k:v.tolist() for k,v in e.items()} for p,e in errors.items()},examples=[dict(index=int(i),positions=positions,prediction=pred[i].tolist(),target=data['y'][i].tolist(),mask=data['mask'][i].tolist()) for i in np.linspace(0,len(rows)-1,4,dtype=int)])
                atomic_json(value,out/f'{split}_{name}_{kind}.json');records[split][name+'_'+kind]=value
            if before!=ur.bb.state_signature(original):raise ValueError('Frozen encoder/decoder/head mutated')
            del original;progress(f'Local readout evaluation: {split}/{name}, original/best/last')
    decision=decide(records,inventories);atomic_json(decision,out/'decision.json');atomic_json(replay,out/'frozen_replay.json')
    inherited=read_json(source/'decision.json');atomic_json(dict(source_status=inherited['status'],original_utility_protocol=inherited['original_utility_protocol'],source_decision_sha256=sha256(source/'decision.json'),encoder_updates=0,global_decoder_updates=0,utility_readout_updates=0,interpretation='Source global/overlap/utility results preserved by unchanged model/reader files and endpoint cache replay. Not newly improved or requalified.'),out/'unchanged_capabilities.json')
    summary={s:{n:{task:v['scores'][task]['metrics']['primary'] for task in ('trained','held')} for n,v in bank.items()} for s,bank in records.items()};atomic_json(dict(summary=summary,decision=decision),out/'local_readapt_metrics.json')
    lines=['# Frozen768 local-reader adaptation','',f'Decision: {decision["status"]}; original capacity verdict unchanged.','', '| dataset/model/head | trained-prefix error | held-prefix error |','|---|---:|---:|']
    for s,bank in summary.items():
        for n,v in bank.items():lines.append(f'| {s}/{n} | {v["trained"]:.6f} | {v["held"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def run(source,out,jobs=2,device='cuda'):
    source,out=source.resolve(),out.resolve();check_output(source,out)
    if jobs not in (1,2):raise ValueError('One or two reader workers supported')
    progress('Verifying frozen capacity checkpoints, data banks and original readers')
    identity=source_identity(source);check_output(source,out,identity);meta=make_manifest(source,identity);runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source runtime')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Source/configuration changed; use new output')
    elif out.exists() and any(out.iterdir()):raise ValueError('Empty output required')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        d=read_json(out/'completion.json')
        if d['status']!='complete' or d['encoder_updates']!=0:raise ValueError('Invalid completion')
        ur.bb.ab.sc.verify_worker_files(out,d['files']);check_selection(meta,out);progress('Already complete; no repeated training');return
    started=time.monotonic()
    try:
        for s in ('train','val','test','cross_research'):shutil.copyfile(source/f'{s}_inventory.json',out/f'{s}_inventory.json')
        atomic_json(gr.rr.up.probe.inventory_audit({s:read_json(out/f'{s}_inventory.json') for s in gr.rr.old.SPLITS}),out/'data_audit.json')
        for s in ('train','val'):prepare(meta,out,s,device)
        run_jobs(out,jobs);lock_selection(meta,out)
        for s in ('test','cross_research'):prepare(meta,out,s,device)
        evaluate(meta,out,device)
        if source_identity(source)!=identity:raise ValueError('Frozen source changed')
    except Exception as exc:atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json');raise
    atomic_json(dict(seconds=time.monotonic()-started,torch=str(torch.__version__),numpy=np.__version__,jobs=jobs,device=str(device)),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md') and p.name not in ('completion.json','failure.json','progress.json')};atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,cells=4,files=files),out/'completion.json')


def main():
    import signal
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker'));p.add_argument('--source',type=Path,default=Path('checkpoints/babel_growth768'));p.add_argument('--out',type=Path,default=Path('checkpoints/babel_local_readapt768'));p.add_argument('--name');p.add_argument('--jobs',type=int,default=2);a=p.parse_args();ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal inference and reader training run on AutoDL CUDA only')
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if a.action=='worker':worker(a.out,a.name);return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Output already running')
        run(a.source,a.out,a.jobs)


if __name__=='__main__':main()
