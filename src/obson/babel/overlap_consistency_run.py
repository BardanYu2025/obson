"""Matched continuation: original supervision versus shared-history consistency."""
import argparse
import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from . import overlap_consistency as oc
from .ae_extend import atomic_json,atomic_save,rng_state,restore_rng
from .dual_state import sha256,verify_files
from .holdout_audit import read_json
from .progress import progress

odr=oc.odr;ur=odr.ur;pf=ur.pf
SCHEMA='babel-paired-consistency768-v1'


def code_identity():
    from . import overlap_consistency_evaluate as ev
    return odr.code_identity()|{Path(m.__file__).name:sha256(m.__file__) for m in (oc,ev)}|{Path(__file__).name:sha256(__file__)}


def bank_identity(identity):
    result=odr.banks(identity);original=Path(identity['manifest']['identity']['manifest']['original_source'])
    am=read_json(original/'manifest.json');bank=Path(am['bank'])
    for split in ('train','val'):
        files={}
        for suffix in ('x.npy','sequences.json'):
            p=bank/f'{split}_{suffix}';digest=am['source_identity'][str(p.resolve())];ur.bb.cov.file_check(p,digest);files[str(p)]=digest
        result[split]=dict(directory=str(bank),files=files)
    return result


def reader_identity(root,identity):
    meta=read_json(root/'manifest.json');done=read_json(root/'completion.json')
    if meta['schema']!='babel-frozen-utility768-v2' or meta['identity']!=identity or done['status']!='complete':raise ValueError('Completed matching utility768_v2 required')
    ur.bb.ab.sc.verify_worker_files(root,done['files'])
    for name,digest in meta['code_sha256'].items():ur.bb.cov.file_check(Path(__file__).parent/name,digest)
    ur.old.check_selection(root)
    return dict(root=str(root),manifest=meta,files=done['files']|{'completion.json':sha256(root/'completion.json')})


def make_manifest(source,identity,packed,reader,epochs=100,micro=64):
    if epochs<2 or not 0<micro<=128 or 128%micro:raise ValueError('At least2 epochs; micro pairs divides128')
    lock=read_json(source/'selection_lock.json');parents={}
    for seed in (42,43):
        name=f'control_s{seed}';history=read_json(source/name/'history.json')
        if len(history)!=100 or history[-1]['epoch']!=100:raise ValueError('Completed300-epoch source required')
        parents[str(seed)]=dict(name=name,last_sha256=lock['weights'][name]['last'],best_sha256=lock['weights'][name]['best'],
            lr=history[-1]['lr'],head_lr=history[-1]['head_lr'],best_total_epoch=200+lock['trials'][name]['selected_epoch'])
    return dict(schema=SCHEMA,source=str(source),identity=identity,packed=packed,reader=reader,parents=parents,code_sha256=code_identity(),
        epochs=epochs,batch=128,micro=micro,budget=4789,evaluation_batch=128,
        experiments=[dict(name=f'{mode}_s{s}',seed=s,weight=w) for s in (42,43) for mode,w in [('control',0.),('consistent',.1)]],
        shifts=list(oc.TRAIN_SHIFTS),evaluation_shifts=[1,16,64],
        initialization='Restore control last300 model, both AdamW moments and RNG. Best298/300 remains frozen capability reference; no best-model/last-optimizer mixing.',
        sampling='Same eligible dense train pool,4789 pairs/epoch without replacement; same shifts1/16, four prefixes, micro order and38 updates in both arms. Two supervised views per pair; held shift64 scored only after lock.',
        objective='Mean original global+0.25 local SmoothL1 of both views; candidate adds0.10 times equal-family SmoothL1 consistency of change1/body/activity on shared historical bars. Both arms compute identical auxiliary graph; control coefficient0.',
        selection='Original full validation global MSE+0.25 local MSE only, epoch0 eligible. All four arms complete before selection lock/research scoring; best primary, last confirmation. No research choice.',
        decision=dict(gap_gain=.10,reconstruction_retention=.05,family_retention=.10,utility_retention=.05,current_r2=.80,current_nmse=.10),
        readers='Frozen previous control-seed Ridge heads and scalers; no refit. Measures existing readout compatibility, not all possible information access.',
        stop='Single fixed-weight paired experiment. Complete budget even if best stays at0; failure does not trigger alpha/LR/architecture sweeps. Reused research data; no automatic promotion.')


def context(meta):return ur.make_manifest(Path(meta['source']),meta['identity'],meta['evaluation_batch'])
def alignment(meta):return ur.alignment(context(meta))
def statistics(meta):
    root=Path(meta['identity']['manifest']['source'])/'cache'
    return read_json(root/'statistics.json'),read_json(root/'local_scales.json')


def parent_job(meta,job):return next(j for j in meta['identity']['manifest']['experiments'] if j['name']==meta['parents'][str(job['seed'])]['name'])


def construct(meta,job,device,optimizers=False):
    sm=meta['identity']['manifest'];pj=parent_job(meta,job);root=Path(meta['source']);parent=meta['parents'][str(job['seed'])]
    ur.bb.cov.file_check(root/pj['name']/'last.pt',parent['last_sha256'])
    ck=torch.load(root/pj['name']/'last.pt',map_location='cpu',weights_only=True)
    if ck['metadata']!=dict(manifest=sm,job=pj) or ck['epoch']!=100:raise ValueError('Invalid continuation parent')
    if optimizers:
        model,enc,head,_=pf.construct(sm,pj,device,True)
        model.load_state_dict(ck['model']);enc.load_state_dict(ck['encoder_optimizer']);head.load_state_dict(ck['head_optimizer']);restore_rng(ck['rng'])
        if any(g['lr']!=parent['lr'] for g in enc.param_groups) or any(g['lr']!=parent['head_lr'] for g in head.param_groups):raise ValueError('Parent optimizer rates differ')
        return model,enc,head,ck
    model=pf.construct(sm,pj,device);model.load_state_dict(ck['model']);return model


def validation(meta,job,model,device):
    stats,local=statistics(meta)
    return pf.validation(meta['identity']['manifest'],parent_job(meta,job),model,ur.bb.load_data(alignment(meta),'val'),stats,local,device)


def verify_run(meta):
    if meta['schema']!=SCHEMA or meta['code_sha256']!=code_identity():raise ValueError('Implementation changed; use a new run')
    for p in meta['parents'].values():ur.bb.cov.file_check(Path(meta['source'])/p['name']/'last.pt',p['last_sha256'])


def prepare(meta,out):
    if (out/'data_lock.json').exists():
        lock=read_json(out/'data_lock.json')
        if lock['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Data manifest changed')
        verify_files(out,lock['files']);return
    source=alignment(meta);stats,_=statistics(meta);rows=ur.inventories(context(meta))
    ur.up.probe.inventory_audit(rows)
    root=Path(meta['packed']['train']['directory']);bank=np.load(root/'train_x.npy',mmap_mode='r',allow_pickle=False);specs=read_json(root/'train_sequences.json')
    sampling=Path(source['sampling_source']);candidates=read_json(sampling/'candidates/inventory.json');pool=ur.bb.load_data(source,'pool')
    selected,excluded=oc.eligible(specs,rows['train'],candidates,len(bank))
    if len(selected)<meta['budget']:raise ValueError('Paired dense pool below fixed exposure budget')
    replay={k:0. for k in ('x','y','mask')}
    for start in range(0,len(selected),256):
        block=selected[start:start+256];raw=np.stack([bank[p['bank_end']-127:p['bank_end']+1] for p in block]);values=odr.normalized(raw,stats);ids=np.array([p['index'] for p in block])
        for key,value in zip(('x','y','mask'),values):
            reference=np.asarray(pool[key][ids]);diff=float(abs(value.astype(float)-reference.astype(float)).max());replay[key]=max(replay[key],diff)
            if not np.allclose(value,reference,atol=1e-6,rtol=2e-5) or (key=='mask' and not np.array_equal(value,reference)):raise ValueError(f'Dense packed replay mismatch: {key}')
    atomic_json(dict(rows=selected,eligible=len(selected),total_candidates=len(candidates),excluded=excluded,replay=replay),out/'train_plan.json')
    # Copy original fixed inventories for unambiguous research/utility row correspondence.
    for split in ('val','test','cross_research'):atomic_json(rows[split],out/f'{split}_inventory.json')
    names=['train_plan.json']+[f'{s}_inventory.json' for s in ('val','test','cross_research')]
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),files={n:sha256(out/n) for n in names}),out/'data_lock.json')
    progress(f'Paired dense train pool: {len(selected)}/{len(candidates)} eligible, replay={replay}, excluded={excluded}')


class Builder:
    def __init__(self,meta,out):
        self.plan=read_json(out/'train_plan.json')['rows'];self.bank=np.load(Path(meta['packed']['train']['directory'])/'train_x.npy',mmap_mode='r',allow_pickle=False)
        self.pool=ur.bb.load_data(alignment(meta),'pool');self.stats=statistics(meta)[0]
    def __call__(self,indices,shifts):
        rows=[self.plan[i] for i in indices];ids=np.array([p['index'] for p in rows])
        raw=np.stack([self.bank[p['bank_end']-int(d)-127:p['bank_end']-int(d)+1] for p,d in zip(rows,shifts)])
        a=dict(zip(('x','y','mask'),odr.normalized(raw,self.stats)));b={k:np.asarray(v[ids]) for k,v in self.pool.items()}
        return a,b


def plan(meta,job,epoch,n):
    ids,ds,ps=oc.schedule(n,meta['budget'],job['seed'],epoch)
    return ids,ds,ps,dict(absolute_epoch=300+epoch,ids=ur.bb.cov.ndarray_hash(ids),shifts=ur.bb.cov.ndarray_hash(ds),prefixes=ur.bb.cov.ndarray_hash(ps))


def verify_history(meta,job,state,n):
    if [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)) or state['epoch']>meta['epochs']:raise ValueError('Invalid epoch history')
    parent=meta['parents'][str(job['seed'])];steps=(meta['budget']+meta['batch']-1)//meta['batch']
    for r in state['history']:
        expected=plan(meta,job,r['epoch'],n)[3]
        if r['sampling']!=expected or r['pairs']!=meta['budget'] or r['views']!=2*meta['budget'] or r['encoder_steps']!=steps or r['head_steps']!=steps:raise ValueError('Paired budget changed')
        if r['lr']!=pf.pf.learning_rate(r['epoch'],meta['epochs'],parent['lr']) or r['head_lr']!=pf.pf.learning_rate(r['epoch'],meta['epochs'],parent['head_lr']):raise ValueError('LR schedule changed')
    best=min([(0,state['initial_validation'])]+[(r['epoch'],r['validation']) for r in state['history']],key=lambda x:x[1]['selection'])
    if best!=(state['best_epoch'],state['best_validation']):raise ValueError('Validation selection changed')


def worker(out,name,device='cuda'):
    meta=read_json(out/'manifest.json');verify_run(meta);job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True);metadata=dict(manifest=meta,job=job)
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['metadata']!=metadata:raise ValueError('Worker metadata changed')
        verify_files(path,done['files']);return
    prepare(meta,out);builder=Builder(meta,out);stats,local=statistics(meta);model,enc,head,parent=construct(meta,job,device,True)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Resume changed')
        verify_history(meta,job,state,len(builder.plan));model.load_state_dict(state['model']);enc.load_state_dict(state['encoder_optimizer']);head.load_state_dict(state['head_optimizer']);restore_rng(state['rng'])
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        pf.ea.require_replay(validation(meta,job,model,device),expected,'paired resume')
    else:
        initial=validation(meta,job,model,device);pf.ea.require_replay(initial,parent['history'][-1]['validation'],'parent300')
        state=dict(metadata=metadata,epoch=0,history=[],initial_validation=initial,best_epoch=0,best_validation=initial,best_model=ur.bb.ab.cpu_state(model))
        atomic_json(dict(validation=initial,parent_epoch=300,parent_best_epoch=meta['parents'][str(job['seed'])]['best_total_epoch'],optimizer_restored=True),path/'initial_validation.json')
    del parent
    state.update(model=ur.bb.ab.cpu_state(model),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state());pf.publish(state,path)
    parent=meta['parents'][str(job['seed'])]
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started=time.monotonic();ids,ds,ps,sampling=plan(meta,job,epoch,len(builder.plan))
        lr=pf.pf.learning_rate(epoch,meta['epochs'],parent['lr']);hlr=pf.pf.learning_rate(epoch,meta['epochs'],parent['head_lr'])
        for g in enc.param_groups:g['lr']=lr
        for g in head.param_groups:g['lr']=hlr
        train,steps=oc.run_epoch(model,builder,ids,ds,ps,stats,local,meta['batch'],meta['micro'],device,job['weight'],enc,head)
        result=validation(meta,job,model,device)
        if result['selection']<state['best_validation']['selection']:state.update(best_epoch=epoch,best_validation=result,best_model=ur.bb.ab.cpu_state(model))
        state['history'].append(dict(epoch=epoch,sampling=sampling,train=train,validation=result,lr=lr,head_lr=hlr,pairs=len(ids),views=2*len(ids),encoder_steps=steps,head_steps=steps,seconds=time.monotonic()-started))
        state.update(epoch=epoch,model=ur.bb.ab.cpu_state(model),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state());pf.publish(state,path)
        progress(f'{name}: epoch={epoch}/{meta["epochs"]}, validation={result["selection"]:.6f}, consistency={train["consistency"]:.6f}, best={state["best_epoch"]}')
    verify_history(meta,job,state,len(builder.plan))
    atomic_json(dict(selected_epoch=state['best_epoch'],epochs=meta['epochs'],validation=state['best_validation'],pairs=sum(r['pairs'] for r in state['history']),views=sum(r['views'] for r in state['history']),encoder_steps=sum(r['encoder_steps'] for r in state['history']),head_steps=sum(r['head_steps'] for r in state['history']),training_seconds=sum(r['seconds'] for r in state['history']),peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None),path/'training_summary.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


def lock_selection(meta,out):
    trials={};weights={};n=read_json(out/'train_plan.json')['eligible']
    for job in meta['experiments']:
        path=out/job['name'];done=read_json(path/'completion.json')
        if done['status']!='complete' or done['metadata']!=dict(manifest=meta,job=job):raise ValueError('Incomplete paired worker')
        verify_files(path,done['files']);last=torch.load(path/'last.pt',map_location='cpu',weights_only=True);best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        verify_history(meta,job,last,n)
        if last['epoch']!=meta['epochs'] or best['epoch']!=last['best_epoch'] or best['validation']!=last['best_validation'] or best['metadata']!=last['metadata']:raise ValueError('Checkpoint selection mismatch')
        if any(not torch.equal(v,best['model'][k]) for k,v in last['best_model'].items()):raise ValueError('Selected model differs')
        summary=read_json(path/'training_summary.json')
        if summary['selected_epoch']!=best['epoch'] or summary['validation']!=best['validation'] or last['history']!=read_json(path/'history.json'):raise ValueError('Selection summary differs')
        for k in ('pairs','views','encoder_steps','head_steps'):
            if summary[k]!=sum(r[k] for r in last['history']):raise ValueError('Exposure summary differs')
        trials[job['name']]=summary;weights[job['name']]={k:sha256(path/f'{k}.pt') for k in ('best','last')}
    result=dict(manifest=meta,trials=trials,weights=weights)
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json')!=result:raise ValueError('Selection changed after lock')
    atomic_json(result,out/'selection_lock.json');return result


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);path=out/job['name'];path.mkdir(exist_ok=True);log=(path/'run.log').open('a')
                try:p=subprocess.Popen([sys.executable,'-m','obson.babel.overlap_consistency_run','worker','--out',str(out),'--name',job['name']],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[job['name']]=(p,log);progress(f'Started {job["name"]}, pid={p.pid}')
            for name,(p,log) in list(active.items()):
                path=out/name/'progress.json'
                if path.exists():
                    try:row=read_json(path)
                    except (OSError,ValueError):row=None
                    if row and seen.get(name)!=row:progress(f'{name}: {row}');seen[name]=row
                code=p.poll()
                if code is not None:
                    log.close();del active[name]
                    if code:raise RuntimeError(f'{name} exited{code}; see run.log')
            if active:time.sleep(1)
    finally:
        for p,log in active.values():
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()
        signal.signal(signal.SIGTERM,handler)


def run(source,reader,out,epochs=100,micro=64,jobs=2,device='cuda'):
    from . import overlap_consistency_evaluate as ev
    source,reader,out=source.resolve(),reader.resolve(),out.resolve();ur.check_output(source,out)
    if jobs<1 or out==reader or reader in out.parents or out in reader.parents:raise ValueError('Separate output and positive jobs required')
    progress('Verifying paired continuation source, banks and frozen utility readers')
    identity=ur.source_identity(source);packed=bank_identity(identity);ri=reader_identity(reader,identity);meta=make_manifest(source,identity,packed,ri,epochs,micro)
    runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore source runtime')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Continuation config/source changed')
    elif out.exists() and any(out.iterdir()):raise ValueError('Empty output required')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    if (out/'completion.json').exists():
        done=read_json(out/'completion.json')
        if done['status']!='complete' or done['cells']!=4:raise ValueError('Invalid completed run')
        ur.bb.ab.sc.verify_worker_files(out,done['files']);lock_selection(meta,out);progress('Already complete');return
    started=time.time()
    try:
        prepare(meta,out);ev.preflight(meta,out,device);run_jobs(out,jobs);lock_selection(meta,out);ev.evaluate(meta,out,device)
        if ur.source_identity(source)!=identity or bank_identity(identity)!=packed or reader_identity(reader,identity)!=ri:raise ValueError('Immutable sources changed')
    except Exception as exc:
        atomic_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),out/'failure.json');raise
    atomic_json(dict(seconds=time.time()-started,torch=str(torch.__version__),numpy=np.__version__,jobs=jobs,device=str(device)),out/'runtime.json')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md') and p.name not in ('completion.json','progress.json')}
    atomic_json(dict(status='complete',source_unchanged=True,cells=4,files=files),out/'completion.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker'));p.add_argument('--source',type=Path,default=Path('checkpoints/babel_path768'));p.add_argument('--reader',type=Path,default=Path('checkpoints/babel_utility768_v2'));p.add_argument('--out',type=Path,default=Path('checkpoints/babel_consistency768'));p.add_argument('--name');p.add_argument('--epochs',type=int,default=100);p.add_argument('--micro',type=int,default=64);p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal training only on AutoDL CUDA')
    if a.action=='worker':worker(a.out,a.name);return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with (a.out.parent/f'.{a.out.name}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Output already running')
        run(a.source,a.reader,a.out,a.epochs,a.micro,a.jobs)


if __name__=='__main__':main()
