"""Frozen epoch200 states at four prefix positions; matched linear/MLP history probes."""
import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import prefix_readout as pr, sampling_benchmark as sb, pca_teacher_benchmark as tb
from . import architecture as ar, architecture_benchmark as ab, capacity_benchmark as cb
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA='babel-prefix-readout-v1'
GOAL='Diagnose whether individual causal bar states preserve observed history, without updating encoders or training trend-rule labels.'


def code_identity():
    return sb.code_identity() | {Path(pr.__file__).name:sha256(pr.__file__),Path(__file__).name:sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256']!=code_identity():raise ValueError('Prefix probe implementation changed; use a new run')


def source_identity(root):
    sm=read_json(root/'manifest.json');done=read_json(root/'completion.json');sb.verify_code(sm)
    if sm['schema']!=sb.SCHEMA or done['status']!='complete' or not done['source_unchanged']:
        raise ValueError('Completed sampling experiment required')
    ab.sc.verify_worker_files(root,done['files']);sb.verify_cache(sm,root)
    teacher=Path(sm['teacher_source'])
    if sb.source_identity(teacher)!=sm['teacher_identity']:raise ValueError('Original plain controls changed')
    lock=read_json(root/'selection_lock.json')
    if lock['manifest']!=sm:raise ValueError('Sampling selection manifest changed')
    files=dict(done['files']);files['completion.json']=sha256(root/'completion.json');variants={}
    for job in sm['experiments']:
        summary=sb.an.selected_trial(root,sm,job,lambda e:0.)
        sb.verify_sampling_history(sm,root,job,read_json(root/job['name']/'history.json'))
        if summary!=lock['all_trials'][job['name']] or summary['candidate_cache_sha256']!=sha256(root/'candidates/index.json'):
            raise ValueError('Sampling training/selection provenance changed')
        for kind in ('best','last'):
            rel=f'{job["name"]}/{kind}.pt';digest=sha256(root/rel)
            if digest!=lock['weights'][job['name']][kind]:raise ValueError('Sampling weights changed')
            files[rel]=digest
        files[f'{job["name"]}/completion.json']=sha256(root/job['name']/'completion.json')
        variants[job['name']]=dict(seed=job['seed'],root=str(root),job=job,sha256=files[f'{job["name"]}/last.pt'])
    for job in sm['teacher_identity']['controls']:
        if job['name'].startswith('plain_'):
            variants[job['name']]=dict(seed=job['seed'],root=str(teacher),job=job,sha256=sm['teacher_identity']['files'][f'{job["name"]}/last.pt'])
    if set(variants)!={'plain_s42','plain_s43','sampled_s42','sampled_s43'}:
        raise ValueError('Expected two immutable encoder pairs')
    return dict(manifest=sm,files=files,variants=variants)


def make_manifest(source, identity, epochs=100, batch=256, hidden=256):
    if min(epochs,batch,hidden)<1:raise ValueError('Positive probe budgets required')
    variants=dict(identity['variants']);variants['current28']=dict(seed=42)
    jobs=[dict(name=f'{name}_p{p}',variant=name,prefix=p,seed=v['seed']+5000+p*11,
        epochs=epochs,lr=1e-3,width=28 if name=='current28' else identity['manifest']['config']['latent'])
        for name,v in variants.items() for p in pr.PREFIXES]
    return dict(schema=SCHEMA,source=str(source.resolve()),identity=identity,variants=variants,experiments=jobs,
        batch=batch,hidden=hidden,extract_batch=64,prefixes=list(pr.PREFIXES),history=pr.HISTORY,alphas=list(pr.ALPHAS),
        code_sha256=code_identity(),goal=GOAL,encoder_checkpoint='fixed_last_epoch200',
        data='Original fixed4789 train and978 validation windows for every probe; research windows unchanged. No sampled-pool expansion for readouts.',
        objective='Past16 bars excluding current bar, locally reanchored close/body/activity. Equal weight of three families; normalized MSE, no trend labels.',
        selection='Train-only scales; ridge alpha and MLP epoch including0 selected by validation primary. All20 cells locked before research extraction.',
        transfer='Apply each prefix128 head with its own training normalization to earlier states without refitting; diagnostic only, not selected on research.',
        limits='Previously opened research sets, uncorrected paired weekly intervals. Independent position heads diagnose readability, not a universal shared state. current28 has fewer inputs/parameters and is a diagnostic baseline. No automatic encoder promotion.')


def upstream(meta):
    return meta['identity']['manifest']


def encoder(meta, name, device):
    sm=upstream(meta);v=meta['variants'][name];path=Path(v['root'])/name/'last.pt'
    if sha256(path)!=v['sha256']:raise ValueError('Frozen checkpoint changed')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['epoch']!=v['job']['epochs']:raise ValueError('Expected fixed budget-end checkpoint')
    model=tb.model_for(sm,Path(meta['source']),v['job'],device)
    model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
    return model


def verify_cache(meta,out,split):
    cache=out/'cache';index=read_json(cache/f'{split}_index.json')
    if index['manifest']!=meta:raise ValueError('Prefix cache identity mismatch')
    verify_files(cache,index['files'])
    if split!='train' and index['target_scales_sha256']!=sha256(cache/'target_scales.json'):
        raise ValueError('Target scales changed after caching')


@torch.no_grad()
def prepare_split(meta,out,split,device='cuda'):
    if split not in ('train','val','test','cross_research'):raise ValueError('Unknown split')
    if split in ('test','cross_research') and not (out/'selection_lock.json').is_file():
        raise ValueError('Research extraction requires all probe selections locked')
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists():verify_cache(meta,out,split);return
    sm=upstream(meta);source=Path(sm['source']);data=ab.load_arrays(source,split)
    original=read_json(source/'cache/statistics.json');target_rows=[pr.targets(data,original,p) for p in pr.PREFIXES]
    files={}
    if split=='train':
        atomic_json(pr.fit_target_scales(target_rows),cache/'target_scales.json')
        files['target_scales.json']=sha256(cache/'target_scales.json')
    stats=read_json(cache/'target_scales.json')
    for p,(y,mask) in zip(pr.PREFIXES,target_rows):
        normalized=pr.normalize(y,mask,stats)
        for k,a in [('y',normalized),('mask',mask),('current28',np.asarray(data['x'][:,p-1]))]:
            path=cache/f'{split}_p{p}_{k}.npy';np.save(path,a,allow_pickle=False);files[path.name]=sha256(path)
        if split=='train':
            mean=(normalized*mask).sum(0,dtype=np.float64)/mask.sum(0).clip(1)
            path=cache/f'mean_p{p}.npy';np.save(path,mean,allow_pickle=False);files[path.name]=sha256(path)
    for name in meta['identity']['variants']:
        model=encoder(meta,name,device);v=meta['variants'][name]
        # Validate loaded checkpoint on original validation objective before extracting states.
        if split=='val':
            ck=torch.load(Path(v['root'])/name/'last.pt',map_location='cpu',weights_only=True)
            reference=ck['history'][-1]['validation']
            original_val=tb.load_data(sm,Path(meta['source']),'val')
            replay,_=tb.run_epoch(model,original_val,original,sm['batch'],sm['micro'],device,0.)
            if any(not np.isclose(replay[k],reference[k],atol=1e-6,rtol=2e-5) for k in replay):
                raise ValueError(f'Frozen original validation mismatch: {name}')
            atomic_json(dict(matched=True,validation=replay,reference=reference,checkpoint=v),out/f'{name}_frozen_validation.json')
            del ck
        for p in pr.PREFIXES:
            parts=[]
            for start in range(0,len(data['x']),meta['extract_batch']):
                # Strictly truncated inputs: later bars never enter the extraction call.
                x=torch.tensor(np.asarray(data['x'][start:start+meta['extract_batch'],:p]),device=device)
                parts.append(model.encoder(x)[:,-1].cpu().numpy())
            z=np.concatenate(parts)
            if not np.isfinite(z).all():raise ValueError('Nonfinite frozen state')
            path=cache/f'{split}_p{p}_{name}.npy';np.save(path,z,allow_pickle=False);files[path.name]=sha256(path)
        if any(param.requires_grad for param in model.parameters()):raise ValueError('Encoder unexpectedly trainable')
        del model
        progress(f'Frozen states cached: {split}/{name}, prefixes={pr.PREFIXES}')
    atomic_json(dict(manifest=meta,files=files,target_scales_sha256=sha256(cache/'target_scales.json'),
        windows=len(data['x']),encoder_updates=0,strict_prefix_inputs=True),cache/f'{split}_index.json')
    verify_cache(meta,out,split)


def load_data(out,split,job,scales):
    p=job['prefix'];cache=out/'cache'
    x=np.load(cache/f'{split}_p{p}_{job["variant"]}.npy',mmap_mode='r',allow_pickle=False)
    return dict(x=pr.design(x,scales),y=np.load(cache/f'{split}_p{p}_y.npy',mmap_mode='r'),
                mask=np.load(cache/f'{split}_p{p}_mask.npy',mmap_mode='r'))


def cache_identity(out):
    return {s:sha256(out/f'cache/{s}_index.json') for s in ('train','val')}


def head_signature(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def publish(state,path):
    atomic_save(state,path/'last.pt')
    atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],validation=state['best_validation'],model=state['best_model']),path/'best.pt')
    atomic_json(state['history'],path/'history.json')
    row=state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'],best_epoch=state['best_epoch'],validation=row.get('validation'),seconds=row.get('seconds')),path/'progress.json')


def worker(out,name,device='cuda'):
    meta=read_json(out/'manifest.json');verify_code(meta)
    job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True)
    for s in ('train','val'):verify_cache(meta,out,s)
    metadata=dict(manifest=meta,job=job,cache_sha256=cache_identity(out))
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['metadata']!=metadata:raise ValueError('Completed probe identity changed')
        verify_files(path,done['files']);return
    raw=np.load(out/f'cache/train_p{job["prefix"]}_{job["variant"]}.npy',mmap_mode='r')
    scales=pr.feature_scales(raw);atomic_json(scales,path/'feature_scales.json')
    data={s:load_data(out,s,job,scales) for s in ('train','val')};stats=read_json(out/'cache/target_scales.json')
    if not (path/'ridge.json').exists():
        fits=pr.ridge_candidates(data['train']['x'],data['train']['y'],data['train']['mask'],meta['alphas'],device)
        choices=[]
        for alpha,fit in fits.items():
            pred=pr.ridge_predict(fit,data['val']['x']);score,_=pr.measure(pred,data['val']['y'],data['val']['mask'],stats)
            choices.append(dict(alpha=alpha,validation=score['metrics']))
        best=min(choices,key=lambda v:v['validation']['primary']);fit=fits[best['alpha']]
        np.savez(path/'ridge.npz',**fit)
        atomic_json(dict(metadata=metadata,candidates=choices,selected=best,weights_sha256=sha256(path/'ridge.npz')),path/'ridge.json')
    ridge=read_json(path/'ridge.json')
    if ridge['metadata']!=metadata or ridge['weights_sha256']!=sha256(path/'ridge.npz'):
        raise ValueError('Ridge checkpoint changed')
    torch.manual_seed(job['seed']);model=pr.Probe(job['width'],job['seed'],meta['hidden']).to(device)
    initial=dict(seed=job['seed'],width=job['width'],hidden=meta['hidden'],sha256=head_signature(model))
    if (path/'head_initialization.json').exists() and read_json(path/'head_initialization.json')!=initial:
        raise ValueError('Probe initialization changed')
    atomic_json(initial,path/'head_initialization.json')
    opt=torch.optim.AdamW(model.parameters(),lr=job['lr'],weight_decay=1e-4)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata or state['epoch']!=len(state['history']) or state['epoch']>job['epochs']:
            raise ValueError('Resume checkpoint/cache identity mismatch')
        verify_history(state['history'],job,len(raw),meta['batch'])
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        replay,_=pr.run_epoch(model,data['val'],meta['batch'],device)
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k],expected[k],atol=1e-6,rtol=2e-5) for k in replay):raise ValueError('Probe resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'],matched=True),path/'resume_validation.json')
    else:
        val,_=pr.run_epoch(model,data['val'],meta['batch'],device)
        state=dict(metadata=metadata,epoch=0,history=[],initial_validation=val,best_epoch=0,best_validation=val,best_model=ab.cpu_state(model))
        state.update(model=ab.cpu_state(model),optimizer=opt.state_dict(),rng=rng_state());publish(state,path)
    for epoch in range(state['epoch']+1,job['epochs']+1):
        started=time.monotonic();lr=ar.learning_rate(epoch,job['epochs'],job['lr'])
        for group in opt.param_groups:group['lr']=lr
        train,updates=pr.run_epoch(model,data['train'],meta['batch'],device,opt,job['seed']+epoch*1009)
        if updates!=(len(raw)+meta['batch']-1)//meta['batch']:raise ValueError('Probe update budget changed')
        val,_=pr.run_epoch(model,data['val'],meta['batch'],device)
        if val['primary']<state['best_validation']['primary']:state.update(best_epoch=epoch,best_validation=val,best_model=ab.cpu_state(model))
        state['history'].append(dict(epoch=epoch,lr=lr,train=train,validation=val,optimizer_steps=updates,windows=len(raw),seconds=time.monotonic()-started))
        state.update(epoch=epoch,model=ab.cpu_state(model),optimizer=opt.state_dict(),rng=rng_state());publish(state,path)
        if epoch%10==0 or epoch==1:progress(f'{name}: epoch={epoch}/{job["epochs"]} val={val["primary"]:.5f} best={state["best_epoch"]}')
    atomic_json(dict(selected_epoch=state['best_epoch'],validation=state['best_validation'],allocated_epochs=job['epochs'],
        optimizer_steps=sum(r['optimizer_steps'] for r in state['history']),parameters=sum(p.numel() for p in model.parameters()),encoder_updates=0),path/'training_summary.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


def verify_history(history,job,n,batch):
    if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>job['epochs']:raise ValueError('Invalid probe history')
    for r in history:
        if r['windows']!=n or r['optimizer_steps']!=(n+batch-1)//batch or r['lr']!=ar.learning_rate(r['epoch'],job['epochs'],job['lr']):
            raise ValueError('Probe budget/LR differs from protocol')


def lock_selection(meta,out):
    rows={};weights={}
    n=read_json(out/'cache/train_index.json')['windows']
    for job in meta['experiments']:
        path=out/job['name'];done=read_json(path/'completion.json')
        expected=dict(manifest=meta,job=job,cache_sha256=cache_identity(out))
        if done['status']!='complete' or done['metadata']!=expected:raise ValueError('Incomplete or changed probe')
        verify_files(path,done['files'])
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=True);best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        history=read_json(path/'history.json');verify_history(history,job,n,meta['batch'])
        if last['metadata']!=expected or best['metadata']!=expected or last['epoch']!=job['epochs'] or last['history']!=history:raise ValueError('Incomplete probe checkpoint budget')
        epoch,val=min([(0,last['initial_validation'])]+[(r['epoch'],r['validation']) for r in history],key=lambda r:r[1]['primary'])
        summary=read_json(path/'training_summary.json');ridge=read_json(path/'ridge.json')
        if (best['epoch']!=epoch or summary['selected_epoch']!=epoch or best['validation']!=val or summary['validation']!=val
                or last['best_epoch']!=epoch or last['best_validation']!=val
                or summary['optimizer_steps']!=sum(r['optimizer_steps'] for r in history)):
            raise ValueError('Probe selection mismatch')
        if (ridge['metadata']!=expected or [r['alpha'] for r in ridge['candidates']]!=meta['alphas']
                or ridge['selected']!=min(ridge['candidates'],key=lambda r:r['validation']['primary'])
                or ridge['weights_sha256']!=sha256(path/'ridge.npz')):
            raise ValueError('Ridge selection mismatch')
        rows[job['name']]=dict(mlp=summary,ridge=ridge['selected']);weights[job['name']]={k:sha256(path/k) for k in ('best.pt','last.pt','ridge.npz','feature_scales.json')}
    for seed in (42,43):
        for p in pr.PREFIXES:
            a=read_json(out/f'plain_s{seed}_p{p}'/'head_initialization.json')
            b=read_json(out/f'sampled_s{seed}_p{p}'/'head_initialization.json')
            if a!=b:raise ValueError('Paired plain/sampled heads did not share initialization')
    result=dict(manifest=meta,selections=rows,weights=weights)
    if (out/'selection_lock.json').exists() and read_json(out/'selection_lock.json')!=result:raise ValueError('Locked probe selections changed')
    atomic_json(result,out/'selection_lock.json');return result


def predict_probe(meta,out,job,split,kind,prefix=None,device='cuda'):
    path=out/job['name'];scales=read_json(path/'feature_scales.json')
    data=load_data(out,split,dict(job,prefix=prefix or job['prefix']),scales)
    if kind=='ridge':
        with np.load(path/'ridge.npz',allow_pickle=False) as fit:pred=pr.ridge_predict(fit,data['x'])
    else:
        model=pr.Probe(job['width'],job['seed'],meta['hidden']).to(device)
        ck=torch.load(path/('last.pt' if kind=='mlp_last' else 'best.pt'),map_location='cpu',weights_only=True)
        model.load_state_dict(ck['model']);pred=pr.predictions(model,data['x'],meta['batch'],device)
    return pred,data


def evaluate(meta,out,device='cuda'):
    lock=lock_selection(meta,out);stats=read_json(out/'cache/target_scales.json')
    # Replay every selected validation before opening research caches.
    for job in meta['experiments']:
        for kind in ('ridge','mlp','mlp_last'):
            pred,data= predict_probe(meta,out,job,'val',kind,device=device)
            actual=pr.measure(pred,data['y'],data['mask'],stats)[0]['metrics']
            expected=(read_json(out/job['name']/'history.json')[-1]['validation'] if kind=='mlp_last'
                      else lock['selections'][job['name']][kind]['validation'])
            if any(not np.isclose(actual[k],v,atol=1e-6,rtol=2e-5) for k,v in expected.items()):raise ValueError('Selected probe validation did not reproduce')
    report=dict(schema=SCHEMA,goal=GOAL,selections=lock['selections'],datasets={},limits=meta['limits'])
    for split in ('test','cross_research'):
        prepare_split(meta,out,split,device)
        inv=read_json(Path(upstream(meta)['source'])/f'cache/{split}_inventory.json');atomic_json(inv,out/f'{split}_inventory.json')
        scores={};errors={}
        def record(label,pred,data):scores[label],errors[label]=pr.measure(pred,data['y'],data['mask'],stats)
        for job in meta['experiments']:
            for kind in ('ridge','mlp','mlp_last'):
                pred,data=predict_probe(meta,out,job,split,kind,device=device);record(job['name']+'/'+kind,pred,data)
                if job['prefix']==128 and kind!='mlp_last':
                    for p in pr.PREFIXES[:-1]:
                        pred,data=predict_probe(meta,out,job,split,kind,p,device)
                        record(f'{job["variant"]}_p{p}/transfer128_{kind}',pred,data)
        for p in pr.PREFIXES:
            mean=np.load(out/f'cache/mean_p{p}.npy')
            data={k:np.load(out/f'cache/{split}_p{p}_{k}.npy') for k in ('y','mask')}
            record(f'mean_p{p}',np.broadcast_to(mean,data['y'].shape).astype(np.float32),data)
        pairs=[]
        for p in pr.PREFIXES:
            for kind in ('ridge','mlp','mlp_last'):
                for seed in (42,43):pairs.append((f'sampled_s{seed}_p{p}/{kind}',f'plain_s{seed}_p{p}/{kind}'))
                for name in meta['identity']['variants']:pairs.append((f'{name}_p{p}/{kind}',f'current28_p{p}/{kind}'))
            if p!=128:
                for kind in ('ridge','mlp'):
                    for name in meta['identity']['variants']:pairs.append((f'{name}_p{p}/transfer128_{kind}',f'{name}_p{p}/{kind}'))
        paired={a+'_minus_'+b:{k:ab.pair_groups(errors[a][k],errors[b][k],inv) for k in pr.METRICS} for a,b in pairs}
        report['datasets'][split]=dict(scores=scores,paired=paired)
        atomic_json({n:{k:v.tolist() for k,v in rows.items()} for n,rows in errors.items()},out/f'{split}_errors.json')
        progress(f'Prefix readouts evaluated: {split}; all four positions and both seeds retained')
    atomic_json(report,out/'prefix_readout_metrics.json')
    lines=['# Frozen bar-state readout','','Past16 observed bars, no encoder updates. Position-specific and transferred heads are distinct diagnostics.','']
    for split,rows in report['datasets'].items():
        lines += [f'## {split}','','| probe | primary | path | body | activity | change1 | close bp |','|---|---:|---:|---:|---:|---:|---:|']
        for name,row in rows['scores'].items():lines.append('| '+name+' | '+' | '.join(f'{row["metrics"][k]:.5f}' for k in pr.METRICS)+' |')
    (out/'summary.md').write_text('\n'.join(lines));return report


def run_jobs(out,jobs):
    pending=list(read_json(out/'manifest.json')['experiments']);active={};seen={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);path=out/job['name'];path.mkdir(exist_ok=True);log=(path/'run.log').open('a')
                try:proc=subprocess.Popen([sys.executable,'-m','obson.babel.prefix_readout_benchmark','worker','--out',str(out),'--name',job['name']],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[job['name']]=(proc,log);progress(f'Started {job["name"]}, pid={proc.pid}')
            for name,(proc,log) in list(active.items()):
                path=out/name/'progress.json'
                if path.exists():
                    try:row=read_json(path)
                    except (ValueError,OSError):row=None
                    if row and seen.get(name)!=row:progress(f'{name}: {row}');seen[name]=row
                code=proc.poll()
                if code is not None:
                    log.close();del active[name]
                    if code:raise RuntimeError(f'{name} exited{code}; see worker run.log')
            if active:time.sleep(1)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        signal.signal(signal.SIGTERM,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('all','preflight','evaluate','worker'))
    parser.add_argument('--source');parser.add_argument('--out',required=True);parser.add_argument('--name');parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args();out=Path(args.out).resolve();ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal encoder extraction and head training run on AutoDL CUDA')
    if args.action=='worker':worker(out,args.name);return
    if not args.source:parser.error('--source required')
    if not 1<=args.jobs<=2:raise ValueError('Use one or two concurrent probe workers')
    source=Path(args.source).resolve();progress('Verifying immutable sampling/plain sources')
    identity=source_identity(source);sm=identity['manifest']
    if any(v['job']['epochs']!=200 for v in identity['variants'].values()):raise ValueError('Both encoder pairs must have200 allocated epochs')
    if sm['windows_per_epoch']!=4789 or sm['config']['latent']!=512 or sm['batch']!=128 or sm['micro']!=64:
        raise ValueError('Expected original fixed-budget512 encoder experiment')
    for dep in [source,Path(sm['source']),Path(sm['teacher_source'])]:
        if out==dep or out in dep.parents or dep in out.parents:raise ValueError('Separate output required')
    runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:raise ValueError('Restore original Torch/NumPy environment')
    meta=make_manifest(source,identity)
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Probe configuration changed')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output without matching manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(),jobs=args.jobs,source_runtime=runtime,started_unix=time.time()),out/'runtime.json')
    for split in ('train','val'):prepare_split(meta,out,split)
    if read_json(out/'cache/train_index.json')['windows']!=4789 or read_json(out/'cache/val_index.json')['windows']!=978:
        raise ValueError('Original fixed probe window counts changed')
    atomic_json(dict(status='passed',encoder_updates=0,train_windows=read_json(out/'cache/train_index.json')['windows'],
        val_windows=read_json(out/'cache/val_index.json')['windows'],strict_prefix_inputs=True,source_validation_replayed=True),out/'preflight.json')
    if args.action=='preflight':return
    if args.action=='all':run_jobs(out,args.jobs)
    evaluate(meta,out)
    if source_identity(source)!=identity:raise ValueError('Frozen sources changed during experiment')
    for split in ('train','val','test','cross_research'):verify_cache(meta,out,split)
    files={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.md') and p.name!='completion.json'}
    files.update({f'cache/{s}_index.json':sha256(out/f'cache/{s}_index.json') for s in ('train','val','test','cross_research')})
    updates=sum(read_json(out/j['name']/'training_summary.json')['optimizer_steps'] for j in meta['experiments'])
    atomic_json(dict(status='complete',cells=len(meta['experiments']),mlp_optimizer_steps=updates,
        encoder_updates=0,source_unchanged=True,automatic_promotion=False,files=files),out/'completion.json')
    progress('Frozen bar-state diagnosis complete; encoder weights unchanged')


if __name__=='__main__':
    import fcntl
    if '--out' not in sys.argv:main()
    else:
        root=Path(sys.argv[sys.argv.index('--out')+1]).resolve();root.parent.mkdir(parents=True,exist_ok=True)
        name=sys.argv[sys.argv.index('--name')+1] if len(sys.argv)>1 and sys.argv[1]=='worker' and '--name' in sys.argv else 'controller'
        with (root.parent/('.'+root.name+'.'+name+'.lock')).open('a') as lockfile:
            try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit('Output/job already running')
            main()
