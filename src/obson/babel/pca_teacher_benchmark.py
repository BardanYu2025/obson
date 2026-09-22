"""PCA-guided causal history compression, with controlled continuation branches."""
import argparse
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import architecture as ar, architecture_benchmark as ab, capacity_benchmark as cb, pca_teacher as pt
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-pca-teacher-v1'
GOAL = dict(stage='Learn causal historical compression using train-fitted PCA as a teacher; separate coordinate guidance, residual readout, and teacher release.',
            automatic_promotion=False, independent_holdout=False,
            context='Same reset128-bar windows; endpoint supervision only. No teacher coordinates or true targets supplied at student inference.')
METRICS = cb.METRICS


def code_identity():
    return cb.code_identity() | {Path(pt.__file__).name: sha256(pt.__file__), Path(__file__).name: sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256'] != code_identity():
        raise ValueError('Teacher experiment code changed; use a new run')


def make_manifest(source, identity, parent_epochs=200, branch_epochs=100, batch=128, micro=64):
    if min(parent_epochs, branch_epochs, batch, micro)<1 or micro>batch or batch%micro:
        raise ValueError('Positive budgets; micro must divide batch')
    jobs = []
    for seed in (42, 43):
        for mode, weight in [('plain', 0.), ('teacher', .25)]:
            jobs.append(dict(name=f'{mode}_s{seed}', phase='base', seed=seed, parent=None,
                             epochs=parent_epochs, lr=3e-4, residual=False, teacher_weight=weight, schedule='constant'))
    for seed in (42, 43):
        for mode, weight in [('plain', 0.), ('teacher', .25)]:
            for branch in (('fixed', 'residual') if mode=='plain' else ('fixed', 'residual', 'release')):
                jobs.append(dict(name=f'{mode}_{branch}_s{seed}', phase='branch', seed=seed,
                    parent=f'{mode}_s{seed}', epochs=branch_epochs, lr=1e-4, residual=branch!='fixed',
                    teacher_weight=weight, schedule='release' if branch=='release' else 'constant'))
    return json.loads(json.dumps(dict(schema=SCHEMA, source=str(source.resolve()), source_identity=identity,
        config=pt.CONFIG, seeds=[42,43], experiments=jobs, batch=batch, micro=micro,
        parent_epochs=parent_epochs, branch_epochs=branch_epochs, release_epochs=max(2,min(50,branch_epochs)),
        precision='fp32_tf32_disabled', code_sha256=code_identity(), goal=GOAL,
        objective=dict(weights=ar.WEIGHTS, horizons=ar.HORIZONS, train='Original SmoothL1 plus lambda*mean SmoothL1 of train-standardized PCA coordinates',
            validation_selection='Original normalized MSE only; alignment loss never selects checkpoints', teacher_strength=.25,
            normalization='Coordinate mean/std from training endpoints only; std floor1e-4'),
        budget='Four200-epoch parents, ten100-epoch continuations. Same budget within each phase; not a direct comparison to old200-epoch architectures.',
        lineage='Branches start from their parent validation-selected best, including identical zero-initialized residual weights. Optimizer reset equally for all siblings; inherited selected epoch recorded.',
        selection='Epoch0 eligible. All parent choices locked before branching; all14 choices locked before neural research evaluation.',
        limitations='Fixed PCA inverse is a strong analytical prior. Do not attribute a comparison with old learned decoders solely to teacher guidance. No guarantee100 continuation epochs converge.')))


def verify_cache(meta, out):
    cb.verify_cache(meta)
    index = read_json(out/'cache/index.json')
    if index['manifest'] != meta: raise ValueError('Teacher cache metadata changed')
    verify_files(out/'cache', index['files'])
    cb.source_pca(meta)


def prepare(meta, out):
    verify_code(meta); cb.verify_cache(meta)
    source = Path(meta['source']); pca = cb.source_pca(meta); cache = out/'cache'; cache.mkdir(exist_ok=True)
    if (cache/'index.json').exists():
        verify_cache(meta, out); return
    files = {}
    # Only training/validation data may enter preparation, including scale fitting.
    for split in ('train', 'val'):
        data = ab.load_arrays(source, split); c = pt.coefficients(pca, data)
        if split=='train':
            scales = pt.fit_coordinate_scales(c)
            atomic_json(scales, cache/'coordinate_scales.json'); files['coordinate_scales.json'] = sha256(cache/'coordinate_scales.json')
        q = pt.normalize_coordinates(c, scales)
        np.save(cache/f'{split}_q.npy', q, allow_pickle=False); files[f'{split}_q.npy'] = sha256(cache/f'{split}_q.npy')
    atomic_json(read_json(source/'cache/statistics.json'), out/'statistics.json')
    atomic_json(dict(teacher_fitted_on='original_train_only', coordinate_scales_fitted_on='original_train_only',
        endpoint_context='All targets are observed history available at endpoint t. No full-window reconstruction is inserted into earlier bar inputs.',
        current_bar_scored=False, targets=ar.NAMES,
        coverage={s: len(ab.load_arrays(source,s)['x']) for s in ('train','val')}), out/'data_audit.json')
    atomic_json(dict(manifest=meta, files=files, pca_sha256=sha256(source/'pca.npz')), cache/'index.json')


def load_data(meta, out, split):
    data = ab.load_arrays(Path(meta['source']), split)
    return dict(data, q=np.load(out/f'cache/{split}_q.npy', mmap_mode='r', allow_pickle=False))


def model_for(meta, out, job, device):
    return pt.Student(meta['config'], job['seed'], cb.source_pca(meta),
        read_json(out/'cache/coordinate_scales.json'), job['residual']).to(device)


def run_epoch(model, data, stats, batch, micro, device, weight, opt=None, order_seed=0):
    training = opt is not None; model.train(training); n=len(data['x'])
    order=np.random.default_rng(order_seed).permutation(n) if training else np.arange(n)
    totals={}; updates=0
    with torch.set_grad_enabled(training):
        for left in range(0,n,batch):
            ids=order[left:left+batch]
            if training: opt.zero_grad(set_to_none=True)
            for start in range(0,len(ids),micro):
                use=ids[start:start+micro]
                b={k:torch.tensor(np.asarray(v[use]),device=device) for k,v in data.items()}
                z=model.encoder(b['x'])[:,-1]; pred=model.decoder(z)
                rows=ar.error_rows(pred,b['y'],b['mask'],stats)
                smooth=ar.error_rows(pred,b['y'],b['mask'],stats,True)['primary']
                aligned=pt.alignment_rows(z,b['q'])
                loss=smooth+weight*aligned['coordinate_smooth']
                rows=dict(rows,**aligned,smooth_objective=smooth,total_objective=loss)
                if any(not torch.isfinite(v).all() for v in rows.values()): raise ValueError('Nonfinite student objective')
                if training:(loss.sum()/len(ids)).backward()
                for k,v in rows.items():totals[k]=totals.get(k,0.)+float(v.detach().sum())
            if training:
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1
    return {k:v/n for k,v in totals.items()},updates


def parent_lineage(meta, out, job):
    if not job['parent']: return None
    parent = next(j for j in meta['experiments'] if j['name']==job['parent'])
    lock=read_json(out/'parent_selection_lock.json')
    if lock['manifest']!=meta: raise ValueError('Parent selection manifest changed')
    path=out/parent['name'];done=read_json(path/'completion.json')
    if done['status']!='complete' or done['metadata']['job']!=parent or done['metadata']['manifest']!=meta: raise ValueError('Incomplete parent')
    verify_files(path,done['files'])
    expected=lock['weights'][parent['name']]['best']
    if sha256(path/'best.pt')!=expected:raise ValueError('Parent selected checkpoint changed')
    return dict(parent=parent['name'], best_sha256=expected, selected_epoch=lock['all_trials'][parent['name']]['selected_epoch'],
                allocated_epochs=parent['epochs'], optimizer='reset equally for sibling branches')


def publish(state, path, improved=True):
    atomic_save(state,path/'last.pt')
    if improved or not (path/'best.pt').exists():
        atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],validation=state['best_validation'],model=state['best_model']),path/'best.pt')
    atomic_json(state['history'],path/'history.json')
    row=state['history'][-1] if state['history'] else {}
    atomic_json(dict(epoch=state['epoch'],epochs=state['metadata']['job']['epochs'],best_epoch=state['best_epoch'],
        train=row.get('train',{}).get('primary'),validation=row.get('validation',{}).get('primary'),
        teacher_weight=row.get('teacher_weight'),seconds=row.get('seconds'),remaining_minutes=row.get('remaining_minutes')),path/'progress.json')


def worker(out, name, device='cuda'):
    meta=read_json(out/'manifest.json');verify_code(meta)
    job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True)
    lineage=parent_lineage(meta,out,job);metadata=dict(manifest=meta,job=job,lineage=lineage)
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['status']!='complete' or done['metadata']!=metadata:raise ValueError('Completed worker identity mismatch')
        verify_files(path,done['files']);return
    verify_cache(meta,out)
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
    model=model_for(meta,out,job,device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=job['lr'],weight_decay=.01)
    stats=read_json(Path(meta['source'])/'cache/statistics.json');data={s:load_data(meta,out,s) for s in ('train','val')}
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata or state['epoch']>job['epochs'] or [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)):
            raise ValueError('Resume metadata/history mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        weight=pt.teacher_weight(job,state['epoch'],meta['release_epochs'])
        replay,_=run_epoch(model,data['val'],stats,meta['batch'],meta['micro'],device,weight)
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k],expected[k],atol=1e-6,rtol=2e-5) for k in replay):raise ValueError('Resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'],max_abs=max(abs(replay[k]-expected[k]) for k in replay)),path/'resume_validation.json')
    else:
        if lineage:
            parent=torch.load(out/job['parent']/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(parent['model'])
        weight=pt.teacher_weight(job,0,meta['release_epochs'])
        initial,_=run_epoch(model,data['val'],stats,meta['batch'],meta['micro'],device,weight)
        if lineage:
            # Teacher release begins at the same weight as its parent, residual starts at exactly zero.
            if any(not np.isclose(initial[k],parent['validation'][k],atol=1e-6,rtol=2e-5) for k in initial):
                raise ValueError('Branch initial validation differs from parent')
            atomic_json(dict(lineage=lineage,validation=initial,matched=True),path/'branch_initialization.json')
        state=dict(metadata=metadata,epoch=0,history=[],initial_validation=initial,best_epoch=0,best_validation=initial,best_model=ab.cpu_state(model))
        state.update(model=ab.cpu_state(model),optimizer=opt.state_dict(),rng=rng_state());publish(state,path)
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1,job['epochs']+1):
        started=time.monotonic();lr=ar.learning_rate(epoch,job['epochs'],job['lr']);weight=pt.teacher_weight(job,epoch,meta['release_epochs'])
        for g in opt.param_groups:g['lr']=lr
        train,updates=run_epoch(model,data['train'],stats,meta['batch'],meta['micro'],device,weight,opt,job['seed']+epoch*1009)
        val,_=run_epoch(model,data['val'],stats,meta['batch'],meta['micro'],device,weight)
        improved=val['primary']<state['best_validation']['primary']
        if improved:state.update(best_epoch=epoch,best_validation=val,best_model=ab.cpu_state(model))
        elapsed=time.monotonic()-started
        state['history'].append(dict(epoch=epoch,train=train,validation=val,lr=lr,teacher_weight=weight,optimizer_steps=updates,
            windows=len(data['train']['x']),seconds=elapsed,remaining_minutes=elapsed*(job['epochs']-epoch)/60))
        state.update(epoch=epoch,model=ab.cpu_state(model),optimizer=opt.state_dict(),rng=rng_state());publish(state,path,improved)
        progress(f'{name} epoch={epoch}/{job["epochs"]} val={val["primary"]:.5f} coordinate_mse={val["coordinate_mse"]:.5f} best={state["best_epoch"]} seconds={elapsed:.1f}')
    publish(state,path)
    atomic_json(dict(selected_epoch=state['best_epoch'],validation=state['best_validation'],lineage=lineage,
        trained_selection=state['best_epoch']>0 or (lineage is not None and lineage['selected_epoch']>0),
        selected_postbranch_update=state['best_epoch']>0 if lineage else None,selected_at_budget_end=state['best_epoch']==job['epochs'],
        parameters=dict(encoder=sum(p.numel() for p in model.encoder.parameters()),decoder=sum(p.numel() for p in model.decoder.parameters()),
                        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
        training_seconds=sum(r['seconds'] for r in state['history']),optimizer_steps=sum(r['optimizer_steps'] for r in state['history']),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None),path/'training_summary.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


def lock_selection(meta,out,parents_only=False):
    jobs=[j for j in meta['experiments'] if not parents_only or j['phase']=='base'];rows={};weights={}
    for job in jobs:
        path=out/job['name'];done=read_json(path/'completion.json')
        lineage=parent_lineage(meta,out,job)
        metadata=dict(manifest=meta,job=job,lineage=lineage)
        if done['status']!='complete' or done['metadata']!=metadata:raise ValueError('Incomplete or mismatched trial')
        verify_files(path,done['files'])
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        if last['metadata']!=metadata or best['metadata']!=metadata or last['epoch']!=job['epochs'] or [r['epoch'] for r in last['history']]!=list(range(1,job['epochs']+1)):
            raise ValueError('Trial budget/history/checkpoint mismatch')
        choices=[(0,last['initial_validation'])]+[(r['epoch'],r['validation']) for r in last['history']]
        epoch,val=min(choices,key=lambda r:r[1]['primary'])
        if best['epoch']!=epoch or last['best_epoch']!=epoch or best['validation']!=val:
            raise ValueError('Selection is not the first validation-primary minimum')
        summary=read_json(path/'training_summary.json')
        if summary['selected_epoch']!=epoch or summary['validation']!=val or summary['lineage']!=lineage:
            raise ValueError('Selected summary mismatch')
        rows[job['name']]=summary
        weights[job['name']]={k:sha256(path/f'{k}.pt') for k in ('best','last')}
    lock=dict(manifest=meta,all_trials=rows,weights=weights)
    name='parent_selection_lock.json' if parents_only else 'selection_lock.json'
    if (out/name).exists() and read_json(out/name)!=lock:raise ValueError('Locked selection changed')
    atomic_json(lock,out/name);return lock


@torch.no_grad()
def preflight(meta,out,device='cuda'):
    results={};pca=cb.source_pca(meta);scales=read_json(out/'cache/coordinate_scales.json')
    for residual in (False,True):
        model=pt.Student(meta['config'],999,pca,scales,residual).to(device)
        x=torch.randn(meta['micro'],128,28,device=device)
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        with torch.enable_grad():
            z=model.encoder(x)[:,-1];pred=model.decoder(z)
            target=torch.zeros_like(pred);mask=torch.ones_like(pred,dtype=torch.bool);mask[:,-1]=False
            loss=ar.error_rows(pred,target,mask,dict(y_scale=[1.]*7,delta_scale=[1.]*4),True)['primary'].mean()
            loss=(loss+.25*pt.alignment_rows(z,torch.zeros_like(z))['coordinate_smooth'].mean());loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite synthetic gradient')
        model.eval();original=model.encoder(x[:2]);changed=x[:2].clone();changed[:,65:]+=5
        delta=float((original[:,:65]-model.encoder(changed)[:,:65]).abs().max())
        if delta>2e-5:raise ValueError('Student prefix causality failed')
        results[str(residual)]=dict(causal_prefix_maxdiff=delta,output_shape=list(pred.shape),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
        del model,x,z,pred,loss,original,changed
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    # Known coordinates must reproduce the original PCA reconstruction before any training.
    data=load_data(meta,out,'val');decoder=pt.PCADecoder(meta['config'],pca,scales).to(device)
    actual=np.concatenate([decoder.base(torch.tensor(np.asarray(data['q'][i:i+meta['micro']]),device=device)).cpu().numpy()
                           for i in range(0,len(data['q']),meta['micro'])])
    expected=ab.pca_predict(pca,data);diff=actual-expected
    passed=bool(np.allclose(actual,expected,atol=5e-5,rtol=2e-5))
    atomic_json(dict(results=results,optimizer_updates=0,pca_inverse_replay=dict(passed=passed,max_abs=float(np.abs(diff).max()),
        rms=float(np.sqrt(np.mean(diff**2))),atol=5e-5,rtol=2e-5),scope='Synthetic forward/backward and entire validation PCA-inverse replay; no parameter updates.'),out/'preflight.json')
    if not passed:raise ValueError('PCA inverse validation replay failed; see preflight.json')
    progress('Causal student and PCA inverse preflight passed; no optimizer updates')


def run_phase(out,jobs,phase):
    """Parent phase must finish/lock before any continuation process is launched."""
    pending=[j for j in read_json(out/'manifest.json')['experiments'] if j['phase']==phase];active={};previous={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);name=job['name'];(out/name).mkdir(exist_ok=True);log=(out/name/'run.log').open('a')
                try:proc=subprocess.Popen([sys.executable,'-m','obson.babel.pca_teacher_benchmark','worker','--out',str(out),'--name',name],stdout=log,stderr=subprocess.STDOUT)
                except BaseException:log.close();raise
                active[name]=(proc,log);progress(f'Started {phase} {name} pid={proc.pid}')
            for name,(proc,log) in list(active.items()):
                path=out/name/'progress.json';code=proc.poll()
                if path.exists():
                    try:status=read_json(path)
                    except (OSError,ValueError):status=None
                    if status and status!=previous.get(name):progress(f'{name}: {status}');previous[name]=status
                if code is not None:
                    log.close();del active[name]
                    if code:raise RuntimeError(f'{name} exited{code}; see {out/name/"run.log"}')
                    progress(f'Completed {name}')
            if active:time.sleep(1)
    finally:
        for proc,log in active.values():
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        signal.signal(signal.SIGTERM,handler)


def coordinate_metrics(z,q):
    squared=(z.astype(float)-q.astype(float))**2;width=z.shape[1]
    cuts=sorted(set([0,min(32,width),min(128,width),width]))
    return dict(mse=float(squared.mean()),bands={f'{a+1}-{b}':float(squared[:,a:b].mean()) for a,b in zip(cuts[:-1],cuts[1:])},
                interpretation='MSE of PCA coordinates standardized using training endpoints; coordinate accuracy is auxiliary, not checkpoint selection.')


def contrasts(seeds,width=512):
    pairs=[]
    for s in seeds:
        pairs += [(f'teacher_s{s}',f'plain_s{s}'),
                  (f'plain_residual_s{s}',f'plain_fixed_s{s}'),
                  (f'teacher_residual_s{s}',f'teacher_fixed_s{s}'),
                  (f'teacher_release_s{s}',f'teacher_residual_s{s}'),
                  (f'teacher_residual_s{s}',f'plain_residual_s{s}'),
                  (f'teacher_release_s{s}',f'plain_residual_s{s}')]
        for mode in ('plain_fixed','plain_residual','teacher_fixed','teacher_residual','teacher_release'):
            pairs.append((f'{mode}_s{s}',f'{mode.split("_")[0]}_s{s}'))
        for mode in ('plain','teacher','plain_residual','teacher_residual','teacher_release'):
            pairs.append((f'{mode}_s{s}',f'pca{width}'))
    return pairs


def save_examples(out,cards):
    import html
    colors={'truth':'#222','pca512':'#842db1','plain_s42':'#bd8047','teacher_s42':'#3479c1',
            'plain_fixed_s42':'#957641','plain_residual_s42':'#d4543c',
            'teacher_fixed_s42':'#499aa6','teacher_residual_s42':'#26a076','teacher_release_s42':'#253c88'}
    lines=['<!doctype html><meta charset="utf-8"><title>PCA teacher reconstruction</title>',
        '<style>body{font:14px sans-serif;margin:24px}svg{border:1px solid #ddd;max-width:100%}</style>',
        '<h1>Observed history reconstruction, not forecasts</h1><p>Fixed evenly spaced examples; seed42 displayed, both seeds in metrics. All student curves use predicted state only.</p>']
    for name,color in colors.items():
        lines.append(f'<label style="color:{color}"><input type="checkbox" checked onchange="document.querySelectorAll(\'[data-series={name}]\').forEach(p=>p.style.display=this.checked?\'\':\'none\')">{name}</label> ')
    for row in cards:
        lines.append('<h3>'+html.escape(row['dataset']+' '+row['key']+' '+row['end'])+'</h3>')
        for channel,label in [(0,'Close, log-percent from pre-window anchor'),(2,'Volume relative to prior EMA'),(4,'Observed OI change, asinh')]:
            valid=np.array(row['mask'])[:127,channel]
            values=np.concatenate([np.array(v)[:127,channel][valid] for k,v in row['series'].items() if k in colors])
            lo=float(values.min()) if len(values) else 0.;hi=float(values.max()) if len(values) else 1.;span=max(hi-lo,1e-5)
            lines.append(f'<p>{label} [{lo:.3g},{hi:.3g}]</p><svg viewBox="0 0 900 180" width="900" height="180">')
            for name,color in colors.items():
                if name not in row['series']:continue
                v=np.array(row['series'][name])[:127,channel];parts=[]
                for i in range(len(v)):
                    if valid[i]:parts.append(('M' if i==0 or not valid[i-1] else 'L')+f'{10+i*880/126:.2f},{170-(v[i]-lo)/span*160:.2f}')
                lines.append(f'<path data-series="{name}" d="{" ".join(parts)}" fill="none" stroke="{color}" stroke-width="1.2"/>')
            lines.append('</svg>')
    (out/'examples.html').write_text('\n'.join(lines))


def evaluate(meta,out,device='cuda'):
    verify_code(meta);verify_cache(meta,out);lock=lock_selection(meta,out)
    source=Path(meta['source']);stats=read_json(source/'cache/statistics.json');pca=cb.source_pca(meta)
    scales=read_json(out/'cache/coordinate_scales.json');val=load_data(meta,out,'val')
    report=dict(schema=SCHEMA,goal=GOAL,trials=lock['all_trials'],datasets={},
        limitations='Parents and branches have different allocated budgets. Use same-phase and sibling contrasts for attribution. Parent selected epochs may differ. Research sets are already opened.')
    cards=[]
    for split in ('test','cross_research'):
        data=ab.load_arrays(source,split);q=pt.normalize_coordinates(pt.coefficients(pca,data),scales)
        inv=read_json(source/f'cache/{split}_inventory.json');atomic_json(inv,out/f'{split}_inventory.json')
        ids=list(np.linspace(0,len(data['y'])-1,min(6,len(data['y'])),dtype=int))
        examples={i:dict(dataset=split,key=inv[i]['key'],end=inv[i]['end'],mask=data['mask'][i].tolist(),
            series={'truth':(data['y'][i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()}) for i in ids}
        scores={};errors={};last_scores={}
        def record(name,pred):
            scores[name],errors[name]=ab.measure(pred,data,stats)
            for i in ids:examples[i]['series'][name]=(pred[i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()
        for name,pred in ab.baseline_predictions(data).items():record(name,pred)
        for rank in (meta['config']['latent']//2,meta['config']['latent']):record(f'pca{rank}',ab.pca_predict(cb.prefix_pca(pca,rank),data))
        for job in meta['experiments']:
            model=model_for(meta,out,job,device);path=out/job['name']
            ck=torch.load(path/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
            weight=pt.teacher_weight(job,ck['epoch'],meta['release_epochs'])
            replay,_=run_epoch(model,val,stats,meta['batch'],meta['micro'],device,weight)
            if any(not np.isclose(replay[k],ck['validation'][k],atol=1e-6,rtol=2e-5) for k in replay):raise ValueError('Selected validation did not reproduce')
            pred,z=ab.predictions(model,data['x'],meta['micro'],device);record(job['name'],pred)
            row=scores[job['name']];row['coordinate_alignment']=coordinate_metrics(z,q)
            if job['residual']:
                with torch.no_grad():
                    base=np.concatenate([model.decoder.base(torch.tensor(z[i:i+meta['micro']],device=device)).cpu().numpy() for i in range(0,len(z),meta['micro'])])
                    oracle=np.concatenate([model.decoder(torch.tensor(q[i:i+meta['micro']],device=device)).cpu().numpy() for i in range(0,len(q),meta['micro'])])
                row['residual_disabled']=ab.measure(base,data,stats)[0]
                row['true_pca_coordinate_diagnostic']=ab.measure(oracle,data,stats)[0]
                row['diagnostic_scope']='Teacher-coordinate input is a diagnostic only; main predictions never use it. Residual-off measures current latent, not the original parent.'
            last=torch.load(path/'last.pt',map_location='cpu',weights_only=True);model.load_state_dict(last['model'])
            pred,_=ab.predictions(model,data['x'],meta['micro'],device);last_scores[job['name']]=ab.measure(pred,data,stats)[0]
            del model,ck,last,pred,z
            progress(f'Evaluated {split} {job["name"]}, selected and fixed last')
        paired={a+'_minus_'+b:{k:ab.pair_groups(errors[a][k],errors[b][k],inv) for k in METRICS} for a,b in contrasts(meta['seeds'],meta['config']['latent'])}
        report['datasets'][split]=dict(variants=scores,fixed_last=last_scores,paired=paired)
        atomic_json({n:{k:v.tolist() for k,v in row.items()} for n,row in errors.items()},out/f'{split}_per_window_errors.json')
        cards.extend(examples.values())
    report['exploratory_screen']={}
    for a,b in [('teacher','plain'),('plain_residual','plain_fixed'),('teacher_residual','teacher_fixed'),('teacher_release','teacher_residual')]:
        checks=[];regressions=[]
        for split,rows in report['datasets'].items():
            for seed in meta['seeds']:
                name=f'{a}_s{seed}';other=f'{b}_s{seed}';group=rows['paired'][name+'_minus_'+other]
                ci=group['primary']['all'];trained=all(lock['all_trials'][n]['trained_selection'] for n in (name,other))
                checks.append(trained and ci['supported'] and ci['high'] is not None and ci['high']<0)
                for family in ('path','changes','body','activity'):
                    ci=group[family]['all']
                    if ci['supported'] and ci['low'] is not None and ci['low']>0:regressions.append(dict(dataset=split,seed=seed,family=family,interval=ci))
        report['exploratory_screen'][a+'_minus_'+b]=dict(primary_improvement_all_seeds_and_datasets=all(checks),family_regressions=regressions,
            interpretation='Exploratory uncorrected weekly intervals. No regression detected is not equivalence. No automatic promotion.')
    atomic_json(report,out/'teacher_metrics.json');atomic_json(cards,out/'examples.json');save_examples(out,cards)
    lines=['# PCA teacher observed-history benchmark','','Lower MSE is better. Student inference only uses observed input and its own endpoint state.','']
    for split,rows in report['datasets'].items():
        lines += [f'## {split}','','| variant | primary | path | changes | body | activity | close bp |','|---|---:|---:|---:|---:|---:|---:|']
        for name,row in rows['variants'].items():lines.append(f'| {name} | '+' | '.join(f'{row["metrics"][k]:.5f}' for k in METRICS)+' |')
        lines.append('')
    (out/'summary.md').write_text('\n'.join(lines));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('all','preflight','evaluate','worker'))
    parser.add_argument('--source');parser.add_argument('--out',required=True);parser.add_argument('--name')
    parser.add_argument('--parent-epochs',type=int,default=200);parser.add_argument('--branch-epochs',type=int,default=100)
    parser.add_argument('--batch',type=int,default=128);parser.add_argument('--micro',type=int,default=64);parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args();out=Path(args.out).resolve();ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal student training/evaluation runs on AutoDL CUDA')
    if args.action=='worker':worker(out,args.name);return
    if not args.source:parser.error('--source required')
    if not 1<=args.jobs<=4:raise ValueError('jobs must be1..4')
    source=Path(args.source).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Separate sibling output required')
    progress('Verifying original architecture data/PCA source read-only')
    identity=cb.source_identity(source)
    meta=make_manifest(source,identity,args.parent_epochs,args.branch_epochs,args.batch,args.micro)
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Configuration changed; use a new run')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output without manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    runtime=dict(torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(),jobs=args.jobs,action=args.action,started_unix=time.time())
    atomic_json(runtime,out/'runtime.json')
    with (out/'runtime_history.jsonl').open('a') as f:f.write(json.dumps(runtime)+'\n')
    prepare(meta,out)
    if args.action in ('all','preflight'):
        preflight(meta,out)
        if args.action=='preflight':return
    if args.action=='all':run_phase(out,args.jobs,'base')
    lock_selection(meta,out,parents_only=True)
    if args.action=='all':run_phase(out,args.jobs,'branch')
    evaluate(meta,out)
    if cb.source_identity(source)!=identity:raise ValueError('Original source changed during experiment')
    verify_cache(meta,out)
    files={f.name:sha256(f) for f in out.iterdir() if f.is_file() and f.suffix in ('.json','.jsonl','.md','.html') and f.name!='completion.json'}
    files['cache/index.json']=sha256(out/'cache/index.json')
    atomic_json(dict(status='complete',trials=len(meta['experiments']),source_unchanged=True,automatic_promotion=False,independent_holdout=False,files=files),out/'completion.json')
    progress('PCA teacher benchmark complete; existing production/research models not replaced')


if __name__=='__main__':
    import fcntl
    if '--out' not in sys.argv:main()
    else:
        root=Path(sys.argv[sys.argv.index('--out')+1]).resolve()
        name=sys.argv[sys.argv.index('--name')+1] if len(sys.argv)>1 and sys.argv[1]=='worker' and '--name' in sys.argv else 'controller'
        root.parent.mkdir(parents=True,exist_ok=True)
        with (root.parent/('.'+root.name+'.'+name+'.lock')).open('a') as lockfile:
            try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit('Another process is already running this output/job')
            main()
