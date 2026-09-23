"""Fixed-budget endpoint sampling with immutable plain PCA controls."""
import argparse
import json
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import architecture as ar, architecture_benchmark as ab, capacity_benchmark as cb
from . import pca_teacher as pt, pca_teacher_benchmark as tb, pca_anneal_benchmark as an
from . import window_sampling as ws, coverage_audit as cov
from .ae_extend import atomic_json, rng_state, restore_rng
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-window-sampling-v1'
GOAL = dict(stage='Test train-window sampling diversity at fixed update and exposure budgets for causal historical compression.',
            automatic_promotion=False, independent_holdout=False,
            context=tb.GOAL['context'])
METRICS = tb.METRICS
run_epoch = tb.run_epoch
model_for = tb.model_for
load_data = tb.load_data
publish = tb.publish
preflight = tb.preflight


def code_identity():
    modules = (an, ws, cov, cov.data, cov.ae_context, cov.aa, cov.ar_codec, cov.rp)
    return tb.code_identity() | {Path(m.__file__).name: sha256(m.__file__) for m in modules} | {Path(__file__).name: sha256(__file__)}


def verify_code(meta):
    if meta['code_sha256'] != code_identity():
        raise ValueError('Sampling implementation changed; use a new output')


def teacher_weight(job, epoch):
    if job['schedule'] != 'constant' or job['teacher_weight'] != 0. or job['residual']:
        raise ValueError('Sampling experiment requires plain fixed PCA')
    return 0.


def verify_history(history, job, batch, windows, weight_fn):
    steps=(windows+batch-1)//batch
    if [r['epoch'] for r in history] != list(range(1,len(history)+1)) or len(history)>job['epochs']:
        raise ValueError('Noncontiguous or excessive training history')
    for r in history:
        if (r['lr'] != ar.learning_rate(r['epoch'],job['epochs'],job['lr']) or r['teacher_weight'] != weight_fn(r['epoch'])
                or r['windows'] != windows or r['optimizer_steps'] != steps):
            raise ValueError('Training schedule/exposure/update-count mismatch')


def verify_sampling_history(meta, out, job, history):
    verify_history(history, job, meta['batch'], meta['windows_per_epoch'], lambda e: 0.)
    plan = read_json(out/f'candidates/sampling_s{job["seed"]}.json')
    for row in history:
        expected = plan['epochs'][row['epoch']-1]
        if row['sampling'] != expected:
            raise ValueError('Sampling exposure/history mismatch')
        ids = ws.sample_ids(plan['pool_size'], meta['windows_per_epoch'], job['seed'], row['epoch'])
        if cov.ndarray_hash(ids) != expected['ids_sha256']:
            raise ValueError('Sampling membership no longer reproduces')


def lock_selection(meta, out):
    for job in meta['experiments']:
        summary=an.selected_trial(out, meta, job, lambda e: 0.)
        last=torch.load(out/job['name']/'last.pt',map_location='cpu',weights_only=True)
        digest=sha256(out/'candidates/index.json')
        if last['candidate_cache_sha256']!=digest or summary['candidate_cache_sha256']!=digest:
            raise ValueError('Candidate cache changed after training')
        verify_sampling_history(meta, out, job, read_json(out/job['name']/'history.json'))
    return tb.lock_selection(meta, out)


source_identity = an.source_identity


def make_manifest(teacher, identity, coverage, coverage_id, raw_root):
    sm = identity['manifest']
    if sm['config'] != pt.CONFIG:
        raise ValueError('Keep original student configuration')
    jobs = [dict(name=f'sampled_s{s}', phase='base', seed=s, parent=None, epochs=sm['parent_epochs'],
                 lr=3e-4, residual=False, teacher_weight=0., schedule='constant') for s in (42,43)]
    return dict(schema=SCHEMA, source=sm['source'], source_identity=sm['source_identity'],
        teacher_source=str(teacher.resolve()), teacher_identity=identity, coverage_source=str(coverage.resolve()),
        coverage_identity=coverage_id, raw_root=str(raw_root.resolve()), windows_per_epoch=coverage_id['train_count'],
        config=sm['config'], batch=sm['batch'], micro=sm['micro'], seeds=[42,43], experiments=jobs,
        precision=sm['precision'], goal=GOAL, code_sha256=code_identity(),
        objective=dict(sm['objective'], train='Original SmoothL1; no coordinate auxiliary loss', teacher_strength=0.),
        sampling=ws.SAMPLING_VERSION,
        selection='Original validation-primary minimum including epoch0; both new selections locked before research evaluation. Best and fixed epoch200 reported.',
        schedule='Each epoch samples4789 without replacement from38549 train candidates, then uses original minibatch shuffle. Original200-epoch LR and continuous AdamW. No optimizer reset.',
        controls='Two immutable200-epoch plain runs. All jobs from scratch;7600 updates and957800 window exposures per seed.',
        acceptance='Exploratory: primary improvement against plain in both seeds and research sets, no supported family/close-MAE regression. No detected regression is not equivalence.',
        stop_rule='If unstable, retain original plain route; do not automatically add sampling variants or extend budget.',
        limitations='Repeated research sets and uncorrected intervals. Same update/exposure budget is not equal compute or convergence. Sampling changes positions, some unique-bar coverage and stratum exposure; no pure-position attribution or automatic promotion.')


def verify_cache(meta, out):
    an.verify_cache(meta, out)
    ws.verify_cache(meta, out)


def prepare(meta, out):
    verify_code(meta)
    root = Path(meta['teacher_source']); ab.sc.verify_worker_files(root, meta['teacher_identity']['files'])
    verify_files(root/'cache', meta['teacher_identity']['coordinate_files'])
    coverage = Path(meta['coverage_source']); verify_files(coverage, meta['coverage_identity']['files'])
    copied = out/'coverage'; copied.mkdir(exist_ok=True)
    for name in meta['coverage_identity']['files']:
        shutil.copyfile(coverage/name, copied/name)
    cache = out/'cache'; cache.mkdir(exist_ok=True)
    if not (cache/'index.json').exists():
        for name in meta['teacher_identity']['coordinate_files']:
            shutil.copyfile(root/'cache'/name, cache/name)
        shutil.copyfile(root/'statistics.json', out/'statistics.json')
        for job in meta['teacher_identity']['controls']:
            folder = out/'controls'/job['name']; folder.mkdir(parents=True, exist_ok=True)
            for name in ('history.json','training_summary.json','completion.json'):
                shutil.copyfile(root/job['name']/name, folder/name)
        atomic_json(dict(manifest=meta, files=meta['teacher_identity']['coordinate_files']), cache/'index.json')
    an.verify_cache(meta, out)
    if read_json(cache/'coordinate_scales.json')['count'] != meta['windows_per_epoch']:
        raise ValueError('Old control and new sampling exposure budgets differ')
    ws.prepare_candidates(meta, out)
    verify_cache(meta, out)


def worker(out, name, device='cuda'):
    meta=read_json(out/'manifest.json');verify_code(meta)
    job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True)
    lineage=None;metadata=dict(manifest=meta,job=job,lineage=lineage)
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['status']!='complete' or done['metadata']!=metadata:raise ValueError('Completed worker identity mismatch')
        verify_files(path,done['files']);return
    verify_cache(meta,out)
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
    model=model_for(meta,out,job,device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=job['lr'],weight_decay=.01)
    stats=read_json(Path(meta['source'])/'cache/statistics.json');data=dict(train=ws.load_candidates(out), val=load_data(meta,out,'val'))
    plan=read_json(out/f'candidates/sampling_s{job["seed"]}.json')
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['candidate_cache_sha256']!=sha256(out/'candidates/index.json'):
            raise ValueError('Candidate cache changed since checkpoint')
        if state['metadata']!=metadata or state['epoch']>job['epochs'] or [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)):
            raise ValueError('Resume metadata/history mismatch')
        verify_sampling_history(meta,out,job,state['history'])
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        weight=teacher_weight(job,state['epoch'])
        replay,_=run_epoch(model,data['val'],stats,meta['batch'],meta['micro'],device,weight)
        expected=state['history'][-1]['validation'] if state['history'] else state['initial_validation']
        if any(not np.isclose(replay[k],expected[k],atol=1e-6,rtol=2e-5) for k in replay):raise ValueError('Resume validation mismatch')
        atomic_json(dict(epoch=state['epoch'],max_abs=max(abs(replay[k]-expected[k]) for k in replay)),path/'resume_validation.json')
    else:
        initial,_=run_epoch(model,data['val'],stats,meta['batch'],meta['micro'],device,teacher_weight(job,0))
        # Reconstruct identical random initialization; compare all validation quantities
        # against the plain control's stored epoch0, before any optimizer update.
        control_path=Path(meta['teacher_source'])/f'plain_s{job["seed"]}'/'last.pt'
        if sha256(control_path)!=meta['teacher_identity']['files'][f'plain_s{job["seed"]}/last.pt']:
            raise ValueError('Initialization control checkpoint changed')
        control=torch.load(control_path,map_location='cpu',weights_only=True)
        reference=control['initial_validation']
        if any(not np.isclose(initial[k],reference[k],atol=1e-6,rtol=2e-5) for k in initial):
            raise ValueError('Random initialization validation differs from source control')
        atomic_json(dict(validation=initial,source_validation=reference,matched=True,
            source_checkpoint_sha256=meta['teacher_identity']['files'][f'plain_s{job["seed"]}/last.pt']),path/'initialization.json')
        del control
        state=dict(metadata=metadata,candidate_cache_sha256=sha256(out/'candidates/index.json'),epoch=0,history=[],initial_validation=initial,best_epoch=0,best_validation=initial,best_model=ab.cpu_state(model))
        state.update(model=ab.cpu_state(model),optimizer=opt.state_dict(),rng=rng_state());publish(state,path)
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['epoch']+1,job['epochs']+1):
        started=time.monotonic();lr=ar.learning_rate(epoch,job['epochs'],job['lr']);weight=teacher_weight(job,epoch)
        for g in opt.param_groups:g['lr']=lr
        ids=ws.sample_ids(len(data['train']['x']),meta['windows_per_epoch'],job['seed'],epoch)
        sampling=plan['epochs'][epoch-1]
        if cov.ndarray_hash(ids)!=sampling['ids_sha256']:raise ValueError('Sampling plan changed')
        train,updates=run_epoch(model,ws.subset(data['train'],ids),stats,meta['batch'],meta['micro'],device,weight,opt,job['seed']+epoch*1009)
        if updates != (len(ids)+meta['batch']-1)//meta['batch']:
            raise ValueError('Per-epoch optimizer update budget changed')
        val,_=run_epoch(model,data['val'],stats,meta['batch'],meta['micro'],device,weight)
        improved=val['primary']<state['best_validation']['primary']
        if improved:state.update(best_epoch=epoch,best_validation=val,best_model=ab.cpu_state(model))
        elapsed=time.monotonic()-started
        state['history'].append(dict(epoch=epoch,train=train,validation=val,lr=lr,teacher_weight=weight,optimizer_steps=updates,
            windows=len(ids),sampling=sampling,seconds=elapsed,remaining_minutes=elapsed*(job['epochs']-epoch)/60))
        state.update(epoch=epoch,model=ab.cpu_state(model),optimizer=opt.state_dict(),rng=rng_state());publish(state,path,improved)
        progress(f'{name} epoch={epoch}/{job["epochs"]} val={val["primary"]:.5f} coordinate_mse={val["coordinate_mse"]:.5f} best={state["best_epoch"]} seconds={elapsed:.1f}')
    verify_sampling_history(meta,out,job,state['history'])
    publish(state,path)
    atomic_json(dict(selected_epoch=state['best_epoch'],candidate_cache_sha256=state['candidate_cache_sha256'],sampling=ws.SAMPLING_VERSION,window_exposures=sum(r['windows'] for r in state['history']),validation=state['best_validation'],lineage=lineage,
        trained_selection=state['best_epoch']>0,
        selected_postbranch_update=None,selected_at_budget_end=state['best_epoch']==job['epochs'],
        parameters=dict(encoder=sum(p.numel() for p in model.encoder.parameters()),decoder=sum(p.numel() for p in model.decoder.parameters()),
                        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
        training_seconds=sum(r['seconds'] for r in state['history']),optimizer_steps=sum(r['optimizer_steps'] for r in state['history']),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None),path/'training_summary.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',metadata=metadata,files=files),path/'completion.json')


def run_phase(out,jobs,phase):
    """Run only the two predeclared from-scratch jobs, with bounded concurrency."""
    pending=[j for j in read_json(out/'manifest.json')['experiments'] if j['phase']==phase];active={};previous={}
    def stop(signum,frame):raise SystemExit(128+signum)
    handler=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                job=pending.pop(0);name=job['name'];(out/name).mkdir(exist_ok=True);log=(out/name/'run.log').open('a')
                try:proc=subprocess.Popen([sys.executable,'-m','obson.babel.sampling_benchmark','worker','--out',str(out),'--name',name],stdout=log,stderr=subprocess.STDOUT)
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


def replay_control(root, sm, job, model, val, stats, device):
    ck=torch.load(root/job['name']/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model'])
    row,_=run_epoch(model,val,stats,sm['batch'],sm['micro'],device,job['teacher_weight'])
    if any(not np.isclose(row[k],ck['validation'][k],atol=1e-6,rtol=2e-5) for k in row):
        raise ValueError('Reused control validation failed to reproduce')
    return ck


def evaluate(meta,out,device='cuda'):
    verify_code(meta);verify_cache(meta,out)
    lock=lock_selection(meta,out)
    # Neither neural test inference nor raw research-window audit precedes this lock.
    root=Path(meta['teacher_source']);sm=meta['teacher_identity']['manifest'];source=Path(meta['source'])
    ab.sc.verify_worker_files(root,meta['teacher_identity']['files'])
    old=read_json(root/'teacher_metrics.json');stats=read_json(out/'statistics.json')
    pca=cb.source_pca(meta);scales=read_json(out/'cache/coordinate_scales.json');val=load_data(meta,out,'val')
    report=dict(schema=SCHEMA,goal=GOAL,trials=lock['all_trials'],datasets={},
        controls={j['name']:read_json(root/j['name']/'training_summary.json') for j in meta['teacher_identity']['controls'] if j['name'].startswith('plain_')},
        limitations=meta['limitations'],acceptance=meta['acceptance'])
    cards=[]
    for split in ('test','cross_research'):
        data=ab.load_arrays(source,split);inv=read_json(source/f'cache/{split}_inventory.json')
        atomic_json(inv,out/f'{split}_inventory.json')
        q=pt.normalize_coordinates(pt.coefficients(pca,data),scales);scores={};errors={};last_scores={};last_errors={}
        ids=list(np.linspace(0,len(data['y'])-1,min(6,len(data['y'])),dtype=int))
        examples={i:dict(dataset=split,key=inv[i]['key'],end=inv[i]['end'],mask=data['mask'][i].tolist(),
            series={'truth':(data['y'][i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()}) for i in ids}
        def record(name,pred):
            scores[name],errors[name]=ab.measure(pred,data,stats)
            for i in ids:examples[i]['series'][name]=(pred[i]*np.array(stats['y_scale'])+np.array(stats['y_mean'])).tolist()
        for rank in (meta['config']['latent']//2,meta['config']['latent']):
            record(f'pca{rank}',ab.pca_predict(cb.prefix_pca(pca,rank),data))
        old_errors=read_json(root/f'{split}_per_window_errors.json')
        for rank in (meta['config']['latent']//2,meta['config']['latent']):
            label=f'pca{rank}'
            if label in old_errors:
                for k in METRICS:
                    if not np.allclose(errors[label][k],old_errors[label][k],atol=1e-6,rtol=2e-5):
                        raise ValueError('Pinned PCA research results changed')
        for job in [j for j in meta['teacher_identity']['controls'] if j['name'].startswith('plain_')]+meta['experiments']:
            control=job['name'] in report['controls'];path=(root if control else out)/job['name']
            model=model_for(meta,out,job,device)
            if control: ck=replay_control(root,sm,job,model,val,stats,device)
            else:
                ck=torch.load(path/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
                replay,_=run_epoch(model,val,stats,meta['batch'],meta['micro'],device,teacher_weight(job,ck['epoch']))
                if any(not np.isclose(replay[k],ck['validation'][k],atol=1e-6,rtol=2e-5) for k in replay):
                    raise ValueError('Sampled selected validation failed to reproduce')
            pred,z=ab.predictions(model,data['x'],meta['micro'],device);record(job['name'],pred)
            scores[job['name']]['coordinate_alignment']=tb.coordinate_metrics(z,q)
            last=torch.load(path/'last.pt',map_location='cpu',weights_only=True);model.load_state_dict(last['model'])
            pred,_=ab.predictions(model,data['x'],meta['micro'],device)
            last_scores[job['name']],last_errors[job['name']]=ab.measure(pred,data,stats)
            if control:
                for k in METRICS:
                    if not np.allclose(errors[job['name']][k],old_errors[job['name']][k],atol=1e-6,rtol=2e-5):
                        raise ValueError(f'Control per-window result changed: {split}/{job["name"]}/{k}')
                    if not np.isclose(last_scores[job['name']]['metrics'][k],old['datasets'][split]['fixed_last'][job['name']]['metrics'][k],atol=1e-6,rtol=2e-5):
                        raise ValueError('Control fixed-last result changed')
            del ck,last,model,pred,z
            progress(f'Evaluated {split} {job["name"]}: best and fixed-last, control={control}')
        pairs=[(f'sampled_s{s}',f'plain_s{s}') for s in meta['seeds']]
        paired={a+'_minus_'+b:{k:ab.pair_groups(errors[a][k],errors[b][k],inv) for k in METRICS} for a,b in pairs}
        last_paired={a+'_minus_'+b:{k:ab.pair_groups(last_errors[a][k],last_errors[b][k],inv) for k in METRICS} for a,b in pairs}
        report['datasets'][split]=dict(variants=scores,fixed_last=last_scores,paired=paired,fixed_last_paired=last_paired)
        atomic_json({n:{k:v.tolist() for k,v in r.items()} for n,r in errors.items()},out/f'{split}_per_window_errors.json')
        atomic_json({n:{k:v.tolist() for k,v in r.items()} for n,r in last_errors.items()},out/f'{split}_last_per_window_errors.json')
        cards.extend(examples.values())
    report['exploratory_screen']={}
    for mode in ('plain',):
        checks=[];regressions=[]
        for split,rows in report['datasets'].items():
            for seed in meta['seeds']:
                name=f'sampled_s{seed}';group=rows['paired'][name+f'_minus_{mode}_s{seed}']
                ci=group['primary']['all']
                checks.append(lock['all_trials'][name]['trained_selection'] and ci['supported'] and ci['high'] is not None and ci['high']<0)
                for k in ('path','changes','body','activity','close_mae_bps'):
                    ci=group[k]['all']
                    if ci['supported'] and ci['low'] is not None and ci['low']>0:
                        regressions.append(dict(dataset=split,seed=seed,metric=k,interval=ci))
        report['exploratory_screen']['sampled_minus_'+mode]=dict(primary_improvement_all_seeds_and_datasets=all(checks),
            regressions=regressions,exploratory_criteria_met=all(checks) and not regressions,
            interpretation='Uncorrected intervals; no detected regression is not equivalence. No automatic promotion.')
    atomic_json(report,out/'sampling_metrics.json');atomic_json(cards,out/'examples.json')
    save_examples(out,cards)
    lines=['# Fixed-budget train-window sampling','','All neural comparisons:200 allocated epochs from scratch. Best selected using original validation only.','']
    for split,rows in report['datasets'].items():
        lines += [f'## {split}','','| variant | primary | path | changes | body | activity | close bp |','|---|---:|---:|---:|---:|---:|---:|']
        for name,row in rows['variants'].items():lines.append(f'| {name} | '+' | '.join(f'{row["metrics"][k]:.5f}' for k in METRICS)+' |')
        lines.append('')
    (out/'summary.md').write_text('\n'.join(lines));return report


def save_examples(out,cards):
    import html
    colors={'truth':'#222','pca512':'#842db1','plain_s42':'#bd8047','sampled_s42':'#23945a'}
    lines=['<!doctype html><meta charset="utf-8"><title>Fixed-budget window sampling</title>',
        '<style>body{font:14px sans-serif;margin:24px}svg{border:1px solid #ddd;max-width:100%}</style>',
        '<h1>Observed history reconstruction, not forecasts</h1><p>Fixed evenly spaced examples, seed42. Both seeds and fixed-last results are in metrics.</p>']
    lines.append(' / '.join(f'<span style="color:{c}">{n}</span>' for n,c in colors.items()))
    for row in cards:
        lines.append('<h3>'+html.escape(row['dataset']+' '+row['key']+' '+row['end'])+'</h3>')
        for channel,label in [(0,'Close log-percent from pre-window anchor'),(2,'Volume relative to prior EMA'),(4,'Observed OI change, asinh')]:
            mask=np.asarray(row['mask'])[:127,channel]
            series={n:np.asarray(v)[:127,channel] for n,v in row['series'].items() if n in colors}
            vals=np.concatenate([v[mask] for v in series.values()]);lo=float(vals.min()) if len(vals) else 0.;hi=float(vals.max()) if len(vals) else 1.;span=max(hi-lo,1e-5)
            lines.append(f'<p>{label}</p><svg viewBox="0 0 900 180" width="900" height="180">')
            for name,v in series.items():
                parts=[('M' if i==0 or not mask[i-1] else 'L')+f'{10+i*880/126:.2f},{170-(v[i]-lo)/span*160:.2f}' for i in range(127) if mask[i]]
                lines.append(f'<path d="{" ".join(parts)}" fill="none" stroke="{colors[name]}" stroke-width="1.2"/>')
            lines.append('</svg>')
    (out/'examples.html').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('all','preflight','evaluate','worker'))
    parser.add_argument('--source');parser.add_argument('--out',required=True);parser.add_argument('--name')
    parser.add_argument('--coverage');parser.add_argument('--root',default='data/contracts');parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args();out=Path(args.out).resolve();ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal training/evaluation runs on AutoDL CUDA')
    if args.action=='worker':worker(out,args.name);return
    if not args.source or not args.coverage:parser.error('--source and --coverage required')
    if not 1<=args.jobs<=2:raise ValueError('Two declared jobs; jobs must be1 or2')
    source=Path(args.source).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Separate sibling output required')
    progress('Verifying completed plain controls and original PCA/cache read-only')
    identity=source_identity(source);sm=identity['manifest']
    if sm['parent_epochs']!=200 or sm['batch']!=128 or sm['micro']!=64 or sm['config']!=pt.CONFIG:
        raise ValueError('Formal protocol requires source200 epochs, original512 configuration,batch128,micro64')
    old_runtime=read_json(source/'runtime.json')
    if str(torch.__version__)!=old_runtime['torch'] or np.__version__!=old_runtime['numpy']:
        raise ValueError('Torch/NumPy versions differ from reused controls; restore the source environment')
    original=Path(sm['source']).resolve()
    if out==original or original in out.parents or out in original.parents:raise ValueError('Output must not overlap original data source')
    coverage=Path(args.coverage).resolve()
    coverage_id=ws.coverage_identity(coverage,original)
    if coverage_id['train_count']!=4789 or coverage_id['candidate']['coverage']['windows']!=38549:
        raise ValueError('Formal experiment requires audited4789/38549 window counts')
    for dependency in [coverage, Path(args.root).resolve()]+[Path(p).resolve() for p in coverage_id['sources'].values()]:
        if out==dependency or out in dependency.parents or dependency in out.parents:
            raise ValueError('Output must not overlap any input')
    meta=make_manifest(source,identity,coverage,coverage_id,Path(args.root))
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Configuration changed; use a new run')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output without manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    runtime=dict(torch=str(torch.__version__),numpy=np.__version__,gpu=torch.cuda.get_device_name(),jobs=args.jobs,
        action=args.action,started_unix=time.time(),raw_root=str(Path(args.root).resolve()),source_runtime=old_runtime)
    atomic_json(runtime,out/'runtime.json')
    with (out/'runtime_history.jsonl').open('a') as f:f.write(json.dumps(runtime)+'\n')
    (out/'completion.json').unlink(missing_ok=True)
    prepare(meta,out)
    if args.action in ('all','preflight'):
        preflight(meta,out)
        if args.action=='preflight':return
    if args.action=='all':run_phase(out,args.jobs,'base')
    evaluate(meta,out)
    if ws.coverage_identity(coverage,original)!=coverage_id:raise ValueError('Coverage sources changed during experiment')
    if source_identity(source)!=identity:raise ValueError('Controls changed during experiment')
    verify_cache(meta,out)
    files={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.md','.html') and p.name!='completion.json'}
    files['cache/index.json']=sha256(out/'cache/index.json')
    files['candidates/index.json']=sha256(out/'candidates/index.json')
    atomic_json(dict(status='complete',trials=2,reused_controls=2,source_unchanged=True,automatic_promotion=False,
        independent_holdout=False,files=files),out/'completion.json')
    progress('Fixed-budget window sampling complete; no automatic model promotion')


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
