"""Matched-budget continuation of activity alignment, with two validation selectors."""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import activity_alignment as al
from . import activity_ablation as aa
from . import detail_alignment as da
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import run_jobs
from .dual_state import sha256
from .progress import progress

SCHEMA = 'babel-activity-extend-v1'
GOAL = dict(al.GOAL, stage='Check matched longer training before changing architecture; preserve price selection and add validation past-history selection.',
    limitations='Same pretrained encoder, reused research test, fixed learning rates and two finetuning seeds.100 epochs is a budget, not a convergence certificate.')


def parent_identity(parent):
    meta = json.loads((parent/'manifest.json').read_text())
    if meta['schema'] not in (al.SCHEMA, SCHEMA): raise ValueError('Not an activity-alignment run')
    if json.loads((parent/'completion.json').read_text()).get('status') != 'complete': raise ValueError('Parent run incomplete')
    index = json.loads((parent/'cache/index.json').read_text())
    if index['manifest'] != meta: raise ValueError('Parent cache metadata mismatch')
    al.verify_alignment_cache(parent/'cache', index['files'], Path(meta['activity_run']))
    _, identity = al.source_identity(Path(meta['activity_run']), Path(meta['fusion_run']))
    if identity != meta['upstream']: raise ValueError('Original feature/weight sources changed')
    names = ['manifest.json','completion.json','cache/index.json','alignment_metrics.json']
    for job in meta['experiments']:
        names += [f'{job["name"]}/{s}' for s in ('last.pt','best.pt','per_window_metrics.json')]
        if meta['schema']==SCHEMA: names.append(f'{job["name"]}/activity_best.pt')
    return meta, {name:sha256(parent/name) for name in names}


def make_manifest(parent, old, identity, epochs, every):
    if epochs <= old['epochs'] or every < 1: raise ValueError('Continuation must increase total epochs')
    meta = copy.deepcopy(old)
    meta.update(schema=SCHEMA, epochs=epochs, parent_run=str(parent.resolve()), parent_identity=identity,
        parent_epochs=old['epochs'], selection_every=every, goal=GOAL,
        secondary_selection='Train-fitted standardized multi-output ridge on15 past-only targets, alpha selected by validation; evaluate parent available checkpoints and fixed epoch grid. Price retention relative to same-seed parent control best; no test selection.',
        continuation='Copy exact last model/AdamW/RNG/history and primary best into NEW directory. Same LR, input, streams, seed+absolute_epoch ordering; no scheduler or optimizer reset.')
    return meta


def prepare(meta, out):
    parent=Path(meta['parent_run']); cache=out/'cache'; cache.mkdir(exist_ok=True)
    original=json.loads((parent/'cache/index.json').read_text()); index=cache/'index.json'
    if index.exists():
        saved=json.loads(index.read_text())
        if saved['manifest']!=meta or saved['files']!=original['files']: raise ValueError('Continuation cache identity mismatch')
        al.verify_alignment_cache(cache,saved['files'],Path(meta['activity_run']));return
    for name in original['files']:
        dst=cache/name; src=parent/'cache'/name
        if name in {f'{s}_x.npy' for s in da.SPLITS}:
            if not dst.exists(): dst.symlink_to(src.resolve())
            if dst.resolve()!=src.resolve(): raise ValueError('Wrong linked feature bank')
        else: shutil.copyfile(src,dst)
    al.verify_alignment_cache(cache,original['files'],Path(meta['activity_run']))
    for name in ('coverage.json','target_audit.json'): shutil.copyfile(parent/name,out/name)
    atomic_json(dict(manifest=meta,files=original['files']),index)


def import_state(meta, job):
    parent=Path(meta['parent_run']); old=json.loads((parent/'manifest.json').read_text())
    path=parent/job['name']; expected=dict(manifest=old,job=job)
    for name in (('last.pt','best.pt','activity_best.pt') if old['schema']==SCHEMA else ('last.pt','best.pt')):
        if sha256(path/name)!=meta['parent_identity'][f'{job["name"]}/{name}']: raise ValueError('Parent checkpoint changed')
    state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    best=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
    if state['metadata']!=expected or best['metadata']!=expected: raise ValueError('Parent checkpoint metadata mismatch')
    if state['epoch']!=meta['parent_epochs'] or [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)):
        raise ValueError('Parent training has not completed declared budget')
    if best['epoch']!=state['best_epoch'] or best['validation']!=state['best_validation']: raise ValueError('Parent primary best mismatch')
    if set(best['model'])!=set(state['best_model']) or any(not torch.equal(v,state['best_model'][k]) for k,v in best['model'].items()):
        raise ValueError('Parent primary best tensors mismatch')
    # Only provenance changes. Every tensor, optimizer accumulator and RNG entry survives.
    state['metadata']=dict(manifest=meta,job=job)
    parent_history=state.get('activity_best')
    for key in ('activity_best','selection_history','selectors_initialized'):
        state.pop(key,None)
    state.update(activity_best=None,selection_history=[],selectors_initialized=False)
    if parent_history is not None: state['parent_history_candidate']=parent_history
    return state


def ridge_selection(train_z, val_z, train_y, val_y, train_mask, val_mask):
    """No test inputs; common support and one validation-selected alpha for15 targets."""
    tr=train_mask[:,5:].all(1); va=val_mask[:,5:].all(1)
    if min(int(tr.sum()),int(va.sum()))<2: raise ValueError('Insufficient past-only selection support')
    x=train_z[tr].astype(float); v=val_z[va].astype(float)
    mean=x.mean(0); scale=x.std(0).clip(1e-5)
    x=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    v=np.column_stack(((v-mean)/scale,np.ones(len(v))))
    y=train_y[tr,5:]; target=val_y[va,5:]
    lhs=x.T@x; rhs=x.T@y; best=None
    for alpha in (1.,10.,100.,1000.):
        penalty=np.eye(lhs.shape[0])*alpha;penalty[-1,-1]=0
        weight=np.linalg.solve(lhs+penalty,rhs); errors=(v@weight-target)**2
        score=float(errors.mean())
        if best is None or score<best['validation_past_mse']:
            best=dict(validation_past_mse=score,alpha=alpha,per_target_mse=errors.mean(0).tolist(),train_support=int(tr.sum()),validation_support=int(va.sum()))
    if not np.isfinite(best['validation_past_mse']): raise ValueError('Nonfinite selector')
    return best


def activity_eligible(score, reference):
    return da.eligible(score,reference) and score['detail']<=reference['detail']*1.02+1e-8


def consider_activity(state, model, epoch, score, readout, reference, origin):
    eligible=activity_eligible(score,reference)
    row=dict(epoch=epoch,origin=origin,validation=score,readout=readout,price_retained=eligible)
    state['selection_history'].append(row)
    best=state['activity_best']
    if eligible and (best is None or (readout['validation_past_mse'],score['detail'])<
                    (best['readout']['validation_past_mse'],best['validation']['detail'])):
        state['activity_best']=dict(epoch=epoch,validation=score,readout=readout,origin=origin,
            model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    return row


def publish(state,path):
    da.publish(state,path)
    selected=state['activity_best']
    if selected is None:
        selected=dict(epoch=state['best_epoch'],model=state['best_model'],validation=state['best_validation'])
    ck=dict(selected,metadata=state['metadata'],selection_qualified=state['activity_best'] is not None,
        selection_role='past_history' if state['activity_best'] is not None else 'price_fallback_no_qualified_history_candidate')
    atomic_save(ck,path/'activity_best.pt')
    atomic_json(dict(reference=state['activity_reference'],candidates=state['selection_history'],qualified=ck['selection_qualified'],
        selected_epoch=ck['epoch'],limitations='Parent epochs lacking saved tensors cannot be retrospectively selected. New grid uses validation only.'),path/'selection_history.json')


def optimizer_for(model,meta):
    return torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr']),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=meta['encoder_lr']),
        dict(params=model.activity_head.parameters(),lr=meta['decoder_lr'])],weight_decay=.01)


def worker(out,name,device='cuda'):
    meta=json.loads((out/'manifest.json').read_text()); stats=json.loads((out/'cache/statistics.json').read_text())
    job=next(j for j in meta['experiments'] if j['name']==name);path=out/name;path.mkdir(exist_ok=True)
    state=torch.load(path/'last.pt',map_location='cpu',weights_only=True) if (path/'last.pt').exists() else import_state(meta,job)
    if state['metadata']!=dict(manifest=meta,job=job): raise ValueError('Continuation resume identity mismatch')
    model=al.initial_model(meta,device,job);model.configure(True);opt=optimizer_for(model,meta)
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    data={s:al.arrays_for(meta,out,s,device) for s in ('train','val')}
    streams={s:aa.ActivityStreams(out/'cache',s,job) for s in ('train','val')}
    callback=al.latent_objective(model,job['aux_weight'])
    reference_ck=torch.load(Path(meta['parent_run'])/f'control_s{job["seed"]}'/'best.pt',map_location='cpu',weights_only=True)
    state['activity_reference']=reference_ck['validation']
    restore_rng(state['rng'])
    def run(split,training=False,epoch=None,collect=False):
        return da.run_epoch(model,data[split],streams[split],True,True,stats['scales'],meta['streams'],
            opt if training else None,job['seed']+epoch if training else None,collect=collect,latent_objective_fn=callback)
    def select(epoch,origin):
        # Full chronological eval replay. Protect training RNG against diagnostics.
        rng=rng_state()
        try:
            _,tr,_=run('train',collect=True);score,va,_=run('val',collect=True)
            readout=ridge_selection(tr,va,*(data[s][key].cpu().numpy() for key in ('activity','activity_mask') for s in ('train','val')))
            row=consider_activity(state,model,epoch,score,readout,state['activity_reference'],origin)
            progress(f'{name} history selector epoch={epoch} val_mse={readout["validation_past_mse"]:.4f} price_retained={row["price_retained"]}')
        finally: restore_rng(rng)
    if not state['selectors_initialized']:
        actual=run('val')[0]; expected=state['history'][-1]['validation']
        if not all(np.isclose(actual[k],v,rtol=1e-4,atol=1e-5) for k,v in expected.items()): raise ValueError('Parent last validation does not reproduce')
        atomic_json(dict(resumed_epoch=state['epoch'],parent_last_sha256=meta['parent_identity'][f'{name}/last.pt'],expected=expected,replayed=actual,
            optimizer_groups=[dict(lr=g['lr'],parameters=len(g['params'])) for g in opt.param_groups],optimizer_and_rng_restored=True),path/'resume_validation.json')
        # Old primary winner and last are the only routinely saved parent tensors.
        last={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        select(state['epoch'],'parent_last')
        if state['best_epoch']!=state['epoch']:
            model.load_state_dict(state['best_model']);select(state['best_epoch'],'parent_price_best')
        if state.get('parent_history_candidate') is not None:
            candidate=state.pop('parent_history_candidate')
            model.load_state_dict(candidate['model']);select(candidate['epoch'],'parent_history_best')
        model.load_state_dict(last);state['selectors_initialized']=True
        state.update(model=last,optimizer=opt.state_dict(),rng=rng_state());publish(state,path)
    for e in range(state['epoch']+1,meta['epochs']+1):
        started=time.perf_counter()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        train=run('train',True,e)[0];score=run('val')[0]
        if da.improves(score,state['best_validation'],stats['reference']):
            state.update(best_epoch=e,best_validation=score,best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if e%meta['selection_every']==0 or e==meta['epochs']:select(e,'continuation_grid')
        seconds=time.perf_counter()-started
        state['history'].append(dict(epoch=e,train=train,validation=score,eligible=da.eligible(score,stats['reference']),seconds=seconds,
            remaining_minutes=seconds*(meta['epochs']-e)/60,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None))
        state.update(epoch=e,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state());publish(state,path)
        progress(f'{name} epoch={e}/{meta["epochs"]} detail={score["detail"]:.5f} close={score["close_bps"]:.2f}bp price_best={state["best_epoch"]}')
    publish(state,path)


def build_activity_view(out):
    meta=json.loads((out/'manifest.json').read_text());view=out/'activity_selection';view.mkdir(exist_ok=True)
    atomic_json(meta,view/'manifest.json')
    if not (view/'cache').exists():(view/'cache').symlink_to((out/'cache').resolve(),target_is_directory=True)
    if (view/'cache').resolve()!=(out/'cache').resolve():raise ValueError('Wrong evaluation cache link')
    for job in meta['experiments']:
        folder=view/job['name'];folder.mkdir(exist_ok=True)
        shutil.copyfile(out/job['name']/'activity_best.pt',folder/'best.pt')
    return view


def annotate_activity_view(out):
    meta=json.loads((out/'manifest.json').read_text());view=out/'activity_selection'
    report=json.loads((view/'alignment_metrics.json').read_text());qualified={}
    for job in meta['experiments']:
        ck=torch.load(view/job['name']/'best.pt',map_location='cpu',weights_only=True)
        row=report['variants'][job['name']];qualified[job['name']]=ck['selection_qualified']
        row.update(selection_qualified=ck['selection_qualified'],selection_role=ck['selection_role'],selection_readout=ck.get('readout'))
        atomic_json(row,view/job['name']/'metrics.json')
    for mode,result in report['stage_decisions'].items():
        if not all(qualified[f'{m}_s{s}'] for s in meta['seeds'] for m in ('control',mode)):result['evidence_pass']=False
        result['all_selection_qualified']=all(qualified[f'{m}_s{s}'] for s in meta['seeds'] for m in ('control',mode))
    report['selection_scope']='Secondary validation-history selector; failed selections are explicitly labeled price fallbacks, not accepted history candidates.'
    atomic_json(report,view/'alignment_metrics.json')
    page=view/'summary.md'
    prefix=page.read_text().split('## Evidence screen')[0]
    page.write_text(prefix+'## Selection qualification\n'+json.dumps(qualified,indent=2)+'\n\n## Evidence screen\n'+json.dumps(report['stage_decisions'],indent=2)+'\n')
    return report


def verify_finished(out,meta):
    for job in meta['experiments']:
        state=torch.load(out/job['name']/'last.pt',map_location='cpu',weights_only=True)
        if (state['metadata']!=dict(manifest=meta,job=job) or state['epoch']!=meta['epochs'] or
                [r['epoch'] for r in state['history']]!=list(range(1,meta['epochs']+1))):
            raise ValueError(f'Continuation incomplete: {job["name"]}')


def budget_report(out):
    meta=json.loads((out/'manifest.json').read_text());parent=Path(meta['parent_run'])
    reports=[json.loads((p/'alignment_metrics.json').read_text()) for p in (parent,out,out/'activity_selection')]
    result=dict(goal=GOAL,from_epoch=meta['parent_epochs'],to_epoch=meta['epochs'],variants={},
        limits='Descriptive matched-budget comparisons on reused research test. Primary and secondary are separately declared selectors; do not choose between them using test scores. Last-window summaries do not certify convergence.')
    for job in meta['experiments']:
        name=job['name'];row={}
        for label,report in zip(('parent_price','extended_price','extended_history'),reports):
            v=report['variants'][name]
            row[label]=dict(selected_epoch=v['selected_epoch'],selection_qualified=v.get('selection_qualified',True),validation=v['validation'],
                close_mae_bps=v['test_objectives']['close_bps'],detail=v['test_objectives']['detail'],state_ba=v['state_probe']['test']['ba'],
                past_linear=v['linear_activity_readout']['normalized_error_summary']['all']['past_only'],
                past_nonlinear=v['nonlinear_activity_readout']['normalized_error_summary']['all']['past_only'])
        h=[json.loads(l) for l in (out/name/'history.jsonl').read_text().splitlines()]
        curves={}
        for metric in ('base','detail','close_bps','activity','past_activity'):
            values=np.array([x['validation'][metric] for x in h])
            curves[metric]=dict(best_epoch=h[int(values.argmin())]['epoch'],previous10_median=float(np.median(values[-20:-10])),last10_median=float(np.median(values[-10:])),
                interpretation='Auxiliary-head activity losses are not comparable across weights; control head is untrained.')
        row['validation_trends']=curves;result['variants'][name]=row
    result['stage_decisions']={label:r['stage_decisions'] for label,r in zip(('parent_price','extended_price','extended_history'),reports)}
    atomic_json(result,out/'budget_comparison.json')
    lines=['# Activity training budget audit','',GOAL['stage'],'', '| variant | selector | epoch | close bp | past linear MSE | past MLP MSE | state BA | eligible selector |', '|---|---|---:|---:|---:|---:|---:|---|']
    for name,v in result['variants'].items():
        for selector in ('parent_price','extended_price','extended_history'):
            r=v[selector];lines.append(f'| {name} | {selector} | {r["selected_epoch"]} | {r["close_mae_bps"]:.3f} | {r["past_linear"]:.4f} | {r["past_nonlinear"]:.4f} | {r["state_ba"]:.2%} | {r["selection_qualified"]} |')
    (out/'budget_summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker','evaluate'));p.add_argument('--out',required=True)
    p.add_argument('--parent');p.add_argument('--name');p.add_argument('--epochs',type=int,default=100);p.add_argument('--selection-every',type=int,default=10);p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();out=Path(a.out).resolve()
    if not torch.cuda.is_available():raise ValueError('Formal continuation/readout training requires CUDA')
    torch.set_num_threads(4)
    if a.action=='worker':
        if not a.name:p.error('--name required')
        worker(out,a.name);return
    if not a.parent:p.error('--parent required')
    if not 1<=a.jobs<=4:raise ValueError('jobs must be1..4')
    parent=Path(a.parent).resolve();old,identity=parent_identity(parent)
    if parent==out or parent in out.parents or out in parent.parents:raise ValueError('Use a separate sibling output directory')
    meta=make_manifest(parent,old,identity,a.epochs,a.selection_every)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Continuation settings changed; use another output directory')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty directory lacks manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');atomic_json(GOAL,out/'experiment_goal.json')
    (out/'completion.json').unlink(missing_ok=True);prepare(meta,out)
    atomic_json(dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),jobs=a.jobs),out/'runtime.json')
    progress(f'Matched continuation: all6 jobs epoch{old["epochs"]+1}..{a.epochs}; model/optimizer/RNG preserved')
    if a.action=='all':run_jobs(out,a.jobs,'obson.babel.activity_extend')
    verify_finished(out,meta)
    al.evaluate(out,'cuda')
    view=build_activity_view(out);al.evaluate(view,'cuda');annotate_activity_view(out);budget_report(out)
    if parent_identity(parent)[1]!=identity:raise ValueError('Parent artifacts changed during continuation')
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),parent_unchanged=True,from_epoch=old['epochs'],to_epoch=a.epochs,goal=GOAL),out/'completion.json')
    progress('Budget audit complete; both selectors reported, no automatic replacement')


if __name__=='__main__':main()
