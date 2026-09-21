"""Exploratory transfer probes on frozen price/activity states; no encoder updates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import price_readapt as pr, activity_ablation as aa, detail_alignment as da
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import run_jobs, paired_error_interval
from .dual_state import sha256, verify_files
from .progress import progress

SCHEMA='babel-state-transfer-v1'
HORIZONS=(32,64,128)
FAMILIES=('directional_efficiency','volatility_shift','price_volume_correlation','price_oi_correlation')
NAMES=[f'{family}/{h}' for family in FAMILIES for h in HORIZONS]
GOAL=dict(overall='Reusable causal compression of observed price and activity history.',
    stage='Frozen-state readout of12 descriptors not directly supervised by the20-target activity head, with current/statistical/incremental baselines.',
    limits='Derived historical descriptors, not unseen raw data, market mechanisms or prediction. Reused research endpoints; independent holdout not established.',
    automatic_promotion=False)


def correlation(a,b):
    if len(a)<2 or min(float(np.std(a)),float(np.std(b)))<1e-10:return 0.,False
    return float(np.clip(np.corrcoef(a,b)[0,1],-1,1)),True


def window_descriptors(window):
    """Each h-bar window uses only its h-1 INTERNAL close changes."""
    if window.ndim!=2 or window.shape[1]!=28 or len(window)<max(HORIZONS) or not np.isfinite(window).all():
        raise ValueError('Expected finite128+ by28 causal feature window')
    y=np.zeros((4,3));mask=np.ones((4,3),bool);stats=[]
    for j,h in enumerate(HORIZONS):
        w=np.asarray(window[-h:],dtype=float)
        # Invert un-clipped log-percent geometry, NOT volatility-normalized channels.
        r=(np.sinh(w[1:,0])+np.sinh(w[1:,1]))/100
        dv=np.diff(w[:,9]*10);oi=w[1:,20]
        valid=(w[1:,23]>0)&~((w[1:,22]==0)&(oi!=0))
        valid &= ~((w[1:,24]>0)&(np.abs(w[1:,21])>np.arcsinh(1.)+1e-6))
        denom=np.abs(r).sum();y[0,j]=r.sum()/denom if denom>1e-12 else 0.
        mid=len(r)//2;y[1,j]=np.log((np.mean(r[mid:]**2)+1e-8)/(np.mean(r[:mid]**2)+1e-8))
        y[2,j],mask[2,j]=correlation(r,dv)
        y[3,j],ok=correlation(r[valid],oi[valid]);mask[3,j]=ok and valid.sum()>=np.ceil(.8*len(r))
        # Generic moments only. No target formula or cross-moment is copied in.
        for values in (r,np.abs(r),w[:,9]*10,dv):stats.extend((float(values.mean()),float(values.std())))
        vals=oi[valid]
        stats.extend((float(vals.mean()) if len(vals) else 0.,float(vals.std()) if len(vals) else 0.,float(valid.mean())))
    y[~mask]=0.
    targets=y.ravel().astype(np.float32);features=np.r_[window[-1],stats].astype(np.float32)
    if not np.isfinite(targets).all() or not np.isfinite(features).all():raise ValueError('Nonfinite transfer descriptor')
    return targets,mask.ravel(),features


def targets_from_bank(bank,specs,n):
    y=np.zeros((n,len(NAMES)),np.float32);mask=np.zeros_like(y,bool)
    current=np.zeros((n,28),np.float32);stats=np.zeros((n,61),np.float32);seen=np.zeros(n,bool)
    for spec in specs:
        x=bank[spec['offset']:spec['offset']+spec['length']]
        for end,idx in spec['endpoints']:
            if not 0<=idx<n or end<127 or end>=len(x) or seen[idx]:raise ValueError('Invalid transfer endpoint')
            seen[idx]=True;y[idx],mask[idx],stats[idx]=window_descriptors(x[end-127:end+1]);current[idx]=x[end]
    if not seen.all():raise ValueError('Incomplete transfer endpoint coverage')
    return y,mask,current,stats


def source_identity(parent):
    meta=json.loads((parent/'manifest.json').read_text())
    if meta['schema']!=pr.SCHEMA or json.loads((parent/'completion.json').read_text()).get('status')!='complete':
        raise ValueError('Completed price-readaptation run required')
    _,identity=pr.source_identity(Path(meta['parent_run']))
    if identity!=meta['parent_identity']:raise ValueError('Encoder ancestry changed')
    index=json.loads((parent/'cache/index.json').read_text())
    if index['manifest']!=meta:raise ValueError('Frozen-state cache metadata mismatch')
    verify_files(parent/'cache',index['files'])
    files=['manifest.json','completion.json','cache/index.json','readapt_metrics.json']
    for job in meta['experiments']:files.extend(f'{job["name"]}/{f}' for f in ('best.pt','last.pt','selection.json'))
    return meta,{f:sha256(parent/f) for f in files}


def make_manifest(parent,old,identity,epochs=100,batch=256):
    if epochs<1 or batch<1:raise ValueError('Positive probe epochs and batch required')
    names=['current','statistics']+[j['name'] for j in old['experiments']]+[j['name']+'_plus_statistics' for j in old['experiments']]
    return dict(schema=SCHEMA,parent_run=str(parent.resolve()),parent_identity=identity,
        encoder_run=old['parent_run'],seeds=old['seeds'],epochs=epochs,batch=batch,probe_seeds=[1701,1702],
        decays=[.001,.01],hidden=128,lr=1e-3,names=NAMES,horizons=list(HORIZONS),goal=GOAL,
        experiments=[dict(name=n) for n in names],
        selection='Ridge alpha1/10/100/1000 per target; MLP validation equal-target standardized MSE selects epoch/decay separately for each probe seed. No test selection.',
        primary='Equal-weight12-task MSE on complete-case test endpoints (minimum50); family/target analyses exploratory. No promotion or fresh-holdout claim.')


def coverage_audit(old,parent,raw_root=None):
    encoder=Path(old['parent_run']);em=json.loads((encoder/'manifest.json').read_text())
    lm=Path(em['long_run'])/'manifest.json'
    ref=json.loads(lm.read_text()).get('manifest',{}) if lm.exists() else {}
    prior={s['key']:s for s in ref.get('sources',[])};candidates=[]
    if raw_root and Path(raw_root).exists():
        for f in sorted(Path(raw_root).glob('*/*.csv')):
            try:
                contract,period=f.stem.rsplit('_',1)
                if period not in ('15m','30m','60m'):continue
                with f.open('rb') as stream:
                    stream.seek(max(0,f.stat().st_size-8192));lines=stream.read().splitlines()
                if not lines:continue
                end=next(csv.reader([lines[-1].decode()]))[0];key=f'{f.parent.name}/{period[:-1]}/{contract}'
                if key not in prior or end>prior[key]['end']:
                    candidates.append(dict(key=key,last_bar_start=end,reason='absent_from_recorded_sources' if key not in prior else 'later_than_recorded_source',provenance_verified=False))
            except (ValueError,UnicodeError,IndexError):candidates.append(dict(path=str(f),reason='inventory_unreadable',provenance_verified=False))
    inventory=json.loads((encoder/'cache/test_inventory.json').read_text())
    return dict(independent_holdout=False,evaluation_scope='reused_research_endpoints',
        test_end_min=min(x['end'] for x in inventory),test_end_max=max(x['end'] for x in inventory),
        test_endpoints=len(inventory),pretraining_boundaries=ref.get('boundaries'),recorded_sources=len(prior),
        source_manifest_sha256=sha256(lm) if lm.exists() else None,
        unverified_raw_candidates=candidates,raw_inventory_root=str(Path(raw_root).resolve()) if raw_root else None,
        raw_inventory_available=bool(raw_root and Path(raw_root).exists()),
        limitations='Raw filename/tail inventory only, not verified independent data. Do not ingest candidates or label unused contracts/time as held out without full lineage audit. No new samples are evaluated here.')


def prepare(meta,out,raw_root=None):
    parent=Path(meta['parent_run']);old=json.loads((parent/'manifest.json').read_text());encoder=Path(meta['encoder_run'])
    cache=out/'cache';cache.mkdir(exist_ok=True);index=cache/'index.json'
    if index.exists():
        saved=json.loads(index.read_text())
        if saved['manifest']!=meta:raise ValueError('Transfer cache identity mismatch')
        verify_files(cache,saved['files']);return
    files={};counts={}
    def save(name,value):
        np.save(cache/name,value,allow_pickle=False);files[name]=sha256(cache/name)
    for split in da.SPLITS:
        specs=json.loads((encoder/f'cache/{split}_sequences.json').read_text())
        n=len(np.load(parent/f'cache/{old["experiments"][0]["name"]}_{split}_z.npy',mmap_mode='r'))
        y,mask,current,stats=targets_from_bank(np.load(encoder/f'cache/{split}_x.npy',mmap_mode='r'),specs,n)
        for suffix,value in [('y',y),('mask',mask),('current',current),('statistics',stats)]:save(f'{split}_{suffix}.npy',value)
        for job in old['experiments']:
            z=np.load(parent/f'cache/{job["name"]}_{split}_z.npy',allow_pickle=False)
            if z.ndim!=2 or len(z)!=n or not np.isfinite(z).all():raise ValueError('Frozen embedding alignment mismatch')
            save(f'{split}_{job["name"]}.npy',z);save(f'{split}_{job["name"]}_plus_statistics.npy',np.column_stack((z,stats)))
        counts[split]=dict(endpoints=n,support=mask.sum(0).tolist(),complete_cases=int(mask.all(1).sum()))
    (cache/'test_inventory.json').write_bytes((encoder/'cache/test_inventory.json').read_bytes())
    files['test_inventory.json']=sha256(cache/'test_inventory.json')
    mean,scale,active=target_stats(np.load(cache/'train_y.npy'),np.load(cache/'train_mask.npy'))
    atomic_json(dict(names=NAMES,mean=mean.tolist(),scale=scale.tolist(),active=active.tolist()),cache/'target_statistics.json')
    files['target_statistics.json']=sha256(cache/'target_statistics.json')
    audit=coverage_audit(old,parent,raw_root);atomic_json(audit,out/'coverage_audit.json')
    atomic_json(dict(names=NAMES,splits=counts,scope='Targets use last128 observed rows within each partition/contract. Internal h-1 returns; no look-ahead.'),out/'target_audit.json')
    atomic_json(dict(manifest=meta,files=files),index)


def load_arrays(out,name):
    return [[np.load(out/f'cache/{s}_{k}.npy',allow_pickle=False) for s in da.SPLITS] for k in (name,'y','mask')]


def target_stats(y,mask):
    y=np.asarray(y,dtype=np.float64)
    count=mask.sum(0);mean=np.where(mask,y,0).sum(0)/count.clip(1)
    std=np.sqrt(np.where(mask,(y-mean)**2,0).sum(0)/count.clip(1));active=(count>=2)&(std>1e-8)
    return mean,std.clip(1e-5),active


def masked_mean(err,mask):
    count=mask.sum(0);active=count>0
    if not bool(active.any()):raise ValueError('No valid readout targets')
    return ((err*mask).sum(0)/count.clamp_min(1))[active].mean()


def score_predictions(pred,ys,masks):
    mean,scale,active=target_stats(ys[0],masks[0]);truth=(ys[2]-mean)/scale
    valid=masks[2]&active;errors=np.where(valid,(pred-truth)**2,np.nan);rows=[]
    for j in range(len(NAMES)):
        ok=valid[:,j];n=int(ok.sum());var=float(np.var(truth[ok,j])) if n else 0.
        rows.append(dict(name=NAMES[j],support=n,mse=float(errors[ok,j].mean()) if n else None,
            r2=1-float(errors[ok,j].mean())/var if n>=2 and var>1e-12 else None))
    return rows,errors


def train_probe_epoch(model,opt,x,y,mask,batch):
    if x.device.type!='cuda':raise ValueError('Formal MLP fitting requires AutoDL CUDA')
    model.train();order=torch.randperm(len(x),device=x.device);total=0.;updates=0
    for ids in order.split(batch):
        if not bool(mask[ids].any()):continue
        loss=masked_mean(nn.functional.smooth_l1_loss(model(x[ids]),y[ids],reduction='none'),mask[ids])
        if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        total+=float(loss.detach());updates+=1
    return total/max(updates,1)


def mlp_trial(xs,ys,masks,path,metadata,seed,decay,epochs,batch,hidden,lr,device='cuda'):
    torch.manual_seed(seed)
    xm,xscl=xs[0].mean(0),xs[0].std(0).clip(1e-5);ym,yscl,active=target_stats(ys[0],masks[0])
    # Selection receives train/validation only; test predictions happen after decay selection.
    x=[torch.tensor((v-xm)/xscl,dtype=torch.float32,device=device) for v in xs[:2]]
    y=[torch.tensor((v-ym)/yscl,dtype=torch.float32,device=device) for v in ys[:2]]
    mask=[torch.tensor(v&active,device=device) for v in masks[:2]]
    model=nn.Sequential(nn.Linear(x[0].shape[1],hidden),nn.GELU(),nn.Linear(hidden,len(NAMES))).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
    def validate():
        model.eval()
        with torch.no_grad():score=float(masked_mean((model(x[1])-y[1])**2,mask[1]))
        if not np.isfinite(score):raise ValueError('Nonfinite probe validation')
        return score
    path.mkdir(exist_ok=True)
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Probe resume settings changed')
        if state['epoch']>epochs or [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)):raise ValueError('Probe history incomplete')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    else:
        state=dict(metadata=metadata,epoch=0,history=[],best_epoch=0,best_score=validate(),
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    for e in range(state['epoch']+1,epochs+1):
        loss=train_probe_epoch(model,opt,x[0],y[0],mask[0],batch)
        score=validate()
        if score<state['best_score']:state.update(best_epoch=e,best_score=score,best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        state['history'].append(dict(epoch=e,train_batch_mean_loss=loss,validation_mse=score))
        state.update(epoch=e,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        atomic_save(state,path/'last.pt');atomic_json(state['history'],path/'history.json')
        if e%20==0 or e==epochs:
            atomic_json(dict(probe_seed=seed,decay=decay,epoch=e,epochs=epochs,validation_mse=score,best_epoch=state['best_epoch']),path.parent/'progress.json')
            progress(f'{path.parent.name} probe seed={seed} decay={decay} epoch={e}/{epochs} val={score:.4f}')
    report=dict(selected_epoch=state['best_epoch'],trained_selection=state['best_epoch']>0,validation_mse=state['best_score'],probe_seed=seed,decay=decay,
        parameters=sum(p.numel() for p in model.parameters()))
    atomic_save(dict(metadata=metadata,model=state['best_model'],normalizers=dict(x_mean=xm.tolist(),x_scale=xscl.tolist(),y_mean=ym.tolist(),y_scale=yscl.tolist()),report=report),path/'best.pt')
    return report


@torch.no_grad()
def mlp_errors(path,xs,ys,masks,hidden,device):
    ck=torch.load(path/'best.pt',map_location='cpu',weights_only=True);norm=ck['normalizers']
    model=nn.Sequential(nn.Linear(xs[0].shape[1],hidden),nn.GELU(),nn.Linear(hidden,len(NAMES))).to(device).eval()
    model.load_state_dict(ck['model'])
    x=torch.tensor((xs[2]-np.array(norm['x_mean']))/np.array(norm['x_scale']),dtype=torch.float32,device=device)
    return score_predictions(model(x).cpu().numpy(),ys,masks)


def worker(out,name,device='cuda'):
    meta=json.loads((out/'manifest.json').read_text());index=json.loads((out/'cache/index.json').read_text())
    if name not in [j['name'] for j in meta['experiments']] or index['manifest']!=meta:raise ValueError('Unknown probe/cache')
    verify_files(out/'cache',index['files']);path=out/name;path.mkdir(exist_ok=True)
    if (path/'completion.json').exists():
        done=json.loads((path/'completion.json').read_text())
        if done['manifest']!=meta:raise ValueError('Completed probe metadata mismatch')
        verify_files(path,done['files']);return
    xs,ys,masks=load_arrays(out,name);report=dict(input_dimensions=xs[0].shape[1],scope=GOAL['limits'])
    value,err=aa.fit_activity_probe(xs,ys,masks);value.update(names=NAMES,scope=GOAL['limits'])
    report['linear']=value;np.save(path/'linear_errors.npy',err);window_errors={'linear':np.where(np.isfinite(err),err,None).tolist()}
    for seed in meta['probe_seeds']:
        best=None
        for decay in meta['decays']:
            trial=path/f'p{seed}_wd{decay}';metadata=dict(manifest=meta,name=name,probe_seed=seed,decay=decay)
            value=mlp_trial(xs,ys,masks,trial,metadata,seed,decay,meta['epochs'],meta['batch'],meta['hidden'],meta['lr'],device)
            if best is None or value['validation_mse']<best[0]['validation_mse']:best=(value,trial)
        rows,err=mlp_errors(best[1],xs,ys,masks,meta['hidden'],device)
        key=f'mlp_s{seed}';report[key]=dict(best[0],targets=rows);np.save(path/f'{key}_errors.npy',err)
        window_errors[key]=np.where(np.isfinite(err),err,None).tolist()
    atomic_json(dict(names=NAMES,errors=window_errors),path/'per_window_errors.json')
    atomic_json(report,path/'metrics.json')
    files={p.name:sha256(p) for p in path.iterdir() if p.name in ('metrics.json','per_window_errors.json') or p.name.endswith('_errors.npy')}
    atomic_json(dict(status='complete',manifest=meta,files=files),path/'completion.json')
    atomic_json(dict(status='complete',representation=name),path/'progress.json')


def error_summary(err):
    summary={}
    for name,ids in [('primary',list(range(len(NAMES))))]+[(f,list(range(i*3,i*3+3))) for i,f in enumerate(FAMILIES)]:
        valid=np.isfinite(err[:,ids]).all(1);values=err[valid][:,ids]
        summary[name]=dict(support=int(valid.sum()),mse=float(values.mean()) if len(values) else None)
    return summary


def paired(a,b,weeks):
    result={}
    for name,ids in [('primary',list(range(len(NAMES))))]+[(f,list(range(i*3,i*3+3))) for i,f in enumerate(FAMILIES)]:
        valid=np.isfinite(a[:,ids]).all(1)&np.isfinite(b[:,ids]).all(1)
        if valid.sum()<2:result[name]=dict(support=int(valid.sum()));continue
        row=paired_error_interval(a[valid][:,ids].mean(1),b[valid][:,ids].mean(1),weeks[valid])
        row['delta_mse']=row.pop('delta_close_mae_bps');row['support']=int(valid.sum());result[name]=row
    return result


def evaluate(out):
    meta=json.loads((out/'manifest.json').read_text());encoder=Path(meta['encoder_run'])
    inventory=json.loads((encoder/'cache/test_inventory.json').read_text());weeks=np.array([x['week'] for x in inventory])
    kinds=['linear']+[f'mlp_s{s}' for s in meta['probe_seeds']];errors={}
    report=dict(schema=SCHEMA,goal=GOAL,coverage=json.loads((out/'coverage_audit.json').read_text()),variants={},paired={},
        inherited_price_screen=json.loads((Path(meta['parent_run'])/'readapt_metrics.json').read_text())['stage_screen'],
        limits='Primary aggregate uses complete-case12-task support. Family contrasts and R2 are exploratory; weekly intervals do not correct all multiple comparisons. MLP seeds are readout replicates, not independent encoders.')
    for job in meta['experiments']:
        name=job['name'];done=json.loads((out/name/'completion.json').read_text())
        if done['status']!='complete' or done['manifest']!=meta:raise ValueError('Incomplete transfer worker')
        verify_files(out/name,done['files']);v=json.loads((out/name/'metrics.json').read_text());errors[name]={}
        for kind in kinds:
            err=np.load(out/name/f'{kind}_errors.npy');errors[name][kind]=err
            if err.shape!=(len(weeks),len(NAMES)):raise ValueError('Probe test shape mismatch')
            v[kind]['summary']=error_summary(err)
        report['variants'][name]=v
    contrasts=[]
    for s in meta['seeds']:
        a,b=f'aux020_s{s}',f'control_s{s}'
        contrasts.extend([(a,b),(a,'current'),(a,'statistics'),(a+'_plus_statistics','statistics'),
                          (b+'_plus_statistics','statistics'),(a+'_plus_statistics',b+'_plus_statistics')])
    for a,b in contrasts:report['paired'][a+'_minus_'+b]={k:paired(errors[a][k],errors[b][k],weeks) for k in kinds}
    def support(a,b):
        rows=report['paired'][a+'_minus_'+b]
        return all(rows[k]['primary'].get('support',0)>=50 and rows[k]['primary'].get('high',1)<0 and
            (k=='linear' or (report['variants'][a][k]['trained_selection'] and report['variants'][b][k]['trained_selection'])) for k in kinds)
    tradeoffs=[]
    for s in meta['seeds']:
        for suffix in ('','_plus_statistics'):
            comparison=f'aux020_s{s}{suffix}_minus_control_s{s}{suffix}'
            for kind,groups in report['paired'][comparison].items():
                for family in FAMILIES:
                    row=groups[family]
                    if row.get('support',0)>=50 and row.get('low',-1)>0:
                        tradeoffs.append(dict(comparison=comparison,readout=kind,family=family,interval=row))
    report['evidence_screen']=dict(primary_improvement_across_both_encoder_seeds=all(support(f'aux020_s{s}',f'control_s{s}') for s in meta['seeds']),
        auxiliary_increment_over_statistics=all(support(f'aux020_s{s}_plus_statistics','statistics') for s in meta['seeds']),
        exploratory_family_regressions=tradeoffs,fresh_holdout_confirmed=False,automatic_promotion=False)
    atomic_json(report,out/'transfer_metrics.json')
    lines=['# Frozen-state exploratory transfer','',GOAL['stage'],'',GOAL['limits'],'',
           '| representation | input dim | linear MSE | MLP1701 MSE | MLP1702 MSE | complete cases |',
           '|---|---:|---:|---:|---:|---:|']
    for name,v in report['variants'].items():
        vals=[v[k]['summary']['primary'] for k in kinds]
        text=['—' if x['mse'] is None else f'{x["mse"]:.4f}' for x in vals]
        lines.append(f'| {name} | {v["input_dimensions"]} | '+ ' | '.join(text)+f' | {vals[0]["support"]} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n\n## Exploratory signals\n'+json.dumps(report['evidence_screen'],indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--parent');p.add_argument('--name');p.add_argument('--root',default='data/contracts')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=256);p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();out=Path(a.out).resolve()
    if not torch.cuda.is_available():raise ValueError('Formal readout experiments require AutoDL CUDA')
    torch.set_num_threads(4)
    if a.action=='worker':
        if not a.name:p.error('--name required')
        worker(out,a.name);return
    if not a.parent:p.error('--parent required')
    if not 1<=a.jobs<=4:raise ValueError('jobs must be1..4')
    parent=Path(a.parent).resolve();old,identity=source_identity(parent)
    if parent==out or parent in out.parents or out in parent.parents:raise ValueError('Separate sibling directory required')
    meta=make_manifest(parent,old,identity,a.epochs,a.batch)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Settings changed; use another directory')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output lacks manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    atomic_json(dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),jobs=a.jobs),out/'runtime.json')
    progress('Preparing fixed-state transfer targets and baselines; encoder training disabled')
    prepare(meta,out,a.root)
    if a.action=='all':run_jobs(out,a.jobs,'obson.babel.state_transfer')
    evaluate(out);verify_files(out/'cache',json.loads((out/'cache/index.json').read_text())['files'])
    if source_identity(parent)[1]!=identity:raise ValueError('Frozen sources changed')
    atomic_json(dict(status='complete',representations=len(meta['experiments']),source_unchanged=True,independent_holdout=False,goal=GOAL),out/'completion.json')
    progress('Transfer evaluation complete; exploratory reused-data evidence, no automatic promotion')


if __name__=='__main__':main()
