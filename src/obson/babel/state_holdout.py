"""Score-blind cross-symbol audit, then locked-state evaluation; no SGD updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import state_transfer as st, price_readapt as pr, activity_alignment as al
from . import activity_ablation as aa, detail_alignment as da
from .ae_context import encode_context
from .ae_extend import atomic_json, atomic_save
from .ar_reconstruction import paired_error_interval
from .data import load_series
from .dual_state import sha256, verify_files
from .history_autoencoder import raw_targets
from .holdout_audit import CANDIDATES, read_json, scan_lineage, audit_raw
from .progress import progress
from .representation import time_mask

SCHEMA = 'babel-state-holdout-v1'
KINDS = ('linear', 'mlp_s1701', 'mlp_s1702')


def safe(value):
    if isinstance(value, dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [safe(v) for v in value]
    if isinstance(value, np.ndarray):return safe(value.tolist())
    if isinstance(value, (float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):return int(value)
    return value


def locked_ridge(x, y, mask, rows):
    """Restore selected closed-form fits. Does not accept validation or holdout data."""
    result = []
    for j,row in enumerate(rows):
        if 'alpha' not in row:
            result.append(None);continue
        ok = mask[:,j];xx = x[ok].astype(float);yy = y[ok,j].astype(float)
        mean,scale=xx.mean(0),xx.std(0).clip(1e-5);ym,ys=yy.mean(),max(yy.std(),1e-5)
        xx=np.column_stack(((xx-mean)/scale,np.ones(len(xx))))
        penalty=np.eye(xx.shape[1])*row['alpha'];penalty[-1,-1]=0
        weight=np.linalg.solve(xx.T@xx+penalty,xx.T@((yy-ym)/ys))
        result.append(dict(mean=mean.tolist(),scale=scale.tolist(),ym=float(ym),ys=float(ys),
            weight=weight.tolist(),alpha=row['alpha']))
    return result


def ridge_errors(model,x,y,mask):
    errors=np.full(y.shape,np.nan)
    for j,row in enumerate(model):
        if row is None:continue
        ok=mask[:,j]
        xx=np.column_stack(((x[ok]-row['mean'])/row['scale'],np.ones(int(ok.sum()))))
        errors[ok,j]=(xx@row['weight']-(y[ok,j]-row['ym'])/row['ys'])**2
    return errors


def check_ridge(model,x,y,mask,rows):
    err=ridge_errors(model,x,y,mask)
    for j,row in enumerate(rows):
        if 'test_normalized_mse' in row and not np.isclose(np.nanmean(err[:,j]),row['test_normalized_mse'],rtol=2e-5,atol=1e-7):
            raise ValueError(f'Locked Ridge old-test replay failed target {j}')


@torch.no_grad()
def mlp_errors(ck,x,y,mask,hidden,device):
    norm=ck['normalizers'];model=nn.Sequential(nn.Linear(x.shape[1],hidden),nn.GELU(),nn.Linear(hidden,y.shape[1])).to(device)
    model.load_state_dict(ck['model']);model.eval();pred=[]
    for left in range(0,len(x),256):
        batch=torch.tensor((x[left:left+256]-norm['x_mean'])/norm['x_scale'],dtype=torch.float32,device=device)
        pred.append(model(batch).cpu().numpy())
    error=(np.concatenate(pred)-(y-norm['y_mean'])/norm['y_scale'])**2
    return np.where(mask,error,np.nan)


def freeze_readouts(meta,out,device):
    transfer=Path(meta['transfer']);old=read_json(transfer/'manifest.json');price=Path(old['parent_run']);encoder=Path(old['encoder_run'])
    dest=out/'frozen';dest.mkdir(exist_ok=True);index=dest/'index.json'
    if index.exists():
        info=read_json(index)
        if info['manifest']!=meta:raise ValueError('Frozen bundle settings changed')
        verify_files(dest,info['files']);return
    report=read_json(transfer/'transfer_metrics.json');files={};replays={}
    for job in old['experiments']:
        name=job['name'];progress(f'freeze: restoring fixed transfer readouts {name}')
        xs,ys,masks=st.load_arrays(transfer,name);rows=report['variants'][name]['linear']['targets']
        ridge=locked_ridge(xs[0],ys[0],masks[0],rows);check_ridge(ridge,xs[2],ys[2],masks[2],rows)
        atomic_json(ridge,dest/f'{name}_ridge.json')
        old_errors=read_json(transfer/name/'per_window_errors.json')['errors']
        for kind in KINDS[1:]:
            r=report['variants'][name][kind];path=transfer/name/f'p{r["probe_seed"]}_wd{r["decay"]}'/'best.pt'
            ck=torch.load(path,map_location='cpu',weights_only=True)
            if ck['report']['selected_epoch']!=r['selected_epoch'] or ck['metadata']!=dict(manifest=old,name=name,probe_seed=r['probe_seed'],decay=r['decay']):
                raise ValueError('Selected MLP identity mismatch')
            err=mlp_errors(ck,xs[2],ys[2],masks[2],old['hidden'],device)
            if not np.allclose(err,np.array(old_errors[kind],float),rtol=5e-4,atol=2e-5,equal_nan=True):
                raise ValueError('Selected MLP old-test replay mismatch')
            atomic_save(ck,dest/f'{name}_{kind}.pt')
        replays[name]=dict(ridge=True,mlp=True)
    prior=read_json(encoder/'last_diagnostic/alignment_metrics.json')
    ys=[np.load(encoder/f'cache/{s}_activity.npy') for s in da.SPLITS]
    masks=[np.load(encoder/f'cache/{s}_activity_mask.npy') for s in da.SPLITS]
    for name in meta['encoder_names']+['current_input']:
        xs=[np.load(encoder/f'cache/{s}_current.npy') if name=='current_input' else
            np.load(price/f'cache/{name}_{s}_z.npy') for s in da.SPLITS]
        rows=(prior['current_input_control']['linear'] if name=='current_input' else prior['variants'][name]['linear_activity_readout'])['targets']
        model=locked_ridge(xs[0],ys[0],masks[0],rows);check_ridge(model,xs[2],ys[2],masks[2],rows)
        atomic_json(model,dest/f'{name}_activity_ridge.json');replays[name+'_activity']=True
    for path in dest.iterdir():
        if path.name!='index.json':files[path.name]=sha256(path)
    atomic_json(replays,out/'readout_replay.json');atomic_json(dict(manifest=meta,files=files),index)


def endpoint_keys(series,bounds,cap):
    groups={}
    for i,s in enumerate(series):
        valid=time_mask(s,bounds,'test');bad=np.r_[0,np.cumsum(~valid)]
        for j in range(127,len(s.frame),128):
            if valid[j] and s.main[j] and bad[j+1]==bad[j-127]:
                groups.setdefault((s.code,s.period),[]).append((i,j))
    keys=[]
    for group,items in sorted(groups.items()):
        items.sort(key=lambda ij:(series[ij[0]].frame.datetime.iloc[ij[1]],series[ij[0]].key))
        ids=np.linspace(0,len(items)-1,min(cap,len(items)),dtype=int)
        keys.extend(items[k] for k in ids)
    if not keys:raise ValueError('No eligible test endpoints')
    return np.array(keys,dtype=np.int64)


def prepare_holdout(meta,out,audit):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/'index.json').exists():
        info=read_json(cache/'index.json')
        if info['manifest']!=meta or info['raw_hashes']!=audit['raw_hashes']:raise ValueError('Holdout cache identity changed')
        verify_files(cache,info['files']);return
    expected={r['path'] for r in audit.get('files',[]) if r.get('symbol') in audit['eligible_symbols']}
    if expected:
        actual={str(p.resolve()) for s in audit['eligible_symbols'] for period in (15,30,60) for p in (Path(meta['root'])/s).glob(f'*_{period}m.csv')}
        if actual!=expected:raise ValueError('Candidate file inventory changed after audit')
        for p in expected:
            if sha256(p)!=audit['raw_hashes'][p]:raise ValueError('Candidate content changed after audit')
    progress('holdout: generating causal features; GPU may be idle')
    series,_=load_series(meta['root'],audit['eligible_symbols'],(15,30,60))
    encoded=[];activity_audit=[]
    for s in series:
        e=encode_context(s.frame,s.period,'ema8_32');act,stats=aa.activity_features(s.frame,s.period)
        encoded.append(dict(x=np.column_stack((e['x'],act))));activity_audit.append(dict(key=s.key,**stats))
    keys=endpoint_keys(series,meta['boundaries'],meta['cap']);n=len(keys)
    x,specs=da.sequence_specs(series,encoded,meta['boundaries'],'test',keys)
    y,mask,current,statistics=st.targets_from_bank(x,specs,n)
    ay,am,clean,ac=al.targets_from_bank(x,specs,n)
    py=[];pm=[];inventory=[];raw={i:raw_targets(series[i].frame,series[i].period) for i in np.unique(keys[:,0])}
    for i,end in keys:
        s=series[i];anchor=float(s.frame.close.iloc[end-64]);target=raw[i][end-63:end+1].copy();target[:,:2]-=np.log(anchor)*100
        valid=np.ones_like(target,bool);valid[:,5]=s.frame.oi_available.iloc[end-63:end+1];target[:,5]=np.where(valid[:,5],target[:,5],0.)
        py.append(target);pm.append(valid)
        inventory.append(dict(key=s.key,symbol=s.code,period=s.period,row=int(end),end=str(s.frame.datetime.iloc[end]),
            anchor=anchor,week=str(pd.Timestamp(s.sessions[end]).to_period('W-SUN')),month=str(s.frame.datetime.iloc[end].to_period('M'))))
    arrays=dict(x=x,y=y,mask=mask,current=current,statistics=statistics,activity=ay,activity_mask=am,
        clean=clean,current_input=ac,price_y=np.array(py,np.float32),price_mask=np.array(pm,bool))
    files={}
    for name,value in arrays.items():
        np.save(cache/f'test_{name}.npy',value);files[f'test_{name}.npy']=sha256(cache/f'test_{name}.npy')
    for name,value in [('test_sequences.json',specs),('test_inventory.json',inventory),('activity_audit.json',activity_audit)]:
        atomic_json(value,cache/name);files[name]=sha256(cache/name)
    atomic_json(dict(endpoints=n,symbols=audit['eligible_symbols'],transfer_support=mask.sum(0).tolist(),
        activity_support=am.sum(0).tolist(),clean_endpoints=int(clean.sum()),activity_anomalies=activity_audit,
        endpoint_policy='Stride128 within contract, same old test session bounds, deterministic cap per symbol/period; no labels/scores used.'),out/'coverage.json')
    atomic_json(dict(manifest=meta,raw_hashes=audit['raw_hashes'],files=files),cache/'index.json')


def grouped_errors(err,truth,inventory,names):
    groups={'all':np.arange(len(err))}
    for field in ('symbol','period','month'):
        for key in sorted({str(r[field]) for r in inventory}):
            groups[f'{field}/{key}']=np.array([i for i,r in enumerate(inventory) if str(r[field])==key])
    result={}
    for group,ids in groups.items():
        e=err[ids];y=truth[ids];rows=[]
        for j,name in enumerate(names):
            ok=np.isfinite(e[:,j]);n=int(ok.sum());var=float(np.var(y[ok,j])) if n else 0
            rows.append(dict(name=name,support=n,mse=float(e[ok,j].mean()) if n else None,
                r2=1-float(e[ok,j].mean())/var if n>=2 and var>1e-12 else None))
        good=np.isfinite(e).all(1)
        result[group]=dict(primary_support=int(good.sum()),primary_mse=float(e[good].mean()) if good.any() else None,targets=rows)
    return result


def contrast(a,b,inventory,columns):
    groups={'all':np.arange(len(a))}
    for field in ('symbol','period','month'):
        for key in sorted({str(r[field]) for r in inventory}):groups[f'{field}/{key}']=np.array([i for i,r in enumerate(inventory) if str(r[field])==key])
    result={}
    for group,ids in groups.items():
        x=a[ids][:,columns];y=b[ids][:,columns];ok=np.isfinite(x).all(1)&np.isfinite(y).all(1);ids=ids[ok]
        row=dict(support=len(ids),supported=len(ids)>=50)
        if len(ids):
            row.update(paired_error_interval(x[ok].mean(1),y[ok].mean(1),np.array([inventory[i]['week'] for i in ids])))
            row['delta_mse']=row.pop('delta_close_mae_bps')
            row['interpretation']='Negative favors candidate; paired calendar-week bootstrap within audited cross-symbol sample, not independent-symbol replication.'
        row['supported']=row['supported'] and row.get('weeks',0)>=5
        result[group]=row
    return result


def evidence_screen(report,seeds):
    rows=[];regressions=[]
    def favorable(row):
        return row.get('supported',False) and row.get('high') is not None and row['high']<0
    for seed in seeds:
        a,b=f'aux020_s{seed}',f'control_s{seed}'
        direct=report['paired'][a+'_minus_'+b]
        incremental=report['paired'][a+'_plus_statistics_minus_statistics']
        price_a=report['price'][a]['objectives'];price_b=report['price'][b]['objectives']
        rows.append(dict(seed=seed,
            transfer_primary_supported=all(favorable(direct[k]['primary']['all']) for k in KINDS),
            increment_over_statistics_supported=all(favorable(incremental[k]['primary']['all']) for k in KINDS),
            activity_past_supported=all(favorable(report['paired'][a+'_activity_minus_'+r]['all']) for r in (b,'current_input')),
            price_retained=all(price_a[k]<=price_b[k]*1.02+1e-8 for k in ('close_bps','detail')),
            price_ratios={k:price_a[k]/price_b[k] if price_b[k] else None for k in ('close_bps','detail')}))
        for kind in KINDS:
            for family in st.FAMILIES:
                r=direct[kind][family]['all']
                if r.get('supported') and r.get('low') is not None and r['low']>0:
                    regressions.append(dict(seed=seed,readout=kind,family=family,interval=r))
    return dict(seeds=rows,exploratory_family_regressions=regressions,automatic_promotion=False,
        interpretation='Evidence in registry-audited cross-symbol sample only; old gate failures remain. No all-purpose certification, family comparisons uncorrected.')


@torch.no_grad()
def evaluate(meta,out,device):
    transfer=Path(meta['transfer']);tm=read_json(transfer/'manifest.json');price=Path(tm['parent_run']);pm=read_json(price/'manifest.json')
    encoder=Path(tm['encoder_run']);em=read_json(encoder/'manifest.json');cache=out/'cache';frozen=out/'frozen'
    inventory=read_json(cache/'test_inventory.json');n=len(inventory)
    arr={k:np.load(cache/f'test_{k}.npy') for k in ('y','mask','current','statistics','activity','activity_mask','clean','current_input','price_y','price_mask')}
    scales=read_json(price/'cache/replay.json')['scales'];features={k:arr[k] for k in ('current','statistics')}
    report=dict(schema=SCHEMA,scope='Audited cross-symbol holdout within recorded provenance; not a new-time holdout or a certificate about unrecorded research.',
        automatic_promotion=False,encoder_updates=0,transfer={},activity={},price={},paired={},
        inherited_price_screen=read_json(price/'readapt_metrics.json')['stage_screen'],
        limits='Statistics MLP selected at97/100 epochs in prior budget; no convergence claim. Old activity MLP not persisted: only locked Ridge evaluated for activity. No state-classification head or long-encoder evaluation in this run.')
    price_errors={};activity_errors={};transfer_errors={}
    for job in pm['experiments']:
        name=job['name'];progress(f'holdout: replaying frozen encoder and price head {name}')
        ck=torch.load(encoder/name/'last.pt',map_location='cpu',weights_only=True)
        if ck['metadata']!=dict(manifest=em,job=job) or ck['epoch']!=em['epochs']:raise ValueError('Encoder checkpoint mismatch')
        model=da.DetailModel(**em['config'],input_dim=28)
        model.activity_head=nn.Sequential(nn.LayerNorm(em['config']['width']),nn.Linear(em['config']['width'],20))
        model.load_state_dict(ck['model']);del ck
        head=torch.load(price/name/'best.pt',map_location='cpu',weights_only=True)
        if head['metadata']!=dict(manifest=pm,job=job):raise ValueError('Price head mismatch')
        model.head.load_state_dict(head['model']);model.to(device).eval().requires_grad_(False)
        data={k:torch.tensor(v,device=device) for k,v in dict(z=np.zeros((n,em['config']['width']),np.float32),y=arr['price_y'],mask=arr['price_mask']).items()}
        score,z,pred=da.run_epoch(model,data,aa.ActivityStreams(cache,'test',job),True,True,scales,meta['streams'],collect=True)
        summary,rows=da.describe_predictions(arr['price_y'],pred,inventory)
        detail=da.detail_values(torch.tensor(pred),torch.tensor(arr['price_y']),scales).numpy()
        price_errors[name]=np.column_stack((np.abs(pred[:,:,1]-arr['price_y'][:,:,1]).mean(1)*100,detail))
        report['price'][name]=dict(objectives=score,diagnostics=summary,head_epoch=head['epoch'])
        atomic_json(safe(rows),out/f'{name}_price_windows.json')
        features[name]=z;features[name+'_plus_statistics']=np.column_stack((z,arr['statistics']))
        del model,data
    for name in meta['encoder_names']+['current_input']:
        x=arr['current_input'] if name=='current_input' else features[name]
        ridge=read_json(frozen/f'{name}_activity_ridge.json');err=ridge_errors(ridge,x,arr['activity'],arr['activity_mask']);activity_errors[name]=err
        truth=np.column_stack([(arr['activity'][:,j]-r['ym'])/r['ys'] if r else np.zeros(n) for j,r in enumerate(ridge)])
        report['activity'][name]=dict(groups=grouped_errors(err,truth,inventory,al.NAMES),clean=al.error_summary(err,arr['clean']))
    stats=read_json(transfer/'cache/target_statistics.json');truth=(arr['y']-stats['mean'])/stats['scale'];valid=arr['mask']&np.array(stats['active'])
    for job in tm['experiments']:
        name=job['name'];progress(f'holdout: fixed transfer heads {name}');x=features[name];transfer_errors[name]={};report['transfer'][name]={}
        for kind in KINDS:
            err=(ridge_errors(read_json(frozen/f'{name}_ridge.json'),x,arr['y'],valid) if kind=='linear' else
                mlp_errors(torch.load(frozen/f'{name}_{kind}.pt',map_location='cpu',weights_only=True),x,arr['y'],valid,tm['hidden'],device))
            transfer_errors[name][kind]=err
            report['transfer'][name][kind]=dict(summary=st.error_summary(err),groups=grouped_errors(err,truth,inventory,st.NAMES))
    for seed in tm['seeds']:
        a,b=f'aux020_s{seed}',f'control_s{seed}'
        for left,right in [(a,b),(a,'statistics'),(a+'_plus_statistics','statistics'),(a+'_plus_statistics',b+'_plus_statistics')]:
            report['paired'][left+'_minus_'+right]={k:{'primary':contrast(transfer_errors[left][k],transfer_errors[right][k],inventory,list(range(12))),
                **{f:contrast(transfer_errors[left][k],transfer_errors[right][k],inventory,list(range(i*3,i*3+3))) for i,f in enumerate(st.FAMILIES)}} for k in KINDS}
        report['paired'][a+'_price_minus_'+b]={name:contrast(price_errors[a],price_errors[b],inventory,[i]) for i,name in enumerate(('close_bps','detail'))}
        for right in (b,'current_input'):
            report['paired'][a+'_activity_minus_'+right]=contrast(activity_errors[a],activity_errors[right],inventory,list(range(5,20)))
    report['evidence_screen']=evidence_screen(report,tm['seeds'])
    atomic_json(safe(report),out/'holdout_metrics.json')
    atomic_json(safe(dict(transfer=transfer_errors,activity=activity_errors,price=price_errors)),out/'per_window_errors.json')
    lines=['# Audited cross-symbol holdout','',report['scope'],'','No automatic promotion. Readout budget is fixed, not proven converged.','',
        '| representation | Ridge MSE | MLP1701 MSE | MLP1702 MSE | complete cases |','|---|---:|---:|---:|---:|']
    for name,ks in report['transfer'].items():
        vals=[ks[k]['summary']['primary'] for k in KINDS]
        lines.append('| '+name+' | '+' | '.join('—' if v['mse'] is None else f'{v["mse"]:.4f}' for v in vals)+f' | {vals[0]["support"]} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def sources(transfer):
    tm=read_json(transfer/'manifest.json')
    if tm['schema']!=st.SCHEMA or read_json(transfer/'completion.json').get('status')!='complete':raise ValueError('Completed transfer run required')
    _,identity=st.source_identity(Path(tm['parent_run']))
    if identity!=tm['parent_identity']:raise ValueError('Transfer parent changed')
    verify_files(transfer/'cache',read_json(transfer/'cache/index.json')['files'])
    paths=[transfer/'manifest.json',transfer/'completion.json',transfer/'transfer_metrics.json',transfer/'cache/index.json']
    price=Path(tm['parent_run']);encoder=Path(tm['encoder_run']);pm=read_json(price/'manifest.json')
    for job in pm['experiments']:paths.extend([encoder/job['name']/'last.pt',price/job['name']/'best.pt'])
    for job in tm['experiments']:
        name=job['name'];done=read_json(transfer/name/'completion.json')
        if done['manifest']!=tm or done['status']!='complete':raise ValueError('Transfer worker incomplete')
        verify_files(transfer/name,done['files']);paths.extend([transfer/name/'metrics.json',transfer/name/'per_window_errors.json'])
        for kind in KINDS[1:]:
            r=read_json(transfer/name/'metrics.json')[kind]
            paths.append(transfer/name/f'p{r["probe_seed"]}_wd{r["decay"]}'/'best.pt')
    return tm,{str(p.resolve()):sha256(p) for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('all','audit'))
    parser.add_argument('--transfer',required=True);parser.add_argument('--registry',default='checkpoints');parser.add_argument('--root',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--streams',type=int,default=32);parser.add_argument('--cap',type=int,default=256)
    args=parser.parse_args();out=Path(args.out).resolve();transfer=Path(args.transfer).resolve()
    if args.streams<1 or args.cap<1:raise ValueError('Positive streams and cap required')
    if out==transfer or out in transfer.parents or transfer in out.parents:raise ValueError('Separate output directory required')
    torch.set_num_threads(4);tm,identity=sources(transfer);em=read_json(Path(tm['encoder_run'])/'manifest.json')
    ref=read_json(Path(em['long_run'])/'manifest.json')['manifest']
    meta=dict(schema=SCHEMA,transfer=str(transfer),registry=str(Path(args.registry).resolve()),root=str(Path(args.root).resolve()),
        source_identity=identity,boundaries=ref['boundaries'],symbols=list(CANDIDATES),streams=args.streams,cap=args.cap,
        encoder_names=[f'{mode}_s{seed}' for seed in tm['seeds'] for mode in ('control','aux020')],
        protocol='Freeze existing selected heads; recover Ridge with old selected alpha, train only. No SGD, new normalization, calibration or selection on holdout. Separate all seed results.',
        scope='Cross-symbol evaluation within auditable local records, not a new future interval; no automatic promotion.')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Run settings changed; new output required')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output lacks manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    lineage=scan_lineage(args.registry,transfer,out);atomic_json(lineage,out/'lineage_audit.json')
    audit=audit_raw(args.root,lineage);atomic_json(audit,out/'raw_audit.json')
    if not audit['eligible_symbols']:
        atomic_json(dict(status='blocked',reason='No source/quality-qualified symbols',evaluation_performed=False),out/'audit_status.json')
        progress('Audit blocked: no certified candidate within recorded provenance; see lineage_audit.json and raw_audit.json')
        raise SystemExit(3)
    atomic_json(dict(status='eligible',symbols=audit['eligible_symbols'],evaluation_performed=False),out/'audit_status.json')
    if args.action=='audit':return
    if not torch.cuda.is_available():raise ValueError('Run real checkpoint replay on AutoDL CUDA')
    freeze_readouts(meta,out,'cuda')
    # Bind score-blind audit and selection before constructing or scoring held-out endpoints.
    lock=dict(manifest=meta,lineage_sha256=sha256(out/'lineage_audit.json'),raw_audit_sha256=sha256(out/'raw_audit.json'),
        frozen_index_sha256=sha256(out/'frozen/index.json'))
    if (out/'evaluation_lock.json').exists() and read_json(out/'evaluation_lock.json')!=lock:raise ValueError('Holdout lock changed; do not reselect after observing results')
    atomic_json(lock,out/'evaluation_lock.json')
    try:
        prepare_holdout(meta,out,audit)
    except ValueError as error:
        if str(error) != 'No eligible test endpoints':raise
        atomic_json(dict(status='blocked',reason=str(error),evaluation_performed=False),out/'audit_status.json')
        raise SystemExit(3)
    evaluate(meta,out,'cuda')
    if sources(transfer)[1]!=identity:raise ValueError('Source artifacts changed during evaluation')
    for path,digest in {**audit['raw_hashes'],**audit['source_raw_hashes'],**lineage['manifests']}.items():
        if sha256(path)!=digest:raise ValueError(f'Audited source changed: {path}')
    verify_files(out/'frozen',read_json(out/'frozen/index.json')['files']);verify_files(out/'cache',read_json(out/'cache/index.json')['files'])
    atomic_json(dict(status='complete',evaluated_symbols=audit['eligible_symbols'],source_unchanged=True,
        provenance_scope=lineage['scope'],automatic_promotion=False),out/'completion.json')
    progress('Cross-symbol evaluation complete; original failed price gate remains in report')


if __name__=='__main__':main()
