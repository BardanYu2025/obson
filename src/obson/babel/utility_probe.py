"""Masked frozen-state diagnostics; observed current features never score as utility."""
import numpy as np
from . import window_state_probe as probe

CURRENT_NAMES = ('gap_asinh_logpct', 'body_asinh_logpct', 'volume_vs_prior_ema32',
                 'delta_log_volume', 'oi_change_percent_asinh', 'oi_per_volume_asinh', 'turnover_asinh')
NAMES = probe.NAMES + tuple('current/'+n for n in CURRENT_NAMES)
GROUPS = dict(utility=(0,2,3,5), direction=(0,3), volatility=(2,5), trend_diagnostic=(1,4))


def restore_raw(x,stats):
    """Invert the pinned float32 cache; canonicalize only OI zero sentinels.

    Mean/scale round trips do not necessarily reproduce exact zero. Compare
    with the cached encoding of zero before applying source zero predicates.
    Sub-quantization values cannot be distinguished from that same zero code.
    """
    x=np.asarray(x);mean=np.asarray(stats['x_mean']);scale=np.asarray(stats['x_scale'])
    raw=x.astype(np.float64)*scale+mean
    for i in (20,22):raw[...,i]=np.where(x[...,i]==np.float32(-mean[i]/scale[i]),0.,raw[...,i])
    return raw


def targets(raw):
    raw=np.asarray(raw,dtype=np.float64)
    state=probe.descriptors(raw);last=raw[:,-1]
    current=last[:,[0,1,18,19,20,21,22]].copy()
    valid=np.ones_like(current,dtype=bool)
    valid[:,3]=last[:,27]>0;valid[:,4]=last[:,23]>0
    valid[:,5]=last[:,24]>0;valid[:,6]=last[:,25]>0
    suspect=((last[:,23]>0)&(last[:,22]==0)&(last[:,20]!=0))|((last[:,24]>0)&(np.abs(last[:,21])>np.arcsinh(1.)+1e-6))
    valid[:,4:6]&=~suspect[:,None]
    y=np.concatenate((state,np.where(valid,current,0)),axis=1)
    mask=np.concatenate((np.ones_like(state,dtype=bool),valid),axis=1)
    return y,mask


def target_scales(y,mask):
    means=[];scales=[];counts=[]
    for i in range(y.shape[1]):
        values=y[mask[:,i],i]
        if len(values)<2 or not np.isfinite(values).all():raise ValueError('Insufficient finite training targets')
        means.append(float(values.mean()));scales.append(max(float(values.std()),1e-6));counts.append(len(values))
    return dict(mean=means,scale=scales,support=counts,fitted_on='train_only_available_rows')


def fit_heads(train,y,mask,val,vy,vmask,stats,device):
    """One five-alpha selection per target, same opportunity for every representation.

    Cache eigensystems for identical availability masks; missing targets never
    become zero-valued examples. Each feature scaler uses that target's train rows.
    """
    heads=[];weights=[];intercepts=[];eigen_cache={}
    for i in range(y.shape[1]):
        valid=mask[:,i];vv=vmask[:,i]
        if valid.sum()<2 or vv.sum()<2:raise ValueError('Insufficient train/validation availability')
        key=valid.tobytes()
        if key not in eigen_cache:
            xs=probe.scales(train[valid]);eigen_cache[key]=probe.eigensystem(probe.normalize(train[valid],xs),device)
        sy=((y[valid,i]-stats['mean'][i])/stats['scale'][i])[:,None]
        sv=((vy[vv,i]-stats['mean'][i])/stats['scale'][i])[:,None]
        head,grid=probe.ridge_select(train[valid],sy,val[vv],sv,device,eigen_cache[key])
        weights.append(head.pop('weights')[:,0]);intercepts.append(float(head.pop('intercept')[0]))
        heads.append(dict(**head,candidates=grid,target=NAMES[i],training_support=int(valid.sum()),validation_support=int(vv.sum())))
    return heads,np.column_stack(weights),np.array(intercepts)


def predict(heads,weights,intercepts,x,stats):
    normalized=np.column_stack([probe.normalize(x,h['statistics'])@weights[:,i]+intercepts[i] for i,h in enumerate(heads)])
    return normalized*np.array(stats['scale'])+np.array(stats['mean'])


def measure(pred,y,mask,stats):
    pred,y,mask=np.asarray(pred,float),np.asarray(y,float),np.asarray(mask,bool)
    if pred.shape!=y.shape or mask.shape!=y.shape or y.shape[1]!=len(NAMES) or not np.isfinite(pred).all():
        raise ValueError('Finite matched thirteen-target predictions required')
    if not np.isfinite(y[mask]).all():raise ValueError('Nonfinite observed targets')
    error=np.where(mask,((pred-y)/stats['scale'])**2,0)
    reports=[]
    for i,name in enumerate(NAMES):
        valid=mask[:,i];n=int(valid.sum());var=float(np.var(y[valid,i])) if n else 0.
        reports.append(dict(name=name,support=n,nmse=float(error[valid,i].mean()) if n else None,
            mae=float(np.abs(pred[valid,i]-y[valid,i]).mean()) if n else None,
            r2=float(1-np.square(pred[valid,i]-y[valid,i]).mean()/var) if var>1e-12 else None))
    per_window={k:error[:,ids].mean(1) for k,ids in GROUPS.items()}
    per_window.update({name:error[:,i] for i,name in enumerate(NAMES)})
    return dict(windows=len(y),targets=reports,groups={k:float(v.mean()) for k,v in per_window.items() if k in GROUPS},
        direction={str(h):probe.classification(pred[:,i*3],y[:,i*3]) for i,h in enumerate(probe.HORIZONS)}),per_window
