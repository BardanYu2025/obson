"""Frozen-state local-head adaptation; all neural updates are confined to a reader."""
import numpy as np
import torch
from torch import nn
from . import bar_alignment as ba


def schedule(n,seed,epoch):
    order=np.random.default_rng(np.random.SeedSequence([seed,epoch,20260926])).permutation(n)
    prefixes=ba.position_plan(n,seed+101000,epoch)
    return order,prefixes


def scales(x,batch=128):
    total=np.zeros(x.shape[-1]);square=total.copy();count=0
    for start in range(0,len(x),batch):
        a=np.asarray(x[start:start+batch],dtype=np.float64).reshape(-1,x.shape[-1])
        if not np.isfinite(a).all():raise ValueError('Nonfinite frozen states')
        total+=a.sum(0);square+=(a*a).sum(0);count+=len(a)
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,0)).clip(1e-6)
    return dict(mean=mean.tolist(),scale=std.tolist(),fitted_on='train_all_nonheld_prefix_states',rows=count)


def reader(width,hidden,seed,device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+20260926)
        model=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,ba.HISTORY*7))
    return model.to(device)


def normalize(x,stats):return (x-x.new_tensor(stats['mean']))/x.new_tensor(stats['scale'])


def predict(head,x,stats):
    return head(normalize(x,stats)).reshape(*x.shape[:-1],ba.HISTORY,7)


def epoch(head,data,stats,positions,order,prefixes,batch,optimizer=None):
    training=optimizer is not None
    if batch<1 or data['x'].requires_grad or data['y'].requires_grad:raise ValueError('Positive batch and frozen states/targets required')
    if len(order)!=len(data['x']) or sorted(order.tolist())!=list(range(len(order))):raise ValueError('Exactly one pass over every training window required')
    if prefixes.shape!=(len(order),4) or not np.isin(prefixes,ba.TRAIN_PREFIXES).all() or np.any(np.diff(np.sort(prefixes,axis=1),axis=1)==0):raise ValueError('Four distinct nonheld training positions required')
    lookup={p:i for i,p in enumerate(positions)};head.train(training);total=0.;steps=0
    with torch.set_grad_enabled(training):
        for start in range(0,len(order),batch):
            ids=order[start:start+batch];selected=np.array([[lookup[int(p)] for p in row] for row in prefixes[ids]])
            ii=torch.tensor(ids,device=data['x'].device)[:,None];pp=torch.tensor(selected,device=data['x'].device)
            x,y,mask=(data[k][ii,pp] for k in ('x','y','mask'))
            if x.requires_grad or y.requires_grad:raise ValueError('Only frozen states/targets allowed')
            pred=predict(head,x,stats);loss=ba.local_rows(pred,y,mask,True)['primary'].mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite local objective')
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
            total+=float(loss.detach())*len(ids)
    return dict(smooth=total/len(order),windows=len(order),prefix_exposures=4*len(order),steps=steps)


@torch.no_grad()
def validate(head,data,stats,batch):
    head.eval();total={}
    for start in range(0,len(data['x']),batch):
        pred=predict(head,data['x'][start:start+batch],stats)
        rows=ba.local_rows(pred,data['y'][start:start+batch],data['mask'][start:start+batch])
        for k,v in rows.items():total[k]=total.get(k,0.)+float(v.double().sum())
    return {k:v/len(data['x']) for k,v in total.items()}
