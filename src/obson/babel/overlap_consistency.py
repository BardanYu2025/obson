"""Paired historical supervision with a symmetric, observed-overlap auxiliary loss."""
import numpy as np
import torch
from torch.nn import functional as F
from . import overlap_diagnostic_run as odr

ur=odr.ur
TRAIN_SHIFTS=(1,16)


def eligible(specs,original,candidates,bank_length):
    # Validate the entire old index first, including partition/contract identities.
    odr.od.pair_plan(specs,original,bank_length)
    mapping={}
    for spec in specs:
        key=original[spec['endpoints'][0][1]]['key']
        if key in mapping:raise ValueError('Repeated packed contract')
        mapping[key]=spec
    selected=[];excluded={}
    for index,row in enumerate(candidates):
        spec=mapping.get(row['key']);reason=None
        if spec is None:reason='no_original_packed_contract'
        else:
            local=row['row']-spec['lo']
            if local>=spec['length']:reason='beyond_packed_endpoint'
            elif local-64<127 or row['row']-64<511:reason='partition_or_warmup'
        if reason:excluded[reason]=excluded.get(reason,0)+1;continue
        selected.append(dict(index=index,bank_end=spec['offset']+local))
    if not selected:raise ValueError('No eligible paired training windows')
    return selected,excluded


def schedule(size,budget,seed,epoch):
    ids=ur.bb.ws.sample_ids(size,budget,seed,300+epoch)
    rng=np.random.default_rng(np.random.SeedSequence([seed,300+epoch,20260925]))
    ids=ids[rng.permutation(len(ids))]
    shifts=np.array(TRAIN_SHIFTS)[rng.permutation(np.arange(len(ids))%2)]
    prefixes=ur.bb.ba.position_plan(len(ids),seed,300+epoch)
    return ids,shifts,prefixes


def consistency_rows(a,b,ma,mb,shifts,stats,smooth=True):
    """Both predictions normalized with the same fixed training y statistics.

    Align A[d:127] to B[:127-d]; no current bar, truth anchor, detach or extra input.
    Equal weights for change1, body and activity. No activity support -> omit family.
    """
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(128,7):raise ValueError('Matched128x7 predictions required')
    if ma.shape!=a.shape or mb.shape!=a.shape or len(shifts)!=len(a):raise ValueError('Invalid overlap masks/shifts')
    output=a.new_zeros(len(a));ys=a.new_tensor(stats['y_scale'])
    loss=lambda x,y:F.smooth_l1_loss(x,y,reduction='none') if smooth else (x-y).square()
    for d in torch.unique(shifts).tolist():
        if int(d)!=d or not 0<d<126:raise ValueError('Invalid overlap shift')
        d=int(d);ids=shifts==d;x=a[ids,d:127];y=b[ids,:127-d];mask=ma[ids,d:127]&mb[ids,:127-d]
        if not torch.equal(ma[ids,d:127],mb[ids,:127-d]) or not mask[:,:,:2].all():raise ValueError('Shared masks differ or missing price')
        dx=x[:,1:,0]-x[:,:-1,0];dy=y[:,1:,0]-y[:,:-1,0]
        change=loss(dx*ys[0]/stats['delta_scale'][0],dy*ys[0]/stats['delta_scale'][0]).mean(1)
        body=loss(x[:,:,1],y[:,:,1]).mean(1)
        valid=mask[:,:,2:];counts=valid.sum(1);ok=counts>0
        activity=((loss(x[:,:,2:],y[:,:,2:])*valid).sum(1)/counts.clamp_min(1)*ok).sum(1)/ok.sum(1).clamp_min(1)
        observed=ok.any(1);output[ids]=(change+body+activity*observed)/(2+observed.to(a.dtype))
    return output


def supervised(model,batch,prefixes,stats,local):
    prediction,detail=model(batch['x'],prefixes,True)
    target,mask=ur.bb.ba.local_targets(batch['y'],batch['mask'],prefixes,stats,local)
    loss=ur.bb.ar.error_rows(prediction,batch['y'],batch['mask'],stats,True)['primary']
    loss=loss+.25*ur.bb.ba.local_rows(detail,target,mask,True)['primary']
    return prediction,loss


def run_epoch(model,builder,indices,shifts,prefixes,stats,local,batch,micro,device,weight,encoder=None,head=None):
    training=encoder is not None
    if training!=(head is not None):raise ValueError('Both optimizers required')
    if not 0<micro<=batch or batch%micro or len(indices)!=len(shifts) or prefixes.shape!=(len(indices),4):raise ValueError('Invalid paired budget')
    model.train(training);total=np.zeros(3);steps=0
    with torch.set_grad_enabled(training):
        for left in range(0,len(indices),batch):
            size=min(batch,len(indices)-left)
            if training:encoder.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
            for start in range(left,left+size,micro):
                stop=min(start+micro,left+size);a,b=builder(indices[start:stop],shifts[start:stop])
                convert=lambda v:{k:torch.tensor(x,device=device) for k,x in v.items()}
                a,b=convert(a),convert(b);ps=torch.tensor(prefixes[start:stop],device=device);ds=torch.tensor(shifts[start:stop],device=device)
                # Both arms do identical forwards, true-history losses and auxiliary computation.
                joined={k:torch.cat((a[k],b[k])) for k in a};pred,sup=supervised(model,joined,torch.cat((ps,ps)),stats,local)
                n=len(ps);base=(sup[:n]+sup[n:])/2
                aux=consistency_rows(pred[:n],pred[n:],a['mask'],b['mask'],ds,stats)
                objective=base+weight*aux
                if not all(torch.isfinite(v).all() for v in (base,aux,objective)):raise ValueError('Nonfinite paired loss')
                if training:(objective.sum()/size).backward()
                total+=np.array([float(v.detach().sum()) for v in (base,aux,objective)])
            if training:
                torch.nn.utils.clip_grad_norm_(model.core.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(model.local_head.parameters(),1.,error_if_nonfinite=True)
                encoder.step();head.step();steps+=1
    return dict(zip(('supervised','consistency','objective'),(total/len(indices)).tolist())),steps
