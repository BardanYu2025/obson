"""Frozen long/short representations with additive or gated causal residual fusion."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .ae_diagnostics import fingerprint, metrics, summarize
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .history_autoencoder import reconstruction_loss, to_ohlc
from .large_history import LargeHistory, HierWindows, move, write_global_review
from .memory_benchmark import regression_metrics
from .metrics import balanced_accuracy
from .progress import progress
from .representation import time_mask
from .short_state import ShortState
from .stage_audit import load_data

MODES=('long','short','add','gated','long_gated')


class ResidualFusion(nn.Module):
    def __init__(self,mode,width=512):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown fusion mode')
        self.mode=mode
        # Identical seed gives every variant the same initial readout parameters.
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4))
        if mode in ('add','gated','long_gated'):
            self.short_norm=nn.LayerNorm(width)
            self.project=nn.Linear(width,width)
            nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
        if mode in ('gated','long_gated'):
            self.long_norm=nn.LayerNorm(width)
            self.gate=nn.Linear(2*width,width)
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9))

    def forward(self,long,short):
        if long.shape!=short.shape or long.ndim!=2:raise ValueError('Aligned [batch,width] inputs required')
        if self.mode=='short':z=short;gate=torch.zeros_like(short)
        elif self.mode=='long':z=long;gate=torch.zeros_like(long)
        else:
            source=long if self.mode=='long_gated' else short
            delta=self.project(self.short_norm(source))
            gate=(torch.sigmoid(self.gate(torch.cat((self.long_norm(long),self.short_norm(source)),-1)))
                  if self.mode!='add' else torch.ones_like(delta))
            z=long+gate*delta
        return dict(z=z,logits=self.head(z),gate=gate)


def class_weights(labels):
    counts=torch.bincount(labels,minlength=4)
    return len(labels)/(int((counts>0).sum())*counts.clamp_min(1).float())


def losses(result,long,labels,weights,preserve,mode):
    ce=nn.functional.cross_entropy(result['logits'],labels,weight=weights,reduction='none')
    drift=((result['z']-long).square().mean(1)/long.square().mean(1).clamp_min(1e-6)
           if mode not in ('long','short') else torch.zeros_like(ce))
    return ce+preserve*drift,ce,drift


@torch.no_grad()
def extract_short(model,base,bounds,split,keys,batch,device):
    """Read each contract chronologically; collect exact requested endpoints only."""
    model.eval();groups={};result=np.empty((len(keys),model.width),np.float32);seen=np.zeros(len(keys),bool)
    for j,(i,end) in enumerate(keys):groups.setdefault(int(i),[]).append((int(end),j))
    sequences=[]
    for i,ends in groups.items():
        rows=np.flatnonzero(time_mask(base.series[i],bounds,split))
        if not len(rows) or np.any(np.diff(rows)!=1):raise ValueError('Noncontiguous split')
        if min(e for e,_ in ends)<rows[0] or max(e for e,_ in ends)>rows[-1]:raise ValueError('Endpoint outside split')
        sequences.append((i,int(rows[0]),max(e for e,_ in ends)+1,sorted(ends)))
    for start in range(0,len(sequences),batch):
        group=sequences[start:start+batch];hidden=None
        for offset in range(0,max(hi-lo for _,lo,hi,_ in group),128):
            x=np.zeros((len(group),128,18),np.float32);pick=[]
            for lane,(i,lo,hi,ends) in enumerate(group):
                left=lo+offset;right=min(left+128,hi)
                if left>=right:continue
                x[lane,:right-left]=base.encoded[i]['x'][left:right]
                pick.extend((lane,end-left,j) for end,j in ends if left<=end<right)
            h,hidden=model(torch.tensor(x,device=device),hidden)
            if pick:
                lanes,steps,ids=zip(*pick);result[list(ids)]=h[list(lanes),list(steps)].cpu().numpy();seen[list(ids)]=True
        progress(f'Fusion short cache {split}: contracts={min(start+batch,len(sequences))}/{len(sequences)}')
    if not seen.all() or not np.isfinite(result).all():raise ValueError('Incomplete short-state alignment')
    return result


@torch.no_grad()
def feature_cache(long_model,short_model,ds,bounds,split,path,identity,encode_batch,stream_batch,device):
    path.mkdir(parents=True,exist_ok=True);file=path/f'{split}.npz';index=path/f'{split}.json'
    keys=np.asarray([(i,end) for i,end,_ in ds.items],np.int64)
    if file.exists() and index.exists():
        info=json.loads(index.read_text())
        if info.get('identity')==identity and info.get('sha256')==fingerprint(file):
            with np.load(file,allow_pickle=False) as data:values={k:data[k] for k in data.files}
            if not np.array_equal(values['keys'],keys):raise ValueError('Cached endpoint mismatch')
            if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite cache')
            return values
    long_model.eval();longs=[];labels=[];targets=[];blocks=[]
    for step,batch in enumerate(DataLoader(ds,batch_size=encode_batch)):
        b=move(batch,device);n,k,w,f=b['x'].shape
        local=long_model.local.encode(b['x'].reshape(n*k,w,f))[:,-1].reshape(n,k,-1)
        z=long_model.summarize(local,b['offsets'],b['valid'])
        longs.extend(z.cpu().numpy());targets.extend(batch['targets'].numpy());blocks.extend(batch['blocks'].tolist())
        labels.extend(int(ds.base.series[i].labels['state'][end,1]) for i,end in zip(batch['series'].tolist(),batch['row'].tolist()))
        if step%50==0:progress(f'Fusion long cache {split}: {len(longs)}/{len(ds)}')
    values=dict(long=np.asarray(longs),short=extract_short(short_model,ds.base,bounds,split,keys,stream_batch,device),
                labels=np.asarray(labels,np.int64),targets=np.asarray(targets),blocks=np.asarray(blocks),keys=keys)
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite extracted features')
    with file.with_suffix('.tmp').open('wb') as f:np.savez(f,**values)
    file.with_suffix('.tmp').replace(file);atomic_json(dict(identity=identity,sha256=fingerprint(file)),index)
    return values


def device_features(data,device):
    return {k:torch.tensor(data[k],device=device) for k in ('long','short','labels')}


@torch.no_grad()
def score(model,data,weights,preserve,batch):
    model.eval();totals=np.zeros(3);pred=[];n=len(data['labels']);gates=[];relative=[]
    for start in range(0,n,batch):
        b={k:v[start:start+batch] for k,v in data.items()};r=model(b['long'],b['short'])
        loss,ce,drift=losses(r,b['long'],b['labels'],weights,preserve,model.mode)
        totals+=np.array([float(x.sum()) for x in (loss,ce,drift)])
        pred.extend(r['logits'].argmax(1).cpu().tolist())
        if model.mode not in ('long','short'):
            relative.extend(((r['z']-b['long']).norm(dim=1)/b['long'].norm(dim=1).clamp_min(1e-6)).cpu().tolist())
            gates.extend(r['gate'].mean(1).cpu().tolist())
    return dict(objective=float(totals[0]/n),weighted_ce=float(totals[1]/n),relative_squared_drift=float(totals[2]/n),
                state=balanced_accuracy(np.asarray(pred),data['labels'].cpu().numpy(),4),
                mean_gate=float(np.mean(gates)) if gates else None,
                mean_relative_correction=float(np.mean(relative)) if relative else None)


def save_training(state,path):
    atomic_save(state,path/'last.pt')
    atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],model=state['best_model']),path/'best.pt')
    temp=path/'history.jsonl.tmp';temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in state['history']));temp.replace(path/'history.jsonl')


def train_variant(mode,train,val,path,meta,device='cuda'):
    random.seed(meta['seed']);np.random.seed(meta['seed']);torch.manual_seed(meta['seed'])
    model=ResidualFusion(mode).to(device);opt=torch.optim.AdamW(model.parameters(),lr=meta['lr'],weight_decay=.01)
    train=device_features(train,device);val=device_features(val,device);weights=class_weights(train['labels'])
    metadata={**meta,'mode':mode};batch=meta['batch'];preserve=meta['preserve']
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Fusion recovery metadata mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    else:
        if path.exists() and any(path.iterdir()):raise ValueError('Nonempty unrecoverable variant directory')
        path.mkdir(parents=True,exist_ok=True)
        initial=score(model,val,weights,preserve,batch)
        state=dict(metadata=metadata,epoch=0,best_epoch=0,best=initial['objective'],history=[],
                   best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        state.update(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        save_training(state,path)
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started=time.perf_counter();model.train();total=0.;n=len(train['labels']);order=torch.randperm(n,device=device)
        for offset in range(0,n,batch):
            ids=order[offset:offset+batch];b={k:v[ids] for k,v in train.items()}
            r=model(b['long'],b['short']);loss=losses(r,b['long'],b['labels'],weights,preserve,mode)[0].mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite fusion loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(ids)
        validation=score(model,val,weights,preserve,batch)
        if not np.isfinite(validation['objective']):raise ValueError('Nonfinite validation loss')
        if validation['objective']<state['best']:
            state.update(best=validation['objective'],best_epoch=epoch,best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        state['history'].append(dict(epoch=epoch,train_objective=total/n,validation=validation,seconds=time.perf_counter()-started))
        state.update(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        save_training(state,path)
        if epoch==1 or epoch%10==0:progress(f'Fusion {mode} epoch={epoch} val={validation["objective"]:.5f} BA={validation["state"]["ba"]:.4f} best={state["best_epoch"]}')
    # Recreate best/history if a previous run stopped between atomic file writes.
    save_training(state,path)
    model.load_state_dict(state['best_model']);model.eval()
    return model,state['best_epoch']


@torch.no_grad()
def retention(model,long_model,ds,features,mean,scale,batch,device,out,examples=False):
    """Frozen original decoder and structure head; no access to short/raw bars at decode."""
    model.eval();long_model.eval();records={};targets=[];predictions=[];cards=[];loss_sum=0.;offset=0
    chosen=set(np.linspace(0,len(ds)-1,min(6,len(ds)),dtype=int))
    for data in DataLoader(ds,batch_size=batch):
        n=len(data['x']);m=torch.tensor(features['long'][offset:offset+n],device=device)
        s=torch.tensor(features['short'][offset:offset+n],device=device);z=model(m,s)['z']
        decoded=long_model.decode_history(z)['reconstruction']
        predictions.extend((long_model.structure(z)*scale+mean).cpu().numpy());targets.extend(data['targets'].numpy())
        for j in range(n):
            valid=data['valid'][j].numpy();truth=data['y'][j].numpy()[valid];pred=decoded[j].cpu().numpy()[valid]
            mask=data['mask'][j].numpy()[valid]
            loss_sum+=float(reconstruction_loss(torch.tensor(pred),torch.tensor(truth),torch.tensor(mask)).mean())
            for name,y,p in (('history',truth.reshape(-1,7),pred.reshape(-1,7)),('recent64',truth[-1,-64:],pred[-1,-64:])):
                records.setdefault(name,[]).append(metrics(y,p))
            if examples and offset+j in chosen:
                i=int(data['series'][j]);end=int(data['row'][j]);anchor=float(data['anchor'][j])
                cards.append(dict(source=ds.base.series[i].key,end=str(ds.base.series[i].frame.datetime.iloc[end]),blocks=int(valid.sum()),
                    truth_ohlc=to_ohlc(truth.reshape(-1,7),anchor).tolist(),reconstructed_ohlc=to_ohlc(pred.reshape(-1,7),anchor).tolist()))
        offset+=n
        if offset==len(ds) or offset%(batch*50)==0:progress(f'Fusion retention {model.mode}: {offset}/{len(ds)}')
    report=dict(windows=len(ds),history_reconstruction_loss=loss_sum/len(ds),reconstruction=summarize(records),
                historical_structure=regression_metrics(np.asarray(targets),np.asarray(predictions)))
    if examples:
        atomic_json(cards,out/'examples.json');write_global_review(out/'examples.html',cards)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--long-run',required=True);p.add_argument('--short-run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--encode-batch',type=int,default=16);p.add_argument('--stream-batch',type=int,default=128)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--preserve',type=float,default=.1)
    a=p.parse_args()
    if not torch.cuda.is_available() or min(a.batch,a.encode_batch,a.stream_batch,a.epochs)<1 or not math.isfinite(a.preserve) or a.preserve<0:
        raise ValueError('CUDA and valid positive settings required')
    long_run=Path(a.long_run);short_run=Path(a.short_run);out=Path(a.out)
    source_paths=dict(long_manifest=long_run/'manifest.json',long_best=long_run/'joint/best.pt',
                      short_manifest=short_run/'manifest.json',short_best=short_run/'best.pt')
    hashes={k:fingerprint(v) for k,v in source_paths.items()}
    short_meta=json.loads(source_paths['short_manifest'].read_text())
    if short_meta['source_manifest']!=hashes['long_manifest'] or short_meta['local_checkpoint']!=fingerprint(long_run/'local/best.pt'):
        raise ValueError('Short and long source experiment mismatch')
    meta=dict(schema='babel-residual-fusion-v1',sources=hashes,modes=list(MODES),seed=42,epochs=a.epochs,batch=a.batch,lr=3e-4,preserve=a.preserve,
              source_policy='Frozen joint long512 and frozen persistent short512 at identical complete-block endpoints; no recurrent feedback from fused z',
              selection='Validation training-class-balanced CE + preserve * relative squared latent drift; test only after every variant is selected',
              interpretation='Supervised current-rule-state fusion, not unsupervised embedding pretraining or forecasting. Reused research test period is not a fresh final holdout.')
    if out.exists() and any(out.iterdir()):
        if not (out/'manifest.json').exists() or json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Changed fusion settings; use a new directory')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    ref,series,encoded,local_sets=load_data(a.root,long_run)
    # Teacher values are unused: long vectors come from the frozen joint encoder directly.
    dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in series]
    hier=[HierWindows(ds,dummy,ref['boundaries'],split) for ds,split in zip(local_sets,('train','val','test'))]
    long_model=LargeHistory().cuda();ck=torch.load(source_paths['long_best'],map_location='cpu',weights_only=True)
    long_model.load_state_dict(ck['model']);long_model.eval().requires_grad_(False)
    mean=torch.tensor(ck['metadata']['target_mean'],device='cuda');scale=torch.tensor(ck['metadata']['target_scale'],device='cuda')
    short_model=ShortState().cuda();short_ck=torch.load(source_paths['short_best'],map_location='cpu',weights_only=True)
    short_model.load_state_dict(short_ck['model']);short_model.eval().requires_grad_(False)
    arrays=[feature_cache(long_model,short_model,ds,ref['boundaries'],split,out/'feature_cache',hashes,a.encode_batch,a.stream_batch,'cuda')
            for ds,split in zip(hier,('train','val','test'))]
    atomic_json({s:dict(windows=len(ds),blocks={str(k):int((data['blocks']==k).sum()) for k in np.unique(data['blocks'])},
                       state_counts=np.bincount(data['labels'],minlength=4).tolist()) for s,ds,data in zip(('train','val','test'),hier,arrays)},out/'coverage.json')
    # No tests are scored while choosing epochs or fitting weights.
    for mode in MODES:train_variant(mode,arrays[0],arrays[1],out/mode,meta)
    report=dict(schema='babel-residual-fusion-evaluation-v1',variants={},source_epochs=dict(long=ck['epoch'],short=short_ck['epoch']),
                interpretation=meta['interpretation'],selection=meta['selection'])
    weights=class_weights(torch.tensor(arrays[0]['labels'],device='cuda'))
    for mode in MODES:
        model=ResidualFusion(mode).cuda();best=torch.load(out/mode/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(best['model'])
        result=dict(selected_epoch=best['epoch'],parameters=sum(p.numel() for p in model.parameters()),
                    classification={s:score(model,device_features(data,'cuda'),weights,a.preserve,a.batch) for s,data in zip(('val','test'),arrays[1:])})
        if mode!='short':
            result['retention']={s:retention(model,long_model,ds,data,mean,scale,a.encode_batch,'cuda',out/mode,examples=s=='test')
                                 for s,ds,data in zip(('val','test'),hier[1:],arrays[1:])}
        else:result['retention']=None  # Short vectors are not in the long decoder's coordinate system.
        report['variants'][mode]=result;atomic_json(result,out/mode/'metrics.json');atomic_json(report,out/'fusion_metrics.json')
    if hashes!={k:fingerprint(v) for k,v in source_paths.items()}:raise ValueError('Source artifacts changed during fusion')
    atomic_json(dict(sources=hashes,checkpoints={m:fingerprint(out/m/'best.pt') for m in MODES}),out/'artifacts.json')
    progress(f'Fusion complete: {out}')


if __name__=='__main__':main()
