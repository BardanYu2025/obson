"""Causal persistent short state, truncated backpropagation, no forecast targets."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ae_diagnostics import fingerprint, metrics, summarize
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .autoregressive import probe
from .history_autoencoder import positions, reconstruction_loss, to_ohlc
from .large_history import LargeHistory, write_global_review
from .progress import progress
from .representation import time_mask
from .stage_audit import load_data


class ShortState(nn.Module):
    def __init__(self,width=512,layers=2,input_dim=18):
        super().__init__();self.width=width;self.layers=layers
        self.input=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width),nn.GELU())
        self.rnn=nn.GRU(width,width,layers,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.decoder=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,7))

    def forward(self,x,state=None):
        h,state=self.rnn(self.input(x),state)
        return self.norm(h),state

    def decode(self,z):
        if z.ndim!=2:raise ValueError('Decoder accepts a single state vector per sample')
        raw=self.decoder(z[:,None]+positions(64,self.width,z.device,z.dtype))
        return torch.cat((raw[...,:2],nn.functional.softplus(raw[...,2:])),dim=-1)


def multiscale_loss(pred,target,mask):
    return torch.stack([reconstruction_loss(pred[:,-n:],target[:,-n:],mask[:,-n:]) for n in (16,32,64)]).mean(0)


class Streams:
    """Each batch lane is one contract-period and one time partition; never splice contracts."""
    def __init__(self,base,bounds,split):
        self.base=base;self.sequences=[];self.ends={}
        for j,(i,end) in enumerate(base.items):self.ends.setdefault(i,{})[end]=j
        for i in self.ends:
            rows=np.flatnonzero(time_mask(base.series[i],bounds,split))
            if not len(rows) or np.any(np.diff(rows)!=1):raise ValueError('Expected contiguous time partition')
            self.sequences.append((i,int(rows[0]),int(rows[-1])+1))

    def batches(self,batch_size,shuffle=False,chunk=128):
        order=list(range(len(self.sequences)))
        if shuffle:random.shuffle(order)
        for start in range(0,len(order),batch_size):
            group=[self.sequences[j] for j in order[start:start+batch_size]]
            for offset in range(0,max(hi-lo for _,lo,hi in group),chunk):
                x=np.zeros((len(group),chunk,18),np.float32);samples=[];indices=[]
                for lane,(i,lo,hi) in enumerate(group):
                    left=lo+offset;right=min(left+chunk,hi)
                    if left>=right:continue
                    x[lane,:right-left]=self.base.encoded[i]['x'][left:right]
                    for end in range(left,right):
                        if end not in self.ends[i]:continue
                        sample=self.base[self.ends[i][end]]
                        anchor=float(self.base.series[i].frame.close.iloc[end-64])
                        shift=100*np.log(anchor/sample['anchor'])
                        y=sample['y'][-64:].copy();y[:,:2]-=shift
                        samples.append(dict(y=y,mask=sample['mask'][-64:],anchor=anchor,
                                            series=i,row=end,local_x=sample['x'],local_shift=shift))
                        indices.append((lane,end-left))
                yield offset==0,x,indices,samples


def tensors(samples,device):
    y=torch.tensor(np.stack([s['y'] for s in samples]),device=device)
    mask=torch.tensor(np.stack([s['mask'] for s in samples]),device=device)
    return y,mask


def run_epoch(model,streams,batch_size,device,opt=None):
    model.train(opt is not None);hidden=None;total=0.;count=0
    with torch.set_grad_enabled(opt is not None):
        for step,(reset,x,indices,samples) in enumerate(streams.batches(batch_size,shuffle=opt is not None)):
            if reset:hidden=None
            h,hidden=model(torch.tensor(x,device=device),hidden)
            # Carry values across chunks, truncate only the backward graph.
            hidden=hidden.detach()
            if not samples:continue
            lanes,steps=zip(*indices);z=h[list(lanes),list(steps)]
            y,mask=tensors(samples,device);loss=multiscale_loss(model.decode(z),y,mask).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite short-state loss')
            if opt is not None:
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(samples);count+=len(samples)
            if step%100==0:progress(f'Short state chunks={step+1} windows={count} loss={total/count:.6f}')
    if count!=len(streams.base):raise ValueError('Streaming endpoint coverage mismatch')
    return total/count


@torch.no_grad()
def evaluate_state(model,streams,control,batch_size,local_batch,out,device):
    model.eval();control.eval();features=[];records={};cards=[]
    for split,ds in zip(('train','val','test'),streams):
        zs=[];locals_=[];labels=[];hidden=None;seen=0
        for reset,x,indices,samples in ds.batches(batch_size):
            if reset:hidden=None
            h,hidden=model(torch.tensor(x,device=device),hidden)
            if not samples:continue
            lanes,steps=zip(*indices);z=h[list(lanes),list(steps)];pred=model.decode(z).cpu().numpy()
            local_z=[];local_pred=[]
            for start in range(0,len(samples),local_batch):
                sub=samples[start:start+local_batch]
                values=control.local(torch.tensor(np.stack([s['local_x'] for s in sub]),device=device))
                local_z.extend(values['z'].cpu().numpy());v=values['reconstruction'][:,-64:].cpu().numpy()
                v[...,:2]-=np.array([s['local_shift'] for s in sub])[:,None,None];local_pred.extend(v)
            zs.extend(z.cpu().numpy());locals_.extend(local_z)
            for j,s in enumerate(samples):
                labels.append(int(ds.base.series[s['series']].labels['state'][s['row'],1]))
                if split=='test':
                    for name,values in (('short',pred),('frozen_local',local_pred)):
                        for n in (16,32,64):records.setdefault(f'{name}/{n}',[]).append(metrics(s['y'][-n:],values[j][-n:]))
                    # Fixed endpoint indices in the original dataset; never select by error.
                    idx=ds.ends[s['series']][s['row']]
                    if idx in set(np.linspace(0,len(ds.base)-1,6,dtype=int)):
                        series=ds.base.series[s['series']]
                        cards.append(dict(source=series.key,end=str(series.frame.datetime.iloc[s['row']]),blocks=1,
                            truth_ohlc=to_ohlc(s['y'],s['anchor']).tolist(),reconstructed_ohlc=to_ohlc(pred[j],s['anchor']).tolist()))
            seen+=len(samples)
        if seen!=len(ds.base):raise ValueError('Evaluation coverage mismatch')
        features.append((np.asarray(zs),np.asarray(locals_),np.asarray(labels)))
        progress(f'Short state evaluation {split}: {seen}')
    result=dict(schema='babel-short-state-v1',reconstruction=summarize(records),state_probes={},
                interpretation='Persistent GRU vs frozen stage-1 local512 on identical endpoints. GRU sees all preceding bars within partition; baseline sees last128. Different context budgets, not a pure architecture ablation. No long-memory fusion or forecast claims.')
    for name,column in (('short',0),('frozen_local',1)):
        result['state_probes'][name]=probe(*(v for row in features for v in (row[column],row[2])),4)
    atomic_json(result,out/'short_metrics.json');atomic_json(cards,out/'short_examples.json')
    write_global_review(out/'short_examples.html',cards)
    # Use a truthful title for the reused plot renderer.
    file=out/'short_examples.html';file.write_text(file.read_text().replace('512维历史重建','短时状态历史重建').replace('单个512维综合向量重建历史','单个短时状态向量重建最近64根').replace('全局向量解码','短时状态解码'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--local-batch',type=int,default=64)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--evaluate',action='store_true')
    a=p.parse_args()
    if not torch.cuda.is_available() or min(a.batch,a.local_batch,a.epochs)<1:raise ValueError('CUDA and positive settings required')
    source=Path(a.source);out=Path(a.out)
    meta=dict(schema='babel-short-state-v1',source_manifest=fingerprint(source/'manifest.json'),
              local_checkpoint=fingerprint(source/'local/best.pt'),width=512,layers=2,chunk=128,
              batch_streams=a.batch,epochs=a.epochs,seed=42,lr=3e-4,scales=[16,32,64],
              state_policy='Reset per contract/period/partition/epoch; detach per128 bars; no sequence shuffle within lane',
              target='Past64 relative to preceding close; equal16/32/64 reconstruction losses; no future supervision')
    if out.exists() and any(out.iterdir()):
        if not (out/'manifest.json').exists() or json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Settings changed; use a separate run directory')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    ref,series,encoded,sets=load_data(a.root,source)
    streams=[Streams(ds,ref['boundaries'],s) for ds,s in zip(sets,('train','val','test'))]
    atomic_json({s:dict(windows=len(ds.base),streams=len(ds.sequences)) for s,ds in zip(('train','val','test'),streams)},out/'coverage.json')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=ShortState().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    state=dict(epoch=0,best=float('inf'),history=[],best_epoch=0)
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=meta:raise ValueError('Checkpoint metadata mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    if not a.evaluate:
        for epoch in range(state['epoch']+1,a.epochs+1):
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            train=run_epoch(model,streams[0],a.batch,'cuda',opt);val=run_epoch(model,streams[1],a.batch,'cuda')
            if val<state['best']:state.update(best=val,best_epoch=epoch,best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            state['history'].append(dict(epoch=epoch,train_loss=train,validation_loss=val,seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            state.update(epoch=epoch,metadata=meta,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
            atomic_save(state,out/'last.pt');atomic_save(dict(epoch=state['best_epoch'],model=state['best_model'],metadata=meta),out/'best.pt')
            temp=out/'history.jsonl.tmp';temp.write_text(''.join(json.dumps(row)+'\n' for row in state['history']));temp.replace(out/'history.jsonl')
            progress(f'Short state epoch={epoch} val={val:.6f} best={state["best_epoch"]}')
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
    control=LargeHistory().cuda();control.load_state_dict(torch.load(source/'local/best.pt',map_location='cpu',weights_only=True)['model'])
    evaluate_state(model,streams,control,a.batch,a.local_batch,out,'cuda')
    atomic_json(dict(selected_epoch=ck['epoch'],source_local_sha256=meta['local_checkpoint'],best_sha256=fingerprint(out/'best.pt')),out/'artifacts.json')
    progress(f'Short state complete: {out}')


if __name__=='__main__':main()
