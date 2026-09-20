"""Parallel reconstruction-led long/short fusion; no state-label training loss."""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from .ae_diagnostics import fingerprint, metrics, summarize
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .autoregressive import probe
from .fusion_readout_audit import load_cache
from .history_autoencoder import positions, reconstruction_loss, to_ohlc
from .large_history import LargeHistory, HierWindows, move, write_global_review
from .progress import progress
from .residual_fusion import retention
from .stage_audit import load_data

MODES=('add','long','short')


class ReconstructionFusion(nn.Module):
    def __init__(self,mode,width=512,decoder_width=256):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown reconstruction mode')
        self.mode=mode;self.decoder_width=decoder_width
        # Initialize the identical recent decoder before any variant-specific parameters.
        self.expand=nn.Linear(width,decoder_width)
        layer=nn.TransformerEncoderLayer(decoder_width,8,decoder_width*4,dropout=.1,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Sequential(nn.LayerNorm(decoder_width),nn.Linear(decoder_width,7))
        if mode=='add':
            self.norm=nn.LayerNorm(width);self.project=nn.Linear(width,width)
            nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)

    def fuse(self,long,short):
        if long.shape!=short.shape or long.ndim!=2:raise ValueError('Aligned vector pairs required')
        if self.mode=='long':return long
        if self.mode=='short':return short
        return long+self.project(self.norm(short))

    def decode_recent(self,z):
        if z.ndim!=2:raise ValueError('Recent decoder accepts only one vector')
        h=self.expand(z)[:,None]+positions(64,self.decoder_width,z.device,z.dtype)
        raw=self.output(self.decoder(h))
        return torch.cat((raw[...,:2],nn.functional.softplus(raw[...,2:])),dim=-1)

    def forward(self,long,short):
        z=self.fuse(long,short)
        return dict(z=z,recent=self.decode_recent(z))


def recent_loss(pred,y,mask):
    base=torch.stack([reconstruction_loss(pred[:,-n:],y[:,-n:],mask[:,-n:]) for n in (16,32,64)]).mean(0)
    changes=torch.stack([nn.functional.smooth_l1_loss(pred[:,h:,1]-pred[:,:-h,1],y[:,h:,1]-y[:,:-h,1],reduction='none').mean(1) for h in (4,16)]).mean(0)
    return base+.25*changes


def history_values(pred,y,mask,valid):
    rows=valid.nonzero()[:,0];n=len(valid);counts=valid.sum(1).clamp_min(1)
    loss=reconstruction_loss(pred[valid],y[valid],mask[valid])
    close=(pred[valid][...,1]-y[valid][...,1]).abs().mean(1)*100
    return (pred.new_zeros(n).scatter_add_(0,rows,v)/counts for v in (loss,close))


def eligible(validation,baseline,tolerance):
    return all(np.isfinite(validation[k]) and validation[k]<=baseline[k]*(1+tolerance) for k in ('history_loss','history_close_bps'))


class CachedWindows(Dataset):
    def __init__(self,path,split):
        self.arrays={k:np.load(Path(path)/f'{split}_{k}.npy',mmap_mode='r',allow_pickle=False)
                     for k in ('long','short','y','mask','valid','teacher','recent_y','recent_mask')}
    def __len__(self):return len(self.arrays['long'])
    def __getitem__(self,i):return {k:np.array(v[i],copy=True) for k,v in self.arrays.items()}


def load_long(path,device):
    model=LargeHistory().to(device)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model']);model.eval().requires_grad_(False)
    return model,ck


@torch.no_grad()
def prepare_cache(source,long_run,root,out,identity,batch,device):
    cache=out/'target_cache';cache.mkdir(exist_ok=True);index=cache/'index.json'
    if index.exists():
        info=json.loads(index.read_text())
        if info['identity']==identity and all((cache/k).exists() and fingerprint(cache/k)==v for k,v in info['files'].items()):
            progress('Verified shared reconstruction cache');return
    ref,series,encoded,sets=load_data(root,long_run)
    model,_=load_long(long_run/'joint/best.pt',device)
    dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in series]
    sources=json.loads((source/'manifest.json').read_text())['sources'];coverage={};files={}
    for split,base in zip(('train','val','test'),sets):
        features,sha=load_cache(source,split,sources);ds=HierWindows(base,dummy,ref['boundaries'],split)
        if not np.array_equal(features['keys'],np.array([(i,end) for i,end,_ in ds.items])):raise ValueError('Target/feature endpoints do not match')
        n=len(ds);arrays={}
        specs=dict(y=((n,16,128,7),np.float32),teacher=((n,16,128,7),np.float32),mask=((n,16,128,7),bool),
                   valid=((n,16),bool),recent_y=((n,64,7),np.float32),recent_mask=((n,64,7),bool))
        for key,(shape,dtype) in specs.items():arrays[key]=np.lib.format.open_memmap(cache/f'{split}_{key}.tmp',mode='w+',dtype=dtype,shape=shape)
        offset=0
        for data in DataLoader(ds,batch_size=batch):
            count=len(data['x']);m=torch.tensor(features['long'][offset:offset+count],device=device)
            arrays['teacher'][offset:offset+count]=model.decode_history(m)['reconstruction'].cpu().numpy()
            for key in ('y','mask','valid'):arrays[key][offset:offset+count]=data[key].numpy()
            recent=data['y'][:,-1,-64:].numpy().copy()
            for j,(i,end) in enumerate(zip(data['series'].tolist(),data['row'].tolist())):
                anchor64=float(series[i].frame.close.iloc[end-64]);anchor128=float(data['anchor'][j])
                recent[j,:,:2]-=100*np.log(anchor64/anchor128)
            arrays['recent_y'][offset:offset+count]=recent;arrays['recent_mask'][offset:offset+count]=data['mask'][:,-1,-64:].numpy()
            offset+=count
            if offset==n or offset%(batch*50)==0:progress(f'Reconstruction targets {split}: {offset}/{n}')
        for key in list(arrays):
            arrays[key].flush();del arrays[key];(cache/f'{split}_{key}.tmp').replace(cache/f'{split}_{key}.npy')
        for key in ('long','short','labels','keys'):
            with (cache/f'{split}_{key}.tmp').open('wb') as f:np.save(f,features[key],allow_pickle=False)
            (cache/f'{split}_{key}.tmp').replace(cache/f'{split}_{key}.npy')
        for key in (*specs,'long','short','labels','keys'):
            file=cache/f'{split}_{key}.npy';files[file.name]=fingerprint(file)
        coverage[split]=dict(windows=n,source_feature_sha256=sha)
    atomic_json(coverage,out/'coverage.json');atomic_json(dict(identity=identity,files=files),index)


def loss_parts(model,b,long_model=None):
    r=model(b['long'],b['short']);recent=recent_loss(r['recent'],b['recent_y'],b['recent_mask'])
    if model.mode=='add':
        def decode(z):return long_model.decode_history(z)['reconstruction']
        pred=checkpoint(decode,r['z'],use_reentrant=False) if torch.is_grad_enabled() else decode(r['z'])
        hist,close=history_values(pred,b['y'],b['mask'],b['valid'])
        teacher,_=history_values(pred,b['teacher'],b['mask'],b['valid'])
    elif model.mode=='long':
        hist,close=history_values(b['teacher'],b['y'],b['mask'],b['valid']);teacher=torch.zeros_like(recent)
    else:hist=close=teacher=torch.zeros_like(recent)
    # Baselines train only the identical recent decoder; history is constant for long.
    total=recent+hist+5*teacher if model.mode=='add' else recent
    return dict(total=total,recent=recent,history_loss=hist,history_close_bps=close,teacher=teacher)


@torch.no_grad()
def validate(model,ds,batch,device,long_model=None):
    model.eval();sums={};count=0
    for data in DataLoader(ds,batch_size=batch):
        b=move(data,device);parts=loss_parts(model,b,long_model);count+=len(b['long'])
        for k,v in parts.items():sums[k]=sums.get(k,0.)+float(v.sum())
    result={k:v/count for k,v in sums.items()}
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite reconstruction validation')
    return result


@torch.no_grad()
def baseline_metrics(ds,batch,device):
    loss=close=0.;count=0
    for data in DataLoader(ds,batch_size=batch):
        b=move(data,device);a,c=history_values(b['teacher'],b['y'],b['mask'],b['valid'])
        loss+=float(a.sum());close+=float(c.sum());count+=len(a)
    return dict(history_loss=loss/count,history_close_bps=close/count)


def save_state(state,path):
    atomic_save(state,path/'last.pt')
    atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],model=state['best_model']),path/'best.pt')
    tmp=path/'history.jsonl.tmp';tmp.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in state['history']));tmp.replace(path/'history.jsonl')
    atomic_json(dict(epoch=state['epoch'],best_epoch=state['best_epoch']),path/'progress.json')


def worker(out,mode,long_run,device='cuda'):
    meta=json.loads((out/'manifest.json').read_text());path=out/mode;path.mkdir(exist_ok=True)
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=ReconstructionFusion(mode).to(device);long_model=load_long(long_run/'joint/best.pt',device)[0] if mode=='add' else None
    opt=torch.optim.AdamW(model.parameters(),lr=meta['lr'],weight_decay=.01)
    train=CachedWindows(out/'target_cache','train');val=CachedWindows(out/'target_cache','val')
    micro=meta['fusion_micro'] if mode=='add' else meta['batch'];effective=meta['batch'];metadata={**meta,'mode':mode,'micro':micro}
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Reconstruction resume metadata mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    else:
        baseline=baseline_metrics(val,micro,device)
        initial=validate(model,val,micro,device,long_model)
        state=dict(metadata=metadata,epoch=0,best_epoch=0,best=initial['recent'],baseline=baseline,history=[],
                   best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        state.update(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state());save_state(state,path)
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started=time.perf_counter();model.train();opt.zero_grad();totals={};seen=0
        # First fit the new decoder while fusion remains an exact identity mapping.
        if mode=='add':
            model.project.requires_grad_(epoch>meta['warmup_epochs'])
            model.norm.requires_grad_(epoch>meta['warmup_epochs'])
        torch.cuda.reset_peak_memory_stats()
        loader=DataLoader(train,batch_size=micro,shuffle=True,generator=torch.Generator().manual_seed(42+epoch))
        for step,data in enumerate(loader):
            b=move(data,device);n=len(b['long']);parts=loss_parts(model,b,long_model)
            if not torch.isfinite(parts['total']).all():raise ValueError('Nonfinite reconstruction loss')
            group_start=(step//(effective//micro))*effective;group_size=min(effective,len(train)-group_start)
            (parts['total'].sum()/group_size).backward()
            if (step+1)%(effective//micro)==0 or step+1==len(loader):
                nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad()
            for k,v in parts.items():totals[k]=totals.get(k,0.)+float(v.detach().sum())
            seen+=n
        score=validate(model,val,micro,device,long_model)
        passed=mode!='add' or eligible(score,state['baseline'],meta['tolerance'])
        if passed and score['recent']<state['best']:
            state.update(best=score['recent'],best_epoch=epoch,best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        state['history'].append(dict(epoch=epoch,phase='decoder_warmup' if mode=='add' and epoch<=meta['warmup_epochs'] else 'train',train={k:v/seen for k,v in totals.items()},validation=score,eligible=passed,
                                    seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        state.update(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state());save_state(state,path)
        progress(f'Reconstruction {mode} epoch={epoch} recent={score["recent"]:.5f} retention_ok={passed} best={state["best_epoch"]}')
    save_state(state,path);progress(f'Reconstruction worker complete: {mode}')


def run_jobs(out,long_run,jobs):
    pending=list(MODES);active={};last_report={}
    def stop(signum,frame):raise SystemExit(128+signum)
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        while pending or active:
            while pending and len(active)<jobs:
                mode=pending.pop(0);(out/mode).mkdir(exist_ok=True)
                log=(out/mode/'run.log').open('a')
                cmd=[sys.executable,'-m','obson.babel.reconstruction_fusion','worker','--out',str(out),'--long-run',str(long_run),'--mode',mode]
                active[mode]=(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log)
                progress(f'Started parallel reconstruction worker: {mode}')
            for mode,(process,log) in list(active.items()):
                code=process.poll()
                status=out/mode/'progress.json'
                if status.exists():
                    try:report=json.loads(status.read_text())
                    except (ValueError,OSError):report=None
                    if report and report!=last_report.get(mode):progress(f'{mode}: {report}');last_report[mode]=report
                if code is not None:
                    log.close();del active[mode]
                    if code:raise RuntimeError(f'{mode} failed (exit {code}); inspect {out/mode/"run.log"}')
                    progress(f'Completed reconstruction worker: {mode}')
            if active:time.sleep(1)
    finally:
        for process,log in active.values():
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()
        signal.signal(signal.SIGTERM,previous)


@torch.no_grad()
def evaluate(out,source,long_run,root,batch,device):
    ref,series,encoded,sets=load_data(root,long_run)
    dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in series]
    hier=[HierWindows(ds,dummy,ref['boundaries'],s) for ds,s in zip(sets,('train','val','test'))]
    long_model,ck=load_long(long_run/'joint/best.pt',device)
    mean=torch.tensor(ck['metadata']['target_mean'],device=device);scale=torch.tensor(ck['metadata']['target_scale'],device=device)
    cache=out/'target_cache';report=dict(schema='babel-reconstruction-fusion-evaluation-v1',variants={},
        interpretation='Reconstruction-only training. Identical recent decoders. Frozen original historical decoder. All variants chosen on validation before test/readout. Previously reused research test period.')
    sources=json.loads((source/'manifest.json').read_text())['sources']
    original=[load_cache(source,s,sources)[0] for s in ('train','val','test')]
    for mode in MODES:
        model=ReconstructionFusion(mode).to(device);best=torch.load(out/mode/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(best['model']);model.eval();features=[];records={};cards=[]
        for split in ('train','val','test'):
            ds=CachedWindows(cache,split);zs=[];offset=0;chosen=set(np.linspace(0,len(ds)-1,min(6,len(ds)),dtype=int))
            for data in DataLoader(ds,batch_size=batch):
                b=move(data,device);r=model(b['long'],b['short']);zs.extend(r['z'].cpu().numpy())
                if split=='test':
                    for j in range(len(b['long'])):
                        y=data['recent_y'][j].numpy();pred=r['recent'][j].cpu().numpy()
                        for n in (16,32,64):records.setdefault(f'recent/{n}',[]).append(metrics(y[-n:],pred[-n:]))
                        if offset+j in chosen:
                            i,end=original[2]['keys'][offset+j];s=series[i];anchor=float(s.frame.close.iloc[end-64])
                            cards.append(dict(source=s.key,end=str(s.frame.datetime.iloc[end]),blocks=1,
                                              truth_ohlc=to_ohlc(y,anchor).tolist(),reconstructed_ohlc=to_ohlc(pred,anchor).tolist()))
                offset+=len(b['long'])
            features.append(np.asarray(zs))
        meta=best['metadata'];valds=CachedWindows(cache,'val')
        baseline=baseline_metrics(valds,batch,device)
        validation=validate(model,valds,batch,device,long_model if mode=='add' else None)
        result=dict(selected_epoch=best['epoch'],validation=validation,validation_retention_pass=(eligible(validation,baseline,meta['tolerance']) if mode=='add' else None),
                    recent_reconstruction=summarize(records),state_probe=probe(*(v for z,d in zip(features,original) for v in (z,d['labels'])),4))
        atomic_json(cards,out/mode/'recent_examples.json');write_global_review(out/mode/'recent_examples.html',cards)
        page=out/mode/'recent_examples.html';page.write_text(page.read_text().replace('单个512维综合向量重建历史','512维表示重建最近64根').replace('全局向量解码','对应表示经近期解码头重建'))
        if mode!='short':
            result['retention']={s:retention(model,long_model,ds,data,mean,scale,batch,device,out/mode,examples=s=='test')
                                 for s,ds,data in zip(('val','test'),hier[1:],original[1:])}
        else:result['retention']=None
        report['variants'][mode]=result;atomic_json(result,out/mode/'metrics.json');atomic_json(report,out/'reconstruction_metrics.json')
    atomic_json({mode:fingerprint(out/mode/'best.pt') for mode in MODES},out/'artifacts.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker'))
    p.add_argument('--out',required=True);p.add_argument('--long-run',required=True);p.add_argument('--source')
    p.add_argument('--root');p.add_argument('--mode',choices=MODES);p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',type=int,default=128);p.add_argument('--fusion-micro',type=int,default=8)
    p.add_argument('--eval-batch',type=int,default=16)
    a=p.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required; no local training fallback')
    out=Path(a.out);long_run=Path(a.long_run)
    if a.action=='worker':
        if a.mode is None:p.error('worker requires --mode')
        worker(out,a.mode,long_run);return
    if not a.source or not a.root:p.error('all requires --source and --root')
    if min(a.jobs,a.epochs,a.batch,a.fusion_micro,a.eval_batch)<1 or a.jobs>3 or a.batch%a.fusion_micro:raise ValueError('Invalid batch/job settings')
    source=Path(a.source);old=json.loads((source/'manifest.json').read_text())
    for split in ('train','val','test'):load_cache(source,split,old['sources'])
    long_sha=fingerprint(long_run/'joint/best.pt')
    if old['sources']['long_best']!=long_sha or old['sources']['long_manifest']!=fingerprint(long_run/'manifest.json'):raise ValueError('Long checkpoint differs from feature cache')
    identity=dict(fusion_manifest=fingerprint(source/'manifest.json'),long_best=long_sha,
                  feature_indexes={s:fingerprint(source/'feature_cache'/f'{s}.json') for s in ('train','val','test')})
    meta=dict(schema='babel-reconstruction-fusion-v1',sources=identity,modes=list(MODES),seed=42,epochs=a.epochs,batch=a.batch,fusion_micro=a.fusion_micro,
              lr=3e-4,tolerance=.02,warmup_epochs=min(5,a.epochs),loss_weights=dict(recent=1.,history=1.,teacher=5.),
              selection='Minimum validation recent objective, with both historical loss and close MAE <= 1.02 times frozen baseline for add. Initial identity epoch0 is eligible fallback.',
              protocol='Frozen encoders and original historical decoder; same 2-layer width256 recent decoder for all modes; no class-label training loss.')
    if out.exists() and any(out.iterdir()):
        if not (out/'manifest.json').exists() or json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('New settings require new run directory')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json')
    atomic_json(dict(jobs=a.jobs,eval_batch=a.eval_batch,gpu=torch.cuda.get_device_name()),out/'runtime.json')
    prepare_cache(source,long_run,a.root,out,identity,a.eval_batch,'cuda');torch.cuda.empty_cache()
    run_jobs(out,long_run,a.jobs)
    evaluate(out,source,long_run,a.root,a.eval_batch,'cuda')
    if long_sha!=fingerprint(long_run/'joint/best.pt'):raise ValueError('Frozen source changed during run')
    progress(f'Reconstruction matrix complete: {out}')


if __name__=='__main__':main()
