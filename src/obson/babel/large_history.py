"""512-dimensional local and hierarchical history model, trained in three stages."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from .ae_context import encode_context
from .ae_diagnostics import fingerprint,metrics,summarize
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .autoregressive import probe
from .data import load_series, manifest
from .history_autoencoder import HistoryAE, HistoryWindows, positions, reconstruction_loss, evaluate, write_review,to_ohlc
from .memory_benchmark import regression_metrics
from .progress import progress
from .representation import time_mask

CONFIG=dict(hidden=512,layers=8,heads=8,latent=512,window=128,dropout=.1,context="ema8_32")
BLOCKS=16


class CheckpointLocal(HistoryAE):
    """Equivalent local architecture; checkpoint activations to bound joint-training memory."""
    def encode(self,x):
        if x.shape[-1]!=18:raise ValueError("Expected causal EMA inputs")
        t=x.shape[1];mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        h=self.input(x[...,:14])+self.context_input(x[...,14:])+positions(t,self.config["hidden"],x.device,x.dtype)
        for layer in self.encoder.layers:
            if self.training and torch.is_grad_enabled():
                h=checkpoint(lambda z,layer=layer:layer(z,src_mask=mask),h,use_reentrant=False)
            else:h=layer(h,src_mask=mask)
        return self.compress(h)

    def decode(self,z):
        if z.ndim!=2 or z.shape[-1]!=self.config["latent"]:raise ValueError("Single local vector required")
        h=self.expand(z)[:,None]+positions(self.config["window"],self.config["hidden"],z.device,z.dtype)
        for layer in self.decoder.layers:
            if self.training and torch.is_grad_enabled():h=checkpoint(layer,h,use_reentrant=False)
            else:h=layer(h)
        raw=self.output(h)
        return torch.cat((raw[...,:2],nn.functional.softplus(raw[...,2:])),dim=-1)


class LargeHistory(nn.Module):
    def __init__(self,config=None,blocks=BLOCKS,aggregate_layers=4):
        super().__init__();self.config=dict(CONFIG if config is None else config);self.blocks=blocks
        self.local=CheckpointLocal(**self.config)
        d=self.config["latent"]
        self.anchor_input=nn.Linear(1,d)
        layer=nn.TransformerEncoderLayer(d,self.config["heads"],d*4,dropout=self.config["dropout"],batch_first=True,norm_first=True)
        self.aggregate=nn.TransformerEncoder(layer,aggregate_layers,enable_nested_tensor=False)
        self.summary=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.LayerNorm(d))
        self.history_decoder=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
        self.anchor_decoder=nn.Linear(d,1)
        self.structure=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,3))

    def summarize(self,local_z,anchors,valid):
        if local_z.shape[1]!=self.blocks or not valid[:,-1].all():raise ValueError("Current block must be valid")
        # Only final state is exposed: every token and anchor is already known at its time.
        h=local_z+self.anchor_input(torch.asinh(anchors)[...,None])
        h=h+positions(self.blocks,self.config["latent"],h.device,h.dtype)
        for layer in self.aggregate.layers:
            if self.training and torch.is_grad_enabled():
                h=checkpoint(lambda z,layer=layer:layer(z,src_key_padding_mask=~valid),h,use_reentrant=False)
            else:h=layer(h,src_key_padding_mask=~valid)
        return self.summary(h[:,-1])

    def decode_history(self,z):
        if z.ndim!=2 or z.shape[-1]!=self.config["latent"]:raise ValueError("Only one global vector is accepted")
        queries=z[:,None]+positions(self.blocks,self.config["latent"],z.device,z.dtype)
        latents=self.history_decoder(queries)
        anchors=self.anchor_decoder(latents).squeeze(-1)
        anchors=torch.cat((anchors[:,:-1],torch.zeros_like(anchors[:,-1:])),dim=1)
        b,k,d=latents.shape
        local=self.local.decode(latents.reshape(b*k,d)).reshape(b,k,self.config["window"],7)
        price=local[...,:2]+anchors[:,:,None,None]
        return {"reconstruction":torch.cat((price,local[...,2:]),-1),"latents":latents,"anchors":anchors}

    def forward(self,batch,joint=False):
        if joint:
            b,k,w,f=batch["x"].shape
            local_z=self.local.encode(batch["x"].reshape(b*k,w,f))[:,-1].reshape(b,k,-1)
        else:local_z=batch["teacher"]
        z=self.summarize(local_z,batch["offsets"],batch["valid"])
        result={**self.decode_history(z),"z":z,"structure":self.structure(z),"local_z":local_z}
        return result


class HierWindows(Dataset):
    """4–16 nonoverlapping blocks, all scored bars inside the requested time split."""
    def __init__(self,base,banks,bounds,split,blocks=BLOCKS,minimum=4):
        self.base,self.banks,self.blocks=base,banks,blocks;self.items=[]
        starts=[]
        for s in base.series:
            valid=np.flatnonzero(time_mask(s,bounds,split));starts.append(int(valid[0]) if len(valid) else len(s.frame))
        for i,end in base.items:
            if (end-127)%128:continue
            n=min(blocks,(end+1-starts[i])//128)
            if n>=minimum and time_mask(base.series[i],bounds,split)[end+1-n*128:end+1].all():self.items.append((i,end,n))
        if not self.items:raise ValueError(f"No complete hierarchical {split} samples")

    def __len__(self):return len(self.items)

    def __getitem__(self,j):
        i,end,n=self.items[j];s=self.base.series[i];e=self.base.encoded[i];k=self.blocks
        lo=end+1-n*128;current_anchor=float(s.frame.close.iloc[end-128]);reference=np.log(current_anchor)*100
        x=np.zeros((k,128,18),np.float32);y=np.zeros((k,128,7),np.float32);mask=np.zeros((k,128,7),bool)
        teacher=np.zeros((k,self.banks[i].shape[1]),np.float32);offsets=np.zeros(k,np.float32);valid=np.zeros(k,bool)
        x[-n:]=e["x"][lo:end+1].reshape(n,128,18)
        target=self.base.targets[i][lo:end+1].copy();target[:,:2]-=reference
        available=e["available"][lo:end+1];target[:,5]=np.where(available,target[:,5],0)
        y[-n:]=target.reshape(n,128,7);mask[-n:]=True;mask[-n:,:,5]=available.reshape(n,128);valid[-n:]=True
        for slot,block_end in enumerate(range(lo+127,end+1,128),k-n):
            anchor=float(s.frame.close.iloc[block_end-128] if block_end>=128 else s.frame.open.iloc[0])
            offsets[slot]=np.log(anchor)*100-reference
            teacher[slot]=self.banks[i][(block_end-127)//128]
        past=s.frame.iloc[lo:end-127];sigma=float(np.exp(e["x"][end,8]));c=float(s.frame.close.iloc[end])
        highs=past.high.to_numpy().reshape(n-1,128).max(1)[::-1]
        targets=np.array([np.arcsinh(np.log(c/past.high.max())/sigma),np.arcsinh(np.log(c/past.low.min())/sigma),
                          np.argmax(highs)/max(n-2,1)],np.float32)
        return dict(x=x,y=y,mask=mask,teacher=teacher,offsets=offsets,valid=valid,targets=targets,
                    anchor=current_anchor,series=i,row=end,blocks=n)


def objective(model,b,result,mean,scale,joint):
    valid=b["valid"]
    raw=reconstruction_loss(result["reconstruction"][valid],b["y"][valid],b["mask"][valid]).mean()
    latent=nn.functional.smooth_l1_loss(result["latents"][valid],b["teacher"][valid])
    anchors=nn.functional.smooth_l1_loss(result["anchors"][valid],b["offsets"][valid])
    structure=((result["structure"]-(b["targets"]-mean)/scale)**2).mean()
    loss=raw+.1*latent+.1*anchors+.1*structure
    if joint:
        local=model.local.decode(result["local_z"][valid])
        target=b["y"][valid].clone();target[...,:2]-=b["offsets"][valid][:,None,None]
        loss=loss+.25*reconstruction_loss(local,target,b["mask"][valid]).mean()
    return loss,dict(global_reconstruction=raw,latent=latent,anchor=anchors,structure=structure)


def move(batch,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def set_stage(model,stage,training):
    model.train(training)
    for p in model.parameters():p.requires_grad_(stage!="local")
    for p in model.local.parameters():p.requires_grad_(stage!="aggregate")
    if stage=="aggregate":model.local.eval()


@torch.no_grad()
def validation(model,dataset,stage,device,micro,mean,scale):
    model.eval();total=0.;count=0
    for batch in DataLoader(dataset,batch_size=micro):
        b=move(batch,device)
        if stage=="local":loss=reconstruction_loss(model.local(b["x"])["reconstruction"],b["y"],b["mask"]).mean()
        else:loss,_=objective(model,b,model(b,joint=stage=="joint"),mean,scale,stage=="joint")
        total+=float(loss)*len(b["x"]);count+=len(b["x"])
    score=total/count
    if not np.isfinite(score):raise ValueError("Nonfinite validation")
    return score


def publish(state,path):
    atomic_json(state["metadata"],path/"manifest.json")
    identity=dict(metadata=state["metadata"],epoch=state["best_epoch"])
    marker=path/"best_identity.json"
    previous=json.loads(marker.read_text()) if marker.exists() else {}
    # Preserve best.pt bytes across recovery; downstream stage/cache fingerprints depend on them.
    if previous.get("identity")!=identity or not (path/"best.pt").exists() or previous.get("sha256")!=fingerprint(path/"best.pt"):
        atomic_save(dict(**identity,model=state["best_model"]),path/"best.pt")
        atomic_json(dict(identity=identity,sha256=fingerprint(path/"best.pt")),marker)
    temp=path/"history.jsonl.tmp";temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in state["history"]))
    temp.replace(path/"history.jsonl")


def train_stage(model,datasets,path,metadata,stage,device):
    tr,va=datasets[:2];metadata={**metadata,"stage":stage};micro=metadata["micro"];effective=metadata["effective"]
    if effective%micro:raise ValueError("Effective batch must be divisible by micro batch")
    set_stage(model,stage,True)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=metadata["lr"],weight_decay=.01)
    mean=torch.tensor(metadata["target_mean"],device=device);scale=torch.tensor(metadata["target_scale"],device=device)
    if (path/"last.pt").exists():
        state=torch.load(path/"last.pt",map_location="cpu",weights_only=True)
        if state["metadata"]!=metadata:raise ValueError("Stage recovery metadata mismatch")
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);restore_rng(state["rng"])
    else:
        if path.exists() and any(path.iterdir()):raise ValueError("Stage directory must be empty or recoverable")
        score=validation(model,va,stage,device,micro,mean,scale)
        state=dict(metadata=metadata,epoch=0,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                   best_model=copy.deepcopy(model.state_dict()),best_epoch=0,best_loss=score,history=[])
        path.mkdir(parents=True,exist_ok=True);atomic_save(state,path/"last.pt")
    publish(state,path)
    for epoch in range(state["epoch"]+1,metadata["epochs"]+1):
        started=time.perf_counter();set_stage(model,stage,True);loader=DataLoader(tr,batch_size=micro,shuffle=True)
        total=0.;seen=0;opt.zero_grad()
        for step,batch in enumerate(loader):
            b=move(batch,device);n=len(b["x"])
            if stage=="local":loss=reconstruction_loss(model.local(b["x"])["reconstruction"],b["y"],b["mask"]).mean()
            else:loss,_=objective(model,b,model(b,joint=stage=="joint"),mean,scale,stage=="joint")
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            # Sample-weighted accumulation, including the final partial effective batch.
            group_start=(step//(effective//micro))*effective;group_size=min(effective,len(tr)-group_start)
            (loss*n/group_size).backward();total+=float(loss.detach())*n;seen+=n
            if (step+1)%(effective//micro)==0 or step+1==len(loader):
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.);opt.step();opt.zero_grad()
            if step==0 or (step+1)%100==0:progress(f"512 {stage} epoch={epoch} batch={step+1}/{len(loader)} loss={total/seen:.5f}")
        score=validation(model,va,stage,device,micro,mean,scale)
        if score<state["best_loss"]:state.update(best_loss=score,best_epoch=epoch,best_model=copy.deepcopy(model.state_dict()))
        state["history"].append(dict(epoch=epoch,train_loss=total/seen,validation_loss=score,seconds=time.perf_counter()-started))
        state.update(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        atomic_save(state,path/"last.pt");publish(state,path)
        progress(f"512 {stage} epoch={epoch} val={score:.6f} best={state['best_epoch']}")
    model.load_state_dict(state["best_model"])


@torch.no_grad()
def teacher_banks(model,series,encoded,path,source_hash,device,batch_size):
    path.mkdir(exist_ok=True);model.local.eval();banks=[]
    for i,(s,e) in enumerate(zip(series,encoded,strict=True)):
        file=path/f"{i:04d}.npy";index=path/f"{i:04d}.json"
        key=dict(source=source_hash,contract=s.key,data=s.source_hash)
        info=json.loads(index.read_text()) if index.exists() else {}
        if file.exists() and info.get("key")==key and info.get("sha256")==fingerprint(file):bank=np.load(file,allow_pickle=False)
        else:
            ends=np.arange(127,len(s.frame),128);rows=[]
            for start in range(0,len(ends),batch_size):
                x=np.stack([e["x"][end-127:end+1] for end in ends[start:start+batch_size]])
                rows.append(model.local.encode(torch.tensor(x,device=device))[:,-1].cpu().numpy())
            bank=np.concatenate(rows) if rows else np.empty((0,512),np.float32)
            with file.with_suffix(".tmp").open("wb") as f:np.save(f,bank,allow_pickle=False)
            file.with_suffix(".tmp").replace(file);atomic_json(dict(key=key,sha256=fingerprint(file)),index)
        if bank.shape!=(len(range(127,len(s.frame),128)),512) or not np.isfinite(bank).all():raise ValueError("Bad teacher cache")
        banks.append(bank)
        if i==0 or (i+1)%25==0:progress(f"512 teacher cache {i+1}/{len(series)}")
    return banks


def target_stats(dataset):
    y=np.stack([dataset[i]["targets"] for i in range(len(dataset))])
    return y.mean(0).tolist(),y.std(0).clip(.01).tolist()


def preflight_batch(n,device):
    """Keep synthetic inputs and attention masks on the requested device."""
    x=torch.randn(n,16,128,18,device=device)
    y=torch.randn(n,16,128,7,device=device);y[...,2:]=y[...,2:].abs()
    return dict(x=x,y=y,mask=torch.ones_like(y,dtype=torch.bool),teacher=torch.randn(n,16,512,device=device),
                offsets=torch.zeros(n,16,device=device),valid=torch.ones(n,16,dtype=torch.bool,device=device),
                targets=torch.randn(n,3,device=device))


def preflight(path,settings):
    """Disposable synthetic GPU workload including optimizer allocations; no real-data updates."""
    model=LargeHistory().cuda();report={"gpu":torch.cuda.get_device_name(),"stages":{},"completed":False}
    for stage in ("local","aggregate","joint"):
        report["running_stage"]=stage;atomic_json(report,path/"preflight.json")
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();set_stage(model,stage,True)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4)
        n=settings[stage]["micro"];b=preflight_batch(n,"cuda")
        if stage=="local":loss=reconstruction_loss(model.local(b["x"][:,-1])["reconstruction"],b["y"][:,-1],b["mask"][:,-1]).mean()
        else:loss,_=objective(model,b,model(b,joint=stage=="joint"),torch.zeros(3,device="cuda"),torch.ones(3,device="cuda"),stage=="joint")
        if not torch.isfinite(loss):raise ValueError("Preflight loss is not finite")
        loss.backward();opt.step();torch.cuda.synchronize()
        report["stages"][stage]={"micro":n,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                                 "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30}
        atomic_json(report,path/"preflight.json")
        model.zero_grad(set_to_none=True);del loss,opt,b
        progress(f"512 preflight {stage}: {report['stages'][stage]}")
    report.update(completed=True,running_stage=None);atomic_json(report,path/"preflight.json")


@torch.no_grad()
def evaluate_large(model,local_sets,hier_sets,out,mean,scale,micro,local_micro=None):
    model.eval()
    report,local_examples=evaluate(model.local,local_sets,"cuda",local_micro or max(micro,4),42)
    atomic_json(report,out/"local_metrics.json");write_review(out/"local_examples.html",local_examples)
    records={};predictions=[];truth=[];features=[];cards=[]
    chosen=set(np.linspace(0,len(hier_sets[-1])-1,min(6,len(hier_sets[-1])),dtype=int).tolist())
    for split,ds in zip(("train","val","test"),hier_sets,strict=True):
        vectors=[];locals_=[];labels=[];offset=0
        for batch in DataLoader(ds,batch_size=micro):
            b=move(batch,"cuda");result=model(b,joint=True)
            vectors.extend(result["z"].cpu().numpy());locals_.extend(result["local_z"][:,-1].cpu().numpy())
            for i,end in zip(batch["series"].tolist(),batch["row"].tolist(),strict=True):
                labels.append(int(ds.base.series[i].labels["state"][end,1]))
            if split=="test":
                predictions.extend((result["structure"]*scale+mean).cpu().numpy());truth.extend(batch["targets"].numpy())
                for j in range(len(batch["x"])):
                    valid=batch["valid"][j].numpy();y=batch["y"][j].numpy()[valid].reshape(-1,7)
                    pred=result["reconstruction"][j].cpu().numpy()[valid].reshape(-1,7)
                    n=int(valid.sum());row=metrics(y,pred)
                    for group in ("all",f"blocks/{n}"):
                        records.setdefault(group,[]).append(row)
                    if offset+j in chosen:
                        i=int(batch["series"][j]);end=int(batch["row"][j]);anchor=float(batch["anchor"][j])
                        cards.append(dict(source=ds.base.series[i].key,end=str(ds.base.series[i].frame.datetime.iloc[end]),blocks=n,
                                          truth_ohlc=to_ohlc(y,anchor).tolist(),reconstructed_ohlc=to_ohlc(pred,anchor).tolist()))
            offset+=len(batch["x"])
            if offset==len(ds) or offset%(micro*100)==0:progress(f"512 evaluation {split}: {offset}/{len(ds)}")
        features.append((np.asarray(vectors),np.asarray(locals_),np.asarray(labels)))
    result=dict(schema="babel-large-history-evaluation-v1",hierarchical_reconstruction=summarize(records),
                historical_structure=regression_metrics(np.asarray(truth),np.asarray(predictions)),state_probes={})
    for name,column in (("global512",0),("local512",1)):
        result["state_probes"][name]=probe(*(v for row in features for v in (row[column],row[2])),4)
    result["interpretation"]="New staged model; hierarchical windows have 4-16 blocks entirely within split. These samples differ from earlier carry-over history benchmark. Global reconstruction receives only global z plus fixed positions and one external current-window price anchor for physical units. Auxiliary historical targets use variable available history. No forecast claims."
    atomic_json(result,out/"hierarchical_metrics.json");atomic_json(cards,out/"hierarchical_examples.json")
    write_global_review(out/"hierarchical_examples.html",cards)


def write_global_review(path,cards):
    import html
    sections=[]
    for card in cards:
        arrays=[np.asarray(card[k])[:,3] for k in ("truth_ohlc","reconstructed_ohlc")]
        low=min(a.min() for a in arrays);high=max(a.max() for a in arrays);lines=[]
        for a,color in zip(arrays,("#222222","#d65330"),strict=True):
            points=' '.join(f'{20+j*960/max(len(a)-1,1):.2f},{270-(v-low)*240/max(high-low,1e-12):.2f}' for j,v in enumerate(a))
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        sections.append(f'<section><h2>{html.escape(card["source"])} · {html.escape(card["end"])} · {card["blocks"]}片段</h2><svg viewBox="0 0 1000 300">{"".join(lines)}</svg></section>')
    Path(path).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>512维历史重建</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}svg{width:100%}section{margin:25px 0;background:#f5f5f2;padding:15px}</style><h1>单个512维综合向量重建历史</h1><p>黑线：真实历史收盘价；橙线：全局向量解码。每组共用价格轴。固定等距抽取测试样例，不是未来预测。完整OHLC保存在JSON中。</p>'+''.join(sections)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("all","preflight","evaluate"))
    p.add_argument("--root",required=True);p.add_argument("--reference",required=True);p.add_argument("--out",required=True)
    for name,epochs,micro,effective in (("local",200,16,32),("aggregate",30,2,16),("joint",20,1,8)):
        p.add_argument(f"--{name}-epochs",type=int,default=epochs);p.add_argument(f"--{name}-micro",type=int,default=micro)
        p.add_argument(f"--{name}-effective",type=int,default=effective)
    args=p.parse_args()
    if not torch.cuda.is_available():raise ValueError("This pipeline requires AutoDL CUDA; no local training fallback")
    out=Path(args.out);settings={}
    for name in ("local","aggregate","joint"):
        setting=dict(epochs=getattr(args,f"{name}_epochs"),micro=getattr(args,f"{name}_micro"),effective=getattr(args,f"{name}_effective"),lr=3e-5 if name=="joint" else 3e-4)
        if min(setting[k] for k in ("epochs","micro","effective"))<1 or setting["effective"]%setting["micro"]:raise ValueError("Invalid stage batch/budget")
        settings[name]=setting
    ref=json.loads(Path(args.reference).read_text())["manifest"]
    metadata=dict(schema="babel-large-history-v1",config=CONFIG,aggregate_layers=4,max_blocks=16,min_blocks=4,
                  seed=42,settings=settings,reference_sha256=fingerprint(args.reference),manifest=ref,
                  hierarchical_stride=128,hierarchical_policy="all scored history within split; 4-16 valid blocks",
                  objectives=dict(global_reconstruction=1.,teacher_latent=.1,anchor=.1,structure=.1,joint_local=.25),
                  runtime=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)),
                  interpretation="Three-stage training from scratch. No autoregressive forecast loss. Fixed local teacher after stage 1. External single price anchor for physical global decode.")
    if out.exists() and any(out.iterdir()):
        if not (out/"manifest.json").exists() or json.loads((out/"manifest.json").read_text())!=metadata:raise ValueError("Existing large run settings differ; use a new directory")
    out.mkdir(parents=True,exist_ok=True);atomic_json(metadata,out/"manifest.json")
    preflight_ok=(out/"preflight.json").exists() and json.loads((out/"preflight.json").read_text()).get("completed",False)
    if args.stage=="preflight" or not preflight_ok:preflight(out,settings)
    if args.stage=="preflight":return
    keys=[s["key"].split("/") for s in ref["sources"]]
    series,_=load_series(args.root,sorted({k[0] for k in keys}),sorted({int(k[1]) for k in keys}))
    bounds=ref["boundaries"]
    if manifest(series,bounds)!=ref:raise ValueError("512 source fingerprint mismatch")
    encoded=[encode_context(s.frame,s.period,"ema8_32") for s in series]
    local_sets=[HistoryWindows(series,encoded,bounds,split) for split in ("train","val","test")]
    random.seed(42);np.random.seed(42);torch.manual_seed(42);model=LargeHistory().cuda()
    local_meta={**settings["local"],"target_mean":[0.,0.,0.],"target_scale":[1.,1.,1.],"parent":fingerprint(out/"manifest.json")}
    if args.stage=="all":train_stage(model,local_sets,out/"local",local_meta,"local","cuda")
    local_ck=torch.load(out/"local/best.pt",map_location="cpu",weights_only=True);model.load_state_dict(local_ck["model"])
    banks=teacher_banks(model,series,encoded,out/"teacher_cache",fingerprint(out/"local/best.pt"),"cuda",settings["local"]["micro"])
    hier_sets=[HierWindows(base,banks,bounds,split) for base,split in zip(local_sets,("train","val","test"),strict=True)]
    coverage={split:dict(windows=len(ds),full16=sum(n==16 for _,_,n in ds.items),
                        periods={str(period):sum(ds.base.series[i].period==period for i,_,_ in ds.items) for period in sorted({s.period for s in series})})
              for split,ds in zip(("train","val","test"),hier_sets,strict=True)}
    atomic_json(coverage,out/"coverage.json")
    mean,scale=target_stats(hier_sets[0])
    for index,stage in enumerate(("aggregate","joint"),43):
        parent="local" if stage=="aggregate" else "aggregate"
        ck=torch.load(out/parent/"best.pt",map_location="cpu",weights_only=True);model.load_state_dict(ck["model"])
        stage_meta={**settings[stage],"target_mean":mean,"target_scale":scale,"parent":fingerprint(out/parent/"best.pt")}
        random.seed(index);np.random.seed(index);torch.manual_seed(index)
        if args.stage=="all":train_stage(model,hier_sets,out/stage,stage_meta,stage,"cuda")
    final=torch.load(out/"joint/best.pt",map_location="cpu",weights_only=True);model.load_state_dict(final["model"])
    evaluate_large(model,local_sets,hier_sets,out,torch.tensor(mean,device="cuda"),torch.tensor(scale,device="cuda"),settings["joint"]["micro"])
    atomic_json({stage:{name:fingerprint(out/stage/name) for name in ("best.pt","history.jsonl")} for stage in settings},out/"artifacts.json")
    progress(f"512 pipeline complete: {out}")


if __name__=="__main__":main()
