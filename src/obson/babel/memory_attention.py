"""Small supervised historical aggregator, with a matched anchor-only control."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .ae_context import encode_context
from .ae_diagnostics import fingerprint
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .data import load_series, manifest
from .history_autoencoder import HistoryWindows
from .memory_benchmark import TARGETS, dataset_rows, fit_readout, regression_metrics
from .progress import progress


class MemoryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.current=nn.Linear(256,64)
        self.token=nn.Linear(17,64)
        self.age=nn.Linear(1,64,bias=False)
        self.memory=nn.TransformerEncoderLayer(64,4,128,dropout=.1,batch_first=True,norm_first=True)
        self.read=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
        self.norm=nn.LayerNorm(64)
        self.head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))

    def forward(self,x,anchors_only=False):
        if x.ndim!=2 or x.shape[1]!=511:raise ValueError("Expected fixed 511-dimensional benchmark inputs")
        current=self.current(x[:,:256])
        tokens=x[:,256:].reshape(-1,15,17)
        if anchors_only:
            tokens=torch.cat((torch.zeros_like(tokens[...,:16]),tokens[...,16:]),dim=-1)
        age=torch.arange(1,16,device=x.device,dtype=x.dtype).reshape(1,15,1)/15
        memory=self.memory(self.token(tokens)+self.age(age))
        read,_=self.read(current[:,None],memory,memory,need_weights=False)
        context=self.norm(current+read[:,0])
        return self.head(torch.cat((current,context),dim=-1))


def normalization(x,y):
    # Shared across conditions, estimated strictly from training inputs/targets.
    return {"x_mean":torch.tensor(x.mean(0)),"x_scale":torch.tensor(x.std(0).clip(.01)),
            "y_mean":torch.tensor(y.mean(0),dtype=torch.float32),
            "y_scale":torch.tensor(y.std(0).clip(.01),dtype=torch.float32)}


def tensors(x,y,stats):
    return TensorDataset((torch.tensor(x,dtype=torch.float32)-stats["x_mean"])/stats["x_scale"],
                         (torch.tensor(y,dtype=torch.float32)-stats["y_mean"])/stats["y_scale"])


@torch.no_grad()
def predict(model,dataset,device,batch_size,anchors_only):
    model.eval();out=[]
    for x,_ in DataLoader(dataset,batch_size=batch_size):
        out.append(model(x.to(device),anchors_only).cpu())
    return torch.cat(out)


def publish(state,path):
    atomic_json(state["metadata"],path/"manifest.json")
    atomic_save({"metadata":state["metadata"],"epoch":state["best_epoch"],
                 "model":state["best_model"],"normalization":state["normalization"]},path/"best.pt")
    target=path/"history.jsonl";temp=path/"history.jsonl.tmp"
    temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in state["history"]))
    temp.replace(target)


def train_arm(values,path,metadata,mode,device="cuda"):
    """Automatic recovery only when the full frozen experiment metadata matches."""
    metadata={**metadata,"condition":mode};epochs=metadata["epochs"];batch_size=metadata["batch_size"]
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=MemoryAttention().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    if (path/"last.pt").exists():
        state=torch.load(path/"last.pt",map_location="cpu",weights_only=True)
        if state.get("schema")!="babel-memory-attention-resume-v1" or state["metadata"]!=metadata:
            raise ValueError("Aggregator recovery metadata mismatch")
        stats=state["normalization"]
        model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    else:
        if path.exists() and any(path.iterdir()):raise ValueError("New aggregator arm must be empty or contain last.pt")
        stats=normalization(values[0][0]["ordered"],values[0][1])
        state=None
    tr,va=[tensors(f["ordered"],y,stats) for f,y,_ in values[:2]]
    anchors_only=mode=="anchors_only"
    if state is None:
        score=float(((predict(model,va,device,batch_size,anchors_only)-va.tensors[1])**2).mean())
        state={"schema":"babel-memory-attention-resume-v1","metadata":metadata,"epoch":0,
               "model":model.state_dict(),"optimizer":opt.state_dict(),"normalization":stats,
               "rng":rng_state(),"history":[],"best_loss":score,"best_epoch":0,
               "best_model":copy.deepcopy(model.state_dict())}
        path.mkdir(parents=True,exist_ok=True);atomic_save(state,path/"last.pt")
    publish(state,path)
    for epoch in range(state["epoch"]+1,epochs+1):
        start=time.perf_counter();model.train();total=0.;count=0
        for x,y in DataLoader(tr,batch_size=batch_size,shuffle=True):
            pred=model(x.to(device),anchors_only)
            loss=((pred-y.to(device))**2).mean()
            if not torch.isfinite(loss):raise ValueError("Nonfinite aggregator loss")
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=loss.item()*len(x);count+=len(x)
        score=float(((predict(model,va,device,batch_size,anchors_only)-va.tensors[1])**2).mean())
        if not np.isfinite(score):raise ValueError("Nonfinite validation loss")
        if score<state["best_loss"]:
            state.update(best_loss=score,best_epoch=epoch,best_model=copy.deepcopy(model.state_dict()))
        state["history"].append({"epoch":epoch,"train_standardized_mse":total/count,
                                  "validation_standardized_mse":score,"seconds":time.perf_counter()-start})
        state.update(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        atomic_save(state,path/"last.pt");publish(state,path)
        progress(f"{mode} epoch={epoch}/{epochs} train={total/count:.6f} val={score:.6f} best={state['best_epoch']}")


def prepare(root,memory,benchmark):
    protocol=json.loads((benchmark/"manifest.json").read_text())
    if (protocol["schema"]!="babel-distant-history-v1" or protocol["targets"]!=list(TARGETS)
            or protocol["history_bars"]!=2048 or protocol["excluded_current_bars"]!=128
            or protocol["past_block_projection"]!=16 or protocol["projection_seed"]!=42
            or protocol["memory_manifest_sha256"]!=fingerprint(memory/"manifest.json")):
        raise ValueError("Frozen benchmark protocol mismatch")
    ref=protocol["manifest"];keys=[s["key"].split("/") for s in ref["sources"]]
    series,_=load_series(root,sorted({k[0] for k in keys}),sorted({int(k[1]) for k in keys}))
    bounds=ref["boundaries"]
    if manifest(series,bounds)!=ref:raise ValueError("Aggregator data mismatch")
    banks=[]
    for i,s in enumerate(series):
        path=memory/"cache"/f"{i:04d}.npy"
        if fingerprint(path)!=protocol["memory_cache_sha256"][path.name]:raise ValueError("Memory cache changed")
        bank=np.load(path,allow_pickle=False)
        if bank.shape!=(len(range(127,len(s.frame),16)),256) or not np.isfinite(bank).all():raise ValueError("Bad cache")
        banks.append(bank)
    encoded=[encode_context(s.frame,s.period,"ema8_32") for s in series]
    values=[]
    for split in ("train","val","test"):
        progress(f"Preparing fixed historical targets: {split}")
        values.append(dataset_rows(HistoryWindows(series,encoded,bounds,split),banks))
    reference=json.loads((benchmark/"benchmark_metrics.json").read_text())
    if reference["protocol_sha256"]!=fingerprint(benchmark/"manifest.json"):
        raise ValueError("Benchmark report/protocol mismatch")
    if [reference["split_windows"][k] for k in ("train","val","test")]!=[len(v[1]) for v in values]:
        raise ValueError("Benchmark sample counts changed")
    return values,reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True);p.add_argument("--memory-run",required=True)
    p.add_argument("--benchmark-run",required=True);p.add_argument("--out",required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise ValueError("GPU required for aggregator training; no local fallback")
    memory,benchmark,out=Path(args.memory_run),Path(args.benchmark_run),Path(args.out)
    source_hashes={n:fingerprint(benchmark/n) for n in ("manifest.json","benchmark_metrics.json")}
    values,reference=prepare(args.root,memory,benchmark)
    metadata={"schema":"babel-memory-attention-v1","source_benchmark_sha256":source_hashes,
              "epochs":50,"batch_size":256,"seed":42,"hidden":64,"heads":4,
              "memory_self_attention_layers":1,"cross_attention_layers":1,"dropout":.1,
              "parameters":sum(p.numel() for p in MemoryAttention().parameters()),
              "targets":list(TARGETS),"input":"same 511 raw coordinates as frozen linear benchmark; local256 + 15*(projection16+anchor1)",
              "optimizer":{"name":"AdamW","lr":3e-4,"weight_decay":.01,"grad_clip":1.},
              "selection":"best original validation standardized MSE; test only after both arms finish",
              "runtime":{"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__)}}
    if out.exists() and any(out.iterdir()):
        if not (out/"manifest.json").exists() or json.loads((out/"manifest.json").read_text())!=metadata:
            raise ValueError("Use a new run directory for changed experiment settings")
    out.mkdir(parents=True,exist_ok=True);atomic_json(metadata,out/"manifest.json")
    for mode in ("full","anchors_only"):train_arm(values,out/mode,metadata,mode)
    result={"schema":metadata["schema"],"manifest_sha256":fingerprint(out/"manifest.json"),
            "linear_reference":reference,"attention":{}}
    for mode in ("full","anchors_only"):
        path=out/mode;ck=torch.load(path/"best.pt",map_location="cpu",weights_only=True)
        model=MemoryAttention().to("cuda");model.load_state_dict(ck["model"]);stats=ck["normalization"]
        result["attention"][mode]={"selected_epoch":ck["epoch"],"evaluations":{}}
        for kind in ("ordered","shuffled"):
            f,y,_=values[2];ds=tensors(f[kind],y,stats)
            pred=predict(model,ds,"cuda",256,mode=="anchors_only")*stats["y_scale"]+stats["y_mean"]
            result["attention"][mode]["evaluations"][kind]=regression_metrics(y,pred.numpy())
        result["attention"][mode]["artifacts_sha256"]={n:fingerprint(path/n) for n in ("best.pt","history.jsonl")}
    anchor_values=[]
    for f,y,_ in values:
        x=f["ordered"].copy();tokens=x[:,256:].reshape(-1,15,17);tokens[...,:16]=0
        anchor_values.extend((x,y))
    progress("Fitting matched linear anchor-only control")
    result["linear_anchors_only"]=fit_readout(*anchor_values)
    result["interpretation"]="Task-specific supervised historical aggregator, not a universal bar embedding or forecast model. Anchor-only retains current embedding. Shuffled evaluations are test-time interventions on trained models, unlike separately fitted shuffled ridge. Distances are permutation-invariant; age depends on order. Prior inspected test dates remain development evidence."
    if source_hashes!={n:fingerprint(benchmark/n) for n in source_hashes}:raise ValueError("Source benchmark changed")
    atomic_json(result,out/"attention_metrics.json")
    progress(f"Saved {out/'attention_metrics.json'}; frozen local encoder untouched")


if __name__=="__main__":main()
