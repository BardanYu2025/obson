"""Causal distant-history targets and fixed readout/control baselines."""
import argparse
import json
from pathlib import Path

import numpy as np

from .ae_context import encode_context
from .ae_diagnostics import fingerprint
from .ae_extend import atomic_json
from .data import load_series, manifest
from .history_autoencoder import HistoryWindows
from .history_memory import WINDOW, STRIDE, history_indices
from .progress import progress

TARGETS=("distance_from_prior_high", "distance_from_prior_low", "prior_high_age")
BLOCKS=16


def distant_targets(frame, end, sigma):
    """Exclude current 128 bars; prior 15 blocks cover 1920 completed bars."""
    if end < BLOCKS*WINDOW-1 or sigma<=0:
        raise ValueError("Complete 2048-bar history and positive causal scale required")
    past=frame.iloc[end-BLOCKS*WINDOW+1:end-WINDOW+1]
    high=float(past.high.max()); low=float(past.low.min()); close=float(frame.close.iloc[end])
    # Slot 0 is nearest historical block; ties select the nearest block.
    block_high=past.high.to_numpy().reshape(BLOCKS-1,WINDOW).max(1)[::-1]
    return np.array([np.arcsinh(np.log(close/high)/sigma),
                     np.arcsinh(np.log(close/low)/sigma),
                     np.argmax(block_high)/(BLOCKS-2)],np.float64)


def projection():
    # Fixed random projection, no data fitting; preserves 16 coordinates per block.
    q,_=np.linalg.qr(np.random.default_rng(42).normal(size=(256,16)))
    return q.astype(np.float32)


def feature_row(bank, frame, end, sigma, project, shuffle_seed):
    ids=history_indices(end,len(bank),limit=BLOCKS)
    if len(ids)!=BLOCKS:
        raise ValueError("Complete memory required")
    current=bank[ids[0]]
    past=bank[ids[1:]]@project
    ends=127+ids[1:]*STRIDE
    anchors=np.array([frame.close.iloc[e-WINDOW] if e>=WINDOW else frame.open.iloc[0] for e in ends])
    offsets=np.arcsinh(np.log(anchors/float(frame.close.iloc[end]))/sigma)
    tokens=np.column_stack((past,offsets)).astype(np.float32)
    order=np.random.default_rng(shuffle_seed).permutation(BLOCKS-1)
    return {"ordered":np.r_[current,tokens.ravel()],
            "masked":np.r_[current,np.zeros(tokens.size,dtype=np.float32)],
            "shuffled":np.r_[current,tokens[order].ravel()]}


def dataset_rows(dataset,banks):
    rows={k:[] for k in ("ordered","masked","shuffled")};targets=[];anchors=[]
    project=projection()
    for i,end in dataset.items:
        if end<BLOCKS*WINDOW-1: continue
        frame=dataset.series[i].frame
        # The codec stores log(prior sigma) at input column 8, before current return update.
        sigma=float(np.exp(dataset.encoded[i]["x"][end,8]))
        row=feature_row(banks[i],frame,end,sigma,project,np.random.SeedSequence([42,i,end]))
        for k,v in row.items():rows[k].append(v)
        targets.append(distant_targets(frame,end,sigma));anchors.append((i,end))
    if not targets: raise ValueError("No eligible full-history benchmark windows")
    return {k:np.asarray(v) for k,v in rows.items()},np.asarray(targets),anchors


def regression_metrics(truth,pred):
    result={}
    for j,name in enumerate(TARGETS):
        a,b=truth[:,j],pred[:,j]
        variance=float(((a-a.mean())**2).sum())
        result[name]={"mae":float(np.abs(a-b).mean()),"rmse":float(np.sqrt(((a-b)**2).mean())),
                      "r2":float(1-((a-b)**2).sum()/variance) if variance>1e-12 else None}
    return result


def fit_readout(train,y,validation,v,test,t):
    mean,scale=train.mean(0),train.std(0).clip(.01)
    ym,ys=y.mean(0),y.std(0).clip(.01)
    a,b,c=[np.column_stack(((x-mean)/scale,np.ones(len(x)))).astype(np.float64) for x in (train,validation,test)]
    target=(y-ym)/ys
    lhs,rhs=a.T@a,a.T@target
    best=None
    for alpha in (1.,10.,100.):
        penalty=np.eye(a.shape[1])*alpha;penalty[-1,-1]=0
        weight=np.linalg.solve(lhs+penalty,rhs)
        score=float(np.mean((b@weight-(v-ym)/ys)**2))
        if best is None or score<best[0]:best=score,alpha,weight
    predicted=c@best[2]*ys+ym
    return {"alpha":best[1],"validation_standardized_mse":best[0],
            "test":regression_metrics(t,predicted)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True);p.add_argument("--memory-run",required=True);p.add_argument("--out",required=True)
    args=p.parse_args();memory,out=Path(args.memory_run),Path(args.out)
    source=json.loads((memory/"manifest.json").read_text())
    if source["schema"]!="babel-history-memory-v1":raise ValueError("Wrong memory schema")
    ref=source["manifest"]; keys=[s["key"].split("/") for s in ref["sources"]]
    series,_=load_series(args.root,sorted({k[0] for k in keys}),sorted({int(k[1]) for k in keys}))
    bounds=ref["boundaries"]
    if manifest(series,bounds)!=ref:raise ValueError("Benchmark data mismatch")
    banks=[];hashes={}
    for i,s in enumerate(series):
        path=memory/"cache"/f"{i:04d}.npy"
        digest=fingerprint(path)
        if digest!=json.loads(path.with_suffix(".json").read_text())["sha256"]:raise ValueError("Memory checksum mismatch")
        bank=np.load(path,allow_pickle=False)
        if bank.shape!=(len(range(127,len(s.frame),16)),256) or not np.isfinite(bank).all():raise ValueError("Invalid memory array")
        banks.append(bank);hashes[path.name]=digest
    protocol={"schema":"babel-distant-history-v1","memory_manifest_sha256":fingerprint(memory/"manifest.json"),
              "memory_cache_sha256":hashes,"source_checkpoint_sha256":source["source_artifacts_sha256"]["best.pt"],
              "manifest":ref,"targets":list(TARGETS),"history_bars":2048,"excluded_current_bars":128,
              "past_block_projection":16,"projection_seed":42,
              "controls":"fit ordered, masked and within-sample shuffled separately on identical windows",
              "selection":"train-only normalization; alpha 1/10/100 by validation standardized MSE"}
    if out.exists() and any(out.iterdir()):
        if not (out/"manifest.json").exists() or json.loads((out/"manifest.json").read_text())!=protocol:
            raise ValueError("Existing benchmark protocol differs; use a new output directory")
    out.mkdir(parents=True,exist_ok=True);atomic_json(protocol,out/"manifest.json")
    encoded=[encode_context(s.frame,s.period,"ema8_32") for s in series]
    datasets=[HistoryWindows(series,encoded,bounds,split) for split in ("train","val","test")]
    values=[]
    for split,ds in zip(("train","val","test"),datasets,strict=True):
        progress(f"Building distant-history benchmark {split}")
        values.append(dataset_rows(ds,banks))
    report={"schema":protocol["schema"],"protocol_sha256":fingerprint(out/"manifest.json"),
            "split_windows":dict(zip(("train","val","test"),(len(v[1]) for v in values))),
            "feature_dimensions":int(values[0][0]["ordered"].shape[1]),"readouts":{}}
    for kind in ("ordered","masked","shuffled"):
        progress(f"Fitting historical readout: {kind}; neural encoder unchanged")
        report["readouts"][kind]=fit_readout(*(v for f,y,_ in values for v in (f[kind],y)))
    ymean=values[0][1].mean(0)
    report["train_mean"]=regression_metrics(values[2][1],np.broadcast_to(ymean,values[2][1].shape))
    report["interpretation"]="All targets describe known past, not future returns. Prior high/low distances are permutation-invariant; high age tests order. Anchor offsets are legitimate causal inputs, not exact high/low targets. Linear readouts and random projection limit conclusions. Already-inspected test dates are development evidence only."
    atomic_json(report,out/"benchmark_metrics.json")
    progress(f"Distant-history benchmark complete: {out}")


if __name__=="__main__":main()
