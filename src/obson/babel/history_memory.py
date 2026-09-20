"""Frozen local embeddings, causal historical memory and multiscale state probes."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .ae_capacity import config
from .ae_context import encode_context
from .ae_diagnostics import fingerprint
from .ae_extend import atomic_json
from .autoregressive import probe
from .data import load_series, manifest
from .history_autoencoder import HistoryAE, HistoryWindows, SCHEMA
from .progress import progress

WINDOW, STRIDE = 128, 16


@torch.no_grad()
def encode_bank(model, encoded, device, batch_size):
    """Overlapping cache supports anchor alignment; reads select nonoverlapping blocks."""
    ends = np.arange(WINDOW - 1, len(encoded["x"]), STRIDE)
    values = []
    model.eval()
    for start in range(0, len(ends), batch_size):
        batch = ends[start:start + batch_size]
        x = np.stack([encoded["x"][end - WINDOW + 1:end + 1] for end in batch])
        values.append(model.encode(torch.as_tensor(x, device=device))[:, -1].cpu().numpy())
    return np.concatenate(values) if values else np.empty((0, model.config["latent"]), np.float32)


def history_indices(end, bank_length, limit=None, include_current=True):
    if end < WINDOW - 1 or (end - WINDOW + 1) % STRIDE:
        raise ValueError("Memory anchor is not on the frozen window grid")
    index = (end - WINDOW + 1) // STRIDE
    if index >= bank_length:
        raise ValueError("Memory anchor exceeds cache")
    first = index if include_current else index - WINDOW // STRIDE
    indices = np.arange(first, -1, -WINDOW // STRIDE, dtype=int)
    return indices if limit is None else indices[:limit]


def read_memory(bank, end):
    ids = history_indices(end, len(bank))
    current = bank[ids[0]]
    recent = bank[ids[:4]].mean(0)
    long = bank[ids[:16]].mean(0)
    return {"local256":current, "mean4_256":recent, "mean16_256":long,
            "multiscale768":np.concatenate((current, recent, long))}, len(ids)


def retrieve_past(bank, end, topk=3):
    ids = history_indices(end, len(bank), include_current=False)
    if not len(ids):
        return []
    query = bank[(end - WINDOW + 1) // STRIDE]
    norms = np.linalg.norm(bank[ids], axis=1) * np.linalg.norm(query)
    scores = (bank[ids] @ query) / np.maximum(norms, 1e-12)
    order = np.argsort(-scores, kind="stable")[:topk]
    return [{"end_row":int(WINDOW - 1 + ids[j] * STRIDE), "cosine":float(scores[j]),
             "age_bars":int(end - (WINDOW - 1 + ids[j] * STRIDE))} for j in order]


def state_features(dataset, banks):
    features, labels, counts = {}, [], []
    for i, end in dataset.items:
        values, count = read_memory(banks[i], end)
        for name, value in values.items():
            features.setdefault(name, []).append(value)
        labels.append(dataset.series[i].labels["state"][end])
        counts.append(count)
    return {k:np.asarray(v, np.float32) for k,v in features.items()}, np.asarray(labels), np.asarray(counts)


def bank_index(s):
    ends=np.arange(WINDOW-1,len(s.frame),STRIDE)
    lows=ends-WINDOW+1
    anchors=[float(s.frame.close.iloc[lo-1] if lo else s.frame.open.iloc[0]) for lo in lows]
    return {"source":s.key,"period_minutes":int(s.period),"end_rows":ends.tolist(),
            "price_anchors":anchors,
            "available_at":[str(s.frame.datetime.iloc[end]+pd.Timedelta(minutes=s.period)) for end in ends]}


def snapshot(s, end):
    frame = s.frame.iloc[end - WINDOW + 1:end + 1]
    return {"source":s.key, "end_row":int(end), "start":str(frame.datetime.iloc[0]),
            "available_at":str(frame.datetime.iloc[-1] + pd.Timedelta(minutes=s.period)),
            "ohlc":frame[["open","high","low","close"]].to_numpy().tolist()}


def examples(dataset, banks):
    result = []
    for j in np.unique(np.linspace(0, len(dataset) - 1, min(6,len(dataset)), dtype=int)):
        i,end = dataset.items[j]
        matches=[]
        for match in retrieve_past(banks[i], end):
            matches.append({**snapshot(dataset.series[i],match["end_row"]), **match})
        result.append({"query":snapshot(dataset.series[i],end), "past_matches":matches,
                       "available_past_blocks":len(history_indices(end,len(banks[i]),include_current=False))})
    return result


def write_review(path, cards):
    sections=[]
    for card in cards:
        panels=[]
        for index, item in enumerate([card["query"], *card["past_matches"]]):
            bars=np.asarray(item["ohlc"])
            low,high=bars[:,2].min(),bars[:,1].max()
            sy=lambda p:190 - (p-low)/max(high-low,1e-12)*170
            shapes=[]
            for j,(o,h,l,c) in enumerate(bars):
                x=20+j*760/len(bars); color="#b33e39" if c>=o else "#16806a"
                shapes.append(f'<line x1="{x}" x2="{x}" y1="{sy(h)}" y2="{sy(l)}" stroke="{color}"/>')
                shapes.append(f'<rect x="{x-1.5}" y="{min(sy(o),sy(c))}" width="3" height="{max(abs(sy(o)-sy(c)),1)}" fill="{color}"/>')
            title="当前已知窗口" if index==0 else f'此前 {item["age_bars"]} 根 · 相似度 {item["cosine"]:.3f}'
            panels.append(f'<h3>{title}</h3><p>{html.escape(item["start"])} → 可用时间 {html.escape(item["available_at"])}</p><svg viewBox="0 0 800 210">{"".join(shapes)}</svg>')
        sections.append(f'<section><h2>{html.escape(card["query"]["source"])}</h2><p>可检索历史片段：{card["available_past_blocks"]}</p>{"".join(panels)}</section>')
    Path(path).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>因果历史检索</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;background:#f4f5f3}section{background:white;padding:20px;margin:24px 0}svg{width:100%}</style><h1>从当前窗口回看过去</h1><p>全部是真实已知历史，不是预测或重建。仅检索同一合约、同一周期中与当前窗口不重叠的过去片段。每幅图独立缩放价格轴，仅比较形态；余弦相似度不是概率或经过验证的形态标签。样例按测试索引等距选取。</p>'+''.join(sections)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--source-run",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args()
    if not torch.cuda.is_available() or args.batch_size<1:
        raise ValueError("CUDA and positive batch size required for frozen extraction")
    source,out=Path(args.source_run),Path(args.out)
    tracked=("best.pt","manifest.json","history.jsonl","ae_metrics.json")
    hashes={n:fingerprint(source/n) for n in tracked}
    ck=torch.load(source/"best.pt",map_location="cpu",weights_only=True)
    if ck["schema"]!=SCHEMA or ck["config"]!=config(256) or ck.get("training_family")!="capacity":
        raise ValueError("Requires the frozen wide z256 capacity checkpoint")
    ref=ck["manifest"]; keys=[s["key"].split("/") for s in ref["sources"]]
    series,_=load_series(args.root,sorted({k[0] for k in keys}),sorted({int(k[1]) for k in keys}))
    bounds=ref["boundaries"]
    if manifest(series,bounds)!=ref:
        raise ValueError("Memory source/data mismatch")
    metadata={"schema":"babel-history-memory-v1","source_artifacts_sha256":hashes,
              "source_config":ck["config"],"selected_epoch":ck["epoch"],"manifest":ref,
              "window":WINDOW,"cache_stride":STRIDE,"memory_step":WINDOW,
              "context_blocks":[1,4,16],"retrieval":"all available earlier nonoverlapping blocks in same contract and period",
              "selection":"source checkpoint already selected by reconstruction validation; probes select alpha using validation only"}
    if out.exists() and any(out.iterdir()):
        if not (out/"manifest.json").exists() or json.loads((out/"manifest.json").read_text())!=metadata:
            raise ValueError("Existing memory run has incompatible metadata")
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(metadata,out/"manifest.json")
    cache=out/"cache";cache.mkdir(exist_ok=True)
    model=HistoryAE(**ck["config"]).to("cuda").eval(); model.load_state_dict(ck["model"])
    encoded=[encode_context(s.frame,s.period,"ema8_32") for s in series]
    banks=[]
    for i,(s,e) in enumerate(zip(series,encoded,strict=True)):
        path=cache/f"{i:04d}.npy"; checksum=path.with_suffix(".json")
        if path.exists() and checksum.exists() and json.loads(checksum.read_text())["sha256"]==fingerprint(path):
            bank=np.load(path,allow_pickle=False)
        else:
            bank=encode_bank(model,e,"cuda",args.batch_size)
            with path.with_suffix(".tmp").open("wb") as f: np.save(f,bank,allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
            atomic_json({"sha256":fingerprint(path)},checksum)
        expected=(len(range(WINDOW-1,len(e["x"]),STRIDE)),256)
        if bank.shape!=expected or not np.isfinite(bank).all(): raise ValueError("Invalid memory cache")
        atomic_json(bank_index(s),cache/f"{i:04d}_index.json")
        banks.append(bank)
        if i==0 or (i+1)%25==0 or i+1==len(series): progress(f"Frozen memory cache {i+1}/{len(series)}")
    datasets=[HistoryWindows(series,encoded,bounds,split) for split in ("train","val","test")]
    values=[state_features(ds,banks) for ds in datasets]
    report={"schema":"babel-history-memory-v1","source_artifacts_sha256":hashes,
            "split_windows":dict(zip(("train","val","test"),map(len,datasets))),
            "coverage":{},"state_probes":{}}
    for split,(_,_,counts) in zip(("train","val","test"),values,strict=True):
        report["coverage"][split]={"mean_available_blocks":float(counts.mean()),
                                   "at_least_4":int((counts>=4).sum()),"at_least_16":int((counts>=16).sum())}
    for scale,name in enumerate(("short","medium","long")):
        report["state_probes"][name]={}
        for feature in values[0][0]:
            progress(f"State probe {name}/{feature}; frozen encoder, ridge only")
            report["state_probes"][name][feature]=probe(*(v for f,y,_ in values for v in (f[feature],y[:,scale])),4)
    reference=json.loads((source/"ae_metrics.json").read_text())["current_structure_probe"]["pretrained"]
    observed=report["state_probes"]["medium"]["local256"]
    report["local_control"]={"source":reference,"recomputed":observed,
                              "ba_absolute_difference":abs(reference["test"]["ba"]-observed["test"]["ba"])}
    report["interpretation"]="Historical context baseline, not a trained global embedding. Means discard temporal order; concat has 768 dimensions. Rule probes are not trading performance. Earlier train-period memories are valid historical context. Test period has been inspected repeatedly; independent future holdout still required."
    if hashes!={n:fingerprint(source/n) for n in tracked}: raise ValueError("Source artifacts changed during memory evaluation")
    cards=examples(datasets[-1],banks)
    atomic_json(report,out/"memory_metrics.json")
    atomic_json(cards,out/"memory_examples.json")
    write_review(out/"memory_examples.html",cards)
    progress(f"Memory evaluation saved to {out}; neural weights unchanged")


if __name__=="__main__": main()
