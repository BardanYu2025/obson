"""E12-M4 第一步：把全量历史压成 embedding 检索库

对每个品种×周期，用 E12 因果骨干滑窗推理，取窗口末根 bar 的 embedding
落盘。同时落盘元数据：品种/周期/行号/时间戳/三尺度段方向标签/
后续 5/10/20 根 bar 的实际涨跌（ATR 归一化，越界为 NaN）。

用法：
  PYTHONPATH=src python -u scripts/e12_index.py \
      --ckpt checkpoints/e12r1_causal_s42/best.pt \
      --symbols rb hc i sr p j jm m y cu ag TA MA --periods 60 30 \
      --stride 5 --out data/index/e12_index.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from obson.pattern_data import build_pattern_datasets
from obson.pattern_model import PatternEncoder

FREQ_IDS = {5: 1, 15: 2, 30: 3, 60: 4}
FWD_HORIZONS = (5, 10, 20)     # 后续走势观测窗（bar 数）


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=5,
                    help="索引步长：1=每根 bar 都入库（最大最准），5=省 5 倍空间")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--contract", action="store_true", default=True)
    ap.add_argument("--out", default="data/index/e12_index.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = PatternEncoder(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[load] {args.ckpt} → {device}")

    from obson.symbols import SYMBOLS

    embs, metas = [], {k: [] for k in
                       ("code", "period", "row", "dt_ns", "seg_dir", "fwd")}
    for code in args.symbols:
        for period in args.periods:
            try:
                tr, va, te = build_pattern_datasets(
                    code, period, args.window, args.stride,
                    SYMBOLS.get(code, 0), FREQ_IDS.get(period, 7),
                    contract=args.contract)
            except FileNotFoundError as e:
                print(f"  [skip] {code}_{period}m: {e}")
                continue
            ds = tr  # 三个 split 共享底层数组；直接把末根集合换成全量
            ds.indices = np.sort(np.concatenate([d.indices for d in (tr, va, te)]))
            n0 = sum(len(e) for e in embs)
            for lo in range(0, len(ds.indices), args.batch_size):
                x = torch.stack([ds[k]["x"] for k in range(lo, min(lo + args.batch_size, len(ds.indices)))])
                rows = ds.indices[lo:lo + args.batch_size]
                x = x.to(device)
                sid = torch.tensor([ds.symbol_id] * len(rows), device=device)
                fid = torch.tensor([ds.freq_id] * len(rows), device=device)
                out = model(x, sid, fid)
                embs.append(out["h"][:, -1].float().cpu().numpy().astype(np.float16))
                for j in rows:
                    metas["code"].append(code)
                    metas["period"].append(period)
                    metas["row"].append(int(j))
                    metas["dt_ns"].append(int(ds.dt_ns[j]))
                    metas["seg_dir"].append(ds.seg_dir[j].copy())  # [3] 0/1/2
                    fwd = []
                    for h in FWD_HORIZONS:
                        if j + h < len(ds.px):
                            fwd.append((ds.px[j + h, 3] - ds.px[j, 3])
                                       / max(float(ds.atr[j]), 1e-6))
                        else:
                            fwd.append(np.nan)
                    metas["fwd"].append(fwd)
            n1 = sum(len(e) for e in embs)
            print(f"  [{code}_{period}m] +{n1 - n0} 条（累计 {n1}）")

    emb = np.concatenate(embs, 0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        emb=emb,
        code=np.array(metas["code"]),
        period=np.array(metas["period"], np.int32),
        row=np.array(metas["row"], np.int64),
        dt_ns=np.array(metas["dt_ns"], np.int64),
        seg_dir=np.stack(metas["seg_dir"]).astype(np.int8),
        fwd=np.array(metas["fwd"], np.float32),
        horizons=np.array(FWD_HORIZONS, np.int32),
    )
    print(f"[done] {len(emb)} 条 → {args.out} "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
