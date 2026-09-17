"""E12-M4 第二步：形态检索查询 + 门禁 R1-R5

用法：
  # 实时查询（时间掩码自动屏蔽查询时点之后的库条目）
  PYTHONPATH=src python -u scripts/e12_query.py \
      --ckpt checkpoints/e12r1_causal_s42/best.pt --index data/index/e12_index.npz \
      --code rb --period 60 [--datetime "2026-09-15 14:00"] [--plot reports/query.png]

  # 门禁自检（R1 自一致性 / R2 结构一致性 / R3 打乱对照 / R5 分布检验）
  PYTHONPATH=src python -u scripts/e12_query.py \
      --ckpt ... --index ... --self-test 200
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
DIR_NAME = {0: "下降", 1: "无段", 2: "上升"}


def load_index(path):
    z = np.load(path, allow_pickle=False)
    emb = z["emb"].astype(np.float32)
    # v1.1 均值中心化：全部向量共享一个强公共方向（实测 Top-20 余弦挤在
    # 0.994+），减掉库均值再归一化，相似度才反映结构差异而非"市场一般状态"
    mean = emb.mean(0)
    emb = emb - mean
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb, mean, {k: z[k] for k in ("code", "period", "row", "dt_ns", "seg_dir", "fwd", "horizons")}


@torch.no_grad()
def embed_query(model, code, period, window, dt_ns, device):
    """查询 embedding：取 code_period 合约帧中 dt<=dt_ns 的最后一根为窗口末根。"""
    from train_multi_symbol import SYMBOLS
    tr, va, te = build_pattern_datasets(
        code, period, window, 1, SYMBOLS.get(code, 0),
        FREQ_IDS.get(period, 7), contract=True)
    ds = tr
    ds.indices = np.sort(np.concatenate([d.indices for d in (tr, va, te)]))
    if dt_ns is None:
        pos = len(ds.indices) - 1
    else:
        ok = np.where(ds.dt_ns[ds.indices] <= dt_ns)[0]
        if len(ok) == 0:
            raise ValueError("该时点之前没有可用窗口")
        pos = int(ok[-1])
    item = ds[pos]
    x = item["x"].unsqueeze(0).to(device)
    h = model(x, torch.tensor([ds.symbol_id], device=device),
              torch.tensor([ds.freq_id], device=device))["h"][0, -1].float().cpu().numpy()
    j = int(ds.indices[pos])
    return h, ds, j


def report(hits, meta, k, horizons):
    print(f"\n── Top-{k} 命中 ──")
    for r, (i, sim) in enumerate(hits):
        ts = np.datetime64(int(meta["dt_ns"][i]), "ns")
        d = " ".join(f"s{s}:{DIR_NAME[meta['seg_dir'][i, s]]}" for s in range(3))
        fwd = meta["fwd"][i]
        fs = " ".join(f"+{h}bar:{v:+.2f}ATR" if np.isfinite(v) else f"+{h}bar:—"
                      for h, v in zip(horizons, fwd))
        print(f"  #{r + 1:2d} sim={sim:.3f} {meta['code'][i]}_{meta['period'][i]}m "
              f"{str(ts)[:16]} | {d} | {fs}")


def fwd_stats(idx, meta, label):
    fwd = meta["fwd"][idx]
    print(f"  {label}: n={len(idx)}")
    for c, h in enumerate(meta["horizons"]):
        v = fwd[:, c]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        print(f"    +{h}bar: 均值{v.mean():+.3f} ATR | 中位{np.median(v):+.3f} | "
              f"上行率{(v > 0).mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--code", default="rb")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--datetime", default=None, help="查询时点；缺省=数据末根")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--self-test", type=int, default=0, help="R1-R3 自检：随机抽样条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", default=None, help="输出查询vs命中对比图路径")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = PatternEncoder(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    emb, lib_mean, meta = load_index(args.index)
    horizons = meta["horizons"]
    print(f"[load] 索引 {len(emb)} 条 | ckpt → {device}")

    rng = np.random.default_rng(args.seed)

    # ═══ R1/R2/R3 自检模式 ═══
    if args.self_test > 0:
        n = min(args.self_test, len(emb))
        pick = rng.choice(len(emb), n, replace=False)
        r1_hits = 0
        consist = np.zeros(3)        # 三尺度分别统计
        consist_shuf = 0.0
        for i in pick:
            sim = emb @ emb[i]
            order = np.argsort(-sim)
            if order[0] == i:
                r1_hits += 1
            # v1.1 与查询模式同规则：屏蔽自身 + 同组合重叠窗口（|Δrow|<window）
            invalid = ((meta["code"] == meta["code"][i])
                       & (meta["period"] == meta["period"][i])
                       & (np.abs(meta["row"] - meta["row"][i]) < args.window))
            sim[invalid] = -np.inf
            top = np.argsort(-sim)[:args.topk]
            for s in range(3):
                consist[s] += (meta["seg_dir"][top, s] == meta["seg_dir"][i, s]).mean()
            sh = emb.copy()
            for c in range(sh.shape[1]):                     # R3 打乱对照
                sh[:, c] = sh[rng.permutation(len(sh)), c]
            order_s = np.argsort(-(sh @ sh[i]))
            top_s = order_s[order_s != i][:args.topk]
            consist_shuf += (meta["seg_dir"][top_s, 0] == meta["seg_dir"][i, 0]).mean()
        consist /= n
        r1, r2s = r1_hits / n, consist_shuf / n
        r2 = float(consist[0])
        r2m = float(consist.mean())
        print(f"\n══ 检索门禁自检（n={n}，均值中心化后）══")
        print(f"  R1 自一致性 Top-1 = {r1:.3f}（线 1.00）{'✅' if r1 == 1.0 else '❌'}")
        print(f"  R2 段方向一致率 s0={consist[0]:.3f} s1={consist[1]:.3f} "
              f"s2={consist[2]:.3f} 均={r2m:.3f}（线 s0≥0.70）"
              f"{'✅' if r2 >= 0.70 else '❌'}")
        print(f"  R3 打乱对照一致率 = {r2s:.3f}（Δ={r2 - r2s:+.3f}，线 ≥0.15）"
              f"{'✅' if r2 - r2s >= 0.15 else '❌'}")
        return

    # ═══ 查询模式 ═══
    dt_ns = np.datetime64(args.datetime, "ns").astype(np.int64) if args.datetime else None
    q, ds, j = embed_query(model, args.code, args.period, args.window, dt_ns, device)
    q = q - lib_mean                                     # 与库同口径中心化
    q /= np.linalg.norm(q) + 1e-9
    q_dt = int(ds.dt_ns[j])
    print(f"[query] {args.code}_{args.period}m "
          f"{str(np.datetime64(q_dt, 'ns'))[:16]} | 段标签 "
          + " ".join(f"s{s}:{DIR_NAME[ds.seg_dir[j, s]]}" for s in range(3)))

    mask = meta["dt_ns"] < q_dt          # 防泄漏：只看查询时点之前
    # v1.1 重叠屏蔽：同品种同周期且行距 < window 的条目与查询窗口
    # 共享超过一半 bar，本质是"查到自己"，屏蔽
    overlap = (meta["code"] == args.code) & (meta["period"] == args.period) & \
              (np.abs(meta["row"] - j) < args.window)
    mask &= ~overlap
    sims = emb[mask] @ q
    lib_idx = np.where(mask)[0]
    order = np.argsort(-sims)
    hits = [(int(lib_idx[o]), float(sims[o])) for o in order[:args.topk]]
    report(hits, meta, args.topk, horizons)

    # R5：命中集后续分布 vs 先验（只报告，不作判决）
    print("\n══ R5 后续走势分布（只报告）══")
    fwd_stats(np.array([i for i, _ in hits]), meta, f"命中集 Top-{args.topk}")
    fwd_stats(lib_idx, meta, "全库先验（查询时点前）")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        w = ds.window
        qc = ds.px[j - w + 1:j + 1, 3]
        ax.plot((qc - qc[-1]) / max(ds.atr[j], 1e-6), lw=2, label="query")
        for i, _ in hits[:5]:
            key = (meta["code"][i], int(meta["period"][i]))
            try:
                htr, hva, hte = build_pattern_datasets(
                    key[0], key[1], w, 1, 0, FREQ_IDS.get(key[1], 7), contract=True)
                hd = htr
                hj = int(meta["row"][i])
                hc = hd.px[hj - w + 1:hj + 1, 3]
                if len(hc) == w:
                    ax.plot((hc - hc[-1]) / max(hd.atr[hj], 1e-6), alpha=0.5,
                            label=f"{key[0]}_{key[1]}m")
            except Exception:
                pass
        ax.legend(fontsize=8)
        ax.set_title("query vs top hits (close, ATR-norm, end-aligned)")
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, bbox_inches="tight")
        print(f"[plot] → {args.plot}")


if __name__ == "__main__":
    main()
