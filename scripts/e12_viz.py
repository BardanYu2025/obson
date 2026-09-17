"""E12 embedding 可视化：h_t 的 t-SNE 簇结构（"看懂"的直观证据）

三张着色图（同一 embedding 投影）：
  左：段方向（红=上升段 / 绿=下降段 / 灰=未定）
  中：距枢轴的远近（颜色越亮越靠近枢轴，看枢轴是否形成环/带）
  右：品种（看簇是按品种分还是按结构分——按结构分=学会通用语言）

用法：
  PYTHONPATH=src python -u scripts/e12_viz.py \
      --ckpt checkpoints/e12r1_causal_s42/best.pt --symbols rb sr p --periods 60 30
输出：reports/e12_viz_tsne.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from obson.pattern_data import N_SCALES, build_pattern_datasets
from obson.pattern_model import PatternEncoder

FREQ_IDS = {5: 1, 15: 2, 30: 3, 60: 4}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--symbols", nargs="+", default=["rb", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=6000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = PatternEncoder(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    from obson.symbols import SYMBOLS
    H_all, seg_all, amp_all, sym_all = [], [], [], []
    for code in args.symbols:
        for period in args.periods:
            if not Path(f"data/labels/{code}_{period}m_contract_labels.pkl").exists():
                continue
            _, va, _ = build_pattern_datasets(
                code, period, args.window, args.stride,
                symbol_id=SYMBOLS.get(code, 0), freq_id=FREQ_IDS.get(period, 7),
                contract=True)
            dl = DataLoader(va, batch_size=64)
            for bi, batch in enumerate(dl):
                if bi >= 6:  # 每组合约 384 窗 × 256 bar ≈ 10 万点太多，截断
                    break
                out = model(batch["x"].to(device), batch["symbol_id"].to(device),
                            batch["freq_id"].to(device))
                h = out["h"].cpu()
                B, T, D = h.shape
                # 每窗只取后 64 根 bar（因果版后段上下文更足，且降采样）
                take = slice(-64, None)
                H_all.append(h[:, take].reshape(-1, D))
                seg_all.append(batch["seg_dir"][:, take, 1].reshape(-1))       # 中尺度
                amp_all.append(batch["amp_since"][:, take, 1].reshape(-1).abs())
                sym_all.extend([f"{code}_{period}m"] * (B * 64))
    H = torch.cat(H_all).numpy()
    seg = torch.cat(seg_all).numpy()
    amp = torch.cat(amp_all).numpy()
    sym = np.array(sym_all)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(H), min(args.max_samples, len(H)), replace=False)
    H, seg, amp, sym = H[sel], seg[sel], amp[sel], sym[sel]
    print(f"t-SNE 输入 {len(H)} 点...")

    try:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(H)
        method = "t-SNE"
    except ImportError:
        # 无 sklearn：PCA 投影兜底（numpy SVD，零依赖）
        Hc = H - H.mean(0)
        _, _, Vt = np.linalg.svd(Hc[: min(len(Hc), 20000)], full_matrices=False)
        Z = Hc @ Vt[:2].T
        method = "PCA"
    print(f"投影方式: {method}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ("PingFang SC", "Heiti SC", "Arial Unicode MS"):
        from matplotlib import font_manager
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    colors = {0: "gray", 1: "green", 2: "red"}
    names = {0: "未定", 1: "下降段", 2: "上升段"}
    for c in (0, 1, 2):
        m = seg == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.4, c=colors[c], label=names[c])
    axes[0].legend(markerscale=5); axes[0].set_title("段方向着色（中尺度）")

    sc = axes[1].scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.5,
                         c=np.clip(amp, 0, 3), cmap="viridis")
    fig.colorbar(sc, ax=axes[1], label="|距枢轴幅度| (ATR)")
    axes[1].set_title("距枢轴幅度着色（亮点=远离枢轴）")

    uniq = np.unique(sym)
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(uniq):
        m = sym == s
        axes[2].scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.4, color=cmap(i % 20), label=s)
    axes[2].legend(markerscale=5, fontsize=7); axes[2].set_title("品种着色")

    fig.suptitle(f"E12 bar embedding {method} — {Path(args.ckpt).parent.name}")
    fig.tight_layout()
    out = Path("reports/e12_viz_tsne.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
