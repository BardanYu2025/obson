"""枢轴标签可视化审计：随机抽段画图，人工/模型自检标签质量

用法:
  python scripts/audit_pivots.py [--symbol rb] [--period 60] [--n 6]
输出: reports/pivot_audit/{symbol}_{period}m_seg{i}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from obson.pivot_labels import SCALES, zigzag, atr


def render(df: pd.DataFrame, i0: int, i1: int, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # CJK 字体兜底
    for f in ("PingFang SC", "Heiti SC", "Arial Unicode MS"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    seg = df.iloc[i0:i1]
    x = np.arange(len(seg))
    fig, axes = plt.subplots(len(SCALES), 1, figsize=(14, 3.2 * len(SCALES)), sharex=True)
    colors = {1: "red", -1: "green"}
    for si, k in enumerate(SCALES):
        ax = axes[si]
        ax.plot(x, seg["close"].to_numpy(), lw=0.9, color="black", alpha=0.8)
        pivs = [p for p in zigzag(df, k) if i0 - 200 <= p["bar"] < i1]  # 前文枢轴也画，看连续性
        for p in pivs:
            xp = p["bar"] - i0
            if 0 <= xp < len(seg):
                ax.scatter([xp], [p["price"]], marker="v" if p["kind"] > 0 else "^",
                           color=colors[p["kind"]], s=60, zorder=5)
                ax.annotate(f"d={p['confirm_bar']-p['bar']}", (xp, p["price"]),
                            fontsize=7, alpha=0.7)
            else:
                ax.axvline(0, color="blue", alpha=0.3)  # 段首之前有枢轴的提示
        # 连成 zigzag 线
        vis = [p for p in pivs if 0 <= p["bar"] - i0 < len(seg)]
        if len(vis) >= 2:
            ax.plot([p["bar"] - i0 for p in vis], [p["price"] for p in vis],
                    color="blue", lw=1.2, alpha=0.5)
        ax.set_ylabel(f"k={k}×ATR")
        ax.grid(alpha=0.2)
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="rb")
    ap.add_argument("--period", type=int, default=60)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(f"data/{args.symbol}_{args.period}m.csv", parse_dates=["datetime"])
    outdir = Path("reports/pivot_audit")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    win = 500
    for s in range(args.n):
        i0 = int(rng.integers(0, len(df) - win))
        path = outdir / f"{args.symbol}_{args.period}m_seg{s}.png"
        t0, t1 = df['datetime'].iloc[i0], df['datetime'].iloc[i0 + win - 1]
        render(df, i0, i0 + win, path, f"{args.symbol}_{args.period}m  {t0} → {t1}")
        print(f"saved {path}")
    # 全样本枢轴统计
    a = atr(df)
    for k in SCALES:
        pivs = zigzag(df, k)
        delays = np.array([p["confirm_bar"] - p["bar"] for p in pivs])
        amps = np.array([p["leg_amp"] for p in pivs]) / a[[p["bar"] for p in pivs]]
        print(f"k={k}: {len(pivs)} 枢轴 | 确认延迟 中位{np.median(delays):.0f} p90={np.percentile(delays, 90):.0f} bar"
              f" | 腿幅度 中位{np.median(amps):.2f}×ATR | 高低比 "
              f"{sum(p['kind']>0 for p in pivs)}/{sum(p['kind']<0 for p in pivs)}")


if __name__ == "__main__":
    main()
