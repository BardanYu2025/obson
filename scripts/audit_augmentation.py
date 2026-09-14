"""E7-A 第一步：增强语义合法性审计（协议 v3 §3.1）

纯统计脚本，不训练任何模型。对真实 K 线窗口施加协议 §3.2 的窗口级收益率抖动：

    r'_t = r_t + ε_t,  ε_t ~ N(0, α · σ_window / sqrt(L))
    （α = 窗口累计噪声标准差占窗口自身波动的比例）

审计代理（v3 提前固定，不看下游收益）：
  1. 窗口末端相对收益符号翻转率
  2. 线性回归斜率符号翻转率
  3. 最大正/负 excursion 主导方向翻转率
  4. 波动率 regime 分桶漂移率（三分桶，参照该品种滚动历史）
  5. 窗口累计收益 / 末端价格相对变化幅度
  6. 末 3 根 close-path 锚定相对变化
  7. 时间特征与 causal mask 不受影响（抖动只改收益率，设计保证，打印确认）

用法：
  PYTHONPATH=src .venv/bin/python scripts/audit_augmentation.py
  PYTHONPATH=src .venv/bin/python scripts/audit_augmentation.py --symbols rb sr p --periods 60 --n 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = [0.02, 0.05, 0.10]
RNG_SEED = 20260914


def load_windows(csv_path: Path, seq_len: int, n: int, rng: np.random.Generator):
    """从 CSV 随机抽 n 个长度为 seq_len 的连续窗口，返回 closes 数组 [n, seq_len+1]。
    跳过含跳空断点（|ret|>2%，与训练管线断链保护一致）的窗口。"""
    df = pd.read_csv(csv_path)
    close = df["close"].to_numpy(dtype=np.float64)
    if len(close) < seq_len + 2:
        return None
    starts = np.arange(0, len(close) - seq_len - 1)
    rng.shuffle(starts)
    out = []
    for s in starts:
        w = close[s:s + seq_len + 1]  # 多取 1 根用于算 seq_len 个 bar 收益
        r = np.diff(np.log(np.maximum(w, 1e-8)))
        if np.any(np.abs(r) > 0.02):  # 断链保护同训练管线
            continue
        out.append(w)
        if len(out) >= n:
            break
    return np.array(out) if out else None


def proxies(closes: np.ndarray):
    """对每个窗口计算审计代理。closes: [n, L+1]，返回 dict of [n] 数组。"""
    logp = np.log(np.maximum(closes, 1e-8))
    rets = np.diff(logp, axis=1)                       # [n, L]
    cum_ret = closes[:, -1] / closes[:, 0] - 1.0       # 末端相对收益
    # 线性回归斜率（log 价格对时间）
    t = np.arange(logp.shape[1])
    t = t - t.mean()
    slope = (logp * t).sum(axis=1) / (t * t).sum()
    # excursion：窗口内相对起点的最大上/下偏移
    rel = closes / closes[:, [0]] - 1.0
    max_up = rel.max(axis=1)
    max_dn = -rel.min(axis=1)
    exc_dir = np.sign(max_up - max_dn)                 # 主导方向
    sigma_w = rets.std(axis=1)                         # 窗口波动
    anchor = closes[:, -3:].mean(axis=1)               # close-path 锚定（末3根均价）
    return dict(cum_ret=cum_ret, slope=slope, exc_dir=exc_dir,
                sigma_w=sigma_w, anchor=anchor, rets=rets)


def jitter(closes: np.ndarray, alpha: float, rng: np.random.Generator):
    """协议 §3.2 窗口级抖动：对每根 bar 的 log 收益加 ε，重建价格路径。"""
    p = proxies(closes)
    rets = p["rets"]
    L = rets.shape[1]
    eps = rng.normal(0.0, 1.0, size=rets.shape)
    eps *= (alpha * p["sigma_w"] / np.sqrt(L))[:, None]
    new_rets = rets + eps
    new_logp = np.concatenate([np.log(np.maximum(closes[:, [0]], 1e-8)),
                               np.log(np.maximum(closes[:, [0]], 1e-8)) + np.cumsum(new_rets, axis=1)],
                              axis=1)
    return np.exp(new_logp)


def audit_symbol_period(csv_path: Path, seq_len: int, n: int, rng):
    closes = load_windows(csv_path, seq_len, n, rng)
    if closes is None:
        return None
    base = proxies(closes)
    # 波动率 regime 参照：该文件全历史的窗口 σ 三分位
    df = pd.read_csv(csv_path)
    cl = df["close"].to_numpy(dtype=np.float64)
    r_all = np.diff(np.log(np.maximum(cl, 1e-8)))
    sig_all = pd.Series(r_all).rolling(seq_len).std().dropna().to_numpy()
    q1, q2 = np.nanquantile(sig_all, [1 / 3, 2 / 3])

    def bucket(s):
        return np.digitize(s, [q1, q2])

    rows = []
    for alpha in ALPHAS:
        aug = jitter(closes, alpha, rng)
        pa = proxies(aug)
        flip_end = np.mean(np.sign(base["cum_ret"]) != np.sign(pa["cum_ret"]))
        flip_slope = np.mean(np.sign(base["slope"]) != np.sign(pa["slope"]))
        both_nz = (base["exc_dir"] != 0) & (pa["exc_dir"] != 0)
        flip_exc = np.mean(base["exc_dir"][both_nz] != pa["exc_dir"][both_nz]) if both_nz.any() else np.nan
        drift_vol = np.mean(bucket(base["sigma_w"]) != bucket(pa["sigma_w"]))
        d_cum = np.mean(np.abs(pa["cum_ret"] - base["cum_ret"])) * 100
        d_end = np.mean(np.abs(aug[:, -1] / closes[:, -1] - 1.0)) * 100
        d_anchor = np.mean(np.abs(pa["anchor"] / base["anchor"] - 1.0)) * 100
        rows.append(dict(alpha=alpha, n=len(closes),
                         flip_end=flip_end, flip_slope=flip_slope,
                         flip_exc=flip_exc, drift_vol=drift_vol,
                         d_cum_pct=d_cum, d_end_pct=d_end, d_anchor_pct=d_anchor))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", nargs="+", default=["rb", "sr", "p", "m", "y"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--seq-len", type=int, default=None, help="默认 60m→60 根 / 30m→60 根")
    ap.add_argument("--n", type=int, default=300, help="每个 品种×周期 抽多少窗口")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    print("== E7-A 增强语义合法性审计（协议 v3 §3.1）==")
    print(f"样本: 每组合 {args.n} 窗口 | 抖动: ε~N(0, α·σ_w/√L) | seed={RNG_SEED}")
    print("设计保证: 抖动只改收益率序列，不动时间特征/causal mask/volume/OI 通道\n")

    all_rows = []
    for sym in args.symbols:
        for per in args.periods:
            csv_path = Path(args.data_dir) / f"{sym}_{per}m.csv"
            if not csv_path.exists():
                print(f"  !! 缺文件 {csv_path}，跳过")
                continue
            seq_len = args.seq_len or 60
            rows = audit_symbol_period(csv_path, seq_len, args.n, rng)
            if not rows:
                print(f"  !! {sym}_{per}m 有效窗口不足，跳过")
                continue
            for r in rows:
                r.update(symbol=sym, period=per)
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    # 汇总表：按 α 平均各翻转率/变化幅度
    agg = df.groupby("alpha").agg(
        组合数=("n", "count"),
        样本数=("n", "sum"),
        末端符号翻转=("flip_end", "mean"),
        斜率符号翻转=("flip_slope", "mean"),
        excursion方向翻转=("flip_exc", "mean"),
        波动分桶漂移=("drift_vol", "mean"),
        累计收益变化均幅_pct=("d_cum_pct", "mean"),
        末端价格变化均幅_pct=("d_end_pct", "mean"),
        锚定变化均幅_pct=("d_anchor_pct", "mean"),
    )
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n== 分品种明细（α=0.05）==")
    print(df[df.alpha == 0.05].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n== 判读参考（协议：判决只看语义破坏率）==")
    for _, row in agg.iterrows():
        verdict = "✅ 轻微" if row["斜率符号翻转"] < 0.05 else \
                  "🔶 中等" if row["斜率符号翻转"] < 0.15 else "❌ 明显破坏"
        print(f"  α={row.name:.2f}: 斜率翻转 {row['斜率符号翻转']:.1%} | "
              f"excursion翻转 {row['excursion方向翻转']:.1%} | "
              f"波动漂移 {row['波动分桶漂移']:.1%} → {verdict}")
    out = Path("reports/aug_audit.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n明细已存 {out}")


if __name__ == "__main__":
    main()
