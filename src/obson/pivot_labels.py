"""多尺度枢轴（zigzag）标签引擎 —— E12 标签地基

设计原则（与用户/教师对齐）：
- 纯几何定义：枢轴 = 价格反向运行超过 k×ATR 的极值点。不用任何技术指标参与定义，
  ATR 仅作波动率尺子（自适应阈值），不参与形态判定。
- 因果语义显式化：第 t 根的枢轴在确认时刻 c(t) > t 才成立。
  标签打在事件时刻 t（供训练），确认延迟 c(t)-t 一并记录（供下游自知"当时不可知"）。
- 每根 bar 自带多尺度结构身份：段方向 / 时间坐标 / 空间坐标 / 是否枢轴 / 确认延迟，
  切窗口不丢历史。

输出（每根 bar，每尺度 s）：
  seg_dir[s]        +1=上升段（自低点向上）/ -1=下降段 / 0=起始未定
  bars_since_piv[s] 距上一确认枢轴的 bar 数
  amp_since_piv[s]  距上一确认枢轴的幅度（÷ 当根 ATR，可为负? 否——取有符号：上升段为正）
  is_piv_high[s]    本 bar 是尺度 s 的枢轴高点（事件时刻标签）
  is_piv_low[s]     本 bar 是尺度 s 的枢轴低点
  piv_amp[s]        若本 bar 是枢轴：相邻两腿中较短腿的幅度（÷ATR），衡量枢轴"成色"
  confirm_delay[s]  若本 bar 是枢轴：确认延迟 bar 数；否则 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCALES = (0.75, 1.5, 3.0)  # k×ATR：短/中/长


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    hi, lo, cl = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_cl = np.roll(cl, 1)
    prev_cl[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    out = pd.Series(tr).ewm(span=n, adjust=False).mean().to_numpy()
    return np.maximum(out, 1e-9)


def zigzag(df: pd.DataFrame, k: float) -> list[dict]:
    """因果 zigzag：用 high/low 追踪候选极值，反向幅度 ≥ k×ATR 时确认上一个极值为枢轴。

    返回枢轴事件列表：[{bar, kind(+1高/-1低), price, confirm_bar, leg_amp}]，
    leg_amp = 该枢轴前一条腿的幅度（价格单位）。
    """
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    a = atr(df)
    pivots: list[dict] = []
    # 初始方向未定：先追踪同时的两个候选，谁先走出 k×ATR 就定初始方向
    cand_hi, cand_hi_i = hi[0], 0   # 候选高点
    cand_lo, cand_lo_i = lo[0], 0   # 候选低点
    direction = 0                    # +1=当前在上升段（找高点），-1=下降段（找低点）
    last_piv_price = None
    for i in range(1, len(df)):
        if direction >= 0:
            if hi[i] >= cand_hi:
                cand_hi, cand_hi_i = hi[i], i
            elif cand_hi - lo[i] >= k * a[i]:  # 从候选高点回落超阈值 → 确认高点
                leg = cand_hi - (last_piv_price if (last_piv_price is not None and direction > 0) else cand_lo)
                pivots.append(dict(bar=cand_hi_i, kind=1, price=cand_hi,
                                   confirm_bar=i, leg_amp=float(leg)))
                last_piv_price = cand_hi
                direction = -1
                cand_lo, cand_lo_i = lo[i], i
        if direction <= 0:
            if lo[i] <= cand_lo:
                cand_lo, cand_lo_i = lo[i], i
            elif hi[i] - cand_lo >= k * a[i]:  # 从候选低点反弹超阈值 → 确认低点
                leg = (last_piv_price if (last_piv_price is not None and direction < 0) else cand_hi) - cand_lo
                pivots.append(dict(bar=cand_lo_i, kind=-1, price=cand_lo,
                                   confirm_bar=i, leg_amp=float(leg)))
                last_piv_price = cand_lo
                direction = +1
                cand_hi, cand_hi_i = hi[i], i
    return pivots


def bar_labels(df: pd.DataFrame, scales: tuple[float, ...] = SCALES) -> dict[str, np.ndarray]:
    """对每根 bar × 每尺度生成结构身份标签。返回 {name: [N, S] ndarray}。"""
    n = len(df)
    a = atr(df)
    out = {
        "seg_dir": np.zeros((n, len(scales)), np.int8),
        "bars_since_piv": np.zeros((n, len(scales)), np.float32),
        "amp_since_piv": np.zeros((n, len(scales)), np.float32),
        "is_piv_high": np.zeros((n, len(scales)), bool),
        "is_piv_low": np.zeros((n, len(scales)), bool),
        "piv_amp": np.zeros((n, len(scales)), np.float32),
        "confirm_delay": np.zeros((n, len(scales)), np.float32),
    }
    cl = df["close"].to_numpy()
    for si, k in enumerate(scales):
        pivs = zigzag(df, k)
        # 事件标签打在枢轴 bar 上
        for p in pivs:
            b = p["bar"]
            if p["kind"] > 0:
                out["is_piv_high"][b, si] = True
            else:
                out["is_piv_low"][b, si] = True
            out["piv_amp"][b, si] = p["leg_amp"] / a[b]
            out["confirm_delay"][b, si] = p["confirm_bar"] - b
        # 段身份：按（事件时刻）枢轴切段——注意这是"事后全知"口径的训练标签，
        # 下游实盘口径需用确认时刻重建，差异由 confirm_delay 显式携带
        for pi, p in enumerate(pivs):
            start = p["bar"] + 1
            end = pivs[pi + 1]["bar"] + 1 if pi + 1 < len(pivs) else n
            d = -p["kind"]  # 高点之后是下降段，低点之后是上升段
            out["seg_dir"][start:end, si] = d
            idx = np.arange(start, end)
            out["bars_since_piv"][start:end, si] = idx - p["bar"]
            out["amp_since_piv"][start:end, si] = d * (cl[start:end] - p["price"]) / a[start:end]
    return out


def label_frame(df: pd.DataFrame, scales: tuple[float, ...] = SCALES) -> pd.DataFrame:
    """标签展开成 DataFrame，列名带尺度后缀。"""
    lab = bar_labels(df, scales)
    cols = {}
    for name, mat in lab.items():
        for si, k in enumerate(scales):
            cols[f"{name}_s{si}"] = mat[:, si]
    out = pd.DataFrame(cols)
    out.insert(0, "datetime", df["datetime"].to_numpy())
    return out
