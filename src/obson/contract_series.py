# -*- coding: utf-8 -*-
"""按合约分段的训练数据帧构建（contract-mode，做法 B）

核心语义：每个样本的特征窗口、技术指标、标签路径全部来自
"当时主力合约自己的序列"（含其当远月时的历史），零跨合约污染：
  - 拼接：各合约完整生命周期按主力任期顺序纵向堆叠为单一 df
  - seg 边界 = 硬断点（特征窗/标签路径不许跨越）
  - main_mask：该 bar 所在交易日是否属于本段合约的主力任期（只有 True 的 bar 能成为样本锚点）
  - day_ids：段内独立编号（拼接帧时间戳非单调，不能全局编号，否则
    同名交易日的两个合约副本会被并入同一收盘锚组）

配合 dataset.KLineDataset 的 extra_breaks / sample_mask / day_ids_override 参数使用。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from obson.model.dataset import trading_day_ids

CONTRACT_DIR = Path(__file__).parent.parent.parent / "data" / "contracts"


def load_calendar(code: str, data_dir: str | Path = CONTRACT_DIR) -> pd.DataFrame:
    fp = Path(data_dir) / f"{code}_roll_calendar.csv"
    if not fp.exists():
        raise FileNotFoundError(f"缺主力日历 {fp}，先跑 scripts/download_contracts.py")
    cal = pd.read_csv(fp, parse_dates=["date"])
    return cal


def trading_day_dates(timestamps: pd.Series) -> np.ndarray:
    """与 dataset.trading_day_ids 同一口径，但返回每根 bar 的交易日历日（date）。
    夜盘（18:00 后）归下一个工作日——周五夜盘归周一，不能用 +6h 自然日近似
    （周五 21:00 +6h 落周六，不在任何交易日历里，会被误判为非主力）。"""
    d = pd.to_datetime(timestamps)
    bset = pd.bdate_range(
        d.min().normalize(), d.max().normalize() + pd.Timedelta(days=10)
    ).values
    idx = np.searchsorted(bset, d.dt.normalize().values) + (
        d.dt.hour >= 18
    ).to_numpy().astype(int)
    return pd.DatetimeIndex(bset[idx]).date


def build_contract_frame(code: str, period: int, data_dir: str | Path = CONTRACT_DIR):
    """拼接合约段帧。

    :return: (df, extra_breaks, main_mask, day_ids)
        df: 拼接帧（8 列，与主连 CSV 同格式 + seg 列=合约序号）
        extra_breaks: np.ndarray，各段起始 bar 在 df 中的位置（硬断点）
        main_mask: np.ndarray[bool]，bar 级：所在交易日 ∈ 本段合约主力任期
        day_ids: np.ndarray[int]，段内交易日编号（全局唯一，跨段递增）
    """
    cal = load_calendar(code, data_dir)
    tenure: dict[str, set] = {}
    for contract, g in cal.groupby("contract"):
        tenure[contract] = set(g["date"].dt.date)

    frames, breaks, masks, day_ids_all = [], [], [], []
    offset_day = 0
    pos = 0
    kept = []
    for seg_id, (contract, days) in enumerate(sorted(
            tenure.items(), key=lambda kv: min(kv[1]))):  # 按任期首日排序
        fp = Path(data_dir) / code / f"{contract}_{period}m.csv"
        if not fp.exists():
            print(f"  [contract] 警告: 缺 {fp.name}，该合约段跳过")
            continue
        df = pd.read_csv(fp, parse_dates=["datetime"])
        df = df[df["datetime"] > "2001-01-01"].reset_index(drop=True)
        if len(df) < 50:
            continue
        # 段内交易日编号（夜盘归次日口径与全局一致）
        seg_days = trading_day_ids(df["datetime"]).astype(np.int64)
        # bar 的交易日历日（夜盘归次一工作日口径，与 trading_day_ids 一致）
        bar_day = trading_day_dates(df["datetime"])
        mask = np.array([d in days for d in bar_day], dtype=bool)

        df["seg"] = seg_id
        frames.append(df)
        breaks.append(pos)
        masks.append(mask)
        day_ids_all.append(seg_days + offset_day)
        offset_day += int(seg_days.max()) + 1
        pos += len(df)
        kept.append(contract)

    if not frames:
        raise ValueError(f"[{code}] 无可用合约段，检查 {Path(data_dir) / code}")
    df_all = pd.concat(frames, ignore_index=True)
    extra_breaks = np.array(breaks[1:], dtype=np.int64)  # 首段起点不算断点
    main_mask = np.concatenate(masks)
    day_ids = np.concatenate(day_ids_all)
    n_main = int(main_mask.sum())
    print(f"  [contract] {code}_{period}m: {len(kept)} 段拼接 {len(df_all)} 行，"
          f"主力任期 bar {n_main} 根（{n_main / len(df_all):.0%}），断点 {len(extra_breaks)} 处")
    return df_all, extra_breaks, main_mask, day_ids
