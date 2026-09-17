"""E12 per-bar 形态数据集：滑窗 + 18 维结构标签 + 掩码重构素材

设计纪律（E12 设计稿）：
- 输入只有原始 OHLCV(+OI) 与日内时刻——任何技术指标不进输入；
- 价格按窗口末根 close 锚定、窗口末根 ATR14 缩放（站内可比）；
- 标签与行情按行对齐（data/labels/*.pkl，由 build_bar_labels.py 生成）；
- 切分按交易日 70/15/15，与历史项目口径一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from obson.pivot_labels import atr

FEATURE_DIM = 8  # o,h,l,c(归一) + vol,oi(log-z) + tod_sin,tod_cos
N_SCALES = 3


def _tod_feats(dt: pd.Series) -> np.ndarray:
    mins = (dt.dt.hour * 60 + dt.dt.minute).to_numpy(np.float32)
    ang = 2 * np.pi * mins / (24 * 60)
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)


class PatternWindowDataset(Dataset):
    """单品种单周期滑窗数据集。__getitem__ 返回特征 + 稠密标签 + 重构目标。"""

    def __init__(self, df: pd.DataFrame, labels: pd.DataFrame, window: int,
                 indices: np.ndarray, symbol_id: int, freq_id: int):
        self.window = window
        self.symbol_id = symbol_id
        self.freq_id = freq_id
        px = df[["open", "high", "low", "close"]].to_numpy(np.float32)
        vol = np.log1p(df["volume"].to_numpy(np.float32)) if "volume" in df else np.zeros(len(df), np.float32)
        oi = np.log1p(df["open_interest"].to_numpy(np.float32)) if "open_interest" in df else np.zeros(len(df), np.float32)
        self.px, self.vol, self.oi = px, vol, oi
        self.atr = atr(df).astype(np.float32)
        self.tod = _tod_feats(df["datetime"])
        self.day = df["datetime"].dt.date.to_numpy()
        L = labels
        self.seg_dir = np.stack([L[f"seg_dir_s{s}"] for s in range(N_SCALES)], 1).astype(np.int64) + 1  # →0/1/2
        self.bars_since = np.stack([L[f"bars_since_piv_s{s}"] for s in range(N_SCALES)], 1).astype(np.float32)
        self.amp_since = np.stack([L[f"amp_since_piv_s{s}"] for s in range(N_SCALES)], 1).astype(np.float32)
        piv_h = np.stack([L[f"is_piv_high_s{s}"] for s in range(N_SCALES)], 1)
        piv_l = np.stack([L[f"is_piv_low_s{s}"] for s in range(N_SCALES)], 1)
        self.piv_cls = (piv_h.astype(np.int64) + 2 * piv_l.astype(np.int64))  # 0无/1高/2低
        self.piv_amp = np.stack([L[f"piv_amp_s{s}"] for s in range(N_SCALES)], 1).astype(np.float32)
        # 前瞻轨"当时可知"掩码：第 t 根距上枢轴 d 根，该枢轴确认延迟 delay，
        # d >= delay 时这段结构在 t 时刻已可确认（v1.1 A2：只在已确认段评估因果版）
        conf = np.stack([L[f"confirm_delay_s{s}"] for s in range(N_SCALES)], 1).astype(np.float32)
        self.confirmed = np.zeros((len(L), N_SCALES), bool)
        for t in range(len(L)):
            for s in range(N_SCALES):
                d = int(self.bars_since[t, s])
                p = t - d
                if d > 0 and p >= 0 and conf[p, s] <= d:
                    self.confirmed[t, s] = True
        self.indices = indices  # 窗口末根的行号

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        w = self.window
        lo = j - w + 1
        anchor = self.px[j, 3]
        scale = max(float(self.atr[j]), 1e-6)
        px = (self.px[lo:j + 1] - anchor) / scale
        vol = self.vol[lo:j + 1]
        oi = self.oi[lo:j + 1]
        vol = (vol - vol.mean()) / (vol.std() + 1e-6)
        oi = (oi - oi.mean()) / (oi.std() + 1e-6)
        x = np.concatenate([px, vol[:, None], oi[:, None], self.tod[lo:j + 1]], 1).astype(np.float32)
        item = {
            "x": torch.from_numpy(x),                                    # [w,8]
            "recon": torch.from_numpy(px),                               # [w,4] 掩码重构目标
            "seg_dir": torch.from_numpy(self.seg_dir[lo:j + 1]),         # [w,3]
            "bars_since": torch.from_numpy(self.bars_since[lo:j + 1]),   # [w,3]
            "amp_since": torch.from_numpy(self.amp_since[lo:j + 1]),     # [w,3]
            "piv_cls": torch.from_numpy(self.piv_cls[lo:j + 1]),         # [w,3]
            "piv_amp": torch.from_numpy(self.piv_amp[lo:j + 1]),         # [w,3]
            "confirmed": torch.from_numpy(self.confirmed[lo:j + 1]),     # [w,3] 当时可知掩码
            "symbol_id": torch.tensor(self.symbol_id),
            "freq_id": torch.tensor(self.freq_id),
        }
        return item


def build_pattern_datasets(code: str, period: int, window: int, stride: int,
                           symbol_id: int, freq_id: int,
                           train_ratio: float = 0.7, val_ratio: float = 0.15,
                           data_dir: str = "data"):
    """按交易日切分 train/val/test（窗口末根落在段内）。返回三个 Dataset。"""
    f = Path(data_dir) / f"{code}_{period}m.csv"
    lab_f = Path(data_dir) / "labels" / f"{code}_{period}m_labels.pkl"
    df = pd.read_csv(f, parse_dates=["datetime"])
    labels = pd.read_pickle(lab_f)
    assert len(df) == len(labels), f"{code}_{period}m 行情与标签行数不一致"
    ends = np.arange(window - 1, len(df), stride)          # 窗口末根候选
    days = pd.Series(df["datetime"].dt.date).unique()
    n_tr, n_va = int(len(days) * train_ratio), int(len(days) * (train_ratio + val_ratio))
    tr_days, va_days, te_days = set(days[:n_tr]), set(days[n_tr:n_va]), set(days[n_va:])
    end_day = df["datetime"].dt.date.to_numpy()[ends]
    splits = []
    for dayset in (tr_days, va_days, te_days):
        idx = ends[np.isin(end_day, list(dayset))]
        splits.append(PatternWindowDataset(df, labels, window, idx, symbol_id, freq_id))
    print(f"  [{code}_{period}m] window={window} 样本 train/val/test="
          f"{len(splits[0])}/{len(splits[1])}/{len(splits[2])}")
    return tuple(splits)
