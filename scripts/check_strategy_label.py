# -*- coding: utf-8 -*-
"""策略对齐标签预研：对比 原标签 / 策略标签(0.8θ止盈,0.5θ止损) 的三类占比"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from collections import Counter
from obson.model.dataset import build_datasets, weekly_seq_len
from train_multi_symbol import SYMBOLS, JUMP_BREAK_PCT, _theta_c_dynamic, load_foreign


def dist(labels):
    cnt = Counter(labels.tolist())
    n = len(labels)
    return {k: cnt.get(k, 0) / n for k in (2, 1, 0)}


print(f"{'组合':<9}{'':>6}{'原标签 多/无/空':<26}{'策略标签 多/无/空':<26}{'翻转分析'}")
for code in ["rb", "sr", "p"]:
    for period in [60, 30]:
        df = pd.read_csv(f"data/{code}_{period}m.csv", parse_dates=["datetime"])
        seq_len = weekly_seq_len(df)
        n_train = int(len(df) * 0.7)
        c = _theta_c_dynamic(df.iloc[:n_train], seq_len, 0.90)
        common = dict(
            seq_len=seq_len, target_offset=1, train_ratio=0.7, val_ratio=0.15,
            split_mode="time", symbol_id=SYMBOLS[code], label_mode="day_close",
            label_threshold=c, theta_mode="dynamic", jump_break_pct=JUMP_BREAK_PCT[code],
            daily_bars=20, foreign_close=load_foreign(code), foreign_bars=20,
        )
        _, _, base = build_datasets(df, **common)
        _, _, strat = build_datasets(df, **common, strategy_label=True, tp_frac=0.8, stop_frac=0.5)
        lb, ls = np.asarray(base.labels), np.asarray(strat.labels)
        assert len(lb) == len(ls)
        db, dsk = dist(lb), dist(ls)
        gain = int(((lb == 1) & (ls != 1)).sum())   # 原无 → 新方向（止盈轨更近，新增的方向）
        lose = int(((lb != 1) & (ls == 1)).sum())   # 原方向 → 新无（先止损后到位的被杀掉）
        key = f"{code}_{period}m"
        print(f"{key:<10}n={len(lb):<5}"
              f"{db[2]:.1%}/{db[1]:.1%}/{db[0]:.1%}"
              f"{'':<8}{dsk[2]:.1%}/{dsk[1]:.1%}/{dsk[0]:.1%}"
              f"{'':<8}新增方向 {gain} / 杀掉脏方向 {lose}")
