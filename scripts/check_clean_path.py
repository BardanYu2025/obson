# -*- coding: utf-8 -*-
"""clean_path 标签翻转抽查：对比 rb/sr/p × 60/30m 开/关干净路径后的标签分布"""
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
    return {k: (cnt.get(k, 0), cnt.get(k, 0) / n) for k in (2, 1, 0)}


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
        _, _, clean = build_datasets(df, **common, clean_path=True, clean_frac=0.5)
        lb, lc = np.asarray(base.labels), np.asarray(clean.labels)
        assert len(lb) == len(lc), (len(lb), len(lc))
        flip_dir = int(((lb != 1) & (lc == 1)).sum())
        n_dir = int((lb != 1).sum())
        key = f"{code}_{period}m"
        print(f"\n[{key}] 测试集 n={len(lb)}")
        for name, d in [("关", dist(lb)), ("开", dist(lc))]:
            print(f"  clean={name}: 多 {d[2][0]:4d}({d[2][1]:.1%})  无 {d[1][0]:4d}({d[1][1]:.1%})  空 {d[0][0]:4d}({d[0][1]:.1%})")
        print(f"  方向样本翻转: {flip_dir}/{n_dir} = {flip_dir / max(n_dir, 1):.1%}")
        assert set(np.unique(lc)) <= {0, 1, 2}
print("\nOK 标签集合 ∈ {0,1,2}")
