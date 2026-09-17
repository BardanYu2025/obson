"""全量 per-bar 结构标签生成：13 品种 × 全周期 → data/labels/

输出: data/labels/{code}_{period}m_labels.pkl
  每根 bar 一行，18 维结构身份（3 尺度 × 6 字段）+ datetime
审计: 每品种每周期打印枢轴统计；异常（枢轴过少/延迟异常）告警
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from obson.pivot_labels import SCALES, zigzag, atr, label_frame

DATA = Path("data")
OUT = Path("data/labels")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(DATA.glob("*_*.csv")):
    code_period = f.stem
    df = pd.read_csv(f, parse_dates=["datetime"])
    if len(df) < 1000:
        print(f"[{code_period}] 仅 {len(df)} 行，跳过")
        continue
    lab = label_frame(df)
    out = OUT / f"{code_period}_labels.pkl"
    lab.to_pickle(out)

    a = atr(df)
    stats = [f"k={k}: {len(zigzag(df, k))}枢轴" for k in SCALES]
    # 健康检查：长尺度枢轴密度（每千 bar）
    n_long = len(zigzag(df, SCALES[-1]))
    density = n_long / len(df) * 1000
    warn = " ⚠️ 长尺度过稀" if density < 5 else (" ⚠️ 长尺度过密" if density > 150 else "")
    print(f"[{code_period}] {len(df)} bar → {out.name} | {' | '.join(stats)}{warn}")
    rows.append({"file": code_period, "bars": len(df), "long_density_per_1k": round(density, 2)})

print(f"\n共 {len(rows)} 个文件")
pd.DataFrame(rows).to_csv(OUT / "_manifest.csv", index=False)
