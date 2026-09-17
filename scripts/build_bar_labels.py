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

if "--contract" in sys.argv:
    # 合约模式：逐段打标（断点不跨越），拼接帧 + seg/main_mask 一并存
    from obson.contract_series import build_contract_frame
    for cal in sorted((DATA / "contracts").glob("*_roll_calendar.csv")):
        code = cal.stem.replace("_roll_calendar", "")
        for period in (60, 30, 15, 5):
            try:
                cframe, cbreaks, cmask, cday_ids = build_contract_frame(code, period)
            except (FileNotFoundError, ValueError):
                continue
            parts = []
            for seg_id, g in cframe.groupby("seg"):
                g = g.reset_index(drop=True)
                if len(g) < 200:
                    continue
                lg = label_frame(g)
                lg["seg"] = seg_id
                parts.append(lg)
            lab = pd.concat(parts, ignore_index=True)
            # main_mask 与 cframe 行对齐（label_frame 逐段 reset， concat 顺序一致）
            lab["main_mask"] = False
            pos = 0
            for seg_id, g in cframe.groupby("seg"):
                n = len(g)
                if n < 200:
                    continue
                lab.iloc[pos:pos + n, lab.columns.get_loc("main_mask")] = \
                    cmask[cframe["seg"].to_numpy() == seg_id]
                pos += n
            key = f"{code}_{period}m_contract"
            lab.to_pickle(OUT / f"{key}_labels.pkl")
            n_piv = int(lab["is_piv_high_s1"].sum() + lab["is_piv_low_s1"].sum())
            print(f"[{key}] {len(lab)} bar（{lab['seg'].nunique()} 段）→ 标签落盘 | 中尺度枢轴 {n_piv}")
            rows.append({"file": key, "bars": len(lab)})
    pd.DataFrame(rows).to_csv(OUT / "_manifest_contract.csv", index=False)
    print(f"\n共 {len(rows)} 个合约标签文件")
    sys.exit(0)

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
