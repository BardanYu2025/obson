"""预测路径证据表 → Excel

每一行 = 一个样本（信号bar），列：
  时间 | 格子 | 收盘价 | 预测方向(argmax三类) | P(空)/P(无)/P(多)
  顺向最远% | 反向最远% | 先后顺序(顺先/反先/同时) | 顺向达到时刻 | 反向达到时刻
  向上最远% | 向下最远% （方向无关的原始路径，供"无"预测参考）
  数据集(val/test)

"最远" = 从信号bar收盘到当日收盘（收盘锚）之间，最高价/最低价相对信号bar收盘的最大偏移。
先后顺序 = 顺向极值点与反向极值点出现的时间先后（同日 bar 粒度）。
评估集 = 验证+测试合并，绝不用训练集。

用法：
  PYTHONPATH=src python scripts/pred_path_table.py \
    --ckpt checkpoints/q90_softfix_s42/best.pt,checkpoints/q90_softfix_s7/best.pt \
    --theta-q 0.90 --symbols rb hc i sr p j jm m y cu ag TA MA --periods 60 30 \
    --out reports/pred_path_table.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_multi_symbol import SYMBOLS, JUMP_BREAK_PCT, _theta_c_dynamic, load_foreign  # noqa: E402
from signal_live import run_probs  # noqa: E402

from obson.model import KLineTransformer, build_datasets, weekly_seq_len  # noqa: E402

DIR_CN = {0: "空", 1: "无", 2: "多"}


def collect_rows(ds, prob, split_name, key):
    """从一个子数据集还原每个样本的信号bar之后的日内路径"""
    raw = ds.data * ds.std + ds.mean  # 反归一化 → [n_bars, 4] OHLC
    ts = ds.timestamps.to_numpy()
    day_ids = ds.day_ids_global
    n = len(day_ids)
    # 每个交易日的最后一根 bar 索引（收盘锚）
    day_end = {}
    for i in range(n - 1, -1, -1):
        day_end.setdefault(int(day_ids[i]), i)
    rows = []
    for pos in range(ds.n_samples):
        j = int(ds.valid_indices[pos]) + ds.seq_len - 1  # 信号bar（窗口末根）
        a = day_end[int(day_ids[j])]
        if a <= j:
            continue
        base = float(raw[j, 3])
        if base <= 0:
            continue
        hi = raw[j + 1 : a + 1, 1].astype(np.float64) / base - 1.0
        lo = 1.0 - raw[j + 1 : a + 1, 2].astype(np.float64) / base
        i_up, i_dn = int(np.argmax(hi)), int(np.argmax(lo))
        up_pct, dn_pct = float(hi[i_up]) * 100, float(lo[i_dn]) * 100
        t_up, t_dn = pd.Timestamp(ts[j + 1 + i_up]), pd.Timestamp(ts[j + 1 + i_dn])
        p0, p1, p2 = float(prob[pos, 0]), float(prob[pos, 1]), float(prob[pos, 2])
        pred = int(np.argmax([p0, p1, p2]))
        true_label = int(ds.labels[pos])  # 2=先摸上轨 0=先摸下轨 1=无
        theta_pct = float(ds.thetas[pos])  # 该样本的通道宽 θ（%）
        hit = (pred == true_label) and pred != 1
        # 顺向/反向（预测为"无"时不定义顺反，留空）
        if pred == 2:
            fav, opp = up_pct, dn_pct
            t_fav, t_opp = t_up, t_dn
        elif pred == 0:
            fav, opp = dn_pct, up_pct
            t_fav, t_opp = t_dn, t_up
        else:
            fav = opp = np.nan
            t_fav = t_opp = pd.NaT
        if pred == 1:
            order = ""
        elif t_fav < t_opp:
            order = "顺向先"
        elif t_fav > t_opp:
            order = "反向先"
        else:
            order = "同时"
        rows.append({
            "时间": pd.Timestamp(ts[j]), "格子": key, "数据集": split_name,
            "收盘价": round(base, 2), "预测方向": DIR_CN[pred],
            "P(空)": round(p0, 4), "P(无)": round(p1, 4), "P(多)": round(p2, 4),
            "θ%": round(theta_pct, 3), "真实标签": DIR_CN[true_label],
            "命中": ("是" if hit else "否") if pred != 1 else "",
            "顺向摸轨": ("是" if fav >= theta_pct else "否") if pred != 1 else "",
            "顺向最远%": round(fav, 3) if np.isfinite(fav) else "",
            "反向最远%": round(opp, 3) if np.isfinite(opp) else "",
            "先后顺序": order,
            "顺向达到时刻": t_fav if pd.notna(t_fav) else "",
            "反向达到时刻": t_opp if pd.notna(t_opp) else "",
            "向上最远%": round(up_pct, 3), "向下最远%": round(dn_pct, 3),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="逗号分隔=种子集成")
    ap.add_argument("--symbols", nargs="+", default=["rb", "hc", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--theta-q", type=float, default=0.90)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="reports/pred_path_table.xlsx")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in args.ckpt.split(","):
        ckpt = torch.load(path.strip(), map_location="cpu", weights_only=False)
        m = KLineTransformer(ckpt["config"])
        m.load_state_dict(ckpt["model"])
        m.to(device).eval()
        models.append(m)
    daily_bars = int(getattr(ckpt["config"], "daily_bars", 0))
    foreign_bars = int(getattr(ckpt["config"], "foreign_bars", 0))
    fine_bars = int(getattr(ckpt["config"], "fine_bars", 0))
    print(f"加载 {len(models)} 个模型 | device={device} | 评估集=验证+测试")

    sheets: dict[str, list[dict]] = {}
    for code in args.symbols:
        for period in args.periods:
            key = f"{code}_{period}m"
            fp = Path(args.data_dir) / f"{code}_{period}m.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp, parse_dates=["datetime"])
            fine_df = None
            if fine_bars > 0:
                fp_fine = Path(args.data_dir) / f"{code}_5m.csv"
                if fp_fine.exists():
                    fine_df = pd.read_csv(fp_fine, parse_dates=["datetime"])
                else:
                    print(f"  [{code}] 警告: 缺细周期文件 {fp_fine}，该品种无细粒度上下文")
            seq_len = weekly_seq_len(df)
            n_train = int(len(df) * 0.7)
            c = _theta_c_dynamic(df.iloc[:n_train], seq_len, args.theta_q)
            _, val_ds, test_ds = build_datasets(
                df, seq_len=seq_len, target_offset=1,
                train_ratio=0.7, val_ratio=0.15, split_mode="time",
                symbol_id=SYMBOLS[code], label_mode="day_close",
                label_threshold=c, theta_mode="dynamic",
                jump_break_pct=JUMP_BREAK_PCT[code],
                daily_bars=daily_bars,
                foreign_close=load_foreign(code) if foreign_bars > 0 else None,
                foreign_bars=foreign_bars,
                use_vol_oi=bool(getattr(ckpt["config"], "use_vol_oi", False)),
                clean_path=bool(getattr(ckpt["config"], "clean_path", False)),
                clean_frac=float(getattr(ckpt["config"], "clean_frac", 0.5)),
                strategy_label=bool(getattr(ckpt["config"], "strategy_label", False)),
                tp_frac=float(getattr(ckpt["config"], "tp_frac", 0.8)),
                stop_frac=float(getattr(ckpt["config"], "stop_frac", 0.5)),
                fine_df=fine_df, fine_bars=fine_bars,
                fine_period_min=5, base_period_min=period,
            )
            ds_cat = ConcatDataset([val_ds, test_ds])
            prob, _ = run_probs(models, DataLoader(ds_cat, batch_size=args.batch_size), device)
            n_val = val_ds.n_samples
            rows = collect_rows(val_ds, prob[:n_val], "val", key)
            rows += collect_rows(test_ds, prob[n_val:], "test", key)
            sheets[key] = rows
            print(f"[{key}] {len(rows)} 行")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        all_rows = [r for rows in sheets.values() for r in rows]
        pd.DataFrame(all_rows).to_excel(w, sheet_name="全部", index=False)
        for key, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=key[:31], index=False)
    print(f"\n已保存: {out}（{len(sheets)} 个品种sheet + 汇总sheet，共 {len(all_rows)} 行）")


if __name__ == "__main__":
    main()
