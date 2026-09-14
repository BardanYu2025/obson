"""置信度分桶 EV 表 —— 不同信心档位下，信号到底值多少钱

协议（老师审查 #3 修复，2026-09-11 起）：
  1. 分桶表只在【验证集】上生成 —— 这里允许"看数据选规则"，
     输出 confidence_ev_table_val.csv（选择集，带偏差警示）；
  2. 冻结的手册规则（src/obson/playbook.py RULES）只在【测试集】上评估一次，
     输出 confidence_ev_frozen_test.csv —— 这才是可对外说的独立测试数字；
  3. 旧版"val+test 合并分桶"口径已废弃（白名单曾由此选出 = 选择偏差污染），
     历史报告中由该口径得出的结论不得再标注"纯测试集验证"。

另加"无退潮"轨迹分桶（仅验证集）：以上一根 bar 喊出信号为前提，按 Δ无
分桶看摸轨率/收益 —— 验证"信心回落该不该平仓"。

用法：
  PYTHONPATH=src python scripts/confidence_ev_table.py \
    --ckpt checkpoints/q90_softfix_s42/best.pt,checkpoints/q90_softfix_s7/best.pt \
    --theta-q 0.90 --symbols rb hc i sr p j jm m y cu ag TA MA --periods 60 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_multi_symbol import SYMBOLS, JUMP_BREAK_PCT, _theta_c_dynamic, load_foreign  # noqa: E402
from signal_live import run_probs, softmax_np  # noqa: E402

from obson.model import KLineTransformer, build_datasets, weekly_seq_len  # noqa: E402
from obson.playbook import RULES, CANDIDATE_RULES  # noqa: E402

# 信心分桶边界（对多头/空头对称使用）
EDGES = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 1.01]
LABELS = ["<0.30", "0.30~0.35", "0.35~0.40", "0.40~0.45", "0.45~0.50", "≥0.50"]
# Δ无 分桶（本根无信心 − 上根无信心）
DNONE_EDGES = [-1.01, -0.05, 0.0, 0.05, 0.15, 1.01]
DNONE_LABELS = ["无退≥5pt", "无退0~5pt", "无涨0~5pt", "无涨5~15pt", "无涨≥15pt"]


def bucket_table(conf, hit, fwd, maxfav, touch_min, direction):
    """conf=该方向信心, hit=是否摸到该方向轨, fwd=持有到收盘收益(已按方向带符号),
    maxfav=最大顺向偏移%(m*θ), touch_min=触达分钟"""
    rows = []
    for lo, hi, lab in zip(EDGES[:-1], EDGES[1:], LABELS):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "方向": direction, "信心档": lab, "样本数": int(m.sum()),
            "摸轨率": float(np.nanmean(hit[m])),
            "E[收盘收益%]": float(np.nanmean(fwd[m])),
            "E[最大顺向%]": float(np.nanmean(maxfav[m])),
            "中位触达min": float(np.nanmedian(touch_min[m])) if np.isfinite(touch_min[m]).any() else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="逗号分隔=种子集成")
    ap.add_argument("--symbols", nargs="+", default=["rb", "hc", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--theta-q", type=float, default=None,
                    help="默认从 ckpt config 恢复（老 ckpt 回退 0.90）；与 ckpt 不一致将报错")
    ap.add_argument("--force-theta", action="store_true", help="强制用命令行 theta-q 覆盖 ckpt 口径")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="reports")
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
    from obson.runtime import resolve_theta_q
    theta_q, theta_src = resolve_theta_q(args.theta_q, ckpt["config"],
                                         force=args.force_theta, script="confidence_ev_table")
    print(f"加载 {len(models)} 个模型 | device={device} | θ_q={theta_q}（{theta_src}）")
    print("协议: 分桶选规则只用验证集；冻结手册规则只在测试集评估一次（独立测试）")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    all_rows, traj_rows, frozen_rows = [], [], []

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
            c = _theta_c_dynamic(df.iloc[:n_train], seq_len, theta_q)
            _, val_ds, test_ds = build_datasets(
                df, seq_len=seq_len, target_offset=1,
                train_ratio=0.7, val_ratio=0.15, split_mode="time",
                symbol_id=SYMBOLS[code], label_mode="day_close",
                label_threshold=c, theta_mode="dynamic",
                jump_break_pct=JUMP_BREAK_PCT[code],
                daily_bars=daily_bars,
                foreign_close=load_foreign(code) if foreign_bars > 0 else None,
                foreign_bars=foreign_bars,
                soft_label=True,
                use_vol_oi=bool(getattr(ckpt["config"], "use_vol_oi", False)),
                clean_path=bool(getattr(ckpt["config"], "clean_path", False)),
                clean_frac=float(getattr(ckpt["config"], "clean_frac", 0.5)),
                strategy_label=bool(getattr(ckpt["config"], "strategy_label", False)),
                tp_frac=float(getattr(ckpt["config"], "tp_frac", 0.8)),
                stop_frac=float(getattr(ckpt["config"], "stop_frac", 0.5)),
                fine_df=fine_df, fine_bars=fine_bars,
                fine_period_min=5, base_period_min=period,
            )
            # ── 验证集 / 测试集分别跑（不合并，防止选择偏差污染测试集）──
            prob_v, true_v = run_probs(models, DataLoader(val_ds, batch_size=args.batch_size), device)
            prob_t, true_t = run_probs(models, DataLoader(test_ds, batch_size=args.batch_size), device)

            fwd_v, thetas_v = val_ds.fwd_rets, val_ds.thetas
            touch_v = val_ds.touch_minutes
            soft_v = val_ds.soft_targets
            m_dn_v, m_up_v = soft_v[:, 0] * thetas_v, soft_v[:, 1] * thetas_v

            print("\n" + "=" * 78)
            print(f"[{key}] val={len(true_v)} test={len(true_t)} | θ_q={theta_q} c={c:.3f}")

            # ── 多头/空头分桶（仅验证集 = 规则选择集，带选择偏差）──
            rows = bucket_table(prob_v[:, 2], (true_v == 2).astype(float), fwd_v, m_up_v, touch_v, "多")
            rows += bucket_table(prob_v[:, 0], (true_v == 0).astype(float), -fwd_v, m_dn_v, touch_v, "空")
            all_rows += [{"品种": key, **r} for r in rows]
            df_out = pd.DataFrame(rows)
            print("  [选择集·验证集] 分桶（可用于选规则，结论不等于独立测试）:")
            print(df_out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

            # ── 冻结手册规则在测试集上的独立评估（每规则一行，不做任何调参）──
            fwd_t = test_ds.fwd_rets
            rules = RULES.get((code, period), {})
            for direction, cls, fwd_sign in (("多", 2, 1.0), ("空", 0, -1.0)):
                dkey = "long" if cls == 2 else "short"
                if dkey not in rules:
                    continue
                lo, hi = rules[dkey]
                m = (prob_t[:, cls] >= lo) & (prob_t[:, cls] < hi)
                n = int(m.sum())
                if n == 0:
                    frozen_rows.append({"品种": key, "方向": direction, "区间": f"[{lo:.2f},{hi:.2f})",
                                        "样本数": 0, "摸轨率": float("nan"), "E[收盘收益%]": float("nan")})
                    continue
                frozen_rows.append({
                    "品种": key, "方向": direction, "区间": f"[{lo:.2f},{hi:.2f})",
                    "样本数": n,
                    "摸轨率": float((true_t[m] == cls).mean()),
                    "E[收盘收益%]": float(np.nanmean(fwd_sign * fwd_t[m])),
                })
            df_f = pd.DataFrame([r for r in frozen_rows if r["品种"] == key])
            if len(df_f):
                print("  [独立评估·测试集] 冻结手册规则（无调参，可对外报告）:")
                print(df_f.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

            # ── 候选规则的一次性测试集评估（过则进 RULES，败则删除不重试）──
            cand = CANDIDATE_RULES.get((code, period), {})
            cand_rows = []
            for direction, cls, fwd_sign in (("多", 2, 1.0), ("空", 0, -1.0)):
                dkey = "long" if cls == 2 else "short"
                if dkey not in cand:
                    continue
                lo, hi = cand[dkey]
                m = (prob_t[:, cls] >= lo) & (prob_t[:, cls] < hi)
                n = int(m.sum())
                base = float((true_t == cls).mean())  # 测试集基准摸轨率
                cand_rows.append({
                    "品种": key, "方向": direction, "区间": f"[{lo:.2f},{hi:.2f})",
                    "样本数": n,
                    "摸轨率": float((true_t[m] == cls).mean()) if n else float("nan"),
                    "基准": base,
                    "E[收盘收益%]": float(np.nanmean(fwd_sign * fwd_t[m])) if n else float("nan"),
                })
            if cand_rows:
                print("  [独立评估·测试集] 候选规则（一次性：过则进手册，败则删除不重试）:")
                print(pd.DataFrame(cand_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

            # ── "无退潮"轨迹分桶（仅验证集）：上根已喊信号(信心≥0.40)，
            #    看本根 Δ无 与最终是否摸到该方向轨 ──
            prob, true, fwd = prob_v, true_v, fwd_v
            for cls, dname in ((2, "多"), (0, "空")):
                sig_prev = prob[:-1, cls] >= 0.40
                d_none = prob[1:, 1] - prob[:-1, 1]  # 本根无 − 上根无
                hit_next = (true[1:] == cls).astype(float)
                for lo, hi, lab in zip(DNONE_EDGES[:-1], DNONE_EDGES[1:], DNONE_LABELS):
                    m = sig_prev & (d_none >= lo) & (d_none < hi)
                    if m.sum() < 10:
                        continue
                    traj_rows.append({
                        "品种": key, "方向": dname, "Δ无档": lab, "样本数": int(m.sum()),
                        "摸轨率": float(np.nanmean(hit_next[m])),
                        "E[收盘收益%]": float(np.nanmean((fwd[1:] if cls == 2 else -fwd[1:])[m])),
                    })
            df_t = pd.DataFrame([r for r in traj_rows if r["品种"] == key])
            if len(df_t):
                print("  轨迹分桶（上根已喊信号，按本根Δ无分组）:")
                print(df_t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_dir / "confidence_ev_table_val.csv", index=False)
    if frozen_rows:
        pd.DataFrame(frozen_rows).to_csv(out_dir / "confidence_ev_frozen_test.csv", index=False)
    if traj_rows:
        pd.DataFrame(traj_rows).to_csv(out_dir / "confidence_traj_table.csv", index=False)
    print(f"\n已保存: {out_dir/'confidence_ev_table_val.csv'}（选择集·验证集）"
          f", {out_dir/'confidence_ev_frozen_test.csv'}（冻结规则·独立测试）"
          f", {out_dir/'confidence_traj_table.csv'}")


if __name__ == "__main__":
    main()
