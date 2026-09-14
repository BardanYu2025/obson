"""信号回测 —— 严格模拟你的实盘规则

规则（对每个测试集 bar，模型给出方向后）：
  - 信号 bar 收盘价开仓（喊多=开多，喊空=开空）
  - 先碰到有利轨 → 平仓，盈利 +θ（θ = 该 bar 动态边界宽度）
  - 先碰到不利轨 → 止损，亏损 −θ
  - 到当天收盘都没碰 → 按收盘价平仓，盈亏 = 实际涨跌幅
  - 每笔扣除双边成本（手续费+滑点，按品种 2 跳估算，可用 --cost-scale 调整）

信号判定两种模式：
  argmax : 模型直接输出方向（现状）
  auto   : 验证集按目标频率自适应定阈值（每 100 根K线喊 N 次），低于阈值不喊

用法（在训练同款机器上跑，需 best.pt + data/）：
  PYTHONPATH=src python scripts/backtest_signals.py \
    --ckpt checkpoints/edge_s42/best.pt \
    --symbols rb hc i sr p --periods 60 30
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

from obson.model import KLineTransformer, build_datasets, weekly_seq_len  # noqa: E402

# 双边成本（开仓+平仓的手续费+滑点），按约 2 跳估算（%）
COST_RT = {"rb": 0.06, "hc": 0.06, "i": 0.13, "sr": 0.035, "p": 0.05, "j": 0.05, "jm": 0.06,
           "m": 0.065, "y": 0.05, "cu": 0.03, "ag": 0.025, "TA": 0.08, "MA": 0.08}


def collect(ckpt_path: str, symbols: list[str], periods: list[int], batch_size: int,
            theta_q: float, device: str):
    """重建数据集 + 加载模型（逗号分隔=多种子概率平均集成），返回 {key: dict(val=..., test=...)}"""
    paths = [p.strip() for p in ckpt_path.split(",")]
    models, ckpt = [], None
    for pth in paths:
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        m = KLineTransformer(ck["config"])
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        models.append(m)
        ckpt = ck
    if len(models) > 1:
        print(f"  种子集成: {len(models)} 个模型概率平均")
    daily_bars = int(getattr(ckpt["config"], "daily_bars", 0))
    foreign_bars = int(getattr(ckpt["config"], "foreign_bars", 0))
    fine_bars = int(getattr(ckpt["config"], "fine_bars", 0))
    if daily_bars > 0:
        print(f"  该模型启用日线金字塔: daily_bars={daily_bars}")
    if foreign_bars > 0:
        print(f"  该模型启用外盘日线: foreign_bars={foreign_bars}")
    if fine_bars > 0:
        print(f"  该模型启用细粒度上下文: 5m×{fine_bars}根")

    out = {}
    for code in symbols:
        sym_id = SYMBOLS[code]
        for period in periods:
            key = f"{code}_{period}m"
            fp = f"data/{code}_{period}m.csv"
            if not Path(fp).exists():
                print(f"  [{key}] 无数据，跳过")
                continue
            df = pd.read_csv(fp, parse_dates=["datetime"])
            fine_df = None
            if fine_bars > 0:
                fp_fine = f"data/{code}_5m.csv"
                if Path(fp_fine).exists():
                    fine_df = pd.read_csv(fp_fine, parse_dates=["datetime"])
                else:
                    print(f"  [{code}] 警告: 缺细周期文件 {fp_fine}，该品种无细粒度上下文")
            seq_len = weekly_seq_len(df)
            n_train = int(len(df) * 0.7)
            c = _theta_c_dynamic(df.iloc[:n_train], seq_len, theta_q)
            train_ds, val_ds, test_ds = build_datasets(
                df, seq_len=seq_len, target_offset=1,
                train_ratio=0.7, val_ratio=0.15, split_mode="time",
                symbol_id=sym_id, label_mode="day_close",
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
            splits = {}
            for name, ds in [("val", val_ds), ("test", test_ds)]:
                loader = DataLoader(ds, batch_size=batch_size)
                probs, trues, fwds, thetas, tods = [], [], [], [], []
                with torch.no_grad():
                    for batch in loader:
                        x = batch["seq"].to(device)
                        prob_acc = None
                        for model in models:
                            o = model(
                                x,
                                temporal_feat=batch["temporal_feat"].to(device),
                                time_pos=batch["time_pos"].to(device),
                                freq_feat=batch["freq_feat"].to(device),
                                symbol_id=batch["symbol_id"].to(device),
                                daily_ctx=(batch["daily_ctx"].to(device) if "daily_ctx" in batch else None),
                                foreign_ctx=(batch["foreign_ctx"].to(device) if "foreign_ctx" in batch else None),
                                fine_ctx=(batch["fine_ctx"].to(device) if "fine_ctx" in batch else None),
                            )
                            lg = o["logits"].cpu().numpy()
                            lg = lg - lg.max(axis=1, keepdims=True)
                            p1 = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
                            prob_acc = p1 if prob_acc is None else prob_acc + p1
                        probs.append(prob_acc / len(models))
                        trues.append(batch["label"].numpy())
                        fwds.append(batch["fwd_ret"].numpy())
                        thetas.append(batch["theta"].numpy())
                        tods.append(batch["tod_minutes"].numpy())
                splits[name] = {
                    "prob": np.concatenate(probs), "true": np.concatenate(trues),
                    "fwd": np.concatenate(fwds), "theta": np.concatenate(thetas),
                    "tod": np.concatenate(tods),
                }
            out[key] = splits
            print(f"  [{key}] c={c:.3f} | val={len(val_ds)} test={len(test_ds)}")
    return out


# 作战手册 v1：组合 → 允许的方向（"long"/"short"/"both"）；不在表里的组合 playbook 模式下不做
# 唯一权威定义在 src/obson/playbook.py（老师审查 #4 修复：禁止各脚本自维护副本）
from obson.playbook import PLAYBOOK_V1 as PLAYBOOK, signal_mask  # noqa: E402


def report(key: str, d: dict, sig_pos, sig_neg, cost: float) -> dict | None:
    t, fwd, th = d["true"], d["fwd"], d["theta"]
    wins_l, losses_l, flats_l = [], [], []
    for sig, cls in [(sig_pos, 2), (sig_neg, 0)]:
        idx = np.flatnonzero(sig)
        w = t[idx] == cls                # 先摸有利轨：+θ
        l = t[idx] == 2 - cls            # 先摸不利轨：−θ
        f = ~(w | l)                     # 都没碰：收盘平（喊空取反）
        wins_l.append(th[idx][w])
        losses_l.append(-th[idx][l])
        flats_l.append(np.where(cls == 2, 1, -1) * fwd[idx][f])
    wins = np.concatenate(wins_l)
    losses = np.concatenate(losses_l)
    flats = np.concatenate(flats_l)
    n_tr = len(wins) + len(losses) + len(flats)
    if n_tr == 0:
        return None
    gross_all = np.concatenate([wins, losses, flats])
    net_all = gross_all - cost
    return {
        "n": n_tr, "win_rate": len(wins) / n_tr,
        "avg_win": wins.mean() if len(wins) else float("nan"),
        "avg_loss": losses.mean() if len(losses) else float("nan"),
        "n_flat": len(flats),
        "exp_gross": gross_all.mean(), "exp_net": net_all.mean(),
        "total_net": net_all.sum(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--symbols", nargs="+", default=["rb", "hc", "i", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--theta-q", type=float, default=None,
                    help="默认从 ckpt config 恢复（老 ckpt 回退 0.90）；与 ckpt 不一致将报错")
    ap.add_argument("--force-theta", action="store_true", help="强制用命令行 theta-q 覆盖 ckpt 口径")
    ap.add_argument("--target-per-100", type=float, default=5.0,
                    help="auto 模式：验证集上每 100 根K线的目标喊开次数（每个方向各自标定）")
    ap.add_argument("--cost-scale", type=float, default=1.0, help="成本倍率，压力测试用")
    ap.add_argument("--playbook", action="store_true",
                    help="作战手册过滤：只做白名单组合方向 "
                         "(p_60m双向/p_30m空/sr_60m多/y_30m空/m_60m空)，其余信号全部丢弃")
    ap.add_argument("--playbook-v2", action="store_true",
                    help="手册 v2：品种×周期×方向×置信度区间过滤（playbook.py 唯一权威），"
                         "优先级高于 --playbook")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 {args.ckpt} | device={device}")
    from obson.runtime import resolve_theta_q
    ck0 = torch.load(args.ckpt.split(",")[0].strip(), map_location="cpu", weights_only=False)
    theta_q, theta_src = resolve_theta_q(args.theta_q, ck0["config"],
                                         force=args.force_theta, script="backtest_signals")
    print(f"θ_q={theta_q}（{theta_src}）")
    data = collect(args.ckpt, args.symbols, args.periods, args.batch_size, theta_q, device)

    for mode in ["argmax", "auto"]:
        print("\n" + "=" * 78)
        print(f"信号模式: {mode}" + ("" if mode == "argmax" else f"（验证集标定，每方向每100根K线喊 {args.target_per_100:.0f} 次）"))
        print("=" * 78)
        print(f"  {'组合':<9}{'交易数':>6}{'胜率':>7}{'平均盈':>8}{'平均亏':>8}{'收盘平':>7}"
              f"{'单笔期望(毛)':>12}{'单笔期望(净)':>12}{'累计净盈亏':>11}")
        tot_net, tot_n = 0.0, 0
        for key, splits in data.items():
            code = key.split("_")[0]
            cost = COST_RT.get(code, 0.06) * args.cost_scale
            d = splits["test"]
            p = d["prob"]
            if mode == "argmax":
                am = p.argmax(axis=1)
                sig_pos, sig_neg = am == 2, am == 0
            else:
                v = splits["val"]
                thr = {}
                for cls in (2, 0):
                    k = max(int(len(v["prob"]) * args.target_per_100 / 100), 5)
                    top = np.sort(v["prob"][:, cls])[::-1]
                    thr[cls] = float(top[min(k - 1, len(top) - 1)])
                sig_pos, sig_neg = p[:, 2] >= thr[2], p[:, 0] >= thr[0]
            if args.playbook_v2:
                # 手册 v2：置信度区间过滤（唯一权威 playbook.py）；argmax 模式无标定阈值，
                # 传 0 → 只按置信度区间过滤
                period = int(key.rsplit("_", 1)[1].rstrip("m"))
                t_l = thr.get(2, 0.0) if mode != "argmax" else 0.0
                t_s = thr.get(0, 0.0) if mode != "argmax" else 0.0
                m_l, m_s = signal_mask(code, period, p, t_l, t_s)
                sig_pos &= m_l
                sig_neg &= m_s
            elif args.playbook:
                side = PLAYBOOK.get(key)
                if side is None:
                    sig_pos = sig_neg = np.zeros(len(d["true"]), bool)
                elif side == "long":
                    sig_neg = np.zeros_like(sig_neg)
                elif side == "short":
                    sig_pos = np.zeros_like(sig_pos)
            r = report(key, d, sig_pos, sig_neg, cost)
            if r is None:
                print(f"  {key:<9} 无信号")
                continue
            tot_net += r["total_net"]
            tot_n += r["n"]
            tag = PLAYBOOK.get(key, "")
            print(f"  {key:<9}{r['n']:>6}{r['win_rate']:>7.1%}{r['avg_win']:>+8.3f}%{r['avg_loss']:>+8.3f}%"
                  f"{r['n_flat']:>7}{r['exp_gross']:>+11.4f}%{r['exp_net']:>+11.4f}%{r['total_net']:>+10.2f}%"
                  + (f"  [{tag}]" if tag else ""))
        if tot_n:
            print(f"  {'合计':<9}{tot_n:>6}{'':>7}{'':>8}{'':>8}{'':>7}{'':>12}"
                  f"{tot_net / tot_n:>+11.4f}%{tot_net:>+10.2f}%")
        print("  读法: 胜率=摸到有利轨的比例；平均盈/亏≈±θ；收盘平=拖到收盘按实际涨跌结算的笔数；")
        print("        单笔期望(净)已扣双边成本。测试段≈最后15%时间（约3个月）。")


if __name__ == "__main__":
    main()
