# -*- coding: utf-8 -*-
"""路径回测：复刻用户的真实打法
  开仓: 模型喊开（argmax / 验证集标定阈值 auto），按信号 bar 收盘价进场
  止盈: 触及顺向轨的 tp_frac×θ（默认 0.8θ）→ 按该价位平
  止损: 逆向偏移 stop_frac×θ（默认 0.5θ）→ 按该价位平
  兜底: 当日收盘锚（15:00 最后一根 bar）都没触发 → 收盘价平
  同根 bar 止盈止损都触 → 保守记止损（与训练标签口径一致）
用法:
  PYTHONPATH=src python -u scripts/backtest_playbook.py \
    --ckpt models/best.pt --symbols p sr y m rb --periods 60 30 \
    --theta-q 0.90 --playbook
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
from train_multi_symbol import SYMBOLS, JUMP_BREAK_PCT, _theta_c_dynamic, _theta_c_dynamic_ms, load_foreign  # noqa: E402

from obson.model import KLineTransformer, build_datasets, weekly_seq_len  # noqa: E402
from obson.model.dataset import trading_day_ids  # noqa: E402
from obson.playbook import PLAYBOOK_V1 as PLAYBOOK, signal_mask, COST_RT  # noqa: E402


def load_models(ckpt_arg: str, device: str):
    models, cfg = [], None
    for pth in [p.strip() for p in ckpt_arg.split(",")]:
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        m = KLineTransformer(ck["config"])
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        models.append(m)
        cfg = ck["config"]
    return models, cfg


def run_probs(models, ds, batch_size, device):
    probs = []
    loader = DataLoader(ds, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            acc = None
            for m in models:
                o = m(
                    batch["seq"].to(device),
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
                acc = p1 if acc is None else acc + p1
            probs.append(acc / len(models))
    return np.concatenate(probs)


def simulate_trade(hi, lo, cl, j, a, direction, tp_frac, stop_frac, th):
    """单笔路径回放：信号 bar j 收盘进场 → 0.8θ止盈 / 0.5θ止损 / 收盘锚 a 兜底。
    同根双触记止损。返回 (pnl小数未扣成本, exit, 持仓bar数, mae小数, mfe小数, 出场bar下标)"""
    entry = cl[j]
    tp = entry * (1 + direction * tp_frac * th)
    st = entry * (1 - direction * stop_frac * th)
    mae = mfe = 0.0
    for b in range(j + 1, a + 1):
        if direction == 1:
            adverse = entry - lo[b]
            favor = hi[b] - entry
            hit_stop = lo[b] <= st
            hit_tp = hi[b] >= tp
        else:
            adverse = hi[b] - entry
            favor = entry - lo[b]
            hit_stop = hi[b] >= st
            hit_tp = lo[b] <= tp
        mae = max(mae, adverse / entry)
        mfe = max(mfe, favor / entry)
        if hit_stop and hit_tp:
            return direction * (st / entry - 1), "stop(同根双触)", b - j, mae, mfe, b
        if hit_stop:
            return direction * (st / entry - 1), "stop", b - j, mae, mfe, b
        if hit_tp:
            return direction * (tp / entry - 1), "tp", b - j, mae, mfe, b
    return direction * (cl[a] / entry - 1), "close", a - j, mae, mfe, a


def path_backtest(df: pd.DataFrame, ds, probs: np.ndarray, sig: np.ndarray,
                  tp_frac: float, stop_frac: float, cost: float):
    """逐 bar 路径回放（信号质量口径：每个信号独立开仓，允许重叠 —— 仅用于信号质量统计，
    不是资金曲线；策略口径见 single_position_backtest）。
    ds 建在 df 切片上；样本 s 的信号 bar = valid_indices[s]+seq_len-1"""
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()
    # day_ids 优先用数据集注入的口径：合约模式下段内编号（拼接帧时间戳非单调，
    # 全局 trading_day_ids 会把不同合约段的同名交易日错误合并 —— 教师模型指出）
    day_ids = getattr(ds, "day_ids_global", None)
    if day_ids is None:
        day_ids = trading_day_ids(df["datetime"])
    day_last = {}  # day_id -> 收盘锚 bar 下标
    for d in np.unique(day_ids):
        day_last[d] = int(np.flatnonzero(day_ids == d)[-1])
    vi = ds.valid_indices
    thetas = np.asarray(ds.thetas)  # 百分比
    seq_len = ds.seq_len

    pnls, exits, hold_min, mae_list, mfe_list = [], [], [], [], []
    idxs = np.flatnonzero(sig)
    # 每根 bar 时长（分钟，取中位数间隔）
    deltas_min = df["datetime"].diff().dt.total_seconds().dropna() / 60.0
    step_min = float(deltas_min.median()) if len(deltas_min) else 60.0
    for s in idxs:
        j = int(vi[s]) + seq_len - 1          # 信号 bar（收盘进场）
        a = day_last[day_ids[j]]              # 当日收盘锚
        if a <= j:
            continue
        th = thetas[s] / 100.0                # 小数
        direction = 1 if probs[s, 2] >= probs[s, 0] else -1  # 由 sig 构造时保证一致
        pnl, ex, nb, mae, mfe, _b = simulate_trade(
            hi, lo, cl, j, a, direction, tp_frac, stop_frac, th)
        pnls.append(pnl * 100 - cost)          # 扣双边成本，单位 %
        exits.append(ex)
        hold_min.append(nb * step_min)
        mae_list.append(mae * 100)
        mfe_list.append(mfe * 100)
    return (np.array(pnls), np.array(exits), np.array(hold_min),
            np.array(mae_list), np.array(mfe_list))


def single_position_backtest(candidates, cost_by_code):
    """单仓策略回测（老师审查 #2 修复）：同一品种任意时刻只持一笔仓。

    candidates: 按信号时间排序的列表，元素 = dict(
        ts=信号bar时间戳, code, key, direction, hi/lo/cl=该品种该周期df的numpy,
        j=信号bar下标, a=收盘锚下标, th=θ小数, ts_exit_hint 不用)
    规则：空仓时遇到信号 → 开仓；持仓中遇到同向信号 → 跳过；
    持仓中遇到反向信号 → 按该信号 bar 收盘价平仓（exit=flip_close，不反手）。
    返回 (trades 列表, 各品种资金曲线合并后的全组合 equity/回撤 统计)。
    """
    trades = []
    open_pos = {}  # code -> dict(direction, entry, ts_exit)
    for cand in candidates:
        code = cand["code"]
        pos = open_pos.get(code)
        if pos is not None:
            if cand["ts"] <= pos["ts_exit"]:
                # 持仓中：反向信号平仓（覆盖开仓时入账的记录），同向跳过
                if cand["direction"] != pos["direction"]:
                    px_exit = cand["cl"][cand["j"]]
                    pnl = pos["direction"] * (px_exit / pos["entry"] - 1) * 100 - cost_by_code.get(code, 0.06)
                    trades[pos["_ti"]].update({
                        "exit": "flip_close", "pnl": pnl,
                        "ts_exit_real": cand["ts"],
                        "hold_min": (cand["ts"] - pos["ts"]).total_seconds() / 60.0,
                    })
                    del open_pos[code]
                continue
            del open_pos[code]  # 仓位已了结，该品种恢复空仓
        pnl, ex, nb, mae, mfe, b = simulate_trade(
            cand["hi"], cand["lo"], cand["cl"], cand["j"], cand["a"],
            cand["direction"], cand["tp_frac"], cand["stop_frac"], cand["th"])
        pnl_pct = pnl * 100 - cost_by_code.get(code, 0.06)
        ts_exit = cand["ts_index"][b]
        rec = {"code": code, "key": cand["key"], "direction": cand["direction"],
               "ts": cand["ts"], "entry": cand["cl"][cand["j"]], "exit": ex,
               "pnl": pnl_pct, "mae": mae * 100, "mfe": mfe * 100,
               "hold_min": nb * cand["step_min"], "ts_exit": ts_exit,
               "ts_exit_real": ts_exit}
        trades.append(rec)
        # 无论止盈/止损/收盘兜底，仓位在 [ts, ts_exit] 内都占用
        open_pos[code] = {"code": code, "key": cand["key"], "direction": cand["direction"],
                          "entry": cand["cl"][cand["j"]], "ts": cand["ts"],
                          "ts_exit": ts_exit, "_ti": len(trades) - 1}
    # 末尾仍持有的仓位理论上不存在（出场都在当日锚内），防御性忽略
    trades.sort(key=lambda t: (t.get("ts_exit_real", t["ts_exit"]), t["ts"]))
    return trades


def equity_stats(trades):
    """由交易列表算资金曲线指标：累计收益、最大回撤、持仓占用率"""
    if not trades:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    mdd = float((peak - eq).max()) if len(eq) else 0.0
    span_min = (trades[-1].get("ts_exit_real", trades[-1]["ts_exit"])
                - trades[0]["ts"]).total_seconds() / 60.0
    hold_sum = sum(t["hold_min"] for t in trades)
    return {"n": len(trades), "tot": float(eq[-1]), "mdd": mdd,
            "exp": float(pnls.mean()),
            "win": float((pnls > 0).mean()),
            "occupancy": hold_sum / max(span_min, 1.0)}


def report_one(key, side, pnls, exits, hold_min, mae, mfe):
    if len(pnls) == 0:
        return None, 0.0
    n = len(pnls)
    tp_n = int((exits == "tp").sum())
    st_n = int(np.char.startswith(exits.astype(str), "stop").sum())
    fl_n = n - tp_n - st_n
    r = {
        "n": n, "tp": tp_n / n, "stop": st_n / n, "flat": fl_n / n,
        "exp": pnls.mean(), "tot": pnls.sum(),
        "med_hold": float(np.median(hold_min)),
        "mae50": float(np.median(mae)), "mae75": float(np.percentile(mae, 75)),
        "mfe50": float(np.median(mfe)),
    }
    return r, r["tot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="逗号分隔=种子集成")
    ap.add_argument("--symbols", nargs="+", default=["p", "sr", "y", "m", "rb"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--theta-q", type=float, default=None,
                    help="默认从 ckpt config 恢复（老 ckpt 回退 0.90）；与 ckpt 不一致将报错")
    ap.add_argument("--force-theta", action="store_true", help="强制用命令行 theta-q 覆盖 ckpt 口径")
    ap.add_argument("--target-per-100", type=float, default=5.0)
    ap.add_argument("--tp-frac", type=float, default=0.8, help="止盈=顺向轨的比例，默认0.8θ")
    ap.add_argument("--stop-frac", type=float, default=0.5, help="止损=逆向 θ 的比例，默认0.5θ")
    ap.add_argument("--cost-scale", type=float, default=1.0)
    ap.add_argument("--playbook", action="store_true", help="只交易手册白名单组合方向（v1，仅品种×方向）")
    ap.add_argument("--playbook-v2", action="store_true",
                    help="手册 v2：品种×周期×方向×置信度区间过滤（playbook.py 唯一权威）")
    ap.add_argument("--single-position", action="store_true",
                    help="单仓策略口径：同品种同时只持一笔，反向信号平仓不反手；输出资金曲线/最大回撤")
    ap.add_argument("--contract-mode", action="store_true",
                    help="合约模式回测：数据来自 data/contracts 段帧（须与 ckpt 训练口径一致）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, cfg = load_models(args.ckpt, device)
    # 数据口径防呆（教师模型指出）：合约模型 + 主连数据 = 无效组合
    ckpt_contract = bool(getattr(cfg, "contract_mode", False))
    if ckpt_contract and not args.contract_mode:
        raise SystemExit("该 checkpoint 是合约模式训练，回测必须加 --contract-mode；主连数据+合约模型=错配")
    if args.contract_mode and not ckpt_contract:
        print("  警告: ckpt 未记录 contract_mode（旧版训练），按命令行 --contract-mode 口径执行")
    print(f"模型 {args.ckpt}（{'集成×' + str(len(models)) if len(models) > 1 else '单模型'}）| device={device}")
    print(f"规则: 喊开即开 → +{args.tp_frac}θ 止盈 / −{args.stop_frac}θ 止损 / 收盘兜底 | 同根双触记止损")
    daily_bars = int(getattr(cfg, "daily_bars", 0))
    foreign_bars = int(getattr(cfg, "foreign_bars", 0))
    fine_bars = int(getattr(cfg, "fine_bars", 0))
    clean_path = bool(getattr(cfg, "clean_path", False))
    clean_frac = float(getattr(cfg, "clean_frac", 0.5))
    from obson.runtime import resolve_theta_q
    theta_q, theta_src = resolve_theta_q(args.theta_q, cfg,
                                         force=args.force_theta, script="backtest_playbook")
    print(f"θ_q={theta_q}（{theta_src}）")

    for mode in ["argmax", "auto"]:
        print("\n" + "=" * 88)
        print(f"信号模式: {mode}" + ("" if mode == "argmax" else f"（验证集标定，每方向每100根喊 {args.target_per_100:.0f} 次）"))
        print("=" * 88)
        print(f"  {'组合':<9}{'交易数':>6}{'止盈率':>7}{'止损率':>7}{'收盘平':>7}{'单笔期望':>10}"
              f"{'累计净盈亏':>11}{'中位持仓':>9}{'浮亏中位':>9}{'浮亏75分':>9}")
        tot, tot_n = 0.0, 0
        candidates = []  # --single-position 用
        for code in args.symbols:
            cost = COST_RT.get(code, 0.06) * args.cost_scale
            for period in args.periods:
                key = f"{code}_{period}m"
                if args.playbook and key not in PLAYBOOK:
                    continue
                fp = f"data/{code}_{period}m.csv"
                if not args.contract_mode and not Path(fp).exists():
                    continue
                if args.contract_mode:
                    # 合约模式：段帧 + 按段独立 θ 校准（与训练口径一致）
                    from obson.contract_series import build_contract_frame
                    from obson.model.dataset import build_datasets_contract
                    try:
                        cframe, cbreaks, cmask, cday_ids = build_contract_frame(code, period)
                    except (FileNotFoundError, ValueError) as e:
                        print(f"  [{key}] 合约段帧构建失败: {e}")
                        continue
                    last_seg = cframe[cframe["seg"] == cframe["seg"].max()].reset_index(drop=True)
                    seq_len = weekly_seq_len(last_seg)
                    main_days = np.unique(cday_ids[cmask])
                    n_tr_days = int(len(main_days) * 0.7)
                    tr_rows = cmask & (cday_ids <= main_days[n_tr_days - 1])
                    ms_all = []
                    for _, gdf in cframe[tr_rows].groupby("seg"):
                        gdf = gdf.reset_index(drop=True)
                        if len(gdf) < seq_len + 20:
                            continue
                        ms_all.extend(_theta_c_dynamic_ms(gdf, seq_len))
                    if not ms_all:
                        print(f"  [{key}] θ 校准无样本，跳过")
                        continue
                    c = float(np.quantile(ms_all, theta_q))
                    _, val_ds, test_ds = build_datasets_contract(
                        code, period, seq_len=seq_len,
                        train_ratio=0.7, val_ratio=0.15,
                        target_offset=1, symbol_id=SYMBOLS[code],
                        label_mode="day_close", label_threshold=c, theta_mode="dynamic",
                        daily_bars=daily_bars,
                        foreign_close=load_foreign(code) if foreign_bars > 0 else None,
                        foreign_bars=foreign_bars,
                        use_vol_oi=bool(getattr(cfg, "use_vol_oi", False)),
                        clean_path=clean_path, clean_frac=clean_frac,
                        strategy_label=bool(getattr(cfg, "strategy_label", False)),
                        tp_frac=float(getattr(cfg, "tp_frac", 0.8)),
                        stop_frac=float(getattr(cfg, "stop_frac", 0.5)),
                        base_period_min=period,
                    )
                    # 回测价格序列 = 合约段帧（valid_indices 为段帧坐标）；
                    # day_ids 用 test_ds.day_ids_global（段内编号，收盘锚不跨段）
                    test_df = cframe
                else:
                    df = pd.read_csv(fp, parse_dates=["datetime"])
                    fine_df = None
                    if fine_bars > 0:
                        fp_f = Path(f"data/{code}_5m.csv")
                        fine_df = pd.read_csv(fp_f, parse_dates=["datetime"]) if fp_f.exists() else None
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
                        use_vol_oi=bool(getattr(cfg, "use_vol_oi", False)),
                        clean_path=clean_path, clean_frac=clean_frac,
                        strategy_label=bool(getattr(cfg, "strategy_label", False)),
                        tp_frac=float(getattr(cfg, "tp_frac", 0.8)),
                        stop_frac=float(getattr(cfg, "stop_frac", 0.5)),
                        fine_df=fine_df, fine_bars=fine_bars,
                        fine_period_min=5, base_period_min=period,
                    )
                    # warm-up 口径（老师三轮修复）：test_ds 建在完整 df 上，valid_indices
                    # 直接是 df 全历史坐标，回测价格序列也用完整 df（不再切片）
                    test_df = df

                p_test = run_probs(models, test_ds, args.batch_size, device)
                if mode == "argmax":
                    am = p_test.argmax(axis=1)
                    sig_pos, sig_neg = am == 2, am == 0
                else:
                    p_val = run_probs(models, val_ds, args.batch_size, device)
                    thr = {}
                    for cls in (2, 0):
                        k = max(int(len(p_val) * args.target_per_100 / 100), 5)
                        top = np.sort(p_val[:, cls])[::-1]
                        thr[cls] = float(top[min(k - 1, len(top) - 1)])
                    sig_pos, sig_neg = p_test[:, 2] >= thr[2], p_test[:, 0] >= thr[0]
                if args.playbook:
                    side = PLAYBOOK[key]
                    if side == "long":
                        sig_neg = np.zeros_like(sig_neg)
                    elif side == "short":
                        sig_pos = np.zeros_like(sig_pos)
                if args.playbook_v2:
                    # 手册 v2：品种×周期×方向×置信度区间（playbook.py 唯一权威）
                    t_l = thr.get(2, 0.0) if mode != "argmax" else 0.0
                    t_s = thr.get(0, 0.0) if mode != "argmax" else 0.0
                    m_l, m_s = signal_mask(code, period, p_test, t_l, t_s)
                    sig_pos &= m_l
                    sig_neg &= m_s
                # 同一 bar 双向信号互斥：只留概率更大的方向
                both = sig_pos & sig_neg
                sig_pos[both & (p_test[:, 0] >= p_test[:, 2])] = False
                sig_neg[both & (p_test[:, 2] > p_test[:, 0])] = False

                if args.single_position:
                    # 收集候选信号，按品种做单仓状态机回放（不重叠、反向信号平仓）
                    hi_a = test_df["high"].to_numpy()
                    lo_a = test_df["low"].to_numpy()
                    cl_a = test_df["close"].to_numpy()
                    ts_a = test_df["datetime"].to_numpy()
                    day_ids_t = getattr(test_ds, "day_ids_global", None)
                    if day_ids_t is None:
                        day_ids_t = trading_day_ids(test_df["datetime"])
                    day_last_t = {d: int(np.flatnonzero(day_ids_t == d)[-1])
                                  for d in np.unique(day_ids_t)}
                    deltas_min = test_df["datetime"].diff().dt.total_seconds().dropna() / 60.0
                    step_min = float(deltas_min.median()) if len(deltas_min) else float(period)
                    vi = test_ds.valid_indices
                    thetas = np.asarray(test_ds.thetas)
                    for sig_d, direction in [(sig_pos, 1), (sig_neg, -1)]:
                        for s in np.flatnonzero(sig_d):
                            j = int(vi[s]) + test_ds.seq_len - 1
                            a = day_last_t[day_ids_t[j]]
                            if a <= j:
                                continue
                            candidates.append({
                                "ts": pd.Timestamp(ts_a[j]), "code": code, "key": key,
                                "direction": direction, "hi": hi_a, "lo": lo_a, "cl": cl_a,
                                "ts_index": ts_a, "j": j, "a": a,
                                "th": float(thetas[s]) / 100.0,
                                "tp_frac": args.tp_frac, "stop_frac": args.stop_frac,
                                "step_min": step_min,
                            })
                    n_sig = int(sig_pos.sum() + sig_neg.sum())
                    print(f"  {key:<9} 信号 {n_sig:>4} 个 → 并入 {code} 单仓候选池")
                    continue

                res = {}
                for direction, sig in [("多", sig_pos), ("空", sig_neg)]:
                    pnls, exits, hold, mae, mfe = path_backtest(
                        test_df, test_ds, p_test, sig, args.tp_frac, args.stop_frac, cost)
                    res[direction] = (pnls, exits, hold, mae, mfe)
                pnls = np.concatenate([res["多"][0], res["空"][0]]) if len(res["多"][0]) or len(res["空"][0]) else np.array([])
                exits = np.concatenate([res["多"][1], res["空"][1]]) if len(pnls) else np.array([])
                hold = np.concatenate([res["多"][2], res["空"][2]]) if len(pnls) else np.array([])
                mae = np.concatenate([res["多"][3], res["空"][3]]) if len(pnls) else np.array([])
                mfe = np.concatenate([res["多"][4], res["空"][4]]) if len(pnls) else np.array([])
                r, tot_k = report_one(key, PLAYBOOK.get(key, ""), pnls, exits, hold, mae, mfe)
                if r is None:
                    print(f"  {key:<9} 无信号")
                    continue
                tot += tot_k
                tot_n += r["n"]
                print(f"  {key:<9}{r['n']:>6}{r['tp']:>7.1%}{r['stop']:>7.1%}{r['flat']:>7.1%}"
                      f"{r['exp']:>+9.4f}%{r['tot']:>+10.2f}%{r['med_hold']:>7.0f}m"
                      f"{r['mae50']:>8.3f}%{r['mae75']:>8.3f}%")
        if args.single_position:
            candidates.sort(key=lambda c: c["ts"])
            cost_map = {c: COST_RT.get(c, 0.06) * args.cost_scale for c in args.symbols}
            trades = single_position_backtest(candidates, cost_map)
            st = equity_stats(trades)
            if st is None:
                print("  单仓回放：无成交")
            else:
                print(f"\n  ── 单仓策略回测（同品种同时只持一笔，反向信号平仓不反手）──")
                print(f"  成交 {st['n']} 笔 | 胜率 {st['win']:.1%} | 单笔期望 {st['exp']:+.4f}%"
                      f" | 累计净盈亏 {st['tot']:+.2f}% | 最大回撤 {st['mdd']:.2f}%"
                      f" | 持仓占用率 {st['occupancy']:.1%}")
                for code in args.symbols:
                    sub = [t for t in trades if t["code"] == code]
                    if not sub:
                        continue
                    s2 = equity_stats(sub)
                    flips = sum(1 for t in sub if t["exit"] == "flip_close")
                    print(f"    {code:<4} {s2['n']:>4} 笔 | 胜率 {s2['win']:.1%}"
                          f" | 累计 {s2['tot']:+.2f}% | 回撤 {s2['mdd']:.2f}%"
                          f" | 反向平仓 {flips} 次")
                print("  口径说明: 这才是可对照实盘的策略资金曲线；上面逐组合表是信号质量口径"
                      "（独立样本、允许重叠，不代表可执行资金曲线）。")
            continue
        if tot_n:
            print(f"  {'合计':<9}{tot_n:>6}{'':>7}{'':>7}{'':>7}{tot / tot_n:>+9.4f}%{tot:>+10.2f}%")
        print("  读法: 止盈率=摸到 0.8θ 离场；止损率=逆向 0.5θ 离场；收盘平=拖到 15:00 按收盘价结算；")
        print("        单笔期望/累计已扣双边成本；浮亏=持仓过程最大逆向偏移。测试段≈最后15%时间（约3个月）。")


if __name__ == "__main__":
    main()
