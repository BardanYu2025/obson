"""盘中信号脚本 —— 喂最新K线，输出 方向/信心/上下轨/触达时间

三部分功能：
  1. 自检：测试集上按"验证集自适应阈值"跑一遍信号报告（确认 best.pt 正常）
  2. 半成品bar衰减测试：bar 只走 1/4、1/2、3/4 时信号命中率掉多少 → 定刷新节奏
  3. 实盘信号：读取最新CSV最后一根完整bar，输出可下单信息

用法（AutoDL 或本地，需 best.pt + data/ 最新CSV）：
  PYTHONPATH=src python scripts/signal_live.py \
    --ckpt checkpoints/q80_s42/best.pt --theta-q 0.80 \
    --symbols rb hc sr p --periods 60 30
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_multi_symbol import SYMBOLS, JUMP_BREAK_PCT, _theta_c_dynamic, load_foreign  # noqa: E402

from obson.model import KLineTransformer, build_datasets, weekly_seq_len  # noqa: E402
from obson.model.dataset import KLineDataset, trading_day_ids  # noqa: E402

TICK = {"rb": 1.0, "hc": 1.0, "i": 0.5, "sr": 1.0, "p": 2.0, "j": 0.5, "jm": 0.5,
        "m": 1.0, "y": 2.0, "cu": 10.0, "ag": 1.0, "TA": 2.0, "MA": 1.0}


def softmax_np(lg: np.ndarray) -> np.ndarray:
    lg = lg - lg.max(axis=-1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(axis=-1, keepdims=True)


def last_complete_bar(df: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[int, float]:
    """返回 (最后一根已收盘 bar 的行索引 j*, bar 时长分钟)。
    判定：bar 时长 = 同交易日内相邻 bar 间隔的中位数（排除隔夜/休市跳空）；
    now >= 末根 bar 起始时刻 + 时长 → 末根已收盘，j*=len-1；否则 j*=len-2。
    老师审查 #1 修复：模型输入末 bar 必须 = 展示的信号 bar = 最近完整 bar。"""
    ts = df["datetime"]
    day = trading_day_ids(ts)
    diffs = ts.diff().dt.total_seconds().to_numpy() / 60.0
    same_day = np.diff(day) == 0
    intra = diffs[1:][same_day]
    dur = float(np.median(intra)) if len(intra) else 60.0
    if now is None:
        now = pd.Timestamp.now()
    n = len(df) - 1
    if now >= ts.iloc[n] + pd.Timedelta(minutes=dur):
        return n, dur
    return max(n - 1, 0), dur


def run_probs(models, loader, device, mutilate_last_bar: float | None = None):
    """跑一个 loader，返回 probs/labels；多模型时输出概率集成平均；
    mutilate_last_bar=f 时把末根bar改成走了 f 的半成品"""
    probs, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["seq"].clone()
            if mutilate_last_bar is not None:
                f = mutilate_last_bar
                o, h, l, c = x[:, -1, 0], x[:, -1, 1], x[:, -1, 2], x[:, -1, 3]
                nc = o + (c - o) * f
                nh = torch.maximum(o, nc) + (h - torch.maximum(o, c)) * f
                nl = torch.minimum(o, nc) - (torch.minimum(o, c) - l) * f
                x[:, -1, 1], x[:, -1, 2], x[:, -1, 3] = nh, nl, nc
            ens = None
            for model in models:
                out = model(
                    x.to(device),
                    temporal_feat=batch["temporal_feat"].to(device),
                    time_pos=batch["time_pos"].to(device),
                    freq_feat=batch["freq_feat"].to(device),
                    symbol_id=batch["symbol_id"].to(device),
                    daily_ctx=(batch["daily_ctx"].to(device) if "daily_ctx" in batch else None),
                    foreign_ctx=(batch["foreign_ctx"].to(device) if "foreign_ctx" in batch else None),
                    cross_ctx=(batch["cross_ctx"].to(device) if "cross_ctx" in batch else None),
                    cross_mask=(batch["cross_mask"].to(device) if "cross_mask" in batch else None),
                    fine_ctx=(batch["fine_ctx"].to(device) if "fine_ctx" in batch else None),
                )
                p = softmax_np(out["logits"].cpu().numpy())
                ens = p if ens is None else ens + p
            probs.append(ens / len(models))
            trues.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(trues)


def maybe_cross_kwargs(cfg, symbols=None):
    """checkpoint config 含跨品种/板块上下文时，构造数据集所需的日线参数。
    品种集合必须与训练时的 config.cross_symbol_ids 一致（与 args.symbols 无关）。"""
    cb = int(getattr(cfg, "cross_daily_bars", 0))
    if cb <= 0 and not getattr(cfg, "sector_groups", None):
        return {}
    if cb <= 0:
        cb = int(getattr(cfg, "daily_bars", 0)) or 20  # sector 模式窗口与日K金字塔对齐
    from train_multi_symbol import build_cross_daily, SYMBOLS as _S
    ids = getattr(cfg, "cross_symbol_ids", None)
    codes = [c for c, i in _S.items() if ids is None or i in ids]
    return {"cross_daily": build_cross_daily(codes), "cross_daily_bars": cb}


def calibrate(val_prob, per_100: float):
    """验证集自适应阈值：每方向每100根K线喊 per_100 次"""
    thr = {}
    for cls in (2, 0):
        k = max(int(len(val_prob) * per_100 / 100), 5)
        top = np.sort(val_prob[:, cls])[::-1]
        thr[cls] = float(top[min(k - 1, len(top) - 1)])
    return thr


def hit_table(prob, true, thr):
    """返回 (喊多命中率, 喊空命中率, 喊多覆盖率, 喊空覆盖率)"""
    res = []
    for cls in (2, 0):
        sig = prob[:, cls] >= thr[cls]
        res.append(float((true[sig] == cls).mean()) if sig.any() else float("nan"))
        res.append(float(sig.mean()))
    return res  # hit_pos, cov_pos, hit_neg, cov_neg  -> 注意顺序


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint 路径；逗号分隔传多个=种子集成（概率平均，降种子方差）")
    ap.add_argument("--symbols", nargs="+", default=["rb", "hc", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--theta-q", type=float, default=None,
                    help="默认从 ckpt config 恢复（老 ckpt 无此字段回退 0.90）；显式传入且与 ckpt 不一致将报错")
    ap.add_argument("--force-theta", action="store_true", help="强制用命令行 theta-q 覆盖 ckpt 口径")
    ap.add_argument("--target-per-100", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--now", default=None,
                    help="测试用：覆盖当前时刻（如 '2026-09-10 21:30'），判断末根 bar 是否收盘")
    args = ap.parse_args()
    now_override = pd.Timestamp(args.now) if args.now else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for path in args.ckpt.split(","):
        ckpt = torch.load(path.strip(), map_location="cpu", weights_only=False)
        m = KLineTransformer(ckpt["config"])
        m.load_state_dict(ckpt["model"])
        m.to(device).eval()
        models.append(m)
    daily_bars = int(getattr(ckpt["config"], "daily_bars", 0))
    manifest = ckpt.get("data_manifest")
    if manifest:
        print(f"  ckpt 训练口径清单: label={manifest.get('label_anchor')}/{manifest.get('theta_mode')}"
              f" θ_q={manifest.get('theta_q')} soft={manifest.get('soft_label')}"
              f" clean={manifest.get('clean_path')} strat={manifest.get('strategy_label')}"
              f" 窗口={manifest.get('window_days')}天 seed={manifest.get('seed')}")
    foreign_bars = int(getattr(ckpt["config"], "foreign_bars", 0))
    use_vol_oi = bool(getattr(ckpt["config"], "use_vol_oi", False))
    clean_path = bool(getattr(ckpt["config"], "clean_path", False))
    clean_frac = float(getattr(ckpt["config"], "clean_frac", 0.5))
    strategy_label = bool(getattr(ckpt["config"], "strategy_label", False))
    tp_frac = float(getattr(ckpt["config"], "tp_frac", 0.8))
    stop_frac = float(getattr(ckpt["config"], "stop_frac", 0.5))
    fine_bars = int(getattr(ckpt["config"], "fine_bars", 0))
    fine_period = 5  # 细粒度周期目前固定5m（训练侧同口径）
    cross_kw = maybe_cross_kwargs(ckpt["config"], args.symbols)
    from obson.runtime import resolve_theta_q
    theta_q, theta_src = resolve_theta_q(args.theta_q, ckpt["config"],
                                         force=args.force_theta, script="signal_live")
    print(f"加载 {len(models)} 个模型（{'种子集成' if len(models) > 1 else '单模型'}）| device={device}"
          f" | θ_q={theta_q}（{theta_src}）"
          + (f" | 日线金字塔={daily_bars}根" if daily_bars else "")
          + (f" | 外盘日线={foreign_bars}根" if foreign_bars else "")
          + (" | 量仓特征" if use_vol_oi else "")
          + (f" | 细粒度{fine_period}m×{fine_bars}根" if fine_bars else "")
          + (f" | 跨品种日线={cross_kw['cross_daily_bars']}根" if cross_kw else ""))

    # ── 外盘新鲜度检查（fail-closed：stale/unavailable 品种禁止出信号）──
    foreign_blocked: dict[str, str] = {}
    if foreign_bars > 0:
        from obson.foreign_freshness import freshness_report
        for code_f, (status, reason) in freshness_report(args.symbols).items():
            if status == "fresh":
                print(f"  外盘[{code_f}] fresh：{reason}")
            else:
                foreign_blocked[code_f] = f"外盘{status}（{reason}）"
                print(f"  ⚠️ 外盘[{code_f}] {status}：{reason} → 该品种本次禁止出信号")

    for code in args.symbols:
        for period in args.periods:
            key = f"{code}_{period}m"
            fp = Path(args.data_dir) / f"{code}_{period}m.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp, parse_dates=["datetime"])
            fine_df = None
            if fine_bars > 0:
                fp_fine = Path(args.data_dir) / f"{code}_{fine_period}m.csv"
                if fp_fine.exists():
                    fine_df = pd.read_csv(fp_fine, parse_dates=["datetime"])
                else:
                    print(f"  [{code}] 警告: 缺细周期文件 {fp_fine}，该品种无细粒度上下文")
            seq_len = weekly_seq_len(df)
            n_train = int(len(df) * 0.7)
            c = _theta_c_dynamic(df.iloc[:n_train], seq_len, theta_q)
            train_ds, val_ds, test_ds = build_datasets(
                df, seq_len=seq_len, target_offset=1,
                train_ratio=0.7, val_ratio=0.15, split_mode="time",
                symbol_id=SYMBOLS[code], label_mode="day_close",
                label_threshold=c, theta_mode="dynamic",
                jump_break_pct=JUMP_BREAK_PCT[code],
                daily_bars=daily_bars,
                foreign_close=load_foreign(code) if foreign_bars > 0 else None,
                foreign_bars=foreign_bars,
                use_vol_oi=use_vol_oi,
                clean_path=clean_path,
                clean_frac=clean_frac,
                strategy_label=strategy_label,
                tp_frac=tp_frac,
                stop_frac=stop_frac,
                fine_df=fine_df, fine_bars=fine_bars,
                fine_period_min=fine_period, base_period_min=period,
                **cross_kw,
            )
            val_prob, val_true = run_probs(models, DataLoader(val_ds, batch_size=args.batch_size), device)
            thr = calibrate(val_prob, args.target_per_100)

            print("\n" + "=" * 72)
            print(f"[{key}] c={c:.3f} | 自适应阈值: 多={thr[2]:.3f} 空={thr[0]:.3f}"
                  f"（验证集每100根K线各喊 {args.target_per_100:.0f} 次）")

            # ── 1. 自检：测试集完整bar ──
            test_loader = DataLoader(test_ds, batch_size=args.batch_size)
            tp, tt = run_probs(models, test_loader, device)
            hp, cp, hn, cn = hit_table(tp, tt, thr)
            base_p, base_n = (tt == 2).mean(), (tt == 0).mean()
            print(f"  自检(测试集完整bar): 喊多 {hp:.1%}（基准{base_p:.1%}，每100根{cp * 100:.1f}次）"
                  f" | 喊空 {hn:.1%}（基准{base_n:.1%}，每100根{cn * 100:.1f}次）")

            # ── 2. 半成品bar衰减 ──
            print(f"  半成品bar衰减:")
            for f in [0.25, 0.5, 0.75, 1.0]:
                pf, _ = run_probs(models, DataLoader(test_ds, batch_size=args.batch_size), device,
                                  mutilate_last_bar=f if f < 1.0 else None)
                hp2, _, hn2, _ = hit_table(pf, tt, thr)
                tag = "完整bar" if f == 1.0 else f"走了{f:.0%}"
                print(f"    {tag:<8}: 喊多命中 {hp2:.1%} | 喊空命中 {hn2:.1%}")

            # ── 3. 实盘信号（模型输入末 bar = 最近一根已收盘 bar j*）──
            j_star, bar_dur = last_complete_bar(df, now_override)
            # 截断到 j*+1（+1 根供 target_offset 读取，推理不用），
            # 保证最后样本的输入窗口恰好止于 j*
            df_live = df.iloc[: j_star + 2]
            live_ds = KLineDataset(
                df_live, seq_len=seq_len, target_offset=1, normalize=False,
                symbol_id=SYMBOLS[code], jump_break_pct=JUMP_BREAK_PCT[code],
                daily_bars=daily_bars,
                foreign_close=load_foreign(code) if foreign_bars > 0 else None,
                foreign_bars=foreign_bars,
                use_vol_oi=use_vol_oi,
                clean_path=clean_path,
                clean_frac=clean_frac,
                strategy_label=strategy_label,
                tp_frac=tp_frac,
                stop_frac=stop_frac,
                fine_df=fine_df, fine_bars=fine_bars,
                fine_period_min=fine_period, base_period_min=period,
                **cross_kw,
            )
            need_idx = j_star - seq_len + 1
            if int(live_ds.valid_indices[-1]) != need_idx:
                # 断链保护可能剔除了末尾样本；实盘信号强制补回窗口末=j* 的样本
                # （该窗口不跨断点才可补，否则宁可报错也不给出错位信号）
                win_ts = df_live["datetime"].iloc[need_idx: j_star + 1]
                gap = win_ts.diff().dt.total_seconds().max() / 60.0
                if gap > 5 * 24 * 60:
                    raise ValueError(f"[{key}] 末尾样本窗口跨 >5 天断链，拒绝出信号")
                live_ds.valid_indices = np.append(live_ds.valid_indices, need_idx)
                live_ds.n_samples += 1
            item = live_ds[len(live_ds) - 1]
            # 断言：模型输入窗口末 bar 时间 == 待会儿打印的信号 bar 时间
            in_last = live_ds.timestamps.iloc[int(live_ds.valid_indices[-1]) + seq_len - 1]
            assert in_last == df["datetime"].iloc[j_star], \
                f"[{key}] 输入末bar {in_last} != 信号bar {df['datetime'].iloc[j_star]}"
            with torch.no_grad():
                ens = None
                for model in models:
                    out = model(
                        item["seq"].unsqueeze(0).to(device),
                        temporal_feat=item["temporal_feat"].unsqueeze(0).to(device),
                        time_pos=item["time_pos"].unsqueeze(0).to(device),
                        freq_feat=item["freq_feat"].unsqueeze(0).to(device),
                        symbol_id=item["symbol_id"].unsqueeze(0).to(device),
                        daily_ctx=(item["daily_ctx"].unsqueeze(0).to(device)
                                   if "daily_ctx" in item else None),
                        foreign_ctx=(item["foreign_ctx"].unsqueeze(0).to(device)
                                     if "foreign_ctx" in item else None),
                        cross_ctx=(item["cross_ctx"].unsqueeze(0).to(device)
                                   if "cross_ctx" in item else None),
                        cross_mask=(item["cross_mask"].unsqueeze(0).to(device)
                                    if "cross_mask" in item else None),
                        fine_ctx=(item["fine_ctx"].unsqueeze(0).to(device)
                                  if "fine_ctx" in item else None),
                    )
                    p = softmax_np(out["logits"].cpu().numpy())
                    ens = p if ens is None else ens + p
            prob = (ens / len(models))[0]
            # 估计 rem：以 j* 所在交易日为准（已走 bar 含 j*），全日长度取历史中位数
            day_ids = trading_day_ids(df["datetime"])
            day_of_j = day_ids[j_star]
            day_lens = np.unique(day_ids, return_counts=True)[1]
            full_est = int(np.median(day_lens))
            done = int((day_ids[: j_star + 1] == day_of_j).sum())
            rem_est = max(full_est - done, 1)
            closes = df["close"].to_numpy(float)
            log_ret = np.log(closes[1:] / np.maximum(closes[:-1], 1e-8))
            # σ_w 与训练标签同口径：窗口截止到 j*（dataset.py: sigma_w[j]=rolling.iloc[j-1]）
            sigma_w = float(
                pd.Series(log_ret)
                .rolling(seq_len - 1, min_periods=max((seq_len - 1) // 2, 10))
                .std()
                .iloc[j_star - 1]
            )
            theta = c * sigma_w * math.sqrt(rem_est)
            last = df.iloc[j_star]
            px = float(last["close"])
            tick = TICK.get(code, 1.0)
            up, dn = px * (1 + theta), px * (1 - theta)
            ts = last["datetime"]
            sig = "观望"
            if prob[2] >= thr[2]:
                sig = "🔺 喊多"
            elif prob[0] >= thr[0]:
                sig = "🔻 喊空"
            now_str = str(now_override or pd.Timestamp.now())
            incomplete = "（末根bar仍在走，已剔除）" if j_star < len(df) - 1 else ""
            print(f"  实盘信号 @ {ts}（当日已走{done}根/估{full_est}根，剩约{rem_est}根）{incomplete}")
            print(f"    数据截至 {df['datetime'].iloc[-1]} | 当前时刻 {now_str} | bar时长 {bar_dur:.0f}min")
            print(f"    {sig} | 信心 多{prob[2]:.1%}/无{prob[1]:.1%}/空{prob[0]:.1%}"
                  f" | 阈值 多{thr[2]:.2f}/空{thr[0]:.2f}")
            print(f"    现价 {px:.0f} | θ=±{theta:.2%} | 上轨 {up:.0f}（+{(up - px) / tick:.0f}跳）"
                  f" | 下轨 {dn:.0f}（−{(px - dn) / tick:.0f}跳）")
            # 手册 v2 判定（唯一权威规则，src/obson/playbook.py）
            from obson.playbook import decide
            act, reason = decide(code, period, prob, thr)
            if code in foreign_blocked and act != "观望":
                act, reason = "观望", f"{foreign_blocked[code]}，fail-closed 拦截（原判定被覆盖）"
            print(f"    手册判定: {act}（{reason}）")


if __name__ == "__main__":
    main()
