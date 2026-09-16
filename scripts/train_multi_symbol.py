"""多品种多频率混频训练 — RB/HC/I/SR/P × 30m，周窗口，预测当日收盘

数据前提（天勤 TQSDK）:
    TQ_USER=xxx TQ_PASS=xxx uv run python scripts/download_tqsdk_v2.py --symbols rb hc i sr p

用法:
    uv run python scripts/train_multi_symbol.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from obson.model import KLineConfig, KLineTransformer, build_datasets, weekly_seq_len
from obson.model.mixed_trainer import MixedFrequencyTrainer

SYMBOLS = {"rb": 0, "hc": 1, "i": 2, "sr": 3, "p": 4, "j": 5, "jm": 6,
           "m": 7, "y": 8, "cu": 9, "ag": 10, "TA": 11, "MA": 12}
# 各品种跳空断链阈值：铁矿真实隔夜跳空常>2%，放宽避免误杀；
# 白糖/棕榈油波动介于 rb 与 i 之间，暂定 2.5%；
# 焦炭/焦煤波动与铁矿接近，暂定 3.0%（j/jm/sr/p 待东财新数据到位后用 qc_new_data.py 的 P99 校准）
# 2026-09-06 天勤数据 P99/P99.9 校准（60m，8964根）：
#   m/y P99=1.02 → 0.02；j P99=1.02 → 0.03；jm P99=1.24 → 0.035；
#   cu P99=0.88 → 0.015；ag P99=1.76 → 0.03；TA P99=1.51 → 0.025；MA P99=1.37 → 0.025
JUMP_BREAK_PCT = {"rb": 0.02, "hc": 0.02, "i": 0.035, "sr": 0.025, "p": 0.025,
                  "j": 0.03, "jm": 0.035, "m": 0.02, "y": 0.02, "cu": 0.015,
                  "ag": 0.03, "TA": 0.025, "MA": 0.025}

# 品种 → 外盘定价锚（Wind EDB 日线，收盘-only，存 data/foreign/{name}_1d.csv）
FOREIGN_MAP = {"sr": "ice_sugar", "p": "bmd_palm", "y": "cbot_beanoil",
               "m": "cbot_soybean", "rb": "sgx_iron", "i": "sgx_iron",
               "ag": "comex_silver"}


def load_foreign(code: str):
    """读取该品种对应的外盘日线（date, close）；无映射或无文件返回 None"""
    name = FOREIGN_MAP.get(code)
    if name is None:
        return None
    fp = f"data/foreign/{name}_1d.csv"
    if not Path(fp).exists():
        return None
    return pd.read_csv(fp)


def _load_fine(code: str, args):
    """读取该品种的细周期K线（data/{code}_{fine_period}m.csv）；无文件返回 None 并警告"""
    fp = f"data/{code}_{args.fine_period}m.csv"
    if not Path(fp).exists():
        print(f"  [{code}] 警告: 缺细周期文件 {fp}，该品种无细粒度上下文（需补下 {args.fine_period}m 数据）")
        return None
    return pd.read_csv(fp, parse_dates=["datetime"])


def build_cross_daily(symbols=None):
    """跨品种日线原料：{symbol_id: (day_key_ord[F], ohlc[F,4])}

    从各品种 60m 分钟线聚合日K（开=首bar开，高/低=极值，收=末bar收），
    对齐 key = 该交易日最后一根 bar 的日历日期序数（夜盘归入次日，
    与 dataset 内 day_key_ord 同一定义）。symbols=None 用全品种。
    """
    from obson.model.dataset import trading_day_ids

    codes = symbols if symbols is not None else list(SYMBOLS)
    out = {}
    for code in codes:
        fp = Path(f"data/{code}_60m.csv")
        if not fp.exists():
            continue
        df = pd.read_csv(fp, parse_dates=["datetime"])
        day_ids = trading_day_ids(df["datetime"])
        keys, rows = [], []
        for d in np.unique(day_ids):
            idx = np.flatnonzero(day_ids == d)
            s, e = idx[0], idx[-1]
            keys.append(df["datetime"].iloc[e].date().toordinal())
            rows.append([
                df["open"].iloc[s], df["high"].iloc[s:e + 1].max(),
                df["low"].iloc[s:e + 1].min(), df["close"].iloc[e],
            ])
        out[SYMBOLS[code]] = (
            np.array(keys, dtype=np.int64),
            np.array(rows, dtype=np.float32),
        )
    return out
PERIODS = [15, 30, 60]
HORIZON_MINUTES = 60  # 30m→2
MIN_SAMPLES = 100


def _theta_from_quantile(df: pd.DataFrame, offset: int, q: float) -> float:
    """classify 阈值 θ：在训练段上，未来 offset 根内最大单方向偏移（%）的 q 分位数

    只用训练段数据（避免偷看验证/测试），max(向上最大偏移, 向下最大偏移)。
    """
    import numpy as np
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    n = len(df) - offset
    base = c[:n]
    up = np.full(n, -np.inf)
    dn = np.full(n, np.inf)
    for k in range(1, offset + 1):
        np.maximum(up, h[k:k + n], out=up)
        np.minimum(dn, l[k:k + n], out=dn)
    exc = np.maximum(up / base - 1.0, 1.0 - dn / base) * 100  # 百分比
    return float(np.quantile(exc, q))


def _day_theta_base(df: pd.DataFrame, q: float) -> float:
    """day_close 的基准 θ：训练段内每个完整交易日"从当日首根 bar close 到
    收盘锚的最大单方向偏移（%）"的 q 分位数。

    只用训练段数据；锚不在 13-15 点的截断日（换月/数据不全）剔除。
    """
    import numpy as np
    from obson.model.dataset import trading_day_ids

    day_ids = trading_day_ids(df["datetime"])
    ts = pd.to_datetime(df["datetime"])
    hours = (ts.dt.hour + ts.dt.minute / 60).to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    # 完整交易日的最小 bar 数按日长度中位数定（5m≈69, 60m≈7），写死会误杀 60m
    day_lens = np.unique(day_ids, return_counts=True)[1]
    min_day_bars = max(int(np.median(day_lens) * 0.5), 3)
    excs = []
    for d in np.unique(day_ids):
        rows = np.flatnonzero(day_ids == d)
        a = rows[-1]
        if not (13.0 <= hours[a] <= 15.0) or len(rows) < min_day_bars:
            continue
        base = close[rows[0]]
        if base <= 0:
            continue
        up = high[rows[1:]].max() / base - 1.0
        dn = 1.0 - low[rows[1:]].min() / base
        excs.append(max(up, dn) * 100.0)
    if not excs:
        raise ValueError("day_close θ 计算失败：训练段没有完整交易日")
    return float(np.quantile(excs, q))


def _theta_c_dynamic_ms(df: pd.DataFrame, seq_len: int,
                        min_remaining_minutes: int = 30) -> list:
    """dynamic θ 校准的原始 m 值列表（每个样本"摸到任一轨道所需 σ 倍数"）。
    拆出来供合约模式按段独立计算后再合并，避免跨合约段的虚假收益污染 σ_w。"""
    import math
    import numpy as np
    from obson.model.dataset import trading_day_ids

    day_ids = trading_day_ids(df["datetime"])
    ts = pd.to_datetime(df["datetime"])
    hours = (ts.dt.hour + ts.dt.minute / 60).to_numpy()
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    log_ret = np.log(close[1:] / np.maximum(close[:-1], 1e-8))
    sigma_w = (
        pd.Series(log_ret)
        .rolling(seq_len - 1, min_periods=max((seq_len - 1) // 2, 10))
        .std()
        .to_numpy()
    )
    sigma_w = np.concatenate([[np.nan], sigma_w])
    day_lens = np.unique(day_ids, return_counts=True)[1]
    min_day_bars = max(int(np.median(day_lens) * 0.5), 3)
    ms = []
    for d in np.unique(day_ids):
        rows = np.flatnonzero(day_ids == d)
        a = rows[-1]
        if not (13.0 <= hours[a] <= 15.0) or len(rows) < min_day_bars:
            continue
        deltas = np.diff(df["datetime"].iloc[rows].to_numpy()).astype("timedelta64[s]").astype(float) / 60.0
        step = float(np.median(deltas)) if len(deltas) > 0 else 5.0
        for j in rows[:-1]:
            rem = a - j
            if rem * step < min_remaining_minutes:
                continue
            base = close[j]
            sw = sigma_w[j]
            if base <= 0 or not np.isfinite(sw) or sw <= 0:
                continue
            up = high[j + 1 : a + 1].max() / base - 1.0
            dn = 1.0 - low[j + 1 : a + 1].min() / base
            ms.append(max(up, dn) / (sw * math.sqrt(rem)))
    return ms


def _theta_c_dynamic(df: pd.DataFrame, seq_len: int, q: float,
                     min_remaining_minutes: int = 30) -> float:
    """dynamic θ 的 σ 倍数 c：训练段上"摸到任一轨道所需 σ 倍数"的 q 分位数。

    对每个完整交易日的每根 bar j：所需倍数 m = max(向上偏移, 向下偏移)/(σ_w[j]·√剩余bar)，
    取 m 的 q 分位 → P(m ≥ c) = 1-q ≈ 训练段信号率（q=0.7 → 约30%发信号），
    信号密度仍由设计钉死，波动自适应只改"何时发"。
    """
    import numpy as np
    ms = _theta_c_dynamic_ms(df, seq_len, min_remaining_minutes)
    if not ms:
        raise ValueError("dynamic θ 校准失败：训练段没有可用样本")
    return float(np.quantile(ms, q))


def _label_counts(ds) -> "np.ndarray":
    """统计（可能是 ConcatDataset 的）训练集标签分布 [负, 无, 正]"""
    import numpy as np
    from torch.utils.data import ConcatDataset
    counts = np.zeros(3, dtype=np.int64)
    parts = ds.datasets if isinstance(ds, ConcatDataset) else [ds]
    for part in parts:
        if getattr(part, "labels", None) is not None:
            counts += np.bincount(part.labels, minlength=3)
    return counts


def _probe_loaders(train_loaders: dict, val_loaders: dict) -> dict:
    """probe 用：把 train/val 合并成 key_train / key_val 命名的 loader 字典。"""
    from torch.utils.data import DataLoader
    out = {}
    for k, ld in train_loaders.items():
        out[f"{k}_train"] = DataLoader(ld.dataset, batch_size=256)
    for k, ld in val_loaders.items():
        out[f"{k}_val"] = DataLoader(ld.dataset, batch_size=256)
    return out


def _verdict_probe(probes: dict) -> None:
    """协议 v3 §6.2 判决：pretrained vs random-init 的 label probe。
    5% 是告警线；硬暂停需双 seed 同向 + CI 不重叠（需第二次 seed 的结果）。"""
    pre = probes.get("pretrained", {}).get("label_ba")
    rnd = probes.get("random", {}).get("label_ba")
    champ = probes.get("supervised_champ", {}).get("label_ba")
    sym = probes.get("pretrained", {}).get("symbol_ba")
    path = probes.get("pretrained", {}).get("path_ba")
    print("\n" + "=" * 50)
    print("== E7-A probe 门禁判决 ==")
    if pre is None or rnd is None:
        print("  数据不足，无法判决")
        return
    diff = pre - rnd
    print(f"  label probe: pretrained={pre:.4f} vs random={rnd:.4f} → Δ={diff:+.4f}")
    if champ is not None:
        print(f"  监督冠军参考: {champ:.4f}")
        if champ - rnd < 0.02:
            print("  🚨 仪器警报：监督冠军 vs 随机 < 2%，标签信息可能根本不可线性读出")
            print("     → 本次 probe 判决无效，不得据此判 E7 表示失败")
    if sym is not None:
        print(f"  symbol probe: {sym:.4f}" +
              ("（偏高，留意品种身份挤占）" if sym > 0.9 and diff <= 0 else ""))
    if path is not None:
        print(f"  path probe: {path:.4f}")
    if diff <= 0:
        print("  ⚠️ 预训练表示 label probe 不优于随机初始化 → 告警（5% 线）")
        print("  按协议：需另一 seed 复跑确认同向 + CI 不重叠，才判表示失败")
    elif diff < 0.05:
        print("  🔶 略优于随机（<5%）：告警线内，建议跑第二 seed 确认噪声带")
    else:
        print("  ✅ label probe 明显优于随机初始化 → 门禁通过，可进入微调对照")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=list(SYMBOLS),
                    help="只用部分品种（冒烟测试用）")
    ap.add_argument("--periods", nargs="+", type=int, default=PERIODS, choices=PERIODS,
                    help="只用部分频率（冒烟测试用）")
    ap.add_argument("--split", default="time", choices=["time", "striped"],
                    help="time=时间硬切，验证集是纯未来（选模型用它，是测试集的无偏代理）；"
                         "striped=分块交错，验证集与训练同regime（只用于诊断学习能力）")
    ap.add_argument("--no-tech", action="store_true",
                    help="消融实验：关闭 technical 指标分支（ma_position/vol_regime/range_position/"
                         "rsi/macd/boll_squeeze/roc20 置零）")
    ap.add_argument("--task", default="regress", choices=["regress", "classify"],
                    help="regress=预测涨跌幅（原路径）；classify=三分类信号（正/无/负，first-passage 标注）")
    ap.add_argument("--label-anchor", default="day_close", choices=["horizon", "day_close", "next_close"],
                    help="classify 标签锚点：horizon=固定视野（--horizon-min 根内 first-passage）；"
                         "day_close=当日收盘（θ 随剩余时间√缩放，尾盘30min样本剔除）")
    ap.add_argument("--horizon-min", type=int, default=60,
                    help="预测视野分钟数（classify 建议 120；offset=horizon/period）")
    ap.add_argument("--theta-q", type=float, default=0.70,
                    help="classify 阈值 θ 的分位数（在训练段上对未来最大偏移取分位数，θ 越大信号越稀疏）"
                         "；day_close 下为全日偏移分位数，决定信号密度")
    ap.add_argument("--theta-mode", default="frozen", choices=["frozen", "dynamic"],
                    help="day_close 的 θ 口径：frozen=训练段冻结 θ_base 随√时间缩放；"
                         "dynamic=c × 窗口σ × √剩余bar（波动自适应，c 仅在训练段校准一次）")
    ap.add_argument("--lr", type=float, default=3e-3, help="OneCycle 峰值学习率")
    ap.add_argument("--lr-sched", default="onecycle", choices=["onecycle", "constant"],
                    help="学习率调度：onecycle=冠军配方默认；constant=固定 lr（P1 诊断用）")
    # E7-A 自监督预训练（协议 v3）
    ap.add_argument("--pretrain-e7", action="store_true",
                    help="只做 E7-A 双视图一致性预训练（训练段无标签窗口），不跑监督训练")
    ap.add_argument("--pretrain-e7b", action="store_true",
                    help="只做 E7-B 自回归预训练（预测下一根 bar，要求 causal 主干），不跑监督训练")
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--pretrain-lr", type=float, default=3e-4)
    ap.add_argument("--aug-alpha", type=float, default=0.10,
                    help="E7-aug0 窗口级噪声强度（累计噪声标准差占窗口波动比例，审计口径）")
    ap.add_argument("--guard-days", type=int, default=30,
                    help="负样本保护区：同品种锚定日距离 ≤ 此值的样本对不得互为负样本")
    ap.add_argument("--tau", type=float, default=0.1, help="NT-Xent 温度")
    ap.add_argument("--probe-e7", default=None, metavar="CKPT",
                    help="对指定 E7 预训练 ckpt 跑 probe 门禁（对照：随机初始化 + models/best.pt）")
    ap.add_argument("--init-ckpt", default=None, metavar="CKPT",
                    help="从指定 ckpt（如 E7 预训练 encoder）初始化主干再监督训练（微调对照）")
    ap.add_argument("--hidden", type=int, default=512, help="模型 hidden 维度")
    ap.add_argument("--layers", type=int, default=8, help="attention 层数")
    ap.add_argument("--heads", type=int, default=16, help="attention 头数")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--save-dir", default="checkpoints/multi_symbol",
                    help="checkpoint 保存目录（多轮实验用不同目录避免互相覆盖 best.pt）")
    ap.add_argument("--eval-ckpt", default=None, metavar="PATH",
                    help="只评估不训练：加载指定 checkpoint（如 checkpoints/multi_symbol/epoch_10.pt）"
                         "跑测试集评估后退出；需带与训练相同的 --task/--label-anchor/--theta-mode 参数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（多种子验证稳定性用）")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="分类模式的逐bar密集辅助监督权重（0=关闭；建议 0.3）")
    ap.add_argument("--soft-label", action="store_true",
                    help="day_close 软标签：按'实际走了θ的几成'构造软目标分布，"
                         "冲上轨92%%没摸到按0.8成多奖励而非全罚（改善概率校准）")
    ap.add_argument("--bidirectional", action="store_true",
                    help="窗口内双向 attention（分类任务无未来泄露，几何形状识别更完整；"
                         "勿与 --aux-weight 密集监督同开）")
    ap.add_argument("--rope", action="store_true",
                    help="时间感知 RoPE：旋转角度=真实流逝时间（隔夜跳空自动拉开距离），"
                         "叠加在 learned pos emb 之上（Qwen-VL 式双位置编码）")
    ap.add_argument("--readout", default="last", choices=["last", "mean", "cls"],
                    help="分类读出位置：last=最后位置（单向标配）；mean=全局平均；"
                         "cls=ViT 式 [CLS] token（建议配 --bidirectional）")
    ap.add_argument("--margin-weight", type=float, default=0.0,
                    help="方向间隔损失权重（0=关闭；建议 0.3）：赢家方向 logit 须领先输家 "
                         "--margin 以上，惩罚有方向证据样本上的骑墙输出")
    ap.add_argument("--margin", type=float, default=1.0, help="方向间隔门槛（logit 单位）")
    ap.add_argument("--aux-tail", type=int, default=16,
                    help="密集监督只覆盖窗口末尾 N 根 bar（前端上下文不足的不监督）")
    ap.add_argument("--daily-bars", type=int, default=0,
                    help="日线金字塔：每个样本附带 N 根完整日K前缀（0=关闭，建议 60）")
    ap.add_argument("--window-days", type=int, default=5,
                    help="分钟窗口覆盖的交易日数（默认5=一周；20≈一个月，配合长日K前缀使用）")
    ap.add_argument("--foreign-bars", type=int, default=0,
                    help="外盘日线token数（0=关闭，建议20；仅 sr/p/y/m/rb/i/ag 有外盘锚）")
    ap.add_argument("--close-path", action="store_true",
                    help="价格通道全列锚定窗口前3根收盘均值（携带趋势路径；默认关=原版相对自身收盘）")
    ap.add_argument("--use-vol-oi", action="store_true",
                    help="加入成交量/持仓量特征分支（需 CSV 含 volume/open_oi/close_oi 列，缺失自动补零降级）")
    ap.add_argument("--clean-path", action="store_true",
                    help="干净路径标签：方向样本若摸轨前最大逆向偏移 > clean-frac×θ，改判'按兵不动'（仅 day_close）")
    ap.add_argument("--clean-frac", type=float, default=0.5,
                    help="干净路径的逆向容差（θ 的倍数），默认0.5")
    ap.add_argument("--strategy-label", action="store_true",
                    help="策略对齐标签：多=先摸+tp-frac×θ且未先破-stop-frac×θ，空镜像，其余=无（与 clean-path 互斥）")
    ap.add_argument("--tp-frac", type=float, default=0.8, help="策略标签止盈轨（θ 的倍数），默认0.8")
    ap.add_argument("--stop-frac", type=float, default=0.5, help="策略标签止损轨（θ 的倍数），默认0.5")
    ap.add_argument("--fine-bars", type=int, default=0,
                    help="细粒度上下文：最近 N 根已收盘细周期bar作为token（0=关闭，建议 24~48）")
    ap.add_argument("--fine-period", type=int, default=5, choices=[5, 15],
                    help="细粒度上下文的周期（分钟），默认5")
    ap.add_argument("--contract-mode", action="store_true",
                    help="合约模式：按合约段拼接训练（data/contracts/），零跨合约污染；"
                         "仅支持 --task classify --label-anchor day_close")
    ap.add_argument("--hazard-task", action="store_true",
                    help="teacher-level competing-risk hazard 辅助监督（生产分类头仍为主任务）")
    ap.add_argument("--hazard-event-weight", type=float, default=1.0,
                    help="hazard 首触事件的 NLL 权重，默认1.0")
    ap.add_argument("--hazard-loss-weight", type=float, default=0.10,
                    help="hazard 辅助损失相对分类损失的权重，默认0.10")
    ap.add_argument("--serial-path", action="store_true",
                    help="串联路径任务：固定[1,2,4,8]根bar路径表示接入交易分类")
    ap.add_argument("--serial-path-loss-weight", type=float, default=0.10)
    ap.add_argument("--serial-path-fusion-weight", type=float, default=0.10)
    ap.add_argument("--klm-task", action="store_true",
                    help="KLM：双向K线记忆 + future query 分位数路径回归 + trade query")
    ap.add_argument("--klm-reg-loss-weight", type=float, default=0.10,
                    help="KLM 路径分位数回归损失权重，默认0.10")
    ap.add_argument("--path-aux", action="store_true",
                    help="E3 路径状态辅助任务：4节点×3态辅助头（教师模型方案）")
    ap.add_argument("--path-aux-weight", type=float, default=0.1,
                    help="路径状态辅助损失权重，默认 0.1")
    ap.add_argument("--exc-aux", action="store_true",
                    help="E4 excursion 分桶辅助任务：m_dn/m_up 最大偏移占 θ 比例各 6 桶")
    ap.add_argument("--exc-aux-weight", type=float, default=0.1,
                    help="excursion 辅助损失权重，默认 0.1")
    ap.add_argument("--query-decoder", action="store_true",
                    help="E6'：未来时间 query decoder 替代简易路径头（encoder 保持单向）")
    ap.add_argument("--teacher", action="store_true",
                    help="Teacher 训练线：E3 主任务/路径头 + 生产效用辅助头与复合选模")
    ap.add_argument("--teacher-loss-weight", type=float, default=0.15)
    ap.add_argument("--teacher-logit-weight", type=float, default=0.10)
    ap.add_argument("--teacher-selection-weight", type=float, default=0.10)
    ap.add_argument("--hierarchical-task", action="store_true",
                    help="H1 层级任务：先预测是否值得交易，再预测条件方向")
    ap.add_argument("--gate-threshold", type=float, default=0.80,
                    help="机会门控标签的最小生产效用（theta倍数），默认0.80（完整止盈级别）")
    ap.add_argument("--gate-pos-weight", type=float, default=0.0,
                    help="机会门控正例 BCE 权重，0=按训练集正例率自动计算")
    ap.add_argument("--hier-stage", choices=["joint", "gate", "direction", "direction_ft"], default="joint",
                    help="层级训练阶段：gate只训机会塔，direction只训方向塔，direction_ft解冻末层微调，joint联合")
    ap.add_argument("--direction-label-mode", choices=["utility", "close_return"], default="utility",
                    help="方向标签：utility=生产效用最优方向；close_return=gate样本按收盘收益方向")
    ap.add_argument("--hier-unfreeze-layers", type=int, default=2,
                    help="direction_ft 解冻 Transformer 最后几层，默认2")
    ap.add_argument("--hier-direction-active-only", action="store_true",
                    help="方向阶段训练只采样 gate=1 样本；用于可学习性诊断，不改变验证/生产口径")
    ap.add_argument("--hier-direction-all", action="store_true",
                    help="方向头对全样本训练；gate=0 行按 --hier-direction-none-weight 弱监督")
    ap.add_argument("--hier-direction-none-weight", type=float, default=0.2,
                    help="全样本方向训练时 gate=0 行的损失权重，默认0.2")
    ap.add_argument("--dual-tower-task", action="store_true",
                    help="双塔+结果回归：机会塔/方向塔/效用结果塔（回归不参与交易输出）")
    ap.add_argument("--outcome-loss-weight", type=float, default=0.10)
    args = ap.parse_args()
    if args.teacher:
        if args.task != "classify" or args.label_anchor != "day_close" or args.theta_mode != "dynamic":
            raise SystemExit(
                "--teacher 只支持生产语义：--task classify --label-anchor day_close "
                "--theta-mode dynamic"
            )
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.task == "classify" and args.label_anchor == "horizon":
        assert args.horizon_min % min(args.periods) == 0, "horizon-min 需能被最小频率整除"
    if args.contract_mode:
        assert args.task == "classify" and args.label_anchor in ("day_close", "next_close"), \
            "--contract-mode v1 仅支持 --task classify --label-anchor day_close/next_close"
        assert args.theta_mode == "dynamic", "--contract-mode v1 仅支持 --theta-mode dynamic"
    if args.label_anchor in ("day_close", "next_close") and args.task != "classify":
        print(f"  提示: --label-anchor {args.label_anchor} 只在 --task classify 下生效，忽略")

    # T+1 锚（next_close）：day_close 语义族 + anchor_days_ahead=1，泄露双闸在 dataset.py
    _anchor_ahead = 1 if args.label_anchor == "next_close" else 0

    batch_size = args.batch_size
    hidden_size = args.hidden
    num_layers = args.layers
    num_heads = args.heads
    lr = args.lr
    dropout = args.dropout
    weight_decay = 0.1
    max_epochs = args.epochs
    patience = args.patience
    symbols = {k: SYMBOLS[k] for k in args.symbols}
    periods = args.periods

    print("=" * 64)
    if args.task == "classify" and args.label_anchor == "day_close":
        anchor_desc = "当日收盘"
    elif args.task == "classify" and args.label_anchor == "next_close":
        anchor_desc = "次日收盘(T+1)"
    else:
        anchor_desc = f"未来{args.horizon_min}分钟"
    print(f"多品种多频率训练 | 品种={args.symbols} × {args.periods}m | 周窗口 | 预测{anchor_desc} | {args.split}切分")
    print("=" * 64)

    train_loaders, val_loaders, test_loaders = {}, {}, {}
    train_dss: dict = {}
    total_counts = None
    gate_pos_weight = args.gate_pos_weight
    if args.contract_mode:
        from obson.contract_series import build_contract_frame, load_calendar
        from obson.model.dataset import build_datasets_contract
    for code, sym_id in symbols.items():
        for period in periods:
            key = f"{code}_{period}m"
            if args.contract_mode:
                # 合约模式：段帧拼接，窗口/标签零跨合约污染
                try:
                    cframe, cbreaks, cmask, cday_ids = build_contract_frame(code, period)
                except (FileNotFoundError, ValueError) as e:
                    print(f"  [{key}] 合约段帧构建失败: {e}，跳过")
                    continue
                # 窗口长度：用当前主力合约（最后一段）的单段长度估一周 bar 数
                last_seg = cframe[cframe["seg"] == cframe["seg"].max()].reset_index(drop=True)
                seq_len = weekly_seq_len(last_seg, trading_days=args.window_days)
                # θ 常数 c：只用"主力任期 bar 且落在训练段日期（前70%交易日）"的行估，
                # 避免远期月噪声污染波动率分位数。
                # 必须按合约段独立计算再合并（教师模型指出）：拼接帧段交界处的
                # 虚假跳价收益会污染 σ_w 滚动窗口，进而带偏 c
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
                theta = float(np.quantile(ms_all, args.theta_q))
                extra_kwargs = {"label_mode": "day_close", "label_threshold": theta,
                                "theta_mode": "dynamic", "soft_label": args.soft_label,
                                "anchor_days_ahead": _anchor_ahead,
                                "hierarchical_task": args.hierarchical_task,
                                "gate_threshold": args.gate_threshold,
                                "direction_label_mode": args.direction_label_mode}
                try:
                    train_ds, val_ds, test_ds = build_datasets_contract(
                        code, period, seq_len=seq_len,
                        train_ratio=0.7, val_ratio=0.15,
                        target_offset=1, symbol_id=sym_id,
                        daily_bars=args.daily_bars,
                        foreign_close=load_foreign(code) if args.foreign_bars > 0 else None,
                        foreign_bars=args.foreign_bars, **extra_kwargs,
                        use_vol_oi=args.use_vol_oi,
                        clean_path=args.clean_path,
                        clean_frac=args.clean_frac,
                        strategy_label=args.strategy_label,
                        tp_frac=args.tp_frac,
                        stop_frac=args.stop_frac,
                        base_period_min=period,
                    )
                except ValueError as e:
                    print(f"  [{key}] 构造失败: {e}，跳过")
                    continue
            else:
                try:
                    df = pd.read_csv(f"data/{code}_{period}m.csv", parse_dates=["datetime"])
                except FileNotFoundError:
                    print(f"  [{key}] 文件不存在，跳过")
                    continue
                seq_len = weekly_seq_len(df, trading_days=args.window_days)
                extra_kwargs = {}
                theta = None
                if args.task == "classify":
                    n_train = int(len(df) * 0.7)
                    if args.label_anchor in ("day_close", "next_close"):
                        offset = 1  # day_close/next_close 无固定视野，offset 仅用于窗口切片
                        if args.theta_mode == "dynamic":
                            theta = _theta_c_dynamic(df.iloc[:n_train], seq_len, args.theta_q)
                            extra_kwargs = {"label_mode": "day_close", "label_threshold": theta,
                                            "theta_mode": "dynamic", "soft_label": args.soft_label,
                                            "anchor_days_ahead": _anchor_ahead,
                                            "hierarchical_task": args.hierarchical_task,
                                            "gate_threshold": args.gate_threshold,
                                            "direction_label_mode": args.direction_label_mode}
                        else:
                            theta = _day_theta_base(df.iloc[:n_train], args.theta_q)
                            extra_kwargs = {"label_mode": "day_close", "label_threshold": theta,
                                            "soft_label": args.soft_label,
                                            "anchor_days_ahead": _anchor_ahead,
                                            "hierarchical_task": args.hierarchical_task,
                                            "gate_threshold": args.gate_threshold,
                                            "direction_label_mode": args.direction_label_mode}
                    else:
                        offset = args.horizon_min // period
                        theta = _theta_from_quantile(df.iloc[:n_train], offset, args.theta_q)
                        extra_kwargs = {"label_mode": "classify", "label_threshold": theta}
                else:
                    offset = args.horizon_min // period
                try:
                    train_ds, val_ds, test_ds = build_datasets(
                        df, seq_len=seq_len, target_offset=offset,
                        train_ratio=0.7, val_ratio=0.15,
                        split_mode=args.split, symbol_id=sym_id,
                        daily_bars=args.daily_bars,
                        foreign_close=load_foreign(code) if args.foreign_bars > 0 else None,
                        foreign_bars=args.foreign_bars, **extra_kwargs,
                        jump_break_pct=JUMP_BREAK_PCT[code],
                        use_vol_oi=args.use_vol_oi,
                        clean_path=args.clean_path,
                        clean_frac=args.clean_frac,
                        strategy_label=args.strategy_label,
                        tp_frac=args.tp_frac,
                        stop_frac=args.stop_frac,
                        fine_df=_load_fine(code, args) if args.fine_bars > 0 else None,
                        fine_bars=args.fine_bars,
                        fine_period_min=args.fine_period,
                        base_period_min=period,
                    )
                except ValueError as e:
                    print(f"  [{key}] 构造失败: {e}，跳过")
                    continue
            if min(len(train_ds), len(val_ds)) < MIN_SAMPLES:
                print(f"  [{key}] 样本不足，跳过")
                continue
            g = torch.Generator().manual_seed(args.seed)
            train_loaders[key] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
            train_dss[key] = train_ds
            val_loaders[key] = DataLoader(val_ds, batch_size=batch_size)
            test_loaders[key] = DataLoader(test_ds, batch_size=batch_size)
            msg = f"  [{key}] 窗口={seq_len} | 训练: {len(train_ds)} | 验证: {len(val_ds)}"
            if args.task == "classify":
                counts = _label_counts(train_ds)
                total_counts = counts if total_counts is None else total_counts + counts
                frac = counts / counts.sum()
                theta_desc = f"c={theta:.3f}" if args.theta_mode == "dynamic" else f"θ={theta:.3f}%"
                msg += f" | {theta_desc} | 标签分布 负/无/正={frac[0]:.1%}/{frac[1]:.1%}/{frac[2]:.1%}"
                if args.hierarchical_task:
                    g = getattr(train_ds, "gate_targets", None)
                    d = getattr(train_ds, "direction_targets", None)
                    if g is not None and d is not None:
                        active = g > 0.5
                        msg += (
                            f" | hier gate={active.mean():.1%}"
                            f" active空/多={(d[active] == 0).mean() if active.any() else float('nan'):.1%}/"
                            f"{(d[active] == 1).mean() if active.any() else float('nan'):.1%}"
                        )
                        # The gate is derived from the same production utility
                        # convention as __getitem__; verify one sampled row.
                        if len(train_ds) > 0:
                            sample = train_ds[0]
                            if int(sample["direction_target"]) != int(d[0]) or float(sample["gate_target"]) != float(g[0]):
                                raise ValueError(f"{key}: hierarchical array/item label mismatch")
            print(msg)

    if not train_loaders:
        print("没有可用数据，请先运行 download_tqsdk.py")
        return

    if args.hierarchical_task and gate_pos_weight <= 0:
        gate_pos = sum(float(getattr(ds, "gate_targets", np.array([])).sum())
                       for ds in train_dss.values())
        gate_total = sum(float(len(getattr(ds, "gate_targets", [])))
                         for ds in train_dss.values())
        gate_neg = gate_total - gate_pos
        gate_pos_weight = gate_neg / max(gate_pos, 1.0)
        print(f"  [hier] gate训练标签正例率={gate_pos / max(gate_total, 1.0):.2%} "
              f"pos_weight={gate_pos_weight:.3f}")

    # E3 v1.1：逐节点类别权重 = 训练集状态频率求逆（均值归一到 1），
    # 治 90%+ "未触轨"多数类把 masked CE 淹没导致的辅助头塌缩
    path_state_weights = None
    if args.path_aux:
        import numpy as _np
        cnt = _np.zeros((4, 3), dtype=_np.float64)
        n_tot = 0
        for ld in train_loaders.values():
            ps = getattr(ld.dataset, "path_states", None)
            if ps is None:
                continue
            for k in range(4):
                col = ps[:, k]
                col = col[col != -1]
                for c in (0, 1, 2):
                    cnt[k, c] += (col == c).sum()
            n_tot += len(ps)
        if n_tot > 0:
            w = cnt.sum(axis=1, keepdims=True) / (3.0 * _np.clip(cnt, 1.0, None))
            w = w / w.mean()
            path_state_weights = w.flatten().tolist()
            print(f"  [E3] 路径状态权重(4节点×3态): "
                  + " ".join(f"[{w[k,0]:.2f}/{w[k,1]:.2f}/{w[k,2]:.2f}]" for k in range(4))
                  + f" | 节点标签分布(均) 未/上/下={cnt.mean(axis=0)/cnt.mean(axis=0).sum()}")

    # E4：逐侧类别权重 = 训练集桶频率求逆（均值归一到 1），与 v1.1 同款防塌缩
    exc_weights = None
    if args.exc_aux:
        import numpy as _np
        from obson.model.dataset import EXC_N_BINS
        ecnt = _np.zeros((2, EXC_N_BINS), dtype=_np.float64)
        e_tot = 0
        for ld in train_loaders.values():
            el = getattr(ld.dataset, "exc_labels", None)
            if el is None:
                continue
            for side in range(2):
                for c in range(EXC_N_BINS):
                    ecnt[side, c] += (el[:, side] == c).sum()
            e_tot += len(el)
        if e_tot > 0:
            ew = ecnt.sum(axis=1, keepdims=True) / (EXC_N_BINS * _np.clip(ecnt, 1.0, None))
            ew = ew / ew.mean()
            exc_weights = ew.flatten().tolist()
            dist = ecnt / ecnt.sum(axis=1, keepdims=True)
            print(f"  [E4] excursion 权重 dn={np.round(ew[0],2).tolist()} up={np.round(ew[1],2).tolist()}")
            print(f"  [E4] 训练集桶分布 dn={np.round(dist[0],3).tolist()}")
            print(f"  [E4]            up={np.round(dist[1],3).tolist()}")

    model_config = KLineConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        loss_type="huber",
        dropout=dropout,
        num_symbols=len(SYMBOLS),  # embedding 表固定按全品种建，保证符号id稳定
        use_technical=not args.no_tech,
        task=args.task,
        class_weights=(
            (total_counts.sum() / (3.0 * total_counts.clip(min=1))).tolist()
            if args.task == "classify" and total_counts is not None else None
        ),
        dense_loss_weight=args.aux_weight,  # 分类模式的逐bar密集辅助监督（语音式逐帧监督）
        dense_tail=args.aux_tail,
        daily_bars=args.daily_bars,  # 日线金字塔前缀
        foreign_bars=args.foreign_bars,  # 外盘日线前缀
        close_path=args.close_path,      # 价格通道路径锚定
        bidirectional=args.bidirectional,  # 窗口内双向 attention
        rope=args.rope,                    # 时间感知 RoPE
        readout=args.readout,              # 分类读出位置
        margin_weight=args.margin_weight,  # 方向间隔损失
        margin=args.margin,
        use_vol_oi=args.use_vol_oi,        # 成交量/持仓量特征分支
        clean_path=args.clean_path,        # 干净路径标签
        clean_frac=args.clean_frac,
        strategy_label=args.strategy_label,  # 策略对齐标签
        tp_frac=args.tp_frac,
        stop_frac=args.stop_frac,
        fine_bars=args.fine_bars,          # 细粒度上下文 token 数
        theta_q=args.theta_q,              # 标签宽度口径入 ckpt（消费端从这里恢复）
        soft_label=args.soft_label,
        contract_mode=args.contract_mode,  # 合约模式标记入 ckpt（回测/实盘据此选数据口径）
        path_aux=args.path_aux,            # E3 路径状态辅助头
        path_aux_weight=args.path_aux_weight,
        path_state_weights=path_state_weights,  # E3 v1.1 逐节点类别权重（None=不加权）
        exc_aux=args.exc_aux,              # E4 excursion 分桶辅助头
        exc_aux_weight=args.exc_aux_weight,
        exc_weights=exc_weights,           # E4 逐侧类别权重（None=不加权）
        query_decoder=args.query_decoder,  # E6' 未来时间 query decoder
        utility_head=args.teacher,
        utility_loss_weight=args.teacher_loss_weight,
        utility_logit_weight=args.teacher_logit_weight if args.teacher else 0.0,
        utility_selection_weight=args.teacher_selection_weight if args.teacher else 0.0,
        hazard_task=args.hazard_task,
        hazard_bins=4,
        hazard_event_weight=args.hazard_event_weight,
        hazard_loss_weight=args.hazard_loss_weight,
        serial_path=args.serial_path,
        serial_path_loss_weight=args.serial_path_loss_weight,
        serial_path_fusion_weight=args.serial_path_fusion_weight,
        klm_task=args.klm_task,
        klm_reg_loss_weight=args.klm_reg_loss_weight,
        hierarchical_task=args.hierarchical_task,
        gate_threshold=args.gate_threshold,
        gate_pos_weight=gate_pos_weight,
        hier_stage=args.hier_stage,
        hier_unfreeze_layers=args.hier_unfreeze_layers,
        direction_all=args.hier_direction_all,
        direction_none_weight=args.hier_direction_none_weight,
        dual_tower_task=args.dual_tower_task,
        outcome_loss_weight=args.outcome_loss_weight,
    )
    model = KLineTransformer(model_config)
    if args.init_ckpt:
        # E7 微调对照：从预训练 encoder 初始化主干（头部权重即使加载也是预训练时的随机值，
        # 等价于全新头部；strict=False 容忍未来头部结构差异）
        _ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        _miss, _unexp = model.load_state_dict(_ck["model"], strict=False)
        print(f"[init-ckpt] 从 {args.init_ckpt} 初始化：missing={len(_miss)} unexpected={len(_unexp)}")
    if args.hierarchical_task and args.hier_stage in ("direction", "direction_ft"):
        frozen = 0
        n_layers = max(0, min(args.hier_unfreeze_layers, len(model.layers)))
        train_layer_names = {f"layers.{i}." for i in range(len(model.layers) - n_layers, len(model.layers))}
        for name, param in model.named_parameters():
            layer_train = any(name.startswith(prefix) for prefix in train_layer_names)
            if name.startswith("direction_tower") or name.startswith("direction_head") or (
                args.hier_stage == "direction_ft" and layer_train
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen += param.numel()
        suffix = " + 最后%d层encoder" % n_layers if args.hier_stage == "direction_ft" else ""
        print(f"[hier] direction阶段：冻结参数={frozen:,}，训练 direction_tower/direction_head{suffix}")
        if args.hier_direction_active_only:
            active_loaders = {}
            for key, loader in train_loaders.items():
                ds = loader.dataset
                gate = getattr(ds, "gate_targets", None)
                if gate is None:
                    raise ValueError(f"{key}: active-only 采样要求数据集提供 gate_targets")
                indices = np.flatnonzero(np.asarray(gate) > 0.5).tolist()
                if not indices:
                    raise ValueError(f"{key}: 没有 gate-positive 方向样本")
                gen = torch.Generator().manual_seed(args.seed)
                active_loaders[key] = DataLoader(
                    Subset(ds, indices), batch_size=batch_size, shuffle=True, generator=gen
                )
                print(f"[hier] active-only {key}: {len(ds)} -> {len(indices)} samples "
                      f"({len(indices) / max(len(ds), 1):.1%})")
            train_loaders = active_loaders
    elif args.hierarchical_task and args.hier_stage == "gate":
        for name, param in model.named_parameters():
            if name.startswith("direction_tower") or name.startswith("direction_head"):
                param.requires_grad = False
        print("[hier] gate阶段：只训练 gate loss，方向参数冻结")
    if args.task == "classify":
        if args.label_anchor == "day_close":
            horizon_desc = f"当日收盘 (θ={args.theta_mode})"
        elif args.label_anchor == "next_close":
            horizon_desc = f"次日收盘T+1 (θ={args.theta_mode})"
        else:
            horizon_desc = f"{args.horizon_min}min"
        print(f"\n任务=classify | 锚点={horizon_desc} | θ分位数={args.theta_q} | "
              f"类别权重={model_config.class_weights}")
    print(f"模型参数量: {model.count_parameters():,} | 品种数: {len(SYMBOLS)} | 组合数: {len(train_loaders)}")

    if args.probe_e7:
        # E7-A probe 门禁（协议 v3 §6.2）：只用 train/val，测试段不参与。
        from obson.model.probe import collect_reprs, run_probes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        probes = {}
        # 1) 预训练 encoder
        ckpt = torch.load(args.probe_e7, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        probes["pretrained"] = run_probes(collect_reprs(model, _probe_loaders(train_loaders, val_loaders), device), "pretrained", seed=args.seed)
        # 2) 随机初始化对照（同架构同预算，估计噪声/下界）
        import copy as _copy
        torch.manual_seed(args.seed + 1000)
        model_rand = KLineTransformer(model_config)
        probes["random"] = run_probes(collect_reprs(model_rand, _probe_loaders(train_loaders, val_loaders), device), "random-init", seed=args.seed)
        del model_rand
        # 3) 冠军监督上界（若本地有 models/best.pt）
        from pathlib import Path as _P
        champ = _P("models/best.pt")
        if champ.exists():
            c2 = torch.load(champ, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(c2["model"], strict=False)
            if missing or unexpected:
                print(f"  ⚠️ 冠军 ckpt 与当前配置不完全同构（probe 只用主干表示，"
                      f"头部差异不影响）：missing={len(missing)} unexpected={len(unexpected)}")
            probes["supervised_champ"] = run_probes(collect_reprs(model, _probe_loaders(train_loaders, val_loaders), device), "supervised-champ", seed=args.seed)
        _verdict_probe(probes)
        return

    if args.pretrain_e7 or args.pretrain_e7b:
        # E7 预训练：只用 train_dss（训练段），val/test 不进预训练；保存后由 probe 门禁决定能否微调。
        # E7-A = 双视图一致性（已判失败，保留作对照）；E7-B = 自回归预测下一根 bar。
        from torch.utils.data import ConcatDataset, DataLoader as _DL
        from obson.model.pretrain import (
            PretrainViewDataset, PretrainTrainer, ARPretrainTrainer, SeqLenBatchSampler)

        wrapped = []
        for key, ds in train_dss.items():
            ds.return_raw = True  # 增强/AR目标需要原始价量（raw_data 现已始终保存）
            code = key.rsplit("_", 1)[0]
            sym_idx = list(SYMBOLS.keys()).index(code) if code in SYMBOLS else 0
            anchor_bar = ds.valid_indices + ds.seq_len - 1
            anchor_days = ds.day_ids_global[anchor_bar]
            wrapped.append(PretrainViewDataset(ds, sym_idx, anchor_days))
            print(f"  [E7预训练] {key}: {len(ds)} 窗口（仅训练段）")
        concat = ConcatDataset(wrapped)
        seq_lens = [w.feat_sig for w in wrapped for _ in range(len(w))]
        sampler = SeqLenBatchSampler(seq_lens, args.batch_size, seed=args.seed)
        loader = _DL(concat, batch_sampler=sampler)
        if args.pretrain_e7b:
            ptrainer = ARPretrainTrainer(
                model, loader, lr=args.pretrain_lr, max_epochs=args.pretrain_epochs,
                save_dir=args.save_dir)
        else:
            ptrainer = PretrainTrainer(
                model, loader, lr=args.pretrain_lr, max_epochs=args.pretrain_epochs,
                tau=args.tau, alpha=args.aug_alpha, guard_days=args.guard_days,
                save_dir=args.save_dir)
        ptrainer.fit()
        return

    if args.eval_ckpt:
        # 只评估不训练：复现训练时的数据集构造，加载指定 checkpoint 跑测试集
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.eval_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        _evaluate_test(args, model, test_loaders, device, ckpt_name=str(args.eval_ckpt).split("/")[-1])
        return

    trainer = MixedFrequencyTrainer(
        model=model,
        loaders=train_loaders,
        val_loaders=val_loaders,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        lr_sched=args.lr_sched,
    )
    # 完整数据/标签配置清单（老师审查二轮 #5）：ckpt 必须能自解释训练口径，
    # 不靠命令行记忆复现
    trainer.data_manifest = {
        "task": args.task,
        "label_anchor": args.label_anchor,
        "theta_mode": args.theta_mode,
        "theta_q": args.theta_q,
        "soft_label": args.soft_label,
        "clean_path": args.clean_path,
        "clean_frac": args.clean_frac,
        "strategy_label": args.strategy_label,
        "tp_frac": args.tp_frac,
        "stop_frac": args.stop_frac,
        "daily_bars": args.daily_bars,
        "foreign_bars": args.foreign_bars,
        "close_path": args.close_path,
        "use_vol_oi": args.use_vol_oi,
        "fine_bars": args.fine_bars,
        "fine_period": args.fine_period,
        "window_days": args.window_days,
        "min_remaining_minutes": 30,  # build_datasets 默认值（非命令行参数）
        "jump_break_pct": JUMP_BREAK_PCT,  # 按品种的字典，非单一命令行参数
        "symbols": args.symbols,
        "periods": args.periods,
        "seed": args.seed,
        "hazard_task": args.hazard_task,
        "hazard_bins": 4,
        "hazard_event_weight": args.hazard_event_weight,
        "hazard_loss_weight": args.hazard_loss_weight,
        "serial_path": args.serial_path,
        "serial_path_horizons": [1, 2, 4, 8],
        "serial_path_loss_weight": args.serial_path_loss_weight,
        "serial_path_fusion_weight": args.serial_path_fusion_weight,
        "klm_task": args.klm_task,
        "klm_horizons": [1, 2, 4, -1],
        "klm_reg_loss_weight": args.klm_reg_loss_weight,
        "hierarchical_task": args.hierarchical_task,
        "gate_threshold": args.gate_threshold,
        "direction_label_mode": args.direction_label_mode,
        "gate_pos_weight": gate_pos_weight,
        "hier_stage": args.hier_stage,
        "hier_unfreeze_layers": args.hier_unfreeze_layers,
        "dual_tower_task": args.dual_tower_task,
        "outcome_loss_weight": args.outcome_loss_weight,
        "teacher": args.teacher,
        "teacher_loss_weight": args.teacher_loss_weight,
        "teacher_logit_weight": args.teacher_logit_weight if args.teacher else 0.0,
        "teacher_selection_weight": args.teacher_selection_weight if args.teacher else 0.0,
    }
    trainer.fit()

    # ═══ 纯未来测试集评估（最后15%时间，模型从未见过）═══
    ckpt = torch.load(trainer.save_dir / "best.pt", map_location=trainer.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _evaluate_test(args, model, test_loaders, trainer.device)


def _evaluate_test(args, model, test_loaders, device, ckpt_name: str = "best.pt") -> None:
    """纯未来测试集评估（最后15%时间，模型从未见过）；classify 打印分时段细节"""
    print("\n" + "=" * 64)
    print(f"测试集评估（加载 {ckpt_name}）")
    print("=" * 64)
    import numpy as np
    model.eval()
    for key, loader in test_loaders.items():
        preds, trues, fwd, tods, logits, touch = [], [], [], [], [], []
        gate_scores, gate_trues, direction_preds = [], [], []
        has_touch = False
        with torch.no_grad():
            for batch in loader:
                x = batch["seq"].to(device)
                if args.task == "classify":
                    daily_ctx = batch.get("daily_ctx")
                    if daily_ctx is not None:
                        daily_ctx = daily_ctx.to(device)
                    foreign_ctx = batch.get("foreign_ctx")
                    if foreign_ctx is not None:
                        foreign_ctx = foreign_ctx.to(device)
                    cross_ctx = batch.get("cross_ctx")
                    cross_mask = batch.get("cross_mask")
                    if cross_ctx is not None:
                        cross_ctx = cross_ctx.to(device)
                        cross_mask = cross_mask.to(device)
                    fine_ctx = batch.get("fine_ctx")
                    if fine_ctx is not None:
                        fine_ctx = fine_ctx.to(device)
                    out = model(
                        x,
                        temporal_feat=batch["temporal_feat"].to(device),
                        time_pos=batch["time_pos"].to(device),
                        freq_feat=batch["freq_feat"].to(device),
                        symbol_id=batch["symbol_id"].to(device),
                        daily_ctx=daily_ctx,
                        foreign_ctx=foreign_ctx,
                        cross_ctx=cross_ctx,
                        cross_mask=cross_mask,
                        fine_ctx=fine_ctx,
                    )
                    preds.append(out["pred_class"].cpu().numpy())
                    logits.append(out["logits"].cpu().numpy())
                    if args.hierarchical_task:
                        gate_scores.append(out["gate_logit"].cpu().numpy())
                        direction_preds.append(out["direction_logits"].argmax(-1).cpu().numpy())
                        if "gate_target" in batch:
                            gate_trues.append(batch["gate_target"].numpy())
                    trues.append(batch["label"].numpy())
                    fwd.append(batch["fwd_ret"].numpy())
                    tods.append(batch["tod_minutes"].numpy())
                    if "touch_minutes" in batch:
                        has_touch = True
                        touch.append(batch["touch_minutes"].numpy())
                else:
                    daily_ctx = batch.get("daily_ctx")
                    if daily_ctx is not None:
                        daily_ctx = daily_ctx.to(device)
                    foreign_ctx = batch.get("foreign_ctx")
                    if foreign_ctx is not None:
                        foreign_ctx = foreign_ctx.to(device)
                    fine_ctx = batch.get("fine_ctx")
                    if fine_ctx is not None:
                        fine_ctx = fine_ctx.to(device)
                    out = model(
                        x,
                        temporal_feat=batch["temporal_feat"].to(device),
                        time_pos=batch["time_pos"].to(device),
                        freq_feat=batch["freq_feat"].to(device),
                        symbol_id=batch["symbol_id"].to(device),
                        daily_ctx=daily_ctx,
                        foreign_ctx=foreign_ctx,
                        fine_ctx=fine_ctx,
                    )
                    base = x[:, -1, 3]
                    true_close = batch["target"][:, -1, 3].to(device)
                    true_pct = (true_close - base) / (base.abs() + 1e-8) * 100
                    preds.append(out["close_pct"].cpu().numpy())
                    trues.append(true_pct.cpu().numpy())

        p, t = np.concatenate(preds), np.concatenate(trues)
        if args.task == "classify":
            fwd = np.concatenate(fwd)
            if args.hierarchical_task and gate_scores:
                gs = np.concatenate(gate_scores)
                if gate_trues:
                    gt = np.concatenate(gate_trues).astype(int)
                    gate_pred = (gs >= 0.0).astype(int)
                    gate_cm = np.array([
                        [((gt == 0) & (gate_pred == 0)).sum(),
                         ((gt == 0) & (gate_pred == 1)).sum()],
                        [((gt == 1) & (gate_pred == 0)).sum(),
                         ((gt == 1) & (gate_pred == 1)).sum()],
                    ], dtype=np.int64)
                    gate_rec = [gate_cm[c, c] / max(gate_cm[c].sum(), 1) for c in (0, 1)]
                    n10 = max(1, int(np.ceil(len(gs) * 0.10)))
                    top10 = np.argsort(gs)[-n10:]
                    print(f"  [gate-only] rate={gt.mean():.1%} | acc={(gate_pred == gt).mean():.3f} "
                          f"| bal_acc={np.mean(gate_rec):.3f} | top10_precision={gt[top10].mean():.1%} "
                          f"| top10_n={n10}")
                dp = np.concatenate(direction_preds).astype(int)
                n_select = max(1, int(np.ceil(len(gs) * 0.10)))
                chosen = np.zeros(len(gs), dtype=bool)
                chosen[np.argsort(gs)[-n_select:]] = True
                p = np.ones(len(gs), dtype=np.int64)
                p[chosen] = np.where(dp[chosen] == 1, 2, 0)
                print(f"  [hier-cascade] 固定 gate top10%: {chosen.mean():.1%} | "
                      f"方向多/空={int((p == 2).sum())}/{int((p == 0).sum())}")
            n = len(p)
            cm = np.zeros((3, 3), dtype=np.int64)
            for ti, pi in zip(t.astype(np.int64), p.astype(np.int64)):
                cm[ti, pi] += 1
            recalls = [cm[c, c] / max(cm[c].sum(), 1) for c in range(3)]
            bal_acc = float(np.mean(recalls))
            prec = {c: cm[c, c] / max(cm[:, c].sum(), 1) for c in range(3)}
            cover = float((p != 1).mean())
            # 实战含义：模型喊"正/负信号"时，随后2小时的平均实际收益
            ret_pos = float(fwd[p == 2].mean()) if (p == 2).any() else float("nan")
            ret_neg = float(fwd[p == 0].mean()) if (p == 0).any() else float("nan")
            print(f"  [{key}] n={n:5d} | bal_acc={bal_acc:.3f} | "
                  f"P(正)={prec[2]:.1%} P(负)={prec[0]:.1%} | 信号覆盖={cover:.1%} | "
                  f"喊多均收益={ret_pos:+.3f}% 喊空均收益={ret_neg:+.3f}%")
            print(f"      混淆矩阵(行=真实[负无正], 列=预测):\n{cm}")
            rec_by_class = [cm[c, c] / max(cm[c].sum(), 1) for c in range(3)]
            print(f"      召回率: 负={rec_by_class[0]:.2f} 无={rec_by_class[1]:.2f} 正={rec_by_class[2]:.2f}")
            # ═══ 实战信号报告：模型喊开后，价格真的先摸到边界的概率 ═══
            # 标签本身就是 first-passage（先摸上轨=正，先摸下轨=负），
            # 所以"喊多且真实=正"的次数占比 = 你的真实命中率。
            lg = np.concatenate(logits)
            lg = lg - lg.max(axis=1, keepdims=True)
            prob = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
            base_pos = float((t == 2).mean())   # 不喊任何信号时的自然摸到上轨比例
            base_neg = float((t == 0).mean())
            print(f"      ── 实战信号报告（测试段共 {n} 根K线）──")
            print(f"      不喊信号的基准: 任意一根K线后先摸到上轨 {base_pos:.1%} / 先摸到下轨 {base_neg:.1%}")
            print(f"      {'信心阈值':<10}{'喊多次数':>8}{'约每百根K线':>12}{'喊多命中率':>12}{'喊空次数':>8}{'约每百根K线':>12}{'喊空命中率':>12}")
            for thr in [None, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
                tag = "argmax(现状)" if thr is None else f"≥{thr:.2f}"
                if thr is None:
                    sig_pos, sig_neg = p == 2, p == 0
                else:
                    sig_pos, sig_neg = prob[:, 2] >= thr, prob[:, 0] >= thr
                line = f"      {tag:<10}"
                for sig, cls in [(sig_pos, 2), (sig_neg, 0)]:
                    cnt = int(sig.sum())
                    if cnt > 0:
                        hit = float((t[sig] == cls).mean())
                        line += f"{cnt:>8}{cnt / n * 100:>11.1f}次{hit:>11.1%}"
                    else:
                        line += f"{cnt:>8}{0.0:>11.1f}次{'—':>11}"
                print(line)
            print(f"      读法: 命中率比基准(约{base_pos:.0%}/{base_neg:.0%})高得越多，信号越值钱；"
                  f"阈值越高喊得越少、命中率通常越高。")
            # ═══ 触达时间：喊对的情况下，信号发出后多久摸到边界 ═══
            if has_touch:
                tm = np.concatenate(touch)
                for cls, name in [(2, "摸到上轨"), (0, "摸到下轨")]:
                    all_hit = tm[t == cls]
                    all_hit = all_hit[np.isfinite(all_hit)]
                    if len(all_hit) == 0:
                        continue
                    print(f"      触达时间·{name}（全样本基准）: 中位 {np.median(all_hit):.0f} 分钟"
                          f" / 75分位 {np.percentile(all_hit, 75):.0f} 分钟")
                for thr in [None, 0.50]:
                    tag = "argmax" if thr is None else f"≥{thr:.2f}"
                    for cls, name in [(2, "喊多"), (0, "喊空")]:
                        sig = (p == cls) if thr is None else (prob[:, cls] >= thr)
                        hit_t = tm[sig & (t == cls)]
                        hit_t = hit_t[np.isfinite(hit_t)]
                        if len(hit_t) < 3:
                            continue
                        print(f"      触达时间·{tag}{name}且喊对({len(hit_t)}次): "
                              f"中位 {np.median(hit_t):.0f} 分钟"
                              f" / 平均 {hit_t.mean():.0f} 分钟"
                          f" / 75分位 {np.percentile(hit_t, 75):.0f} 分钟")
            # day_close: 分时段评估 —— 信号质量是否全天成立，还是只有尾盘"送分题"
            tod = np.concatenate(tods)
            for name, lo, hi in [("早盘9:00-10:15", 540, 615), ("午盘10:30-11:30", 630, 690),
                                 ("下午13:30-14:30", 810, 875), ("夜盘21:00-23:00", 1259, 1381)]:
                m = (tod >= lo) & (tod < hi)
                if m.sum() < 5:
                    continue
                pm, tm, fm = p[m], t[m], fwd[m]
                cm_b = np.zeros((3, 3), dtype=np.int64)
                for ti, pi in zip(tm.astype(np.int64), pm.astype(np.int64)):
                    cm_b[ti, pi] += 1
                rec = [cm_b[c, c] / max(cm_b[c].sum(), 1) for c in range(3)]
                cov = float((pm != 1).mean())
                rp = float(fm[pm == 2].mean()) if (pm == 2).any() else float("nan")
                print(f"      └ {name}: n={m.sum():4d} | bal_acc={np.mean(rec):.3f} | "
                      f"信号覆盖={cov:.0%} | 喊多均收益={rp:+.3f}%")
        else:
            ic = float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 else float("nan")
            acc = float((np.sign(p) == np.sign(t)).mean() * 100)
            print(f"  [{key}] n={len(p):5d} | IC={ic:+.4f} | 方向准确率={acc:.1f}%")


if __name__ == "__main__":
    main()
