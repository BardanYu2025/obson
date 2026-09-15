"""K 线序列数据集

将 obson DataGateway 输出的 DataFrame 转换为 Transformer 训练用的 tensor 序列。
支持多频率对齐和时间特征返回。
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# E4 excursion 分桶边（占 θ 比例）：0.5 对齐止损轨，0.8/1.0 对齐止盈轨
_EXC_EDGES = (0.25, 0.5, 0.8, 1.0, 1.5)
EXC_N_BINS = len(_EXC_EDGES) + 1  # 6 桶


def trading_day_ids(timestamps: pd.Series) -> np.ndarray:
    """交易日归属：夜盘 bar（18:00 后）算下一个工作日，日盘算当天。

    国内期货的"交易日"从前一晚夜盘开始（如周五 21:00 的夜盘算周一交易日）。
    返回每天一个从 0 递增的整数 id。
    """
    d = pd.to_datetime(timestamps)
    bset = pd.bdate_range(
        d.min().normalize(), d.max().normalize() + pd.Timedelta(days=10)
    ).values
    idx = np.searchsorted(bset, d.dt.normalize().values) + (
        d.dt.hour >= 18
    ).to_numpy().astype(int)
    _, ids = np.unique(bset[idx], return_inverse=True)
    return ids


def _align_hourly_to_5m(df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> np.ndarray:
    """将1小时线对齐到5分钟粒度

    对于5分钟序列中的每个时间戳，找到其所属的1小时K线，
    返回对齐后的1小时OHLCV特征 [n_5m, 5]。
    """
    df_5m = df_5m.copy()
    df_1h = df_1h.copy()

    df_5m["datetime"] = pd.to_datetime(df_5m["datetime"])
    df_1h["datetime"] = pd.to_datetime(df_1h["datetime"])

    # 对齐到"上一根已完成的"1小时K线，避免使用尚未走完的当前小时（未来函数）
    df_5m["hour_key"] = df_5m["datetime"].dt.floor("h") - pd.Timedelta(hours=1)
    df_1h["hour_key"] = df_1h["datetime"].dt.floor("h")

    # merge对齐
    cols = ["open", "high", "low", "close", "volume"]
    merged = df_5m[["datetime", "hour_key"]].merge(
        df_1h[["hour_key"] + cols],
        on="hour_key",
        how="left",
    )
    # 按原始5分钟顺序排序
    merged = merged.sort_values("datetime").reset_index(drop=True)

    hourly_features = merged[cols].to_numpy(dtype=np.float32)
    # 填充缺失值
    hourly_features = np.nan_to_num(hourly_features, nan=0.0, posinf=0.0, neginf=0.0)
    return hourly_features


class KLineDataset(Dataset):
    """K 线序列数据集（v4）

    新增：
    - 返回时间特征（月份sin/cos + 星期几onehot + 日内时间sin/cos）
    - 支持多频率对齐（5分钟线 + 1小时上下文）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 128,
        target_offset: int = 1,
        features: list[str] | None = None,
        normalize: bool = True,
        return_raw: bool = False,
        hourly_df: pd.DataFrame | None = None,
        jump_break_pct: float = 0.02,
        symbol_id: int = 0,
        label_mode: str = "regress",
        label_threshold: float | None = None,
        min_remaining_minutes: int = 30,
        theta_mode: str = "frozen",
        daily_bars: int = 0,
        cross_daily: dict | None = None,
        cross_daily_bars: int = 0,
        foreign_close: pd.DataFrame | None = None,
        foreign_bars: int = 0,
        soft_label: bool = False,
        use_vol_oi: bool = False,
        fine_df: pd.DataFrame | None = None,
        fine_bars: int = 0,
        fine_period_min: int = 5,
        base_period_min: int = 60,
        clean_path: bool = False,
        clean_frac: float = 0.5,
        strategy_label: bool = False,
        tp_frac: float = 0.8,
        stop_frac: float = 0.5,
        extra_breaks: np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
        day_ids_override: np.ndarray | None = None,
        anchor_days_ahead: int = 0,
    ):
        """
        :param df: 主频率DataFrame（如5分钟线），需包含 OHLCV + datetime 列
        :param seq_len: 输入序列长度（回看窗口）
        :param target_offset: 预测目标相对输入的偏移量，1 表示预测下一根 K 线
        :param label_mode: "regress"=回归目标（原路径）；"classify"=三分类标签
            （first-passage：未来 target_offset 根内先摸 +θ 上轨→正向(2)，
            先破 -θ 下轨→负向(0)，都没碰到或同根双触→无信号(1)）；
            "day_close"=当日收盘锚定三分类（first-passage 到本交易日收盘，
            基准为当前 bar 的 close，θ_t = θ_base × √(剩余bar/全日bar)）
        :param label_threshold: 分类阈值 θ，单位百分比（如 0.4 表示 0.4%）；
            day_close 模式下为全日基准 θ_base
        :param min_remaining_minutes: day_close 模式下，距收盘不足该分钟数的
            样本剔除（尾盘标签几乎必然"无信号"，且实盘不可用）
        :param theta_mode: day_close 模式的 θ 口径。"frozen"=训练段冻结的
            θ_base × √(剩余/全日)；"dynamic"=c × σ_w × √剩余，σ_w 为该样本
            输入窗口内的 bar log 收益 std（波动率自适应，因果无泄露），
            此时 label_threshold 传入的是 σ 倍数 c（无量纲）
        :param features: 使用的特征列，默认 ["open", "high", "low", "close"]
        :param normalize: 是否对每列做 Z-Score 标准化
        :param return_raw: 是否在 __getitem__ 中同时返回原始未标准化的序列
        :param hourly_df: 1小时线DataFrame，用于多频率上下文（可选）
        """
        self.seq_len = seq_len
        self.target_offset = target_offset
        self.features = features or ["open", "high", "low", "close"]
        self.normalize = normalize
        self.return_raw = return_raw
        self.symbol_id = symbol_id
        # 软标签（仅 day_close 模式生效）：按"实际走了 θ 的几成"给部分奖惩，
        # 冲到 92% 上轨没摸到 ≠ 全程阴跌，不再同罪
        self.soft_label = bool(soft_label) and label_mode == "day_close"
        # 干净路径约束（仅 day_close）：方向样本要求"摸轨前最大逆向偏移 ≤ clean_frac×θ"，
        # 先深插再摸轨的脏样本改判"按兵不动"（实盘做不下来的单，不教模型喊）
        self.clean_path = bool(clean_path) and label_mode == "day_close"
        self.clean_frac = float(clean_frac)
        # 策略对齐标签（仅 day_close）：多=先摸 +tp_frac×θ（途中未先破 −stop_frac×θ），
        # 空镜像；其余=无。让训练目标 = 实盘打法（0.8θ止盈/0.5θ止损）的胜负定义。
        # 与 clean_path 互斥（strategy_label 已内建止损语义）
        self.strategy_label = bool(strategy_label) and label_mode == "day_close"
        self.tp_frac = float(tp_frac)
        self.stop_frac = float(stop_frac)
        # T+N 锚（仅 day_close 语义族）：0=当日收盘（默认），1=次日收盘（next_close）。
        # 锚日必须落在本数据段内：段末 N 天样本无锚自然剔除（防跨段泄露的第一道闸，
        # 第二道闸 = 合约模式 _restrict_samples_by_dayset 的锚日过滤）
        self.anchor_days_ahead = int(anchor_days_ahead) if label_mode == "day_close" else 0
        self.anchor_days: np.ndarray | None = None  # [N] 每样本锚交易日 id
        self.soft_targets: np.ndarray | None = None  # [N, 2] = (m_dn, m_up)，对 θ 归一
        self.path_states: np.ndarray | None = None  # E3: [N, 4] int64，-1=歧义 mask
        self.exc_labels: np.ndarray | None = None  # E4: [N, 2] int64 (dn桶, up桶)，
        # 桶边 [0.25,0.5,0.8,1.0,1.5]×θ → 6 桶；excursion 是事实量，双触样本不 mask

        # 保存时间戳
        self.timestamps = pd.to_datetime(df["datetime"])

        # 成交量/持仓量通道（可选）：log1p 压缩量级后随全列一起 Z-Score；
        # CSV 缺列时补零并警告（老数据未含 OI，先跑通流程再重下）
        self.use_vol_oi = bool(use_vol_oi)
        if self.use_vol_oi:
            df = df.copy()
            if "volume" not in df.columns:
                print("  [vol_oi] 警告: 缺 volume 列，补零")
                df["volume"] = 0.0
            if "close_oi" not in df.columns:
                print("  [vol_oi] 警告: 缺 close_oi 列，补零（重下 tqsdk 数据后生效）")
                df["close_oi"] = 0.0
            df["volume"] = np.log1p(df["volume"].clip(lower=0))
            df["close_oi"] = np.log1p(df["close_oi"].clip(lower=0))
            features = ["open", "high", "low", "close", "volume", "close_oi"]

        # 提取特征列
        raw = df[self.features if not self.use_vol_oi else features].copy().to_numpy(dtype=np.float32)

        # 处理缺失值
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        if normalize:
            self.mean = raw.mean(axis=0)
            self.std = raw.std(axis=0) + 1e-8
            self.data = (raw - self.mean) / self.std
        else:
            self.mean = np.zeros(raw.shape[1], dtype=np.float32)
            self.std = np.ones(raw.shape[1], dtype=np.float32)
            self.data = raw

        # 原始（未标准化）序列始终保留：return_raw 取样本与 E7-A 预训练增强要用
        self.raw_data = raw

        # 交易日边界信息（用于 intra_day 位置编码和 rem_ratio）
        # day_ids_override：合约拼接模式下由外部按"合约段×交易日"预分配，
        # 防止拼接帧时间戳非单调导致 np.unique 把不同合约段的同名交易日合并
        if day_ids_override is not None:
            self.day_ids_global = np.asarray(day_ids_override, dtype=np.int64)
            if len(self.day_ids_global) != len(self.timestamps):
                raise ValueError("day_ids_override 长度必须与 df 行数一致")
        else:
            self.day_ids_global = trading_day_ids(self.timestamps).astype(np.int64)
        n_days = int(self.day_ids_global.max() + 1)
        day_counts = np.bincount(self.day_ids_global, minlength=n_days)
        self.day_starts_arr = np.cumsum(np.concatenate([[0], day_counts[:-1]])).astype(np.int64)
        self.day_ends_arr = (self.day_starts_arr + day_counts - 1).astype(np.int64)

        # 日线上下文：从分钟bar聚合成日K（开=首bar开，高/低=极值，收=末bar收），
        # 每个样本取"基准bar所在交易日之前"的 N 根完整日K，严格因果
        self.daily_bars = int(daily_bars)
        self.daily_ohlc: np.ndarray | None = None
        if self.daily_bars > 0:
            o = np.zeros(n_days, dtype=np.float32)
            h = np.zeros(n_days, dtype=np.float32)
            l = np.zeros(n_days, dtype=np.float32)
            c = np.zeros(n_days, dtype=np.float32)
            for d in range(n_days):
                s, e = self.day_starts_arr[d], self.day_ends_arr[d]
                o[d] = raw[s, 0]
                h[d] = raw[s: e + 1, 1].max()
                l[d] = raw[s: e + 1, 2].min()
                c[d] = raw[e, 3]
            self.daily_ohlc = np.stack([o, h, l, c], axis=-1)  # [n_days, 4]
            self._raw_ohlc = raw[:, :4].copy()  # 首日兜底用

        # 跨品种日线上下文（iTransformer 式 variate token 的原料）：
        # cross_daily = {symbol_id: (day_key_ord[F_s], ohlc[F_s,4])}，day_key_ord=该交易日
        # 日盘锚的日历日期序数（夜盘归入次日），跨品种按此 key 对齐。
        # 每个样本取 key < 基准bar交易日key 的最近 D 根（严格因果：只用已收盘日K）
        self.cross_daily = cross_daily
        self.cross_daily_bars = int(cross_daily_bars)
        self._cross_ids: list[int] = []
        self.day_key_ord: np.ndarray | None = None
        if self.cross_daily is not None and self.cross_daily_bars > 0:
            self._cross_ids = sorted(self.cross_daily.keys())
            # 本品种每个交易日的对齐 key = 该日最后一根 bar 的日历日（日盘锚在 13-15 点）
            self.day_key_ord = np.array([
                self.timestamps.iloc[int(self.day_ends_arr[d])].date().toordinal()
                for d in range(n_days)
            ], dtype=np.int64)

        # 外盘日线上下文：仅收盘价序列（Wind EDB 口径），严格因果。
        # 可用性规则：基准bar日历日 c → 只允许外盘日期 ≤ c-1
        # （美/欧盘 c-1 日的 bar 在北京时间 c 清晨前已收盘；夜盘 21:00 信号更保守也不会越界）
        self.foreign_bars = int(foreign_bars)
        self.foreign_dates: np.ndarray | None = None  # [F] int64 yyyymmdd
        self.foreign_feat: np.ndarray | None = None   # [F, 3] log_ret / mom5 / pos20
        if foreign_close is not None and self.foreign_bars > 0:
            fc = foreign_close.copy()
            # Wind EDB 的 date 是 yyyymmdd 整数/字符串，必须显式 format，
            # 否则 pd.to_datetime 把它当纳秒时间戳（全部落成 1970-01-01）
            fc["date"] = pd.to_datetime(fc["date"].astype(str), format="%Y%m%d", errors="coerce")
            fc = fc.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            # 用日期序数（ordinal）比较，避免 yyyymmdd 整数跨月溢出（20250701-1=20250700 的坑）
            fc["ord"] = fc["date"].map(lambda t: t.toordinal())
            cl_f = fc["close"].to_numpy(dtype=np.float64)
            eps = 1e-8
            log_ret = np.concatenate([[0.0], np.log(cl_f[1:] / np.maximum(cl_f[:-1], eps))])
            mom5 = np.concatenate([np.zeros(5), cl_f[5:] / np.maximum(cl_f[:-5], eps) - 1.0])
            # 20日区间位置：收盘在近20日高低中的归一化位置
            s = pd.Series(cl_f)
            hi20 = s.rolling(20, min_periods=5).max().to_numpy()
            lo20 = s.rolling(20, min_periods=5).min().to_numpy()
            pos20 = (cl_f - lo20) / np.maximum(hi20 - lo20, eps)
            feat = np.stack([log_ret, mom5, pos20], axis=-1)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            self.foreign_dates = fc["ord"].to_numpy()
            self.foreign_feat = feat

        # 细粒度上下文（5/15m token，与日线金字塔同构的"细端"）：
        # 只负责告诉模型"此刻的微观结构"，不作为预测目标。
        # 严格因果：基准bar（起始时刻 t_b，周期 P_b 分钟）收盘于 t_b+P_b；
        # 细粒度 bar（起始 t_f，周期 P_f）收盘于 t_f+P_f，
        # 仅当 t_f+P_f <= t_b+P_b 时可见（与主窗口末根bar同一收盘时刻对齐）。
        self.fine_bars = int(fine_bars)
        self.fine_ohlc: np.ndarray | None = None    # [F, 4] 原始 OHLC
        self.fine_close_ns: np.ndarray | None = None  # [F] 每根细bar的收盘时刻（纳秒）
        self._base_close_ns: np.ndarray | None = None  # [N_bars] 主bar收盘时刻
        if fine_df is not None and self.fine_bars > 0:
            ff = fine_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            fraw = ff[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)
            fraw = np.nan_to_num(fraw, nan=0.0, posinf=0.0, neginf=0.0)
            self.fine_ohlc = fraw
            fts_ns = pd.to_datetime(ff["datetime"]).to_numpy(dtype="datetime64[ns]").astype("int64")
            self.fine_close_ns = fts_ns + int(fine_period_min) * 60 * 10**9
            base_ns = self.timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
            self._base_close_ns = base_ns + int(base_period_min) * 60 * 10**9
            # 细粒度历史不可得时的中性兜底：基准bar自身OHLC造"平bar"
            # （O=H=L=C=已知量，编码后≈零特征；预测点在基准bar收盘，无未来泄露）
            self._fine_fallback = raw[:, :4].copy()

        # 多频率对齐：1小时上下文
        self.hourly_data = None
        if hourly_df is not None:
            hourly_raw = _align_hourly_to_5m(df, hourly_df)
            # 用训练集的统计量标准化1小时特征
            if normalize:
                self.hourly_mean = hourly_raw.mean(axis=0)
                self.hourly_std = hourly_raw.std(axis=0) + 1e-8
                self.hourly_data = (hourly_raw - self.hourly_mean) / self.hourly_std
            else:
                self.hourly_mean = np.zeros(5, dtype=np.float32)
                self.hourly_std = np.ones(5, dtype=np.float32)
                self.hourly_data = hourly_raw

        # 有效样本数 = 总长度 - seq_len - target_offset + 1
        self.n_samples = len(self.data) - seq_len - target_offset + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"数据长度 {len(df)} 不足以构造 seq_len={seq_len} "
                f"target_offset={target_offset} 的样本"
            )

        # 断链保护：两种情况视为断链，样本窗口（含标签）不允许跨越断点
        # 1. 相邻K线时间差超过5天（长假/数据缺失）
        # 2. 相邻K线收盘价跳空 |ret| > jump_break_pct（主力合约换月/数据拼接缺口，默认2%）
        max_gap = pd.Timedelta(days=5)
        gaps = self.timestamps.diff().to_numpy()  # gaps[i] = ts[i] - ts[i-1]
        break_pos = np.flatnonzero(gaps > max_gap)  # bar i 是新段的开始
        if jump_break_pct is not None:
            closes = df[self.features].to_numpy(dtype=np.float64)[:, 3]
            prev = np.maximum(closes[:-1], 1e-8)
            rets = np.abs(closes[1:] / prev - 1.0)
            jump_breaks = np.flatnonzero(rets > jump_break_pct) + 1
            break_pos = np.union1d(break_pos, jump_breaks).astype(np.int64)
            if len(jump_breaks) > 0:
                print(f"  [断链保护] 检测到 {len(jump_breaks)} 处跳空断点 (|ret|>{jump_break_pct:.0%})")
        # 合约模式：外部注入的段边界断点（换月切换点），样本窗口/标签路径不许跨段
        if extra_breaks is not None and len(extra_breaks) > 0:
            break_pos = np.union1d(break_pos, np.asarray(extra_breaks, dtype=np.int64))
        last_bar_needed = seq_len + target_offset - 1  # 样本用到的最远 bar（含标签）
        valid = np.ones(self.n_samples, dtype=bool)
        for b in break_pos:
            # 样本起点 idx 合法要求：窗口 (idx, idx+last_bar_needed] 内无断点 b
            lo = max(0, b - last_bar_needed)
            hi = min(b, self.n_samples)  # idx < b
            valid[lo:hi] = False
        # 合约模式：外部样本掩码（按基准 bar 全局位置索引，True=保留）；
        # 用于"只保留基准 bar 落在主力任期内的样本"
        if sample_mask is not None:
            sm = np.asarray(sample_mask, dtype=bool)
            if len(sm) != len(self.data):
                raise ValueError("sample_mask 长度必须与 df 行数一致")
            base_pos = np.arange(len(valid)) + seq_len - 1
            valid &= sm[base_pos]
        self.valid_indices = np.flatnonzero(valid)
        self.n_samples = len(self.valid_indices)
        if self.n_samples <= 0:
            raise ValueError("断链保护后无有效样本，请检查数据或放宽 max_gap")

        # 三分类标签（first-passage，用未来窗口的最高/最低，对应"不一定持仓到期"）
        self.label_mode = label_mode
        self.label_threshold = label_threshold
        self.labels: np.ndarray | None = None
        self.fwd_rets: np.ndarray | None = None
        self.touch_minutes: np.ndarray | None = None  # 喊对方向时，多少分钟后摸到边界（无信号=NaN）
        self.thetas: np.ndarray | None = None  # 每个样本的边界宽度 θ（百分比），回测盈亏用
        if label_mode == "classify":
            if label_threshold is None:
                raise ValueError("classify 模式必须提供 label_threshold（百分比）")
            vi = self.valid_indices
            base_i = vi + seq_len - 1
            base = raw[base_i, 3].astype(np.float64)
            thr = label_threshold / 100.0
            offset = target_offset
            # first-passage：未来 offset 根内，上轨/下轨首次被触碰的位置
            first_up = np.full(len(vi), offset + 1, dtype=np.int64)
            first_dn = np.full(len(vi), offset + 1, dtype=np.int64)
            for k in range(1, offset + 1):
                up_hit = raw[base_i + k, 1] / base - 1.0 >= thr
                dn_hit = raw[base_i + k, 2] / base - 1.0 <= -thr
                first_up = np.where(up_hit & (first_up > offset), k, first_up)
                first_dn = np.where(dn_hit & (first_dn > offset), k, first_dn)
            labels = np.ones(len(vi), dtype=np.int64)  # 默认无信号
            labels[first_up < first_dn] = 2   # 先摸上轨 → 正向
            labels[first_dn < first_up] = 0   # 先破下轨 → 负向（同根双触保持1）
            self.labels = labels
            self.fwd_rets = ((raw[base_i + offset, 3] / base - 1.0) * 100).astype(np.float32)
        elif label_mode == "day_close":
            # 当日收盘锚定：窗口末 bar → 本交易日 15:00 收盘的 first-passage
            if label_threshold is None:
                raise ValueError("day_close 模式必须提供 label_threshold（全日基准 θ，百分比）")
            day_ids = (
                np.asarray(day_ids_override, dtype=np.int64)
                if day_ids_override is not None
                else trading_day_ids(self.timestamps)
            )
            hours = (
                self.timestamps.dt.hour + self.timestamps.dt.minute / 60
            ).to_numpy()
            cl = raw[:, 3].astype(np.float64)
            hi = raw[:, 1].astype(np.float64)
            lo = raw[:, 2].astype(np.float64)
            th_base = label_threshold / 100.0
            # dynamic 口径：σ_w = 输入窗口内 bar log 收益 std（window=seq_len，与模型所见一致）
            sigma_w = None
            if theta_mode == "dynamic":
                log_ret = np.log(cl[1:] / np.maximum(cl[:-1], 1e-8))
                sigma_w = (
                    pd.Series(log_ret)
                    .rolling(seq_len - 1, min_periods=max((seq_len - 1) // 2, 10))
                    .std()
                    .to_numpy()
                )
                sigma_w = np.concatenate([[np.nan], sigma_w])  # sigma_w[j]: 覆盖 close[..j]
                th_base = label_threshold  # dynamic 下传进来的是无量纲 σ 倍数 c
            vi = self.valid_indices
            base_i_all = vi + seq_len - 1
            labels_all = np.full(len(vi), -1, dtype=np.int64)  # -1 = 无有效收盘锚/被剔除
            fwd_all = np.full(len(vi), np.nan, dtype=np.float64)
            touch_min_all = np.full(len(vi), np.nan, dtype=np.float64)  # 摸到胜出轨距信号的分钟数
            theta_all = np.full(len(vi), np.nan, dtype=np.float64)  # 每个样本的 θ（%）
            soft_all = np.full((len(vi), 2), np.nan, dtype=np.float64)  # (m_dn, m_up)：双向最大偏移÷θ
            # E3 路径状态：4 节点(剩余25/50/75/100%，ceil 离散索引) × 3 态(0=未触轨/1=已上轨/2=已下轨)，
            # -1=整行 mask（同根双触歧义样本，masked CE 忽略）。吸收态：首次触轨后保持该状态
            path_state_all = np.full((len(vi), 4), -1, dtype=np.int64)
            # E4 excursion 分桶标签：(m_dn桶, m_up桶)，偏移占 θ 比例 digitize 到
            # [0.25,0.5,0.8,1.0,1.5] 六桶（0.8/1.0 对齐止盈轨，0.5 对齐止损轨）
            exc_all = np.full((len(vi), 2), -1, dtype=np.int64)
            ts = self.timestamps.to_numpy()
            # 完整交易日的最小 bar 数：按本频率日长度的中位数定（5m≈69, 60m≈7），
            # 写死阈值会把 60m，整天误杀
            day_lens = np.unique(day_ids, return_counts=True)[1]
            min_day_bars = max(int(np.median(day_lens) * 0.5), 3)
            n_double_touch = 0  # 同根双触计数（训练侧记"无"，执行侧记止损；两阶段语义）
            anchor_day_all = np.full(len(vi), -1, dtype=np.int64)  # 每样本锚交易日 id
            # 先筛出完整交易日清单，再按 anchor_days_ahead 取锚日：
            # 0=当日收盘（原 day_close），1=次日收盘（next_close，T+1 信息量实验）
            ok_days = []
            for d in np.unique(day_ids):
                rows_d = np.flatnonzero(day_ids == d)
                a = rows_d[-1]
                # 截断的交易日（跨切分边界/缺日盘数据）：锚不在 13-15 点，整天剔除
                if (13.0 <= hours[a] <= 15.0) and len(rows_d) >= min_day_bars:
                    ok_days.append(d)
            for i, d in enumerate(ok_days):
                ia = i + self.anchor_days_ahead
                if ia >= len(ok_days):
                    break  # 锚日超出本段（T+1 时段末样本无锚，自然剔除防泄露）
                rows_d = np.flatnonzero(day_ids == d)
                a = np.flatnonzero(day_ids == ok_days[ia])[-1]  # 收盘锚 = 锚日最后一根 bar
                deltas = np.diff(ts[rows_d]).astype("timedelta64[s]").astype(float) / 60.0
                step = float(np.median(deltas)) if len(deltas) > 0 else 5.0
                full = a - rows_d[0]  # 全日 bar 数（夜盘+日盘）
                # 当日内的断链点（换月跳空等）：标签路径跨过断点的样本剔除
                brk = break_pos[(break_pos > rows_d[0]) & (break_pos <= a)]
                sel = np.flatnonzero((base_i_all >= rows_d[0]) & (base_i_all <= rows_d[-1]) & (base_i_all < a))
                for s in sel:
                    j = int(base_i_all[s])
                    rem = a - j
                    if rem * step < min_remaining_minutes:
                        continue  # 尾盘：标签几乎必然"无信号"，且实盘不可用
                    if len(brk):
                        bi = int(np.searchsorted(brk, j, side="right"))
                        if bi < len(brk) and brk[bi] <= a:
                            continue
                    anchor_day_all[s] = ok_days[ia]
                    base = max(cl[j], 1e-8)
                    if sigma_w is not None:
                        sw = sigma_w[j]
                        if not np.isfinite(sw) or sw <= 0:
                            continue  # 窗口初期 σ 不可用
                        th = th_base * sw * math.sqrt(rem)
                    else:
                        th = th_base * math.sqrt(rem / full)
                    up_hit = hi[j + 1 : a + 1] / base - 1.0 >= th
                    dn_hit = 1.0 - lo[j + 1 : a + 1] / base >= th
                    # argmax 找第一个 True；无触碰置极大值表示"先到不了"
                    fu = int(np.argmax(up_hit)) + 1 if up_hit.any() else 10**9
                    fd = int(np.argmax(dn_hit)) + 1 if dn_hit.any() else 10**9
                    labels_all[s] = 2 if fu < fd else (0 if fd < fu else 1)
                    if fu == fd and fu < 10**9:
                        n_double_touch += 1
                    theta_all[s] = th * 100.0  # 存百分比，与 fwd_ret 同单位
                    # 软标签素材：双向最大偏移占 θ 的比例（摸到=≥1，冲一半=0.5）
                    soft_all[s, 0] = (1.0 - lo[j + 1 : a + 1].min() / base) / th
                    soft_all[s, 1] = (hi[j + 1 : a + 1].max() / base - 1.0) / th
                    # E4：excursion 分桶（与软标签同素材，标准 θ 轨、strategy_label 覆盖前）
                    exc_all[s, 0] = np.digitize(soft_all[s, 0], _EXC_EDGES)  # dn 桶
                    exc_all[s, 1] = np.digitize(soft_all[s, 1], _EXC_EDGES)  # up 桶
                    # E3 路径状态（用标准 θ 轨的 fu/fd，strategy_label 覆盖前）：
                    # 节点 k = 剩余 bar 的 25/50/75/100% 处，离散索引 n_k = max(1, ceil(frac*rem))
                    # 状态由"首次触轨"决定（吸收态）；同根双触(fu==fd 且可达)时整行 mask：
                    # OHLC 无法确定先后，任何节点监督都可能错（与主标签=无 对齐）
                    if not (fu == fd and fu < 10**9):
                        for k, frac in enumerate((0.25, 0.5, 0.75, 1.0)):
                            n_k = max(1, math.ceil(rem * frac))
                            if fu <= n_k:
                                path_state_all[s, k] = 1   # 已上轨（吸收态）
                            elif fd <= n_k:
                                path_state_all[s, k] = 2   # 已下轨（吸收态）
                            else:
                                path_state_all[s, k] = 0   # 未触轨
                    if self.strategy_label:
                        # 策略对齐标签：止盈轨 ±tp_frac×θ，止损轨 ∓stop_frac×θ
                        # 多 = 先摸 +tp×θ 且此前未破 −stop×θ（先止损再到位 → 无）
                        up_tp = hi[j + 1 : a + 1] / base - 1.0 >= self.tp_frac * th
                        dn_tp = 1.0 - lo[j + 1 : a + 1] / base >= self.tp_frac * th
                        up_st = hi[j + 1 : a + 1] / base - 1.0 >= self.stop_frac * th
                        dn_st = 1.0 - lo[j + 1 : a + 1] / base >= self.stop_frac * th
                        f_ut = int(np.argmax(up_tp)) + 1 if up_tp.any() else 10**9
                        f_dt = int(np.argmax(dn_tp)) + 1 if dn_tp.any() else 10**9
                        f_us = int(np.argmax(up_st)) + 1 if up_st.any() else 10**9
                        f_ds = int(np.argmax(dn_st)) + 1 if dn_st.any() else 10**9
                        is_long = f_ut < f_ds    # 止盈先到、止损未先到
                        is_short = f_dt < f_us
                        labels_all[s] = 2 if is_long else (0 if is_short else 1)
                        # 软标签分子改对止盈轨归一（摸到止盈=≥1），止盈未到但方向
                        # 有进展的样本仍拿部分奖励；被止损的方向分量清零
                        soft_all[s, 0] = (1.0 - lo[j + 1 : a + 1].min() / base) / (self.tp_frac * th)
                        soft_all[s, 1] = (hi[j + 1 : a + 1].max() / base - 1.0) / (self.tp_frac * th)
                        if labels_all[s] != 2:
                            soft_all[s, 1] = min(soft_all[s, 1], 0.999)  # 摸到止盈但被止损先行 → 不满分
                        if labels_all[s] != 0:
                            soft_all[s, 0] = min(soft_all[s, 0], 0.999)
                        fu, fd = f_ut, f_dt  # 触达时间统计按止盈轨口径
                    elif self.clean_path and fu != fd and min(fu, fd) < 10**9:
                        # 干净路径：摸轨时刻之前的最大逆向偏移 > clean_frac×θ → 脏样本改判"无"
                        t = min(fu, fd)
                        if fu < fd:  # 多：逆向 = 向下跌破
                            mae = 1.0 - lo[j + 1 : j + t + 1].min() / base
                        else:        # 空：逆向 = 向上反弹
                            mae = hi[j + 1 : j + t + 1].max() / base - 1.0
                        if mae > self.clean_frac * th:
                            labels_all[s] = 1
                            # 软标签：脏方向的分清零（先深插再到位不奖励）；
                            # 反向偏移保留为事实，"无"分量由 (1−m_up)(1−m_dn) 自然变大
                            if fu < fd:
                                soft_all[s, 1] = 0.0
                            else:
                                soft_all[s, 0] = 0.0
                    if fu != fd and min(fu, fd) < 10**9:
                        touch_min_all[s] = min(fu, fd) * step  # 信号→摸到边界的分钟数
                    fwd_all[s] = (cl[a] / base - 1.0) * 100.0
            keep = labels_all >= 0
            n_dropped = int((~keep).sum())
            n_kept = int(keep.sum())
            _anchor_tag = "day_close" if self.anchor_days_ahead == 0 else f"day_close+{self.anchor_days_ahead}"
            print(
                f"  [{_anchor_tag}] 剔除 {n_dropped} 个样本"
                f"（尾盘{min_remaining_minutes}min内/截断日/日内断链）"
                f" | 有效 {n_kept} | 标签分布 空/无/多="
                f"{int((labels_all[keep] == 0).sum())}/{int((labels_all[keep] == 1).sum())}/"
                f"{int((labels_all[keep] == 2).sum())}"
                f" | 同根双触 {n_double_touch}（{n_double_touch / max(n_kept, 1):.2%}）"
            )
            self.labels = labels_all[keep]
            self.anchor_days = anchor_day_all[keep]  # T+N 泄露防护：每样本锚交易日 id
            self.fwd_rets = fwd_all[keep].astype(np.float32)  # 持有到收盘的实际收益(%)
            self.touch_minutes = touch_min_all[keep].astype(np.float32)
            self.thetas = theta_all[keep].astype(np.float32)
            self.path_states = path_state_all[keep]  # E3：int64 [N, 4]，-1=歧义 mask
            self.exc_labels = exc_all[keep]          # E4：int64 [N, 2] (dn桶, up桶)
            if self.soft_label:
                self.soft_targets = soft_all[keep].astype(np.float32)
            self.valid_indices = vi[keep]
            self.n_samples = len(self.valid_indices)
            if self.n_samples <= 0:
                raise ValueError("day_close 标注后无有效样本，请检查数据")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pos = idx  # labels/fwd_rets 按 valid 顺序存储，用位置索引
        idx = int(self.valid_indices[idx])
        # 输入序列: [idx, idx+seq_len)
        x = self.data[idx : idx + self.seq_len]
        # 目标序列: [idx+target_offset, idx+target_offset+seq_len)
        y = self.data[
            idx + self.target_offset : idx + self.target_offset + self.seq_len
        ]

        # 时间特征预处理
        # 1. 月份 sin/cos（周期性）
        # 2. 交易时段 onehot：早盘/午盘/下午/夜盘
        # 3. 交易日内相对位置 sin/cos（替代 24h 周期，解决夜盘-日盘跳变）
        # 4. 剩余时间比例 rem_ratio（显式对齐 θ 变化）
        ts = self.timestamps.iloc[idx : idx + self.seq_len]
        month = ts.dt.month.values.astype(np.float32)
        month_sin = np.sin(2 * np.pi * month / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * month / 12).astype(np.float32)

        # 交易时段 onehot：早盘(9-10:15) / 午盘(10:30-11:30) / 下午(13:30-15:00) / 夜盘(21-23/1:00)
        tod = (ts.dt.hour.values * 60 + ts.dt.minute.values).astype(np.int64)
        session = np.zeros((len(ts), 4), dtype=np.float32)
        session[:, 0] = ((tod >= 540) & (tod < 615)).astype(np.float32)   # 早盘
        session[:, 1] = ((tod >= 630) & (tod < 690)).astype(np.float32)   # 午盘
        session[:, 2] = ((tod >= 810) & (tod <= 900)).astype(np.float32)  # 下午
        session[:, 3] = ((tod >= 1260) | (tod <= 60)).astype(np.float32)  # 夜盘

        # 交易日内相对位置 + 剩余时间比例
        day_ids_win = self.day_ids_global[idx : idx + self.seq_len]
        starts = self.day_starts_arr[day_ids_win]
        ends = self.day_ends_arr[day_ids_win]
        day_lens = np.maximum(ends - starts + 1, 1)
        bar_indices = np.arange(idx, idx + self.seq_len) - starts
        rems = ends - np.arange(idx, idx + self.seq_len) + 1
        intra_day_pos = bar_indices.astype(np.float32) / day_lens.astype(np.float32)
        intra_day_sin = np.sin(2 * np.pi * intra_day_pos).astype(np.float32)
        intra_day_cos = np.cos(2 * np.pi * intra_day_pos).astype(np.float32)
        rem_ratio = (rems.astype(np.float32) / day_lens.astype(np.float32)).clip(0, 1)

        temporal_feat = np.concatenate([
            month_sin[:, None], month_cos[:, None],
            session,
            intra_day_sin[:, None], intra_day_cos[:, None],
            rem_ratio[:, None],
        ], axis=-1)  # [S, 9]

        # 不规则时间位置：以窗口内相邻bar间隔的中位数为 1 个单位，
        # 隔夜/休市跳空会累积成更大的位置差，供时间感知 ALiBi 使用
        deltas = np.diff(ts.values).astype("timedelta64[s]").astype(np.float32) / 60.0
        median_dt = np.median(deltas) if len(deltas) > 0 else 1.0
        if median_dt <= 0:
            median_dt = 1.0
        steps = deltas / median_dt
        time_pos = np.concatenate([[0.0], np.cumsum(steps)]).astype(np.float32)  # [S]

        # 频率标识：让模型区分 15m / 30m 等不同尺度的输入
        freq_map = {5.0: 0, 15.0: 1, 30.0: 2, 60.0: 3}
        freq_feat = freq_map.get(int(round(median_dt)), 1)

        item = {
            "seq": torch.from_numpy(x),
            "target": torch.from_numpy(y),
            "temporal_feat": torch.from_numpy(temporal_feat),
            "time_pos": torch.from_numpy(time_pos),
            "freq_feat": torch.tensor(freq_feat, dtype=torch.float32),
            "symbol_id": torch.tensor(self.symbol_id, dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[pos], dtype=torch.long)
            item["fwd_ret"] = torch.tensor(self.fwd_rets[pos], dtype=torch.float32)
            # Teacher utility target：与生产止盈/止损语义一致，单位为 theta 倍数。
            # 命中方向=+0.8，反向先触=−0.5，未触轨保留持有到锚点的实际收益。
            # 这是现有标签和事实收益的确定性派生，不引入新标签或未来信息。
            if self.thetas is not None:
                th = max(float(self.thetas[pos]), 1e-6)
                fwd_theta = float(self.fwd_rets[pos]) / th
                label = int(self.labels[pos])
                short_u = 0.8 if label == 0 else (-0.5 if label == 2 else -fwd_theta)
                long_u = 0.8 if label == 2 else (-0.5 if label == 0 else fwd_theta)
                item["utility_target"] = torch.tensor(
                    np.clip([short_u, long_u], -2.0, 2.0), dtype=torch.float32
                )
            if self.exc_labels is not None:
                item["exc_labels"] = torch.from_numpy(self.exc_labels[pos])  # [2] long (dn,up)
            base_ts = self.timestamps.iloc[idx + self.seq_len - 1]
            item["tod_minutes"] = torch.tensor(
                base_ts.hour * 60 + base_ts.minute, dtype=torch.long
            )
            if self.touch_minutes is not None:
                item["touch_minutes"] = torch.tensor(
                    self.touch_minutes[pos], dtype=torch.float32
                )
            if self.thetas is not None:
                item["theta"] = torch.tensor(self.thetas[pos], dtype=torch.float32)
            if self.path_states is not None:
                item["path_states"] = torch.from_numpy(self.path_states[pos])  # [4] long
            if self.soft_targets is not None:
                # 软目标 v2（修正版）：
                # - 方向权重 = 偏移占比²（锐化：摸到≈one-hot，冲一半仍有重赏）
                # - "无" = (1−m_up)(1−m_dn)（离两边都远才算无；v1 的 1−max 给得太薄，
                #   导致震荡样本方向质量占优，模型从不喊"无"）
                m_dn, m_up = float(self.soft_targets[pos, 0]), float(self.soft_targets[pos, 1])
                w_neg, w_pos = m_dn * m_dn, m_up * m_up
                w_none = max(1.0 - m_up, 0.0) * max(1.0 - m_dn, 0.0)
                t = np.array([w_neg, w_none, w_pos], dtype=np.float32)
                item["soft_label"] = torch.from_numpy(t / max(t.sum(), 1e-8))
        if self.daily_ohlc is not None:
            base_bar = idx + self.seq_len - 1
            d = int(self.day_ids_global[base_bar])  # 基准bar所在交易日
            D = self.daily_bars
            ctx = self.daily_ohlc[max(0, d - D): d]  # 只取已收盘的完整日K（不含当日）
            if len(ctx) == 0:
                ctx = self._raw_ohlc[base_bar: base_bar + 1].copy()  # 首日无历史：用当根bar兜底
            if len(ctx) < D:  # 左侧 replicate 填充，保持定长
                ctx = np.concatenate([np.repeat(ctx[:1], D - len(ctx), axis=0), ctx], axis=0)
            item["daily_ctx"] = torch.from_numpy(ctx.astype(np.float32))
        if self.foreign_feat is not None:
            base_ts = self.timestamps.iloc[idx + self.seq_len - 1]
            c_ord = base_ts.date().toordinal()
            # 严格因果：只允许 ≤ c-1 日的外盘 bar（c-1 日的外盘在北京时间 c 清晨前已收盘）
            hi_f = int(np.searchsorted(self.foreign_dates, c_ord - 1, side="right"))
            D2 = self.foreign_bars
            fctx = self.foreign_feat[max(0, hi_f - D2): hi_f]
            if len(fctx) == 0:
                fctx = self.foreign_feat[:1]  # 数据起点前：首行兜底
            if len(fctx) < D2:  # 左侧零填充：无信息，区别于 replicate（不伪造趋势）
                fctx = np.concatenate([np.zeros((D2 - len(fctx), fctx.shape[1]), np.float32), fctx], axis=0)
            item["foreign_ctx"] = torch.from_numpy(fctx.astype(np.float32))
        if self.cross_daily is not None and self.cross_daily_bars > 0:
            base_bar = idx + self.seq_len - 1
            d = int(self.day_ids_global[base_bar])
            key = int(self.day_key_ord[d])
            D3 = self.cross_daily_bars
            NS = len(self._cross_ids)
            cross = np.zeros((NS, D3, 4), dtype=np.float32)
            mask = np.zeros(NS, dtype=np.float32)
            for k, sid in enumerate(self._cross_ids):
                keys, ohlc = self.cross_daily[sid]
                hi_c = int(np.searchsorted(keys, key - 1, side="right"))  # 只取已收盘日K
                sub = ohlc[max(0, hi_c - D3): hi_c]
                if len(sub) == 0:
                    continue  # 该品种在此日期前无任何数据：留零 + mask=0
                if len(sub) < D3:
                    sub = np.concatenate([np.repeat(sub[:1], D3 - len(sub), axis=0), sub], axis=0)
                cross[k] = sub
                mask[k] = 1.0
            item["cross_ctx"] = torch.from_numpy(cross)
            item["cross_mask"] = torch.from_numpy(mask)
        if self.fine_ohlc is not None:
            base_bar = idx + self.seq_len - 1
            bc = self._base_close_ns[base_bar]
            # 严格因果：细bar收盘时刻 <= 基准bar收盘时刻
            hi_fine = int(np.searchsorted(self.fine_close_ns, bc, side="right"))
            F = self.fine_bars
            if hi_fine == 0:
                # 该时点之前无任何细粒度数据（免费版翻页窗口限制）：
                # 用基准bar收盘价造中性"平bar"（O=H=L=C=已知量，编码后≈零特征），
                # 绝不取未来第一行
                flat = np.full((F, 4), self._fine_fallback[base_bar, 3], dtype=np.float32)
                ftx = flat
            else:
                ftx = self.fine_ohlc[max(0, hi_fine - F): hi_fine]
                if len(ftx) < F:  # 左侧 replicate 已有历史，保持定长（与日线前缀同口径）
                    ftx = np.concatenate([np.repeat(ftx[:1], F - len(ftx), axis=0), ftx], axis=0)
            item["fine_ctx"] = torch.from_numpy(ftx.astype(np.float32))
        if self.return_raw:
            item["raw_seq"] = torch.from_numpy(
                self.raw_data[idx : idx + self.seq_len]
            )
        if self.hourly_data is not None:
            h = self.hourly_data[idx : idx + self.seq_len]
            item["hourly_ctx"] = torch.from_numpy(h)
        return item

    def inverse_transform(
        self,
        normalized: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """将标准化后的数据还原为原始尺度"""
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.detach().cpu().numpy()
        return normalized * self.std + self.mean

    def denormalize_prediction(
        self,
        pred: torch.Tensor,
    ) -> np.ndarray:
        """将模型输出的标准化预测还原为原始价格"""
        return self.inverse_transform(pred)


def _striped_split(
    df: pd.DataFrame,
    seq_len: int,
    n_blocks: int = 10,
    val_residues: tuple[int, ...] = (2, 7),
    test_ratio: float = 0.15,
    **kwargs,
) -> tuple:
    """分块交替切分（striped blocks）+ 块间天然 embargo

    最后 15% 时间保留为纯未来测试集；剩余部分切成 n_blocks 个连续块，
    块号 % 5 ∈ val_residues 的进验证集，其余进训练集。
    每个块独立构造 KLineDataset 再 Concat —— 窗口不可能跨越块边界，
    等效于在每个边界自动加了 seq_len 长度的隔离带，无泄露。
    """
    from torch.utils.data import ConcatDataset

    n = len(df)
    n_test = int(n * test_ratio)
    main_df = df.iloc[: n - n_test].reset_index(drop=True)
    test_df = df.iloc[n - n_test:].reset_index(drop=True)

    indices = np.array_split(np.arange(len(main_df)), n_blocks)
    blocks = [main_df.iloc[idx].reset_index(drop=True) for idx in indices]
    train_parts, val_parts = [], []
    for i, block in enumerate(blocks):
        ds = KLineDataset(block, seq_len=seq_len, normalize=False, **kwargs)
        if len(ds) == 0:
            continue
        (val_parts if i % 5 in val_residues else train_parts).append(ds)

    train_ds = ConcatDataset(train_parts)
    val_ds = ConcatDataset(val_parts)
    test_ds = KLineDataset(test_df, seq_len=seq_len, normalize=False, **kwargs)
    # inverse_transform 兼容（normalize=False 时本来就是 0/1）
    train_ds.mean = train_parts[0].mean
    train_ds.std = train_parts[0].std
    return train_ds, val_ds, test_ds


def _restrict_samples(ds, min_base_bar: int, max_label_bar: int | None = None) -> None:
    """把数据集的样本限制为"窗口末 bar（valid_indices+seq_len-1）≥ min_base_bar"。
    warm-up 配套：数据集建在含历史的完整 df 上，此函数截出评估段样本，
    labels/fwd_rets/touch_minutes/thetas/soft_targets 与 valid_indices 同步过滤。

    越界防护（老师四轮确认轮补充）：
    - day_close 模式：切分点已吸附交易日边界，标签锚必然落在本段内，只需 min_base_bar；
    - 其他标签模式（如 classify/horizon）：标签覆盖 base+1 .. base+target_offset，
      必须再传 max_label_bar（= 本段最后一根 bar 下标），防止评估段末尾样本的
      标签偷看下一段数据。"""
    base = ds.valid_indices + ds.seq_len - 1
    keep = base >= min_base_bar
    if max_label_bar is not None:
        keep &= (base + ds.target_offset) <= max_label_bar
    ds.valid_indices = ds.valid_indices[keep]
    ds.n_samples = len(ds.valid_indices)
    for attr in ("labels", "fwd_rets", "touch_minutes", "thetas", "soft_targets", "path_states", "exc_labels", "anchor_days"):
        arr = getattr(ds, attr, None)
        if arr is not None:
            setattr(ds, attr, arr[keep])
    if ds.n_samples <= 0:
        raise ValueError("warm-up 截样后评估段无样本，检查切分点")


def build_datasets(
    df: pd.DataFrame,
    seq_len: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    hourly_df: pd.DataFrame | None = None,
    split_mode: str = "time",
    **kwargs,
) -> tuple:
    """切分训练/验证/测试集

    :param split_mode: "time"=时间硬切（默认，验证集=纯未来）；
        "striped"=分块交替（训练/验证覆盖全时段各种regime，测试集仍是纯未来）
    """
    if split_mode == "striped":
        return _striped_split(df, seq_len=seq_len, test_ratio=1 - train_ratio - val_ratio, **kwargs)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # day_close 模式：切分点吸附到交易日边界，避免把同一交易日的
    # "样本"和"收盘锚"切进两个数据集（既是泄露，也让标签锚不完整）
    if kwargs.get("label_mode") == "day_close":
        day_ids = trading_day_ids(df["datetime"])
        change = np.flatnonzero(np.diff(day_ids)) + 1
        bounds = np.concatenate([[0], change, [n]])
        n_train_raw, n_val_raw = n_train, n_val
        n_train = int(bounds[max(int(np.searchsorted(bounds, n_train_raw, side="right")) - 1, 1)])
        n_train_val = int(bounds[max(int(np.searchsorted(bounds, n_train_raw + n_val_raw, side="right")) - 1, 1)])
        n_val = n_train_val - n_train
        print(f"  [day_close] 切分点已吸附到交易日边界: train={n_train} val={n_val} test={n - n_train - n_val}")

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    # 多频率对齐：1小时线也按同样比例切分
    train_hourly = val_hourly = test_hourly = None
    if hourly_df is not None:
        nh = len(hourly_df)
        nh_train = int(nh * train_ratio)
        nh_val = int(nh * val_ratio)
        train_hourly = hourly_df.iloc[:nh_train].reset_index(drop=True)
        val_hourly = hourly_df.iloc[nh_train : nh_train + nh_val].reset_index(drop=True)
        test_hourly = hourly_df.iloc[nh_train + nh_val :].reset_index(drop=True)

    # 取消标准化：所有数据使用原始价格（模型直接预测原始 close_delta）
    train_ds = KLineDataset(train_df, seq_len=seq_len, normalize=False, hourly_df=train_hourly, **kwargs)
    # warm-up（老师三轮 #2/#4 修复）：val/test 的数据集建在含历史的完整 df 上，
    # 再按"窗口末 bar ≥ 切分点"截出本段样本。边界样本由此看到真实的日K/分钟历史
    # （与生产一致），但标签与评估样本不越界；训练集构造不变（权重不失效）。
    val_ds = KLineDataset(df.iloc[: n_train + n_val].reset_index(drop=True),
                          seq_len=seq_len, normalize=False, hourly_df=val_hourly, **kwargs)
    # 非 day_close 模式（horizon classify 等）：标签向右延 target_offset 根，
    # val 末尾样本的标签不能偷看 test 段 → max_label_bar = val 段末 bar
    val_max_label = None if kwargs.get("label_mode") == "day_close" else n_train + n_val - 1
    _restrict_samples(val_ds, n_train, max_label_bar=val_max_label)
    test_ds = KLineDataset(df, seq_len=seq_len, normalize=False, hourly_df=test_hourly, **kwargs)
    _restrict_samples(test_ds, n_train + n_val)

    # 统一 mean/std 引用（均为 0/1），保证 inverse_transform 兼容
    val_ds.mean = train_ds.mean.copy()
    val_ds.std = train_ds.std.copy()
    test_ds.mean = train_ds.mean.copy()
    test_ds.std = train_ds.std.copy()
    if hourly_df is not None:
        val_ds.hourly_mean = train_ds.hourly_mean.copy()
        val_ds.hourly_std = train_ds.hourly_std.copy()
        test_ds.hourly_mean = train_ds.hourly_mean.copy()
        test_ds.hourly_std = train_ds.hourly_std.copy()

    return train_ds, val_ds, test_ds


def _restrict_samples_by_dayset(ds, day_set: set[int], tag: str) -> None:
    """按"窗口末 bar 所属交易日 id ∈ day_set"截样本（合约模式切分用）。
    labels/fwd_rets/touch_minutes/thetas/soft_targets 与 valid_indices 同步过滤。
    T+N 锚（next_close）：锚日也必须在 day_set 内——否则段末样本的标签
    偷看下一段的收盘价（第二道泄露闸，第一道是构造时锚日超段自然剔除）。"""
    base = ds.valid_indices + ds.seq_len - 1
    dayord = ds.day_ids_global[base]
    keep = np.array([int(d) in day_set for d in dayord], dtype=bool)
    ad = getattr(ds, "anchor_days", None)
    if ad is not None:
        keep &= np.array([int(x) in day_set for x in ad], dtype=bool)
    ds.valid_indices = ds.valid_indices[keep]
    ds.n_samples = len(ds.valid_indices)
    for attr in ("labels", "fwd_rets", "touch_minutes", "thetas", "soft_targets", "path_states", "exc_labels", "anchor_days"):
        arr = getattr(ds, attr, None)
        if arr is not None:
            setattr(ds, attr, arr[keep])
    if ds.n_samples <= 0:
        raise ValueError(f"[contract] {tag} 截样后无样本，检查切分比例或数据量")


def build_datasets_contract(
    code: str,
    period: int,
    seq_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    data_dir=None,
    **kwargs,
) -> tuple:
    """合约模式切分（做法 B）：段帧拼接 + 按样本锚交易日 70/15/15 切分。

    与 build_datasets 的差异：
    - 数据来自 contract_series.build_contract_frame（各合约完整生命周期拼接），
      不是主连 CSV；窗口/标签零跨合约污染；
    - 只支持 day_close 标签模式（v1）；
    - 三分割按"样本锚交易日序数"切（段内编号全局递增，时序有序），
      天然吸附日边界；val/test 建在完整段帧上（等价 warm-up，边界样本可见历史）。

    :return: (train_ds, val_ds, test_ds)
    """
    import copy as _copy

    from obson.contract_series import build_contract_frame

    if kwargs.get("label_mode") != "day_close":
        raise ValueError("build_datasets_contract v1 只支持 label_mode='day_close'")

    df, extra_breaks, main_mask, day_ids = build_contract_frame(code, period, data_dir or _contract_default_dir())
    full = KLineDataset(
        df, seq_len=seq_len, normalize=False,
        jump_break_pct=None,  # 段边界已由 extra_breaks 显式给出
        extra_breaks=extra_breaks, sample_mask=main_mask, day_ids_override=day_ids,
        **kwargs,
    )

    # 按样本锚交易日序数切分（day_ids 段内编号、跨段递增 → 时序有序）
    base = full.valid_indices + seq_len - 1
    sample_dayord = full.day_ids_global[base]
    uniq = np.unique(sample_dayord)
    n_tr = int(len(uniq) * train_ratio)
    n_va = int(len(uniq) * val_ratio)
    tr_set = set(uniq[:n_tr].tolist())
    va_set = set(uniq[n_tr:n_tr + n_va].tolist())
    te_set = set(uniq[n_tr + n_va:].tolist())
    print(f"  [contract] 交易日切分: train={n_tr} 日 / val={n_va} 日 / test={len(uniq) - n_tr - n_va} 日")

    train_ds = _copy.copy(full)
    _restrict_samples_by_dayset(train_ds, tr_set, "train")
    val_ds = _copy.copy(full)
    _restrict_samples_by_dayset(val_ds, va_set, "val")
    test_ds = _copy.copy(full)
    _restrict_samples_by_dayset(test_ds, te_set, "test")

    # mean/std 互抄保持 inverse_transform 兼容（normalize=False 时为 0/1）
    for ds in (val_ds, test_ds):
        ds.mean = train_ds.mean.copy()
        ds.std = train_ds.std.copy()
    return train_ds, val_ds, test_ds


def _contract_default_dir():
    from obson.contract_series import CONTRACT_DIR
    return CONTRACT_DIR


def weekly_seq_len(df: pd.DataFrame, trading_days: int = 5, cap: int = 448) -> int:
    """估算"一个交易周"对应的 bar 数：单交易日 bar 数中位数 × trading_days

    按 trading_day_ids（夜盘归次日）统计，与自然日口径的区别：夜盘 bar 不再被
    算到前一个自然日，各品种/时段的实际回看天数一致（老师三轮 #3 修复）。

    :param df: 含 datetime 列的分钟线 DataFrame
    :param trading_days: 一周交易天数（国内期货 5 天）
    :param cap: 上限（防止脏数据把 seq_len 撑爆显存）
    :return: 建议的 seq_len
    """
    ts = pd.to_datetime(df["datetime"])
    per_day = pd.Series(trading_day_ids(ts)).value_counts()
    if len(per_day) == 0:
        raise ValueError("无法估算周窗口：数据为空")
    median = int(per_day.median())
    return min(median * trading_days, cap)
