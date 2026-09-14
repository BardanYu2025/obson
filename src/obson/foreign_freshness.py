# -*- coding: utf-8 -*-
"""外盘数据新鲜度检查（P0-A，2026-09-11 老师审查意见落地）

三态语义（fail-closed，不做未经验证的零特征降级）：
  fresh       外盘数据在允许延迟内，正常使用
  stale       最新日期距今超过 max_gap_days 个日历日，禁止该品种出信号
  unavailable 文件缺失/为空/无法解析/数据非法，禁止该品种出信号
  none        该品种无外盘映射，不受本检查影响

外盘市场（BMD/SGX/COMEX/CBOT/ICE）节假日与中国不同步，
故允许的最大滞后按"日历日缺口"计（默认 5 天，覆盖周末+小长假），
不按固定 1~2 天，避免节假日误锁。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 与 train_multi_symbol.FOREIGN_MAP 保持一致（此处复制以避免脚本层反向依赖）
FOREIGN_MAP = {"sr": "ice_sugar", "p": "bmd_palm", "y": "cbot_beanoil",
               "m": "cbot_soybean", "rb": "sgx_iron", "i": "sgx_iron",
               "ag": "comex_silver"}

MAX_GAP_DAYS = 5  # 周末+海外小长假的最大日历缺口


def check_foreign_freshness(code: str, foreign_dir: str | Path = "data/foreign",
                            now: pd.Timestamp | None = None,
                            max_gap_days: int = MAX_GAP_DAYS) -> tuple[str, str]:
    """返回 (status, reason)。status ∈ {none, fresh, stale, unavailable}。"""
    name = FOREIGN_MAP.get(code)
    if name is None:
        return "none", "无外盘映射"
    fp = Path(foreign_dir) / f"{name}_1d.csv"
    if not fp.exists():
        return "unavailable", f"缺文件 {fp}"
    try:
        df = pd.read_csv(fp)
    except Exception as e:  # 解析失败
        return "unavailable", f"无法解析 {fp}: {e}"
    if len(df) == 0:
        return "unavailable", f"空文件 {fp}"
    if not {"date", "close"} <= set(df.columns):
        return "unavailable", f"{fp} 缺 date/close 列"
    if (df["close"] <= 0).any() or df["close"].isna().any():
        return "unavailable", f"{fp} close 存在非正/缺失值"
    if not df["date"].is_monotonic_increasing:
        return "unavailable", f"{fp} 日期非单调递增"
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    last_date = pd.to_datetime(str(int(df["date"].iloc[-1])), format="%Y%m%d")
    gap = (now.normalize() - last_date).days
    if gap > max_gap_days:
        return "stale", f"最新外盘日期 {last_date.date()}，距今 {gap} 天 > {max_gap_days} 天"
    return "fresh", f"最新 {last_date.date()}，滞后 {gap} 天"


def freshness_report(codes, foreign_dir: str | Path = "data/foreign",
                     now: pd.Timestamp | None = None) -> dict[str, tuple[str, str]]:
    """批量检查，返回 {code: (status, reason)}，仅含有外盘映射的品种。"""
    return {c: check_foreign_freshness(c, foreign_dir, now)
            for c in codes if c in FOREIGN_MAP}
