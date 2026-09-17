# -*- coding: utf-8 -*-
"""生产作战手册 v2 —— 唯一权威策略定义（single source of truth）

signal_live.py / backtest_playbook.py / backtest_signals.py 必须从这里读规则，
禁止各自维护。规则 = 品种×周期×方向 × 置信度区间。

规则来源：2026-09-11 置信度分桶 EV 表（注意：该表曾用 val+test 混合选出本规则，
存在选择偏差；自本模块落地起，规则冻结，此后测试集只做最终评估）。
"""
from __future__ import annotations

# (品种, 周期分钟) -> {"long"/"short": (信心下限, 信心上限)}，上限为开区间
# 信心 = 集成模型对该方向的概率。不在表里的 品种×周期×方向 = 不做。
RULES: dict[tuple[str, int], dict[str, tuple[float, float]]] = {
    ("p", 60): {"long": (0.35, 0.50), "short": (0.50, 1.01)},
    ("sr", 60): {"long": (0.40, 0.50)},
    # 2026-09-11 转正：验证集选出→测试集一次性独立评估过关（过线标准：摸轨率≥2×基准且n≥30）
    ("m", 60): {"short": (0.40, 0.50)},   # 测试集 n=78 摸轨率15.4%（基准4.6%，3.3x）E=+0.106%
    ("m", 30): {"short": (0.40, 0.45)},   # 测试集 n=45 摸轨率13.3%（基准3.0%，4.4x）E=+0.183%
}

# 方向过滤（旧手册 v1，仅品种×方向，无置信度区间；backtest_signals --playbook 用）
PLAYBOOK_V1 = {
    "p_60m": "both", "p_30m": "short", "sr_60m": "long",
    "y_30m": "short", "m_60m": "short",
}

# 候选规则区（验证集选出，待测试集一次性独立评估）
# 纪律：只准被 confidence_ev_table.py 读取做【一次性】测试集评估；
#       decide()/signal_mask()/任何回测与实盘脚本禁止使用。
#       测试集评估过关 → 移入 RULES；不过关 → 删除且不得以相同区间重试。
# （2026-09-11：m_60m/m_30m 空头候选双双过关，已转正，候选区清空）
CANDIDATE_RULES: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}

# 黑名单（任何方向都不做）
BLACKLIST = {("i", 30), ("j", 30), ("MA", 30), ("ag", 30)}

# 双边成本（%，唯一权威定义；回测脚本统一从这里取）
COST_RT = {"rb": 0.06, "hc": 0.06, "i": 0.13, "sr": 0.035, "p": 0.05, "j": 0.05, "jm": 0.06,
           "m": 0.05, "y": 0.05, "cu": 0.05, "ag": 0.08, "TA": 0.05, "MA": 0.05}


def rule_for(code: str, period: int) -> dict[str, tuple[float, float]]:
    return RULES.get((code, period), {})


def decide(code: str, period: int, prob, thr) -> tuple[str, str]:
    """手册 v2 判定。prob=(p空,p无,p多)，thr={0:空阈值,2:多阈值}（验证集标定）。
    返回 (动作, 理由)。动作 ∈ {开多, 开空, 观望}。"""
    if (code, period) in BLACKLIST:
        return "观望", "黑名单品种"
    rules = RULES.get((code, period))
    if not rules:
        return "观望", "手册外品种"
    p_short, p_long = float(prob[0]), float(prob[2])
    if "long" in rules:
        lo, hi = rules["long"]
        if p_long >= max(thr.get(2, 0.0), lo) and p_long < hi:
            return "开多", f"信心{p_long:.1%}∈[{lo:.0%},{hi:.0%})且过阈值{thr.get(2, 0):.2f}"
    if "short" in rules:
        lo, hi = rules["short"]
        if p_short >= max(thr.get(0, 0.0), lo) and p_short < hi:
            return "开空", f"信心{p_short:.1%}∈[{lo:.0%},{hi:.0%})且过阈值{thr.get(0, 0):.2f}"
    return "观望", "信心未进白名单区间"


def signal_mask(code: str, period: int, prob, thr_long: float, thr_short: float):
    """回测用：返回 (sig_long, sig_short) 布尔数组，已按手册 v2 过滤。"""
    import numpy as np
    n = len(prob)
    sig_long = np.zeros(n, bool)
    sig_short = np.zeros(n, bool)
    if (code, period) in BLACKLIST:
        return sig_long, sig_short
    rules = RULES.get((code, period), {})
    if "long" in rules:
        lo, hi = rules["long"]
        sig_long = (prob[:, 2] >= max(thr_long, lo)) & (prob[:, 2] < hi)
    if "short" in rules:
        lo, hi = rules["short"]
        sig_short = (prob[:, 0] >= max(thr_short, lo)) & (prob[:, 0] < hi)
    return sig_long, sig_short
