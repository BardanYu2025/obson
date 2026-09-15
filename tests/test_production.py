# -*- coding: utf-8 -*-
"""生产链路回归测试（老师审查 #7 补齐）

覆盖 7 条：
  1. signal_live 末 bar 口径：last_complete_bar 判定 + 输入末 bar == 信号 bar
  2. valid_indices → df 索引映射（窗口末 = idx+seq_len-1）
  3. 单仓模式：持仓中同向信号跳过、不重叠开仓
  4. 同根 bar 止盈止损双触 → 记止损
  5. 手册 v2 置信度区间边界（0.349/0.35/0.50/0.51）
  6. EV 表协议：标定只用验证集（源码级断言：不再合并 val+test）
  7. 外盘上下文严格因果：只允许 ≤ c-1 日的外盘 bar
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from obson.model.dataset import KLineDataset  # noqa: E402
from obson.playbook import decide  # noqa: E402
from signal_live import last_complete_bar  # noqa: E402
from backtest_playbook import simulate_trade, single_position_backtest  # noqa: E402


def _mk_df(n=80, start="2026-09-07 09:00", freq="60min", base=3000.0):
    ts = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(0)
    close = base + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "datetime": ts,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
    })


# ── 1. 末 bar 口径 ──────────────────────────────────────────────
def test_last_complete_bar():
    df = _mk_df()  # 60m bars, 最后一根 2026-09-10 12:00 开始
    ts_last = df["datetime"].iloc[-1]
    # 末根已收盘（now 超过 起始+60min）→ j* = len-1
    j, dur = last_complete_bar(df, ts_last + pd.Timedelta(minutes=61))
    assert j == len(df) - 1 and dur == 60.0
    # 末根仍在走（now = 起始+30min）→ j* = len-2
    j2, _ = last_complete_bar(df, ts_last + pd.Timedelta(minutes=30))
    assert j2 == len(df) - 2


def test_live_input_end_equals_signal_bar():
    """数据集最后样本的输入窗口末 bar 时间必须 == 信号打印的 bar 时间（j*）"""
    df = _mk_df(n=40)
    seq_len = 10
    j_star, _ = last_complete_bar(df, df["datetime"].iloc[-1] + pd.Timedelta(minutes=30))
    df_live = df.iloc[: j_star + 2]
    ds = KLineDataset(df_live, seq_len=seq_len, target_offset=1,
                      normalize=False, jump_break_pct=None)
    need_idx = j_star - seq_len + 1
    if int(ds.valid_indices[-1]) != need_idx:
        ds.valid_indices = np.append(ds.valid_indices, need_idx)
        ds.n_samples += 1
    in_last = ds.timestamps.iloc[int(ds.valid_indices[-1]) + seq_len - 1]
    assert in_last == df["datetime"].iloc[j_star]


# ── 2. valid_indices 映射 ───────────────────────────────────────
def test_valid_indices_mapping():
    df = _mk_df(n=50)
    seq_len = 8
    ds = KLineDataset(df, seq_len=seq_len, target_offset=1,
                      normalize=False, jump_break_pct=None)
    item = ds[len(ds) - 1]
    idx = int(ds.valid_indices[-1])
    # seq 最后一行 close 与原始 df 窗口末一致（normalize=False；float32 精度容差 0.01）
    assert abs(float(item["seq"][-1, 3]) - float(df["close"].iloc[idx + seq_len - 1])) < 0.01
    # 窗口末行索引 == valid_indices + seq_len - 1（映射不自漂移）
    assert idx + seq_len - 1 <= len(df) - 1


# ── 3. 单仓不重叠 ───────────────────────────────────────────────
def _mk_candidate(ts, code, direction, j, a, cl_exit_flat=True):
    n = a + 1
    hi = np.full(n, 101.0)
    lo = np.full(n, 99.0)
    cl = np.full(n, 100.0)
    ts_index = pd.date_range("2026-09-07 09:00", periods=n, freq="60min").to_numpy()
    return {"ts": pd.Timestamp(ts), "code": code, "key": f"{code}_60m",
            "direction": direction, "hi": hi, "lo": lo, "cl": cl,
            "ts_index": ts_index, "j": j, "a": a, "th": 0.01,
            "tp_frac": 0.8, "stop_frac": 0.5, "step_min": 60.0}


def test_single_position_no_overlap():
    # 候选1：9:00 开多（价格不动 → 拖到锚 a=5 收盘平仓，占用到 ts_index[5]=14:00）
    c1 = _mk_candidate("2026-09-07 09:00", "p", 1, j=0, a=5)
    # 候选2：10:00 同向信号 → 应被跳过（持仓中）
    c2 = _mk_candidate("2026-09-07 10:00", "p", 1, j=1, a=5)
    # 候选3：15:00（仓位已了结）→ 应开仓
    c3 = _mk_candidate("2026-09-07 15:00", "p", 1, j=0, a=5)
    trades = single_position_backtest([c1, c2, c3], {"p": 0.05})
    assert len(trades) == 2, f"应成交2笔（跳过重叠的1笔），实际 {len(trades)}"
    assert trades[0]["ts"] == pd.Timestamp("2026-09-07 09:00")
    assert trades[1]["ts"] == pd.Timestamp("2026-09-07 15:00")


def test_single_position_flip_close():
    # 持仓多时遇到反向信号 → 按信号 bar 收盘平（flip_close），不反手
    c1 = _mk_candidate("2026-09-07 09:00", "p", 1, j=0, a=5)
    c2 = _mk_candidate("2026-09-07 10:00", "p", -1, j=1, a=5)
    trades = single_position_backtest([c1, c2], {"p": 0.05})
    assert len(trades) == 1
    assert trades[0]["exit"] == "flip_close"
    # 多仓 100 进场，c2 的 cl[j]=100 平 → pnl = -cost
    assert abs(trades[0]["pnl"] - (-0.05)) < 1e-9


# ── 4. 同根双触记止损 ───────────────────────────────────────────
def test_double_touch_counts_as_stop():
    # entry=100, θ=1% → tp=100.8, stop=99.5；下一根 bar 高低 [99, 101] 双触
    hi = np.array([100.0, 101.0])
    lo = np.array([100.0, 99.0])
    cl = np.array([100.0, 100.2])
    pnl, ex, nb, mae, mfe, b = simulate_trade(hi, lo, cl, j=0, a=1, direction=1,
                                              tp_frac=0.8, stop_frac=0.5, th=0.01)
    assert ex == "stop(同根双触)"
    assert abs(pnl - (-0.005)) < 1e-9  # 记止损价 −0.5%


# ── 5. 手册 v2 置信度边界 ───────────────────────────────────────
def test_playbook_decide_boundaries():
    thr = {2: 0.35, 0: 0.40}
    # p_60m 多白名单 [0.35, 0.50)
    assert decide("p", 60, [0.2, 0.2, 0.349], thr)[0] == "观望"   # 低于下限
    assert decide("p", 60, [0.2, 0.2, 0.350], thr)[0] == "开多"   # 恰在下限
    assert decide("p", 60, [0.2, 0.2, 0.499], thr)[0] == "开多"
    assert decide("p", 60, [0.2, 0.2, 0.500], thr)[0] == "观望"   # 上限开区间
    # p_60m 空白名单 [0.50, 1.01)
    assert decide("p", 60, [0.50, 0.2, 0.2], thr)[0] == "开空"
    assert decide("p", 60, [0.49, 0.2, 0.2], thr)[0] == "观望"
    # 手册外品种一律观望
    assert decide("rb", 60, [0.1, 0.1, 0.9], thr)[0] == "观望"
    # 黑名单
    assert decide("i", 30, [0.9, 0.05, 0.05], thr)[0] == "观望"


# ── 6. EV 表协议（源码级）──────────────────────────────────────
def test_ev_table_protocol_val_only():
    src = (ROOT / "scripts" / "confidence_ev_table.py").read_text(encoding="utf-8")
    assert "ConcatDataset" not in src, "EV 表不得再合并 val+test"
    assert "confidence_ev_frozen_test" in src, "冻结规则必须在测试集单独成表"
    assert "confidence_ev_table_val" in src


# ── 6b. 手册规则唯一权威（防止脚本重新自维护 PLAYBOOK 副本）─────
def test_playbook_single_source_of_truth():
    import re
    for name in ("backtest_playbook.py", "signal_live.py"):
        src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "from obson.playbook import" in src, f"{name} 必须从 obson.playbook 导入手册"
        # 禁止本地再定义规则字典（PLAYBOOK = { / RULES = {）
        assert not re.search(r"^\s*(PLAYBOOK|RULES)\s*=\s*\{", src, re.M), \
            f"{name} 不得自维护手册副本"
    # signal_mask 必须被回测脚本真正使用（v2 接线）；
    # backtest_signals.py 已于 2026-09-14 随清理删除，此后再引入须同样接线 v2
    for name in ("backtest_playbook.py",):
        src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "signal_mask(" in src, f"{name} 未接入 signal_mask（手册 v2）"
        assert "--playbook-v2" in src, f"{name} 缺 --playbook-v2 入口"
    assert not (ROOT / "scripts" / "backtest_signals.py").exists(), \
        "backtest_signals.py（v1 残留）不得复活"


# ── 7. 外盘严格因果 ─────────────────────────────────────────────
def test_foreign_context_strictly_past():
    df = _mk_df(n=40, start="2026-09-07 09:00", freq="60min")
    seq_len = 8
    # 外盘 7 天：最后一天 09-08 = 基准bar当日，必须被排除（若泄露，log_ret 会出现大值）
    foreign = pd.DataFrame({
        "date": [20260831, 20260901, 20260902, 20260903, 20260904, 20260907, 20260908],
        "close": [100.0, 100.5, 99.8, 100.2, 100.1, 100.3, 200.0],
    })
    ds = KLineDataset(df, seq_len=seq_len, target_offset=1, normalize=False,
                      jump_break_pct=None, foreign_close=foreign, foreign_bars=5)
    # 取窗口末落在 09-08 的样本
    item = None
    for i in range(len(ds)):
        idx = int(ds.valid_indices[i])
        base_ts = df["datetime"].iloc[idx + seq_len - 1]
        if base_ts.date() == pd.Timestamp("2026-09-08").date():
            item = ds[i]
            break
    assert item is not None and "foreign_ctx" in item
    fctx = item["foreign_ctx"].numpy()
    # 09-08 当天外盘必须排除：真实行的 log_ret（第0列）都 ≤0.5%；
    # 若泄露 09-08（100.3→200，log_ret≈0.69）则必超阈值
    assert np.abs(fctx[:, 0]).max() < 0.1, f"疑似泄露了当日外盘: {fctx}"


# ── 8. weekly_seq_len 口径统一后数值不变（26 组合全部锁定）─────
def test_weekly_seq_len_trading_day_based():
    """检查表 tests/seq_len_check.csv 由 2026-09-11 实测生成（两口径中位数+seq_len）。
    任何组合 seq_len 偏离表中 new_seq_len = 输入定义变化 = 必须重训并更新检查表。"""
    from obson.model.dataset import weekly_seq_len
    table = pd.read_csv(ROOT / "tests" / "seq_len_check.csv")
    assert len(table) == 26, "检查表应覆盖 26 个品种×周期组合"
    checked = 0
    for _, r in table.iterrows():
        p = ROOT / "data" / f"{r['symbol']}_{r['period']}m.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["datetime"])
        got = weekly_seq_len(df)
        assert got == int(r["new_seq_len"]), \
            f"{r['symbol']}_{r['period']}m: seq_len={got} != 检查表 {r['new_seq_len']}（变了就要重训！）"
        assert not bool(r["changed"]), f"{r['symbol']}_{r['period']}m 检查表标记了口径变化"
        checked += 1
    if checked == 0:
        print("  [skip] 无本地数据文件（如 CI/新克隆环境），跳过实测核验")
        return
    assert checked >= 20, f"实际核验组合数不足：{checked}/26"


# ── 9. 同根双触：训练记"无"（两阶段语义的一致面）───────────────
def test_double_touch_label_is_none():
    """训练标签：同根 bar 双触上下轨 → 无信号（不奖励任何方向）；
    执行侧记止损由 test_double_touch_counts_as_stop 覆盖。两侧合起来 =
    "不知道就不开仓；已开仓则保守结算"（老师三轮 #2：保留该语义，不改标签定义）"""
    # 一个交易日：09,10,11,14,15 五根 60m bar；15:00 为收盘锚
    ts = pd.to_datetime(["2026-09-08 09:00", "2026-09-08 10:00", "2026-09-08 11:00",
                         "2026-09-08 14:00", "2026-09-08 15:00"])
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.2, 100.2, 100.2, 100.2, 102.0])  # 末根上穿 +1% 轨
    low = np.array([99.8, 99.8, 99.8, 99.8, 97.5])        # 末根下破 −1% 轨 → 双触
    df = pd.DataFrame({"datetime": ts, "open": close, "high": high, "low": low, "close": close})
    ds = KLineDataset(df, seq_len=3, target_offset=1, normalize=False,
                      jump_break_pct=None, label_mode="day_close",
                      label_threshold=1.0, theta_mode="frozen")
    assert ds.n_samples >= 1
    assert (ds.labels == 1).all(), f"同根双触必须全部记无信号，实际 labels={ds.labels}"


# ── 10. warm-up：val/test 看到切分点前的历史，样本不越界 ───────
def test_warmup_val_test_see_history():
    from obson.model import build_datasets
    # 20 个交易日 × 每天 5 根 60m bar（锚在 15:00，小时∈[13,15] 合法）
    days = pd.bdate_range("2026-08-10", periods=20)
    hours = ["09:00", "10:00", "11:00", "14:00", "15:00"]
    ts, close = [], []
    px = 100.0
    rng = np.random.default_rng(1)
    for d in days:
        for h in hours:
            ts.append(pd.Timestamp(f"{d.date()} {h}"))
            px += rng.normal(0, 0.2)
            close.append(px)
    df = pd.DataFrame({"datetime": ts, "open": close, "high": np.array(close) + 0.3,
                       "low": np.array(close) - 0.3, "close": close})
    seq_len = 8
    train_ds, val_ds, test_ds = build_datasets(
        df, seq_len=seq_len, target_offset=1, train_ratio=0.6, val_ratio=0.2,
        split_mode="time", label_mode="day_close", label_threshold=0.8,
        theta_mode="frozen", jump_break_pct=None, daily_bars=3)
    n = len(df)
    n_tr = int(n * 0.6)
    # 吸附交易日边界后的真实切点：训练集最后一天结束处
    tr_last_ts = train_ds.timestamps.iloc[-1]
    split_bar = int(np.flatnonzero(df["datetime"].to_numpy() > tr_last_ts)[0])
    # val_ds 建在含历史的完整前缀上
    assert len(val_ds.timestamps) > n - split_bar, "val 应包含训练段历史（warm-up）"
    # 所有 val 样本窗口末 ≥ 切点；且存在窗口起点 < 切点的边界样本（旧口径下不存在）
    bases = val_ds.valid_indices + seq_len - 1
    assert bases.min() >= split_bar, "val 样本越界进入训练段"
    starts = val_ds.valid_indices
    assert starts.min() < split_bar, "边界样本应能回看训练段历史（warm-up 生效）"
    # test 同理
    bases_t = test_ds.valid_indices + seq_len - 1
    assert bases_t.min() >= bases.max() or bases_t.min() > split_bar


# ── 11. horizon classify 模式：val 末尾样本标签不偷看 test 段 ───
def test_warmup_horizon_label_no_peek():
    from obson.model import build_datasets
    n = 120
    df = _mk_df(n=n, start="2026-08-10 09:00", freq="60min")
    seq_len, offset = 8, 4
    train_ds, val_ds, test_ds = build_datasets(
        df, seq_len=seq_len, target_offset=offset, train_ratio=0.6, val_ratio=0.2,
        split_mode="time", label_mode="classify", label_threshold=0.5,
        jump_break_pct=None)
    n_tr = int(n * 0.6)
    val_end = n_tr + int(n * 0.2) - 1  # val 段最后一根 bar
    bases = val_ds.valid_indices + seq_len - 1
    assert bases.min() >= n_tr, "val 样本窗口末越界进入训练段"
    assert (bases + offset).max() <= val_end, \
        f"horizon 模式 val 标签偷看 test 段: max base+offset={(bases + offset).max()} > {val_end}"
    # warm-up 生效：存在窗口起点在训练段的边界样本
    assert val_ds.valid_indices.min() < n_tr


def test_candidate_rules_isolated_from_production():
    """候选规则区不得影响生产判定/回测/实盘路径。"""
    from obson.playbook import RULES, CANDIDATE_RULES, decide, signal_mask
    import numpy as np
    # 候选区与生产规则区不得重叠（防双写漂移）
    for k in CANDIDATE_RULES:
        assert k not in RULES, f"候选规则 {k} 已在生产 RULES 中，应先从候选区移除"
    # 候选品种在生产路径里必须是"手册外品种"
    for (code, period), rules in CANDIDATE_RULES.items():
        prob = (0.45, 0.10, 0.45)  # 双方向高信心，足以触发任何候选
        action, reason = decide(code, period, prob, {0: 0.0, 2: 0.0})
        assert action == "观望" and "手册外" in reason, \
            f"候选规则 {code}_{period}m 泄漏进生产 decide(): {action} {reason}"
        sig_l, sig_s = signal_mask(code, period, np.array([prob]), 0.0, 0.0)
        assert not sig_l.any() and not sig_s.any(), \
            f"候选规则 {code}_{period}m 泄漏进 signal_mask()"


def _mk_foreign_csv(dirp: Path, name: str, rows: list[tuple[int, float]]):
    fp = dirp / f"{name}_1d.csv"
    with open(fp, "w") as f:
        f.write("date,close\n")
        for d, c in rows:
            f.write(f"{d},{c}\n")
    return fp


def test_foreign_freshness_states(tmp_path=None):
    """外盘新鲜度三态 + 周末不误判 + 无映射不受影响。"""
    import tempfile
    from obson.foreign_freshness import check_foreign_freshness, MAX_GAP_DAYS
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # fresh：最新=今天
        _mk_foreign_csv(d, "bmd_palm", [(20260909, 4968.0), (20260911, 4884.0)])
        st, _ = check_foreign_freshness("p", d, now=pd.Timestamp("2026-09-11 21:00"))
        assert st == "fresh", st
        # 周末缺口不误判：最新=周五，now=下周一（3 天 ≤ MAX_GAP_DAYS）
        _mk_foreign_csv(d, "sgx_iron", [(20260904, 100.0)])  # 2026-09-04 是周五
        st, _ = check_foreign_freshness("rb", d, now=pd.Timestamp("2026-09-07 21:00"))
        assert st == "fresh", st
        # stale：10 天未更新
        _mk_foreign_csv(d, "comex_silver", [(20260901, 40.0)])
        st, r = check_foreign_freshness("ag", d, now=pd.Timestamp("2026-09-11 09:00"))
        assert st == "stale" and "10 天" in r, (st, r)
        # unavailable：文件缺失 / 空文件 / close 非正 / 日期乱序
        st, _ = check_foreign_freshness("m", d, now=pd.Timestamp("2026-09-11"))
        assert st == "unavailable", st  # cbot_soybean 文件不存在
        _mk_foreign_csv(d, "cbot_soybean", [])
        open(d / "cbot_soybean_1d.csv", "w").write("date,close\n")
        st, _ = check_foreign_freshness("m", d, now=pd.Timestamp("2026-09-11"))
        assert st == "unavailable", st  # 空文件
        _mk_foreign_csv(d, "ice_sugar", [(20260910, 15.0), (20260911, -1.0)])
        st, _ = check_foreign_freshness("sr", d, now=pd.Timestamp("2026-09-11"))
        assert st == "unavailable", st  # close 非正
        _mk_foreign_csv(d, "cbot_beanoil", [(20260911, 55.0), (20260910, 54.0)])
        st, _ = check_foreign_freshness("y", d, now=pd.Timestamp("2026-09-11"))
        assert st == "unavailable", st  # 日期乱序
        # none：无外盘映射品种不受影响
        st, _ = check_foreign_freshness("hc", d, now=pd.Timestamp("2026-09-11"))
        assert st == "none", st
        # 边界：恰好 MAX_GAP_DAYS 天不算过期
        _mk_foreign_csv(d, "bmd_palm", [(20260906, 4900.0)])
        st, _ = check_foreign_freshness("p", d, now=pd.Timestamp(f"2026-{9:02d}-{6 + MAX_GAP_DAYS:02d}"))
        assert st == "fresh", st


# ── 合约模式（做法 B）────────────────────────────────────────────
def _mk_contract_dir(code="xx", n_days=30, seed0=1):
    """造两合约合成数据：A/B 各覆盖全部 n_days 个工作日（60m×5根/日），
    A 主力任期=前 2/3，B=后 1/3。返回临时目录路径。"""
    import tempfile
    dirp = Path(tempfile.mkdtemp())
    (dirp / code).mkdir(parents=True)
    days = pd.bdate_range("2026-06-01", periods=n_days)
    hours = [9, 10, 11, 14, 15]
    split = int(n_days * 2 / 3)
    cal_rows = []
    for contract, tenure_days, base in [("xx2601", days[:split], 3000.0),
                                        ("xx2605", days[split:], 4000.0)]:
        rng = np.random.default_rng(seed0 + (0 if contract == "xx2601" else 100))
        ts, closes = [], base
        rows = []
        for d in days:
            for h in hours:
                closes += rng.normal(0, 2)
                rows.append((pd.Timestamp(d) + pd.Timedelta(hours=h), closes))
        df = pd.DataFrame(rows, columns=["datetime", "close"])
        df["open"] = df["close"]; df["high"] = df["close"] + 1; df["low"] = df["close"] - 1
        df["volume"] = 100; df["open_oi"] = 1000; df["close_oi"] = 1000
        df = df[["datetime", "open", "high", "low", "close", "volume", "open_oi", "close_oi"]]
        df.to_csv(dirp / code / f"{contract}_60m.csv", index=False)
        for d in tenure_days:
            cal_rows.append({"date": d, "contract": contract, "volume": 100, "close_oi": 1000})
    pd.DataFrame(cal_rows).to_csv(dirp / f"{code}_roll_calendar.csv", index=False)
    return dirp


def _mk_contract_ds(dirp, code="xx", seq_len=12, **kw):
    from obson.contract_series import build_contract_frame
    df, brk, mask, did = build_contract_frame(code, 60, dirp)
    defaults = dict(label_mode="day_close", label_threshold=0.9,
                    theta_mode="dynamic", normalize=False, jump_break_pct=None,
                    extra_breaks=brk, sample_mask=mask, day_ids_override=did)
    defaults.update(kw)
    ds = KLineDataset(df, seq_len=seq_len, target_offset=1, **defaults)
    return ds, df, brk, mask, did


def test_contract_no_sample_crosses_segment():
    """合约模式：样本窗口与标签路径都不许跨段"""
    dirp = _mk_contract_dir()
    ds, df, brk, mask, did = _mk_contract_ds(dirp)
    base = ds.valid_indices + ds.seq_len - 1
    # 窗口不跨段：窗口 (idx, base] 内无断点
    for b in brk:
        assert not ((ds.valid_indices < b) & (base >= b)).any(), f"窗口跨段 @{b}"
    # 标签锚（同交易日最后一根 bar）与基准 bar 同段：seg 列一致
    segs = df["seg"].to_numpy()
    for i, j in zip(ds.valid_indices, base):
        d = did[j]
        anchor = np.flatnonzero(did == d)[-1]
        assert segs[anchor] == segs[j], "收盘锚跨段"
    # 样本锚全部落在本段主力任期内（sample_mask 生效）
    assert mask[base].all()


def test_contract_label_parity_with_single_contract():
    """同一段内的样本，标签必须与"该合约独立建数据集"逐一对得上"""
    dirp = _mk_contract_dir()
    ds, df, brk, mask, did = _mk_contract_ds(dirp)
    # 段 B（seg=1）独立建数据集
    seg_b = df[df["seg"] == 1].reset_index(drop=True)
    ds_b = KLineDataset(seg_b, seq_len=ds.seq_len, target_offset=1,
                        label_mode="day_close", label_threshold=0.9,
                        theta_mode="dynamic", normalize=False, jump_break_pct=None)
    # 独立数据集基准 bar 时间 → 标签 映射
    base_b = ds_b.valid_indices + ds_b.seq_len - 1
    lab_by_ts = {seg_b["datetime"].iloc[j]: int(l) for j, l in zip(base_b, ds_b.labels)}
    # 合约模式里属于 seg B 的样本逐一比对
    n_cmp = 0
    for vi, l in zip(ds.valid_indices, ds.labels):
        j = int(vi) + ds.seq_len - 1
        if df["seg"].iloc[j] != 1:
            continue
        ts = df["datetime"].iloc[j]
        if ts in lab_by_ts:
            assert lab_by_ts[ts] == int(l), f"标签不一致 @{ts}: 单合约={lab_by_ts[ts]} 拼接={l}"
            n_cmp += 1
    assert n_cmp > 30, f"比对样本太少: {n_cmp}"


def test_contract_split_days_disjoint():
    """三分割按交易日切：日集合两两不交、合计覆盖全部样本日"""
    from obson.model.dataset import build_datasets_contract
    dirp = _mk_contract_dir()
    tr, va, te = build_datasets_contract(
        "xx", 60, seq_len=12, label_mode="day_close", label_threshold=0.9,
        theta_mode="dynamic", data_dir=dirp)
    day_sets = []
    for ds in (tr, va, te):
        base = ds.valid_indices + ds.seq_len - 1
        day_sets.append(set(ds.day_ids_global[base].tolist()))
    assert day_sets[0].isdisjoint(day_sets[1])
    assert day_sets[0].isdisjoint(day_sets[2])
    assert day_sets[1].isdisjoint(day_sets[2])
    # 时序有序：train 全部 < val 全部 < test 全部
    assert max(day_sets[0]) < min(day_sets[1]) < max(day_sets[1]) < min(day_sets[2])
    # 标签数组与 valid_indices 长度同步
    for ds in (tr, va, te):
        assert len(ds.labels) == len(ds.valid_indices) == ds.n_samples


# ── E3 路径状态辅助标签 ──────────────────────────────────────────
def _mk_pathstate_ds(highs, lows):
    """单日 5 根 60m bar（9/10/11/14/15 点），seq_len=1（样本 j 从 9:00 起），
    frozen θ=1%。样本 j=9:00 的 rem=4，节点=偏移 1/2/3/4 根。"""
    ts = pd.to_datetime(["2026-09-08 09:00", "2026-09-08 10:00", "2026-09-08 11:00",
                         "2026-09-08 14:00", "2026-09-08 15:00"])
    close = np.array([100.0] * 5)
    df = pd.DataFrame({"datetime": ts, "open": close,
                       "high": np.array(highs), "low": np.array(lows), "close": close})
    ds = KLineDataset(df, seq_len=1, target_offset=1, normalize=False,
                      jump_break_pct=None, label_mode="day_close",
                      label_threshold=1.0, theta_mode="frozen")
    return ds


def test_path_states_up_touch():
    """第 3 根（14:00）摸上轨 → 节点 1/2=未触轨，节点 3/4=已上轨（吸收态），主标签=多"""
    ds = _mk_pathstate_ds(highs=[100.2, 100.2, 100.2, 101.5, 100.2],
                          lows=[99.8] * 5)
    assert ds.n_samples >= 1 and ds.path_states is not None
    st = ds.path_states[0].tolist()
    assert st == [0, 0, 1, 1], f"路径状态错误: {st}"
    assert ds.labels[0] == 2
    item = ds[0]
    assert "path_states" in item and item["path_states"].tolist() == st


def test_path_states_double_touch_masked():
    """第 2 根双触 → 整行 -1 全 mask（OHLC 无法定先后，任何节点监督都可能错），主标签=无"""
    ds = _mk_pathstate_ds(highs=[100.2, 100.2, 101.5, 100.2, 100.2],
                          lows=[99.8, 99.8, 98.5, 99.8, 99.8])
    st = ds.path_states[0].tolist()
    assert st == [-1, -1, -1, -1], f"双触整行 mask 错误: {st}"
    assert ds.labels[0] == 1


def test_path_states_none_and_dn():
    """全程未触轨 → 全 0 且主标签=无；下破后保持已下轨"""
    ds_none = _mk_pathstate_ds(highs=[100.2] * 5, lows=[99.8] * 5)
    assert ds_none.path_states[0].tolist() == [0, 0, 0, 0]
    assert ds_none.labels[0] == 1
    ds_dn = _mk_pathstate_ds(highs=[100.2] * 5,
                             lows=[99.8, 99.8, 98.5, 99.0, 99.0])
    assert ds_dn.path_states[0].tolist() == [0, 2, 2, 2]
    assert ds_dn.labels[0] == 0


def test_path_aux_label_never_crosses_contract_segment():
    """教师模型点名：路径标签不得跨合约段边界。
    在日内 bar3 处注入段断点（换月切换），凡标签路径跨过断点的样本必须被剔除，
    剩余样本的基准 bar 全部 ≥ 断点位置。"""
    ts = pd.to_datetime(["2026-09-08 09:00", "2026-09-08 10:00", "2026-09-08 11:00",
                         "2026-09-08 14:00", "2026-09-08 15:00"])
    close = np.array([100.0] * 5)
    df = pd.DataFrame({"datetime": ts, "open": close, "high": close + 0.2,
                       "low": close - 0.2, "close": close})
    ds = KLineDataset(df, seq_len=1, target_offset=1, normalize=False,
                      jump_break_pct=None, label_mode="day_close",
                      label_threshold=1.0, theta_mode="frozen",
                      extra_breaks=[3])  # bar3 是新段的开始
    base = ds.valid_indices + ds.seq_len - 1
    assert (base >= 3).all(), f"存在跨段样本: base={base.tolist()}"
    # 断点后的样本路径状态合法（无 -1 残留歧义，因为该段内无双触）
    assert (ds.path_states != -1).all()


def test_path_loss_per_node_class_weights():
    """E3 v1.1：逐节点类别权重生效——少数类样本的梯度贡献应显著大于不加权情形"""
    import torch
    from obson.model.mixed_trainer import MixedFrequencyTrainer
    from obson.model.transformer import KLineConfig, KLineTransformer
    cfg = KLineConfig(task="classify", path_aux=True)
    model = KLineTransformer(cfg)
    model.eval()
    tr = MixedFrequencyTrainer.__new__(MixedFrequencyTrainer)
    tr.model, tr.device = model, "cpu"
    torch.manual_seed(0)
    pl = torch.randn(8, 4, 3)
    ps = torch.zeros(8, 4, dtype=torch.long)  # 全 0：多数类
    ps[0, 0] = 1                               # 唯一少数类
    l_now = tr._path_loss(pl, ps)
    cfg2 = KLineConfig(task="classify", path_aux=True,
                       path_state_weights=[0.1, 5.0, 5.0] * 4)
    tr.model = KLineTransformer(cfg2)
    tr.model.config.path_state_weights = [0.1, 5.0, 5.0] * 4
    l_w = tr._path_loss(pl, ps)
    assert l_w.item() != l_now.item()
    # 加权后总损失应向少数类误差倾斜：把唯一少数类样本预测扰动到正确，损失下降幅度更大
    pl2 = pl.clone(); pl2[0, 0] = torch.tensor([0.0, 10.0, 0.0])
    drop_w = l_w.item() - tr._path_loss(pl2, ps).item()
    cfg3 = KLineConfig(task="classify", path_aux=True)
    tr.model = KLineTransformer(cfg3)
    drop_n = l_now.item() - tr._path_loss(pl2, ps).item()
    assert drop_w > drop_n, f"加权未放大少数类梯度: {drop_w} vs {drop_n}"


# ── E4 excursion 分桶 ────────────────────────────────────────────
def test_exc_labels_bucketed():
    """E4：m_up/m_dn 占 θ 比例 digitize 到 [0.25,0.5,0.8,1.0,1.5] 六桶。
    fixture：θ=1%、base=100，最高 100.3 → 上偏移 0.3θ → 桶1；最低 99.1 → 下偏移 0.9θ → 桶3"""
    ds = _mk_pathstate_ds(highs=[100.2, 100.3, 100.2, 100.2, 100.2],
                          lows=[99.8, 99.1, 99.8, 99.8, 99.8])
    assert ds.exc_labels is not None
    dn_b, up_b = ds.exc_labels[0].tolist()
    assert up_b == 1, f"up 桶错误: {up_b}（期望 1，0.3θ ∈ [0.25,0.5)）"
    assert dn_b == 3, f"dn 桶错误: {dn_b}（期望 3，0.9θ ∈ [0.8,1.0)）"
    item = ds[0]
    assert item["exc_labels"].tolist() == [dn_b, up_b]
    # 未触轨样本（全程窄幅）：两向偏移都 < 0.25θ → 桶 0
    ds0 = _mk_pathstate_ds(highs=[100.1] * 5, lows=[99.9] * 5)
    assert ds0.exc_labels[0].tolist() == [0, 0]


def test_exc_loss_side_weights():
    """E4：逐侧类别权重生效——加权损失与不加权不同，且对少数桶更敏感"""
    import torch
    from obson.model.mixed_trainer import MixedFrequencyTrainer
    from obson.model.transformer import KLineConfig, KLineTransformer
    tr = MixedFrequencyTrainer.__new__(MixedFrequencyTrainer)
    tr.device = "cpu"
    torch.manual_seed(0)
    lg_dn = torch.randn(8, 6); lg_up = torch.randn(8, 6)
    el = torch.zeros(8, 2, dtype=torch.long)
    el[0, 1] = 4  # 唯一少数桶
    tr.model = KLineTransformer(KLineConfig(task="classify"))
    l_plain = tr._exc_loss(lg_dn, lg_up, el)
    cfg = KLineConfig(task="classify", exc_weights=[1.0]*6 + [0.2, 0.2, 0.2, 0.2, 8.0, 0.2])
    tr.model = KLineTransformer(cfg)
    l_w = tr._exc_loss(lg_dn, lg_up, el)
    assert l_w.item() != l_plain.item()
    lg2 = lg_up.clone(); lg2[0] = torch.tensor([0., 0., 0., 0., 10., 0.])
    drop_w = l_w.item() - tr._exc_loss(lg_dn, lg2, el).item()
    tr.model = KLineTransformer(KLineConfig(task="classify"))
    drop_p = l_plain.item() - tr._exc_loss(lg_dn, lg2, el).item()
    assert drop_w > drop_p, f"加权未放大少数桶梯度: {drop_w} vs {drop_p}"


# ── E6' 未来时间 query decoder ──────────────────────────────────
def test_query_decoder_path_head():
    """E6'：query decoder 形态正确、梯度贯通 encoder、与 node_emb 头输出不同"""
    import torch
    from obson.model.transformer import KLineConfig, KLineTransformer
    torch.manual_seed(0)
    B, T, F_in = 2, 30, 8
    cfg = KLineConfig(task="classify", path_aux=True, query_decoder=True,
                      hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      intermediate_size=128, kline_dim=F_in)
    m = KLineTransformer(cfg)
    x = torch.randn(B, T, F_in)
    ps = torch.zeros(B, 4, dtype=torch.long); ps[0, 0] = 1
    out = m(x)
    assert out["path_logits"].shape == (B, 4, 3), out["path_logits"].shape
    loss = out["path_logits"].reshape(-1, 3).softmax(-1).log().gather(
        1, ps.reshape(-1, 1)).neg().mean()
    loss.backward()
    # query/cross-attn 收到梯度，且 encoder 层也收到（梯度贯通，不是断头路）
    assert m.path_queries.grad is not None and m.path_queries.grad.abs().sum() > 0
    assert m.layers[0].self_attn.q_proj.weight.grad is not None
    # 与 E3 简易头同输入下输出不同（确认 decoder 真的在干活）
    cfg2 = KLineConfig(task="classify", path_aux=True, query_decoder=False,
                       hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       intermediate_size=128, kline_dim=F_in)
    m2 = KLineTransformer(cfg2)
    out2 = m2(x)
    assert not torch.allclose(out["path_logits"], out2["path_logits"])
    # 老冠军兼容：默认 query_decoder=False → 走 node_emb 头
    assert hasattr(m2, "path_node_emb") and not hasattr(m2, "path_queries")


# ── Teacher-level competing-risk hazard ──────────────────────────
def test_hazard_probability_aggregation_and_nll():
    """Hazards must aggregate to [down, none, up] probabilities summing to one."""
    import torch
    from obson.model.hazard import hazard_nll, hazard_to_probs

    logits = torch.zeros(2, 4, 3, requires_grad=True)
    targets = torch.tensor([
        [0, 1, -1, -1],  # first upper event in bin 1
        [0, 0, 0, 0],    # censored through the final bin
    ])
    loss = hazard_nll(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    probs = hazard_to_probs(logits.detach())
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)

    # An event-weighted objective must react more strongly to an event-bin
    # mistake than the unweighted objective.
    wrong = torch.zeros(1, 2, 3)
    targets_event = torch.tensor([[0, 1]])
    wrong[0, 1] = torch.tensor([8.0, -8.0, -8.0])
    plain = hazard_nll(wrong, targets_event, event_weight=1.0)
    weighted = hazard_nll(wrong, targets_event, event_weight=5.0)
    assert weighted > plain


def test_dataset_hazard_states_follow_first_event():
    """Dataset hazard labels are survival until one absorbing first event."""
    from obson.model.dataset import KLineDataset
    ts = pd.to_datetime([
        "2026-09-08 09:00", "2026-09-08 10:00", "2026-09-08 11:00",
        "2026-09-08 13:00", "2026-09-08 14:00", "2026-09-08 15:00",
    ])
    close = np.full(6, 100.0)
    high = np.array([100.1, 100.1, 100.1, 101.5, 101.0, 100.1])
    low = np.array([99.9, 99.9, 99.9, 99.8, 99.8, 99.8])
    df = pd.DataFrame({"datetime": ts, "open": close, "high": high,
                       "low": low, "close": close})
    ds = KLineDataset(df, seq_len=3, target_offset=1, normalize=False,
                      jump_break_pct=None, label_mode="day_close",
                      label_threshold=1.0, theta_mode="frozen")
    states = ds.hazard_states
    assert states is not None and len(states) > 0
    for row in states:
        events = np.flatnonzero((row == 1) | (row == 2))
        if len(events):
            first = int(events[0])
            assert np.all(row[:first] == 0)
            assert row[first] in (1, 2)
            assert np.all(row[first + 1:] == -1)
        else:
            assert np.all(row == 0) or np.all(row == -1)


def test_hazard_task_model_output():
    """Hazard mode keeps the classifier public and exposes hazards as auxiliary output."""
    import torch
    from obson.model.transformer import KLineConfig, KLineTransformer
    cfg = KLineConfig(task="classify", hazard_task=True, hazard_bins=4,
                      hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, head_dim=16,
                      intermediate_size=64, kline_dim=4)
    model = KLineTransformer(cfg)
    model.eval()
    x = torch.randn(2, 20, 4)
    targets = torch.zeros(2, 4, dtype=torch.long)
    out = model(x, labels=torch.tensor([1, 1]), hazard_targets=targets)
    assert out["hazard_logits"].shape == (2, 4, 3)
    assert out["logits"].shape == (2, 3)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_hazard"])
    with torch.no_grad():
        model.hazard_head.weight.zero_()
        model.hazard_head.bias.zero_()
    out_zero = model(x, labels=torch.tensor([1, 1]), hazard_targets=targets)
    assert torch.allclose(out["logits"], out_zero["logits"])


def test_serial_path_model_is_prediction_only():
    """Serial path fusion uses predicted representations, never path labels."""
    import torch
    from obson.model.transformer import KLineConfig, KLineTransformer
    cfg = KLineConfig(task="classify", serial_path=True,
                      hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, head_dim=16,
                      intermediate_size=64, kline_dim=4)
    model = KLineTransformer(cfg)
    model.eval()
    x = torch.randn(2, 20, 4)
    labels = torch.tensor([1, 2])
    out_a = model(x, labels=labels,
                  serial_path_targets=torch.tensor([[0, 0, -1, -1], [1, 1, 1, -1]]))
    out_b = model(x, labels=labels,
                  serial_path_targets=torch.tensor([[2, 2, 2, 2], [0, 0, 0, 0]]))
    assert out_a["serial_path_logits"].shape == (2, 4, 3)
    assert torch.allclose(out_a["logits"], out_b["logits"])
    assert torch.isfinite(out_a["loss_serial_path"])


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    sys.exit(1 if failed else 0)
