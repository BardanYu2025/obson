"""按合约号下载期货分钟线 + 自建主力切换日历（数据底座与平台解耦）

动机：
  - 主连（KQ.m@）是天勤自家拼接规则，换平台口径就对不上；合约号（rb2601）是交易所标准，
    天勤/Wind/米筐/掘金/CTP 全通用，数据层不绑死任何平台。
  - 训练样本不跨合约 ⇒ 换月跳空问题从源头消失（不再靠 |ret|>2% 断链剔除），
    满足"标签路径不跨换月"的审查要求。

输出（不覆盖 data/ 现有文件）：
  data/contracts/{code}/{CONTRACT}_{period}m.csv     各合约分钟线（8列，与主连同格式）
  data/contracts/{code}_roll_calendar.csv            主力切换日历：date,contract,volume,close_oi

主力规则：按日聚合各合约成交量，量最大者为当日主力（自洽，无需外部源）。

用法:
    TQ_USER=xxx TQ_PASS=xxx uv run python scripts/download_contracts.py --symbols rb
    TQ_USER=xxx TQ_PASS=xxx uv run python scripts/download_contracts.py --symbols rb --periods 60 30
    uv run python scripts/download_contracts.py --symbols rb --calendar-only   # 只重建日历
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from download_tqsdk_v2 import PERIODS, _dt_ns_to_sh  # noqa: E402

CONTRACT_DIR = Path(__file__).parent.parent / "data" / "contracts"
PAGE = 8964


def _fetch_contract(api, code: str, contract: str, period: int):
    """拉单个合约分钟线。合约生命周期 ~1 年，60/30/15m 多数 < 8964 上限，
    第一页即全量（min_id<=0 = 服务端已从上市日给齐），此时跳过锚定翻页
    （每次翻页固定浪费 30s 超时）。仅当第一页 min_id>0 才走完整翻页。"""
    from download_tqsdk_v2 import _fetch_one, SYMBOLS as _TQ_SYMBOLS
    _TQ_SYMBOLS.setdefault(code, ("", code))  # _fetch_one 仅用于显示名称
    dur_s = PERIODS[period]
    df = api.get_kline_serial(contract, dur_s, data_length=PAGE)
    df = df.dropna(subset=["datetime"]).copy()
    if len(df) == 0:
        return None
    df["id"] = df["id"].astype(int)
    min_id = int(df["id"].iloc[0])
    if min_id > 0:
        # 第一页没到合约上市日（数据量超上限），走完整翻页流程
        return _fetch_one(api, code, contract, period, max_back_years=10)
    for c in ("open_oi", "close_oi"):
        if c not in df.columns:
            df[c] = 0
    out = df[["id", "datetime", "open", "high", "low", "close", "volume",
              "open_oi", "close_oi"]].assign(datetime=df["datetime"].apply(_dt_ns_to_sh))
    out = out[pd.to_datetime(out["datetime"]) > "2001-01-01"].reset_index(drop=True)
    out["datetime"] = out["datetime"].astype(str)
    out = out[["datetime", "open", "high", "low", "close", "volume", "open_oi", "close_oi"]]
    # CZCE 3 位年码十年一循环（MA701=2017-01 与 2027-01 共用代码），
    # 服务端会把两代合约拼进同一序列。按 >90 天断点切段，只保留最新一代。
    ts = pd.to_datetime(out["datetime"])
    gap_pos = ts.diff().dt.days > 90
    if gap_pos.any():
        cut = int(gap_pos.idxmax())
        dropped = len(out) - len(out.iloc[cut:])
        print(f"    !! {contract} 含上一代同名合约数据（{out['datetime'].iloc[0][:10]} 段），"
              f"剔除 {len(out) - len(out.iloc[cut:])} 行，保留 {out['datetime'].iloc[cut][:10]} 起")
        out = out.iloc[cut:].reset_index(drop=True)
    return out

# 品种 → (交易所, product_id)。合约全名 = {exchange}.{product}{YYMM}（天勤格式）
PRODUCTS = {
    "rb": ("SHFE", "rb"), "hc": ("SHFE", "hc"), "cu": ("SHFE", "cu"), "ag": ("SHFE", "ag"),
    "ru": ("SHFE", "ru"), "au": ("SHFE", "au"),
    "i": ("DCE", "i"), "p": ("DCE", "p"), "m": ("DCE", "m"), "y": ("DCE", "y"),
    "j": ("DCE", "j"), "jm": ("DCE", "jm"), "c": ("DCE", "c"), "cs": ("DCE", "cs"),
    "sr": ("CZCE", "SR"), "TA": ("CZCE", "TA"), "MA": ("CZCE", "MA"), "RM": ("CZCE", "RM"),
}

# 各品种主力合约月份（交易所惯例；只下载这些月份，省 3/4 下载量。
# 日历仍按成交量自动选主力，月份过滤只是粗筛，选错不污染结果）
MAIN_MONTHS = {
    "rb": (1, 5, 10), "hc": (1, 5, 10),
    "i": (1, 5, 9), "p": (1, 5, 9), "m": (1, 5, 9), "y": (1, 5, 9),
    "j": (1, 5, 9), "jm": (1, 5, 9), "c": (1, 5, 9), "cs": (1, 5, 9),
    "sr": (1, 5, 9), "TA": (1, 5, 9), "MA": (1, 5, 9), "RM": (1, 5, 9),
    "ru": (1, 5, 9),
    "ag": (6, 12), "au": (6, 12),
    "cu": tuple(range(1, 13)),  # 铜逐月轮转，全下
}


def _contract_month(symbol: str) -> int:
    """从合约名解析月份：SHFE/DCE 4 位（rb2601→01），CZCE 3 位（SR601→01）。"""
    return int(symbol[-2:])


def list_contracts(api, code: str) -> list[str]:
    """枚举品种主力月份合约，按合约名排序。

    天勤 query_quotes 的 expired=True/False 是两个不相交集合
    （True=只返回已摘牌，False=只返回当前挂牌），必须取并集。"""
    exchange, product = PRODUCTS[code]
    quotes: set[str] = set()
    for expired in (False, True):
        try:
            quotes |= {str(q) for q in api.query_quotes(
                ins_class="FUTURE", exchange_id=exchange,
                product_id=product, expired=expired)}
        except TypeError:  # 旧版 tqsdk 无 expired 参数，一次调用拿全集
            quotes |= {str(q) for q in api.query_quotes(
                ins_class="FUTURE", exchange_id=exchange, product_id=product)}
            break
    months = MAIN_MONTHS[code]
    return sorted(q for q in quotes if _contract_month(q) in months)


def download_contracts(api, code: str, periods: list[int], max_back_years: float,
                       limit: int | None = None) -> list[str]:
    """下载该品种所有合约的分钟线，返回成功保存的合约名列表。"""
    out_dir = CONTRACT_DIR / code
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts = list_contracts(api, code)
    print(f"[{code}] 共 {len(contracts)} 个合约: {contracts[:3]} ... {contracts[-3:]}")
    if limit:
        contracts = contracts[-limit:]
        print(f"  试点模式：只取最近 {limit} 个合约 {contracts}")
    saved = []
    for k, contract in enumerate(contracts, 1):
        for period in periods:
            fp = out_dir / f"{contract}_{period}m.csv"
            if fp.exists() and len(pd.read_csv(fp)) > 100:
                print(f"  [{k}/{len(contracts)}] {contract} {period}m 已存在，跳过")
                saved.append(contract)
                continue
            try:
                df = _fetch_contract(api, code, contract, period)
            except Exception as e:
                print(f"  [{k}/{len(contracts)}] {contract} {period}m 异常: {e}")
                continue
            if df is None or len(df) < 10:
                print(f"  [{k}/{len(contracts)}] {contract} {period}m 无有效数据（未交易/已退市过深）")
                continue
            df = df[pd.to_datetime(df["datetime"]) > "2001-01-01"].reset_index(drop=True)  # 剔 1970 占位行
            df.to_csv(fp, index=False)
            print(f"  [{k}/{len(contracts)}] 已保存 {fp.name}: {len(df)} 条 "
                  f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
            saved.append(contract)
        time.sleep(0.3)
    return sorted(set(saved))


def build_roll_calendar(code: str, period: int = 60, confirm_days: int = 2) -> pd.DataFrame:
    """按日聚合各合约成交量，带迟滞地选当日主力 → 主力切换日历。

    迟滞规则：挑战者须连续 confirm_days 天成交量超过现任主力才切换，
    避免换月交叉期在两合约间反复横跳。
    注意：日历只在已下载合约的并集覆盖范围内有效，缺合约会产生伪切换。"""
    out_dir = CONTRACT_DIR / code
    frames = []
    for fp in sorted(out_dir.glob(f"*_{period}m.csv")):
        contract = fp.name.replace(f"_{period}m.csv", "")
        df = pd.read_csv(fp, parse_dates=["datetime"])
        df = df[df["datetime"] > "2001-01-01"]
        df["date"] = (df["datetime"] + pd.Timedelta(hours=6)).dt.date  # 夜盘归入次日（与训练口径一致）
        g = df.groupby("date", as_index=False).agg(volume=("volume", "sum"),
                                                   close_oi=("close_oi", "last"))
        g["contract"] = contract
        frames.append(g)
    if not frames:
        raise SystemExit(f"[{code}] {out_dir} 无合约数据，先跑下载")
    allv = pd.concat(frames, ignore_index=True)
    pivot = allv.pivot_table(index="date", columns="contract", values="volume",
                             aggfunc="sum").fillna(0.0).sort_index()
    oi = allv.pivot_table(index="date", columns="contract", values="close_oi",
                          aggfunc="last").reindex(pivot.index)

    current = None
    challenger, streak = None, 0
    rows = []
    for date, vols in pivot.iterrows():
        day_best = vols.idxmax()
        if current is None:
            current = day_best
        elif day_best != current and vols[day_best] > vols.get(current, 0.0):
            if challenger == day_best:
                streak += 1
            else:
                challenger, streak = day_best, 1
            if streak >= confirm_days:
                current, challenger, streak = day_best, None, 0
        else:
            challenger, streak = None, 0
        rows.append({"date": date, "contract": current,
                     "volume": float(vols.get(current, 0.0)),
                     "close_oi": float(oi.loc[date, current]) if current in oi.columns else 0.0})
    cal = pd.DataFrame(rows)
    out_fp = CONTRACT_DIR / f"{code}_roll_calendar.csv"
    cal.to_csv(out_fp, index=False)
    switches = int((cal["contract"] != cal["contract"].shift()).sum() - 1)
    print(f"[{code}] 主力日历: {len(cal)} 个交易日，{switches} 次切换（迟滞{confirm_days}天）→ {out_fp}")
    print(cal.tail(10).to_string(index=False))
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["rb"], choices=list(PRODUCTS))
    ap.add_argument("--periods", nargs="+", type=int, default=[60], choices=list(PERIODS))
    ap.add_argument("--max-back-years", type=float, default=5.0)
    ap.add_argument("--limit-contracts", type=int, default=None,
                    help="试点：只下载最近 N 个合约")
    ap.add_argument("--calendar-only", action="store_true", help="只重建主力切换日历")
    args = ap.parse_args()

    api = None
    if not args.calendar_only:
        from tqsdk import TqApi, TqAuth, TqSim
        user, password = os.environ.get("TQ_USER"), os.environ.get("TQ_PASS")
        if not user or not password:
            raise SystemExit("请先设置环境变量: TQ_USER=xxx TQ_PASS=xxx")
        api = TqApi(account=TqSim(), auth=TqAuth(user, password))
        print("登录成功")

    try:
        for code in args.symbols:
            if not args.calendar_only:
                download_contracts(api, code, args.periods, args.max_back_years,
                                   limit=args.limit_contracts)
            build_roll_calendar(code, period=args.periods[0])
    finally:
        if api is not None:
            api.close()


if __name__ == "__main__":
    main()
