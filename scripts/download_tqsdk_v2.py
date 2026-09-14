"""从天勤 TQSDK 免费版下载期货主连分钟线（分块锚定翻页版）

免费版单 chart 上限 8964 根，但服务端支持 focus_datetime/left_kline_id 锚定。
策略：
  1) get_kline_serial 拿最近 8964 根（顺带验证合约与权限）
  2) focus_datetime 向前锚定逐段翻页，直到服务端不再返回更早数据
     （免费版天花板钳在最近 8964 根，或翻到合约起点）

输出 data/raw_tqsdk/{code}_{period}m.csv，不覆盖 data/ 现有文件。

用法:
    TQ_USER=xxx TQ_PASS=xxx uv run python scripts/download_tqsdk_v2.py
    TQ_USER=xxx TQ_PASS=xxx uv run python scripts/download_tqsdk_v2.py --symbols rb sr --periods 5 15
    TQ_USER=xxx TQ_PASS=xxx uv run python scripts/download_tqsdk_v2.py --max-back-years 3
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.utils import _generate_uuid

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_tqsdk"
PERIODS = {5: 300, 15: 900, 30: 1800, 60: 3600}
PAGE = 8964  # 免费版单 chart 上限

# 品种 → (天勤主连符号, 中文名)
SYMBOLS = {
    "rb": ("KQ.m@SHFE.rb", "螺纹钢"),
    "hc": ("KQ.m@SHFE.hc", "热卷"),
    "i": ("KQ.m@DCE.i", "铁矿石"),
    "sr": ("KQ.m@CZCE.SR", "白糖"),
    "p": ("KQ.m@DCE.p", "棕榈油"),
    "m": ("KQ.m@DCE.m", "豆粕"),
    "y": ("KQ.m@DCE.y", "豆油"),
    "j": ("KQ.m@DCE.j", "焦炭"),
    "jm": ("KQ.m@DCE.jm", "焦煤"),
    "cu": ("KQ.m@SHFE.cu", "铜"),
    "ag": ("KQ.m@SHFE.ag", "白银"),
    "TA": ("KQ.m@CZCE.TA", "PTA"),
    "MA": ("KQ.m@CZCE.MA", "甲醇"),
}


def _dt_ns_to_sh(ns: int) -> pd.Timestamp:
    return pd.to_datetime(ns, utc=True).tz_convert("Asia/Shanghai").tz_localize(None)


def _read_chart_window(api, chart_id: str, symbol: str, dur_ns: int):
    node = api._data["charts"].get(chart_id)
    if node is None:
        return None
    left_id = node.get("left_id", -1)
    right_id = node.get("right_id", -1)
    if left_id < 0 or right_id < left_id:
        return None
    data = api._data["klines"][symbol][str(dur_ns)]["data"]
    rows = []
    for kid in range(left_id, right_id + 1):
        item = data.get(str(kid))
        if item is None or item.get("datetime", 0) == 0:
            continue
        rows.append({
            "id": kid,
            "datetime_ns": int(item["datetime"]),
            "open": item["open"], "high": item["high"],
            "low": item["low"], "close": item["close"],
            "volume": item["volume"],
            "open_oi": item.get("open_oi", 0), "close_oi": item.get("close_oi", 0),
        })
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["id", "datetime_ns", "open", "high", "low", "close",
                                       "volume", "open_oi", "close_oi"])


def _wait_chart_ready(api, chart_id: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        node = api._data["charts"].get(chart_id)
        if node is not None and node.get("ready"):
            return True
        try:
            api.wait_update(deadline=time.time() + 2)
        except Exception:
            pass
    node = api._data["charts"].get(chart_id)
    return bool(node and node.get("ready"))


def _close_chart(api, chart_id: str, dur_ns: int) -> None:
    try:
        api._send_pack({
            "aid": "set_chart",
            "chart_id": chart_id,
            "ins_list": "",
            "duration": int(dur_ns),
            "view_width": 2000,
        })
    except Exception:
        pass


def _fetch_one(api, code: str, symbol: str, period: int, max_back_years: float,
               stop_dt: pd.Timestamp | None = None):
    dur_s = PERIODS[period]
    dur_ns = dur_s * 1_000_000_000
    name = SYMBOLS[code][1]
    print(f"\n===== {code} {name} {period}m =====")

    try:
        df_latest = api.get_kline_serial(symbol, dur_s, data_length=PAGE)
    except Exception as e:
        print(f"  !! kline_serial 失败: {e}")
        return None
    df_latest = df_latest.dropna(subset=["datetime"]).copy()
    if len(df_latest) == 0:
        print("  !! 无数据")
        return None
    df_latest["id"] = df_latest["id"].astype(int)
    min_id = int(df_latest["id"].iloc[0])
    max_id = int(df_latest["id"].iloc[-1])
    min_dt = _dt_ns_to_sh(int(df_latest["datetime"].iloc[0]))
    latest_dt = _dt_ns_to_sh(int(df_latest["datetime"].iloc[-1]))
    print(f"  最近一页: {len(df_latest)} 条 | id {min_id}..{max_id} | {min_dt} ~ {latest_dt}")

    for c in ("open_oi", "close_oi"):
        if c not in df_latest.columns:
            df_latest[c] = 0
    pages = [df_latest[["id", "datetime", "open", "high", "low", "close", "volume",
                        "open_oi", "close_oi"]].assign(
        datetime=df_latest["datetime"].apply(_dt_ns_to_sh))]
    cutoff_dt = min_dt - pd.Timedelta(days=int(max_back_years * 365.25))
    # 增量快路径：最近一页（8964 根，60m≈3.5年）起点已早于旧文件末尾，
    # 说明新旧数据有重叠，公开接口一页就够，完全跳过私有锚定翻页
    # （"锚定页等待超时"的根因：夜盘增量只需几根新 bar 却在翻私有页）
    skip_paging = stop_dt is not None and min_dt <= stop_dt
    if stop_dt is not None:
        if skip_paging:
            print(f"  增量模式: 一页已覆盖旧末尾 {stop_dt}，跳过锚定翻页")
        else:
            # 增量模式：只需翻到与已有数据重叠即可
            cutoff_dt = max(cutoff_dt, stop_dt - pd.Timedelta(days=2))
            print(f"  增量模式: 数据断档较久，锚定翻到 {cutoff_dt} 即停")
    page_no = 1

    while not skip_paging:
        span_days = max((min_dt - latest_dt).days, 1)
        anchor_dt = min_dt - pd.Timedelta(days=span_days // 2 + 1)
        if anchor_dt < cutoff_dt:
            anchor_dt = cutoff_dt
        cid = _generate_uuid(f"OBSON_BACK_{code}_{period}_{page_no}")
        pack = {
            "aid": "set_chart",
            "chart_id": cid,
            "ins_list": symbol,
            "duration": int(dur_ns),
            "view_width": PAGE,
            "focus_datetime": int(pd.Timestamp(anchor_dt).tz_localize("Asia/Shanghai").value),
            "focus_position": 0,
        }
        df_page = None
        try:
            api._send_pack(pack)
            if not _wait_chart_ready(api, cid, timeout=30):
                print("  !! 锚定页等待超时")
                break
            df_page = _read_chart_window(api, cid, symbol, dur_ns)
        except Exception as e:
            print(f"  !! 锚定页失败: {e}")
            break
        finally:
            _close_chart(api, cid, dur_ns)

        if df_page is None or len(df_page) == 0:
            print("  翻页到底（无数据）")
            break
        new_min_id = int(df_page["id"].iloc[0])
        new_min_dt = _dt_ns_to_sh(int(df_page["datetime_ns"].iloc[0]))
        if new_min_id >= min_id:
            print(f"  服务端钳在 id {min_id}（免费版可见窗口），停止翻页")
            break
        df_page["datetime"] = df_page["datetime_ns"].apply(_dt_ns_to_sh)
        pages.append(df_page.drop(columns=["datetime_ns"]))
        print(f"  +{len(df_page)} 条 | 到 {new_min_dt}")
        min_id = new_min_id
        min_dt = new_min_dt
        page_no += 1
        if min_dt <= cutoff_dt:
            print("  达到 --max-back-years 目标")
            break
        if min_id == 0:
            print("  已到合约最早")
            break
        time.sleep(0.2)

    out = (pd.concat(pages, ignore_index=True)
             .drop_duplicates(subset=["id"])
             .sort_values("datetime")
             .reset_index(drop=True))
    out["datetime"] = out["datetime"].astype(str)
    out = out[["datetime", "open", "high", "low", "close", "volume", "open_oi", "close_oi"]]
    print(f"  合计 {len(out)} 条 | {out['datetime'].iloc[0]} ~ {out['datetime'].iloc[-1]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=list(SYMBOLS),
                    help="品种，默认全部")
    ap.add_argument("--periods", nargs="+", type=int, default=list(PERIODS), choices=list(PERIODS),
                    help="频率，默认 5/15/30/60")
    ap.add_argument("--max-back-years", type=float, default=5.0,
                    help="最多向前拉多少年，默认 5")
    ap.add_argument("--incremental", action="store_true",
                    help="增量模式：已有 CSV 时只补最后一根之后的新数据，与旧文件合并（不重拉全量历史）")
    args = ap.parse_args()

    user, password = os.environ.get("TQ_USER"), os.environ.get("TQ_PASS")
    if not user or not password:
        raise SystemExit("请先设置环境变量: TQ_USER=xxx TQ_PASS=xxx")

    api = TqApi(account=TqSim(), auth=TqAuth(user, password))
    print("登录成功")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for code in args.symbols:
        symbol, _name = SYMBOLS[code]
        for period in args.periods:
            out = RAW_DIR / f"{code}_{period}m.csv"
            old_df = None
            if args.incremental and out.exists():
                old_df = pd.read_csv(out)
            stop_dt = pd.Timestamp(old_df["datetime"].iloc[-1]) if old_df is not None else None
            try:
                df = _fetch_one(api, code, symbol, period, args.max_back_years, stop_dt=stop_dt)
            except Exception as e:
                print(f"  !! {code} {period}m 异常: {e}")
                continue
            if df is None:
                continue
            if old_df is not None:
                # 增量合并：新数据覆盖重叠区（当天 bar 是进行中的，以新为准），不重写历史
                merged = (pd.concat([old_df, df], ignore_index=True)
                          .drop_duplicates(subset=["datetime"], keep="last")
                          .sort_values("datetime").reset_index(drop=True))
                bak = out.with_suffix(f".bak_{datetime.now():%Y%m%d_%H%M%S}.csv")
                out.rename(bak)
                merged.to_csv(out, index=False)
                n_new = len(merged) - len(old_df)
                print(f"  增量合并: {len(old_df)} → {len(merged)}（+{n_new} 行），旧文件备份 {bak.name}")
                continue
            if out.exists():
                existing = len(pd.read_csv(out))
                if len(df) <= existing:
                    print(f"  已有 {existing} 条不短于新数据，跳过")
                    continue
                bak = out.with_suffix(f".bak_{datetime.now():%Y%m%d_%H%M%S}.csv")
                out.rename(bak)
                print(f"  旧文件已备份: {bak.name}")
            df.to_csv(out, index=False)
            print(f"  已保存: {out}")

    api.close()


if __name__ == "__main__":
    main()
