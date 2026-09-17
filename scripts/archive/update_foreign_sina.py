# -*- coding: utf-8 -*-
"""外盘日线补更：新浪 GlobalFuturesService 日K接口
用法: PYTHONPATH=src python -u scripts/update_foreign_sina.py
覆盖: bmd_palm(FCPO 马棕油) / sgx_iron(FEF 新交所铁矿) / comex_silver(SI 纽约银)
ice_sugar 单位口径与新浪不一致，不碰；cbot 系列另有源。
当天行是进行中行情（非结算价），下次运行会被同名日期覆盖修正。
"""
import json
import re
from pathlib import Path

import pandas as pd
import requests

SINA_SYMBOL = {"bmd_palm": "FCPO", "sgx_iron": "FEF", "comex_silver": "SI"}
URL = ("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/x/"
       "GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={sym}")
HDR = {"Referer": "https://finance.sina.com.cn"}


def fetch_daily(sym: str) -> pd.DataFrame:
    r = requests.get(URL.format(sym=sym), headers=HDR, timeout=20)
    m = re.search(r"x\((.*)\)", r.text, re.S)
    rows = json.loads(m.group(1))
    df = pd.DataFrame({"date": [x["date"].replace("-", "") for x in rows],
                       "close": [float(x["close"]) for x in rows]})
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def main():
    for name, sym in SINA_SYMBOL.items():
        fp = Path(f"data/foreign/{name}_1d.csv")
        old = pd.read_csv(fp, dtype={"date": str})
        new = fetch_daily(sym)
        last_date, last_close = old["date"].iloc[-1], float(old["close"].iloc[-1])
        # 只追加比现有最后日期更新的行；历史一概不碰（源口径不同，重写会污染训练集）
        add = new[new["date"] > last_date].copy()
        if not len(add):
            print(f"[{name}] {sym}: 已是最新（至 {last_date}）")
            continue
        #  sanity 1：新段首行与旧末行价差 >3% → 疑似换月/单位问题，拒绝并报警
        gap = add["close"].iloc[0] / last_close - 1
        # sanity 2：剔除新段内的单日尖刺（相对前后邻居 >10% 的孤立点，多为脏 tick）
        closes = add["close"].to_numpy()
        keep = []
        for i, c in enumerate(closes):
            prev_c = closes[i - 1] if i > 0 else last_close
            next_c = closes[i + 1] if i + 1 < len(closes) else None
            spike = abs(c / prev_c - 1) > 0.10 and (next_c is None or abs(c / next_c - 1) > 0.10)
            keep.append(not spike)
        add_clean = add[keep]
        n_spike = len(add) - len(add_clean)
        if abs(gap) > 0.03:
            print(f"[{name}] {sym}: ⚠️ 新段首行 {add['close'].iloc[0]} 与旧末行 {last_close} 差 {gap:+.1%}"
                  f"（疑似换月），已拒绝，请人工核对")
            continue
        comb = pd.concat([old, add_clean]).sort_values("date").reset_index(drop=True)
        comb.to_csv(fp, index=False)
        print(f"[{name}] {sym}: {last_date}({last_close}) → {comb['date'].iloc[-1]}({comb['close'].iloc[-1]})"
              f" | 追加 {len(add_clean)} 行" + (f"，剔除尖刺 {n_spike} 行" if n_spike else ""))
        print(add_clean.to_string(index=False, header=False))


if __name__ == "__main__":
    main()
