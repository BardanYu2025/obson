"""新浪财经期货数据源"""

from __future__ import annotations

import json
import re
import time
import warnings
from typing import Any

import requests

from obson.core.exceptions import DataSourceError, NetworkError
from obson.core.models import DataResponse, Field, KLine, Period, Quote
from obson.data.sources.base import DataSource

# ── 常量 ──
SINA_HQ = "https://hq.sinajs.cn/list={symbols}"
SINA_KLINE = (
    "https://stock2.finance.sina.com.cn/futures/api/json.php/"
    "IndexService.getInnerFuturesMiniKLine{period}?symbol={symbol}"
)
SINA_DAILY_KLINE = (
    "https://stock2.finance.sina.com.cn/futures/api/json.php/"
    "IndexService.getInnerFuturesDailyKLine?symbol={symbol}"
)
CFFEX_KLINE = (
    "https://stock2.finance.sina.com.cn/futures/api/json.php/"
    "CffexFuturesService.getCffexFuturesMiniKLine{period}?symbol={symbol}"
)
CFFEX_DAILY_KLINE = (
    "https://stock2.finance.sina.com.cn/futures/api/json.php/"
    "CffexFuturesService.getCffexFuturesDailyKLine?symbol={symbol}"
)

HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
}

# 中金所品种判断
CFFEX_PREFIXES = ("IF", "IC", "IH", "IM", "T", "TF", "TS", "TL")

# 分钟周期映射: 1m 新浪不支持，用 5m 兜底
PERIOD_MAP: dict[Period, str] = {
    Period.MIN1: "5m",      # 新浪无 1m，降级到 5m
    Period.MIN5: "5m",
    Period.MIN15: "15m",
    Period.MIN30: "30m",
    Period.MIN60: "60m",
    Period.DAILY: "1d",
}


class SinaFuturesSource(DataSource):
    """新浪财经期货行情数据源"""

    name = "sina"

    def __init__(self, timeout: int = 15, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    # ── 公共接口 ──

    def is_available(self) -> bool:
        try:
            self._fetch(SINA_HQ.format(symbols="nf_AU0"))
            return True
        except Exception:
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        period: Period = Period.DAILY,
        count: int = 100,
        fields: list[Field] | None = None,
        **kwargs: Any,
    ) -> DataResponse:
        if fields is None:
            fields = [Field.DATETIME, Field.OPEN, Field.HIGH, Field.LOW, Field.CLOSE, Field.VOLUME]

        sina_period = PERIOD_MAP.get(period, "1d")
        is_cffex = symbol.startswith(CFFEX_PREFIXES)

        if sina_period == "1d":
            url = CFFEX_DAILY_KLINE.format(symbol=symbol) if is_cffex else SINA_DAILY_KLINE.format(symbol=symbol)
        else:
            url = (
                CFFEX_KLINE.format(period=sina_period, symbol=symbol)
                if is_cffex
                else SINA_KLINE.format(period=sina_period, symbol=symbol)
            )

        text = self._fetch(url)
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]

        try:
            raw_data = json.loads(text)
        except json.JSONDecodeError as e:
            raise DataSourceError(f"新浪K线JSON解析失败: {e}") from e

        records = []
        for item in raw_data[-count:]:
            if isinstance(item, list) and len(item) >= 5:
                records.append({
                    "datetime": item[0],
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": int(float(item[5])) if len(item) > 5 else 0,
                })
            elif isinstance(item, dict):
                records.append({
                    "datetime": item.get("d", ""),
                    "open": float(item.get("o", 0)),
                    "high": float(item.get("h", 0)),
                    "low": float(item.get("l", 0)),
                    "close": float(item.get("c", 0)),
                    "volume": int(float(item.get("v", 0))),
                })

        df = self._to_dataframe(records, fields, symbol)
        return DataResponse(
            product=kwargs.get("product", symbol),
            symbol=symbol,
            source=self.name,
            period=period,
            data=df,
            fields=fields,
        )

    def fetch_quote(self, symbol: str, **kwargs: Any) -> Quote:
        results = self._fetch_quotes([symbol])
        if not results:
            raise DataSourceError(f"未获取到 {symbol} 行情")
        return results[0]

    def fetch_quotes(self, symbols: list[str], **kwargs: Any) -> list[Quote]:
        return self._fetch_quotes(symbols)

    # ── 内部实现 ──

    def _fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        """批量获取实时行情（内部）"""
        sina_symbols = []
        for s in symbols:
            s = s.strip().upper()
            sina_symbols.append(s if s.startswith("CFF_RE_") else f"nf_{s}")

        url = SINA_HQ.format(symbols=",".join(sina_symbols))
        text = self._fetch(url)

        results = []
        for line in text.split(";"):
            line = line.strip()
            if not line or "var hq_str_" not in line:
                continue
            q = self._parse_quote_line(line)
            if q:
                results.append(q)

        return results

    def _parse_quote_line(self, line: str) -> Quote | None:
        match = re.search(r'var hq_str_(.+?)="(.+?)"', line)
        if not match:
            return None

        symbol_key = match.group(1)
        fields = match.group(2).split(",")
        if len(fields) < 10:
            return None

        symbol = symbol_key.replace("nf_", "").replace("CFF_RE_", "")

        def _f(idx: int) -> float:
            return float(fields[idx]) if idx < len(fields) and fields[idx] else 0.0

        def _i(idx: int) -> int:
            return int(float(fields[idx])) if idx < len(fields) and fields[idx] else 0

        try:
            return Quote(
                symbol=symbol,
                name=fields[0] if fields[0] else symbol,
                exchange=fields[15] if len(fields) > 15 and fields[15] else "",
                open=_f(2),
                high=_f(3),
                low=_f(4),
                prev_close=_f(5),
                bid=_f(6),
                ask=_f(7),
                close=_f(8),
                settle=_f(9),
                prev_settle=_f(10),
                bid_vol=_i(11),
                ask_vol=_i(12),
                open_interest=_i(13),
                volume=_i(14),
                datetime=fields[17] if len(fields) > 17 and fields[17] else "",
                source=self.name,
                raw={"fields": fields},
            )
        except (ValueError, IndexError) as e:
            warnings.warn(f"解析 {symbol} 行情失败: {e}")
            return None

    def _fetch(self, url: str) -> str:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.get(url, timeout=self.timeout)
                resp.encoding = "gb2312"
                if resp.status_code == 200:
                    return resp.text
                last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise NetworkError(f"请求失败 ({self.max_retries + 1} 次重试): {last_err}\nURL: {url}")

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
