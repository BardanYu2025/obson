"""核心数据模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class Period(str, Enum):
    """支持的时间周期"""
    MIN1 = "1m"
    MIN5 = "5m"
    MIN15 = "15m"
    MIN30 = "30m"
    MIN60 = "60m"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class Field(str, Enum):
    """支持的字段"""
    DATETIME = "datetime"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    OPEN_INTEREST = "open_interest"
    SETTLE = "settle"
    PREV_SETTLE = "prev_settle"
    BID = "bid"
    ASK = "ask"
    BID_VOL = "bid_vol"
    ASK_VOL = "ask_vol"
    CHANGE = "change"
    CHANGE_PCT = "change_pct"


DEFAULT_OHLCV_FIELDS = [
    Field.DATETIME,
    Field.OPEN,
    Field.HIGH,
    Field.LOW,
    Field.CLOSE,
    Field.VOLUME,
]


@dataclass
class Quote:
    """实时行情快照"""
    symbol: str
    name: str = ""
    exchange: str = ""

    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    prev_close: float = 0.0
    settle: float = 0.0
    prev_settle: float = 0.0

    bid: float = 0.0
    ask: float = 0.0
    bid_vol: int = 0
    ask_vol: int = 0

    volume: int = 0
    open_interest: int = 0

    datetime: str = ""
    source: str = ""

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def change(self) -> float:
        return self.close - self.prev_settle if self.prev_settle else 0.0

    @property
    def change_pct(self) -> float:
        if self.prev_settle and self.prev_settle != 0:
            return round((self.close - self.prev_settle) / self.prev_settle * 100, 4)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "prev_settle": self.prev_settle,
            "change": self.change,
            "change_pct": self.change_pct,
            "bid": self.bid,
            "ask": self.ask,
            "bid_vol": self.bid_vol,
            "ask_vol": self.ask_vol,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "datetime": self.datetime,
            "source": self.source,
        }


@dataclass
class KLine:
    """K线数据"""
    datetime: datetime | str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    open_interest: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "datetime": self.datetime,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "open_interest": self.open_interest,
        }


@dataclass
class DataResponse:
    """统一数据响应"""
    product: str
    symbol: str
    source: str
    period: Period
    data: pd.DataFrame = field(repr=False)
    fields: list[Field] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dicts(self) -> list[dict[str, Any]]:
        """转为字典列表"""
        return self.data.to_dict("records")

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.data.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self.data.tail(n)
