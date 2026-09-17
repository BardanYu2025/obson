"""数据源抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from obson.core.models import DataResponse, Field, Period, Quote


class DataSource(ABC):
    """数据源抽象基类 —— 所有具体数据源必须实现此接口"""

    name: str = ""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        period: Period = Period.DAILY,
        count: int = 100,
        fields: list[Field] | None = None,
        **kwargs: Any,
    ) -> DataResponse:
        """
        获取历史K线数据 (OHLCV)

        :param symbol: 合约代码，如 "RB0"
        :param period: 周期
        :param count: 条数
        :param fields: 指定返回字段
        :return: DataResponse
        """
        ...

    @abstractmethod
    def fetch_quote(self, symbol: str, **kwargs: Any) -> Quote:
        """
        获取实时行情

        :param symbol: 合约代码
        :return: Quote
        """
        ...

    @abstractmethod
    def fetch_quotes(self, symbols: list[str], **kwargs: Any) -> list[Quote]:
        """
        批量获取实时行情

        :param symbols: 合约代码列表
        :return: Quote 列表
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        ...

    def _to_dataframe(
        self,
        records: list[dict[str, Any]],
        fields: list[Field],
        symbol: str,
    ) -> pd.DataFrame:
        """通用：将记录转为 DataFrame 并按字段过滤"""
        df = pd.DataFrame(records)
        if df.empty:
            return df

        # 字段过滤
        field_names = [f.value for f in fields]
        available = [c for c in field_names if c in df.columns]
        df = df[available]

        # 类型转换
        for col in ["open", "high", "low", "close", "settle", "prev_settle", "bid", "ask"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["volume", "open_interest", "bid_vol", "ask_vol"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # 时间列
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        return df
