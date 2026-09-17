"""统一数据调用接口 —— DataGateway

对上层屏蔽数据源差异，输入商品名称即可获取数据。
"""

from __future__ import annotations

from typing import Any

from obson.core.exceptions import DataSourceError, ProductNotFoundError, UnsupportedPeriodError
from obson.core.models import DataResponse, Field, Period, Quote
from obson.data.catalog import resolve_symbol
from obson.data.sources.akshare import AKShareSource
from obson.data.sources.base import DataSource
from obson.data.sources.ifind import iFinDSource
from obson.data.sources.sina import SinaFuturesSource


class DataGateway:
    """
    数据统一网关

    用法:
        gateway = DataGateway()

        # 获取K线 —— 输入商品名称即可
        df = gateway.get_ohlcv("螺纹钢", period="1d", count=20)

        # 获取实时行情
        quote = gateway.get_quote("黄金")

        # 批量行情
        quotes = gateway.get_quotes(["黄金", "白银", "原油"])
    """

    _SOURCES: dict[str, type[DataSource]] = {
        "sina": SinaFuturesSource,
        "akshare": AKShareSource,
        "ifind": iFinDSource,
    }

    def __init__(self, default_source: str = "sina", auto_fallback: bool = True):
        """
        :param default_source: 默认数据源，可选 sina / akshare / ifind
        :param auto_fallback: 主数据源失败时是否自动尝试其他源
        """
        self.default_source = default_source
        self.auto_fallback = auto_fallback
        self._instances: dict[str, DataSource] = {}

    # ── 核心接口 ──

    def get_ohlcv(
        self,
        product: str,
        period: str | Period = "1d",
        count: int = 100,
        fields: list[str] | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> DataResponse:
        """
        获取历史K线数据

        :param product: 商品名称，如 "螺纹钢"、"黄金"、"原油"
        :param period: 周期，支持 1d / 5m / 15m / 30m / 60m
        :param count: 返回条数
        :param fields: 指定字段，如 ["open", "high", "low", "close", "volume"]
        :param source: 指定数据源，None 用默认
        :return: DataResponse
        """
        symbol = resolve_symbol(product)
        period_enum = Period(period) if isinstance(period, str) else period

        field_enums = None
        if fields:
            field_enums = [Field(f) for f in fields]

        src_name = source or self.default_source

        # 尝试主数据源
        try:
            src = self._get_source(src_name)
            return src.fetch_ohlcv(
                symbol=symbol,
                period=period_enum,
                count=count,
                fields=field_enums,
                product=product,
                **kwargs,
            )
        except Exception as e:
            if not self.auto_fallback:
                raise

        # 自动降级
        for fallback_name in ["akshare", "sina"]:
            if fallback_name == src_name:
                continue
            try:
                src = self._get_source(fallback_name)
                return src.fetch_ohlcv(
                    symbol=symbol,
                    period=period_enum,
                    count=count,
                    fields=field_enums,
                    product=product,
                    **kwargs,
                )
            except Exception:
                continue

        raise DataSourceError(
            f"所有数据源均无法获取 {product}({symbol}) 的 {period} 数据。"
            f"最后错误: {e}"
        )

    def get_quote(self, product: str, source: str | None = None, **kwargs: Any) -> Quote:
        """
        获取实时行情

        :param product: 商品名称，如 "螺纹钢"
        :param source: 指定数据源
        :return: Quote
        """
        symbol = resolve_symbol(product)
        src_name = source or self.default_source

        try:
            src = self._get_source(src_name)
            return src.fetch_quote(symbol, **kwargs)
        except Exception as e:
            if not self.auto_fallback:
                raise

        for fallback_name in ["akshare", "sina"]:
            if fallback_name == src_name:
                continue
            try:
                src = self._get_source(fallback_name)
                return src.fetch_quote(symbol, **kwargs)
            except Exception:
                continue

        raise DataSourceError(f"无法获取 {product}({symbol}) 实时行情")

    def get_quotes(
        self,
        products: list[str],
        source: str | None = None,
        **kwargs: Any,
    ) -> list[Quote]:
        """
        批量获取实时行情

        :param products: 商品名称列表，如 ["黄金", "螺纹钢", "原油"]
        :param source: 指定数据源
        :return: Quote 列表
        """
        symbols = [resolve_symbol(p) for p in products]
        src_name = source or self.default_source

        try:
            src = self._get_source(src_name)
            return src.fetch_quotes(symbols, **kwargs)
        except Exception:
            if not self.auto_fallback:
                raise

        for fallback_name in ["akshare", "sina"]:
            if fallback_name == src_name:
                continue
            try:
                src = self._get_source(fallback_name)
                return src.fetch_quotes(symbols, **kwargs)
            except Exception:
                continue

        raise DataSourceError(f"无法批量获取行情: {products}")

    def available_sources(self) -> list[str]:
        """返回当前可用的数据源列表"""
        available = []
        for name, cls in self._SOURCES.items():
            try:
                src = self._get_source(name)
                if src.is_available():
                    available.append(name)
            except Exception:
                pass
        return available

    # ── 内部 ──

    def _get_source(self, name: str) -> DataSource:
        """获取或创建数据源实例"""
        if name not in self._instances:
            if name not in self._SOURCES:
                raise DataSourceError(f"未知数据源: {name}。可用: {list(self._SOURCES.keys())}")
            self._instances[name] = self._SOURCES[name]()
        return self._instances[name]

    def close(self) -> None:
        for src in self._instances.values():
            src.close()
        self._instances.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
