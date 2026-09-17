"""iFinD 数据源封装"""

from __future__ import annotations

from typing import Any

from obson.core.exceptions import DataSourceError
from obson.core.models import DataResponse, Field, Period, Quote
from obson.data.sources.base import DataSource


class iFinDSource(DataSource):
    """iFinD 金融数据源 —— 支持商品历史价格"""

    name = "ifind"

    def __init__(self):
        self._available = False
        try:
            import agent_gw
            self._gw = agent_gw
            self._available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._available

    def fetch_ohlcv(
        self,
        symbol: str,
        period: Period = Period.DAILY,
        count: int = 100,
        fields: list[Field] | None = None,
        **kwargs: Any,
    ) -> DataResponse:
        if not self._available:
            raise DataSourceError("iFinD SDK 未安装或不可用")

        if fields is None:
            fields = [Field.DATETIME, Field.OPEN, Field.HIGH, Field.LOW, Field.CLOSE, Field.VOLUME]

        # iFinD 的商品代码格式: AU9999.SHG (上期所黄金现货)
        # 这里 symbol 传的是期货代码如 AU0，需要转换
        ifind_symbol = self._to_ifind_symbol(symbol)

        # 计算日期范围
        from datetime import datetime, timedelta
        end = datetime.now()
        # iFinD 限制最多3年
        start = end - timedelta(days=min(count * 2, 1095))

        try:
            result = self._gw.call_data_source_tool(
                data_source_name="ifind",
                api_name="ifind_get_price",
                params={
                    "ticker": ifind_symbol,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d"),
                    "interval": "D" if period == Period.DAILY else "D",
                    "file_path": "/tmp/ifind_temp.csv",
                },
            )
            if not result.is_success:
                raise DataSourceError(f"iFinD 查询失败: {result.error}")
        except Exception as e:
            raise DataSourceError(f"iFinD 请求异常: {e}") from e

        import pandas as pd
        df = pd.read_csv("/tmp/ifind_temp.csv")
        df = df.rename(columns={
            col: col.lower()
            for col in df.columns
        })

        return DataResponse(
            product=kwargs.get("product", symbol),
            symbol=symbol,
            source=self.name,
            period=period,
            data=df,
            fields=fields,
        )

    def fetch_quote(self, symbol: str, **kwargs: Any) -> Quote:
        raise NotImplementedError("iFinD 暂不支持期货实时行情，请使用 sina 或 akshare")

    def fetch_quotes(self, symbols: list[str], **kwargs: Any) -> list[Quote]:
        raise NotImplementedError("iFinD 暂不支持期货实时行情")

    def _to_ifind_symbol(self, symbol: str) -> str:
        """期货代码转 iFinD 商品代码"""
        # 简化映射
        mapping = {
            "AU0": "AU9999.SHG",
            "AG0": "AG9999.SHG",
            "CU0": "CU9999.SHG",
            "RB0": "RB9999.SHG",
            "M0": "M9999.DCE",
        }
        return mapping.get(symbol, symbol)

    def close(self) -> None:
        pass
