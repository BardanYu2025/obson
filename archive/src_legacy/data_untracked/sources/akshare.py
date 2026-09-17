"""AKShare 数据源封装"""

from __future__ import annotations

from typing import Any

from obson.core.exceptions import DataSourceError
from obson.core.models import DataResponse, Field, Period, Quote
from obson.data.sources.base import DataSource


class AKShareSource(DataSource):
    """AKShare 期货数据源 —— 需要安装 akshare"""

    name = "akshare"

    def __init__(self):
        try:
            import akshare as _ak
            self._ak = _ak
        except ImportError as e:
            raise ImportError(
                "使用 AKShare 数据源需要安装 akshare。"
                "运行: uv add akshare 或 pip install akshare"
            ) from e

    def is_available(self) -> bool:
        try:
            self._ak.futures_zh_realtime(symbol="RB0")
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

        # AKShare 的主力连续合约格式: 如 "RB0"
        # 历史K线接口
        try:
            # 尝试获取历史数据
            if period == Period.DAILY:
                df_hist = self._ak.futures_zh_daily(symbol=symbol, start_date="20000101")
            else:
                # AKShare 分钟线接口有限，这里用日线兜底
                df_hist = self._ak.futures_zh_daily(symbol=symbol, start_date="20000101")
        except Exception as e:
            raise DataSourceError(f"AKShare 获取 {symbol} K线失败: {e}") from e

        if df_hist.empty:
            raise DataSourceError(f"AKShare 返回 {symbol} 空数据")

        # 列名映射
        df_hist = df_hist.rename(columns={
            "date": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        df_hist = df_hist.tail(count)
        df_hist["datetime"] = df_hist["datetime"].astype(str)

        # 字段过滤
        field_names = [f.value for f in fields if f.value in df_hist.columns]
        df = df_hist[field_names]

        return DataResponse(
            product=kwargs.get("product", symbol),
            symbol=symbol,
            source=self.name,
            period=period,
            data=df,
            fields=fields,
        )

    def fetch_quote(self, symbol: str, **kwargs: Any) -> Quote:
        df = self._ak.futures_zh_realtime(symbol=symbol)
        if df.empty:
            raise DataSourceError(f"AKShare 未返回 {symbol} 数据")
        row = df.iloc[0]

        def _f(col: str) -> float:
            v = row.get(col, 0)
            return float(v) if v else 0.0

        def _i(col: str) -> int:
            v = row.get(col, 0)
            return int(float(v)) if v else 0

        cp = row.get("涨跌幅", "")
        change_pct = 0.0
        if isinstance(cp, str) and "%" in cp:
            change_pct = float(cp.replace("%", ""))
        elif isinstance(cp, (int, float)):
            change_pct = float(cp)

        return Quote(
            symbol=symbol,
            name=str(row.get("名称", symbol)),
            close=_f("最新价"),
            open=_f("开盘价"),
            high=_f("最高价"),
            low=_f("最低价"),
            prev_settle=_f("昨结"),
            bid=_f("买一"),
            ask=_f("卖一"),
            volume=_i("成交量"),
            open_interest=_i("持仓量"),
            source=self.name,
            raw=row.to_dict(),
        )

    def fetch_quotes(self, symbols: list[str], **kwargs: Any) -> list[Quote]:
        return [self.fetch_quote(s) for s in symbols]

    def close(self) -> None:
        pass
