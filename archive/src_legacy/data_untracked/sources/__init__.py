"""数据源封装"""

from obson.data.sources.base import DataSource
from obson.data.sources.sina import SinaFuturesSource

__all__ = ["DataSource", "SinaFuturesSource"]
