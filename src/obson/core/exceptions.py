"""核心异常"""


class ObsonError(Exception):
    """框架根异常"""
    pass


class DataSourceError(ObsonError):
    """数据源异常"""
    pass


class ProductNotFoundError(ObsonError):
    """商品未找到"""
    pass


class UnsupportedPeriodError(ObsonError):
    """不支持的周期"""
    pass


class UnsupportedFieldError(ObsonError):
    """不支持的字段"""
    pass


class NetworkError(ObsonError):
    """网络请求异常"""
    pass
