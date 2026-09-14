"""数据层端到端测试"""

import sys
from pathlib import Path

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from obson.data.catalog import list_products, resolve_symbol
from obson.data.unified import DataGateway


def test_catalog():
    print("=== 测试商品目录 ===\n")

    # 名称解析
    cases = [
        ("螺纹钢", "RB0"),
        ("rb", "RB0"),
        ("黄金", "AU0"),
        ("au", "AU0"),
        ("原油", "SC0"),
        ("豆粕", "M0"),
        ("IF0", "IF0"),   # 已经是代码
    ]
    for name, expected in cases:
        result = resolve_symbol(name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name:8s} → {result}")
    print()

    # 列出支持的商品
    products = list_products()
    print(f"  共支持 {len(products)} 个主力品种")
    print(f"  示例: {', '.join(p['name'] for p in products[:5])}...")
    print()


def test_sina_quotes():
    print("=== 测试新浪实时行情 ===\n")

    gateway = DataGateway(default_source="sina")

    # 单合约
    print("1. 单合约: 黄金")
    q = gateway.get_quote("黄金")
    print(f"   代码: {q.symbol}, 名称: {q.name}")
    print(f"   最新: {q.close}, 涨跌: {q.change_pct:+.2f}%")
    print(f"   开盘: {q.open}, 最高: {q.high}, 最低: {q.low}")
    print(f"   持仓: {q.open_interest}, 成交: {q.volume}")
    print()

    # 批量
    print("2. 批量: 黄金, 螺纹钢, 豆粕, 棉花")
    quotes = gateway.get_quotes(["黄金", "螺纹钢", "豆粕", "棉花"])
    for q in quotes:
        print(f"   {q.name:10s} {q.symbol:6s} 最新: {q.close:>10.2f} 涨跌: {q.change_pct:>+7.2f}%")
    print()

    gateway.close()


def test_sina_kline():
    print("=== 测试新浪K线数据 ===\n")

    gateway = DataGateway(default_source="sina")

    # 日K
    print("1. 螺纹钢 日K (最近10条)")
    resp = gateway.get_ohlcv("螺纹钢", period="1d", count=10)
    print(f"   数据源: {resp.source}, 返回 {len(resp.data)} 条")
    print(resp.data.tail(3).to_string(index=False))
    print()

    # 5分钟K
    print("2. 黄金 5分钟K (最近5条)")
    resp = gateway.get_ohlcv("黄金", period="5m", count=5)
    print(f"   数据源: {resp.source}, 返回 {len(resp.data)} 条")
    print(resp.data.tail(3).to_string(index=False))
    print()

    gateway.close()


def test_dataframe_api():
    print("=== 测试 DataFrame 操作 ===\n")

    gateway = DataGateway()
    resp = gateway.get_ohlcv("螺纹钢", period="1d", count=30)

    df = resp.data
    print(f"数据维度: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print()
    print("最近5条:")
    print(df.tail().to_string(index=False))
    print()

    # 简单统计
    if "close" in df.columns:
        print(f"收盘价统计:")
        print(f"  均值: {df['close'].mean():.2f}")
        print(f"  最高: {df['close'].max():.2f}")
        print(f"  最低: {df['close'].min():.2f}")
    print()

    gateway.close()


def test_available_sources():
    print("=== 测试数据源可用性 ===\n")

    gateway = DataGateway()
    sources = gateway.available_sources()
    print(f"  可用数据源: {sources}")
    print()

    gateway.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Obson 数据层端到端测试")
    print("=" * 60)
    print()

    test_catalog()
    test_available_sources()
    test_sina_quotes()
    test_sina_kline()
    test_dataframe_api()

    print("=" * 60)
    print("全部测试完成")
    print("=" * 60)
