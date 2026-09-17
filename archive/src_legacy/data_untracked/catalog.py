"""商品目录 —— 商品名称 ↔ 合约代码映射"""

from __future__ import annotations

from obson.core.exceptions import ProductNotFoundError

# ── 商品名称 → 主力连续合约代码 ──
PRODUCT_TO_SYMBOL: dict[str, str] = {
    # 上期所
    "黄金": "AU0", "au": "AU0", "gold": "AU0",
    "白银": "AG0", "ag": "AG0", "silver": "AG0",
    "铜": "CU0", "cu": "CU0", "copper": "CU0",
    "铝": "AL0", "al": "AL0", "aluminum": "AL0",
    "锌": "ZN0", "zn": "ZN0", "zinc": "ZN0",
    "铅": "PB0", "pb": "PB0", "lead": "PB0",
    "镍": "NI0", "ni": "NI0", "nickel": "NI0",
    "锡": "SN0", "sn": "SN0", "tin": "SN0",
    "不锈钢": "SS0", "ss": "SS0",
    "螺纹钢": "RB0", "rb": "RB0", "rebar": "RB0",
    "热卷": "HC0", "hc": "HC0", "hrc": "HC0",
    "铁矿石": "I0", "i": "I0", "ironore": "I0",
    "焦炭": "J0", "j": "J0", "coke": "J0",
    "焦煤": "JM0", "jm": "JM0",
    "玻璃": "FG0", "fg": "FG0", "glass": "FG0",
    "纯碱": "SA0", "sa": "SA0", "sodaash": "SA0",
    "纸浆": "SP0", "sp": "SP0", "pulp": "SP0",
    "橡胶": "RU0", "ru": "RU0", "rubber": "RU0",
    "20号胶": "NR0", "nr": "NR0",
    "沥青": "BU0", "bu": "BU0", "bitumen": "BU0",
    "燃料油": "FU0", "fu": "FU0", "fueloil": "FU0",
    "低硫燃料油": "LU0", "lu": "LU0",
    "原油": "SC0", "sc": "SC0", "crude": "SC0", "原油sc": "SC0",

    # 大商所
    "豆粕": "M0", "m": "M0", "soymeal": "M0",
    "豆油": "Y0", "y": "Y0", "soybeanoil": "Y0",
    "棕榈油": "P0", "p": "P0", "palmoil": "P0",
    "豆一": "A0", "a": "A0",
    "豆二": "B0", "b": "B0",
    "玉米": "C0", "c": "C0", "corn": "C0",
    "淀粉": "CS0", "cs": "CS0",
    "生猪": "LH0", "lh": "LH0", "livehog": "LH0",
    "乙二醇": "EG0", "eg": "EG0", "meg": "EG0",
    "聚丙烯": "PP0", "pp": "PP0",
    "塑料": "L0", "l": "L0", "pe": "L0",
    "pvc": "V0", "v": "V0",
    "苯乙烯": "EB0", "eb": "EB0", "styrene": "EB0",
    "lpg": "PG0", "pg": "PG0",
    "粳米": "RR0", "rr": "RR0",

    # 郑商所
    "棉花": "CF0", "cf": "CF0", "cotton": "CF0",
    "白糖": "SR0", "sr": "SR0", "sugar": "SR0",
    "pta": "TA0", "ta": "TA0",
    "甲醇": "MA0", "ma": "MA0", "methanol": "MA0",
    "动力煤": "ZC0", "zc": "ZC0", "thermalcoal": "ZC0",
    "尿素": "UR0", "ur": "UR0", "urea": "UR0",
    "菜粕": "RM0", "rm": "RM0", "rapeseedmeal": "RM0",
    "菜油": "OI0", "oi": "OI0", "rapeseedoil": "OI0",
    "苹果": "AP0", "ap": "AP0", "apple": "AP0",
    "红枣": "CJ0", "cj": "CJ0", "jujube": "CJ0",
    "硅铁": "SF0", "sf": "SF0", "ferrosilicon": "SF0",
    "锰硅": "SM0", "sm": "SM0", "silicomanganese": "SM0",
    "花生": "PK0", "pk": "PK0", "peanut": "PK0",

    # 中金所
    "沪深300": "IF0", "if": "IF0",
    "中证500": "IC0", "ic": "IC0",
    "上证50": "IH0", "ih": "IH0",
    "中证1000": "IM0", "im": "IM0",
    "10年国债": "T0", "t": "T0",
    "5年国债": "TF0", "tf": "TF0",
    "2年国债": "TS0", "ts": "TS0",
    "30年国债": "TL0", "tl": "TL0",

    # 广期所
    "工业硅": "SI0", "si": "SI0", "silicon": "SI0",
    "碳酸锂": "LC0", "lc": "LC0", "lithium": "LC0",
}

# 反向映射
SYMBOL_TO_PRODUCT: dict[str, str] = {
    v: k for k, v in PRODUCT_TO_SYMBOL.items() if len(k) <= 4 and "0" not in k
}


def resolve_symbol(name: str) -> str:
    """
    将商品名称/别名解析为标准合约代码

    :param name: 商品名称，如 "螺纹钢"、"rb"、"RB0"
    :return: 主力连续合约代码，如 "RB0"
    :raises ProductNotFoundError: 无法解析时抛出
    """
    key = name.strip()

    # 如果已经是标准代码（含数字），直接返回
    if any(d in key for d in "0123456789"):
        return key.upper()

    key_lower = key.lower()

    # 精确匹配
    if key_lower in PRODUCT_TO_SYMBOL:
        return PRODUCT_TO_SYMBOL[key_lower]

    # 大小写不敏感匹配中文
    for k, v in PRODUCT_TO_SYMBOL.items():
        if k.lower() == key_lower:
            return v

    # 模糊匹配（包含关系）
    matches = [(k, v) for k, v in PRODUCT_TO_SYMBOL.items() if key_lower in k.lower() or k.lower() in key_lower]
    if len(matches) == 1:
        return matches[0][1]

    raise ProductNotFoundError(
        f"无法识别商品 '{name}'。"
        f"{' 匹配到多个: ' + ', '.join(m[0] for m in matches) if len(matches) > 1 else ''}"
    )


def list_products() -> list[dict[str, str]]:
    """列出所有支持的商品"""
    seen = set()
    result = []
    for name, symbol in PRODUCT_TO_SYMBOL.items():
        if symbol not in seen and len(name) > 1 and not name.isascii():
            result.append({"name": name, "symbol": symbol})
            seen.add(symbol)
    return sorted(result, key=lambda x: x["symbol"])
