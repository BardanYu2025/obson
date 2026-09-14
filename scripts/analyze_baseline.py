import pandas as pd
import numpy as np

for period in [15, 30]:
    df = pd.read_csv(f'data/rebar_{period}m.csv')
    df['close_pct'] = df['close'].pct_change() * 100
    
    # 去掉NaN
    c = df['close_pct'].dropna()
    
    print(f"\n[{period}m] close_pct 统计 (N={len(c)}):")
    print(f"  均值: {c.mean():.4f}%")
    print(f"  标准差: {c.std():.4f}%")
    print(f"  |close_pct| 均值 (naive baseline MAE): {c.abs().mean():.4f}%")
    print(f"  最大涨: {c.max():.4f}% | 最大跌: {c.min():.4f}%")
    
    # 自相关：用前一根预测当前
    c_lag = c.shift(1).dropna()
    c_cur = c.iloc[1:]
    naive_mae = (c_cur - c_lag).abs().mean()
    
    # 方向准确率 naive
    same_dir = ((c_cur > 0) == (c_lag > 0)).mean()
    
    print(f"  naive baseline (prev->cur) MAE: {naive_mae:.4f}%")
    print(f"  naive baseline 方向准确率: {same_dir*100:.1f}%")
    print(f"  涨的比例: {(c > 0).mean()*100:.1f}%")
