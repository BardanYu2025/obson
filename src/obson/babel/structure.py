"""Observable structure, with separate extreme and confirmation timestamps.

All arrays are prefix invariant. No annotation is backfilled at an extreme.
Rules are explicit baselines/weak supervision, never a claim of market prediction.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SCALES = (0.75, 1.5, 3.0)
TIME_EDGES = np.array([2, 4, 8, 16, 32, 64])
AMP_EDGES = np.array([-3, -1.5, -0.5, 0.5, 1.5, 3])
STATE_NAMES = ("未形成", "上升结构", "下降结构", "区间震荡")


@dataclass(frozen=True)
class Pivot:
    scale: int
    kind: int  # 1 high, -1 low
    event: int
    confirmed: int
    price: float


def causal_atr(df, span=14):
    prev = df.close.shift(1).fillna(df.close.iloc[0])
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(span=span, adjust=False).mean().clip(lower=1e-6).to_numpy()


def annotate(df):
    """Single forward pass; outputs at t are only functions of df[:t+1].

    Outside bars use an explicit conservative convention: an updated extreme
    cannot also be confirmed on that bar, since intrabar order is unknown.
    """
    n = len(df)
    a = causal_atr(df)
    hi, lo, close = (df[c].to_numpy() for c in ("high", "low", "close"))
    direction = np.ones((n, 3), dtype=np.int64)  # 0 down / 1 unknown / 2 up
    age = np.zeros((n, 3), np.float32)
    amplitude = np.zeros((n, 3), np.float32)
    event = np.zeros((n, 3), np.int64)  # confirmation now: 0 none / 1 high / 2 low
    state = np.zeros((n, 3), np.int64)
    levels = np.full((n, 3, 2), np.nan)  # last confirmed low / high
    pivots = []
    for s, k in enumerate(SCALES):
        mode, ch, cl, ih, il = 0, hi[0], lo[0], 0, 0
        last = None
        highs, lows = [], []
        for t in range(1, n):
            p = None
            if mode == 0:
                old_hi, old_lo = ch, cl
                if hi[t] >= ch:
                    ch, ih = hi[t], t
                if lo[t] <= cl:
                    cl, il = lo[t], t
                down = ih < t and old_hi - lo[t] >= k * a[t]
                up = il < t and hi[t] - old_lo >= k * a[t]
                if down and not up:
                    p = Pivot(s, 1, ih, t, ch)
                elif up and not down:
                    p = Pivot(s, -1, il, t, cl)
            elif mode == 1:
                if hi[t] >= ch:
                    ch, ih = hi[t], t
                elif ch - lo[t] >= k * a[t]:
                    p = Pivot(s, 1, ih, t, ch)
            else:
                if lo[t] <= cl:
                    cl, il = lo[t], t
                elif hi[t] - cl >= k * a[t]:
                    p = Pivot(s, -1, il, t, cl)
            if p is not None:
                pivots.append(p)
                last = p
                mode = -p.kind
                ch, cl, ih, il = hi[t], lo[t], t, t
                (highs if p.kind == 1 else lows).append(p.price)
                event[t, s] = 1 if p.kind == 1 else 2
            if last is not None:
                direction[t, s] = mode + 1
                age[t, s] = t - last.event
                amplitude[t, s] = (close[t] - last.price) / a[t]
            if lows:
                levels[t, s, 0] = lows[-1]
            if highs:
                levels[t, s, 1] = highs[-1]
            if len(highs) >= 2 and len(lows) >= 2:
                hh, ll = highs[-1] > highs[-2], lows[-1] > lows[-2]
                state[t, s] = 1 if hh and ll else 2 if not hh and not ll else 3
    rolling_range = (
        df.high.rolling(20, min_periods=5).max() - df.low.rolling(20, min_periods=5).min()
    ) / a
    efficiency = df.close.diff(20).abs() / df.close.diff().abs().rolling(20).sum().clip(lower=1e-6)
    slow = pd.Series(a).rolling(64, min_periods=16).mean()
    compression = (a / slow.to_numpy()) < 0.75
    return {
        "atr": a,
        "direction": direction,
        "age": age,
        "amplitude": amplitude,
        "event": event,
        "state": state,
        "levels": levels,
        "pivots": pivots,
        "compression": compression,
        "efficiency": efficiency.fillna(0).to_numpy(),
        "range_atr": rolling_range.fillna(0).to_numpy(),
    }


def describe(df, labels, row):
    """Human-readable current evidence, never a forecast or trade instruction."""
    a, price = float(labels["atr"][row]), float(df.close.iloc[row])
    known = [p for p in labels["pivots"] if p.confirmed <= row]
    scales = []
    for s, k in enumerate(SCALES):
        support, resistance = labels["levels"][row, s]
        scales.append(
            {
                "scale": k,
                "state": STATE_NAMES[labels["state"][row, s]],
                "leg": ("下降摆动", "未确认", "上升摆动")[labels["direction"][row, s]],
                "bars_since_extreme": int(labels["age"][row, s]),
                "amplitude_atr": float(labels["amplitude"][row, s]),
                "support": float(support) if np.isfinite(support) else None,
                "resistance": float(resistance) if np.isfinite(resistance) else None,
            }
        )
    patterns = []
    for s in range(3):
        ps = [p for p in known if p.scale == s]
        if len(ps) >= 3:
            x, y, z = ps[-3:]
            if x.kind == z.kind and abs(x.price - z.price) <= 0.6 * a:
                broken = price < y.price if z.kind == 1 else price > y.price
                patterns.append(
                    {
                        "name": "双顶" if z.kind == 1 else "双底",
                        "scale": s,
                        "status": "颈线已突破" if broken else "候选，颈线未突破",
                        "neckline": y.price,
                        "invalidation": max(x.price, z.price)
                        if z.kind == 1
                        else min(x.price, z.price),
                    }
                )
        if len(ps) >= 5:
            x, y, z, u, v = ps[-5:]
            sign = x.kind
            if (
                x.kind == z.kind == v.kind
                and abs(x.price - v.price) <= a
                and sign
                * (
                    z.price - max(x.price, v.price)
                    if sign == 1
                    else z.price - min(x.price, v.price)
                )
                >= 0.5 * a
            ):
                neck = y.price + (u.price - y.price) * (row - y.event) / max(u.event - y.event, 1)
                broken = price < neck if sign == 1 else price > neck
                patterns.append(
                    {
                        "name": "头肩顶" if sign == 1 else "头肩底",
                        "scale": s,
                        "status": "颈线已突破" if broken else "候选，颈线未突破",
                        "neckline": float(neck),
                        "invalidation": z.price,
                    }
                )
    # Previously confirmed levels, so the event is meaningful at this bar.
    alerts = []
    if row > 0:
        prev = float(df.close.iloc[row - 1])
        for s in range(3):
            support, resistance = labels["levels"][row - 1, s]
            if np.isfinite(resistance) and prev <= resistance < price:
                alerts.append(f"尺度{s}：收盘越过此前确认高点 {resistance:.2f}")
            if np.isfinite(support) and prev >= support > price:
                alerts.append(f"尺度{s}：收盘跌破此前确认低点 {support:.2f}")
            if labels["event"][row, s]:
                alerts.append(
                    f"尺度{s}：新确认一个{'高' if labels['event'][row, s] == 1 else '低'}点"
                )
    return {
        "scales": scales,
        "patterns": patterns,
        "alerts": alerts,
        "compression": bool(labels["compression"][row]),
        "efficiency": float(labels["efficiency"][row]),
        "pivots": [asdict(p) for p in known[-120:]],
    }


def descriptor(labels, row):
    """Explicit, inspectable retrieval baseline; signed geometry and multi-scale state."""
    d = labels["direction"][row] - 1
    amp = np.clip(labels["amplitude"][row] / 5, -2, 2)
    age = np.log1p(labels["age"][row]) / np.log(65)
    onehot = np.eye(4)[labels["state"][row]].ravel()
    return np.r_[
        d,
        amp,
        age,
        onehot,
        float(labels["compression"][row]),
        labels["efficiency"][row],
        labels["range_atr"][row] / 10,
    ].astype(np.float32)
