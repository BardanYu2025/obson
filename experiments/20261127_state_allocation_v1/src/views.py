"""Construct the common visible input, addressed queries, and loss-only truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import CAPS

U = np.array([[-1] * 8 + [1] * 8, [-1] * 4 + [1] * 8 + [-1] * 4], dtype=float).T / 4
HORIZONS = (16, 64, 256)


def blank_stats():
    return {
        k: np.full(shape, value, dtype=float)
        for k, shape, value in (
            ("floor", (4,), 1e-6),
            ("fallback", (4,), 0.001),
            ("delta", (4,), 1.0),
            ("block_delta", (4,), 1.0),
            ("desc_delta", (3,), 1.0),
            ("x_mu", (4, 9), 0.0),
            ("x_sd", (4, 9), 1.0),
            ("y_mu", (4, 9), 0.0),
            ("y_sd", (4, 9), 1.0),
            ("f_mu", (4, 4), 0.0),
            ("f_sd", (4, 4), 1.0),
            ("f_on", (4,), 1.0),
            ("d_mu", (18,), 0.0),
            ("d_sd", (18,), 1.0),
        )
    }


def spans_for_mask(shard, end, rng):
    minute = shard.bars[0]
    start = max(0, end - 255)
    times = shard.time[start:end]
    good = np.isfinite(minute["v"][start:end, 3])
    if len(times) < 16:
        raise ValueError("No legal mask pair")
    windows = np.lib.stride_tricks.sliding_window_view
    starts = np.flatnonzero(windows(good, 8).all(1) & (times[7:] - times[:-7] == 7))
    second = np.searchsorted(starts, starts + 8)
    count = np.maximum(0, len(starts) - second)
    pick = int(rng.integers(count.sum()))
    first = int(np.searchsorted(np.cumsum(count), pick, side="right"))
    offset = pick - int(count[:first].sum())
    return start + starts[[first, second[first] + offset]]


def description(shard, end, stats):
    raw = shard.bars[0]["v"]
    if end < 256 or not shard.flags[end, 3]:
        return np.full(18, np.nan)
    out = []
    for j, horizon in enumerate(HORIZONS):
        vals = raw[end - horizon : end + 1]
        logc = np.log(vals[:, 3])
        delta = np.diff(logc)
        net, travel = logc[-1] - logc[0], np.abs(delta).sum()
        out.extend(
            (
                net,
                abs(net) / travel if travel else 0.0,
                np.linalg.norm(delta),
                (np.maximum.accumulate(logc) - logc).max(),
                np.log1p(vals[1:, 4].sum()),
                np.arcsinh((vals[-1, 6] - vals[1, 5]) / stats["desc_delta"][j]),
            )
        )
    return np.array(out)


def make_features(table, hidden, stats, period):
    """The synthetic removal acts on raw-minute support, before any price transform."""
    v = table["v"].copy()
    a, b = table["a"], table["b"]
    affected = np.zeros(v.shape, dtype=bool)
    for start in hidden:
        stop = start + 7
        first = (start <= a) & (a <= stop)
        last = (start <= b) & (b <= stop)
        touch = (a <= stop) & (b >= start)
        affected |= np.stack((first, touch, touch, last, touch, first, last), axis=1)
    v[affected] = np.nan
    o, h, low, c, vol, oi0, oi1 = v.T
    prev = np.r_[np.nan, c[:-1]]
    prev[~table["link"]] = np.nan
    gap, body = np.log(o / prev), np.log(c / o)
    upper, lower = np.log(h / np.maximum(o, c)), np.log(np.minimum(o, c) / low)
    ret = np.log(c / prev)
    h_close = affected[:, 3]
    prev_hidden = np.r_[False, h_close[:-1]] & table["link"]
    dep_ret = h_close | prev_hidden
    idx = np.arange(len(v))

    def prefix(values):
        return np.r_[0, np.cumsum(values)]

    left = np.maximum(0, idx - 64)
    valid = np.isfinite(ret)
    count = prefix(valid)[idx] - prefix(valid)[left]
    sums = prefix(np.where(valid, ret * ret, 0.0))
    volat = np.maximum(
        stats["floor"][period], np.sqrt(np.maximum(0.0, sums[idx] - sums[left]) / 64)
    )
    cold = count < 64
    volat[cold] = stats["fallback"][period]
    deps = prefix(dep_ret)
    affected_scale = deps[idx] > deps[left]
    fields = np.column_stack(
        (
            gap / volat,
            body / volat,
            upper / volat,
            lower / volat,
            np.log1p(vol),
            np.log1p(oi0),
            np.log1p(oi1),
            np.arcsinh((oi1 - oi0) / stats["delta"][period]),
            np.log(volat),
        )
    )
    flags = np.column_stack(
        (
            affected[:, 0] | prev_hidden,
            affected[:, 0] | h_close,
            affected[:, 1] | affected[:, 0] | h_close,
            affected[:, 2] | affected[:, 0] | h_close,
            affected[:, 4],
            affected[:, 5],
            affected[:, 6],
            affected[:, 5] | affected[:, 6],
            affected_scale,
        )
    )
    flags[:, :4] |= affected_scale[:, None]
    visible = np.isfinite(fields)
    x = np.where(visible, (fields - stats["x_mu"][period]) / stats["x_sd"][period], 0.0)
    return np.column_stack((x, visible, flags, cold)).astype("f4"), count / 64


@dataclass
class Sample:
    x: np.ndarray
    context: np.ndarray
    period: np.ndarray
    geom: np.ndarray
    valid: np.ndarray
    query: np.ndarray
    block: np.ndarray
    slot: np.ndarray
    previous: np.ndarray
    current: np.ndarray
    age: np.ndarray
    target: np.ndarray
    physical: np.ndarray
    facts: np.ndarray
    hidden: np.ndarray
    desc: np.ndarray
    anchor: float
    reachable: np.ndarray
    visible_close: np.ndarray


def block_facts(x, period, group, phase, stats):
    facts = np.zeros((32, 5), dtype="f4")
    for g in np.unique(group[group >= 0]):
        take = np.flatnonzero(group == g)
        if len(take) != 16 or not np.array_equal(phase[take], np.arange(16)):
            continue
        p = period[take[0]]
        vis = x[take, 9:18].astype(bool)
        if (
            not stats["f_on"][p]
            or not vis[1:, :2].all()
            or not vis[:, 4].all()
            or not vis[0, 5]
            or not vis[-1, 6]
            or not vis[:, 8].all()
        ):
            continue
        common = x[take, :9].astype("f8") * stats["x_sd"][p] + stats["x_mu"][p]
        path = np.r_[0.0, np.cumsum(common[1:, :2].sum(1) * np.exp(common[1:, 8]))]
        volume = np.expm1(common[:, 4]).sum()
        d_oi = np.expm1(common[-1, 6]) - np.expm1(common[0, 5])
        facts[g] = [
            *(path @ U),
            np.log1p(max(0.0, volume)),
            np.arcsinh(d_oi / stats["block_delta"][p]),
            1.0,
        ]
    return facts


def sample(shard, end, stats, hidden=()):
    clock = shard.time
    anchor = float(shard.bars[0]["v"][end, 3])
    if not np.isfinite(anchor) or anchor <= 0:
        raise ValueError("Endpoint lacks a current close")
    stack = []
    offset = base = 0
    for p, (bars, cap) in enumerate(zip(shard.bars, CAPS)):
        stop = int(np.searchsorted(bars["b"], end, side="right"))
        first = max(0, stop - cap)
        warm = max(0, first - 65)
        if first == stop:
            base += cap // 16
            continue
        table = bars[warm:stop]
        x, warmup = make_features(table, hidden, stats, p)
        trim = first - warm
        x, warmup, table = x[trim:], warmup[trim:], table[trim:]
        a, b = table["a"], table["b"]
        period = np.full(len(b), p)
        age, wall_age = end - b, clock[end] - clock[b]
        current = age == 0
        rank = np.arange(len(b) - 1, -1, -1) - int(current[-1])
        group = np.where(current, -1, base + rank // 16)
        phase = 15 - rank % 16
        duration, wall = b + 1 - a, clock[b] + 1 - clock[a]
        times = np.column_stack(
            (np.log1p(age), np.log1p(wall_age), np.log1p(duration), np.log1p(wall))
        )
        angle = (clock[b] + 1) % 1440 * (2 * np.pi / 1440)
        context = np.column_stack(
            (times, table["coverage"], warmup, np.sin(angle), np.cos(angle))
        )
        circle = 2 * np.pi * phase / 16
        query = np.column_stack(
            (
                times,
                np.eye(4)[period],
                np.where(current, 0.0, np.sin(circle)),
                np.where(current, 0.0, np.cos(circle)),
            )
        )
        # Coordinates relative to t keep integer minute differences exact in FP32.
        geom = np.column_stack(
            (
                a - end - 1,
                b - end,
                clock[a] - clock[end] - 1,
                clock[b] - clock[end],
                table["coverage"],
            )
        )
        v = table["v"]
        o, h, low, c, volume, oi0, oi1 = v.T
        previous = np.arange(offset - 1, offset + len(b) - 1)
        previous[~table["link"]] = -1
        previous[0] = -1
        prev_c = np.r_[np.nan, c[:-1]]
        prev_c[previous < 0] = np.nan
        y = np.column_stack(
            (
                np.log(c / anchor),
                np.log(c / prev_c),
                np.log(c / o),
                np.log(h / np.maximum(o, c)),
                np.log(np.minimum(o, c) / low),
                np.log1p(volume),
                np.log1p(oi0),
                np.log1p(oi1),
                np.arcsinh((oi1 - oi0) / stats["delta"][p]),
            )
        )
        marked = np.zeros(len(b), bool)
        if p == 0:
            for m in hidden:
                marked |= (a >= m) & (a < m + 8)
        stack.append(
            (
                x,
                context,
                period,
                geom,
                table["coverage"] > 0,
                query,
                group,
                phase,
                previous,
                current,
                age,
                y,
                v,
                marked,
                b,
            )
        )
        offset += len(b)
        base += cap // 16
    x, ctx, p, geom, valid, q, g, j, prev, cur, age, y, raw, mark, ends = map(
        np.concatenate, zip(*stack)
    )
    facts = block_facts(x, p, g, j, stats)
    # Share closes at identical physical boundaries; traverse visible consecutive price links.
    nodes = np.unique(ends)
    ids = np.searchsorted(nodes, ends)
    graph = [[] for _ in nodes]
    physical = x[:, :9].astype(float) * stats["x_sd"][p] + stats["x_mu"][p]
    returns = physical[:, :2].sum(1) * np.exp(physical[:, 8])
    for k in np.flatnonzero((prev >= 0) & (x[:, 9] > 0) & (x[:, 10] > 0)):
        old, new = ids[prev[k]], ids[k]
        graph[old].append((new, returns[k]))
        graph[new].append((old, -returns[k]))
    anchor_node = np.searchsorted(nodes, end)
    price = np.full(len(nodes), np.nan)
    price[anchor_node] = 0.0
    queue = [anchor_node]
    for node in queue:
        for other, delta in graph[node]:
            if not np.isfinite(price[other]):
                price[other] = price[node] + delta
                queue.append(other)
    visible_close = price[ids]
    reachable = np.isfinite(visible_close) & np.isfinite(y[:, 0])
    return Sample(
        x,
        ctx.astype("f4"),
        p,
        geom.astype("f4"),
        valid,
        q.astype("f4"),
        g,
        j,
        prev,
        cur,
        age,
        y,
        raw,
        facts,
        mark,
        description(shard, end, stats),
        anchor,
        reachable,
        visible_close,
    )
