"""Deterministic fitting on train-time, non-held contracts only."""

from __future__ import annotations

import numpy as np

from .views import HORIZONS, blank_stats, description, sample


def finish(buffer, width):
    if not buffer:
        return np.zeros(width), np.ones(width)
    arr = np.vstack(buffer)
    mu, sd = np.zeros(width), np.ones(width)
    for k in range(width):
        v = arr[np.isfinite(arr[:, k]), k]
        if len(v):
            mu[k], sd[k] = v.mean(), max(v.std(), 1e-5)
    return mu, sd


def median(values, default=1.0):
    a = np.asarray(values)
    good = np.abs(a[np.isfinite(a) & (a != 0)])
    return float(np.median(good)) if len(good) else default


def fit(corpus, n):
    rng = np.random.default_rng(1847)
    clean = [corpus.draw(0, 0, rng) for _ in range(n)]
    desc = [corpus.draw(0, 2, rng) for _ in range(n)]
    stats = blank_stats()
    delta, bd, rms = ([[] for _ in range(4)] for _ in range(3))
    hd = [[] for _ in HORIZONS]
    for sid, end in clean:
        s = sample(corpus.shard(sid), end, stats)
        for p in range(4):
            ix = np.flatnonzero(s.period == p)[::8]
            delta[p].extend(s.physical[ix, 6] - s.physical[ix, 5])
            warm = ix[(s.x[ix, 27] == 0) & (s.x[ix, 8] > np.log(1e-6) + 1e-6)]
            rms[p].extend(np.exp(s.x[warm, 8]))
        for g in np.flatnonzero(s.facts[:, 4]):
            take = np.where(s.block == g)[0]
            bd[s.period[take[0]]].append(
                s.physical[take[-1], 6] - s.physical[take[0], 5]
            )
    for sid, end in desc:
        v = corpus.shard(sid).bars[0]["v"]
        for k, h in enumerate(HORIZONS):
            hd[k].append(v[end, 6] - v[end - h + 1, 5])
    stats["desc_delta"] = np.array([max(1.0, median(a)) for a in hd])
    for p in range(4):
        stats["delta"][p] = max(1.0, median(delta[p]))
        stats["block_delta"][p] = max(1.0, median(bd[p]))
        stats["floor"][p] = (
            max(1e-6, np.quantile(rms[p], 0.05)) if len(rms[p]) else 1e-6
        )
        stats["fallback"][p] = max(stats["floor"][p], median(rms[p], 0.001))
    xx, yy = ([[] for _ in range(4)] for _ in range(2))
    for sid, end in clean:
        s = sample(corpus.shard(sid), end, stats)
        for p in range(4):
            slots = np.flatnonzero(s.period == p)
            # Fixed cap on calibration observations per endpoint and view.
            ix = rng.choice(slots, min(16, len(slots)), replace=False)
            xx[p].append(np.where(s.x[ix, 9:18] > 0, s.x[ix, :9], np.nan))
            yy[p].append(s.target[ix])
    for p in range(4):
        stats["x_mu"][p], stats["x_sd"][p] = finish(xx[p], 9)
        stats["y_mu"][p], stats["y_sd"][p] = finish(yy[p], 9)
    facts = [[] for _ in range(4)]
    for sid, end in clean:
        s = sample(corpus.shard(sid), end, stats)
        for p, (a, b) in enumerate(((0, 16), (16, 24), (24, 28), (28, 32))):
            valid = s.facts[a:b, 4].astype(bool)
            if valid.any():
                facts[p].append(s.facts[a:b, :4][valid])
    for p in range(4):
        mean, sd = finish(facts[p], 4)
        stats["f_mu"][p], stats["f_on"][p] = mean, float(bool(facts[p]))
        stats["f_sd"][p] = (
            np.maximum(np.vstack(facts[p]).astype(float).std(0), 1e-6)
            if facts[p]
            else sd
        )
    stats["d_mu"], stats["d_sd"] = finish(
        [description(corpus.shard(sid), r, stats) for sid, r in desc], 18
    )
    return stats
