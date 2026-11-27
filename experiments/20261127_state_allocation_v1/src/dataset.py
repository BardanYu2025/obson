"""Build portable, read-only shards from the frozen month-contract Parquet files."""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .calendar_rules import ages, boundaries, session_spans, split_codes
from .common import (
    CAPS,
    HELD,
    PERIODS,
    ROOT,
    digest,
    exclusive,
    identity,
    json_file,
    read_json,
)

FIELDS = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
BAR = np.dtype(
    [("a", "<i4"), ("b", "<i4"), ("link", "?"), ("coverage", "<f4"), ("v", "<f8", 7)]
)
CUTOFF = int(np.datetime64("2026-09-12T03:00", "m").astype(int))


def verified_bytes(path, expected):
    import hashlib

    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Source SHA256 mismatch: {path}")
    return data


def minute_times(table):
    ns = table["datetime_nano"].to_numpy()
    wall = ns // 60_000_000_000 + 480
    text = pc.cast(table["datetime"], pa.timestamp("s")).cast(pa.int64()).to_numpy()
    if (
        not len(ns)
        or (ns % 60_000_000_000).any()
        or (np.diff(ns) <= 0).any()
        or not np.array_equal(text, wall * 60)
    ):
        raise ValueError("Minute starts must be unique and ordered in Shanghai time")
    return wall


def calendar_grid(month, main, financial):
    reference = main[(main >= month[0]) & (main <= month[-1])]
    witnessed = np.union1d(reference, month)
    origins, limits = session_spans(witnessed, financial)
    if (origins < 0).any():
        raise ValueError("Timestamp outside the frozen session grid")
    starts = np.r_[0, np.flatnonzero(np.diff(origins)) + 1]
    stops = np.r_[starts[1:], len(witnessed)]
    # A missing minute inside a witnessed session keeps its place. A file edge
    # never grants access to another contract or a session outside its window.
    segments = []
    for a, b in zip(starts, stops):
        origin = origins[a]
        day = (origin % 1440) < 1260
        stop = min(limits[a], month[-1] + 1) if day else witnessed[b - 1] + 1
        segments.append(np.arange(max(origin, month[0]), stop))
    return np.concatenate(segments)


def aggregate(grid, raw, observed, strict, financial, period):
    origin, limit = session_spans(grid, financial)
    left = origin + (grid - origin) // period * period
    right = np.minimum(left + period, limit)
    starts = np.r_[0, np.flatnonzero(np.diff(left)) + 1]
    ends = np.r_[starts[1:] - 1, len(grid) - 1]
    # Unknown final night hours give no partial night close. Daytime tails
    # have a fixed scheduled close.
    closed = grid[ends] + 1 == right[starts]
    a, b = starts[closed], ends[closed]
    output = np.zeros(len(a), dtype=BAR)
    output["a"], output["b"] = a, b
    count = np.r_[0, np.cumsum(observed)]
    scheduled = right[starts[closed]] - left[starts[closed]]
    output["coverage"] = (count[b + 1] - count[a]) / scheduled
    full = (output["coverage"] == 1) & (grid[a] == left[starts[closed]])
    v = np.column_stack(
        (
            raw[a, 0],
            np.maximum.reduceat(raw[:, 1], starts)[closed],
            np.minimum.reduceat(raw[:, 2], starts)[closed],
            raw[b, 3],
            np.add.reduceat(raw[:, 4], starts)[closed],
            raw[a, 5],
            raw[b, 6],
        )
    )
    v[~full] = np.nan
    output["v"] = v
    if len(a) > 1:
        output["link"][1:] = (
            (a[1:] == b[:-1] + 1) & strict[a[1:]] & full[1:] & full[:-1]
        )
    return output


def pool_flags(grid, raw, bars, strict, financial, held):
    idx = np.arange(len(grid))
    part = split_codes(grid, held, CUTOFF)
    part[grid < np.datetime64("2016-01-01", "m").astype(int)] = 8
    floor = np.maximum.accumulate(np.where(np.r_[True, np.diff(part) != 0], idx, 0))
    pure = np.ones(len(grid), dtype=bool)
    for view, cap in zip(bars, CAPS):
        stop = np.searchsorted(view["b"], idx, side="right")
        start = np.maximum(0, stop - cap - 65)
        if len(view):
            pure &= (stop == 0) | (view["a"][np.minimum(start, len(view) - 1)] >= floor)
    ok = np.isfinite(raw[:, :4]).all(axis=1)
    clean = pure & ok & (grid >= np.datetime64("2016-01-01", "m").astype(int))
    segment = session_spans(grid, financial)[0]
    run = ages(
        np.r_[False, np.diff(grid) == 1]
        & np.r_[False, segment[1:] == segment[:-1]]
        & np.r_[False, ok[:-1]],
        ok,
    )
    ends = np.flatnonzero((run >= 8) & ok)
    starts = ends - 7
    recent = np.searchsorted(ends, idx) - 1
    mask = np.zeros(len(idx), dtype=bool)
    if len(ends):
        e = starts[np.maximum(recent, 0)]
        other = np.searchsorted(ends, e) - 1
        mask = (
            (other >= 0)
            & (recent >= 0)
            & (starts[np.maximum(other, 0)] >= np.maximum(idx - 255, floor))
        )
    connected = ages(strict, ok)
    qok = np.isfinite(raw[:, 4:]).all(axis=1)
    qa = ages(np.r_[False, qok[:-1]], qok)
    desc = (connected >= 257) & (qa >= 256) & (idx - floor >= 256)
    same = np.r_[False, raw[1:, 3] == raw[:-1, 3]]
    flat_run = ages(same, ok)
    ordered = (connected >= 37) & (flat_run < 37) & (idx - floor >= 36)
    return np.column_stack(
        (part, clean, clean & mask, clean & desc, clean & ordered)
    ).astype(np.uint8)


def make_shard(out, record, times, values, main, days):
    out.mkdir()
    financial = record["symbol"].startswith("CFFEX.")
    grid = calendar_grid(times, main, financial)
    at = np.searchsorted(grid, times)
    raw = np.full((len(grid), 7), np.nan)
    raw[at] = values
    o, h, low, c = raw[:, :4].T
    good = (
        np.isfinite(raw[:, :4]).all(axis=1)
        & (low > 0)
        & (low <= np.minimum(o, c))
        & (np.maximum(o, c) <= h)
    )
    raw[~good, :4] = np.nan
    qty = raw[:, 4:]
    qty[(qty < 0) | ~np.isfinite(qty)] = np.nan
    _, strict, _ = boundaries(grid, main, days, financial, good)
    bars = [aggregate(grid, raw, good, strict, financial, p) for p in PERIODS]
    flags = pool_flags(grid, raw, bars, strict, financial, record["product"] in HELD)
    for name, array in (
        ("time", grid.astype("<i4")),
        ("flags", flags),
        *((str(p), bar) for p, bar in zip(PERIODS, bars)),
    ):
        np.save(out / (name + ".npy"), array, allow_pickle=False)
    months = grid.astype("datetime64[m]").astype("datetime64[M]").astype(int)
    first = np.r_[
        0, np.flatnonzero((np.diff(months) != 0) | (np.diff(flags[:, 0]) != 0)) + 1
    ]
    groups = []
    for a, b in zip(first, np.r_[first[1:], len(grid)]):
        counts = flags[a:b, 1:].sum(axis=0).tolist()
        if counts[0]:
            groups.append(
                [
                    record["id"],
                    record["product"],
                    record["contract"],
                    int(months[a]),
                    int(flags[a, 0]),
                    int(a),
                    int(b),
                    *counts,
                ]
            )
    return groups


def snapshot():
    return json.loads(gzip.decompress((ROOT / "configs/snapshot.json.gz").read_bytes()))


@lru_cache(maxsize=1)
def preparation_recipe():
    paths = (
        "src/calendar_rules.py",
        "src/common.py",
        "src/dataset.py",
        "src/scope.py",
        "src/views.py",
        "src/scales.py",
        "configs/protocol.json",
        "requirements.lock",
    )
    return identity({name: digest(ROOT / name) for name in paths})


def build(raw_root, working, manifest=None):
    raw_root, working = Path(raw_root), Path(working)
    manifest = snapshot() if manifest is None else manifest
    src_id = identity(manifest)
    with exclusive(working / ".prepare.lock"):
        catalogue = working / "catalogue.json"
        if catalogue.exists():
            verify_cache(working, src_id)
            return
        records = manifest["files"]
        expected = {r["path"] for r in records}
        actual = {
            p.relative_to(raw_root).as_posix()
            for p in (raw_root / "month_contracts/bars").glob("*/*.parquet")
        }
        if expected - actual or (not manifest.get("scope") and actual - expected):
            raise ValueError(
                f"Month inventory differs: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
            )
        groups, hashes = [], {}
        for product, info in sorted(manifest["clocks"].items()):
            main_bytes = verified_bytes(
                raw_root / "main" / (product + ".parquet"), info["main"]
            )
            main = minute_times(
                pq.read_table(
                    pa.BufferReader(main_bytes), columns=["datetime_nano", "datetime"]
                )
            )
            mapping = verified_bytes(
                raw_root / "mapping" / (product + ".csv"), info["mapping"]
            )
            dates = {
                r["date"]
                for r in csv.DictReader(io.StringIO(mapping.decode("utf-8-sig")))
            }
            days = np.array(sorted(dates), dtype="datetime64[D]").astype(int)
            for rec in (r for r in records if r["product"] == product):
                dest = working / f"s{rec['id']:04d}"
                batch = verified_bytes(raw_root / rec["path"], rec["sha256"])
                if len(batch) != rec["bytes"]:
                    raise ValueError(rec["path"])
                receipt = dest / "receipt.json"
                if not receipt.exists():
                    tmp = working / ("." + dest.name + ".building")
                    if tmp.exists():
                        shutil.rmtree(tmp)
                    table = pq.read_table(
                        pa.BufferReader(batch),
                        columns=["datetime_nano", "datetime", *FIELDS],
                    )
                    times = minute_times(table)
                    if len(times) != rec["rows"] or any(
                        str(np.datetime64(int(t), "m")) != stamp[:16].replace(" ", "T")
                        for t, stamp in [
                            (times[0], rec["first"]),
                            (times[-1], rec["last"]),
                        ]
                    ):
                        raise ValueError(rec["path"])
                    rows = np.column_stack(
                        [table[k].to_numpy() for k in FIELDS]
                    ).astype(float)
                    rows[:, 4] *= rec.get("volume_multiplier", 1.0)
                    rows[:, 5:7] *= rec.get("oi_multiplier", 1.0)
                    sub = make_shard(tmp, rec, times, rows, main, days)
                    json_file(
                        tmp / "receipt.json",
                        {
                            "source": rec["sha256"],
                            "manifest": src_id,
                            "recipe": preparation_recipe(),
                            "groups": sub,
                            "files": {
                                p.name: digest(p) for p in sorted(tmp.glob("*.npy"))
                            },
                        },
                    )
                    tmp.rename(dest)
                detail = read_json(receipt)
                if (
                    detail["source"] != rec["sha256"]
                    or detail["manifest"] != src_id
                    or detail.get("recipe") != preparation_recipe()
                ):
                    raise ValueError(f"Different source in {dest}")
                for name, sha in detail["files"].items():
                    if digest(dest / name) != sha:
                        raise ValueError(f"Damaged cached array: {dest / name}")
                    hashes[f"{dest.name}/{name}"] = sha
                hashes[f"{dest.name}/receipt.json"] = digest(receipt)
                groups.extend(detail["groups"])
            print(f"Prepared {product}", flush=True)
        json_file(
            catalogue,
            {
                "manifest": src_id,
                "recipe": preparation_recipe(),
                "groups": groups,
                "files": hashes,
            },
        )


def verify_cache(working, expected=None):
    working = Path(working)
    data = read_json(working / "catalogue.json")
    if data.get("recipe") != preparation_recipe():
        raise ValueError("Prepared cache belongs to different processing code")
    if expected is not None and data["manifest"] != expected:
        raise ValueError("Prepared cache belongs to another source snapshot")
    for name, sha in data["files"].items():
        if digest(working / name) != sha:
            raise ValueError(f"Prepared file changed: {name}")
    return data


class Shard:
    def __init__(self, folder):
        def load(name):
            return np.load(
                Path(folder) / (name + ".npy"), mmap_mode="r", allow_pickle=False
            )

        self.time, self.flags = load("time"), load("flags")
        self.bars = [load(str(p)) for p in PERIODS]


class Corpus:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.groups = read_json(self.folder / "catalogue.json")["groups"]

    @lru_cache(maxsize=16)
    def shard(self, sid):
        return Shard(self.folder / f"s{sid:04d}")

    @lru_cache(maxsize=512)
    def positions(self, sid, a, b, task):
        return np.flatnonzero(self.shard(sid).flags[a:b, task + 1]) + a

    @lru_cache(maxsize=32)
    def distribution(self, split, task):
        from collections import defaultdict

        products, contracts, months = (
            defaultdict(set),
            defaultdict(set),
            defaultdict(int),
        )
        groups = [g for g in self.groups if g[4] == split]
        for _, p, c, ym, _, _, _, n, *_ in groups:
            products[p].add(c)
            contracts[p, c].add(ym)
            months[p, c, ym] += n
        weights = [
            g[7 + task]
            / (
                len(products[g[1]])
                * len(contracts[g[1], g[2]])
                * months[g[1], g[2], g[3]]
            )
            for g in groups
        ]
        cdf = np.cumsum(weights, dtype=float)
        if not len(cdf) or cdf[-1] <= 0:
            raise ValueError(f"No qualified samples for split={split}, task={task}")
        return groups, cdf / cdf[-1]

    def draw(self, split, task, rng):
        groups, cdf = self.distribution(split, task)
        g = groups[np.searchsorted(cdf, rng.random(), side="right")]
        row = int(rng.choice(self.positions(g[0], g[5], g[6], task)))
        return g[0], row
