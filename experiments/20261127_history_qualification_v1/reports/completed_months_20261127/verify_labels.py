"""Recompute every natural label directly from raw month closes and the frozen boundary decisions."""

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPORT = Path(__file__).resolve().parent
EXPERIMENT = REPORT.parents[1]
ASSETS = EXPERIMENT / "runs" / REPORT.name
CONFIG = json.loads((EXPERIMENT / "configs/protocol.json").read_text())
ROOT = Path(CONFIG["root"])
PERIOD_CAPS = ((1, 256), (5, 128), (15, 64), (60, 64))


def natural_order(closes):
    """Visit the 37 chronological positions, retaining the earliest occurrence of either extreme."""
    size = len(closes) - 36
    first = closes[:size]
    low, high = first.copy(), first.copy()
    first_low = np.zeros(size, dtype=np.uint8)
    first_high = np.zeros(size, dtype=np.uint8)
    for offset in range(1, 37):
        value = closes[offset : offset + size]
        higher, lower = value > high, value < low
        high[higher], first_high[higher] = value[higher], offset
        low[lower], first_low[lower] = value[lower], offset
    return np.where(first_high == first_low, 3, np.where(first_high < first_low, 1, 2))


def direction(closes):
    logs = np.log(closes)
    travel = np.abs(np.diff(logs))
    length = np.convolve(travel, np.ones(64), mode="valid")
    net = logs[64:] - logs[:-64]
    for j in np.flatnonzero((length > 0) & (np.abs(2 * np.abs(net) - length) < 1e-12)):
        length[j] = math.fsum(travel[j : j + 64])
    up = (length > 0) & (2 * net >= length)
    down = (length > 0) & (-2 * net >= length)
    return np.where(up, 3, np.where(down, 1, 2))


def continuous_support(boundaries, size, allowed):
    """A window of size closes uses only the size-1 interior edges."""
    breaks = np.flatnonzero(~np.isin(boundaries, allowed))
    rows = np.arange(len(boundaries))
    return (rows >= size - 1) & (
        breaks.searchsorted(rows, side="right")
        == breaks.searchsorted(rows - size + 1, side="right")
    )


def partition_support(index, market_time, held_product):
    year_24 = int(np.datetime64("2024-01-01", "m").astype(int))
    year_25 = int(np.datetime64("2025-01-01", "m").astype(int))
    old = int(
        np.datetime64(CONFIG["cutoff_last_old_minute_start_shanghai"], "m").astype(int)
    )
    partition = (
        (market_time >= year_24).astype(int)
        + (market_time >= year_25)
        + (market_time > old)
    )
    if held_product:
        partition += 4
    assert np.array_equal(partition, index["split"])
    rows = np.arange(len(market_time))
    split_starts = np.r_[0, np.flatnonzero(np.diff(partition)) + 1]
    first_in_partition = split_starts[split_starts.searchsorted(rows, side="right") - 1]
    input_ok = np.ones(len(rows), dtype=bool)
    for period, capacity in PERIOD_CAPS:
        starts = index[f"bar_start_{period}"]
        completed = index[f"bar_end_{period}"].searchsorted(rows, side="right")
        first_needed = completed - capacity - 65
        has_enough = first_needed >= 0
        after_boundary = np.zeros(len(rows), dtype=bool)
        after_boundary[has_enough] = (
            starts[first_needed[has_enough]] >= first_in_partition[has_enough]
        )
        input_ok &= (first_in_partition == 0) | after_boundary
    assert np.array_equal(input_ok, index["input_split_safe"])
    counts_in_partition = rows - first_in_partition + 1
    for window in (37, 64, 18):
        required = {37: 37, 64: 65, 18: 257}[window]
        assert np.array_equal(
            counts_in_partition >= required, index[f"label{window}_in_split"]
        )
    return input_ok, counts_in_partition


def review():
    with gzip.open(REPORT / "source_index.json.gz", "rt") as f:
        sources = json.load(f)
    totals = Counter()
    for source in sources:
        data = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        table = pq.read_table(
            pa.BufferReader(data), columns=["datetime_nano", "close"], use_threads=False
        )
        times = table.column(0).to_numpy() // 60_000_000_000 + 480
        closes = table.column(1).to_numpy()
        assert np.all(np.isfinite(closes) & (closes > 0))
        with np.load(ASSETS / f"{source['id']:04d}.npz") as index:
            assert np.array_equal(times, index["minute_start_local"])
            input_ok, in_partition = partition_support(
                index, times, source["product"] in CONFIG["held_products"]
            )
            boundary = index["boundary"]
            class64 = np.zeros(len(times), dtype=np.uint8)
            order37 = np.zeros(len(times), dtype=np.uint8)
            if len(times) >= 65:
                class64[64:] = direction(closes)
            if len(times) >= 37:
                order37[36:] = natural_order(closes)
            class64[~continuous_support(boundary, 65, (1, 2, 3))] = 0
            order37[~continuous_support(boundary, 37, (1, 2, 3))] = 0
            assert np.array_equal(class64, index["class64"]), source["contract"]
            assert np.array_equal(order37, index["order37"]), source["contract"]
            strict_class = continuous_support(boundary, 65, (1, 2))
            strict_order = continuous_support(boundary, 37, (1, 2))
            assert np.array_equal(strict_class, index["strict64"])
            assert np.array_equal(strict_order, index["strict37"])
            class_pool = input_ok & (in_partition >= 65) & strict_class
            order_pool = input_ok & (in_partition >= 37) & strict_order
            for code, name in enumerate(("down", "low_direction", "up"), start=1):
                totals[f"class_strict_purged_{name}"] += int(
                    np.sum(class_pool & (class64 == code))
                )
            for code, name in enumerate(("high_first", "low_first"), start=1):
                totals[f"order_strict_purged_{name}"] += int(
                    np.sum(order_pool & (order37 == code))
                )
            totals["rows"] += len(times)
    published = json.loads((REPORT / "summary.json").read_text())
    assert all(published["totals"][key] == value for key, value in totals.items())
    return {
        "source_commit": published["source_commit"],
        "source_files_and_sha_verified": len(sources),
        "labels_and_ineligibility_codes_verified_at_all_endpoints": totals["rows"],
        "full_input_and_label_temporal_purge_verified_at_all_endpoints": totals["rows"],
        "strict_counts_recomputed": dict(totals),
        "scope": "Independent natural-label arithmetic and date-support checks conditional on frozen source boundary and bar decisions. No extra calendar evidence or model performance inferred.",
    }


if __name__ == "__main__":
    print(json.dumps(review(), indent=2))
