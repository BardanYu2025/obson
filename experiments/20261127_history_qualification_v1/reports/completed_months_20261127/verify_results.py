"""Independent source readback of the delivered row index and fixed counterfactual sample.

Run from the repo root with the pinned CPU Python. Writes JSON to stdout; no fitted parameters.
"""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPORT = Path(__file__).resolve().parent
EXPERIMENT = REPORT.parents[1]
ROOT = Path(json.loads((EXPERIMENT / "configs/protocol.json").read_text())["root"])
ASSETS = EXPERIMENT / "runs" / REPORT.name
PERIODS = (1, 5, 15, 60)
CAPS = (256, 128, 64, 64)
FIELDS = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_string(minute):
    return str(np.datetime64(int(minute), "m")).replace("T", " ")


def raw_descriptor(block):
    values = []
    for n in (16, 64, 256):
        lp = np.log(block[-1 - n :, 3])
        increments = np.diff(lp)
        length = math.fsum(abs(x) for x in increments)
        net = lp[-1] - lp[0]
        high = lp[0]
        drawdown = 0.0
        for price in lp[1:]:
            high = max(high, price)
            drawdown = max(drawdown, high - price)
        values.extend(
            (
                net,
                abs(net) / length if length else 0.0,
                math.sqrt(math.fsum(x * x for x in increments)),
                drawdown,
                math.log1p(math.fsum(block[-n:, 4])),
                math.asinh(block[-1, 6] - block[-n, 5]),
            )
        )
    return np.array(values, dtype=">f8").tobytes()


def aggregate(b):
    return [
        b[0, 0],
        b[:, 1].max(),
        b[:, 2].min(),
        b[-1, 3],
        math.fsum(b[:, 4]),
        b[0, 5],
        b[-1, 6],
    ]


def encode_blocks(bars, current, cap):
    strict = bars[:-1] if current else bars
    facts = []
    for k in range(cap // 16):
        stop = len(strict) - 16 * k
        if stop < 16:
            facts.extend([0.0] * 5)
            continue
        b = strict[stop - 16 : stop]
        if not np.isfinite(b).all() or (b[:, 4:] < 0).any():
            facts.extend([0.0] * 5)
            continue
        level = [0.0]
        for prev, now in zip(b, b[1:]):
            # Same defined scalar operation order as the released representation arithmetic.
            level.append(
                level[-1] + math.log(now[0] / prev[3]) + math.log(now[3] / now[0])
            )
        c1 = math.fsum(x * (0.25 if j >= 8 else -0.25) for j, x in enumerate(level))
        c2 = math.fsum(
            x * (0.25 if 4 <= j < 12 else -0.25) for j, x in enumerate(level)
        )
        facts.extend(
            (
                c1,
                c2,
                math.log1p(math.fsum(b[:, 4])),
                math.asinh(b[-1, 6] - b[0, 5]),
                1.0,
            )
        )
    return facts


def examine_pair(i, raw, addresses, cached, tick):
    start = i - 256
    for k in range(i - 36, i - 1):
        slot = 15 - ((i - 1 - k) % 16)
        if slot // 4 != (slot + 1) // 4 or min(raw[k : k + 3, 4]) <= 0:
            continue
        parents = {}
        for p in PERIODS[1:]:
            starts, ends = addresses[p]
            j = int(ends.searchsorted(k))
            if j < len(ends) and starts[j] <= k and k + 2 <= ends[j] <= i:
                parents[p] = j
        if len(parents) == 3:
            break
    else:
        raise AssertionError("Stored accepted case has no eligible first pair")
    rest = np.r_[raw[i - 36 : k, 3], raw[k + 2 : i + 1, 3]]
    hi, lo = float(max(rest) + tick), float(min(rest) - tick)
    vecs, descs, agg, order = [], [], [], []
    covered_by_minute_range = True
    for a, b in ((hi, lo), (lo, hi)):
        patch = raw[k : k + 3].copy()
        patch[:2, 3] = [a, b]
        previous = raw[k - 1, 3]
        for bar in patch:
            bar[0] = previous
            bar[1], bar[2] = max(previous, bar[3]), min(previous, bar[3])
            previous = bar[3]
        covered_by_minute_range &= bool(
            np.all(patch[:, 1] <= raw[k : k + 3, 1])
            and np.all(patch[:, 2] >= raw[k : k + 3, 2])
        )
        window = raw[start : i + 1].copy()
        window[k - start : k - start + 3] = patch
        descs.append(raw_descriptor(window))
        recent = window[-37:, 3].tolist()
        order.append(recent.index(max(recent)) < recent.index(min(recent)))
        coords = encode_blocks(window[-256:], True, 256)
        parents_full = []
        for p, cap in zip(PERIODS[1:], CAPS[1:]):
            starts, ends = addresses[p]
            parent = parents[p]
            block = raw[starts[parent] : ends[parent] + 1].copy()
            offset = k - starts[parent]
            block[offset : offset + 3] = patch
            changed = aggregate(block)
            parents_full.extend(changed)
            stop = int(ends.searchsorted(i, side="right"))
            history = cached[p][stop - cap : stop].copy()
            if stop - cap <= parent < stop:
                history[parent - (stop - cap)] = changed
            coords.extend(encode_blocks(history, bool(ends[stop - 1] == i), cap))
        vecs.append(np.array(coords, dtype=">f4").tobytes())
        agg.append(parents_full)
    assert order == [True, False] and agg[0] == agg[1]
    return vecs[0] == vecs[1], descs[0] == descs[1], covered_by_minute_range


def oracle(i, raw, arrays):
    close = raw[:, 3]
    kind = arrays["boundary"]

    # Left-inclusive window of closes, so the relation of its first close to the previous row is irrelevant.
    def connected(n, extended):
        return (
            i + 1 >= n
            and np.all(arrays["price_good"][i + 1 - n : i + 1])
            and np.isin(
                kind[i + 2 - n : i + 1], (1, 2, 3) if extended else (1, 2)
            ).all()
        )

    assert connected(65, False) == bool(arrays["strict64"][i])
    assert connected(37, False) == bool(arrays["strict37"][i])
    cls = 0
    if connected(65, True):
        lp = np.log(close[i - 64 : i + 1])
        travel = math.fsum(abs(x - y) for x, y in zip(lp[1:], lp))
        net = lp[-1] - lp[0]
        cls = (
            2
            if not travel
            else (3 if 2 * net >= travel else 1 if -2 * net >= travel else 2)
        )
    order = 0
    if connected(37, True):
        win = list(close[i - 36 : i + 1])
        order = (
            3
            if max(win) == min(win)
            else (1 if win.index(max(win)) < win.index(min(win)) else 2)
        )
    assert cls == arrays["class64"][i] and order == arrays["order37"][i]
    peak_error, peak_rel, count = 0.0, 0.0, 0
    for p in PERIODS:
        starts, ends = arrays[f"bar_start_{p}"], arrays[f"bar_end_{p}"]
        j = int(ends.searchsorted(i, side="right")) - 1
        for tier in ("strict", "screen"):
            n = int(arrays[f"{tier}{p}"][i])
            if not n:
                continue
            last = ends[j]
            allowed = (1, 2) if tier == "strict" else (1, 2, 3)
            assert np.isin(kind[last + 1 : i + 1], allowed).all() and i - last < 256
            # Transport C_t through the visible minute bridge in multiplicative coordinates.
            anchor = close[i]
            for t in range(i, last, -1):
                anchor /= (raw[t, 0] / close[t - 1]) * (close[t] / raw[t, 0])
            for b in range(j, j - n, -1):
                src = np.array(aggregate(raw[starts[b] : ends[b] + 1]))[:4]
                source_open, high, low, source_close = src
                body_ratio = source_close / source_open
                open_hat = anchor / body_ratio
                hats = np.array(
                    [
                        open_hat,
                        max(open_hat, anchor) * (high / max(source_open, source_close)),
                        min(open_hat, anchor) / (min(source_open, source_close) / low),
                        anchor,
                    ]
                )
                peak_error = max(peak_error, float(np.max(np.abs(hats - src))))
                peak_rel = max(peak_rel, float(np.max(np.abs(hats - src) / src)))
                count += 1
                if b > j - n + 1:
                    assert starts[b] == ends[b - 1] + 1 and kind[starts[b]] in allowed
                    anchor /= (source_open / close[ends[b - 1]]) * body_ratio
    return count, peak_error, peak_rel


def review():
    summary = json.loads((REPORT / "summary.json").read_text())
    receipt = json.loads((REPORT / "receipt.json").read_text())
    assert all(sha(REPORT / k) == v for k, v in receipt["reports"].items())
    source_snapshot = REPORT / "source_index.json.gz"
    with gzip.open(source_snapshot, "rt") as f:
        sources = json.load(f)
    previous = (
        EXPERIMENT.parent
        / "20261124_minute_audit_v1/reports/20261124/inventory.json.gz"
    )
    previous_hash = "f0a4a462fe5034a3efcc37a0ebcf045a5b7fb23b47c847923af4167ef58ab187"
    assert sha(previous) == previous_hash
    with gzip.open(previous, "rt") as f:
        old = {
            row["path"]: row
            for row in json.load(f)
            if row["path"].startswith("month_contracts/bars/")
        }
    assert set(old) == {row["path"] for row in sources}
    for row in sources:
        assert all(
            row[k] == old[row["path"]][k]
            for k in ("rows", "bytes", "sha256", "first", "last")
        )
    assert sha(ASSETS / "counterfactual_sample.csv.gz") == summary["pair_lookup_sha256"]
    strong_rows = defaultdict(list)
    with gzip.open(ASSETS / "counterfactual_sample.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["reason"] in (
                "matched_160_only",
                "matched_160_and_18",
                "unequal_packed_facts",
            ):
                strong_rows[int(r["source_id"])].append(r)
    prod = defaultdict(list)
    for r in sources:
        prod[r["product"]].append(r["id"])
    fixed = {i for ids in prod.values() for i in (ids[0], ids[len(ids) // 2], ids[-1])}
    stats = Counter()
    checks = Counter()
    maximum = relative = 0.0
    coverage = Counter()
    strong_by_time = Counter()
    full_cap_shortfalls = Counter()
    for r in sources:
        sid = r["id"]
        path = ASSETS / f"{sid:04d}.npz"
        assert (
            sha(path) == r["index_sha256"] and path.stat().st_size == r["index_bytes"]
        )
        with np.load(path) as index:
            kinds = index["boundary"]
            rows = np.arange(len(kinds))
            last_cut = np.maximum.accumulate(
                np.where(np.isin(kinds, (1, 2, 3)), 0, rows)
            )
            usable = np.minimum(256, rows - last_cut + 1)
            assert np.array_equal(usable, index["screen1"])
            full_cap_shortfalls["within_first_255_rows_of_source_window"] += int(
                np.sum((usable < 256) & (rows < 255))
            )
            for kind, label in (
                (4, "last_cut_has_intervening_main_minutes"),
                (5, "last_cut_has_insufficient_evidence"),
            ):
                full_cap_shortfalls[label] += int(
                    np.sum((usable < 256) & (rows >= 255) & (kinds[last_cut] == kind))
                )
        if sid not in fixed and not strong_rows[sid]:
            continue
        table = pq.read_table(ROOT / r["path"], columns=FIELDS, use_threads=False)
        raw = np.column_stack([table.column(f).to_numpy() for f in FIELDS])
        with np.load(path) as z:
            addresses = {p: (z[f"bar_start_{p}"], z[f"bar_end_{p}"]) for p in PERIODS}
            if strong_rows[sid]:
                # Assemble source parents once per file. Replace the full affected parent separately for each path.
                cached = {}
                for p in PERIODS[1:]:
                    starts, ends = addresses[p]
                    cached[p] = np.column_stack(
                        (
                            raw[starts, 0],
                            np.zeros(len(starts)),
                            np.zeros(len(starts)),
                            raw[ends, 3],
                            np.cumsum(np.r_[0, raw[:, 4]])[ends + 1]
                            - np.cumsum(np.r_[0, raw[:, 4]])[starts],
                            raw[starts, 5],
                            raw[ends, 6],
                        )
                    )
                for row in strong_rows[sid]:
                    equal160, equal18, within = examine_pair(
                        int(row["source_row"]), raw, addresses, cached, r["meta_tick"]
                    )
                    outcome = (
                        "unequal_packed_facts"
                        if not equal160
                        else "matched_160_and_18"
                        if equal18
                        else "matched_160_only"
                    )
                    assert outcome == row["reason"], (
                        r["id"],
                        row["source_row"],
                        outcome,
                        row["reason"],
                    )
                    checks[outcome] += 1
                    if equal160 and equal18:
                        strong_by_time[row["split"]] += 1
                        coverage[
                            "all_three_hypothetical_ranges_inside_observed_ranges"
                        ] += int(within)
            if sid in fixed:
                arrays = {
                    k: z[k]
                    for k in (
                        "boundary",
                        "price_good",
                        "class64",
                        "order37",
                        "strict64",
                        "strict37",
                    )
                }
                for p in PERIODS:
                    arrays[f"bar_start_{p}"], arrays[f"bar_end_{p}"] = addresses[p]
                    for tier in ("strict", "screen"):
                        arrays[f"{tier}{p}"] = z[f"{tier}{p}"]
                for i in np.unique(np.linspace(0, len(raw) - 1, 9, dtype=int)):
                    num, err, rel = oracle(i, raw, arrays)
                    stats["natural_endpoint_checks"] += 1
                    stats["original_quote_bar_queries"] += num
                    maximum = max(maximum, err)
                    relative = max(relative, rel)
    assert relative < 1e-10
    assert (
        sum(full_cap_shortfalls.values())
        == summary["totals"]["rows"] - summary["totals"]["screen_1m_full_cap"]
    )
    return {
        "frozen_source_commit": summary["source_commit"],
        "all_endpoint_assets_sha_verified": len(sources),
        "previous_inventory_sha256": previous_hash,
        "month_files_unchanged_from_previous_inventory": len(sources),
        "expanded_256_minute_shortfalls_disjoint": dict(full_cap_shortfalls),
        "products_in_fixed_source_check": len(prod),
        "fixed_files_in_natural_check": len(fixed),
        **stats,
        "max_abs_quote_error_native_in_checked_queries": maximum,
        "max_relative_quote_error_in_checked_queries": relative,
        "counterfactual_recomputed_from_actual_patched_parents": dict(checks),
        "strong_screen_by_split": dict(strong_by_time),
        "hypothetical_price_range_diagnostic": dict(coverage),
        "selection": "First/middle/last file in manifest order per product; 9 equally spaced endpoint rows per file. Every preliminary same-160 candidate and the single rejected packed-precision case independently checked.",
        "meaning": "Structural source checks and retrospective sample eligibility only. No model, exchange band or independent test performance certified.",
    }


if __name__ == "__main__":
    print(json.dumps(review(), ensure_ascii=False, indent=2))
