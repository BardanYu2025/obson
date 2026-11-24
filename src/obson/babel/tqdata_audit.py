"""Read-only CPU inventory of a TqSdk delivery; not a training qualification gate.

No source files are modified, no checkpoints are loaded, and no market observations
are selected by model performance. A delivery still being copied is NOT frozen.
Requires pandas, numpy and pyarrow (Parquet reader).
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume", "open_oi", "close_oi")
FLOOR = "2026-09-12 03:00:00"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inspect_frame(frame, cutoff=FLOOR):
    required = {"datetime", "datetime_nano", *FIELDS}
    if required - set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if len(frame) == 0:
        raise ValueError("Empty Parquet file")
    dt = pd.to_datetime(frame.datetime, errors="raise")
    if dt.dt.tz is not None:
        raise ValueError("Expected declared naive Shanghai bar-start timestamps")
    # Nanoseconds are UTC epoch; naive datetime is local clock, NOT UTC.
    ns_clock = (
        pd.to_datetime(frame.datetime_nano, unit="ns", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    v = frame.loc[:, FIELDS].to_numpy(dtype=np.float64)
    o, h, low, c, vol, oi0, oi1 = v.T
    flat_zero = (vol == 0) & (o == h) & (h == low) & (low == c) & (oi0 == oi1)
    # Run length is in adjacent minute records; overnight breaks reset the run.
    ns = dt.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    starts = np.r_[True, (~flat_zero[:-1]) | (np.diff(ns) != 60_000_000_000)]
    indices = np.arange(len(frame))
    last_start = np.maximum.accumulate(np.where(starts | ~flat_zero, indices, 0))
    runs = np.where(flat_zero, indices - last_start + 1, 0)
    after = dt > pd.Timestamp(cutoff)
    return {
        "rows": len(frame),
        "first": str(dt.min()),
        "last": str(dt.max()),
        "nominal_clock_bins_not_qualified": {
            str(period): int(dt.dt.floor(f"{period}min").nunique()) for period in (15, 30, 60)
        },
        "years": {str(k): int(n) for k, n in dt.dt.year.value_counts().sort_index().items()},
        "after_old_cutoff_rows": int(after.sum()),
        "after_old_cutoff_calendar_weeks": sorted(set(dt[after].dt.strftime("%G-W%V"))),
        "checks": {
            "nat": int(dt.isna().sum()),
            "timestamp_clock_mismatch": int((dt != ns_clock).sum()),
            "duplicate_timestamps": int(dt.duplicated().sum()),
            "backward_steps": int((dt.diff().dt.total_seconds() < 0).sum()),
            "not_minute_aligned": int(
                ((dt.dt.second != 0) | (dt.dt.microsecond != 0) | (dt.dt.nanosecond != 0)).sum()
            ),
            "nonfinite_numeric_rows": int((~np.isfinite(v).all(axis=1)).sum()),
            "invalid_ohlc": int(
                ((h < np.maximum.reduce([o, low, c])) | (low > np.minimum.reduce([o, h, c]))).sum()
            ),
            "nonpositive_price_rows": int((v[:, :4] <= 0).any(axis=1).sum()),
            "negative_volume_or_oi_rows": int((v[:, 4:] < 0).any(axis=1).sum()),
        },
        "zero_volume_rows": int((vol == 0).sum()),
        "flat_zero_unchanged_oi_rows": int(flat_zero.sum()),
        "max_contiguous_flat_zero_minutes": int(runs.max()),
        "zero_close_oi_rows": int((oi1 == 0).sum()),
    }


def signature(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def inventory(root):
    return sorted(
        [
            *root.glob("main/*.parquet"),
            *root.glob("index/*.parquet"),
            *root.glob("month_contracts/bars/*/*.parquet"),
        ]
    )


def audit(root, out, cutoff=FLOOR):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError("Write audit outside the input delivery")
    if out.exists():
        raise FileExistsError("Audit output must be new; preserve earlier observations")
    paths = inventory(root)
    if not paths:
        raise ValueError("No main or month_contracts Parquet files")
    out.mkdir(parents=True)
    initial = {p: signature(p) for p in paths}
    records, failures = [], []
    contract_ranges = defaultdict(list)
    for i, path in enumerate(paths):
        family = path.parent.name if path.parent.name in ("main", "index") else "month"
        product = path.stem if family in ("main", "index") else path.parent.name
        record = {
            "path": str(path.relative_to(root)),
            "family": family,
            "product": product,
            "bytes": initial[path][0],
        }
        try:
            before = signature(path)
            record["sha256"] = digest(path)
            frame = pd.read_parquet(path)
            record.update(inspect_frame(frame, cutoff))
            record["columns"] = {k: str(v) for k, v in frame.dtypes.items()}
            record["stable_during_read"] = initial[path] == before == signature(path)
            if family == "month":
                # Include date window in identity: CZCE short codes repeat each decade.
                contract = path.stem.rsplit("_", 2)[0]
                record["file_contract_code"] = contract
                contract_ranges[(product, contract)].append(
                    (record["first"], record["last"], record["path"])
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(record["path"])
        records.append(record)
        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            print(f"CPU data audit {i + 1}/{len(paths)} files", flush=True)
    now = inventory(root)
    changed = [
        str(p.relative_to(root)) for p in paths if not p.exists() or signature(p) != initial[p]
    ]
    overlaps = []
    for (product, contract), intervals in sorted(contract_ranges.items()):
        ordered = sorted(intervals)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if right[0] > left[1]:
                    break
                overlaps.append(
                    {"product": product, "contract": contract, "files": [left[2], right[2]]}
                )
    groups = {}
    for family in ("main", "month", "index"):
        rows = [r for r in records if r["family"] == family and "error" not in r]
        counts, years = Counter(), Counter()
        for row in rows:
            counts.update(row["checks"])
            years.update(row["years"])
        groups[family] = {
            "files": len(rows),
            "products": sorted({r["product"] for r in rows}),
            "rows": sum(r["rows"] for r in rows),
            "bytes": sum(r["bytes"] for r in rows),
            "first": min((r["first"] for r in rows), default=None),
            "last": max((r["last"] for r in rows), default=None),
            "years": dict(sorted(years.items())),
            "checks": dict(counts),
            "nominal_clock_bins_not_qualified": {
                str(period): sum(r["nominal_clock_bins_not_qualified"][str(period)] for r in rows)
                for period in (15, 30, 60)
            },
            "after_old_cutoff_rows": sum(r["after_old_cutoff_rows"] for r in rows),
            "after_old_cutoff_calendar_weeks": sorted(
                {w for r in rows for w in r["after_old_cutoff_calendar_weeks"]}
            ),
            "zero_volume_rows": sum(r["zero_volume_rows"] for r in rows),
            "flat_zero_unchanged_oi_rows": sum(r["flat_zero_unchanged_oi_rows"] for r in rows),
            "max_contiguous_flat_zero_minutes": max(
                (r["max_contiguous_flat_zero_minutes"] for r in rows), default=0
            ),
        }
    summary = {
        "schema": "babel.tqdata.inventory.v1",
        "root": str(root),
        "cutoff": cutoff,
        "execution": "CPU_data_only_no_model_no_training_no_performance_selection",
        "source_code_sha256": digest(Path(__file__)),
        "groups": groups,
        "missing_month_products_vs_main": sorted(
            set(groups["main"]["products"]) - set(groups["month"]["products"])
        ),
        "read_failures": failures,
        "changed_or_removed_during_audit": changed,
        "added_during_audit": [str(p.relative_to(root)) for p in now if p not in initial],
        "overlapping_same_contract_file_ranges": overlaps,
        "limits": [
            "Inventory during delivery, not a locked complete training snapshot",
            "Flat zero-trade bars flag a pattern, not proof of fabricated/missing trades",
            "No session resampling, as-of mapping qualification or usable-window count",
            "After-cutoff counts are raw minutes, not independent V13 windows",
            "No claim that all numerical rows are semantically valid market observations",
        ],
    }
    for name, value in (("inventory.json", records), ("summary.json", summary)):
        (out / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", default=FLOOR)
    args = parser.parse_args()
    audit(args.root, args.out, args.cutoff)


if __name__ == "__main__":
    main()
