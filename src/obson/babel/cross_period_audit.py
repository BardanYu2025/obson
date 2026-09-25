"""CPU-only raw bar alignment audit; no model import, fitting, or split selection.

Timestamps are interpreted as nominal bar starts, as in the current Babel pipeline.
A value match on a partially covered interval is NOT evidence of complete coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ((15, 30), (15, 60), (30, 60))
FIELDS = ("open", "high", "low", "close", "volume", "open_oi", "close_oi")
ATOL = 1e-6
MINUTE_NS = 60_000_000_000
PROTOCOL = "raw_cross_period_v1"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_bars(raw: bytes, period: int) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw))
    required = {"datetime", *FIELDS[:5]}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Empty file or missing datetime/OHLCV columns")
    dates = pd.to_datetime(frame["datetime"], errors="raise")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    ticks = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if dates.isna().any() or (np.diff(ticks) < period * MINUTE_NS).any():
        raise ValueError("Missing, duplicate, unsorted or overlapping bar timestamps")
    frame["datetime"] = dates
    for name in FIELDS:
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    values = frame[list(FIELDS[:5])].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("OHLCV must be finite, prices positive and volume nonnegative")
    op, high, low, close = (frame[n].to_numpy() for n in FIELDS[:4])
    if ((high < np.maximum(op, close)) | (low > np.minimum(op, close)) | (high < low)).any():
        raise ValueError("Invalid OHLC geometry")
    # Unknown/invalid OI does not invalidate price evidence; it cannot pass OI eligibility.
    for name in FIELDS[5:]:
        frame.loc[~np.isfinite(frame[name]) | (frame[name] < 0), name] = np.nan
    return frame


def compare(
    fine: pd.DataFrame, coarse: pd.DataFrame, fine_minutes: int, coarse_minutes: int
) -> dict:
    """Match by timestamps, independently within one symbol and contract."""
    if coarse_minutes <= fine_minutes or coarse_minutes % fine_minutes:
        raise ValueError("Periods must have an integer coarse/fine ratio greater than one")
    ft = fine["datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ct = coarse["datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ratio = coarse_minutes // fine_minutes
    fstep, cstep = fine_minutes * MINUTE_NS, coarse_minutes * MINUTE_NS
    left = np.searchsorted(ft, ct)
    right = np.searchsorted(ft, ct + cstep)
    count = right - left
    if (count > ratio).any():
        raise ValueError("Fine timestamps overlap; call read_bars before compare")
    offsets = np.arange(ratio)
    indices = np.minimum(left[:, None] + offsets, len(ft) - 1)
    present = offsets[None, :] < count[:, None]
    prior_overlap = (left > 0) & (ft[np.maximum(left - 1, 0)] + fstep > ct)
    end_overlap = (count > 0) & (ft[np.maximum(right - 1, 0)] + fstep > ct + cstep)
    straddled = prior_overlap | end_overlap
    full = (count == ratio) & (ft[indices] == ct[:, None] + offsets * fstep).all(axis=1)
    full &= ~straddled
    nonempty = count > 0
    contained = nonempty & ~straddled
    aggregates = np.empty((len(ct), len(FIELDS)), dtype=np.float64)
    first = np.minimum(left, len(ft) - 1)
    last = np.maximum(right - 1, 0)
    aggregates[:, 0] = fine["open"].to_numpy()[first]
    aggregates[:, 1] = np.where(present, fine["high"].to_numpy()[indices], -np.inf).max(axis=1)
    aggregates[:, 2] = np.where(present, fine["low"].to_numpy()[indices], np.inf).min(axis=1)
    aggregates[:, 3] = fine["close"].to_numpy()[last]
    aggregates[:, 4] = np.where(present, fine["volume"].to_numpy()[indices], 0).sum(axis=1)
    aggregates[:, 5] = fine["open_oi"].to_numpy()[first]
    aggregates[:, 6] = fine["close_oi"].to_numpy()[last]
    expected = coarse[list(FIELDS)].to_numpy(dtype=np.float64)
    finite = np.isfinite(aggregates) & np.isfinite(expected) & contained[:, None]
    delta = np.zeros_like(aggregates)
    np.subtract(aggregates, expected, out=delta, where=finite)
    matches = finite & (np.abs(delta) <= ATOL)
    price_match = matches[:, :4].all(axis=1)
    ohlcv_match = matches[:, :5].all(axis=1)
    oi_known = finite[:, 5:].all(axis=1)
    oi_match = matches[:, 5:].all(axis=1)
    price_eligible = full & price_match
    ohlcv_eligible = full & ohlcv_match
    activity_eligible = ohlcv_eligible & oi_match
    context128 = (right >= 128) & (np.arange(len(ct)) >= 127)
    # Diagnostic only: some feeds carry stale prices in zero-volume bars. Never
    # repair the source or relax eligibility based on this alternative aggregate.
    active = present & (fine["volume"].to_numpy()[indices] > 0)
    has_active = active.any(axis=1)
    active_first = indices[np.arange(len(ct)), active.argmax(axis=1)]
    active_last = indices[np.arange(len(ct)), ratio - 1 - active[:, ::-1].argmax(axis=1)]
    active_prices = np.column_stack(
        (
            fine["open"].to_numpy()[active_first],
            np.where(active, fine["high"].to_numpy()[indices], -np.inf).max(axis=1),
            np.where(active, fine["low"].to_numpy()[indices], np.inf).min(axis=1),
            fine["close"].to_numpy()[active_last],
        )
    )
    active_match = has_active & (np.abs(active_prices - expected[:, :4]) <= ATOL).all(axis=1)
    mismatch = full & ~ohlcv_match
    counts = {
        "coarse_rows": len(ct),
        "full_coverage": int(full.sum()),
        "partial_coverage": int((nonempty & ~full).sum()),
        "empty_coverage": int((~nonempty).sum()),
        "straddled_intervals": int(straddled.sum()),
        "contained_ohlcv_match": int(ohlcv_match.sum()),
        "partial_ohlcv_match_not_eligible": int((~full & ohlcv_match).sum()),
        "full_price_match": int(price_eligible.sum()),
        "full_price_mismatch": int((full & ~price_match).sum()),
        "full_ohlcv_match": int(ohlcv_eligible.sum()),
        "full_ohlcv_mismatch": int((full & ~ohlcv_match).sum()),
        "full_mismatch_first_child_zero_volume": int(
            (mismatch & (fine["volume"].to_numpy()[first] == 0)).sum()
        ),
        "full_mismatch_all_children_zero_volume": int((mismatch & ~has_active).sum()),
        "full_mismatch_active_only_prices_match_diagnostic": int((mismatch & active_match).sum()),
        "full_oi_comparable": int((full & oi_known).sum()),
        "full_oi_match": int((full & oi_match).sum()),
        "full_oi_mismatch": int((full & oi_known & ~oi_match).sum()),
        "full_ohlcv_and_oi_match": int(activity_eligible.sum()),
        "price_matches_with_128_rows_each": int((price_eligible & context128).sum()),
        "ohlcv_matches_with_128_rows_each": int((ohlcv_eligible & context128).sum()),
    }
    field_stats = {}
    for j, name in enumerate(FIELDS):
        valid = full & finite[:, j]
        field_stats[name] = {
            "full_comparable": int(valid.sum()),
            "full_mismatches": int((valid & ~matches[:, j]).sum()),
            "full_max_abs_difference": float(np.abs(delta[valid, j]).max())
            if valid.any()
            else None,
        }

    def examples(mask: np.ndarray) -> list[dict]:
        result = []
        for i in np.flatnonzero(mask)[:3]:
            observed = {
                name: {"aggregated": float(aggregates[i, j]), "coarse": float(expected[i, j])}
                for j, name in enumerate(FIELDS)
                if finite[i, j] and not matches[i, j]
            }
            result.append(
                {
                    "coarse_row": int(i),
                    "start": str(coarse["datetime"].iloc[i]),
                    "nominal_end": str(pd.Timestamp(ct[i] + cstep)),
                    "fine_rows": [int(left[i]), int(right[i])],
                    "fine_row_end_exclusive": True,
                    "fine_starts": [str(d) for d in fine["datetime"].iloc[left[i] : right[i]]],
                    "child_count": int(count[i]),
                    "full_coverage": bool(full[i]),
                    "straddled": bool(straddled[i]),
                    "is_first_or_last_coarse_row": bool(i in (0, len(ct) - 1)),
                    "differing_fields": observed,
                }
            )
        return result

    return {
        "counts": counts,
        "child_count_histogram": dict(sorted(Counter(map(str, count)).items())),
        "partial_start_clock_histogram": dict(
            sorted(Counter(coarse.loc[nonempty & ~full, "datetime"].dt.strftime("%H:%M")).items())
        ),
        "fields": field_stats,
        "examples": {
            "full_ohlcv_mismatch": examples(full & ~ohlcv_match),
            "full_oi_mismatch": examples(full & oi_known & ~oi_match),
            "partial": examples(nonempty & ~full),
            "empty": examples(~nonempty),
            "straddled": examples(straddled),
        },
    }


def prepare_output(root: Path, out: Path) -> tuple[Path, Path]:
    root, out = root.resolve(), out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Audit output must be separate from raw data tree")
    if not root.is_dir():
        raise ValueError(f"Raw root is not a directory: {root}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output is not empty; use a new output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def run(root: Path, out: Path) -> dict:
    root, out = prepare_output(root, out)
    started = time.monotonic()
    write_json(
        out / "manifest.json",
        {
            "protocol": PROTOCOL,
            "root": str(root),
            "timestamp_semantics": "nominal_bar_start",
            "timezone_for_naive_timestamps": "Asia/Shanghai",
            "pairs": PAIRS,
            "atol": ATOL,
            "rtol": 0,
            "strict_coverage": "exact nominal grid; no straddled bars",
            "code_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "model_updates": 0,
            "model_inference": False,
            "eligibility_scope": "Raw geometry only; NO split, main-contract, warmup or model-source certification",
        },
    )
    try:
        return audit_files(root, out, started)
    except Exception as error:
        write_json(out / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


def audit_files(root: Path, out: Path, started: float) -> dict:
    groups: dict[tuple[str, str], dict[int, Path]] = {}
    ignored = []
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root)
        match = re.fullmatch(r"(.+)_(15|30|60)m\.csv", path.name)
        if not match or len(relative.parts) != 2 or path.is_symlink():
            ignored.append(str(relative))
            continue
        key = (relative.parts[0], match[1])
        groups.setdefault(key, {})[int(match[2])] = path
    if not groups:
        raise ValueError("No symbol/contract_{15,30,60}m.csv inputs found")
    inventory, records, errors, unavailable = [], [], [], []
    summaries = {
        f"{f}->{c}": {
            "counts": Counter(),
            "child_count_histogram": Counter(),
            "partial_start_clock_histogram": Counter(),
            "fields": {},
            "contracts": 0,
        }
        for f, c in PAIRS
    }
    for number, ((symbol, contract), paths) in enumerate(sorted(groups.items()), 1):
        frames = {}
        for period, path in sorted(paths.items()):
            raw = path.read_bytes()
            entry = {
                "path": str(path.relative_to(root)),
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
            }
            try:
                frame = read_bars(raw, period)
                frames[period] = frame
                entry.update(
                    rows=len(frame),
                    first=str(frame.datetime.iloc[0]),
                    last=str(frame.datetime.iloc[-1]),
                    valid=True,
                    oi_unknown_rows={n: int(frame[n].isna().sum()) for n in FIELDS[5:]},
                )
            except (ValueError, TypeError, OverflowError) as error:
                entry.update(valid=False, error=str(error))
                errors.append({"path": entry["path"], "error": str(error)})
            inventory.append(entry)
        for f, c in PAIRS:
            pair = f"{f}->{c}"
            if f not in frames or c not in frames:
                unavailable.append(
                    {
                        "symbol": symbol,
                        "contract": contract,
                        "pair": pair,
                        "reason": "missing_or_invalid_counterpart",
                    }
                )
                continue
            result = compare(frames[f], frames[c], f, c)
            record = {"symbol": symbol, "contract": contract, "pair": pair, **result}
            records.append(record)
            total = summaries[pair]
            total["contracts"] += 1
            for category in ("counts", "child_count_histogram", "partial_start_clock_histogram"):
                total[category].update(result[category])
            for field, values in result["fields"].items():
                acc = total["fields"].setdefault(
                    field,
                    {"full_comparable": 0, "full_mismatches": 0, "full_max_abs_difference": None},
                )
                for name in ("full_comparable", "full_mismatches"):
                    acc[name] += values[name]
                if values["full_max_abs_difference"] is not None:
                    acc["full_max_abs_difference"] = max(
                        acc["full_max_abs_difference"] or 0, values["full_max_abs_difference"]
                    )
        if number == 1 or number % 25 == 0 or number == len(groups):
            print(
                f"Raw alignment {number}/{len(groups)} contracts; {time.monotonic() - started:.1f}s",
                flush=True,
            )
    write_json(out / "files.json", inventory)
    write_json(out / "contracts.json", records)
    digest = sha256_bytes(
        json.dumps([(r["path"], r["sha256"]) for r in inventory], separators=(",", ":")).encode()
    )
    summary = {
        "protocol": PROTOCOL,
        "file_count": len(inventory),
        "contract_count": len(groups),
        "valid_bar_rows": sum(r.get("rows", 0) for r in inventory),
        "raw_inventory_sha256": digest,
        "comparisons": summaries,
        "invalid_files": errors,
        "unavailable_comparisons": unavailable,
        "ignored_files": ignored,
        "elapsed_seconds": time.monotonic() - started,
        "status": "invalid_inputs" if errors else "raw_audit_complete",
        "training_pairs_certified": False,
        "limitations": [
            "Nominal bar-start semantics are an assumption checked for numerical consistency, not vendor provenance.",
            "Partial coverage can reflect sessions or missing records; no exchange calendar was supplied.",
            "128-row counts certify row availability only, not unbroken calendar time or split eligibility.",
            "Raw snapshot validity cannot identify historical revisions or certify as-of availability.",
            "All raw files were audited, including research periods; no model loss or parameters were selected.",
            "No embeddings, normalized features, EMA or future forecasts were compared.",
        ],
    }
    write_json(out / "alignment_metrics.json", summary)
    if errors:
        raise ValueError(f"{len(errors)} invalid input files; inspect alignment_metrics.json")
    write_json(
        out / "completion.json",
        {
            "status": "raw_audit_complete",
            "training_pairs_certified": False,
            "report_sha256": {
                n: sha256_bytes((out / n).read_bytes())
                for n in ("manifest.json", "files.json", "contracts.json", "alignment_metrics.json")
            },
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.out)
    for pair, values in result["comparisons"].items():
        print(pair, json.dumps(values["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
