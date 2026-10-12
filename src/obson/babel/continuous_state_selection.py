"""F11 fixed-size consecutive research segments, selected without model outputs.

The caller supplies original representation.time_mask(..., 'test') masks for
both research splits. Timestamps are local bar starts; as-of uses nominal ends.
"""

import hashlib

import numpy as np
import pandas as pd

SCHEMA = "babel-continuous-state-selection-v1"
ASOF = "2026-09-12 03:00:00"
SPLITS = ("test", "cross_research")
PERIODS = (15, 30, 60)
SEGMENT_LENGTH = 512
PER_CELL = 2
ORACLE_OFFSETS = (0, 31, 63, 95, 127, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 511)
STRESS = {
    "split": "test",
    "source_index": 171,
    "key": "ag/15/SHFE.ag2606",
    "row": 5759,
    "first": 5632,
    "last": 6143,
}


def protocol():
    return {
        "schema": SCHEMA,
        "asof": ASOF,
        "length": SEGMENT_LENGTH,
        "per_cell": PER_CELL,
        "cells": [[split, period] for split in SPLITS for period in PERIODS],
        "ranking": "sha256(utf8(split + '/' + key)); distinct symbol per cell",
        "endpoint": "earliest original report row with512 eligible consecutive endpoints",
        "context": "each128-row history wholly inside original test time mask",
        "oracle_offsets": list(ORACLE_OFFSETS),
        "stress": dict(STRESS),
        "selection_uses_model_outputs": False,
        "automatic_size_reduction": False,
    }


def _timestamps(series, key):
    parts = key.split("/")
    if (
        len(parts) != 3
        or not all(parts)
        or series.key != key
        or series.code != parts[0]
        or type(series.period) is not int
        or series.period not in PERIODS
        or str(series.period) != parts[1]
    ):
        raise ValueError(f"Single-contract series identity mismatch: {key}")
    dates = pd.DatetimeIndex(series.frame.datetime)
    if dates.tz is not None or dates.hasnans or not len(dates):
        raise ValueError(f"Naive valid Asia/Shanghai raw bar timestamps required: {key}")
    if np.any(
        np.diff(dates.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        < series.period * 60 * 1_000_000_000
    ):
        raise ValueError(f"Duplicate, unordered or sub-period raw bars: {key}")
    return dates


def _eligible(first, dates, same, period):
    last = first + SEGMENT_LENGTH - 1
    return (
        first >= 127
        and last < len(dates)
        and bool(np.all(same[first - 127 : last + 1]))
        and dates[last] + pd.Timedelta(minutes=period) <= pd.Timestamp(ASOF)
    )


def _record(split, index, row, series, dates, first, role, identifier):
    last = first + SEGMENT_LENGTH - 1
    return {
        "id": identifier,
        "split": split,
        "key": row["key"],
        "period": series.period,
        "symbol": series.code,
        "first": first,
        "last": last,
        "source_index": index,
        "source_row": row["row"],
        "source_offset": row["row"] - first,
        "source_original_index": row.get("original_index"),
        "role": role,
        "first_time": str(dates[first]),
        "last_time": str(dates[last]),
        "input_start": str(dates[first - 127]),
        "feature_first_time": str(dates[0]),
        "source_end": row["end"],
        "source_input_start": row["input_start"],
        "available_at": str(dates[last] + pd.Timedelta(minutes=series.period)),
        "asof": ASOF,
        "oracle_offsets": list(ORACLE_OFFSETS),
    }


def select_segments(series_by_split, rows_by_split, masks_by_split):
    """Return12 coverage segments plus the separately reported known-flat stress.

    Every supplied source row is checked before selection. No raw price, label,
    metric or neural output influences the coverage choice. Missing support is
    an error, never permission to shorten a segment or reduce the cell count.
    """
    lookup = {}
    grouped = {}
    for split in SPLITS:
        if any(
            split not in mapping for mapping in (series_by_split, rows_by_split, masks_by_split)
        ):
            raise ValueError(f"Missing required research split: {split}")
        by_key = {}
        for index, row in enumerate(rows_by_split[split]):
            key = row.get("key")
            if key not in series_by_split[split] or key not in masks_by_split[split]:
                raise ValueError(f"Missing raw series/original time mask: {split}/{key}")
            if key not in by_key:
                series = series_by_split[split][key]
                dates = _timestamps(series, key)
                same = np.asarray(masks_by_split[split][key])
                if same.dtype != np.bool_ or same.shape != (len(dates),):
                    raise ValueError(f"Original time mask must be a matching boolean vector: {key}")
                lookup[split, key] = (series, dates, same)
                by_key[key] = []
            series, dates, same = lookup[split, key]
            end = row.get("row")
            if (
                type(end) is not int
                or not 254 <= end < len(dates)
                or row.get("period") != series.period
                or row.get("symbol") != series.code
                or row.get("end") != str(dates[end])
                or row.get("input_start") != str(dates[end - 254])
                or not np.all(same[end - 254 : end + 1])
                or dates[end] + pd.Timedelta(minutes=series.period) > pd.Timestamp(ASOF)
            ):
                raise ValueError(
                    f"Original full255 source row identity/partition differs: {split}:{index}"
                )
            by_key[key].append((index, row))
        grouped[split] = by_key
    selected = []
    for split in SPLITS:
        for period in PERIODS:
            candidates = []
            for key, rows in grouped[split].items():
                series, dates, same = lookup[split, key]
                if series.period != period:
                    continue
                earliest = next(
                    (
                        (i, r)
                        for i, r in sorted(rows, key=lambda x: (x[1]["row"], x[0]))
                        if _eligible(r["row"], dates, same, period)
                    ),
                    None,
                )
                if earliest is not None:
                    rank = hashlib.sha256(f"{split}/{key}".encode()).hexdigest()
                    candidates.append((rank, key, earliest))
            symbols = set()
            cell = []
            for _, key, (index, row) in sorted(candidates):
                series, dates, _ = lookup[split, key]
                if series.code in symbols:
                    continue
                symbols.add(series.code)
                cell.append(
                    _record(
                        split,
                        index,
                        row,
                        series,
                        dates,
                        row["row"],
                        "coverage",
                        f"{split}_p{period}_{len(cell) + 1}",
                    )
                )
                if len(cell) == PER_CELL:
                    break
            if len(cell) != PER_CELL:
                raise ValueError(
                    f"Insufficient512-endpoint distinct-symbol coverage: {split}/{period}"
                )
            selected.extend(cell)
    split, index = STRESS["split"], STRESS["source_index"]
    if len(rows_by_split[split]) <= index:
        raise ValueError("Missing locked known-flat source row test:171")
    row = rows_by_split[split][index]
    if row["key"] != STRESS["key"] or row["row"] != STRESS["row"]:
        raise ValueError("Locked known-flat test:171 identity differs")
    series, dates, same = lookup[split, row["key"]]
    if not _eligible(STRESS["first"], dates, same, series.period):
        raise ValueError("Locked known-flat512-endpoint stress context unavailable")
    selected.append(
        _record(
            split,
            index,
            row,
            series,
            dates,
            STRESS["first"],
            "known_flat_stress",
            "known_flat_test171",
        )
    )
    return selected
