"""Restrict a formal study to whole, historically qualified source windows."""

from datetime import date

import numpy as np


def select_sources(snapshot, scope):
    if scope is None:
        return snapshot
    ranges = scope["ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Need at least one dated unit/calendar scope")
    for r in ranges:
        if r["product"] not in snapshot["clocks"] or date.fromisoformat(
            r["first"]
        ) > date.fromisoformat(r["last"]):
            raise ValueError("Invalid product or date range")
        for key in (
            "calendar_evidence",
            "volume_evidence",
            "oi_evidence",
            "contract_selection",
        ):
            if not isinstance(r[key], str) or not r[key].strip():
                raise ValueError(f"Missing {key}")
        for key in ("volume_multiplier", "oi_multiplier"):
            if not np.isfinite(r[key]) or r[key] <= 0:
                raise ValueError(f"Invalid {key}")
    for i, r in enumerate(ranges):
        for other in ranges[:i]:
            if (
                other["product"] == r["product"]
                and other["first"] <= r["last"]
                and r["first"] <= other["last"]
            ):
                raise ValueError("Overlapping dated evidence ranges")
    records = []
    for rec in snapshot["files"]:
        matches = [
            r
            for r in ranges
            if r["product"] == rec["product"]
            and r["first"] <= rec["first"][:10]
            and rec["last"][:10] <= r["last"]
        ]
        if matches:
            r = matches[0]
            records.append(
                {
                    **rec,
                    "volume_multiplier": r["volume_multiplier"],
                    "oi_multiplier": r["oi_multiplier"],
                }
            )
    if not records:
        raise ValueError("No entire signed source window fits the dated scope")
    products = {r["product"] for r in records}
    return {
        **snapshot,
        "files": records,
        "clocks": {p: snapshot["clocks"][p] for p in sorted(products)},
        "scope": scope,
    }
