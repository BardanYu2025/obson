"""Read-only trace of blind charts to fingerprint-verified contract records."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_frame


def window_quality(frame, period):
    """Descriptive flags, not an exchange-calendar missing-bar detector."""
    ranges = (frame.high - frame.low).to_numpy()
    median = float(np.median(ranges))
    gaps = frame.open.to_numpy()[1:] - frame.close.to_numpy()[:-1]
    elapsed = frame.datetime.diff().dt.total_seconds().to_numpy()[1:] / 60
    order = np.argsort(-np.abs(gaps), kind="stable")[:5]
    transitions = []
    for j in order:
        before, after = frame.iloc[int(j)], frame.iloc[int(j) + 1]
        transitions.append({
            "bar_1based": int(j) + 2,
            "previous_time": str(before.datetime), "time": str(after.datetime),
            "previous_close": float(before.close), "open": float(after.open),
            "signed_gap": float(gaps[j]),
            "gap_over_median_range": float(abs(gaps[j]) / median) if median > 0 else None,
            "elapsed_minutes": float(elapsed[j]),
            "previous_volume": float(before.volume), "volume": float(after.volume),
        })
    return {
        "bars": len(frame), "start": str(frame.datetime.iloc[0]),
        "end": str(frame.datetime.iloc[-1]), "median_bar_range": median,
        "zero_volume_bars": int((frame.volume == 0).sum()),
        "flat_bars": int((ranges == 0).sum()),
        "intervals_longer_than_period": int((elapsed > period).sum()),
        "max_elapsed_minutes": float(elapsed.max()) if len(elapsed) else None,
        "largest_price_gaps": transitions,
    }


def audit_pairs(root, index, pairs, answer_key):
    packet = json.loads(Path(pairs).read_text())
    key = json.loads(Path(answer_key).read_text())
    if packet.get("schema") != "babel-pairs-v1" or key.get("schema") != "babel-pairs-v1":
        raise ValueError("Expected a pair-review packet and key")
    if packet["packet_id"] != key["packet_id"]:
        raise ValueError("Packet/key mismatch")
    with np.load(index, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        sid, rows = z["series"], z["row"]
    if meta.get("checkpoint") != key.get("checkpoint_sha256"):
        raise ValueError("Checkpoint/index/key mismatch")
    cache, charts = {}, {}
    for case in packet["cases"]:
        info = key["cases"][case["id"]]
        for side in ("query", "left", "right"):
            idx = info["query_index"] if side == "query" else info["hit_ids"][info[side]]
            if not isinstance(idx, int) or not 0 <= idx < len(rows):
                raise ValueError("Invalid index entry")
            source = meta["manifest"]["sources"][int(sid[idx])]
            code, period, contract = source["key"].split("/")
            if source["key"] not in cache:
                path = Path(root) / code / f"{contract}_{period}m.csv"
                df = validate_frame(pd.read_csv(path), str(path))
                digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
                if digest != source["sha256"]:
                    raise ValueError(f"Source fingerprint mismatch: {path}")
                cache[source["key"]] = df
            df = cache[source["key"]]
            end = int(rows[idx])
            lo = end - int(meta["window"]) + 1
            if lo < 0 or end >= len(df):
                raise ValueError("Window outside source records")
            frame = df.iloc[lo:end + 1]
            actual = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
            shown = np.array([[b[c] for c in ("open", "high", "low", "close")] for b in case[side]])
            if actual.shape != shown.shape or not np.array_equal(actual, shown):
                raise ValueError(f"Chart/source mismatch: {case['id']} {side}")
            if idx not in charts:
                charts[idx] = {"index_entry": idx, "source": source["key"],
                               "source_sha256": source["sha256"], "occurrences": [],
                               **window_quality(frame, int(period))}
            charts[idx]["occurrences"].append({"case": case["id"], "side": side})
    return {
        "packet_id": packet["packet_id"], "unique_charts": len(charts),
        "verified_sources": len(cache), "charts": list(charts.values()),
        "interpretation": "Raw charts matched to fingerprint-verified sources. Long intervals may be normal session breaks/holidays; price gaps alone do not establish bad data. No filtering, filling, adjustment or training performed. Main-contract history eligibility requires a separate audit.",
    }
