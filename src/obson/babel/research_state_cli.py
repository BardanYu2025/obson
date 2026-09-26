"""Consume one explicitly named contract CSV with a validated control600 research bundle."""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import research_state as rs
from . import window_state as ws
from .ae_extend import atomic_json
from .holdout_audit import read_json


def run(
    bundle,
    seed,
    key,
    period,
    csv,
    as_of,
    out,
    device="cpu",
    last_only=False,
    snapshot_in=None,
    snapshot_out=None,
):
    bundle, csv, out = Path(bundle).resolve(), Path(csv).resolve(), Path(out).resolve()
    observed = ws.streaming.timestamp(as_of)
    ws.check_key(key, period)
    inputs = [csv] + ([Path(snapshot_in).resolve()] if snapshot_in else [])
    outputs = [out] + ([Path(snapshot_out).resolve()] if snapshot_out else [])
    if len(set(outputs)) != len(outputs):
        raise ValueError("Distinct output paths required")
    for p in outputs:
        if p in inputs or p == bundle or bundle in p.parents or p in bundle.parents:
            raise ValueError("Output overlaps input or model bundle")
        if p.exists():
            raise FileExistsError(f"Use a new output path: {p}")
    engine = rs.load_bundle(bundle, seed, key, period, device)
    if snapshot_in:
        engine.restore(read_json(Path(snapshot_in)))
    if engine.count and engine.features.price.last_time + pd.Timedelta(minutes=period) > observed:
        raise ValueError("Snapshot contains bars later than as_of")
    raw = pd.read_csv(csv)
    # Filter by bar availability before reading any future OHLCV values.
    starts = raw.datetime.map(ws.streaming.timestamp)
    frame = ws.data.validate_frame(
        raw.loc[starts + pd.Timedelta(minutes=period) <= observed].copy(), str(csv)
    )
    bars = list(ws.frame_bars(frame, key, period))
    out.parent.mkdir(parents=True, exist_ok=True)
    if (
        snapshot_in
        and engine.count
        and bars
        and ws.streaming.timestamp(bars[0].datetime) <= engine.features.price.last_time
    ):
        raise ValueError("Snapshot continuation requires an append-only CSV of new bars")
    temp = out.with_name(out.name + ".tmp")
    if temp.exists():
        raise FileExistsError(f"Existing partial output: {temp}")
    try:
        with temp.open("x") as handle:
            if not bars:
                handle.write(
                    json.dumps(engine.current(), ensure_ascii=False, allow_nan=False) + "\n"
                )
            for i, bar in enumerate(bars):
                if last_only and i < len(bars) - 1:
                    engine.warm(bar, as_of=observed)
                    continue
                result = engine.push(bar, as_of=observed)
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        if snapshot_out:
            Path(snapshot_out).parent.mkdir(parents=True, exist_ok=True)
            atomic_json(engine.snapshot(), Path(snapshot_out))
        temp.replace(out)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {"rows": len(bars), "ready": engine.count >= ws.WARMUP, "output": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "key", "csv", "as-of", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42, choices=(42, 43))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--last-only", action="store_true")
    parser.add_argument("--snapshot-in")
    parser.add_argument("--snapshot-out")
    a = parser.parse_args()
    ws.ea.bb.ab.configure_runtime()
    print(
        json.dumps(
            run(
                a.bundle,
                a.seed,
                a.key,
                a.period,
                a.csv,
                a.as_of,
                a.out,
                a.device,
                a.last_only,
                a.snapshot_in,
                a.snapshot_out,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
