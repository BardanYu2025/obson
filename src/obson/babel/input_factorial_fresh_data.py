"""Rebuild a new, standalone cohort after loss of historical AutoDL artifacts.

CPU raw-data arithmetic only. Historical sample identities/scalers are NOT
claimed to be recovered. Every arm uses this same newly locked cohort.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data as raw_data
from . import representation as partitions
from . import state_coverage_data as audit
from . import state_rollout_data as features
from .ae_extend import atomic_json
from .progress import progress
from .raw_snapshot import digest, no_links

SCHEMA = "babel-input-factorial-fresh-data-v1"
COUNTS = {"train": 4789, "val": 978, "test": 984, "cross_research": 439}
TRAIN_SYMBOLS = ("MA", "TA", "ag", "cu", "hc", "i", "j", "jm", "m", "p", "rb", "sr", "y")
CROSS_SYMBOLS = ("RM", "au", "c", "cs", "ru")
BOUNDS = {"train_until": "2024-08-07", "val_until": "2025-08-25", "test_until": "2026-09-11"}
ASOF = "2026-09-12 03:00:00"


def implementation():
    from . import activity_ablation, ae_context, ar_codec, structure

    modules = (raw_data, partitions, features, activity_ablation, ae_context, ar_codec, structure)
    return {Path(m.__file__).name: digest(m.__file__) for m in modules} | {
        Path(__file__).name: digest(__file__)
    }


def recipe():
    return {
        "schema": SCHEMA,
        "counts": COUNTS,
        "train_symbols": list(TRAIN_SYMBOLS),
        "cross_symbols": list(CROSS_SYMBOLS),
        "periods": [15, 30, 60],
        "bounds": BOUNDS,
        "asof": ASOF,
        "input": "own128 from full255 inside same contract/session partition",
        "sampling": "eligible main endpoints at fixed raw-row stride16; lowest SHA256(20261115|split|key|row), then key,row order",
        "scalers": "new train-only unique physical rows; masked population mean/std with fixed1e-5 floor",
        "legacy_equivalence": False,
        "forecast_targets": False,
    }


def choose_rows(series, split, count):
    candidates = []
    for s in series:
        same = partitions.time_mask(s, BOUNDS, "test" if split == "cross_research" else split)
        cumulative = np.r_[0, np.cumsum(~same)]
        for end in range(254, len(s.frame), 16):
            if not s.main[end] or cumulative[end + 1] != cumulative[end - 254]:
                continue
            row = {
                "key": s.key,
                "symbol": s.code,
                "period": s.period,
                "row": end,
                "end": str(s.frame.datetime.iloc[end]),
                "input_start": str(s.frame.datetime.iloc[end - 254]),
                "month": str(s.frame.datetime.iloc[end].to_period("M")),
            }
            rank = hashlib.sha256(f"20261115|{split}|{s.key}|{end}".encode()).hexdigest()
            candidates.append((rank, row))
    if len(candidates) < count:
        raise ValueError(
            f"Insufficient predeclared {split} candidates: {len(candidates)} < {count}"
        )
    chosen = sorted(
        [r for _, r in sorted(candidates, key=lambda v: v[0])[:count]],
        key=lambda r: (r["key"], r["row"]),
    )
    return [dict(r, original_index=i) for i, r in enumerate(chosen)], len(candidates)


def activity_mask(values):
    mask = np.ones((len(values), 7), bool)
    for output, flag in ((3, 27), (4, 23), (5, 24), (6, 25)):
        mask[:, output] = values[:, flag] > 0
    bad = ((values[:, 23] > 0) & (values[:, 22] == 0) & (values[:, 20] != 0)) | (
        (values[:, 24] > 0) & (np.abs(values[:, 21]) > np.arcsinh(1) + 1e-6)
    )
    mask[bad, 4:6] = False
    return mask


def fit_statistics(by_key, encoded, rows, *, split):
    if split != "train":
        raise ValueError("Only locked training rows may fit normalization")
    unique = defaultdict(set)
    for row in rows:
        unique[row["key"]].update(range(row["row"] - 254, row["row"] + 1))
    xs, ys, masks, changes = [], [], [], []
    for key in sorted(unique):
        indices = np.array(sorted(unique[key]))
        x = encoded[key][indices].astype(np.float64)
        close = by_key[key].frame.close.to_numpy(np.float64)
        previous = np.r_[by_key[key].frame.open.iloc[0], close[:-1]]
        delta = 100 * np.log(close[indices] / previous[indices])
        xs.append(x)
        ys.append(np.column_stack((np.zeros(len(x)), np.sinh(x[:, 1]), x[:, 18:23])))
        masks.append(activity_mask(x))
        changes.append(delta)
    x, y, mask = np.concatenate(xs), np.concatenate(ys), np.concatenate(masks)
    counts = mask.sum(0)
    if (counts < 2).any():
        raise ValueError("Insufficient training target support")
    means = np.where(mask, y, 0).sum(0) / counts
    scales = np.sqrt(np.where(mask, (y - means) ** 2, 0).sum(0) / counts)
    means[0], scales[0] = 0, 1  # Price queries use delta_scale and raw endpoint anchors.
    return {
        "x_mean": x.mean(0).tolist(),
        "x_scale": np.maximum(x.std(0), 1e-5).tolist(),
        "y_mean": means.tolist(),
        "y_scale": np.maximum(scales, 1e-5).tolist(),
        "delta_scale": [max(float(np.concatenate(changes).std()), 1e-5)],
        "fit_split": "train",
        "unique_physical_rows": len(x),
        "target_support": counts.tolist(),
    }


def build(raw_root, out):
    raw_root, out = no_links(raw_root), no_links(out)
    if out.exists():
        raise ValueError("Build into a new dataset directory; never overwrite a locked cohort")
    inventory = {str(p.relative_to(raw_root)): digest(p) for p in sorted(raw_root.glob("*/*m.csv"))}
    if len(inventory) != 1251:
        raise ValueError("Expected1251 single-contract CSVs; aggregate root CSVs are excluded")
    out.mkdir(parents=True)
    main, _ = raw_data.load_series(raw_root, TRAIN_SYMBOLS, (15, 30, 60), asof=ASOF)
    cross, _ = raw_data.load_series(raw_root, CROSS_SYMBOLS, (15, 30, 60), asof=ASOF)
    all_series = [*main, *cross]
    rows, pools = {}, {}
    for split, count in COUNTS.items():
        rows[split], pools[split] = choose_rows(
            cross if split == "cross_research" else main, split, count
        )
    partition = audit.row_audit(rows["train"], rows["val"])
    audit.row_audit(rows["val"], rows["test"])
    audit.row_audit(rows["train"], rows["test"])
    used = {row["key"] for rr in rows.values() for row in rr}
    by_key = {s.key: s for s in all_series if s.key in used}
    encoded = {}
    for i, (key, s) in enumerate(by_key.items(), 1):
        encoded[key] = features.encode(s.frame, s.period)
        if (s.frame.close <= 0).any() or not np.isfinite(encoded[key]).all():
            raise ValueError("Invalid raw prices/features")
        # One fixed per-contract truncation checks causal feature recurrence.
        end = min(r["row"] for rr in rows.values() for r in rr if r["key"] == key)
        if not np.array_equal(
            features.encode(s.frame.iloc[: end + 1], s.period), encoded[key][: end + 1]
        ):
            raise ValueError(f"Noncausal feature prefix: {key}")
        if i % 50 == 0:
            progress(
                f"fresh CPU features: {i}/{len(by_key)} selected contracts; no model execution"
            )
    stats = fit_statistics(by_key, encoded, rows["train"], split="train")
    atomic_json(stats, out / "statistics.json")
    atomic_json(inventory, out / "raw_inventory.json")
    for split, rr in rows.items():
        xx, raw = [], []
        for r in rr:
            end = r["row"]
            block = encoded[r["key"]][end - 254 : end + 1]
            xx.append(((block - stats["x_mean"]) / stats["x_scale"]).astype("float32"))
            raw.append(by_key[r["key"]].frame.close.iloc[end - 254 : end + 1].to_numpy(np.float64))
        np.savez_compressed(out / f"{split}.npz", x=np.stack(xx), close=np.stack(raw))
        atomic_json(rr, out / f"{split}_rows.json")
        progress(f"fresh {split}: {len(rr)} full255 blocks locked from {pools[split]} candidates")
    for relative, expected in inventory.items():
        if digest(raw_root / relative) != expected:
            raise ValueError("Raw snapshot changed during data construction")
    files = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    metadata = {
        "schema": SCHEMA,
        "recipe": recipe(),
        "implementation": implementation(),
        "files": files,
        "counts": COUNTS,
        "candidate_counts": pools,
        "partition_audit": partition,
        "statistics": stats,
        "selected_contracts": len(by_key),
        "raw_files": len(inventory),
        "historical_cache_recovered": False,
        "model_executions": 0,
        "limits": "New deterministic sample and train-only scales; not byte/numeric replay of historical cohorts; research data remain reused.",
    }
    atomic_json(metadata, out / "manifest.json")
    return metadata


def verify(root):
    root = no_links(root)
    meta = json.loads((root / "manifest.json").read_text())
    if (
        meta.get("schema") != SCHEMA
        or meta.get("recipe") != recipe()
        or meta.get("counts") != COUNTS
        or meta.get("implementation") != implementation()
    ):
        raise ValueError("Locked fresh dataset protocol differs")
    expected = {
        "statistics.json",
        "raw_inventory.json",
        *[f"{s}{suffix}" for s in COUNTS for suffix in (".npz", "_rows.json")],
    }
    if set(meta["files"]) != expected:
        raise ValueError("Incomplete fresh dataset")
    for relative, sha in meta["files"].items():
        if digest(no_links(root / relative)) != sha:
            raise ValueError(f"Fresh dataset fingerprint mismatch: {relative}")
    if json.loads((root / "statistics.json").read_text()) != meta["statistics"]:
        raise ValueError("Statistics differ")
    return meta


def cache(root, split):
    if split not in COUNTS:
        raise ValueError("Unknown split")
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_text())
    for name in (f"{split}.npz", f"{split}_rows.json"):
        if digest(no_links(root / name)) != meta["files"][name]:
            raise ValueError("Data cache changed")
    with np.load(root / f"{split}.npz", allow_pickle=False) as f:
        x, close = f["x"], f["close"]
    rows = json.loads((root / f"{split}_rows.json").read_text())
    if (
        len(rows) != COUNTS[split]
        or x.shape != (len(rows), 255, 28)
        or x.dtype != np.float32
        or close.shape != (len(rows), 255)
        or close.dtype != np.float64
        or not np.isfinite(x).all()
        or not np.isfinite(close).all()
        or not (close > 0).all()
    ):
        raise ValueError("Invalid locked feature/price arrays")
    return x, close, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.raw_root, args.out)
