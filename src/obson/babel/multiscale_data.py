"""F01/F03 immutable source, raw-price and train-only scale preparation.

The reviewed content-retention data adapter is reused read-only. This module
does not restore any source neural model or optimizer.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import content_retention_data as original
from . import multiscale_content as content
from . import multiscale_training as training
from . import state_coverage_data as coverage
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json

COMPLETION = original.COMPLETION
COUNTS = dict(original.COUNTS)
ASOF = datetime(2026, 9, 12, 3)
SCHEMA = "babel-multiscale-data-v1"


def safe_path(value):
    """Reject traversal and existing symlink components before any write."""
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("Parent traversal is not an experiment path")
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink experiment paths are forbidden")
    return path


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def implementation():
    return {Path(m.__file__).name: sha256(m.__file__) for m in (original, content, training)} | {
        Path(__file__).name: sha256(__file__)
    }


def raw_root(meta):
    """The actual historical load_raw root, derived from its bound source meta."""
    training_manifest = original.source.source.old.parent.tm(meta["coverage_meta"]["source_meta"])
    return safe_path(training_manifest["raw_root"])


def _separate_data(out, meta):
    for root in (
        safe_path(meta["coverage"]),
        safe_path(meta["coverage_meta"]["context"]),
        raw_root(meta),
    ):
        if out == root or out in root.parents or root in out.parents:
            raise ValueError("Output must be separate from every source and raw data root")


def verify(source, out):
    source, out = safe_path(source), safe_path(out)
    if not source.is_dir() or source == out or source in out.parents or out in source.parents:
        raise ValueError("Separate source and output directories required")
    for name in ("completion.json", "manifest.json"):
        if not safe_path(source / name).is_file():
            raise ValueError(f"Missing immutable source {name}")
    result = original.verify(source, out)
    _separate_data(out, result)
    if result["statistics"] != result["coverage_meta"]["statistics"]:
        raise ValueError("Source training statistics differ")
    return result


def _manifest(meta, out):
    out = safe_path(out)
    _separate_data(out, meta)
    # Historical atomic writers use deterministic sibling .tmp names. Reject
    # those symlinks too, before delegating any write to their unchanged code.
    if out.exists():
        for directory, dirs, files in os.walk(out, followlinks=False):
            for name in (*dirs, *files):
                safe_path(Path(directory) / name)
    path = safe_path(out / "manifest.json")
    if not path.is_file() or read_json(path) != meta:
        raise ValueError("Write the immutable matching manifest before data preparation")
    if meta["statistics"] != meta["coverage_meta"]["statistics"]:
        raise ValueError("Fixed training statistics differ from their source")
    return out, sha256(path)


def _rows(rows, split):
    if len(rows) != COUNTS[split]:
        raise ValueError("Original cohort count changed")
    seen = set()
    for row in rows:
        key = row["key"]
        parts = key.split("/")
        period = row["period"]
        index = row["row"]
        if (
            len(parts) != 3
            or str(period) != parts[1]
            or int(period) not in (15, 30, 60)
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 254
        ):
            raise ValueError("Invalid same-contract full255 row identity")
        start, end = datetime.fromisoformat(row["input_start"]), datetime.fromisoformat(row["end"])
        if start.tzinfo is not None or end.tzinfo is not None or not start < end:
            raise ValueError("Naive Asia/Shanghai full255 interval required")
        if end + timedelta(minutes=int(period)) > ASOF:
            raise ValueError("Endpoint was not closed at the original as-of")
        if (key, index) in seen:
            raise ValueError("Duplicate source parent endpoint")
        seen.add((key, index))


def cache(meta, split):
    if split not in COUNTS:
        raise ValueError("Unknown cohort")
    x, rows = original.cache(meta, split)
    _rows(rows, split)
    if x.shape != (len(rows), 255, 28) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Bound full255 normalized float32 features required")
    return x, rows


def _immutable_json(value, path):
    safe_path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable data receipt changed: {path.name}")
    else:
        atomic_json(value, path)


def _partition_audit(meta, out, manifest_hash):
    _, train = cache(meta, "train")
    _, val = cache(meta, "val")
    result = coverage.row_audit(train, val) | {
        "manifest_sha256": manifest_hash,
        "source_indexes": {s: meta["coverage_meta"]["source_indexes"][s] for s in ("train", "val")},
        "train_rows_sha256": json_hash(train),
        "val_rows_sha256": json_hash(val),
        "scope": "complete255 observed feature spans; inherited EMA prehistory unchanged",
    }
    _immutable_json(result, out / "data_partition_audit.json")


def _raw(meta, out, split):
    for name in (f"raw/{split}.npz", f"raw/{split}.json", f"{split}_rows.json"):
        if not safe_path(out / name).is_file():
            raise ValueError(f"Missing bound raw evidence: {name}")
    value = original.closes(out, split)
    if (
        value.dtype != np.float64
        or value.shape != (COUNTS[split], 255)
        or not np.isfinite(value).all()
        or not (value > 0).all()
    ):
        raise ValueError("Original positive float64 full255 close labels required")
    raw_receipt = read_json(out / "raw" / f"{split}.json")
    if raw_receipt["binding"]["source_index"] != meta["coverage_meta"]["source_indexes"][split]:
        raise ValueError("Raw source row index changed")
    return value


def _raw_binding(meta, out, split, rows):
    return {
        "schema": SCHEMA,
        "split": split,
        "manifest_sha256": sha256(out / "manifest.json"),
        "source_completion_sha256": COMPLETION,
        "source_index_sha256": meta["coverage_meta"]["source_indexes"][split],
        "statistics_sha256": json_hash(meta["statistics"]),
        "rows": len(rows),
        "rows_sha256": json_hash(rows),
        "rows_file_sha256": sha256(out / f"{split}_rows.json"),
        "raw_sha256": sha256(out / "raw" / f"{split}.npz"),
        "raw_receipt_sha256": sha256(out / "raw" / f"{split}.json"),
        "implementation": implementation(),
    }


def prepare(meta, out, splits):
    out, manifest_hash = _manifest(meta, out)
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(s not in COUNTS for s in splits):
        raise ValueError("Unique declared cohorts required")
    if any((s == "cross_research") != (splits[0] == "cross_research") for s in splits):
        raise ValueError("Prepare original and cross cohorts separately")
    _partition_audit(meta, out, manifest_hash)
    source_rows = {split: cache(meta, split)[1] for split in splits}
    safe_path(out / "raw")
    for split in splits:
        for suffix in ("npz", "json"):
            safe_path(out / "raw" / f"{split}.{suffix}")
        safe_path(out / f"{split}_rows.json")
    original.prepare(meta, out, splits)
    result = {}
    for split in splits:
        _raw(meta, out, split)
        if read_json(out / f"{split}_rows.json") != source_rows[split]:
            raise ValueError("Prepared raw row identities differ from source")
        receipt = _raw_binding(meta, out, split, source_rows[split])
        _immutable_json(receipt, out / f"multiscale_raw_{split}.json")
        result[split] = receipt
    return result


def closes(out, split):
    if split not in COUNTS:
        raise ValueError("Unknown cohort")
    out = safe_path(out)
    meta = read_json(safe_path(out / "manifest.json"))
    _manifest(meta, out)
    value = _raw(meta, out, split)
    rows = read_json(out / f"{split}_rows.json")
    _rows(rows, split)
    receipt = safe_path(out / f"multiscale_raw_{split}.json")
    if read_json(receipt) != _raw_binding(meta, out, split, rows):
        raise ValueError("Multiscale raw provenance changed")
    return value


def fit_scales(meta, out, raw, train_rows):
    """Fit once on exactly all locked training endpoints; verify only on reentry."""
    out, manifest_hash = _manifest(meta, out)
    _, rows = cache(meta, "train")
    if train_rows != rows:
        raise ValueError("Scale fitting training rows differ from locked source")
    bound = closes(out, "train")
    raw = np.asarray(raw)
    if raw.dtype != np.float64 or not np.array_equal(raw, bound):
        raise ValueError("Scale fitting must use the bound original float64 raw training prices")
    binding = {
        "manifest_sha256": manifest_hash,
        "source_completion_sha256": COMPLETION,
        "source_index_sha256": meta["coverage_meta"]["source_indexes"]["train"],
        "statistics_sha256": json_hash(meta["statistics"]),
        "rows_sha256": json_hash(rows),
        "raw_receipt_sha256": sha256(out / "multiscale_raw_train.json"),
        "raw_array_sha256": training.array_hash(raw),
        "implementation": implementation(),
    }
    report_path = safe_path(out / "multiscale_scales.json")
    receipt_path = safe_path(out / "multiscale_scales_receipt.json")
    if report_path.exists() or receipt_path.exists():
        if not report_path.is_file() or not receipt_path.is_file():
            raise ValueError("Incomplete scale commit; do not silently refit")
        receipt = read_json(receipt_path)
        if receipt["binding"] != binding or receipt["sha256"] != sha256(report_path):
            raise ValueError("Frozen train scales or provenance changed")
        report = read_json(report_path)
    else:
        report = training.fit_train_scales(raw, meta["statistics"], split="train")
        atomic_json(report, report_path)
        atomic_json({"binding": binding, "sha256": sha256(report_path)}, receipt_path)
    if report["contexts"] != COUNTS["train"] * 128 or report["training_rows"] != COUNTS["train"]:
        raise ValueError("Scale fit did not cover every locked training endpoint")
    content.huber_coordinate_weights(report["scales"])
    return report
