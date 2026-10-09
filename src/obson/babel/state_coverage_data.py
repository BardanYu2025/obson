"""Immutable V06-qualified inputs; new target boundaries checked against raw replay."""

from collections import Counter
from pathlib import Path

import numpy as np

from . import recovery_context_run as source
from . import state_coverage as core
from . import uniform_context_data as original
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

COMPLETION = "d794c37e0fd8fe5717b1617a4225ffc9e39f667c4945c05254170eca985c01f2"
SPLITS = ("train", "val", "test", "cross_research")


def verify(root, out):
    if sha256(root / "completion.json") != COMPLETION:
        raise ValueError("Expected reviewed V06 completion")
    source.r0.required_file(
        root, "manifest.json", read_json(root / "completion.json")["files"]["manifest.json"]
    )
    manifest = read_json(root / "manifest.json")
    if manifest["implementation"] != source.implementation():
        raise ValueError("Historical V06 implementation changed")
    model_meta, _, context, _ = source.verify_source(Path(manifest["source"]), out)
    source.interface.separate(out, root, context)
    stats = source.old.parent.stats(model_meta)[0]
    if sha256(context / "statistics.json") != source.STATISTICS_SHA256:
        raise ValueError("Wrong train scalers")
    indexes = {s: sha256(context / f"cache/{s}_index.json") for s in SPLITS}
    return {
        "source_meta": model_meta,
        "context": str(context),
        "statistics": stats,
        "source_indexes": indexes,
        "source_completion_sha256": COMPLETION,
        "source_manifest_sha256": sha256(root / "manifest.json"),
    }


def cache(meta, out, split):
    if split not in SPLITS:
        raise ValueError("Unknown split")
    if split in SPLITS[2:]:
        from .state_coverage_evaluate import check_readouts

        check_readouts(out)
    root = Path(meta["context"])
    if sha256(root / f"cache/{split}_index.json") != meta["source_indexes"][split]:
        raise ValueError("Input index changed")
    x, rows = source.old.data.cache(root, split)
    if x.shape != (len(rows), 255, 28) or not np.isfinite(x).all():
        raise ValueError("Invalid full255 input")
    if split in ("train", "val") and len(rows) != {"train": 4789, "val": 978}[split]:
        raise ValueError("Locked training/validation inventory changed")
    return x, rows


def row_audit(train, val):
    by = {}
    for r in train:
        by.setdefault(r["key"], []).append((r["input_start"], r["end"]))
    overlap = [
        r
        for r in val
        if any(a <= r["end"] and r["input_start"] <= b for a, b in by.get(r["key"], []))
    ]
    if overlap:
        raise ValueError("Full255 training/validation overlap")
    return {"train": len(train), "val": len(val), "overlapping_intervals": 0}


def raw_audit(meta, out, splits):
    """Read-only raw feature rebuild; no model evaluation or fitting."""
    tag = "_".join(splits)
    path = out / f"raw_{tag}.json"
    cross = splits == ("cross_research",)
    # Loading checks the original raw file inventory even on a resumed session.
    series, bounds = source.old.parent.xr.xd.load_raw(
        source.old.parent.tm(meta["source_meta"]), cross=cross
    )
    if path.exists():
        report = read_json(path)
        if report["source_indexes"] != {s: meta["source_indexes"][s] for s in splits}:
            raise ValueError("Raw audit lineage changed")
        return report
    lookup = {s.key: s for s in series}
    stats = meta["statistics"]
    report = {}
    for split in splits:
        x, rows = cache(meta, out, split)
        by = {}
        counts = Counter()
        for i, r in enumerate(rows):
            by.setdefault(r["key"], []).append((i, r))
            for p in range(1, 129):
                counts[(r["key"], int(r["row"]) - 128 + p)] += 1
        maxdiff = 0.0
        count = 0
        labeldiff = 0.0
        for key, items in by.items():
            s = lookup[key]
            features = original.original.encode(s.frame, s.period)
            same = original.rp.time_mask(s, bounds, "test" if cross else split)
            first = int(items[0][1]["row"])
            if not np.array_equal(
                features[: first + 1], original.original.encode(s.frame.iloc[: first + 1], s.period)
            ):
                raise ValueError("Raw feature future leak")
            for i, r in items:
                end = int(r["row"])
                if (
                    not original.eligible(end, same)
                    or str(s.frame.datetime.iloc[end]) != r["end"]
                    or str(s.frame.datetime.iloc[end - 254]) != r["input_start"]
                ):
                    raise ValueError("Raw full255 identity/partition mismatch")
                rebuilt = (
                    (features[end - 254 : end + 1] - stats["x_mean"]) / stats["x_scale"]
                ).astype("float32")
                d = float(np.max(np.abs(rebuilt - x[i])))
                maxdiff = max(maxdiff, d)
                if not np.allclose(rebuilt, x[i], atol=1e-6, rtol=2e-5):
                    raise ValueError("Raw input replay differs")
            # Independent per-window NumPy target versus new torch gather at all128 positions.
            import torch

            i = items[0][0]
            ps = np.arange(1, 129)[None]
            y, m = core.targets(torch.tensor(x[i : i + 1]), torch.tensor(ps), stats)
            oy, om = core.oracle_targets(x[i : i + 1], ps, stats)
            labeldiff = max(labeldiff, float(np.max(np.abs(y.numpy() - oy))))
            if not np.array_equal(m.numpy(), om) or not np.allclose(
                y.numpy(), oy, atol=1e-5, rtol=2e-5
            ):
                raise ValueError("Independent full-position labels differ")
            count += 1
            if count % 25 == 0:
                progress(f"F01 raw {split}: {count}/{len(by)} contracts; zero updates")
        report[split] = {
            "rows": len(rows),
            "raw_max_abs": maxdiff,
            "label_max_abs": labeldiff,
            "contracts": count,
            "unique_physical_endpoints": len(counts),
            "endpoint_occurrences": sum(counts.values()),
            "maximum_endpoint_multiplicity": max(counts.values()),
            "multiplicity_histogram": dict(sorted(Counter(counts.values()).items())),
        }
        atomic_json(rows, out / f"{split}_rows.json")
    result = {"source_indexes": {s: meta["source_indexes"][s] for s in splits}, "splits": report}
    atomic_json(result, path)
    return result
