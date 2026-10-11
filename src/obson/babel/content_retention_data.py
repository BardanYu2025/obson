"""Read-only qualified source lineage and original raw labels for F03."""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from . import content_retention as core
from . import overlap_content_run as diagnostic
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json

COMPLETION = "835cce0452c27173f808b530871624b59041dceb71552074bf32e4c722e4139d"
COUNTS = {"train": 4789, "val": 978, "test": 984, "cross_research": 439}
source = diagnostic.data
save_arrays = diagnostic.prior.ev.save_arrays


def verify(root, out):
    if sha256(root / "completion.json") != COMPLETION:
        raise ValueError("Expected reviewed overlap-content completion, not another run")
    done = diagnostic.check_complete(root)
    if done["status"] != "f03_diagnostic_complete_requires_review" or done["optimizer_updates"]:
        raise ValueError("Completed zero-update diagnostic required")
    spec = read_json(root / "manifest.json")
    coverage, meta = diagnostic.verify_source(Path(spec["source"]), out)
    source.source.interface.separate(out, root, coverage, Path(meta["context"]))
    return {"coverage": str(coverage), "coverage_meta": meta, "statistics": meta["statistics"]}


def cache(meta, split):
    if split not in COUNTS:
        raise ValueError("Unknown cohort")
    x, rows = source.cache(meta["coverage_meta"], Path(meta["coverage"]), split)
    if len(rows) != COUNTS[split]:
        raise ValueError("Original cohort count changed")
    return x, rows


def prepare(meta, out, splits):
    """CPU feature/raw replay. No source-directory writes and no model access."""
    if not splits or any(
        (s == "cross_research") != (splits[0] == "cross_research") for s in splits
    ):
        raise ValueError("Prepare original and cross cohorts separately")
    cm = meta["coverage_meta"]
    series, bounds = source.source.old.parent.xr.xd.load_raw(
        source.source.old.parent.tm(cm["source_meta"]), cross=splits[0] == "cross_research"
    )
    lookup = {s.key: s for s in series}
    folder = out / "raw"
    folder.mkdir(exist_ok=True)
    result = {}
    for split in splits:
        x, rows = cache(meta, split)
        binding = {
            "source_index": cm["source_indexes"][split],
            "manifest": sha256(out / "manifest.json"),
        }
        path, receipt = folder / f"{split}.npz", folder / f"{split}.json"
        if receipt.exists():
            saved = read_json(receipt)
            if saved["binding"] != binding or sha256(path) != saved["sha256"]:
                raise ValueError("Raw-label cache changed")
            result[split] = saved
            continue
        raw = np.empty((len(rows), 255), dtype=np.float64)
        by = defaultdict(list)
        for i, row in enumerate(rows):
            by[row["key"]].append((i, row))
        maximum = 0.0
        for key, items in by.items():
            item = lookup[key]
            features = source.original.original.encode(item.frame, item.period)
            same = source.original.rp.time_mask(
                item, bounds, "test" if split == "cross_research" else split
            )
            for i, row in items:
                end = int(row["row"])
                if datetime.fromisoformat(row["end"]) + timedelta(
                    minutes=int(row["period"])
                ) > datetime(2026, 9, 12, 3):
                    raise ValueError("Raw endpoint is not closed by the original as-of")
                if (
                    not source.original.eligible(end, same)
                    or str(item.frame.datetime.iloc[end]) != row["end"]
                    or str(item.frame.datetime.iloc[end - 254]) != row["input_start"]
                ):
                    raise ValueError("Full255 physical row/partition changed")
                rebuilt = (
                    (features[end - 254 : end + 1] - cm["statistics"]["x_mean"])
                    / cm["statistics"]["x_scale"]
                ).astype("float32")
                maximum = max(maximum, float(np.max(np.abs(x[i] - rebuilt))))
                if not np.allclose(x[i], rebuilt, atol=1e-6, rtol=2e-5):
                    raise ValueError("Raw feature replay differs")
                raw[i] = item.frame.close.iloc[end - 254 : end + 1].to_numpy(dtype=np.float64)
        ps = torch.tensor(core.POSITIONS).expand(len(x), -1)
        y, mask = core.source.targets(torch.tensor(x), ps, cm["statistics"])
        raw_y = core.raw_targets(torch.tensor(raw), ps, cm["statistics"])
        check = diagnostic.prior.r0.compare(raw_y.numpy(), y.numpy()[..., 0], 1e-5, 2e-5)
        diagnostic.prior.r0.save_checks(out, f"{split}_raw_price", {"raw_price": check})
        save_arrays(path, {"close": raw})
        saved = {
            "binding": binding,
            "sha256": sha256(path),
            "rows": len(rows),
            "contracts": len(by),
            "feature_max_abs": maximum,
            "price_target_check": check,
            "raw_inventory": "original load_raw hash-bound inventory",
            "raw_activity_certification": False,
        }
        atomic_json(saved, receipt)
        atomic_json(rows, out / f"{split}_rows.json")
        result[split] = saved
    return result


def closes(out, split):
    receipt = read_json(out / "raw" / f"{split}.json")
    path = out / "raw" / f"{split}.npz"
    if receipt["binding"]["manifest"] != sha256(out / "manifest.json") or receipt[
        "sha256"
    ] != sha256(path):
        raise ValueError("Bound raw-price target changed")
    with np.load(path, allow_pickle=False) as saved:
        return saved["close"]


def load_source(meta, seed, device):
    # ev.load verifies the D fixed-last120 binding. Macro remains the external
    # research reference; it is NOT silently substituted as this trial's start.
    return diagnostic.prior.ev.load(
        meta["coverage_meta"], Path(meta["coverage"]), f"D_s{seed}", device
    )
