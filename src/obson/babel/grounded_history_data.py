"""Immutable common-price targets, with an independent validation-time cohort."""

from pathlib import Path

import numpy as np

from . import grounded_history as core
from . import shared_history_data as sd
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json


def eligible(rows, band=(103, 115), shift=16):
    return [
        i for i, r in enumerate(rows) if len(sd.geometry(r, band, shift) or ()) >= 2 * core.SLOTS
    ]


def schedule(rows, seed, epoch, budget):
    ids = eligible(rows)
    chosen, shifts = sd.schedule([rows[i] for i in ids], seed, epoch, budget)
    return np.asarray(ids)[chosen], shifts


def fixed_pairs(rows, band=(103, 115), shift=16):
    ids = eligible(rows, band, shift)
    selected = [rows[i] for i in ids]
    groups, _ = sd.strata(selected, band, shift)
    return [selected[i] for group in groups for i in group[: len(group) // 2 * 2]]


def verify(meta, out):
    lock = read_json(out / "grounded_data_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Grounded data manifest differs")
    verify_files(out, lock["files"])
    return lock


def prepare(meta, out):
    from . import grounded_history_run as run

    if (out / "grounded_data_lock.json").exists():
        verify(meta, out)
        return
    sd.prepare(meta, out, run.root(meta))
    xd = run.xr.xd
    source_meta = run.tm(meta)
    series, bounds = xd.load_raw(source_meta)
    _, specs_path = xd.bank_info(source_meta, "val")
    specs = read_json(specs_path)
    originals = read_json(Path(source_meta["source"]) / "val_inventory.json")
    anchors = xd.evaluation_anchors(series, bounds, "val", specs)
    rows, excluded = xd.plan_pairs(series, bounds, "val", specs, originals, anchors)
    rows = fixed_pairs(rows)
    # Match the planned trained cross task.30->60 remains a held research task.
    rows = [r for r in rows if r["pair"] == "15->30"]
    if len(rows) < 50 or len({r["key"] for r in rows}) < 5:
        atomic_json({"rows": len(rows), "excluded": excluded}, out / "support_failure.json")
        raise ValueError("Qualification requires50 validation pairs and5 contracts")
    names = ["val_grounded_plan.json"]
    atomic_json({"rows": rows, "excluded": excluded}, out / names[0])
    # Filter research plans on geometry alone, before inspecting any outcomes.
    for split in ("test", "cross_research"):
        p = read_json(out / f"{split}_held_shared_plan.json")
        p["rows"] = fixed_pairs(p["rows"], tuple(p["band"]), p["shift"])
        p.pop("ids")
        for pair in ("15->30", "30->60"):
            rs = [r for r in p["rows"] if r["pair"] == pair]
            if len(rs) < 50 or len({r["key"] for r in rs}) < 5:
                raise ValueError(f"Insufficient grounded research cohort: {split}/{pair}")
        name = f"{split}_grounded_plan.json"
        atomic_json(p, out / name)
        names.append(name)
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {n: sha256(out / n) for n in names},
        },
        out / "grounded_data_lock.json",
    )


def evaluation_batch(meta, rows, split, shift, device, band=(103, 115)):
    from . import grounded_history_run as run

    xd = run.xr.xd
    bank = np.load(xd.bank_info(run.tm(meta), split)[0], mmap_mode="r", allow_pickle=False)
    views = []
    for key, offset in (("fine_bank_end", shift), ("fine_bank_end", 0), ("coarse_bank_end", 0)):
        raw = np.stack([bank[r[key] - offset - 127 : r[key] - offset + 1] for r in rows])
        if raw.shape[1] != 128:
            raise ValueError("Incomplete source window")
        views.append(
            dict(
                zip(("x", "y", "mask"), xd.cr.odr.normalized(raw, run.stats(meta)[0]), strict=True)
            )
        )
    batch = run.parent.batch_tensors(
        {k: np.concatenate([v[k] for v in views]) for k in views[0]}, device
    )
    ages, mask, active = sd.tensors(rows, np.full(len(rows), shift), device, band, shift)
    q = core.queries(ages, mask)
    return batch, q, active, core.targets(batch, q, run.stats(meta)[0])
