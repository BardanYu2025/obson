"""Shared macro geometry from exact, raw-audited same-contract endpoint links."""

from pathlib import Path

import numpy as np
import torch

from . import macro_history as core
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json


def geometry(row, bands=core.TRAIN_BANDS, shift=16):
    if len(bands) != 2 or bands[0][1] > bands[1][0] or any(hi - lo != 16 for lo, hi in bands):
        raise ValueError("Two disjoint ordered sixteen-coarse-bar query bands required")
    if not 1 <= shift <= 16:
        raise ValueError("Historical view shift1..16 required")
    nodes = {}
    for a, b, c, d in row["links"]:
        if not (0 <= a < b < 128 and 0 <= c < d < 128):
            raise ValueError("Invalid raw-audited interval")
        for fine, coarse in ((a, c), (b, d)):
            if coarse in nodes and nodes[coarse] != fine:
                raise ValueError("Ambiguous physical endpoint mapping")
            nodes[coarse] = fine
    queries = []
    for lo, hi in bands:
        selected = sorted(
            (c, f) for c, f in nodes.items() if lo <= c < hi and f + shift < 127 and c < 127
        )
        if len(selected) < 8:
            return None
        # Coarse-period valid links can be irregular (session aggregation). Split
        # the SAME verified nodes into four chronological groups, without invented
        # interpolation or requiring four equal wall-clock bins.
        bins = [
            i for i, group in enumerate(np.array_split(np.arange(len(selected)), 4)) for _ in group
        ]
        fine = [f for _, f in selected]
        if any(a >= b for a, b in zip(fine, fine[1:], strict=False)):
            raise ValueError("Nonmonotonic physical mapping")
        queries.append((selected, bins))
    return queries


def tensors(rows, shifts, device, bands=core.TRAIN_BANDS):
    if len(rows) != len(shifts) or not len(rows):
        raise ValueError("Rows and shifts required")
    ages = np.ones((3, len(rows), 2, 16), np.int64)
    mask = np.zeros((len(rows), 2, 16), bool)
    bins = np.zeros(mask.shape, np.int64)
    duration = np.zeros((len(rows), 2), np.float64)
    for i, (r, s) in enumerate(zip(rows, shifts, strict=True)):
        g = geometry(r, bands, int(s))
        if g is None:
            raise ValueError("Insufficient audited macro support")
        for q, (nodes, slots) in enumerate(g):
            c, f = np.asarray(nodes).T
            n = len(nodes)
            ages[:, i, q, :n] = np.stack([127 - f - int(s), 127 - f, 127 - c])
            mask[i, q, :n] = True
            bins[i, q, :n] = slots
            duration[i, q] = f[-1] - f[0]
    return {
        k: torch.tensor(v, device=device)
        for k, v in {"ages": ages, "mask": mask, "bins": bins, "duration": duration}.items()
    }, torch.tensor([r["pair"] == "15->30" for r in rows], device=device)


def eligible(rows, bands=core.TRAIN_BANDS, shift=16):
    return [i for i, r in enumerate(rows) if geometry(r, bands, shift) is not None]


def schedule(rows, seed, epoch, budget, eligible_ids=None):
    ids = eligible(rows) if eligible_ids is None else eligible_ids
    if not ids:
        raise ValueError("No eligible macro training geometry")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261002]))
    chosen = rng.choice(ids, budget, replace=True)
    shifts = rng.choice([1, 16], budget)
    return chosen, shifts


def verify(meta, out):
    lock = read_json(out / "macro_data_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Macro geometry manifest changed")
    verify_files(out, lock["files"])
    verify_files(Path(meta["task_source"]["source"]), lock["source_files"])
    for name in ("train", "val", "test", "cross_research"):
        source = read_json(out / f"{name}_macro_plan.json")["source"]
        if sha256(Path(source["source_path"])) != source["source_sha256"]:
            raise ValueError("Raw-linked source plan changed")
    return lock


def prepare(meta, out):
    from . import macro_history_run as run

    if (out / "macro_data_lock.json").exists():
        return verify(meta, out)
    names = []
    support = {}
    for split in ("train", "val", "test", "cross_research"):
        p = (
            (Path(meta["task_source"]["grounded_source"]) / "val_grounded_plan.json")
            if split == "val"
            else run.root(meta) / f"{split}_cross_plan.json"
        )
        raw = read_json(p)["rows"]
        bands = core.HELD_BANDS if split in ("test", "cross_research") else core.TRAIN_BANDS
        shift = 8 if split in ("test", "cross_research") else 16
        rows = [raw[i] for i in eligible(raw, bands, shift)]
        if split == "val":
            rows = [r for r in rows if r["pair"] == "15->30"]
        counts = {
            pair: sum(r["pair"] == pair for r in rows) for pair in sorted({r["pair"] for r in rows})
        }
        support[split] = {
            "source_rows": len(raw),
            "retained": len(rows),
            "excluded": len(raw) - len(rows),
            "pairs": counts,
            "source_path": str(p),
            "source_sha256": sha256(p),
        }
        atomic_json(support, out / "macro_support.json")
        required = ("15->30", "30->60") if split in ("test", "cross_research") else ("15->30",)
        for pair in required:
            rs = [r for r in rows if r["pair"] == pair]
            if len(rs) < 50 or len({r["key"] for r in rs}) < 5:
                raise ValueError(f"Insufficient macro support: {split}/{pair}; no training started")
        name = f"{split}_macro_plan.json"
        names.append(name)
        atomic_json(
            {"rows": rows, "bands": bands, "shift": shift, "source": support[split]}, out / name
        )
    names.append("macro_support.json")
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {n: sha256(out / n) for n in names},
            "source_files": {"manifest.json": meta["task_source"]["files"]["manifest.json"]},
        },
        out / "macro_data_lock.json",
    )


def evaluation_batch(meta, rows, split, shift, device, bands):
    from . import macro_history_run as run

    xd = run.xr.xd
    bank = np.load(xd.bank_info(run.tm(meta), split)[0], mmap_mode="r", allow_pickle=False)
    views = []
    for key, offset in (("fine_bank_end", shift), ("fine_bank_end", 0), ("coarse_bank_end", 0)):
        raw = np.stack([bank[r[key] - offset - 127 : r[key] - offset + 1] for r in rows])
        if raw.shape[1] != 128:
            raise ValueError("Incomplete physical window")
        views.append(
            dict(
                zip(("x", "y", "mask"), xd.cr.odr.normalized(raw, run.stats(meta)[0]), strict=True)
            )
        )
    batch = run.parent.batch_tensors(
        {k: np.concatenate([v[k] for v in views]) for k in views[0]}, device
    )
    geom, active = tensors(rows, np.full(len(rows), shift), device, bands)
    return batch, geom, active, core.targets(batch, geom, run.stats(meta)[0])
