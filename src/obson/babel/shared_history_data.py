"""Immutable raw-audited common intervals and nuisance-matched time pairs."""

from collections import Counter, defaultdict

import numpy as np
import torch

from . import shared_history as core
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json


def geometry(row, band=core.TRAIN_BAND, max_shift=16):
    # Each link was verified against exact same-contract raw OHLCV/OI aggregation.
    links = [
        x for x in row["links"] if band[0] <= x[2] < x[3] <= band[1] and x[1] + max_shift < 127
    ]
    if len(links) < core.MIN_INTERVALS:
        return None
    if any(not (0 <= a < b < 127 and 0 <= c < d < 127) for a, b, c, d in links):
        raise ValueError("Current/future interval or inverted mapping")
    if len({tuple(v) for v in links}) != len(links):
        raise ValueError("Duplicate common interval")
    return tuple(map(tuple, links))


def strata(rows, band=core.TRAIN_BAND, max_shift=16):
    result = defaultdict(list)
    excluded = Counter()
    for i, row in enumerate(rows):
        g = geometry(row, band, max_shift)
        if g is None:
            excluded["insufficient_common_intervals"] += 1
            continue
        result[(row["key"], row["pair"], g)].append(i)
    result = [ids for _, ids in sorted(result.items()) if len(ids) >= 2]
    # Each retained row must describe a distinct end time inside its stratum.
    for ids in result:
        if len({rows[i]["end"] for i in ids}) != len(ids):
            raise ValueError("Duplicate physical endpoint in stratum")
    excluded["singleton_geometry"] = (
        len(rows) - sum(map(len, result)) - excluded["insufficient_common_intervals"]
    )
    return result, dict(excluded)


def schedule(rows, seed, epoch, budget):
    if budget % 2:
        raise ValueError("Even exposure budget for matched temporal pairs required")
    groups, _ = strata(rows)
    if not groups:
        raise ValueError("No distinct-time, identical-contract/geometry pairs")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260930]))
    probabilities = np.asarray(list(map(len, groups)), float)
    chosen = rng.choice(len(groups), budget // 2, p=probabilities / probabilities.sum())
    ids = np.concatenate([rng.choice(groups[i], 2, replace=False) for i in chosen])
    shifts = np.repeat(rng.choice([1, 16], budget // 2), 2)
    return ids, shifts


def tensors(rows, shifts, device, band=core.TRAIN_BAND, max_shift=16):
    geometry_rows = [geometry(r, band, max_shift) for r in rows]
    if any(g is None for g in geometry_rows):
        raise ValueError("Selected pair lacks validated common intervals")
    for i in range(0, len(rows), 2):
        if (
            i + 1 >= len(rows)
            or (rows[i]["key"], rows[i]["pair"], geometry_rows[i], int(shifts[i]))
            != (rows[i + 1]["key"], rows[i + 1]["pair"], geometry_rows[i + 1], int(shifts[i + 1]))
            or rows[i]["end"] == rows[i + 1]["end"]
        ):
            raise ValueError("Temporal nuisance pair identity/geometry mismatch")
    max_intervals = band[1] - band[0]
    index = np.zeros((len(rows), max_intervals, 4), np.int64)
    valid = np.zeros((len(rows), max_intervals), bool)
    for i, g in enumerate(geometry_rows):
        index[i, : len(g)] = g
        valid[i, : len(g)] = True
    a = index[:, :, :2] + np.asarray(shifts)[:, None, None]
    coords = np.stack([127 - a, 127 - index[:, :, :2], 127 - index[:, :, 2:]])
    return (
        torch.tensor(coords, device=device),
        torch.tensor(valid, device=device),
        torch.tensor([r["pair"] == "15->30" for r in rows], device=device),
    )


def prepare(meta, out, training_root):
    source_files = {
        f"{s}_cross_plan.json": sha256(training_root / f"{s}_cross_plan.json")
        for s in ("train", "test", "cross_research")
    }
    if (out / "shared_data_lock.json").exists():
        lock = read_json(out / "shared_data_lock.json")
        if (
            lock["manifest_sha256"] != sha256(out / "manifest.json")
            or lock["sources"] != source_files
        ):
            raise ValueError("Shared geometry source/configuration changed")
        for n, h in lock["files"].items():
            if sha256(out / n) != h:
                raise ValueError("Shared geometry fingerprint changed")
        return
    summaries = {}
    plans = {}
    for split in ("train", "test", "cross_research"):
        rows = read_json(training_root / f"{split}_cross_plan.json")["rows"]
        for name, band, shift in [("trained", core.TRAIN_BAND, 16), ("held", core.HELD_BAND, 8)]:
            groups, excluded = strata(rows, band, shift)
            ids = (
                np.concatenate([np.asarray(g[: len(g) // 2 * 2]) for g in groups])
                if groups
                else np.array([], int)
            )
            # Evaluation matches fixed distinct-time pairs before seeing model scores.
            summary = {
                "rows": len(rows),
                "paired_rows": len(ids),
                "strata": len(groups),
                "excluded": excluded,
                "pairs": dict(Counter(rows[i]["pair"] for i in ids)),
            }
            summaries[f"{split}/{name}"] = summary
            if split != "train":
                for pair in ("15->30", "30->60"):
                    selected = [rows[i] for i in ids if rows[i]["pair"] == pair]
                    if (
                        len(selected) < 50
                        or len({r["week"] for r in selected}) < 5
                        or len({r["key"] for r in selected}) < 5
                    ):
                        atomic_json(summaries, out / "shared_support.json")
                        raise ValueError(
                            f"Insufficient raw-only evaluation support: {split}/{name}/{pair}"
                        )
                plans[f"{split}_{name}_shared_plan.json"] = {
                    "ids": ids.tolist(),
                    "rows": [rows[i] for i in ids],
                    "band": list(band),
                    "shift": shift,
                    "summary": summary,
                }
    for n, p in plans.items():
        atomic_json(p, out / n)
    atomic_json(summaries, out / "shared_support.json")
    names = [*plans, "shared_support.json"]
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "sources": source_files,
            "files": {n: sha256(out / n) for n in names},
        },
        out / "shared_data_lock.json",
    )
