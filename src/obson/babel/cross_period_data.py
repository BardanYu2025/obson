"""Pinned packed-feature triplets, same-close timestamps and original partition gates."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import capacity_growth_run as gr
from . import cross_period_audit as raw_audit
from . import cross_period_consistency as xp
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

cr = gr.cr
ur = gr.ur
cov = ur.bb.cov


def audit_identity(root, raw_root):
    done = read_json(root / "completion.json")
    meta = read_json(root / "manifest.json")
    if done["status"] != "raw_audit_complete" or meta["code_sha256"] != sha256(raw_audit.__file__):
        raise ValueError("Completed matching raw cross-period audit required")
    verify_files(root, done["report_sha256"])
    inventory = read_json(root / "files.json")
    for row in inventory:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or not row["valid"]:
            raise ValueError("Invalid raw inventory path or file")
        cov.file_check(raw_root / relative, row["sha256"])
    return {
        "manifest": meta,
        "completion_sha256": sha256(root / "completion.json"),
        "files": done["report_sha256"],
        "raw_inventory": read_json(root / "alignment_metrics.json")["raw_inventory_sha256"],
    }


def source_context(meta):
    gm = meta["identity"]["manifest"]
    cm = gr.parent_meta(gm)
    alignment = cr.alignment(cm)
    sampling = read_json(Path(alignment["sampling_source"]) / "manifest.json")
    return gm, cm, alignment, sampling


def load_raw(meta, cross=False):
    from . import data

    gm, cm, alignment, sampling = source_context(meta)
    root = Path(meta["raw_root"])
    bounds = sampling["coverage_identity"]["boundaries"]
    if not cross:
        ref = read_json(Path(sampling["coverage_identity"]["sources"]["long"]) / "manifest.json")[
            "manifest"
        ]
        keys = [r["key"].split("/") for r in ref["sources"]]
        series, _ = data.load_series(
            root, sorted({r[0] for r in keys}), sorted({int(r[1]) for r in keys})
        )
        if data.manifest(series, bounds) != ref:
            raise ValueError("Training raw lineage changed")
    else:
        am = read_json(Path(alignment["original_source"]) / "manifest.json")
        audit = read_json(Path(am["cross_run"]) / "raw_audit.json")
        series, _ = data.load_series(root, audit["eligible_symbols"], (15, 30, 60))
        expected = {
            (r["symbol"], r["period"], r["contract"]): r
            for r in audit["files"]
            if r["symbol"] in audit["eligible_symbols"]
        }
        if set(expected) != {(s.code, s.period, s.contract) for s in series}:
            raise ValueError("Cross raw keys changed")
        for s in series:
            row = expected[s.code, s.period, s.contract]
            cov.file_check(root / s.code / f"{s.contract}_{s.period}m.csv", row["sha256"])
            if s.source_hash != row["source_hash"]:
                raise ValueError("Cross normalized raw changed")
    return series, bounds


def bank_info(meta, split):
    _, cm, _, _ = source_context(meta)
    folder = Path(cm["packed"][split]["directory"])
    stem = "test" if split == "cross_research" else split
    return folder / f"{stem}_x.npy", folder / f"{stem}_sequences.json"


def evaluation_anchors(series, bounds, split, specs):
    """Fixed stride16 extra cohort for NEW cross-period metrics only, no score filter."""
    packed = {series[s["series"]].key: s for s in specs}
    result = []
    for s in series:
        if s.period not in (15, 30) or s.key not in packed:
            continue
        spec = packed[s.key]
        for e in cov.endpoints(s, bounds, "test" if split == "cross_research" else split, 16, 512):
            if e < spec["lo"] + 511 or e >= spec["lo"] + spec["length"]:
                continue
            dt = s.frame.datetime.iloc[e]
            result.append(
                {
                    "key": s.key,
                    "symbol": s.code,
                    "period": s.period,
                    "row": int(e),
                    "end": str(dt),
                    "month": str(dt.to_period("M")),
                }
            )
    return result


def plan_pairs(series, bounds, split, specs, originals, anchors, train=False):
    """Anchors are previously pinned windows; counterpart need not lie on stride16 grid."""
    from .representation import time_mask

    by_key = {s.key: s for s in series}
    ends = {s.key: s.ends for s in series}
    cov.unpack_specs(
        specs, series, bounds, "test" if split == "cross_research" else split, len(originals)
    )
    packed = {}
    for spec in specs:
        key = series[spec["series"]].key
        for end, index in spec["endpoints"]:
            row = originals[index]
            if row["key"] != key or row["row"] != spec["lo"] + end:
                raise ValueError("Packed/raw endpoint identity changed")
        packed[key] = spec
    gates = {}
    maps = {}
    result = []
    excluded = Counter()
    for s in series:
        same = time_mask(s, bounds, "test" if split == "cross_research" else split)
        bad = np.r_[0, np.cumsum(~same)]
        eligible = np.zeros(len(s.frame), bool)
        ids = np.arange(511, len(s.frame))
        eligible[ids] = s.main[ids] & (bad[ids + 1] == bad[ids - 511])
        gates[s.key] = eligible
    for index, anchor in enumerate(anchors):
        fine = by_key[anchor["key"]]
        fend = anchor["row"]
        if str(fine.frame.datetime.iloc[fend]) != anchor["end"]:
            raise ValueError("Anchor timestamp changed")
        choices = (
            (30, 60) if fine.period == 15 else (60,) if not train and fine.period == 30 else ()
        )
        for period in choices:
            key = f"{fine.code}/{period}/{fine.contract}"
            coarse = by_key.get(key)
            if coarse is None or key not in packed:
                excluded["no_coarse_packed_partition"] += 1
                continue
            stamp = ends[fine.key][fend]
            cend = int(np.searchsorted(ends[key], stamp))
            if cend >= len(coarse.frame) or ends[key][cend] != stamp:
                excluded["no_simultaneous_close"] += 1
                continue
            if (
                fine.sessions[fend] != coarse.sessions[cend]
                or not gates[fine.key][fend]
                or not gates[key][cend]
            ):
                excluded["main_contract_session_or_512_partition_warmup"] += 1
                continue
            fs, cs = packed[fine.key], packed[key]
            # Earlier fine A supports shifts1/16 inside the same original partition.
            if (
                fend - (16 if train else 0) - fs["lo"] < 511
                or cend - cs["lo"] < 511
                or cend >= cs["lo"] + cs["length"]
            ):
                excluded["packed_context_boundary"] += 1
                continue
            mapping_key = (fine.key, key)
            if mapping_key not in maps:
                maps[mapping_key] = xp.strict_map(fine.frame, coarse.frame, fine.period, period)
            last, valid = maps[mapping_key]
            if not valid[cend] or last[cend] != fend:
                excluded["current_interval_not_strict"] += 1
                continue
            common = xp.links(fend, cend, last, valid, coarse.frame.datetime.to_numpy(), period)
            if len(common) < xp.MIN_LINKS:
                excluded["insufficient_common_changes"] += 1
                continue
            result.append(
                dict(
                    anchor_index=index,
                    fine_bank_end=fs["offset"] + fend - fs["lo"],
                    coarse_bank_end=cs["offset"] + cend - cs["lo"],
                    coarse_row=cend,
                    coarse_key=key,
                    pair=f"{fine.period}->{period}",
                    links=common,
                    **(
                        anchor | {"week": str(pd.Timestamp(fine.sessions[fend]).to_period("W-SUN"))}
                    ),
                )
            )
    return result, dict(excluded)


def verify_data(meta, out):
    lock = read_json(out / "cross_data_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Cross data configuration changed")
    verify_files(out, lock["files"])
    return lock


def prepare(meta, out):
    if (out / "cross_data_lock.json").exists():
        verify_data(meta, out)
        return
    gm, cm, alignment, _ = source_context(meta)
    series, bounds = load_raw(meta)
    for split in ("train", "test", "cross_research"):
        if split == "cross_research":
            series, bounds = load_raw(meta, True)
        bank_path, spec_path = bank_info(meta, split)
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        specs = read_json(spec_path)
        originals = read_json(Path(meta["source"]) / f"{split}_inventory.json")
        if split == "train":
            pool = read_json(Path(alignment["sampling_source"]) / "candidates/inventory.json")
            base_plan = read_json(Path(meta["source"]) / "train_plan.json")["rows"]
            anchors = [pool[p["index"]] | {"base_index": i} for i, p in enumerate(base_plan)]
        else:
            anchors = evaluation_anchors(series, bounds, split, specs)
        rows, excluded = plan_pairs(
            series, bounds, split, specs, originals, anchors, split == "train"
        )
        if not rows or split == "train" and len(rows) < 128:
            raise ValueError(f"Insufficient paired support: {split}/{len(rows)}")
        if split != "train":
            for pair in ("15->30", "15->60", "30->60"):
                group = [r for r in rows if r["pair"] == pair]
                if len(group) < 50 or len({r["week"] for r in group}) < 5:
                    atomic_json(
                        {
                            "pair": pair,
                            "windows": len(group),
                            "weeks": len({r["week"] for r in group}),
                            "excluded": excluded,
                        },
                        out / f"{split}_insufficient_support.json",
                    )
                    raise ValueError(
                        f"Cross evaluation requires50 pairs/5 weeks before training: {split}/{pair}"
                    )
        stats = gr.statistics(gm)[0]
        maxdiff = 0.0
        for start in range(0, len(rows), 256):
            block = rows[start : start + 256]
            views = []
            for key in ("fine_bank_end", "coarse_bank_end"):
                raw = np.stack([bank[r[key] - 127 : r[key] + 1] for r in block])
                x, y, mask = cr.odr.normalized(raw, stats)
                views.append(y)
                if not mask[:, :127, 0].all():
                    raise ValueError("Missing common price target")
            for r, a, b in zip(block, *views, strict=False):
                ix = np.asarray(r["links"])
                da = (a[ix[:, 1], 0] - a[ix[:, 0], 0]) * stats["y_scale"][0]
                db = (b[ix[:, 3], 0] - b[ix[:, 2], 0]) * stats["y_scale"][0]
                diff = float(abs(da - db).max())
                maxdiff = max(maxdiff, diff)
                if not np.allclose(da, db, atol=2e-5, rtol=2e-5):
                    raise ValueError("Encoded common targets disagree with raw alignment")
        summary = {
            "rows": rows,
            "eligible": len(rows),
            "excluded": excluded,
            "pairs": dict(Counter(r["pair"] for r in rows)),
            "common_change_count": sum(len(r["links"]) for r in rows),
            "target_replay_max_abs_log_percent": maxdiff,
        }
        atomic_json(summary, out / f"{split}_cross_plan.json")
        progress(
            f"Cross-period {split}: {len(rows)} pairs; groups={summary['pairs']}; exclusions={excluded}; target maxdiff={maxdiff:.3g}"
        )
    names = [f"{s}_cross_plan.json" for s in ("train", "test", "cross_research")]
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {n: sha256(out / n) for n in names},
        },
        out / "cross_data_lock.json",
    )


class Builder:
    def __init__(self, meta, out):
        gm, cm, _, _ = source_context(meta)
        self.plan = read_json(out / "train_cross_plan.json")["rows"]
        self.original = cr.Builder(cm, gr.parent_root(gm))
        self.stats = gr.statistics(gm)[0]
        self.bank = np.load(bank_info(meta, "train")[0], mmap_mode="r", allow_pickle=False)

    def __call__(self, ids, shifts):
        rows = [self.plan[i] for i in ids]
        a, b = self.original(np.array([r["base_index"] for r in rows]), shifts)
        raw = np.stack(
            [self.bank[r["coarse_bank_end"] - 127 : r["coarse_bank_end"] + 1] for r in rows]
        )
        c = dict(zip(("x", "y", "mask"), cr.odr.normalized(raw, self.stats), strict=False))
        ix, mask = xp.padded_links(rows)
        return a, b, c, ix, mask


def evaluation_data(meta, out, split):
    rows = read_json(out / f"{split}_cross_plan.json")["rows"]
    stats = gr.statistics(meta["identity"]["manifest"])[0]
    bank = np.load(bank_info(meta, split)[0], mmap_mode="r", allow_pickle=False)
    views = []
    for key in ("fine_bank_end", "coarse_bank_end"):
        raw = np.stack([bank[r[key] - 127 : r[key] + 1] for r in rows])
        views.append(dict(zip(("x", "y", "mask"), cr.odr.normalized(raw, stats), strict=False)))
    return rows, views, *xp.padded_links(rows)
