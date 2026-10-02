"""Original endpoints, additional observed burn-in, unchanged causal raw features."""

import numpy as np

from . import linear_history_run as parent
from . import representation as rp
from . import state_rollout_data as original
from .ae_extend import atomic_json
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

SPLITS = ("train", "val", "test", "cross_research")


def cache(out, split):
    folder = out / "cache"
    index = read_json(folder / f"{split}_index.json")
    if index["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Input cache manifest changed")
    verify_files(folder, index["files"])
    with np.load(folder / f"{split}.npz", allow_pickle=False) as f:
        x = f["x"]
    return x, read_json(folder / f"{split}_rows.json")


def eligible(end, same):
    return end >= 254 and end < len(same) and bool(np.all(same[end - 254 : end + 1]))


def prepare(meta, out, splits, cross=False):
    if any(s not in SPLITS or (s == "cross_research") != cross for s in splits):
        raise ValueError("Invalid split/partition")
    if any(s in SPLITS[2:] for s in splits):
        from .uniform_context_evaluate import check_readouts

        check_readouts(out)
    pending = []
    for split in splits:
        if (out / f"cache/{split}_index.json").exists():
            cache(out, split)
        else:
            pending.append(split)
    if not pending:
        return
    source = parent.tm(meta)
    series, bounds = parent.xr.xd.load_raw(source, cross=cross)
    lookup = {s.key: s for s in series}
    stats = parent.stats(meta)[0]
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    for split in pending:
        rows = read_json(parent.root(meta) / f"{split}_inventory.json")
        previous = parent.data(meta, split)["x"]
        if previous.shape != (len(rows), 128, 28):
            raise ValueError("Original inventory/input dimensions differ")
        grouped = {}
        for i, r in enumerate(rows):
            grouped.setdefault(r["key"], []).append((i, r))
        blocks, kept, excluded, diffs = [], [], [], []
        for key, items in grouped.items():
            s = lookup[key]
            raw = original.encode(s.frame, s.period)
            same = rp.time_mask(s, bounds, "test" if cross else split)
            for i, r in items:
                e = int(r["row"])
                if e >= len(raw) or e < 127 or str(s.frame.datetime.iloc[e]) != r["end"]:
                    raise ValueError("Original endpoint identity differs")
                replay = original.rolling_inputs(raw, [e], stats)[0]
                diff = float(np.abs(replay - previous[i]).max())
                diffs.append(diff)
                if not np.allclose(replay, previous[i], atol=1e-6, rtol=2e-5):
                    raise ValueError(f"Original input replay differs: {key}/{e}/{diff}")
                if not eligible(e, same):
                    excluded.append(
                        {
                            "index": i,
                            "key": key,
                            "row": e,
                            "reason": "255 features not wholly in contract/partition",
                        }
                    )
                    continue
                block = ((raw[e - 254 : e + 1] - stats["x_mean"]) / stats["x_scale"]).astype(
                    "float32"
                )
                if not np.isfinite(block).all():
                    raise ValueError("Nonfinite full-context inputs")
                blocks.append(block)
                kept.append(
                    dict(r, original_index=i, input_start=str(s.frame.datetime.iloc[e - 254]))
                )
            e = int(items[0][1]["row"])
            if not np.array_equal(original.encode(s.frame.iloc[: e + 1], s.period), raw[: e + 1]):
                raise ValueError("Future truncation changed observed input features")
        if len(kept) < 100:
            raise ValueError(f"Insufficient eligible {split} endpoints: {len(kept)}")
        order = np.argsort([r["original_index"] for r in kept])
        x = np.asarray(blocks)[order]
        kept = [kept[i] for i in order]
        np.savez(folder / f"{split}.tmp.npz", x=x)
        (folder / f"{split}.tmp.npz").replace(folder / f"{split}.npz")
        atomic_json(kept, folder / f"{split}_rows.json")
        atomic_json(
            {
                "original": len(rows),
                "bounds": {k: str(v) for k, v in bounds.items()},
                "eligible": len(kept),
                "excluded": excluded,
                "replay_max_abs": max(diffs),
                "scope": "255 causal feature rows within original partition; EMA may summarize earlier observed history",
            },
            folder / f"{split}_audit.json",
        )
        names = [f"{split}.npz", f"{split}_rows.json", f"{split}_audit.json"]
        atomic_json(
            {
                "manifest_sha256": sha256(out / "manifest.json"),
                "files": {n: sha256(folder / n) for n in names},
            },
            folder / f"{split}_index.json",
        )
        progress(
            f"{split}: {len(kept)}/{len(rows)} original endpoints eligible for full-context study"
        )
