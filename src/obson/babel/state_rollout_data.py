"""Immutable lineage and deduplicated, independently re-encoded rolling128 states."""

import gc
import time
from collections import Counter

import numpy as np
import torch

from . import activity_ablation as aa
from . import ae_context
from . import endpoint_readout as er
from . import linear_history_run as parent
from . import representation as rp
from . import state_rollout as core
from .ae_extend import atomic_json
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

SPLITS = ("train", "val", "test", "cross_research")


def encode(frame, period):
    return np.column_stack(
        (
            ae_context.encode_context(frame, period, "ema8_32")["x"],
            aa.activity_features(frame, period)[0],
        )
    )


def cache(out, split):
    folder = out / "cache"
    index = read_json(folder / f"{split}_index.json")
    if index["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Cache manifest changed")
    verify_files(folder, index["files"])
    with np.load(folder / f"{split}.npz", allow_pickle=False) as f:
        bank = dict(f)
    return bank, read_json(folder / f"{split}_rows.json")


def commit_cache(out, split, bank, rows, audit, endpoints):
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    np.savez(folder / f"{split}.tmp.npz", **bank)
    (folder / f"{split}.tmp.npz").replace(folder / f"{split}.npz")
    for suffix, obj in (("rows", rows), ("audit", audit), ("endpoints", endpoints)):
        atomic_json(obj, folder / f"{split}_{suffix}.json")
    names = [f"{split}.npz"] + [f"{split}_{v}.json" for v in ("rows", "audit", "endpoints")]
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {n: sha256(folder / n) for n in names},
        },
        folder / f"{split}_index.json",
    )


def rolling_inputs(features, ends, statistics):
    # Only a batch is expanded, never origin x future x128 full input caches.
    return np.stack(
        [
            ((features[e - 127 : e + 1] - statistics["x_mean"]) / statistics["x_scale"]).astype(
                "float32"
            )
            for e in ends
        ]
    )


@torch.inference_mode()
def prepare(meta, out, splits, cross, device):
    if any(s not in SPLITS or (s == "cross_research") != cross for s in splits):
        raise ValueError("Invalid raw partition")
    if any(s in SPLITS[2:] for s in splits):
        from .state_rollout_run import check_selection

        check_selection(meta, out)
    pending = []
    for split in splits:
        if (out / f"cache/{split}_index.json").exists():
            cache(out, split)
        else:
            pending.append(split)
    if not pending:
        return
    tm = parent.tm(meta)
    series, bounds = parent.xr.xd.load_raw(tm, cross=cross)
    lookup = {s.key: s for s in series}
    statistics = parent.stats(meta)[0]
    for split in pending:
        started = time.monotonic()
        progress(f"{split}: replaying original inputs and checking128+32 partition containment")
        original = read_json(parent.root(meta) / f"{split}_inventory.json")
        x = parent.data(meta, split)["x"]
        if x.shape != (len(original), 128, 28):
            raise ValueError("Original inputs differ")
        grouped = {}
        for i, r in enumerate(original):
            grouped.setdefault(r["key"], []).append((i, r))
        rows, selected, ys, pasts, gaps, excluded = [], [], [], [], [], []
        requests, feature_bank, maxdiff = {}, {}, 0.0
        horizon = core.TRAIN_H if split == "train" else core.EVAL_H
        for key, items in grouped.items():
            s = lookup[key]
            features = encode(s.frame, s.period)
            same = rp.time_mask(s, bounds, "test" if cross else split)
            close = s.frame.close.to_numpy(float)
            valid = []
            for i, r in items:
                e = int(r["row"])
                if e < 127 or e >= len(close) or str(s.frame.datetime.iloc[e]) != r["end"]:
                    raise ValueError("Raw origin identity differs")
                observed = rolling_inputs(features, [e], statistics)[0]
                diff = float(np.max(np.abs(observed - x[i])))
                maxdiff = max(maxdiff, diff)
                if not np.allclose(observed, x[i], atol=1e-6, rtol=2e-5):
                    raise ValueError(f"Original feature replay differs: {split}/{key}/{e}/{diff}")
                y, reason = core.labels(close, e, same)
                if reason:
                    excluded.append({"index": i, "key": key, "row": e, "reason": reason})
                    continue
                selected.append(i)
                rows.append(
                    dict(
                        r,
                        original_index=i,
                        input_start=str(s.frame.datetime.iloc[e - 127]),
                        target_end=str(s.frame.datetime.iloc[e + 32]),
                        target_session_end=str(s.sessions[e + 32]),
                    )
                )
                ys.append(y)
                pasts.append(core.past(close, e))
                minutes = (
                    np.diff(s.frame.datetime.iloc[e : e + 33].to_numpy())
                    .astype("timedelta64[s]")
                    .astype(float)
                    / 60
                )
                gaps.append(minutes > s.period * 1.01)
                valid.append(e)
            if valid:
                e = valid[0]
                if not np.array_equal(encode(s.frame.iloc[: e + 1], s.period), features[: e + 1]):
                    raise ValueError("Feature causality prefix replay differs")
                feature_bank[key] = features
                requests[key] = sorted({e + h for e in valid for h in range(horizon + 1)})
        if len(rows) < 100:
            raise ValueError(f"Insufficient eligible {split} origins: {len(rows)}")
        order = np.argsort(selected)
        rows = [rows[i] for i in order]
        endpoints = [{"key": k, "row": e} for k in sorted(requests) for e in requests[k]]
        ids = {(r["key"], r["row"]): i for i, r in enumerate(endpoints)}
        mapping = np.array(
            [[ids[r["key"], r["row"] + h] for h in range(horizon + 1)] for r in rows]
        )
        bank = {
            "x": np.asarray(x[np.asarray(selected)[order]], "float32"),
            "targets": np.asarray(ys)[order],
            "past": np.asarray(pasts)[order],
            "gaps": np.asarray(gaps)[order],
            "period": np.array([r["period"] for r in rows]),
            "mapping": mapping,
            "close": np.array([lookup[r["key"]].frame.close.iloc[r["row"]] for r in rows]),
            "history_close": np.array(
                [
                    lookup[r["key"]].frame.close.iloc[r["row"] - 127 : r["row"] + 1].to_numpy()
                    for r in rows
                ]
            ),
        }
        audits = {}
        for seed in (42, 43):
            model, query = parent.construct(meta, seed, device)
            model.eval().requires_grad_(False)
            query.eval().requires_grad_(False)
            before = parent.ur.bb.state_signature(model)
            causal = er.trained_causality(model, torch.as_tensor(bank["x"][:2], device=device))
            if causal["status"] == "failed":
                raise ValueError("Encoder causality failed")
            result = np.empty((len(endpoints), 768), "float32")
            batch = meta["extraction_batch"]
            count = 0
            for key in sorted(requests):
                ends = requests[key]
                for left in range(0, len(ends), batch):
                    es = ends[left : left + batch]
                    xx = rolling_inputs(feature_bank[key], es, statistics)
                    z = model.core.encoder(torch.as_tensor(xx, device=device))[:, -1]
                    result[[ids[key, e] for e in es]] = z.cpu().numpy()
                    count += len(es)
                    if count == len(endpoints) or count % (batch * 128) < batch:
                        elapsed = time.monotonic() - started
                        progress(
                            f"{split}/s{seed}: {count}/{len(endpoints)} unique rolling endpoints; cache stage {elapsed / 60:.1f}min"
                        )
            # Reconstruct fixed states one-by-one to detect batch/position shortcuts.
            sample = sorted({0, len(rows) // 2, len(rows) - 1})
            single = np.stack(
                [
                    model.core.encoder(torch.as_tensor(bank["x"][i : i + 1], device=device))[:, -1]
                    .cpu()
                    .numpy()[0]
                    for i in sample
                ]
            )
            expected = result[mapping[sample, 0]]
            diff = float(np.max(np.abs(single - expected)))
            if not np.allclose(single, expected, atol=1e-4, rtol=2e-4):
                atomic_json(
                    {"split": split, "seed": seed, "max_abs": diff}, out / "numeric_failure.json"
                )
                raise ValueError("Batch/rolling origin state replay differs")
            if before != parent.ur.bb.state_signature(model):
                raise ValueError("Frozen encoder changed")
            bank[f"z{seed}"] = result
            audits[str(seed)] = {
                "signature": before,
                "unchanged": True,
                "causality": causal,
                "single_batch_max_abs": diff,
            }
            del model, query
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        if not all(np.isfinite(v).all() for v in bank.values()):
            raise ValueError("Nonfinite cache")
        audit = {
            "input_rows": len(original),
            "kept": len(rows),
            "excluded": excluded,
            "exclusions": dict(Counter(r["reason"] for r in excluded)),
            "bounds": bounds,
            "input_replay_max_abs": maxdiff,
            "encoders": audits,
            "unique_endpoints": len(endpoints),
            "expanded_inputs_saved": False,
            "future_enters_prediction": False,
            "sample_ids": sample,
            "seconds": time.monotonic() - started,
        }
        commit_cache(out, split, bank, rows, audit, endpoints)
        progress(f"{split} cache complete: {len(rows)} origins/{len(endpoints)} unique states")
        del feature_bank, bank
        gc.collect()


def transformations(out):
    if (out / "transforms_lock.json").exists():
        lock = read_json(out / "transforms_lock.json")
        if lock["manifest_sha256"] != sha256(out / "manifest.json"):
            raise ValueError("Training transforms binding changed")
        verify_files(out, lock["files"])
        return read_json(out / "transforms.json")
    bank, _ = cache(out, "train")
    pairs = np.stack((bank["mapping"][:, :-1], bank["mapping"][:, 1:]), -1)
    raw = bank["x"].reshape(len(bank["x"]), -1).astype(float)
    fit = {
        "price": core.price_fit(bank),
        "raw": {"mean": raw.mean(0).tolist(), "scale": raw.std(0).clip(1e-6).tolist()},
        "states": {str(s): core.state_scaler(bank[f"z{s}"], pairs) for s in (42, 43)},
        "scope": "Eligible TRAIN only. Unique endpoint state moments/unique adjacent pairs; raw moments per origin. Simple mean-price baseline uses32 training labels; learned arms use8.",
    }
    atomic_json(fit, out / "transforms.json")
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {
                "transforms.json": sha256(out / "transforms.json"),
                "cache/train_index.json": sha256(out / "cache/train_index.json"),
            },
        },
        out / "transforms_lock.json",
    )
    return fit
