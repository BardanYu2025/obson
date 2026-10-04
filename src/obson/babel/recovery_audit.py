"""R0 only: immutable replay and gradient diagnostics, never optimizer updates.

A completed audit is evidence for review, not permission to start R1/R2. Existing
training modules remain byte-identical so their historical bindings stay valid.
"""

import argparse
import gc
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import architecture as ar
from . import context_transfer_run as previous
from . import history_query as hq
from . import representation as rp
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-recovery-r0-v1"
# Pinned to the already reviewed user-supplied Context report, not a mutable latest run.
SOURCE_MANIFEST = "fb108cf33be08126a6c761800be93e9efbdf4e8bdaa3c0254b0974ecfc2a00a2"
SOURCE_COMPLETION = "136e0dce63581038bd0bd152bf755c16c9e36c82a948d34d5d44a0b1a17857b5"
# Fixed before any new real-weight replay. A failure is retained, never relaxed.
ATOL, RTOL = 1e-6, 2e-5
TARGET_ATOL = 1e-5  # Prior independent raw-vs-float32 target pipeline contract.
STATE_ATOL, STATE_RTOL = 1e-4, 2e-4  # Existing Context preflight contract.


def compare(actual, expected, atol=ATOL, rtol=RTOL):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return {"passed": False, "actual_shape": list(a.shape), "expected_shape": list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return {
        "passed": finite and bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs": float(np.max(np.abs(a - b))) if finite and a.size else None,
        "atol": atol,
        "rtol": rtol,
        "finite": finite,
    }


def nested_compare(actual, expected, path=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return {path or "fields": {"passed": False, "reason": "field identities differ"}}
        result = {}
        for key in expected:
            result.update(nested_compare(actual[key], expected[key], path + "/" + key))
        return result
    return {path: compare(actual, expected)}


def require(checks, label):
    if not checks or not all(c["passed"] for c in checks.values()):
        failed = [k for k, c in checks.items() if not c["passed"]]
        raise ValueError(f"{label}: mismatched {failed}; numeric evidence saved before stopping")


def save_checks(out, name, checks, **extra):
    atomic_json({"checks": checks, **extra}, out / (name + ".json"))
    require(checks, name)


def safe_output(source, out):
    source, out = source.resolve(), out.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Audit output must be separate from source, never inside/above it")
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Nonempty audit output: choose a new directory; evidence is never overwritten"
        )
    return source, out


def required_file(root, rel, digest):
    root = Path(root).resolve()
    path = (root / rel).resolve()
    if root not in path.parents:
        raise ValueError(f"Invalid bound relative path: {rel}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing bound asset (not proof retraining needed): {path}")
    actual = sha256(path)
    if actual != digest:
        raise ValueError(f"Immutable file mismatch: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def identities(source, out):
    if (
        sha256(source / "manifest.json") != SOURCE_MANIFEST
        or sha256(source / "completion.json") != SOURCE_COMPLETION
    ):
        raise ValueError(
            "Source differs from the reviewed Context report; do not silently change the audit baseline"
        )
    meta, done = read_json(source / "manifest.json"), read_json(source / "completion.json")
    if (
        meta["schema"] != previous.SCHEMA
        or done["status"] != "complete"
        or not done["source_unchanged"]
    ):
        raise ValueError("Completed unchanged Context Transfer required")
    if meta["code_sha256"] != previous.code_identity():
        raise ValueError("Bound historical implementation changed")
    files = {}
    # Validate all archived report bindings, all four input caches and six replay weights.
    for rel, digest in done["files"].items():
        if Path(rel).suffix in (".json", ".jsonl") or rel.startswith("cache/"):
            files[rel] = required_file(source, rel, digest)
    for seed in (42, 43):
        rel = f"rolling_s{seed}/update_last.pt"
        files[rel] = required_file(source, rel, done["files"][rel])
    macro = Path(meta["task_source"]["source"])
    identity = previous.parent.source_identity(macro)
    if identity != meta["task_source"]:
        raise ValueError("Macro selected weights/source ancestry changed")
    cached = previous.data.source_identity(Path(meta["cache_source"]["source"]), identity)
    if cached != meta["cache_source"]:
        raise ValueError("Uniform cache identity changed")
    mm = read_json(macro / "manifest.json")
    shared = Path(mm["task_source"]["source"])
    sm = read_json(shared / "manifest.json")
    edges = []
    for consumer, ancestor, edge in (
        (source, macro, meta["task_source"]),
        (macro, shared, mm["task_source"]),
    ):
        for seed in (42, 43):
            item = edge["chosen"][str(seed)]
            asset = required_file(ancestor, item["path"], item["sha256"])
            edges.append(dict(consumer=str(consumer), seed=seed, role="loaded_weight", **asset))
    stats, _ = previous.parent.stats(meta)
    if stats != read_json(source / "statistics.json") or stats.get("fitted_on") != "train_only":
        raise ValueError("Inherited train-only feature/target scaling binding changed")
    for key, length in (
        ("x_mean", 28),
        ("x_scale", 28),
        ("y_mean", 7),
        ("y_scale", 7),
        ("delta_scale", 4),
    ):
        a = np.asarray(stats[key])
        if a.shape != (length,) or not np.isfinite(a).all() or ("scale" in key and (a <= 0).any()):
            raise ValueError(f"Invalid frozen scaling: {key}")
    rates = {}
    for seed in (42, 43):
        chosen = meta["task_source"]["chosen"][str(seed)]
        history = read_json(macro / f"control_s{seed}/history.json")
        row = history[chosen["epoch"] - 1]
        expected = previous.parent.base.rates(mm, chosen["epoch"])
        if row["epoch"] != chosen["epoch"] or any(row[k] != v for k, v in expected.items()):
            raise ValueError("Macro chosen-epoch learning-rate evidence differs")
        rates[str(seed)] = {
            "source_epoch": chosen["epoch"],
            "source_rates": expected,
            "context_to_source_encoder_ratio": meta["encoder_lr"] / expected["encoder_lr"],
            "context_to_source_query_ratio": meta["query_lr"] / expected["query_lr"],
        }
    report = {
        "learning_rate_bridge": rates,
        "source": str(source),
        "manifest_sha256": sha256(source / "manifest.json"),
        "completion_sha256": sha256(source / "completion.json"),
        "bound_files": files,
        "selected_weight_edges": edges,
        "cache_edge": {"role": "input_cache_not_neural_weights", "source": cached["source"]},
        "ancestral_verification": "Existing recursive source identities, selected hashes and code bindings verified; not independent full historical training replication",
        "scaling_scope": "Inherited train_only statistics and bound fit implementation; no refit and no independent reproduction of original scaler fitting in R0",
    }
    atomic_json(report, out / "identity.json")
    return meta, mm, sm, stats


def schedule_audit(source, meta, n, out):
    result = {}
    for seed in (42, 43):
        rows = read_json(source / f"prefix_s{seed}/history.json")
        if len(rows) < 5:
            raise ValueError("Five high-LR reference epochs required")
        records = []
        for epoch, row in enumerate(rows[:5], 1):
            order, ps = previous.core.schedule(n, seed, epoch)
            expected = {
                "order_hash": previous.parent.ur.bb.cov.ndarray_hash(order),
                "prefix_hash": previous.parent.ur.bb.cov.ndarray_hash(ps),
                "steps": math.ceil(n / meta["batch"]),
                "examples": n,
                "supervised_states": 5 * n,
            }
            if row["epoch"] != epoch or any(row["train"][k] != v for k, v in expected.items()):
                raise ValueError(f"High-LR reference schedule differs: s{seed}/{epoch}")
            if row["encoder_lr"] != meta["encoder_lr"] or row["query_lr"] != meta["query_lr"]:
                raise ValueError("High-LR reference rates differ")
            records.append(dict(epoch=epoch, **expected))
        result[str(seed)] = records
    atomic_json(result, out / "reference_schedule.json")


def target_contract(x, stats):
    ps = torch.tensor([32, 64, 96, 128]).expand(len(x), -1)
    b = torch.as_tensor(x)
    full, fm = previous.core.original.targets(b, ps, stats, False)
    native, nm = previous.core.original.targets(b, ps, stats, True)
    ages = torch.arange(1, 128)
    expected_mask = fm & (ages[None, None] < ps[..., None])[..., None]
    # This is the precise B-arm common mask; it must not be sent to the decoder.
    checks = {
        "A_B_mask": {"passed": bool(torch.equal(nm, expected_mask))},
        "A_B_common_labels": compare(native.numpy(), torch.where(nm, full, 0).numpy(), 0, 0),
    }
    band_support = []
    for mask in (nm, fm):
        counts = torch.stack([mask[..., lo - 1 : hi, 0].any(-1) for lo, hi in hq.BANDS], -1)
        band_support.append((counts / counts.sum(-1, keepdim=True)).double().mean(0).tolist())
    return checks, {
        "prefixes": [32, 64, 96, 128],
        "native_band_weights": band_support[0],
        "full_band_weights": band_support[1],
    }


def raw_replay(source, meta, stats, out):
    """Rebuild every eligible255 block and check excluded boundary decisions."""
    from . import uniform_context_data as ud

    summaries = {}
    for cross, splits in ((False, ("train", "val", "test")), (True, ("cross_research",))):
        series, bounds = previous.parent.xr.xd.load_raw(previous.parent.tm(meta), cross=cross)
        lookup = {s.key: s for s in series}
        for split in splits:
            x, rows = previous.data.cache(source, split)
            audit = read_json(source / f"cache/{split}_audit.json")
            if audit["bounds"] != {k: str(v) for k, v in bounds.items()}:
                raise ValueError(f"Partition boundaries changed: {split}")
            originals = read_json(previous.parent.root(meta) / f"{split}_inventory.json")
            grouped = {}
            for i, r in enumerate(originals):
                grouped.setdefault(r["key"], []).append((i, r))
            positions = {r["original_index"]: j for j, r in enumerate(rows)}
            if len(positions) != len(rows) or x.shape != (len(rows), 255, 28):
                raise ValueError("Duplicated row identity or invalid cache dimensions")
            old = previous.parent.data(meta, split)
            kept, excluded, maxdiff, target_diff, contracts = [], [], 0.0, 0.0, 0
            checks = {}
            for key, members in grouped.items():
                s = lookup[key]
                raw = ud.original.encode(s.frame, s.period)
                same = rp.time_mask(s, bounds, "test" if cross else split)
                cut = int(members[0][1]["row"])
                checks[f"{key}/raw_causality"] = {
                    "passed": bool(
                        np.array_equal(
                            ud.original.encode(s.frame.iloc[: cut + 1], s.period), raw[: cut + 1]
                        )
                    )
                }
                for original_index, row in members:
                    e = int(row["row"])
                    if e >= len(raw) or e < 127 or str(s.frame.datetime.iloc[e]) != row["end"]:
                        raise ValueError(f"Raw endpoint mismatch: {split}/{key}/{e}")
                    if not ud.eligible(e, same):
                        excluded.append(original_index)
                        continue
                    kept.append(original_index)
                    if original_index not in positions:
                        raise ValueError(
                            f"Eligible original missing from cache: {split}/{original_index}"
                        )
                    j = positions[original_index]
                    if rows[j] != dict(
                        row,
                        original_index=original_index,
                        input_start=str(s.frame.datetime.iloc[e - 254]),
                    ):
                        raise ValueError("Cached row identity changed")
                    rebuilt = ((raw[e - 254 : e + 1] - stats["x_mean"]) / stats["x_scale"]).astype(
                        "float32"
                    )
                    ck = compare(rebuilt, x[j])
                    maxdiff = max(maxdiff, ck["max_abs"] or 0.0)
                    if not ck["passed"]:
                        save_checks(out, "raw_failure", {f"{split}/{key}/{e}": ck})
                    # Original labels independently generated before normalization;
                    # compare both canonical old cache and rebuilt128 pipeline.
                    rb = raw[None, e - 127 : e + 1]
                    yy, mm = ar.ordered_targets(rb)
                    xx, yy, mm = ar.normalize(rb, yy, mm, stats)
                    ps = torch.tensor([[32, 64, 96, 128]])
                    canonical, cm = hq.targets(
                        {"x": torch.tensor(xx), "y": torch.tensor(yy), "mask": torch.tensor(mm)},
                        ps,
                        stats,
                    )
                    actual, am = previous.core.original.targets(
                        torch.tensor(rebuilt[None]), ps, stats, True
                    )
                    tc = compare(actual.numpy(), canonical.numpy(), TARGET_ATOL, RTOL)
                    target_diff = max(target_diff, tc["max_abs"] or 0.0)
                    legacy, legacy_mask = hq.targets(
                        {
                            k: torch.tensor(old[k][original_index : original_index + 1])
                            for k in ("x", "y", "mask")
                        },
                        ps,
                        stats,
                    )
                    target_checks = {
                        "original_cached_target": compare(
                            actual.numpy(), legacy.numpy(), TARGET_ATOL, RTOL
                        ),
                        "original_cached_mask": {"passed": bool(torch.equal(am, legacy_mask))},
                        "independent_target": tc,
                        "independent_mask": {"passed": bool(torch.equal(am, cm))},
                        "old128_input": compare(x[j, -128:], old["x"][original_index]),
                    }
                    if not all(c["passed"] for c in target_checks.values()):
                        save_checks(
                            out, "target_failure", target_checks, split=split, key=key, row=e
                        )
                contracts += 1
                if contracts % 25 == 0:
                    progress(f"R0 raw {split}: {contracts}/{len(grouped)} contracts, no updates")
            checks["eligible_rows"] = {
                "passed": sorted(kept) == [r["original_index"] for r in rows]
            }
            checks["excluded_rows"] = {
                "passed": sorted(excluded) == sorted(r["index"] for r in audit["excluded"])
            }
            support = None
            for start in range(0, len(x), 32):
                common, support = target_contract(x[start : start + 32], stats)
                for name, c in common.items():
                    checks[f"{name}/batch{start}"] = c
            save_checks(
                out,
                f"raw_{split}",
                checks,
                windows=len(rows),
                original=len(originals),
                max_input_difference=maxdiff,
                max_target_difference=target_diff,
                support=support,
            )
            summaries[split] = {
                "windows": len(rows),
                "contracts": contracts,
                "excluded": len(excluded),
                "max_input_difference": maxdiff,
                "max_target_difference": target_diff,
            }
            del old, x
        del series, lookup
        gc.collect()
    return summaries


def price_band_losses(pred, target, mask, stats):
    """Exact age-band decomposition, including change1 crossing band boundaries."""
    error = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    p, y = hq.physical(pred, stats), hq.physical(target, stats)
    delta = ((p[..., 1:, 0] - p[..., :-1, 0]) - (y[..., 1:, 0] - y[..., :-1, 0])) / stats[
        "delta_scale"
    ][0]
    de = torch.nn.functional.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
    supports, losses = [], []
    for lo, hi in hq.BANDS:
        a, b = lo - 1, hi
        valid = mask[..., a:b, :]
        support = valid[..., 0].any(-1)
        channel = (error[..., a:b, :] * valid).sum(-2) / valid.sum(-2).clamp_min(1)
        da, db = max(0, a - 1), b - 1
        dm = mask[..., da + 1 : db + 1, 0] & mask[..., da:db, 0]
        change = (de[..., da:db] * dm).sum(-1) / dm.sum(-1).clamp_min(1)
        losses.append((0.4 * channel[..., 0] + 0.2 * change + 0.1 * channel[..., 1]) * support)
        supports.append(support)
    denominator = torch.stack(supports, -1).sum(-1).clamp_min(1)
    return {
        name: (value / denominator).mean()
        for name, value in zip(("price_near", "price_mid", "price_far"), losses, strict=True)
    }


def gradient_stats(encoder, query, x, stats, native):
    """Separate task gradients on a fixed TRAIN batch; no parameter mutation."""
    ps = torch.tensor([[32, 64, 96, 128]], device=x.device).expand(len(x), -1)
    _, pred = previous.core.predict(encoder, query, x, ps, native)
    y, mask = previous.core.original.targets(x, ps, stats, native)
    parts = previous.core.original.loss_rows(pred, y, mask, stats, True)
    losses = {
        "path": 0.4 * parts["path"].mean(),
        "change1": 0.2 * parts["change1"].mean(),
        "body": 0.1 * parts["body"].mean(),
        "activity": 0.3 * parts["activity"].mean(),
        "structure": parts["structure"].mean(),
    }
    base_names = tuple(losses)
    bands = price_band_losses(pred, y, mask, stats)
    expected_price = losses["path"] + losses["change1"] + losses["body"]
    if not torch.allclose(sum(bands.values()), expected_price, atol=1e-7, rtol=1e-6):
        raise ValueError("Age-gradient decomposition changed the training objective")
    losses.update(bands)
    groups = (list(encoder.parameters()), list(query.parameters()))
    params = groups[0] + groups[1]
    result, grads = {}, {}
    for name, loss in losses.items():
        values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        packed = [
            torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).detach().reshape(-1).cpu()
                    for p, g in zip(gs, vs, strict=True)
                ]
            )
            for gs, vs in (
                (groups[0], values[: len(groups[0])]),
                (groups[1], values[len(groups[0]) :]),
            )
        ]
        if not all(torch.isfinite(v).all() for v in packed):
            raise ValueError(f"Nonfinite gradient: {name}")
        grads[name] = packed
        result[name] = {
            "weighted_loss": float(loss.detach()),
            "encoder_norm": float(packed[0].norm()),
            "query_norm": float(packed[1].norm()),
        }
    # CPU dot products avoid retaining all task-gradient vectors on the GPU.
    cosines = {}
    names = list(grads)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ga, gb = grads[a][0], grads[b][0]
            denom = float(ga.norm()) * float(gb.norm())
            cosines[f"{a}/{b}"] = float(torch.dot(ga, gb)) / denom if denom > 0 else None
    total = [sum(grads[k][g] for k in base_names) for g in (0, 1)]
    norm = math.sqrt(sum(float(t.square().sum()) for t in total))
    return {
        "tasks": result,
        "summed_tasks_for_joint_norm": list(base_names),
        "price_band_scope": "Exact decomposition of price objective; additional diagnostics, never double-counted in joint norm",
        "encoder_task_cosines": cosines,
        "total_joint_norm": norm,
        "hypothetical_global_clip_factor": min(1.0, 1.0 / (norm + 1e-6)),
        "scope": "One fixed small train batch, not whole-training gradient attribution",
    }


@torch.no_grad()
def causality(encoder, query, x, out, label):
    ps = torch.tensor([[32, 64, 96, 128]], device=x.device).expand(len(x), -1)
    state, pred = previous.core.predict(encoder, query, x, ps, True)
    checks = {}
    for index, p in enumerate((32, 64, 96)):
        changed = x.clone()
        changed[:, 127 + p :] += 0.125
        z, y = previous.core.predict(encoder, query, changed, ps[:, index : index + 1], True)
        checks[f"prefix{p}/state"] = compare(
            z.cpu(), state[:, index : index + 1].cpu(), STATE_ATOL, STATE_RTOL
        )
        checks[f"prefix{p}/readout"] = compare(
            y.cpu(), pred[:, index : index + 1].cpu(), STATE_ATOL, STATE_RTOL
        )
    full = previous.core.states(encoder, x, ps[:, -1:])
    checks["endpoint_native_full"] = compare(
        full.cpu(), state[:, -1:].cpu(), STATE_ATOL, STATE_RTOL
    )
    # Per-age head independence: querying one age may not read other queries.
    one = query(state[:, -1], torch.tensor([20], device=x.device))
    checks["query20_independent"] = compare(
        one.cpu(), pred[:, -1, 19:20].cpu(), STATE_ATOL, STATE_RTOL
    )
    save_checks(out, label + "_causality", checks)


def checkpoint_report(ck, path):
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": ck["epoch"],
        "keys": sorted(ck),
        "optimizer_in_selected_snapshot": any("optim" in k for k in ck),
        "precise_resume_claim": False,
    }


def neural_replay(source, meta, mm, sm, stats, out, device="cuda"):
    macro = Path(meta["task_source"]["source"])
    shared = Path(mm["task_source"]["source"])
    train, _ = previous.data.cache(source, "train")
    val, _ = previous.data.cache(source, "val")
    schedule_audit(source, meta, len(train), out)
    report = {}
    for seed in (42, 43):
        progress(f"R0 s{seed}: restoring Macro and replaying original validation")
        model, query = previous.parent.construct(meta, seed, device)
        encoder = model.core.encoder
        item = meta["task_source"]["chosen"][str(seed)]
        path = macro / item["path"]
        ck = torch.load(path, map_location="cpu", weights_only=True)
        actual = previous.parent.base.validation(mm, model, query, device)
        save_checks(
            out,
            f"macro_s{seed}_original",
            nested_compare(actual, ck["validation"]),
            actual=actual,
            expected=ck["validation"],
        )
        before = (
            previous.parent.ur.bb.state_signature(encoder),
            previous.parent.ur.bb.state_signature(query),
        )
        recorded = read_json(source / f"prefix_s{seed}/completion.json")
        if before != (recorded["initial_encoder_hash"], recorded["initial_query_hash"]):
            raise ValueError("Old high-LR reference initialization differs")
        with torch.no_grad():
            current = previous.validation(meta, encoder, query, val, device)
        save_checks(
            out,
            f"macro_s{seed}_context",
            nested_compare(current, recorded["initial_validation"]),
            actual=current,
            expected=recorded["initial_validation"],
        )
        encoder.requires_grad_(True)
        query.requires_grad_(True)
        model.local_head.requires_grad_(True)
        b = torch.tensor(train[:2], device=device)
        causality(encoder, query, b, out, f"macro_s{seed}")
        diagnostic = {
            mode: gradient_stats(encoder, query, b, stats, mode == "prefix")
            for mode in ("prefix", "rolling")
        }
        # Exact inherited local head route: true historical local labels, detached states.
        local = previous.parent.stats(meta)[1]
        old = previous.parent.data(meta, "train")
        batch = {k: torch.tensor(old[k][:2], device=device) for k in ("x", "y", "mask")}
        p = torch.tensor([[32, 64, 96, 128]], device=device).expand(2, -1)
        z = encoder(batch["x"])
        detail = model.local_head(
            z.detach()[torch.arange(2, device=device)[:, None], p - 1]
        ).reshape(2, 4, 16, 7)
        ly, lm = hq.ba.local_targets(batch["y"], batch["mask"], p, stats, local)
        loss = hq.ba.local_rows(detail, ly, lm, True)["primary"].mean()
        grads = torch.autograd.grad(loss, list(encoder.parameters()), allow_unused=True)
        diagnostic["detached_local"] = {
            "loss": float(loss.detach()),
            "encoder_all_unused": all(g is None for g in grads),
        }
        if not diagnostic["detached_local"]["encoder_all_unused"]:
            raise ValueError("Historical detached-head route unexpectedly changed")
        after = (
            previous.parent.ur.bb.state_signature(encoder),
            previous.parent.ur.bb.state_signature(query),
        )
        if before != after:
            raise ValueError("Read-only diagnostic mutated selected weights/buffers")
        atomic_json(diagnostic, out / f"gradients_s{seed}.json")
        report[f"macro_s{seed}"] = checkpoint_report(ck, path)
        del ck, old, batch, z, detail, loss, grads, ly, lm
        progress(f"R0 s{seed}: replaying rolling last using its own saved validation")
        path = source / f"rolling_s{seed}/update_last.pt"
        ck = torch.load(path, map_location="cpu", weights_only=True)
        expected_binding = previous.binding(
            source, {"name": f"rolling_s{seed}", "mode": "rolling", "seed": seed}
        )
        if ck["metadata"] != expected_binding or ck["epoch"] != 60:
            raise ValueError("Rolling snapshot metadata/epoch differs")
        encoder.load_state_dict(ck["encoder"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
        actual = previous.validation(meta, encoder, query, val, device)
        save_checks(
            out,
            f"rolling_s{seed}",
            nested_compare(actual, ck["validation"]),
            actual=actual,
            expected=ck["validation"],
        )
        causality(encoder, query, b, out, f"rolling_s{seed}")
        report[f"rolling_s{seed}"] = checkpoint_report(ck, path)
        del encoder, query, model, ck, b
        gc.collect()
        torch.cuda.empty_cache()
        progress(f"R0 s{seed}: replaying direct Shared control ancestor")
        sr = previous.parent.base.prior
        model, query = sr.construct(sm, seed, device)
        item = mm["task_source"]["chosen"][str(seed)]
        path = shared / item["path"]
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["epoch"] != item["selected_epoch"] or ck["metadata"]["manifest_sha256"] != sha256(
            shared / "manifest.json"
        ):
            raise ValueError("Shared selected snapshot metadata differs")
        model.load_state_dict(ck["model"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
        actual = sr.validation(sm, model, query, device)
        save_checks(
            out,
            f"shared_s{seed}",
            nested_compare(actual, ck["validation"]),
            actual=actual,
            expected=ck["validation"],
        )
        causality(
            model.core.encoder,
            query,
            torch.tensor(train[:2], device=device),
            out,
            f"shared_s{seed}",
        )
        report[f"shared_s{seed}"] = checkpoint_report(ck, path)
        del model, query, ck
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(report, out / "checkpoint_replays.json")
    return report


def run(source, out):
    source, out = safe_output(source, out)
    out.mkdir(parents=True)
    started = time.monotonic()
    stage = "environment"
    status = {"schema": SCHEMA, "status": "running", "optimizer_updates": 0, "r1_authorized": False}
    atomic_json(status, out / "audit_status.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Real checkpoint replay is AutoDL CUDA-only; local tests use synthetic fixtures"
            )
        atomic_json(
            {
                "schema": SCHEMA,
                "source": str(source),
                "code_sha256": {Path(__file__).name: sha256(__file__)},
                "historical_code": previous.code_identity(),
                "numeric_contract": {
                    "atol": ATOL,
                    "rtol": RTOL,
                    "target_atol": TARGET_ATOL,
                    "state_atol": STATE_ATOL,
                    "state_rtol": STATE_RTOL,
                },
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(),
                "optimizer_updates": 0,
            },
            out / "manifest.json",
        )
        stage = "identity"
        atomic_json(
            dict(status, stage=stage, elapsed_seconds=time.monotonic() - started),
            out / "audit_status.json",
        )
        progress(
            "R0 verifying immutable weights, cache ancestry and bound source code; zero training"
        )
        meta, mm, sm, stats = identities(source, out)
        initial_identity = read_json(out / "identity.json")
        stage = "raw_and_targets"
        atomic_json(
            dict(status, stage=stage, elapsed_seconds=time.monotonic() - started),
            out / "audit_status.json",
        )
        progress("R0 rebuilding raw inputs/targets and partition eligibility on CPU")
        raw = raw_replay(source, meta, stats, out)
        stage = "checkpoint_replay_and_gradients"
        atomic_json(
            dict(status, stage=stage, elapsed_seconds=time.monotonic() - started),
            out / "audit_status.json",
        )
        replays = neural_replay(source, meta, mm, sm, stats, out)
        stage = "source_unchanged"
        atomic_json(
            dict(status, stage=stage, elapsed_seconds=time.monotonic() - started),
            out / "audit_status.json",
        )
        final_meta = identities(source, out)
        if (
            final_meta != (meta, mm, sm, stats)
            or read_json(out / "identity.json") != initial_identity
        ):
            atomic_json(initial_identity, out / "identity_before.json")
            raise ValueError("Source identity changed during audit")
        status.update(
            status="audit_complete_requires_review",
            elapsed_seconds=time.monotonic() - started,
            stage="complete",
            replayed_checkpoints=len(replays),
            raw=raw,
            limitations=[
                "No independent reproduction of all ancestral training or original scaler fitting",
                "Gradient diagnostics require interpretation and do not themselves prove the cause of near-history errors",
                "Audit completion never starts R1/R2",
            ],
        )
        atomic_json(status, out / "audit_status.json")
        atomic_json(
            {**status, "files": {str(p.relative_to(out)): sha256(p) for p in out.rglob("*.json")}},
            out / "completion.json",
        )
        progress("R0 finished: zero updates; reports require review before R1")
    except Exception as error:
        status.update(
            status="failed",
            elapsed_seconds=time.monotonic() - started,
            stage=stage,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        atomic_json(status, out / "audit_status.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == "__main__":
    main()
