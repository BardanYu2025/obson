"""Locked relative-state evaluation with matched endpoints and utility gates."""

from pathlib import Path

import numpy as np
import torch

from . import depth_scaling_run as run
from . import history_query as hq
from . import history_structure as core
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

ur, xr = run.ur, run.xr
up = ur.up
SPLITS = ("test", "cross_research")
PREFIXES = tuple(sorted(hq.ba.VAL_PREFIXES + hq.ba.HELD_PREFIXES))


def entries(meta):
    return [(f"{j['name']}_{k}", j, k) for j in run.variants(meta) for k in ("best", "last")]


def check_models(meta, out):
    lock = read_json(out / "model_selection_lock.json")
    expected = {v["name"] for v in run.variants(meta)}
    if (
        lock["manifest_sha256"] != sha256(out / "manifest.json")
        or set(lock["weights"]) != expected
        or set(lock["selections"]) != expected
    ):
        raise ValueError("All depth/budget/LR selections must be locked")
    for name, weights in lock["weights"].items():
        if set(weights) != {"best", "last"}:
            raise ValueError("Missing best/last")
        selected = lock["selections"][name]
        winner = run.choose_learning_rate(selected["candidates"])
        if any(selected[k] != winner[k] for k in winner):
            raise ValueError("LR selection differs from protected validation candidates")
        for candidate in selected["candidates"]:
            score = run.selected_value(
                candidate["validation"], lock["trials"][candidate["trial"]]["initial_validation"]
            )
            if list(score) != candidate["score"]:
                raise ValueError("LR validation score changed")
        for kind, item in weights.items():
            rel = f"{selected['trial']}/e{selected['budget']}_{kind}.pt"
            if item["path"] != rel or sha256(out / rel) != item["sha256"]:
                raise ValueError("Selected depth/budget weights changed")
    return lock


def load(meta, out, job, kind, device):
    lock = read_json(out / "model_selection_lock.json")
    selected = lock["selections"][job["name"]]
    model, query = run.construct(meta, job["seed"], device, f"d{job['depth']}")
    rel = lock["weights"][job["name"]][kind]
    if sha256(out / rel["path"]) != rel["sha256"]:
        raise ValueError("Selected weight hash mismatch")
    ck = torch.load(out / rel["path"], map_location="cpu", weights_only=True)
    trial = next(j for j in meta["experiments"] if j["name"] == selected["trial"])
    expected = {"manifest_sha256": sha256(out / "manifest.json"), "job": trial, "phase": "main"}
    if (
        ck["metadata"] != expected
        or ck["budget_epoch"] != job["budget"]
        or ck["epoch"] != (selected["selected_epoch"] if kind == "best" else job["budget"])
    ):
        raise ValueError("Selected checkpoint identity/budget differs")
    model.load_state_dict(ck["model"], strict=True)
    query.load_state_dict(ck["query"], strict=True)
    return model.eval().requires_grad_(False), query.eval().requires_grad_(False)


def load_prior(meta, mode, seed, device):
    model, query = run.construct(meta, seed, device)
    path = Path(meta["start"]["source"]) / f"{mode}_s{seed}/best.pt"
    if sha256(path) != meta["start"]["files"][f"{mode}_s{seed}/best.pt"]:
        raise ValueError("Retained source weight changed")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"], strict=True)
    query.load_state_dict(ck["query"], strict=True)
    return model.eval().requires_grad_(False), query.eval().requires_grad_(False)


@torch.inference_mode()
def audit_sources(meta, out, device):
    """Only replay already-completed fixed sources; no candidate or fit exists yet."""
    results = []
    for seed in (42, 43):
        for mode in ("plain",):
            model, query = load_prior(meta, mode, seed, device)
            for split in SPLITS:
                d = run.data(meta, split)
                reference = read_json(
                    Path(meta["start"]["source"]) / f"{split}_{mode}_s{seed}_best.json"
                )
                q, _ = query_scores(meta, model, query, d, device)
                actual = {"query": q, "matched": matched_scores(meta, model, query, d, device)}
                expected = {k: reference[k] for k in actual}
                differences = run.warm.recheck.mismatches(actual, expected)
                results.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "split": split,
                        "passed": not differences,
                        "differences": differences,
                    }
                )
                atomic_json(results, out / "source_replay_preflight.json")
                if differences:
                    raise ValueError(
                        "Fixed source numeric path differs before training; see source_replay_preflight.json"
                    )
                progress(f"Fixed source replay passed before training: {split}/{mode}_s{seed}")
    return results


def bank(meta, split):
    with np.load(run.root(meta) / f"cache/{split}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in ("targets", "mask")}


@torch.inference_mode()
def states(model, d, batch, device):
    return np.concatenate(
        [
            model.core.encoder(torch.tensor(np.asarray(d["x"][i : i + batch]), device=device))[
                :, -1
            ]
            .cpu()
            .numpy()
            for i in range(0, len(d["x"]), batch)
        ]
    )


def check_readouts(meta, out):
    check_models(meta, out)
    lock = read_json(out / "readout_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json") or lock[
        "model_lock_sha256"
    ] != sha256(out / "model_selection_lock.json"):
        raise ValueError("Reader/model binding differs")
    run.verify_files(out, lock["files"])
    return lock


@torch.inference_mode()
def fit_readouts(meta, out, device):
    check_models(meta, out)
    if (out / "readout_selection_lock.json").exists():
        check_readouts(meta, out)
        return
    d = {s: run.data(meta, s) for s in ("train", "val")}
    b = {s: bank(meta, s) for s in d}
    statistics = run.stats(meta)[0]
    for s in d:
        y, m = up.targets(up.restore_raw(d[s]["x"], statistics))
        if not np.array_equal(m, b[s]["mask"]) or not np.allclose(
            y, b[s]["targets"], atol=1e-6, rtol=2e-5
        ):
            raise ValueError("Reader target/input lineage mismatch")
    target_stats = read_json(run.root(meta) / "fit.json")["target_stats"]
    heads = {}
    arrays = {}
    inputs = {}
    for name, job, kind in entries(meta):
        model, _ = load(meta, out, job, kind, device)
        z = {s: states(model, d[s], meta["evaluation_batch"], device) for s in d}
        heads[name], w, bias = up.fit_heads(
            z["train"],
            b["train"]["targets"],
            b["train"]["mask"],
            z["val"],
            b["val"]["targets"],
            b["val"]["mask"],
            target_stats,
            device,
        )
        arrays[name + "_weights"] = w
        arrays[name + "_intercepts"] = bias
        inputs[name] = {s: ur.bb.cov.ndarray_hash(z[s]) for s in z}
        progress(f"{name}: calibrated13 readouts, fixed original5 alphas")
    np.savez(out / "heads.npz", **arrays)
    atomic_json(
        {
            "heads": heads,
            "target_stats": target_stats,
            "input_hashes": inputs,
            "candidates": len(heads) * 13 * 5,
            "selection": "Original train/validation only; all models locked first",
        },
        out / "fit.json",
    )
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "model_lock_sha256": sha256(out / "model_selection_lock.json"),
            "files": {n: sha256(out / n) for n in ("heads.npz", "fit.json")},
        },
        out / "readout_selection_lock.json",
    )


@torch.inference_mode()
def query_scores(meta, model, query, d, device):
    stats, local = run.stats(meta)
    result = {}
    recent = {}
    oldest = []
    structure = {"held": [], "p128": [], "far65": []}
    zs = []
    examples = []
    chosen = set(np.linspace(0, len(d["x"]) - 1, 4, dtype=int).tolist())
    for start in range(0, len(d["x"]), meta["evaluation_batch"]):
        b = run.batch_tensors(
            {
                k: v[start : start + meta["evaluation_batch"]]
                for k, v in d.items()
                if k in ("x", "y", "mask")
            },
            device,
        )
        ps = torch.tensor([PREFIXES] * len(b["x"]), device=device)
        z, p = hq.predict_states(model, query, b["x"], ps)
        y, mask = hq.targets(b, ps, stats)
        measures = hq.metrics(p, y, mask, stats)
        coarse = core.metrics(p, y, mask, stats)
        held = [PREFIXES.index(v) for v in hq.ba.HELD_PREFIXES]
        for group, ids in (("held", held), ("p128", [-1])):
            structure[group].append({k: v[:, ids].mean(1).cpu().numpy() for k, v in coarse.items()})
        far = core.metrics(p[:, -1:], y[:, -1:], mask[:, -1:], stats, lo=65)
        structure["far65"].append({k: v[:, 0].cpu().numpy() for k, v in far.items()})
        oldest_mask = mask[:, -1:].clone()
        oldest_mask[..., :64, :] = False
        oldest.append(
            {
                k: v[:, 0].cpu().numpy()
                for k, v in hq.metrics(p[:, -1:], y[:, -1:], oldest_mask, stats).items()
            }
        )
        zs.append(z[:, -1].cpu().numpy())
        old_y, old_mask = hq.ba.local_targets(b["y"], b["mask"], ps, stats, local)
        r = hq.recent_prediction(p, stats, local)
        for j, prefix in enumerate(PREFIXES):
            dst = result.setdefault(f"p{prefix}", {k: [] for k in measures})
            for k, v in measures.items():
                dst[k].append(v[:, j].cpu().numpy())
            _, err = ur.bb.pr.measure(
                r[:, j].cpu().numpy(),
                old_y[:, j].cpu().numpy(),
                old_mask[:, j].cpu().numpy(),
                local,
            )
            dst = recent.setdefault(f"p{prefix}", {k: [] for k in err})
            for k, v in err.items():
                dst[k].append(v)
        for i in chosen:
            if start <= i < start + len(ps):
                j = i - start
                examples.append(
                    {
                        "index": i,
                        "prefixes": list(PREFIXES),
                        "ages": list(hq.AGES),
                        "prediction": hq.physical(p[j], stats).cpu().tolist(),
                        "target": hq.physical(y[j], stats).cpu().tolist(),
                        "mask": mask[j].cpu().tolist(),
                        "semantics": "Historical close log-percent relative to observed current close; ages increase into past. No target values are decoder inputs.",
                    }
                )

    def finish(bank):
        bank = {
            k: {m: np.concatenate(v) for m, v in metrics.items()} for k, metrics in bank.items()
        }
        for name, positions in (("held", hq.ba.HELD_PREFIXES), ("trained", hq.ba.VAL_PREFIXES)):
            bank[name] = {
                k: np.mean([bank[f"p{p}"][k] for p in positions], axis=0) for k in bank["p128"]
            }
        return {
            "scores": {k: {m: float(v.mean()) for m, v in e.items()} for k, e in bank.items()},
            "errors": {k: {m: v.tolist() for m, v in e.items()} for k, e in bank.items()},
        }

    return {
        "structure": {
            g: {k: np.concatenate([v[k] for v in vals]).tolist() for k in vals[0]}
            for g, vals in structure.items()
        },
        "oldest_endpoint": {k: np.concatenate([r[k] for r in oldest]).tolist() for k in oldest[0]},
        "native": finish(result),
        "recent": finish(recent),
        "examples": examples,
    }, np.concatenate(zs)


@torch.inference_mode()
def query_overlap(meta, model, query, paired, device):
    stats, _ = run.stats(meta)
    predictions = {}
    for shift, view in paired["views"].items():
        arrays = []
        for start in range(0, len(view["x"]), meta["evaluation_batch"]):
            x = torch.tensor(
                np.asarray(view["x"][start : start + meta["evaluation_batch"]]), device=device
            )
            z = model.core.encoder(x)[:, -1]
            values = hq.physical(query(z), stats).flip(1)
            # Common price CHANGES cancel the different current-close origins.
            full = torch.cat((values, values.new_zeros(len(values), 1, 7)), 1)
            full = (full - full.new_tensor(stats["y_mean"])) / full.new_tensor(stats["y_scale"])
            arrays.append(full.float().cpu())
        predictions[shift] = torch.cat(arrays)
    result = {}
    for shift in (1, 16, 64):
        a, b = paired["views"][shift], paired["views"][0]
        gap = xr.xp.oc.consistency_rows(
            predictions[shift],
            predictions[0],
            torch.tensor(a["mask"]),
            torch.tensor(b["mask"]),
            torch.full((len(a["x"]),), shift),
            stats,
            False,
        )
        result[str(shift)] = {"combined_gap": gap.tolist(), "mean": float(gap.mean())}
    return result


@torch.inference_mode()
def structure_overlap(meta, model, query, paired, device):
    statistics, _ = run.stats(meta)
    result = {}
    for shift in (1, 16, 64):
        errors = []
        for start in range(0, len(paired["rows"]), meta["evaluation_batch"]):
            views = []
            for which in (shift, 0):
                b = run.batch_tensors(
                    {
                        k: v[start : start + meta["evaluation_batch"]]
                        for k, v in paired["views"][which].items()
                    },
                    device,
                )
                ps = torch.full((len(b["x"]), 1), 128, device=device)
                _, valid = hq.targets(b, ps, statistics)
                views.append((query(model.core.encoder(b["x"])[:, -1]), valid[:, 0]))
            p, a = views[0]
            q, b = views[1]
            errors.append(
                core.overlap(p, q, a, b, torch.full((len(p),), shift, device=device), statistics)
                .cpu()
                .numpy()
            )
        result[str(shift)] = np.concatenate(errors).tolist()
    return result


@torch.inference_mode()
def matched_scores(meta, model, query, data, device, lengths=(32, 64, 80, 128), pair=(80, 128)):
    """Identical endpoint and previous16 truth across direct lengths32/64/80/128."""
    statistics, local = run.stats(meta)
    if len(pair) != 2 or any(n not in lengths for n in pair):
        raise ValueError("Declared paired contexts must be evaluated")
    predictions = {n: [] for n in lengths}
    truth, masks = [], []
    for start in range(0, len(data["x"]), meta["evaluation_batch"]):
        b = run.batch_tensors(
            {
                k: v[start : start + meta["evaluation_batch"]]
                for k, v in data.items()
                if k in ("x", "y", "mask")
            },
            device,
        )
        ps = torch.full((len(b["x"]), 1), 128, device=device)
        target, mask = hq.ba.local_targets(b["y"], b["mask"], ps, statistics, local)
        truth.append(target[:, 0].cpu().numpy())
        masks.append(mask[:, 0].cpu().numpy())
        full_y, full_mask = hq.targets(b, ps, statistics)
        for length in lengths:
            short = {k: v[:, -length:] for k, v in b.items()}
            short_ps = torch.full_like(ps, length)
            short_y, short_mask = hq.targets(short, short_ps, statistics)
            # Only labels are compared here; observed history never feeds the decoder.
            if not torch.equal(full_y[:, :, :17], short_y[:, :, :17]) or not torch.equal(
                full_mask[:, :, :17], short_mask[:, :, :17]
            ):
                raise ValueError("Matched endpoint changed recent history labels")
            z = model.core.encoder(short["x"])[:, -1]
            predictions[length].append(
                hq.recent_prediction(query(z), statistics, local).cpu().numpy()
            )
    y, mask = np.concatenate(truth), np.concatenate(masks)
    arrays = {n: np.concatenate(v) for n, v in predictions.items()}
    records = {}
    for n in lengths:
        scores, errors = ur.bb.pr.measure(arrays[n], y, mask, local)
        records[str(n)] = {"scores": scores, "errors": {k: v.tolist() for k, v in errors.items()}}
    _, gap = ur.bb.pr.measure(arrays[pair[0]], arrays[pair[1]], mask, local)
    combined = {
        k: (
            (
                np.array(records[str(pair[0])]["errors"][k])
                + np.array(records[str(pair[1])]["errors"][k])
            )
            / 2
        ).tolist()
        for k in gap
    }
    ids = np.linspace(0, len(y) - 1, min(4, len(y)), dtype=int)
    return {
        "lengths": records,
        f"paired{pair[0]}_{pair[1]}": {
            "gap": {k: v.tolist() for k, v in gap.items()},
            "actual_error": combined,
        },
        "target_sha256": ur.bb.cov.ndarray_hash(y),
        "mask_sha256": ur.bb.cov.ndarray_hash(mask),
        "examples": [
            {
                "index": int(i),
                "target": y[i].tolist(),
                "mask": mask[i].tolist(),
                "predictions": {str(n): arrays[n][i].tolist() for n in lengths},
            }
            for i in ids
        ],
        "scope": "Same endpoint/previous16, different direct context AND position; retained causal EMA features.80 held from training/selection. Not pure position causality."
        if pair == (80, 128)
        else "Validation same endpoint/previous16;64/128 only, no held80 queries.",
    }


def decision(meta, records, inventories, overlap_rows, masks, absolute):
    references = ("plain",)
    expected = (
        {f"parent_s{s}" for s in (42, 43)}
        | {f"prior_{mode}_s{s}" for mode in references for s in (42, 43)}
        | {n for n, _, _ in entries(meta)}
    )
    if set(records) != set(SPLITS) or any(set(bank) != expected for bank in records.values()):
        raise ValueError("All locked states and fixed references required")
    checks = []
    comparisons = (("d8", "d4"), ("d12", "d4"), ("d12", "d8"), ("d8", "d4_c8"), ("d12", "d4_c12"))
    for mode, base in comparisons:
        comparison = f"{mode}_vs_{base}"
        for split, bank_ in records.items():
            rows = inventories[split]
            for seed in (42, 43):
                for kind in ("best", "last"):
                    a = bank_[f"{mode}_s{seed}_{kind}"]
                    b = bank_[f"{base}_s{seed}_{kind}"]

                    def check(
                        group,
                        label,
                        x,
                        y,
                        factor,
                        cohort=rows,
                        extra=True,
                        comparison=comparison,
                        mode=mode,
                        base=base,
                        split=split,
                        seed=seed,
                        kind=kind,
                    ):
                        x, y = np.asarray(x), np.asarray(y)
                        if (
                            x.shape != y.shape
                            or x.shape != (len(cohort),)
                            or not np.isfinite(x).all()
                            or not np.isfinite(y).all()
                        ):
                            raise ValueError("Invalid paired decision arrays")
                        ci = ur.interval(x, factor * y, cohort)
                        checks.append(
                            {
                                "comparison": comparison,
                                "mode": mode,
                                "baseline": base,
                                "dataset": split,
                                "seed": seed,
                                "checkpoint": kind,
                                "group": group,
                                "metric": label,
                                "factor": factor,
                                "interval": ci,
                                "passed": bool(
                                    extra
                                    and ci["supported"]
                                    and ci["high"] is not None
                                    and ci["high"] <= 0
                                ),
                            }
                        )

                    near_a, near_b = [
                        v["matched"]["paired80_128"]["actual_error"]["primary"] for v in (a, b)
                    ]
                    far_a, far_b = [v["query"]["structure"]["far65"]["primary"] for v in (a, b)]
                    check("near_gain", "matched/actual_error", near_a, near_b, 0.95)
                    check("far_gain", "far65/primary", far_a, far_b, 0.95)
                    check("retention", "matched/actual_error", near_a, near_b, 1.05)
                    check("retention", "far65/primary", far_a, far_b, 1.05)
                    for field in ("level", "trend"):
                        check(
                            "retention",
                            "far65/" + field,
                            a["query"]["structure"]["far65"][field],
                            b["query"]["structure"]["far65"][field],
                            1.10,
                        )
                    for what in ("gap", "actual_error"):
                        for ref in (b, bank_[f"prior_plain_s{seed}"]):
                            label = (
                                "matched/" + what + (" vs baseline" if ref is b else " vs source")
                            )
                            check(
                                "retention",
                                label,
                                a["matched"]["paired80_128"][what]["primary"],
                                ref["matched"]["paired80_128"][what]["primary"],
                                1.05,
                            )
                    for task in ("held", "p128"):
                        for field in ("primary", "path", "body", "activity", "change1"):
                            check(
                                "retention",
                                f"query/{task}/{field}",
                                a["query"]["native"]["errors"][task][field],
                                b["query"]["native"]["errors"][task][field],
                                1.05 if field == "primary" else 1.10,
                            )
                    for field in ("primary", "path", "body", "activity", "change1"):
                        check(
                            "retention",
                            "oldest_pointwise/" + field,
                            a["query"]["oldest_endpoint"][field],
                            b["query"]["oldest_endpoint"][field],
                            1.05 if field == "primary" else 1.10,
                        )
                    for shift in ("1", "16", "64"):
                        check(
                            "retention",
                            "overlap/" + shift,
                            a["query_overlap"][shift]["combined_gap"],
                            b["query_overlap"][shift]["combined_gap"],
                            1.05,
                            overlap_rows[split],
                        )
                    for ref in (
                        f"{base}_s{seed}_{kind}",
                        f"parent_s{seed}",
                        *[f"prior_{m}_s{seed}" for m in references],
                    ):
                        for group in ("utility", "direction", "volatility"):
                            check(
                                "retention",
                                f"{group} vs {ref}",
                                a["utility"]["errors"][group],
                                bank_[ref]["utility"]["errors"][group],
                                1.05 if group == "utility" else 1.10,
                            )
                    check(
                        "utility_gain",
                        "utility",
                        a["utility"]["errors"]["utility"],
                        b["utility"]["errors"]["utility"],
                        0.95,
                    )
                    for i, target in enumerate(up.NAMES[6:], 6):
                        ids = np.flatnonzero(masks[split][:, i])
                        r2 = a["utility"]["scores"]["targets"][i]["r2"]
                        check(
                            "retention",
                            target,
                            np.asarray(a["utility"]["errors"][target])[ids],
                            np.full(len(ids), 0.1),
                            1.0,
                            [rows[j] for j in ids],
                            r2 is not None and r2 >= 0.8,
                        )
    branches = {}
    for mode, base in comparisons:
        name = f"{mode}_vs_{base}"
        groups = {
            g: all(c["passed"] for c in checks if c["comparison"] == name and c["group"] == g)
            for g in ("near_gain", "far_gain", "retention", "utility_gain")
        }
        gain = groups["near_gain"] or groups["far_gain"]
        status = (
            "representation_candidate"
            if gain and groups["retention"] and groups["utility_gain"]
            else "balanced_state_gain"
            if gain and groups["retention"]
            else "gain_with_tradeoffs"
            if gain
            else "no_balanced_gain"
        )
        branches[name] = {"status": status, "groups": groups}
    depth_conclusions = {}
    for depth in (8, 12):
        exposure = branches[f"d{depth}_vs_d4"]
        compute = branches[f"d{depth}_vs_d4_c{depth}"]
        accepted = {"balanced_state_gain", "representation_candidate"}
        both = exposure["status"] in accepted and compute["status"] in accepted
        depth_conclusions[f"d{depth}"] = {
            "both_budget_axes_supported": both,
            "status": "depth_candidate_on_both_axes" if both else "no_confirmed_gain_on_both_axes",
            "scope": "Fixed continuation and two-LR matrix; does not prove sufficient convergence or all deeper architectures ineffective.",
        }
    return {
        "status": "matrix_complete_no_automatic_promotion",
        "branches": branches,
        "depth_conclusions": depth_conclusions,
        "checks": checks,
        "original_utility_protocol": absolute,
        "automatic_promotion": False,
        "scope": "Two prespecified improvement routes reported across all cohorts/seeds/best+last. Exposure comparisons and shallow matmul-proxy controls are separate; global depth gain requires both. LR chosen only on validation. Research sets reused; absolute utility protocol separate.",
    }


@torch.inference_mode()
def evaluate(meta, out, device):
    check_readouts(meta, out)
    cm = xr.parent_meta(run.tm(meta))
    odr = xr.cr.odr
    om = odr.make_manifest(
        Path(cm["source"]),
        cm["identity"],
        {s: cm["packed"][s] for s in odr.SPLITS},
        meta["evaluation_batch"],
    )
    paired, _ = odr.prepare(om, out)
    fit = read_json(out / "fit.json")
    with np.load(out / "heads.npz", allow_pickle=False) as f:
        heads = dict(f)
    source = Path(meta["identity"]["manifest"]["source"])
    records = {}
    inventories = {}
    masks = {}
    absolute = {}
    all_entries = (
        [(f"parent_s{s}", None, None, s) for s in (42, 43)]
        + [(n, j, k, j["seed"]) for n, j, k in entries(meta)]
        + [
            (f"prior_{mode}_s{seed}", {"prior": mode, "seed": seed}, "best", seed)
            for seed in (42, 43)
            for mode in ("plain",)
        ]
    )
    for split in SPLITS:
        d = run.data(meta, split)
        rows = read_json(run.root(meta) / f"{split}_inventory.json")
        b = bank(meta, split)
        inventories[split] = rows
        masks[split] = b["mask"]
        records[split] = {}
        truth, valid = up.targets(up.restore_raw(d["x"], run.stats(meta)[0]))
        if not np.array_equal(valid, b["mask"]) or not np.allclose(
            truth, b["targets"], atol=1e-6, rtol=2e-5
        ):
            raise ValueError("Research target identity changed")
        scores = {}
        errors = {}
        predictions = {}
        oldpred = read_json(source / f"{split}_predictions.json")
        for baseline in ("raw3584", "pca768", "current28"):
            prediction = np.array(oldpred["predictions"][baseline])
            scores[baseline], errors[baseline] = up.measure(
                prediction, b["targets"], b["mask"], fit["target_stats"]
            )
            xr.ce.require_nested(
                scores[baseline],
                read_json(source / "readout_metrics.json")["datasets"][split]["scores"][baseline],
                baseline + " reuse",
            )
            predictions[baseline] = prediction.tolist()
        for name, job, kind, seed in all_entries:
            if job is None:
                engine = run.rs.load_bundle(
                    Path(meta["source"]) / "bundle", seed, "MA/15/CZCE.MA601", 15, device
                )
                model = engine.aligned
                query = None
                z = states(model, d, meta["evaluation_batch"], device)
                u = engine.utility
                pred = up.predict(
                    u["heads"],
                    np.asarray(u["weights"]),
                    np.asarray(u["intercepts"]),
                    z,
                    u["target_stats"],
                )
            else:
                if "prior" in job:
                    model, query = load_prior(meta, job["prior"], seed, device)
                else:
                    model, query = load(meta, out, job, kind, device)
                causal = ur.pf.er.trained_causality(
                    model, torch.tensor(np.asarray(run.data(meta, "val")["x"][:4]), device=device)
                )
                if causal["status"] == "failed":
                    raise ValueError("Trained encoder causality failed")
                atomic_json(causal, out / f"{name}_causality.json")
                qresult, z = query_scores(meta, model, query, d, device)
                if "prior" in job:
                    previous = read_json(
                        Path(meta["start"]["source"]) / f"{split}_{job['prior']}_s{seed}_best.json"
                    )
                    pred = np.asarray(previous["utility"]["predictions"])
                    old_query = qresult
                    if run.warm.recheck.mismatches(old_query, previous["query"]):
                        raise ValueError("Fixed prior query replay differs")
                else:
                    pred = up.predict(
                        fit["heads"][name],
                        heads[name + "_weights"],
                        heads[name + "_intercepts"],
                        z,
                        fit["target_stats"],
                    )

            before = ur.bb.state_signature(model)
            original, err, _, _ = ur.pf.score(
                model, d, *run.stats(meta), meta["evaluation_batch"], device
            )
            scores[name], errors[name] = up.measure(
                pred, b["targets"], b["mask"], fit["target_stats"]
            )
            predictions[name] = pred.tolist()
            record = {
                "original": {
                    "scores": original,
                    "errors": {k: {m: v.tolist() for m, v in e.items()} for k, e in err.items()},
                },
                "utility": {
                    "scores": scores[name],
                    "errors": {k: v.tolist() for k, v in errors[name].items()},
                    "predictions": pred.tolist(),
                },
                "overlap": xr.ce.overlap(
                    model, paired[split], run.stats(meta)[0], meta["evaluation_batch"], device
                ),
            }
            if query is not None:
                record.update(
                    matched=matched_scores(meta, model, query, d, device),
                    query=qresult,
                    query_overlap=query_overlap(meta, model, query, paired[split], device),
                    structure_overlap=structure_overlap(meta, model, query, paired[split], device),
                )
            else:
                previous = read_json(source / f"{split}_control_s{seed}_best.json")
                replay = {
                    "reconstruction": record["original"],
                    "utility": record["utility"],
                    "overlap": record["overlap"],
                }
                expected = {
                    "reconstruction": {
                        k: previous["reconstruction"][k] for k in ("scores", "errors")
                    },
                    "utility": previous["utility"],
                    "overlap": previous["overlap"],
                }
                differences = run.warm.recheck.mismatches(replay, expected)
                atomic_json(
                    {"passed": not differences, "differences": differences},
                    out / f"{split}_{name}_replay.json",
                )
                if differences:
                    raise ValueError("Original source replay differs; see numeric diagnostic")
            if job is not None and "prior" in job:
                reference = read_json(
                    Path(meta["start"]["source"]) / f"{split}_{job['prior']}_s{seed}_best.json"
                )
                old_record = record
                mismatches = run.warm.recheck.mismatches(old_record, reference)
                atomic_json(
                    {"passed": not mismatches, "differences": mismatches},
                    out / f"{split}_{name}_replay.json",
                )
                if mismatches:
                    raise ValueError("Fixed prior source replay differs; see diagnostic")
            if before != ur.bb.state_signature(model):
                raise ValueError("Evaluation changed model")
            atomic_json(record, out / f"{split}_{name}.json")
            records[split][name] = record
            progress(f"Locked evaluation: {split}/{name}")
        absolute[split] = ur.decide(
            {
                "models": list(records[split]),
                "pca_rank": 768,
                "decision": cm["reader"]["manifest"]["decision"],
            },
            scores,
            errors,
            b["mask"],
            rows,
        )
        atomic_json(
            {
                "scores": scores,
                "targets": b["targets"].tolist(),
                "mask": b["mask"].tolist(),
                "predictions": predictions,
                "errors": {n: {k: v.tolist() for k, v in e.items()} for n, e in errors.items()},
            },
            out / f"{split}_readouts.json",
        )
    decided = decision(
        meta, records, inventories, {s: paired[s]["rows"] for s in SPLITS}, masks, absolute
    )
    atomic_json(decided, out / "decision.json")
    summary = {
        s: {
            n: {
                "far_structure": None
                if "query" not in r
                else float(np.mean(r["query"]["structure"]["far65"]["primary"])),
                "old_global": r["original"]["scores"]["global"]["metrics"]["primary"],
                "utility": r["utility"]["scores"]["groups"]["utility"],
                "matched80_128_error": None
                if "matched" not in r
                else float(np.mean(r["matched"]["paired80_128"]["actual_error"]["primary"])),
                "matched80_128_gap": None
                if "matched" not in r
                else float(np.mean(r["matched"]["paired80_128"]["gap"]["primary"])),
                "query_held": None
                if "query" not in r
                else r["query"]["native"]["scores"]["held"]["primary"],
                "query_recent_held": None
                if "query" not in r
                else r["query"]["recent"]["scores"]["held"]["primary"],
            }
            for n, r in b.items()
        }
        for s, b in records.items()
    }
    atomic_json(summary, out / "metrics.json")
    (out / "summary.md").write_text(
        "# Depth scaling: exposure and matmul-proxy budgets\n\n"
        + decided["status"]
        + "\n\nAll selections locked before research; full absolute utility protocol is separate. No automatic promotion.\n"
    )
