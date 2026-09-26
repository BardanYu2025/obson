"""Locked state readouts and paired query/original-capability research evaluation."""

from pathlib import Path

import numpy as np
import torch

from . import history_query as hq
from . import history_query_run as run
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

ur, xr = run.ur, run.xr
up = ur.up
SPLITS = ("test", "cross_research")
PREFIXES = tuple(sorted(hq.ba.VAL_PREFIXES + hq.ba.HELD_PREFIXES))


def entries(meta):
    return [(f"{j['name']}_{k}", j, k) for j in meta["experiments"] for k in ("best", "last")]


def check_models(meta, out):
    lock = read_json(out / "model_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json") or set(lock["weights"]) != {
        j["name"] for j in meta["experiments"]
    }:
        raise ValueError("All model selections must be locked")
    for name, weights in lock["weights"].items():
        if set(weights) != {"best", "last"}:
            raise ValueError("Missing best/last")
        for k, h in weights.items():
            if sha256(out / name / f"{k}.pt") != h:
                raise ValueError("Selected weights changed")
    return lock


def load(meta, out, job, kind, device):
    model, query = run.construct(meta, job["seed"], device)
    ck = torch.load(out / job["name"] / f"{kind}.pt", map_location="cpu", weights_only=True)
    expected = {"manifest_sha256": sha256(out / "manifest.json"), "job": job, "phase": "main"}
    if ck["metadata"] != expected:
        raise ValueError("Selected checkpoint metadata differs")
    lock = read_json(out / "model_selection_lock.json")["trials"][job["name"]]
    if ck["epoch"] != (lock["selected_epoch"] if kind == "best" else meta["epochs"]):
        raise ValueError("Selected epoch differs")
    model.load_state_dict(ck["model"])
    query.load_state_dict(ck["query"])
    return model.eval().requires_grad_(False), query.eval().requires_grad_(False)


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
    verify_files(out, lock["files"])
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


def decision(meta, records, inventories, overlap_rows, masks, absolute):
    expected = {f"parent_s{s}" for s in (42, 43)} | {n for n, _, _ in entries(meta)}
    if set(records) != set(SPLITS) or any(set(bank) != expected for bank in records.values()):
        raise ValueError("Both fixed research cohorts and all locked model records required")
    checks = []
    for split, bank in records.items():
        rows = inventories[split]
        for seed in (42, 43):
            parent = f"parent_s{seed}"
            for kind in ("best", "last"):
                name = f"joint_s{seed}_{kind}"
                a = bank[name]

                def check(
                    label, x, y, factor, cohort=rows, extra=True, split=split, seed=seed, kind=kind
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
                            "dataset": split,
                            "seed": seed,
                            "checkpoint": kind,
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

                for mode in ("frozen", "control"):
                    ref = f"{mode}_s{seed}_{kind}"
                    b = bank[ref]
                    for family in ("native", "recent"):
                        check(
                            f"gain/query_{family}_held vs {ref}",
                            a["query"][family]["errors"]["held"]["primary"],
                            b["query"][family]["errors"]["held"]["primary"],
                            0.95,
                        )
                        check(
                            f"retain/query_{family}_endpoint vs {ref}",
                            a["query"][family]["errors"]["p128"]["primary"],
                            b["query"][family]["errors"]["p128"]["primary"],
                            1.05,
                        )
                        for field in ("path", "body", "activity", "change1"):
                            check(
                                f"retain/query_{family}_held_{field} vs {ref}",
                                a["query"][family]["errors"]["held"][field],
                                b["query"][family]["errors"]["held"][field],
                                1.10,
                            )
                    for shift in (1, 16, 64):
                        check(
                            f"retain/query_overlap{shift} vs {ref}",
                            a["query_overlap"][str(shift)]["combined_gap"],
                            b["query_overlap"][str(shift)]["combined_gap"],
                            1.05,
                            overlap_rows[split],
                        )
                for ref in (f"control_s{seed}_{kind}", parent):
                    b = bank[ref]
                    for task, fields in (
                        ("global", ("primary", "path", "body", "activity", "changes")),
                        ("held", ("primary",)),
                        ("recent", ("primary",)),
                    ):
                        for field in fields:
                            check(
                                f"retain/original_{task}_{field} vs {ref}",
                                a["original"]["errors"][task][field],
                                b["original"]["errors"][task][field],
                                1.05 if field == "primary" else 1.10,
                            )
                    for group in ("utility", "direction", "volatility"):
                        check(
                            f"retain/calibrated_{group} vs {ref}",
                            a["utility"]["errors"][group],
                            b["utility"]["errors"][group],
                            1.05 if group == "utility" else 1.10,
                        )
                    for shift in (1, 16, 64):
                        av, bv = a["overlap"][str(shift)], b["overlap"][str(shift)]
                        ids = np.flatnonzero(
                            np.asarray(av["valid"]["combined_gap"])
                            & np.asarray(bv["valid"]["combined_gap"])
                        )
                        check(
                            f"retain/original_overlap{shift} vs {ref}",
                            np.asarray(av["per_pair"]["combined_gap"])[ids],
                            np.asarray(bv["per_pair"]["combined_gap"])[ids],
                            1.05,
                            [overlap_rows[split][i] for i in ids],
                        )
                for i, target in enumerate(up.NAMES[6:], 6):
                    ids = np.flatnonzero(masks[split][:, i])
                    r2 = a["utility"]["scores"]["targets"][i]["r2"]
                    check(
                        "retain/" + target,
                        np.asarray(a["utility"]["errors"][target])[ids],
                        np.full(len(ids), 0.1),
                        1.0,
                        [rows[j] for j in ids],
                        r2 is not None and r2 >= 0.8,
                    )
    gains = [c for c in checks if c["metric"].startswith("gain/")]
    gain = all(c["passed"] for c in gains)
    retention = all(c["passed"] for c in checks if not c["metric"].startswith("gain/"))
    return {
        "status": "history_query_research_candidate"
        if gain and retention
        else "query_gain_with_tradeoffs"
        if gain
        else "no_joint_query_gain",
        "gain_passed": gain,
        "retention_passed": retention,
        "checks": checks,
        "original_utility_protocol": absolute,
        "automatic_promotion": False,
        "scope": "Paired reused research cohorts; no forecasting/universal representation claim. Failure can reflect decoder optimization. Original interfaces and absolute utility reported separately.",
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
    all_entries = [(f"parent_s{s}", None, None, s) for s in (42, 43)] + [
        (n, j, k, j["seed"]) for n, j, k in entries(meta)
    ]
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
                model, query = load(meta, out, job, kind, device)
                causal = ur.pf.er.trained_causality(
                    model, torch.tensor(np.asarray(run.data(meta, "val")["x"][:4]), device=device)
                )
                if causal["status"] == "failed":
                    raise ValueError("Trained encoder causality failed")
                atomic_json(causal, out / f"{name}_causality.json")
                qresult, z = query_scores(meta, model, query, d, device)
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
                    query=qresult,
                    query_overlap=query_overlap(meta, model, query, paired[split], device),
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
                "old_global": r["original"]["scores"]["global"]["metrics"]["primary"],
                "utility": r["utility"]["scores"]["groups"]["utility"],
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
        "# Unified historical query research\n\n"
        + decided["status"]
        + "\n\nAll selections locked before research; full absolute utility protocol is separate. No automatic promotion.\n"
    )
