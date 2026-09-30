"""Locked original-state utility and shared-interval diagnostics."""

from pathlib import Path

import numpy as np
import torch

from . import depth_scaling_evaluate as de
from . import macro_history_run as run
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SPLITS = de.SPLITS
up, ur = de.up, de.ur


def entries(meta):
    return [
        (f"{j['name']}_{kind}", j, kind) for j in meta["experiments"] for kind in ("best", "last")
    ]


def check_models(meta, out):
    lock = read_json(out / "model_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json") or set(lock["weights"]) != {
        j["name"] for j in meta["experiments"]
    }:
        raise ValueError("Incomplete model lock")
    for job in meta["experiments"]:
        path = out / job["name"]
        state = read_json(path / "training_state.json")
        done = read_json(path / "completion.json")
        if (
            done["status"] != "complete"
            or done["metadata"]
            != {"manifest_sha256": lock["manifest_sha256"], "job": job, "phase": "main"}
            or state["epoch"] != meta["epochs"]
        ):
            raise ValueError("Incomplete worker")
        run.parent.verify_files(path, done["files"])
        run.verify_history(
            meta, job, state, read_json(run.root(meta) / "train_cross_plan.json")["eligible"]
        )
        if (
            state["metadata"] != done["metadata"]
            or state["fork_sha256"] != meta["task_source"]["chosen"][str(job["seed"])]["sha256"]
        ):
            raise ValueError("Worker source identity changed")
        run.verify_summary(meta, state, lock["trials"][job["name"]])
        if state["history"] != read_json(path / "history.json"):
            raise ValueError("Worker history changed")
        if lock["trials"][job["name"]] != read_json(path / "training_summary.json"):
            raise ValueError("Trial lock changed")
        weights = lock["weights"][job["name"]]
        if set(weights) != {"best", "last"}:
            raise ValueError("Both checkpoint kinds required")
        for kind, item in weights.items():
            expected = state["best_epoch"] if kind == "best" else meta["epochs"]
            if (
                item["path"] != f"{job['name']}/e{meta['epochs']}_{kind}.pt"
                or item["epoch"] != expected
                or sha256(out / item["path"]) != item["sha256"]
            ):
                raise ValueError("Weight lock mismatch")
    return lock


def load(meta, out, job, kind, device):
    lock = read_json(out / "model_selection_lock.json")
    item = lock["weights"][job["name"]][kind]
    if sha256(out / item["path"]) != item["sha256"]:
        raise ValueError("Checkpoint changed")
    ck = torch.load(out / item["path"], map_location="cpu", weights_only=True)
    if (
        ck["metadata"]
        != {"manifest_sha256": sha256(out / "manifest.json"), "job": job, "phase": "main"}
        or ck["epoch"] != item["epoch"]
        or ck["budget_epoch"] != meta["epochs"]
    ):
        raise ValueError("Wrong checkpoint")
    model, query = run.construct(meta, job["seed"], device)
    model.load_state_dict(ck["model"], strict=True)
    query.load_state_dict(ck["query"], strict=True)
    return model.eval().requires_grad_(False), query.eval().requires_grad_(False)


@torch.inference_mode()
def audit_sources(meta, out, device):
    results = []
    source = Path(meta["task_source"]["source"])
    for seed in (42, 43):
        model, query = run.construct(meta, seed, device)
        model.eval().requires_grad_(False)
        query.eval().requires_grad_(False)
        for split in SPLITS:
            d = run.data(meta, split)
            q, _ = de.query_scores(meta, model, query, d, device)
            actual = {"query": q, "matched": de.matched_scores(meta, model, query, d, device)}
            reference = read_json(source / f"{split}_control_s{seed}_best.json")
            mismatches = run.parent.warm.recheck.mismatches(
                actual, {k: reference[k] for k in actual}
            )
            results.append(
                {"seed": seed, "split": split, "passed": not mismatches, "differences": mismatches}
            )
            atomic_json(results, out / "source_replay_preflight.json")
            if mismatches:
                raise ValueError("Source numeric replay differs before training; saved diagnostic")
            progress(f"Source replay passed: {split}/s{seed}")


def check_readouts(meta, out):
    check_models(meta, out)
    lock = read_json(out / "readout_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json") or lock[
        "model_lock_sha256"
    ] != sha256(out / "model_selection_lock.json"):
        raise ValueError("Readout model binding changed")
    run.parent.verify_files(out, lock["files"])


@torch.inference_mode()
def fit_readouts(meta, out, device):
    check_models(meta, out)
    if (out / "readout_selection_lock.json").exists():
        check_readouts(meta, out)
        return
    d = {s: run.data(meta, s) for s in ("train", "val")}
    b = {s: de.bank(meta, s) for s in d}
    for s in d:
        y, m = up.targets(up.restore_raw(d[s]["x"], run.stats(meta)[0]))
        if not np.array_equal(m, b[s]["mask"]) or not np.allclose(
            y, b[s]["targets"], atol=1e-6, rtol=2e-5
        ):
            raise ValueError("Target lineage differs")
    scales = read_json(run.root(meta) / "fit.json")["target_stats"]
    heads = {}
    arrays = {}
    inputs = {}
    for name, job, kind in entries(meta):
        model, _ = load(meta, out, job, kind, device)
        z = {s: de.states(model, d[s], meta["evaluation_batch"], device) for s in d}
        heads[name], w, bias = up.fit_heads(
            z["train"],
            b["train"]["targets"],
            b["train"]["mask"],
            z["val"],
            b["val"]["targets"],
            b["val"]["mask"],
            scales,
            device,
        )
        arrays[name + "_weights"] = w
        arrays[name + "_intercepts"] = bias
        inputs[name] = {s: ur.bb.cov.ndarray_hash(v) for s, v in z.items()}
        progress(f"{name}: locked 13 readers from original5 alphas")
    np.savez(out / "heads.npz", **arrays)
    atomic_json(
        {
            "heads": heads,
            "target_stats": scales,
            "input_hashes": inputs,
            "candidates": len(heads) * 13 * 5,
            "selection": "Identical train/validation calibration after ALL six model selections locked",
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


def decide(meta, records, inventories, overlap_rows, masks, absolute):
    expected = {n for n, _, _ in entries(meta)} | {f"source_s{s}" for s in (42, 43)}
    if set(records) != set(SPLITS) or any(set(v) != expected for v in records.values()):
        raise ValueError("Missing fixed comparisons")
    checks = []
    for mode, reference in (("content", "control"), ("aligned", "content")):
        for split, bank in records.items():
            rows = inventories[split]
            for seed in (42, 43):
                for kind in ("best", "last"):
                    a = bank[f"{mode}_s{seed}_{kind}"]
                    b = bank[f"{reference}_s{seed}_{kind}"]
                    source = bank[f"source_s{seed}"]

                    def check(
                        group,
                        label,
                        x,
                        y,
                        factor,
                        cohort=rows,
                        extra=True,
                        mode=mode,
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
                            raise ValueError("Invalid paired arrays")
                        ci = ur.interval(x, factor * y, cohort)
                        checks.append(
                            {
                                "mode": mode,
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

                    def near(v, k="path"):
                        return v["matched"]["paired80_128"]["actual_error"][k]

                    check(
                        "utility_gain",
                        "utility",
                        a["utility"]["errors"]["utility"],
                        b["utility"]["errors"]["utility"],
                        0.95,
                    )
                    if mode == "aligned":
                        check(
                            "utility_gain",
                            "utility/control",
                            a["utility"]["errors"]["utility"],
                            bank[f"control_s{seed}_{kind}"]["utility"]["errors"]["utility"],
                            0.95,
                        )
                    check("price_gain", "matched_close", near(a), near(b), 0.95)
                    for refname, ref in ((reference, b), ("source", source)):
                        for family in ("utility", "direction", "volatility"):
                            check(
                                "retention",
                                f"{family}/{refname}",
                                a["utility"]["errors"][family],
                                ref["utility"]["errors"][family],
                                1.05 if family == "utility" else 1.10,
                            )
                        for field in ("path", "change1", "body"):
                            check(
                                "retention",
                                f"matched/{field}/{refname}",
                                near(a, field),
                                near(ref, field),
                                1.10 if field != "path" else 1.05,
                            )
                        for field in ("primary", "level", "trend"):
                            check(
                                "retention",
                                f"far/{field}/{refname}",
                                a["query"]["structure"]["far65"][field],
                                ref["query"]["structure"]["far65"][field],
                                1.05 if field == "primary" else 1.10,
                            )
                        for task in ("held", "p128"):
                            for field in ("path", "change1", "body"):
                                check(
                                    "retention",
                                    f"query/{task}/{field}/{refname}",
                                    a["query"]["native"]["errors"][task][field],
                                    ref["query"]["native"]["errors"][task][field],
                                    1.10,
                                )
                            check(
                                "activity_retention",
                                f"activity/{task}/{refname}",
                                a["query"]["native"]["errors"][task]["activity"],
                                ref["query"]["native"]["errors"][task]["activity"],
                                1.10,
                            )
                    for shift in ("1", "16", "64"):
                        check(
                            "retention",
                            f"overlap/{shift}",
                            a["query_overlap"][shift]["combined_gap"],
                            b["query_overlap"][shift]["combined_gap"],
                            1.05,
                            overlap_rows[split],
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
    for mode, _reference in (("content", "control"), ("aligned", "content")):
        groups = {
            g: all(c["passed"] for c in checks if c["mode"] == mode and c["group"] == g)
            for g in ("utility_gain", "price_gain", "retention", "activity_retention")
        }
        status = (
            "balanced_utility_candidate"
            if groups["utility_gain"] and groups["retention"] and groups["activity_retention"]
            else "utility_gain_with_activity_tradeoff"
            if groups["utility_gain"] and groups["retention"]
            else "price_only_gain"
            if groups["price_gain"] and groups["retention"]
            else "no_confirmed_utility_upgrade"
        )
        branches[mode] = {"status": status, "groups": groups}
    return {
        "status": "matrix_complete_no_automatic_promotion",
        "branches": branches,
        "checks": checks,
        "original_utility_protocol": absolute,
        "automatic_promotion": False,
        "scope": "All seeds/cohorts/best+last required. Original weekly information checks; shared ratios use contract bootstrap preserving matched temporal pairs. Reused research data, uncorrected intervals.",
    }


@torch.inference_mode()
def evaluate(meta, out, device):
    check_readouts(meta, out)
    fit = read_json(out / "fit.json")
    source = Path(meta["task_source"]["source"])
    with np.load(out / "heads.npz", allow_pickle=False) as f:
        heads = dict(f)
    cm = run.xr.parent_meta(run.tm(meta))
    odr = run.xr.cr.odr
    om = odr.make_manifest(
        Path(cm["source"]),
        cm["identity"],
        {s: cm["packed"][s] for s in odr.SPLITS},
        meta["evaluation_batch"],
    )
    paired, _ = odr.prepare(om, out)
    records = {}
    inventories = {}
    masks = {}
    absolute = {}
    summaries = {}
    for split in SPLITS:
        d = run.data(meta, split)
        bank = de.bank(meta, split)
        rows = read_json(run.root(meta) / f"{split}_inventory.json")
        inventories[split] = rows
        masks[split] = bank["mask"]
        y, m = up.targets(up.restore_raw(d["x"], run.stats(meta)[0]))
        if not np.array_equal(m, bank["mask"]) or not np.allclose(
            y, bank["targets"], atol=1e-6, rtol=2e-5
        ):
            raise ValueError("Research label lineage differs")
        previous = read_json(source / f"{split}_readouts.json")
        scores = {}
        errors = {}
        predictions = {}
        records[split] = {}
        summaries[split] = {}
        for name in ("raw3584", "pca768", "current28"):
            pred = np.asarray(previous["predictions"][name])
            sc, er = up.measure(pred, bank["targets"], bank["mask"], fit["target_stats"])
            run.xr.ce.require_nested(sc, previous["scores"][name], name + " baseline replay")
            scores[name] = sc
            errors[name] = er
            predictions[name] = pred.tolist()
        for seed in (42, 43):
            name = f"source_s{seed}"
            rec = read_json(source / f"{split}_control_s{seed}_best.json")
            pred = np.asarray(rec["utility"]["predictions"])
            sc, er = up.measure(pred, bank["targets"], bank["mask"], fit["target_stats"])
            run.xr.ce.require_nested(sc, rec["utility"]["scores"], name + " readout replay")
            scores[name] = sc
            errors[name] = er
            predictions[name] = pred.tolist()
            records[split][name] = rec
            atomic_json(rec, out / f"{split}_{name}.json")
        for name, job, kind in entries(meta):
            model, query = load(meta, out, job, kind, device)
            causal = ur.pf.er.trained_causality(
                model, torch.tensor(run.data(meta, "val")["x"][:4], device=device)
            )
            if causal["status"] == "failed":
                raise ValueError("Candidate causality failed")
            atomic_json(causal, out / f"{name}_causality.json")
            q, z = de.query_scores(meta, model, query, d, device)
            pred = up.predict(
                fit["heads"][name],
                heads[name + "_weights"],
                heads[name + "_intercepts"],
                z,
                fit["target_stats"],
            )
            sc, er = up.measure(pred, bank["targets"], bank["mask"], fit["target_stats"])
            scores[name] = sc
            errors[name] = er
            predictions[name] = pred.tolist()
            rec = {
                "query": q,
                "matched": de.matched_scores(meta, model, query, d, device),
                "query_overlap": de.query_overlap(meta, model, query, paired[split], device),
                "structure_overlap": de.structure_overlap(
                    meta, model, query, paired[split], device
                ),
                "utility": {
                    "scores": sc,
                    "errors": {k: v.tolist() for k, v in er.items()},
                    "predictions": pred.tolist(),
                },
            }
            records[split][name] = rec
            atomic_json(rec, out / f"{split}_{name}.json")
            progress(f"Locked evaluation {split}/{name}")
        absolute[split] = ur.decide(
            {
                "models": list(records[split]),
                "pca_rank": 768,
                "decision": cm["reader"]["manifest"]["decision"],
            },
            scores,
            errors,
            bank["mask"],
            rows,
        )
        for name, rec in records[split].items():
            q = rec["query"]["native"]["scores"]["held"]
            summaries[split][name] = {
                "query_price": float(run.core.price(q)),
                "query_fixed": q["primary"],
                "activity": q["activity"],
                "close_mae_bps": q["close_mae_bps"],
                "near_price": float(
                    np.mean(rec["matched"]["paired80_128"]["actual_error"]["path"])
                ),
                "far": float(np.mean(rec["query"]["structure"]["far65"]["primary"])),
                "utility": rec["utility"]["scores"]["groups"]["utility"],
            }
        atomic_json(
            {
                "scores": scores,
                "errors": {n: {k: v.tolist() for k, v in er.items()} for n, er in errors.items()},
                "predictions": predictions,
                "targets": bank["targets"].tolist(),
                "mask": bank["mask"].tolist(),
            },
            out / f"{split}_readouts.json",
        )
    result = decide(
        meta, records, inventories, {s: paired[s]["rows"] for s in SPLITS}, masks, absolute
    )
    macro = evaluate_macro(meta, out, device)
    result = macro_decision(result, macro)
    atomic_json(result, out / "decision.json")
    atomic_json(summaries, out / "metrics.json")
    (out / "summary.md").write_text(
        "# Shared history representation\n\n"
        + result["status"]
        + "\n\nFixed source/input/budget; decoded information and state utility protected. Fixed macro readout alone never promotes a state. No automatic promotion.\n"
    )


def row_errors(predicted, target):
    e = np.asarray(predicted, float) - np.asarray(target, float)
    return 0.5 * ((e[..., :4] ** 2).mean(-1) + (e[..., 4:] ** 2).mean(-1))


def scores(payload):
    p = np.asarray(payload["predictions"], float)
    y = np.asarray(payload["targets"], float)
    duration = np.asarray(payload["duration"], float)
    if (
        p.shape != y.shape
        or p.shape != (3, len(payload["rows"]), 2, 7)
        or not np.isfinite([p, y]).all()
    ):
        raise ValueError("Finite physical macro content with row identities required")
    if duration.shape != (len(payload["rows"]), 2) or (duration <= 0).any():
        raise ValueError("Invalid recorded physical durations")
    # Counterfactual converts the other query back into THIS query's units.
    wrong = p[:, :, ::-1] * np.sqrt(duration[:, ::-1] / duration)[None, :, :, None]
    wrong_truth = y[:, :, ::-1] * np.sqrt(duration[:, ::-1] / duration)[None, :, :, None]
    informative = row_errors(y[1], wrong_truth[1]).mean(-1) >= 0.05
    return {
        "correct": row_errors(p, y).mean(-1),
        "wrong_query": row_errors(wrong, y).mean(-1),
        "zero": row_errors(np.zeros_like(y), y).mean(-1),
        "informative": informative,
        "within": row_errors(p[0], p[1]).mean(-1),
        "cross": row_errors(p[1], p[2]).mean(-1),
    }


@torch.inference_mode()
def macro_predictions(meta, out, model, reader, split, device):
    md = run.macro_data
    plan = read_json(out / f"{split}_macro_plan.json")
    rows = plan["rows"]
    ps = []
    ys = []
    durations = []
    ages = []
    masks = []
    bins = []
    for start in range(0, len(rows), meta["evaluation_batch"]):
        batch, geometry, _, target = md.evaluation_batch(
            meta,
            rows[start : start + meta["evaluation_batch"]],
            split,
            plan["shift"],
            device,
            plan["bands"],
        )
        z = model.core.encoder(batch["x"])[:, -1]
        p = run.core.describe(reader(z), geometry, run.stats(meta)[0])
        ps.append(p.cpu().numpy())
        ys.append(target.cpu().numpy())
        durations.extend(geometry["duration"].cpu().tolist())
        masks.extend(geometry["mask"].cpu().tolist())
        bins.extend(geometry["bins"].cpu().tolist())
        ages.append(geometry["ages"].cpu().numpy())
    return {
        "rows": rows,
        "predictions": np.concatenate(ps, 1).tolist(),
        "targets": np.concatenate(ys, 1).tolist(),
        "duration": durations,
        "ages": np.concatenate(ages, 1).tolist(),
        "mask": masks,
        "bins": bins,
        "bands": plan["bands"],
        "shift": plan["shift"],
        "units": "training close delta_scale times sqrt(recorded physical fine-bar duration)",
        "delta_scale": run.stats(meta)[0]["delta_scale"][0],
    }


def gate_metrics(payload):
    from . import grounded_history_warmup as qw

    values = scores(payload)
    keep = values["informative"]
    rows = [r for r, k in zip(payload["rows"], keep, strict=True) if k]
    checks = []
    for view in range(3):
        for ref in ("zero", "wrong_query"):
            ci = qw.interval(
                values["correct"][view, keep], values[ref][view, keep], rows, factor=0.9
            )
            checks.append(
                {
                    "view": view,
                    "reference": ref,
                    "interval": ci,
                    "passed": bool(ci["supported"] and ci["high"] is not None and ci["high"] < 0),
                }
            )
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "informative_ids": np.flatnonzero(keep).tolist(),
    }


@torch.inference_mode()
def baseline_gate(meta, out, seed, device):
    model, reader = run.construct(meta, seed, device)
    payload = macro_predictions(meta, out, model, reader, "val", device)
    name = f"s{seed}_macro_baseline.json"
    atomic_json(payload, out / name)
    result = dict(
        gate_metrics(payload),
        manifest_sha256=sha256(out / "manifest.json"),
        seed=seed,
        files={name: sha256(out / name)},
        updates=0,
        fits=0,
        scope="Validation-only zero-fit readiness gate; no model promotion.",
    )
    atomic_json(result, out / f"s{seed}_macro_gate.json")
    return result


@torch.inference_mode()
def evaluate_macro(meta, out, device):
    result = {s: {} for s in SPLITS}
    for seed in (42, 43):
        source_model, reader = run.construct(meta, seed, device)
        reader.requires_grad_(False)
        for split in SPLITS:
            payload = macro_predictions(meta, out, source_model, reader, split, device)
            name = f"source_s{seed}"
            result[split][name] = payload
            atomic_json(payload, out / f"{split}_{name}_macro.json")
        del source_model
        for name, job, kind in entries(meta):
            if job["seed"] != seed:
                continue
            model, query = load(meta, out, job, kind, device)
            del query
            for split in SPLITS:
                payload = macro_predictions(meta, out, model, reader, split, device)
                result[split][name] = payload
                atomic_json(payload, out / f"{split}_{name}_macro.json")
            del model
            progress(f"Fixed original-reader macro evaluation: {name}")
        del reader
    return result


def macro_decision(result, records):
    from . import grounded_history_warmup as qw

    checks = []
    for split, bank in records.items():
        for seed in (42, 43):
            for kind in ("best", "last"):
                for mode, ref in (("content", "control"), ("aligned", "content")):
                    a = bank[f"{mode}_s{seed}_{kind}"]
                    b = bank[f"{ref}_s{seed}_{kind}"]
                    source = bank[f"source_s{seed}"]
                    if (
                        a["rows"] != b["rows"]
                        or a["rows"] != source["rows"]
                        or a["targets"] != b["targets"]
                        or a["targets"] != source["targets"]
                    ):
                        raise ValueError("Macro paired cohort or truth changed")
                    sa, sb, ss = map(scores, (a, b, source))
                    rows = a["rows"]
                    for pair in ("15->30", "30->60", "15->60"):
                        ids = np.flatnonzero([r["pair"] == pair for r in rows])
                        if not len(ids):
                            continue
                        rs = [rows[i] for i in ids]

                        def check(
                            group,
                            label,
                            x,
                            y,
                            factor,
                            ids=ids,
                            rs=rs,
                            split=split,
                            seed=seed,
                            kind=kind,
                            mode=mode,
                            pair=pair,
                        ):
                            ci = qw.interval(x[ids], y[ids], rs, factor=factor)
                            checks.append(
                                {
                                    "dataset": split,
                                    "seed": seed,
                                    "checkpoint": kind,
                                    "mode": mode,
                                    "pair": pair,
                                    "group": group,
                                    "metric": label,
                                    "interval": ci,
                                    "passed": bool(
                                        ci["supported"]
                                        and ci["high"] is not None
                                        and ci["high"] <= 0
                                    ),
                                }
                            )

                        for view in range(3):
                            for label, sc in ((ref, sb), ("source", ss)):
                                check(
                                    "macro_retention",
                                    f"content{view}/{label}",
                                    sa["correct"][view],
                                    sc["correct"][view],
                                    1.05,
                                )
                            if mode == "content":
                                check(
                                    "macro_gain",
                                    f"content{view}/{ref}",
                                    sa["correct"][view],
                                    sb["correct"][view],
                                    0.95,
                                )
                            check(
                                "content_identity",
                                f"content{view}/zero",
                                sa["correct"][view],
                                sa["zero"][view],
                                0.9,
                            )
                        informative_ids = ids[sa["informative"][ids]]
                        informative_rows = [rows[i] for i in informative_ids]
                        for view in range(3):
                            check(
                                "content_identity",
                                f"content{view}/wrong_query",
                                sa["correct"][view],
                                sa["wrong_query"][view],
                                0.9,
                                ids=informative_ids,
                                rs=informative_rows,
                            )
                        if mode == "aligned":
                            for key in ("within", "cross"):
                                check("macro_gain", key, sa[key], sb[key], 0.90)
    for mode, branch in result["branches"].items():
        # Trained15->30 and held30->60 both required;15->60 reported if present.
        required = [c for c in checks if c["mode"] == mode and c["pair"] in ("15->30", "30->60")]
        groups = {
            g: bool([c for c in required if c["group"] == g])
            and all(c["passed"] for c in required if c["group"] == g)
            for g in ("macro_retention", "macro_gain", "content_identity")
        }
        groups["coverage"] = all(
            any(
                c["mode"] == mode
                and c["dataset"] == s
                and c["pair"] == p
                and c["seed"] == seed
                and c["checkpoint"] == kind
                for c in checks
            )
            for s in SPLITS
            for p in ("15->30", "30->60")
            for seed in (42, 43)
            for kind in ("best", "last")
        )
        branch["macro_groups"] = groups
        branch["macro_status"] = (
            "physical_macro_candidate"
            if all(groups.values())
            and branch["groups"]["retention"]
            and branch["groups"]["activity_retention"]
            else "no_balanced_macro_upgrade"
        )
        branch["representation_candidate"] = bool(
            branch["macro_status"] == "physical_macro_candidate"
            and branch["groups"]["utility_gain"]
        )
    result["macro_checks"] = checks
    result["scope"] = (
        "Repeated research data, uncorrected whole-contract macro intervals. Fixed original reader; accuracy protection against candidate/source prevents collapse-only wins. Macro gain is not state-utility gain or deployment approval."
    )
    return result
