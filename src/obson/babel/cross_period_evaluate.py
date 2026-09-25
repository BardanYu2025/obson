"""Locked cross-period, reconstruction, same-period stability and utility checks."""

from pathlib import Path

import numpy as np
import torch

from . import cross_period_consistency as xp
from . import cross_period_data as xd
from . import cross_period_run as run
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

rr = run.rr
ur = run.ur
cr = run.cr
ce = run.ce
up = rr.up
old = rr.old


# These source helpers are already generic over manifest experiment entries.
# Reuse the unchanged, hash-bound lock and fixed-grid fitting implementation.
entries = run.ge.entries
check_models = run.ge.check_models
check_readouts = run.ge.check_readouts
fit = run.ge.fit


def load(meta, out, job, kind, device):
    model = run.construct(meta, job, device)
    ck = torch.load(out / job["name"] / f"{kind}.pt", map_location="cpu", weights_only=True)
    lock = read_json(out / "model_selection_lock.json")["trials"][job["name"]]
    if ck["metadata"] != {"manifest": meta, "job": job}:
        raise ValueError("Evaluation checkpoint identity differs")
    if kind == "best":
        if ck["epoch"] != lock["selected_epoch"] or ck["validation"] != lock["validation"]:
            raise ValueError("Best checkpoint changed")
        expected = ck["validation"]
    else:
        if ck["epoch"] != meta["epochs"]:
            raise ValueError("Incomplete fixed last")
        expected = ck["history"][-1]["validation"]
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    return model, expected


@torch.inference_mode()
def prepare(meta, out, split, device):
    check_models(out)
    if split in old.SPLITS[2:]:
        check_readouts(out)
    elif split not in old.SPLITS:
        raise ValueError("Unknown split")
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    if (cache / f"{split}_index.json").exists():
        old.verify_cache(meta, out, split)
        return
    source = Path(meta["source"])
    old.verify_cache(meta["identity"]["manifest"], source, split)
    bank = old.arrays(source, split)
    rows = read_json(out / f"{split}_inventory.json")
    if rows != read_json(source / f"{split}_inventory.json"):
        raise ValueError("Original row/week identity changed")
    data = ur.bb.load_data(cr.alignment(run.parent_meta(meta)), split)
    x = np.asarray(data["x"])
    stats, _ = run.statistics(meta)
    rr.check_inputs(bank, x, stats, rows)
    values = {k: bank[k] for k in ("raw", "current", "targets", "mask")}
    for seed in (42, 43):
        values[f"parent_s{seed}"] = bank[f"deep_s{seed}_best"]
    for name, job, kind in entries(meta):
        model, expected = load(meta, out, job, kind, device)
        before = ur.bb.state_signature(model)
        if split == "train":
            actual = run.validation(meta, job, model, device)
            ur.pf.ea.require_replay(actual, expected, name + "/trained validation")
            val = ur.bb.load_data(cr.alignment(run.parent_meta(meta)), "val")
            causal = ur.pf.er.trained_causality(
                model, torch.tensor(np.asarray(val["x"][:4]), device=device)
            )
            if causal["status"] == "failed":
                raise ValueError("Trained expanded model noncausal")
            path = out / "trained_validation_causality.json"
            checks = read_json(path) if path.exists() else {}
            checks[name] = {"validation": actual, "causality": causal}
            atomic_json(checks, path)
        zs = []
        for start in range(0, len(x), meta["evaluation_batch"]):
            zs.append(
                model.core.encoder(
                    torch.tensor(x[start : start + meta["evaluation_batch"]], device=device)
                )[:, -1]
                .cpu()
                .numpy()
            )
        values[name] = np.concatenate(zs)
        if (
            values[name].shape != (len(rows), meta["state_width"])
            or before != ur.bb.state_signature(model)
            or any(p.requires_grad for p in model.parameters())
        ):
            raise ValueError("State shape/frozen weight failure")
        del model
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite extraction")
    np.savez(cache / f"{split}.npz", **values)
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "windows": len(rows),
            "files": {f"{split}.npz": sha256(cache / f"{split}.npz")},
        },
        cache / f"{split}_index.json",
    )
    progress(f"Extracted8 checkpoints: {split}, {len(rows)} windows")


@torch.inference_mode()
def cross_scores(model, data, stats, batch, device):
    rows, views, index, valid = data
    accum = {k: [] for k in ("gap", "fine_error", "coarse_error")}
    for start in range(0, len(rows), batch):
        stop = min(start + batch, len(rows))
        n = stop - start
        x = torch.tensor(np.concatenate([v["x"][start:stop] for v in views]), device=device)
        z = model.core.encoder(x)[:, -1]
        pred = model.core.decoder(z)
        a, b = (torch.tensor(v["y"][start:stop], device=device) for v in views)
        result = xp.common_rows(
            pred[:n],
            pred[n:],
            torch.tensor(index[start:stop], device=device),
            torch.tensor(valid[start:stop], device=device),
            stats,
            a,
            b,
            smooth=False,
        )
        for k, v in result.items():
            if not torch.isfinite(v).all():
                raise ValueError("Nonfinite cross-period evaluation")
            accum[k].extend(v.cpu().tolist())
    groups = {}
    for pair in sorted({r["pair"] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r["pair"] == pair]
        groups[pair] = {k: float(np.asarray(v)[ids].mean()) for k, v in accum.items()}
        groups[pair]["pairs"] = len(ids)
    return {"per_pair": accum, "groups": groups}


def decide(meta, records, rows, overlap_rows, masks, cross_rows):
    checks = []
    for split, bank in records.items():
        for seed in (42, 43):
            for kind in ("best", "last"):
                a = f"cross_s{seed}_{kind}"
                b = f"control_s{seed}_{kind}"
                parent = f"parent_s{seed}"

                def check(
                    metric,
                    x,
                    y,
                    factor=1.0,
                    extra=True,
                    cohort=None,
                    *,
                    split=split,
                    seed=seed,
                    kind=kind,
                ):
                    ci = ur.interval(
                        np.asarray(x),
                        factor * np.asarray(y),
                        rows[split] if cohort is None else cohort,
                    )
                    checks.append(
                        {
                            "dataset": split,
                            "seed": seed,
                            "checkpoint": kind,
                            "metric": metric,
                            "factor": factor,
                            "interval": ci,
                            "extra_condition": bool(extra),
                            "passed": bool(
                                extra
                                and ci["supported"]
                                and ci["high"] is not None
                                and ci["high"] <= 0
                            ),
                        }
                    )

                for pair in ("15->30", "15->60", "30->60"):
                    ids = np.array(
                        [i for i, r in enumerate(cross_rows[split]) if r["pair"] == pair], dtype=int
                    )
                    cohort = [cross_rows[split][i] for i in ids]
                    if not len(ids):
                        checks.append(
                            {
                                "dataset": split,
                                "seed": seed,
                                "checkpoint": kind,
                                "metric": f"cross/{pair}/support",
                                "passed": False,
                            }
                        )
                        continue
                    for ref in (b, parent):
                        for metric in ("gap", "fine_error", "coarse_error"):
                            factor = (
                                0.90 if metric == "gap" and ref == b and pair != "30->60" else 1.05
                            )
                            check(
                                f"cross/{pair}/{metric} vs {ref}",
                                np.array(bank[a]["cross_period"]["per_pair"][metric])[ids],
                                np.array(bank[ref]["cross_period"]["per_pair"][metric])[ids],
                                factor,
                                cohort=cohort,
                            )
                for ref in (b, parent):
                    for task, metric in [
                        ("global", "primary"),
                        ("global", "path"),
                        ("global", "changes"),
                        ("global", "body"),
                        ("global", "activity"),
                        ("held", "primary"),
                        ("recent", "primary"),
                    ]:
                        factor = 1.10 if task == "global" and metric != "primary" else 1.05
                        check(
                            f"{task}/{metric} vs {ref}",
                            bank[a]["reconstruction"]["errors"][task][metric],
                            bank[ref]["reconstruction"]["errors"][task][metric],
                            factor,
                        )
                    readable = all(
                        bank[n]["utility"]["scores"]["targets"][i]["r2"] is not None
                        and bank[n]["utility"]["scores"]["targets"][i]["r2"] >= 0.5
                        for n in (a, ref)
                        for i in (0, 2, 3, 5)
                    )
                    for task, factor in [
                        ("utility", 1.05),
                        ("direction", 1.10),
                        ("volatility", 1.10),
                    ]:
                        check(
                            f"{task} vs {ref}",
                            bank[a]["utility"]["errors"][task],
                            bank[ref]["utility"]["errors"][task],
                            factor,
                            readable,
                        )
                    for d in ("1", "16", "64"):
                        av, bv = bank[a]["overlap"][d], bank[ref]["overlap"][d]
                        for metric in (
                            "combined_gap",
                            "combined_error_a",
                            "combined_error_b",
                            "path_shape_gap_bps",
                            "path_native_mae_a_bps",
                            "path_native_mae_b_bps",
                        ):
                            ids = np.flatnonzero(
                                np.array(av["valid"][metric]) & np.array(bv["valid"][metric])
                            )
                            cohort = [overlap_rows[split][i] for i in ids]
                            check(
                                f"overlap{d}/{metric} vs {ref}",
                                np.array(av["per_pair"][metric])[ids],
                                np.array(bv["per_pair"][metric])[ids],
                                1.05,
                                cohort=cohort,
                            )
                for i, name in enumerate(up.NAMES[6:], 6):
                    ids = np.flatnonzero(masks[split][:, i])
                    r2 = bank[a]["utility"]["scores"]["targets"][i]["r2"]
                    check(
                        name,
                        np.array(bank[a]["utility"]["errors"][name])[ids],
                        np.full(len(ids), 0.1),
                        extra=r2 is not None and r2 >= 0.8,
                        cohort=[rows[split][j] for j in ids],
                    )
    groups = {}
    for label, predicate in [
        (
            "cross_period_gain",
            lambda c: c["metric"].startswith("cross/") and c.get("factor") == 0.90,
        ),
        (
            "cross_period_retention",
            lambda c: c["metric"].startswith("cross/") and c.get("factor") != 0.90,
        ),
        ("original_capabilities", lambda c: not c["metric"].startswith("cross/")),
    ]:
        subset = [c for c in checks if predicate(c)]
        groups[label] = bool(subset) and all(c["passed"] for c in subset)
    passed = bool(checks) and all(c["passed"] for c in checks)
    return {
        "status": "cross_period_stability_candidate" if passed else "no_cross_period_upgrade",
        "checks": checks,
        "groups": groups,
        "automatic_promotion": False,
        "scope": "Both seeds, reused original/cross-symbol research, best AND last; weekly paired intervals. Fixed100 epochs, not convergence proof. Gap reduction alone is insufficient: true-history accuracy and original capability protection also required.",
    }


@torch.inference_mode()
def evaluate(meta, out, device):
    check_models(out)
    check_readouts(out)
    cm = run.parent_meta(meta)
    stats, local = run.statistics(meta)
    source = Path(meta["source"])
    original_reader = Path(cm["reader"]["root"])
    om = cr.odr.make_manifest(
        Path(cm["source"]),
        cm["identity"],
        {s: cm["packed"][s] for s in cr.odr.SPLITS},
        meta["evaluation_batch"],
    )
    paired, _ = cr.odr.prepare(om, out)
    fitted = read_json(out / "fit.json")
    parentfit = read_json(source / "fit.json")
    originalfit = read_json(original_reader / "fit.json")
    atomic_json(
        {"reconstruction": stats, "local": local, "utility_targets": fitted["target_stats"]},
        out / "evaluation_scales.json",
    )

    def arrays(path):
        with np.load(path, allow_pickle=False) as f:
            return dict(f)

    heads = arrays(out / "heads.npz")
    pheads = arrays(source / "heads.npz")
    oheads = arrays(original_reader / "heads.npz")
    pca = arrays(original_reader / "pca.npz")["components"]
    records = {}
    inventories = {}
    masks = {}
    absolute = {}
    reader_rows = {}
    cross_rows = {}
    all_entries = [(f"parent_s{s}", None, "best", s) for s in (42, 43)] + [
        (n, j, k, j["seed"]) for n, j, k in entries(meta)
    ]
    for split in ("test", "cross_research"):
        old.verify_cache(meta, out, split)
        bank = old.arrays(out, split)
        rows = read_json(out / f"{split}_inventory.json")
        inventories[split] = rows
        masks[split] = bank["mask"]
        records[split] = {}
        scores = {}
        errors = {}
        predictions = {}
        groups = {}
        data = ur.bb.load_data(cr.alignment(cm), split)
        cross_data = xd.evaluation_data(meta, out, split)
        cross_rows[split] = cross_data[0]

        def score_reader(
            name,
            pred,
            *,
            bank=bank,
            scores=scores,
            errors=errors,
            predictions=predictions,
            groups=groups,
            rows=rows,
        ):
            score, err = up.measure(pred, bank["targets"], bank["mask"], fitted["target_stats"])
            scores[name] = score
            errors[name] = err
            predictions[name] = pred.tolist()
            groups[name] = {}
            for field in ("symbol", "period"):
                for value in sorted({str(r[field]) for r in rows}):
                    ids = np.array([str(r[field]) == value for r in rows])
                    groups[name][field + "/" + value] = up.measure(
                        pred[ids], bank["targets"][ids], bank["mask"][ids], fitted["target_stats"]
                    )[0]
            return {
                "scores": score,
                "errors": {k: v.tolist() for k, v in err.items()},
                "predictions": pred.tolist(),
            }

        for name, job, kind, seed in all_entries:
            if job is None:
                model, _ = run.ge.load(
                    meta["identity"]["manifest"],
                    source,
                    run.parent_job(meta, {"seed": seed}),
                    "best",
                    device,
                )
                hn = f"deep_s{seed}_best"
                fit = parentfit
                weights = pheads
            else:
                model, _ = load(meta, out, job, kind, device)
                hn = name
                fit = fitted
                weights = heads
            model.eval().requires_grad_(False)
            before = ur.bb.state_signature(model)
            recon, err, decomp, pred = ur.pf.score(
                model, data, stats, local, meta["evaluation_batch"], device
            )
            reader_pred = up.predict(
                fit["heads"][hn]["targets"],
                weights[hn + "_weights"],
                weights[hn + "_intercepts"],
                bank[name],
                fit["target_stats"],
            )
            utility = score_reader(name, reader_pred)
            record = {
                "cross_period": cross_scores(
                    model, cross_data, stats, meta["evaluation_batch"], device
                ),
                "reconstruction": {
                    "scores": recon,
                    "errors": {t: {k: v.tolist() for k, v in e.items()} for t, e in err.items()},
                    "path_decomposition": {k: v.tolist() for k, v in decomp.items()},
                },
                "utility": utility,
                "overlap": ce.overlap(
                    model, paired[split], stats, meta["evaluation_batch"], device
                ),
                "examples": [
                    {
                        "index": int(i),
                        "prediction": pred[i].tolist(),
                        "target": data["y"][i].tolist(),
                        "mask": data["mask"][i].tolist(),
                    }
                    for i in np.linspace(0, len(rows) - 1, 4, dtype=int)
                ],
            }
            if job is None:
                oldrecon = read_json(source / f"{split}_deep_s{seed}_best.json")
                ce.require_nested(
                    recon, oldrecon["reconstruction"]["scores"], "frozen500 reconstruction replay"
                )
                ce.require_nested(
                    utility["scores"],
                    read_json(source / "readout_metrics.json")["datasets"][split]["scores"][hn],
                    "parent calibrated utility replay",
                )
                for d in ("1", "16", "64"):
                    ce.require_nested(
                        record["overlap"][d]["summary"],
                        oldrecon["overlap"][d]["summary"],
                        "parent stability replay",
                    )
            if before != ur.bb.state_signature(model):
                raise ValueError("Scoring mutated weights")
            atomic_json(record, out / f"{split}_{name}.json")
            records[split][name] = record
            del model
            progress(
                f"Evaluated {split}/{name}: reconstruction, stability and recalibrated utility"
            )
        for name in rr.BASELINES:
            x = old.representation(name, bank, pca, originalfit["raw_stats"])
            score_reader(
                name,
                up.predict(
                    originalfit["heads"][name]["targets"],
                    oheads[name + "_weights"],
                    oheads[name + "_intercepts"],
                    x,
                    originalfit["target_stats"],
                ),
            )
            ce.require_nested(
                scores[name],
                read_json(source / "readout_metrics.json")["datasets"][split]["scores"][name],
                name + "/baseline replay",
            )
        score_reader("train_mean", np.tile(fitted["target_stats"]["mean"], (len(rows), 1)))
        absolute[split] = ur.decide(
            {
                "models": [n for n, _, _ in entries(meta)],
                "pca_rank": 768,
                "decision": cm["reader"]["manifest"]["decision"],
            },
            scores,
            errors,
            bank["mask"],
            rows,
        )
        reader_rows[split] = {"scores": scores, "groups": groups}
        atomic_json(
            {
                "target_names": list(up.NAMES),
                "targets": bank["targets"].tolist(),
                "mask": bank["mask"].tolist(),
                "predictions": predictions,
                "per_window_errors": {
                    n: {k: v.tolist() for k, v in e.items()} for n, e in errors.items()
                },
            },
            out / f"{split}_predictions.json",
        )
    decision = decide(
        meta, records, inventories, {s: paired[s]["rows"] for s in paired}, masks, cross_rows
    )
    decision["original_utility_protocol"] = absolute
    atomic_json(decision, out / "decision.json")
    atomic_json({"datasets": reader_rows, "decision": decision}, out / "readout_metrics.json")
    summary = {
        s: {
            n: {
                "global_primary": v["reconstruction"]["scores"]["global"]["metrics"]["primary"],
                "utility": v["utility"]["scores"]["groups"]["utility"],
                "gap1": v["overlap"]["1"]["summary"]["all"]["combined_gap"]["mean"],
                "cross_period": v["cross_period"]["groups"],
            }
            for n, v in bank.items()
        }
        for s, bank in records.items()
    }
    atomic_json({"summary": summary, "decision": decision}, out / "cross_period_metrics.json")
    lines = [
        "# Fixed768 cross-period common-history continuation",
        "",
        f"Decision: {decision['status']}; no automatic promotion.",
        "",
        "| dataset/model | global primary | calibrated utility | overlap gap1 |",
        "|---|---:|---:|---:|",
    ]
    for s, bank in summary.items():
        for n, v in bank.items():
            lines.append(
                f"| {s}/{n} | {v['global_primary']:.6f} | {v['utility']:.6f} | {v['gap1']:.6f} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
