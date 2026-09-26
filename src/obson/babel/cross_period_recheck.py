"""Evaluation-only recovery with source-batch replay; immutable training inputs.

The scoring loop is versioned from cross_period_evaluate.evaluate. Objective,
readers, cohorts, scoring batch and decision functions remain bound to that module.
Only parent overlap replay gains a source-batch check and numerical diagnostics.
"""

import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from . import cross_period_evaluate as ev
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

run, xd, rr, ur, cr, ce, up, old = ev.run, ev.xd, ev.rr, ev.ur, ev.cr, ev.ce, ev.up, ev.old
entries, check_models, check_readouts = ev.entries, ev.check_models, ev.check_readouts


def mismatches(actual, expected, path="", atol=1e-6, rtol=2e-5):
    """Report ALL differences using the original summary replay tolerances."""
    result = []

    def walk(a, b, key):
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                result.append({"path": key, "reason": "keys differ"})
                return
            for k in b:
                walk(a[k], b[k], key + "/" + k)
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                result.append({"path": key, "reason": "length differs"})
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, key + "/" + str(i))
        elif b is None or isinstance(b, (str, bool, int)):
            if type(a) is not type(b) or a != b:
                result.append({"path": key, "reason": "value differs", "actual": a, "expected": b})
        elif not np.isclose(a, b, atol=atol, rtol=rtol):
            result.append(
                {
                    "path": key,
                    "reason": "numerical replay differs",
                    "actual": float(a),
                    "expected": float(b),
                    "abs_diff": float(abs(a - b)),
                    "allowed": float(atol + rtol * abs(b)),
                }
            )

    walk(actual, expected, path)
    return result


def array_replay(actual, expected):
    # Existing same-window batch control in overlap_diagnostic_run.evaluate_cell.
    atol, rtol = 5e-5, 2e-4
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape mismatch"}
    diff = np.abs(actual.astype(float) - expected.astype(float))
    failed = ~np.isclose(actual, expected, atol=atol, rtol=rtol)
    return {
        "passed": bool(not failed.any()),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "failed_elements": int(failed.sum()),
        "elements": int(diff.size),
    }


def parent_overlap_replay(
    model, paired, stats, current, expected, batch, reference_batch, device, path
):
    def summary(v):
        return {d: v[d]["summary"] for d in ("1", "16", "64")}

    report = {
        "scoring_batch": batch,
        "reference_batch": reference_batch,
        "summary_atol": 1e-6,
        "summary_rtol": 2e-5,
        "current_summary_differences": mismatches(summary(current), summary(expected)),
        "status": "checking",
        "research_scores_changed": False,
    }
    atomic_json(report, path)
    if not report["current_summary_differences"]:
        report["status"] = "original_replay_passed"
        atomic_json(report, path)
        return report
    if batch == reference_batch:
        report["status"] = "failed_same_batch_reference"
        atomic_json(report, path)
        raise ValueError(f"Parent overlap replay differs at source batch; see {path}")
    reference = ce.overlap(model, paired, stats, reference_batch, device)
    report["source_batch_summary_differences"] = mismatches(summary(reference), summary(expected))
    report["batch_controls"] = {}
    atomic_json(report, path)
    # All original paired windows, including every earlier shifted view, not a sample.
    for shift in (0, 1, 16, 64):
        x = paired["views"][shift]["x"]
        za, a = cr.odr.predict(model, x, stats, batch, device)
        zb, b = cr.odr.predict(model, x, stats, reference_batch, device)
        report["batch_controls"][str(shift)] = {
            "state": array_replay(za, zb),
            "normalized_prediction": array_replay(
                (a - stats["y_mean"]) / stats["y_scale"],
                (b - stats["y_mean"]) / stats["y_scale"],
            ),
        }
        atomic_json(report, path)
    passed = not report["source_batch_summary_differences"] and all(
        v["passed"] for row in report["batch_controls"].values() for v in row.values()
    )
    report["status"] = "source_batch_replay_and_batch_controls_passed" if passed else "failed"
    atomic_json(report, path)
    if not passed:
        raise ValueError(f"Parent numerical replay remains unverified; see {path}")
    progress(
        f"Parent reference batch{reference_batch} replay passed; batch{batch} scores retained; {path.name}"
    )


@torch.inference_mode()
def evaluate(meta, out, report_out, device):
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
    paired, _ = cr.odr.prepare(om, report_out)
    fitted = read_json(out / "fit.json")
    parentfit = read_json(source / "fit.json")
    originalfit = read_json(original_reader / "fit.json")
    atomic_json(
        {"reconstruction": stats, "local": local, "utility_targets": fitted["target_stats"]},
        report_out / "evaluation_scales.json",
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
                model, _ = ev.load(meta, out, job, kind, device)
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
                "cross_period": ev.cross_scores(
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
                parent_overlap_replay(
                    model,
                    paired[split],
                    stats,
                    record["overlap"],
                    oldrecon["overlap"],
                    meta["evaluation_batch"],
                    meta["identity"]["manifest"]["evaluation_batch"],
                    device,
                    report_out / f"{split}_{name}_numeric_replay.json",
                )
            if before != ur.bb.state_signature(model):
                raise ValueError("Scoring mutated weights")
            atomic_json(record, report_out / f"{split}_{name}.json")
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
            report_out / f"{split}_predictions.json",
        )
    decision = ev.decide(
        meta, records, inventories, {s: paired[s]["rows"] for s in paired}, masks, cross_rows
    )
    decision["original_utility_protocol"] = absolute
    atomic_json(decision, report_out / "decision.json")
    atomic_json(
        {"datasets": reader_rows, "decision": decision}, report_out / "readout_metrics.json"
    )
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
    atomic_json(
        {"summary": summary, "decision": decision}, report_out / "cross_period_metrics.json"
    )
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
    (report_out / "summary.md").write_text("\n".join(lines) + "\n")


def verify_inputs(source):
    meta = read_json(source / "manifest.json")
    if meta["schema"] != run.SCHEMA or meta["code_sha256"] != run.code_identity():
        raise ValueError("Original cross-period code/configuration changed")
    if run.source_identity(Path(meta["source"])) != meta["identity"]:
        raise ValueError("Original parent lineage changed")
    if (
        xd.audit_identity(Path(meta["raw_audit"]), Path(meta["raw_root"]))
        != meta["raw_audit_identity"]
    ):
        raise ValueError("Raw audit/data changed")
    lock = check_models(source)
    check_readouts(source)
    xd.verify_data(meta, source)
    if meta["epochs"] != 100 or len(meta["experiments"]) != 4:
        raise ValueError("Four complete100-epoch arms required")
    n = read_json(source / "train_cross_plan.json")["eligible"]
    for job in meta["experiments"]:
        cell = source / job["name"]
        done = read_json(cell / "completion.json")
        summary = read_json(cell / "training_summary.json")
        history = read_json(cell / "history.json")
        if done["status"] != "complete" or done["metadata"] != {"manifest": meta, "job": job}:
            raise ValueError("Training incomplete or metadata changed")
        verify_files(cell, done["files"])
        if lock["trials"][job["name"]] != summary or summary["epochs"] != 100:
            raise ValueError("Locked training summary changed")
        run.verify_history(
            meta,
            job,
            {
                "epoch": 100,
                "history": history,
                "initial_validation": read_json(cell / "initial_validation.json")["validation"],
                "best_epoch": summary["selected_epoch"],
                "best_validation": summary["validation"],
            },
            n,
        )
        for key in ("pairs", "views", "encoder_steps", "head_steps"):
            if summary[key] != sum(h[key] for h in history):
                raise ValueError("Training budget changed")
    for split in old.SPLITS:
        old.verify_cache(meta, source, split)
        if read_json(source / f"{split}_inventory.json") != read_json(
            Path(meta["source"]) / f"{split}_inventory.json"
        ):
            raise ValueError("Original inventory changed")
    if read_json(source / "fit.json")["candidates"] != 520:
        raise ValueError("All locked readouts required; this command never fits readers")
    files = {
        str(p.relative_to(source)): sha256(p)
        for p in source.rglob("*")
        if p.is_file() and p.suffix != ".log"
    }
    return meta, files


def recheck(source, out, device="cuda"):
    source, out = source.resolve(), out.resolve()
    run.check_output(source, out)
    progress(
        "Verifying completed four-arm training, locked readers and immutable data; evaluation only"
    )
    meta, files = verify_inputs(source)
    run.check_output(Path(meta["source"]), out, meta["identity"])
    for key in ("raw_root", "raw_audit"):
        run.check_output(Path(meta[key]), out)
    identity = {
        "schema": "babel-cross-period-evaluation-recheck-v1",
        "source": str(source),
        "source_files": files,
        "original_manifest": meta,
        "evaluator_sha256": sha256(__file__),
        "training_updates": 0,
        "reader_fits": 0,
        "research_scoring_batch": meta["evaluation_batch"],
        "parent_reference_batch": meta["identity"]["manifest"]["evaluation_batch"],
        "policy": "Original summary tolerance retained. Source-batch replay plus existing same-window FP32 control; no replacement of research scores or selection changes.",
    }
    if (out / "manifest.json").exists():
        if read_json(out / "manifest.json") != identity:
            raise ValueError("Evaluation source/code changed; new output directory required")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Empty evaluation output required")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(identity, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        if done["status"] != "complete":
            raise ValueError("Invalid evaluation completion")
        verify_files(out, done["files"])
        progress("Evaluation already complete")
        return
    started = time.time()
    try:
        evaluate(meta, source, out, device)
        _, after = verify_inputs(source)
        if after != files:
            raise ValueError("Evaluation changed original inputs")
    except Exception as exc:
        atomic_json(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "training_updates": 0,
                "reader_fits": 0,
            },
            out / "failure.json",
        )
        raise
    atomic_json(
        {
            "seconds": time.time() - started,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name()
            if str(device).startswith("cuda")
            else "synthetic CPU",
            "training_updates": 0,
            "reader_fits": 0,
        },
        out / "runtime.json",
    )
    artifacts = {
        str(p.relative_to(out)): sha256(p)
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".md")
        and p.name not in ("completion.json", "failure.json")
    }
    atomic_json(
        {"status": "complete", "source_unchanged": True, "files": artifacts},
        out / "completion.json",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("checkpoints/babel_cross_period768_v2"))
    parser.add_argument(
        "--out", type=Path, default=Path("checkpoints/babel_cross_period768_v2_recheck")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("Real-model evaluation requires AutoDL CUDA")
    ur.bb.ab.configure_runtime()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / f".{args.out.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Evaluation output already running") from None
        recheck(args.source, args.out)


if __name__ == "__main__":
    main()
