"""Bounded F05/F01/F11 input structure × initialization experiment.

Eight scratch workers share Q, exposure and optimizer. Two previously completed
bar models are frozen historical anchors, loaded only after the matrix is locked.
Historical execution modules are unchanged; only explicit pure/transaction
interfaces are reused. Real weights/market forward execution require AutoDL CUDA.
"""

import argparse
import fcntl
import gc
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch

from . import history_query as hq
from . import input_factorial as model
from . import input_factorial_evaluate as evaluation
from . import input_factorial_source as source
from . import input_factorial_training as training
from . import input_patch_run as previous
from . import multiscale_content as content
from . import multiscale_data as data
from . import multiscale_evaluate as scoring
from . import observed_history_export_v2 as score_schema
from . import observed_history_run as transactions
from . import recovery_audit as runtime
from .ae_extend import atomic_json, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .observed_history_run import (
    _cpu,
    commit_checkpoint,
    immutable_json,
    publish_history,
    recover_checkpoint,
    verify_files,
)
from .progress import progress

SCHEMA = "babel-input-factorial768-v1"
MODULE = "obson.babel.input_factorial_run"
EPOCHS, BATCH, MICRO = 120, 128, 8
SESSION_CAP = 24 * 3600
DIAGNOSTIC_EPOCHS = (0, 10, 120)
REPO = Path(__file__).resolve().parents[3]
# The checkpoint schema, AdamW recipe, schedule and cumulative24h clock are
# unchanged. These interfaces accept the actual models; no old module globals
# are replaced. Every reused implementation is bound below.
validate_saved_state = previous.validate_saved_state
validate_optimizer = previous.validate_optimizer
reserve_budget = previous.reserve_budget
remaining_seconds = previous.remaining_seconds
worker_deadline = previous.worker_deadline


def jobs():
    return [
        {"name": f"{arm}_s{seed}", "arm": arm, "seed": seed}
        for seed in (42, 43)
        for arm in (model.ARMS if seed == 42 else tuple(reversed(model.ARMS)))
    ]


def evaluation_jobs():
    return jobs() + [
        {"name": f"reference_bar_s{seed}", "arm": "reference_bar", "seed": seed, "reference": True}
        for seed in (42, 43)
    ]


def model_contract(arm=None, config=None):
    if arm == "reference_bar":
        return previous.model.contract("bar", config)
    return (
        model.contract(arm, config) if arm else {a: model.contract(a, config) for a in model.ARMS}
    )


def protocol():
    return {
        "task_ids": ["F05", "F01", "F11"],
        "jobs": jobs(),
        "evaluation_jobs": evaluation_jobs(),
        "variables": "2x2 input organization dense/hybrid and context initialization random/endpoint; no staged training",
        "model_contracts": model_contract(),
        "epochs": EPOCHS,
        "batch": BATCH,
        "micro": MICRO,
        "workers": 1,
        "session_cap_seconds": SESSION_CAP,
        "source": {
            "manifest": source.MANIFEST_SHA256,
            "completion": source.COMPLETION_SHA256,
            "bar_checkpoints": {str(k): v for k, v in source.CHECKPOINT_SHA256.items()},
        },
        "data": {
            "source_completion": data.COMPLETION,
            "counts": dict(data.COUNTS),
            "context": "Each sampled endpoint has its own observed trailing128 from original full255",
            "price_labels": "original float64 closes",
            "statistics": "unchanged train-only source statistics",
        },
        "initialization": "scratch; common tensors identical; same parameter count; only prescribed context initialization differs",
        "training": {
            "objective": "unchanged original Q",
            "schedule": "original epoch1..120 uniform5",
            "optimizer": "fresh single-group AdamW beta0.9/0.999 eps1e-8 wd0.01 peak3e-4 warmup10 clip1",
            "freeze": False,
            "source_optimizer_loaded": False,
            "source_weights_loaded": False,
        },
        "runtime": runtime.RUNTIME_PROFILES["context"],
        "validation": {
            "positions": list(model.VAL_POSITIONS),
            "epochs": [0, *range(5, EPOCHS + 1, 5)],
            "selection": "fixed_last120_no_best_search",
        },
        "research": {
            "positions": list(range(1, 129)),
            "sample_positions": list(evaluation.SAMPLE_POSITIONS),
            "cohorts": ["test", "cross_research"],
            "models": 10,
            "banks": 20,
            "metric_contract": evaluation.metric_contract(),
            "reference_use": "historical_frozen_anchor_not_concurrent_factorial_control",
            "reference_checkpoint_scope": "Complete frozen checkpoints include Adam tensors for integrity validation; no source Adam or RNG is restored into new training",
        },
        "diagnostics": {
            "epochs": list(DIAGNOSTIC_EPOCHS),
            "parents": "first8_locked_training_rows",
            "positions": "seed-matched epoch1 uniform5",
            "updates": 0,
            "fits": 0,
            "preflight": "one additional zero-update gradient panel per actual arm for peak/time qualification",
        },
        "budget": {
            "worker_epochs": 960,
            "updates": 36480,
            "parent_exposures": 4597440,
            "state_exposures": 22987200,
            "equal_FLOPs": False,
            "readout_fits": 0,
        },
        "stopping": "all fixed120 or cumulative24h; insufficient resources fail closed; no partial-arm selection or automatic extension",
        "acceptance": "predeclared complete-matrix factorial comparisons; five-percent gain and ten-percent content/P95 protections; no automatic promotion",
    }


def implementation():
    from . import input_factorial_export as exporter

    # Bind the complete old data/transaction/runtime dependency closure too.
    inherited = previous.implementation()
    modules = (
        model,
        training,
        evaluation,
        exporter,
        source,
        previous,
        data,
        content,
        transactions,
        runtime,
        hq,
        scoring,
        score_schema,
    )
    return {
        "inherited": inherited,
        "files": {
            **{Path(m.__file__).name: sha256(m.__file__) for m in modules},
            Path(__file__).name: sha256(__file__),
            "babel_input_factorial768_autodl.sh": sha256(
                REPO / "scripts/babel_input_factorial768_autodl.sh"
            ),
        },
    }


def binding(out, job):
    return {
        "manifest": sha256(out / "manifest.json"),
        "job": job,
        "model_contract": model_contract(job["arm"], read_json(out / "manifest.json")["encoder"]),
        "initialization": sha256(out / f"initialization_s{job['seed']}.json"),
    }


def prepare_initializations(meta, out, device):
    for seed in (42, 43):
        value = model.initialization_receipt(seed, device, meta["encoder"])
        if not value.get("passed") or not all(value.get("checks", {}).values()):
            raise ValueError("Matched scratch initialization/endpoint parity failed")
        immutable_json(value, out / f"initialization_s{seed}.json")
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def expected_initial(out, job):
    value = read_json(out / f"initialization_s{job['seed']}.json")
    if not value.get("passed") or not all(value.get("checks", {}).values()):
        raise ValueError("Invalid initialization qualification")
    return value["arms"][job["arm"]]["signature"]


def history_check(history, job, n, batch=None, *, config=None):
    batch = BATCH if batch is None else batch
    if len(history) > EPOCHS:
        raise ValueError("History exceeds fixed epoch budget")
    for epoch, item in enumerate(history, 1):
        order, ps = training.schedule(n, job["seed"], epoch)
        expected = {
            "arm": job["arm"],
            "epoch": epoch,
            "steps": math.ceil(n / batch),
            "examples": n,
            "supervised_states": n * 5,
            "lr": training.learning_rate(epoch),
            "batch_size": batch,
            "micro_size": MICRO,
            "order_hash": training.array_hash(order),
            "prefix_hash": training.array_hash(ps),
            "position_counts": np.bincount(ps.ravel() - 1, minlength=128).tolist(),
            "model_contract": model.contract(job["arm"], config),
        }
        if any(item.get(k) != v for k, v in expected.items()):
            raise ValueError(f"History recipe/exposure mismatch at epoch{epoch}")
        if bool(item.get("validation") is not None) != (epoch % 5 == 0 or epoch == EPOCHS):
            raise ValueError("Validation cadence changed")
        json.dumps(item, allow_nan=False)
    return len(history) * math.ceil(n / batch)


def load_pair(meta, job, device):
    module, arm = (previous.model, "bar") if job.get("reference") else (model, job["arm"])
    pair = module.make_models(arm, job["seed"], device, meta["encoder"])
    if module.model_contract(pair[0]) != model_contract(job["arm"], meta["encoder"]):
        raise ValueError("Constructed model differs from immutable contract")
    return pair


@torch.no_grad()
def validation(meta, encoder, query, x, raw, arm, device):
    if model.model_contract(encoder)["arm"] != arm:
        raise ValueError("Validation arm differs from actual encoder")
    encoder.eval()
    query.eval()
    totals, physical, contexts = {}, {}, 0
    for start in range(0, len(x), MICRO):
        features = torch.as_tensor(x[start : start + MICRO], device=device)
        closes = torch.as_tensor(raw[start : start + MICRO], device=device)
        for left in range(0, len(model.VAL_POSITIONS), 5):
            ps = torch.tensor(model.VAL_POSITIONS[left : left + 5], device=device).expand(
                len(features), -1
            )
            _, prediction = model.predict(encoder, query, features, ps)
            target, mask = model.targets(features, ps, meta["statistics"])
            target = content.replace_price_targets(target, closes, ps, meta["statistics"])
            terms = hq.metrics(prediction, target, mask, meta["statistics"], smooth=True)
            terms = terms | {"objective": terms["primary"], "query": terms["primary"]}
            scores = scoring.score_rows(
                prediction.cpu().numpy(),
                target.cpu().numpy(),
                mask.cpu().numpy(),
                raw[start : start + MICRO],
                meta["statistics"],
                ps[0].cpu().numpy(),
            )
            for key, value in scores["errors"].items():
                if key.startswith("price/"):
                    physical[key] = physical.get(key, 0.0) + float(value.sum())
            weight = len(features) * ps.shape[1]
            for key, value in content.detached_summary(terms).items():
                totals[key] = totals.get(key, 0.0) + np.asarray(value) * weight
            contexts += weight
    result = {key: (value / contexts).tolist() for key, value in totals.items()}
    json.dumps(result, allow_nan=False)
    return {
        "positions": list(model.VAL_POSITIONS),
        "parents": len(x),
        "contexts": contexts,
        "components": result,
        "physical_price_mse": {k.removeprefix("price/"): v / contexts for k, v in physical.items()},
        "nonprice_scope": "body/activity components remain original Q, not physical price MSE",
    }


def diagnostic(meta, out, job, encoder, query, x, raw, epoch, device):
    """Bound zero-update diagnostics. Never silently repeat an interrupted one."""
    folder = out / job["name"]
    path = folder / f"gradient_diagnostic_epoch{epoch}.json"
    pending = folder / f"gradient_diagnostic_epoch{epoch}.pending.json"
    bind = {
        "binding": binding(out, job),
        "epoch": epoch,
        "signature": training.parameter_signature(encoder, query),
        "x_sha256": training.array_hash(x[:8]),
        "raw_sha256": training.array_hash(raw[:8]),
    }
    if path.exists():
        value = read_json(path)
        if value.get("binding") != bind or not value.get("complete"):
            raise ValueError("Committed gradient diagnostic binding differs")
        if pending.exists():
            if read_json(pending) != bind:
                raise ValueError("Gradient diagnostic reservation differs")
            pending.unlink()
        if value.get("report", {}).get("passed") is not True:
            raise ValueError("Committed gradient diagnostic failed; evidence retained")
        return value
    if pending.exists():
        raise ValueError("Uncommitted gradient diagnostic is export-only")
    atomic_json(bind, pending)
    report = training.gradient_diagnostic(
        encoder, query, x[:8], raw[:8], meta["statistics"], job["seed"], device
    )
    if training.parameter_signature(encoder, query) != bind["signature"]:
        raise ValueError("Gradient diagnostic changed model")
    value = {"binding": bind, "complete": True, "updates": 0, "fits": 0, "report": report}
    atomic_json(value, path)
    pending.unlink()
    if report.get("passed") is not True:
        raise ValueError("Gradient diagnostic failed; numeric evidence saved")
    return value


def _diagnostic_names():
    return tuple(f"gradient_diagnostic_epoch{epoch}.json" for epoch in DIAGNOSTIC_EPOCHS)


def check_job(meta, out, job):
    folder = out / job["name"]
    done = read_json(folder / "completion.json")
    if done.get("binding") != binding(out, job) or done.get("epoch") != EPOCHS:
        raise ValueError("Worker completion is not fixed last120")
    required = {
        "last.pt",
        "checkpoint.json",
        "history.jsonl",
        "training_summary.json",
        "causal_audit.json",
        "final_causal_audit.json",
        *_diagnostic_names(),
    }
    if set(done.get("files", {})) != required:
        raise ValueError("Incomplete fixed worker/gradient evidence")
    verify_files(folder, done["files"])
    if any(folder.glob("*.pending.json")) or any(
        (folder / p).exists() for p in ("pending_epoch.json", "commit_intent.json")
    ):
        raise ValueError("Completed worker has an unresolved transaction")
    summary = read_json(folder / "training_summary.json")
    steps = history_check(summary["history"], job, meta["counts"]["train"], config=meta["encoder"])
    receipt = read_json(folder / "checkpoint.json")
    if (
        len(summary["history"]) != EPOCHS
        or summary["epoch"] != EPOCHS
        or summary["updates"] != steps
        or done["updates"] != steps
        or summary["examples"] != meta["counts"]["train"] * EPOCHS
        or summary["supervised_states"] != meta["counts"]["train"] * EPOCHS * 5
        or any(summary[k] != done[k] for k in ("binding", "signature", "initial_signature"))
        or done["initial_signature"] != expected_initial(out, job)
        or receipt
        != {"epoch": EPOCHS, "sha256": done["files"]["last.pt"], "binding": done["binding"]}
        or [json.loads(v) for v in (folder / "history.jsonl").read_text().splitlines()]
        != summary["history"]
    ):
        raise ValueError("Completed checkpoint/history exposure or identity differs")
    for name in ("causal_audit.json", "final_causal_audit.json"):
        value = read_json(folder / name)
        if not value["passed"] or not all(value["checks"].values()):
            raise ValueError("Completed causal audit failed")
    panels = set()
    order, positions = training.schedule(8, job["seed"], 1)
    for epoch, name in zip(DIAGNOSTIC_EPOCHS, _diagnostic_names(), strict=True):
        value = read_json(folder / name)
        signature = (
            done["initial_signature"]
            if epoch == 0
            else summary["history"][epoch - 1]["parameter_signature"]
        )
        panels.add((value["binding"].get("x_sha256"), value["binding"].get("raw_sha256")))
        report = value.get("report", {})
        if (
            report.get("arm") != job["arm"]
            or report.get("seed") != job["seed"]
            or report.get("parents") != 8
            or report.get("supervised_states") != 40
            or report.get("order_hash") != training.array_hash(order)
            or report.get("prefix_hash") != training.array_hash(positions)
            or report.get("panel") != "first8_locked_train_parents_epoch1_uniform5_sampling_slots"
            or value.get("complete") is not True
            or value.get("report", {}).get("passed") is not True
            or value.get("updates") != 0
            or value.get("fits") != 0
            or value["binding"]["binding"] != done["binding"]
            or value["binding"]["epoch"] != epoch
            or value["binding"]["signature"] != signature
        ):
            raise ValueError("Completed gradient diagnostic identity differs")
    if len(panels) != 1 or any(
        not isinstance(h, str) or len(h) != 64 for pair in panels for h in pair
    ):
        raise ValueError("Fixed gradient training panel changed between epochs")
    return done


def train_job(meta, out, job, x, raw, val, val_raw, device):
    folder = data.safe_path(out / job["name"])
    folder.mkdir(exist_ok=True)
    if (folder / "completion.json").exists():
        return check_job(meta, out, job)
    encoder, query = load_pair(meta, job, device)
    initial = training.parameter_signature(encoder, query)
    if initial != expected_initial(out, job):
        raise ValueError("Scratch initialization differs from qualification")
    opt, bind = training.new_optimizer(encoder, query), binding(out, job)
    ck = recover_checkpoint(folder, bind)
    if ck is None:
        if any(folder.iterdir()):
            raise ValueError("Unbound worker output; do not overwrite")
        audit = model.audit(
            encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"]
        )
        atomic_json(audit, folder / "causal_audit.json")
        if not audit["passed"]:
            raise ValueError("Scratch causal audit failed")
        ck = {
            "binding": bind,
            "epoch": 0,
            "history": [],
            "initial_signature": initial,
            "initial_validation": validation(
                meta, encoder, query, val, val_raw, job["arm"], device
            ),
            "encoder": _cpu(encoder),
            "query": _cpu(query),
            "optimizer": opt.state_dict(),
            "rng": rng_state(),
            "signature": initial,
        }
        commit_checkpoint(folder, ck)
    else:
        steps = history_check(ck["history"], job, len(x), config=meta["encoder"])
        if ck["initial_signature"] != initial or ck["epoch"] != len(ck["history"]):
            raise ValueError("Resume initialization/epoch differs")
        validate_saved_state(ck, encoder, query, ck["epoch"], steps)
        encoder.load_state_dict(ck["encoder"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
        if training.parameter_signature(encoder, query) != ck["signature"]:
            raise ValueError("Resume parameter signature differs")
        opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
    history, epoch, baseline = ck["history"], ck["epoch"], ck["initial_validation"]
    validate_optimizer(
        opt, encoder, query, epoch, history_check(history, job, len(x), config=meta["encoder"])
    )
    publish_history(folder, history)
    del ck
    if epoch in DIAGNOSTIC_EPOCHS:
        diagnostic(meta, out, job, encoder, query, x, raw, epoch, device)
    for current in range(epoch + 1, EPOCHS + 1):
        pending = {"binding": bind, "epoch": current, "observed_steps": 0}
        atomic_json(pending, folder / "pending_epoch.json")

        def on_step(receipt, pending=pending):
            pending.update(observed_steps=receipt["step"], last_update=receipt)
            atomic_json(pending, folder / "pending_epoch.json")

        report = training.train_epoch(
            encoder,
            query,
            x,
            raw,
            meta["statistics"],
            opt,
            arm=job["arm"],
            seed=job["seed"],
            epoch=current,
            device=device,
            batch_size=BATCH,
            micro_size=MICRO,
            on_step=on_step,
        )
        report["validation"] = (
            validation(meta, encoder, query, val, val_raw, job["arm"], device)
            if current % 5 == 0 or current == EPOCHS
            else None
        )
        report["parameter_signature"] = training.parameter_signature(encoder, query)
        history.append(report)
        steps = history_check(history, job, len(x), config=meta["encoder"])
        validate_optimizer(opt, encoder, query, current, steps)
        state = {
            "binding": bind,
            "epoch": current,
            "history": history,
            "initial_signature": initial,
            "initial_validation": baseline,
            "encoder": _cpu(encoder),
            "query": _cpu(query),
            "optimizer": opt.state_dict(),
            "rng": rng_state(),
            "signature": report["parameter_signature"],
        }
        commit_checkpoint(folder, state)
        del state
        publish_history(folder, history)
        (folder / "pending_epoch.json").unlink()
        if current in DIAGNOSTIC_EPOCHS:
            diagnostic(meta, out, job, encoder, query, x, raw, current, device)
        progress(
            f"{job['name']}: {current}/{EPOCHS}, Q={report['loss']:.6f}, {report['seconds']:.1f}s; fixed last"
        )
    audit = model.audit(encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"])
    atomic_json(audit, folder / "final_causal_audit.json")
    if not audit["passed"]:
        raise ValueError("Trained causal audit failed")
    summary = {
        "binding": bind,
        "epoch": EPOCHS,
        "initial_signature": initial,
        "signature": training.parameter_signature(encoder, query),
        "history": history,
        "initial_validation": baseline,
        "updates": steps,
        "examples": len(x) * EPOCHS,
        "supervised_states": len(x) * EPOCHS * 5,
    }
    atomic_json(summary, folder / "training_summary.json")
    names = (
        "last.pt",
        "checkpoint.json",
        "history.jsonl",
        "training_summary.json",
        "causal_audit.json",
        "final_causal_audit.json",
        *_diagnostic_names(),
    )
    files = {name: sha256(folder / name) for name in names}
    done = {
        k: summary[k] for k in ("binding", "epoch", "initial_signature", "signature", "updates")
    }
    done.update(
        files=files,
        file=f"{job['name']}/last.pt",
        sha256=files["last.pt"],
        arm=job["arm"],
        model_contract=model.contract(job["arm"], meta["encoder"]),
        initialization=bind["initialization"],
    )
    atomic_json(done, folder / "completion.json")
    del encoder, query, opt
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def reference_records(meta):
    return {
        f"reference_bar_s{seed}": {
            "reference": True,
            "seed": seed,
            "arm": "reference_bar",
            "epoch": EPOCHS,
            "updates": 0,
            "source_updates": entry["updates"],
            "signature": entry["signature"],
            "source_binding": entry["binding"],
            "model_contract": previous.model.contract("bar", meta["encoder"]),
            "source_checkpoint": str(Path(meta["model_source"]) / entry["checkpoint"]),
            "file": str(Path(meta["model_source"]) / entry["checkpoint"]),
            "sha256": entry["checkpoint_sha256"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
        }
        for seed in (42, 43)
        for entry in (meta["source_identity"]["controls"][str(seed)],)
    }


def lock_models(meta, out):
    value = {
        "schema": SCHEMA,
        "selection": "fixed_last120",
        "manifest": sha256(out / "manifest.json"),
        "models": {job["name"]: check_job(meta, out, job) for job in jobs()},
        "references": reference_records(meta),
        "source": meta["source_identity"],
        "readout_fits": 0,
        "initialization": {str(s): sha256(out / f"initialization_s{s}.json") for s in (42, 43)},
    }
    immutable_json(value, out / "model_selection_lock.json")
    return value


def verify_lock(meta, out, name, split):
    if name not in {j["name"] for j in evaluation_jobs()} or split not in (
        "test",
        "cross_research",
    ):
        raise ValueError("Unknown locked model/cohort")
    if (
        read_json(out / "manifest.json") != meta
        or meta["implementation"] != implementation()
        or meta["protocol"] != protocol()
    ):
        raise ValueError("Immutable evaluation context/code changed")
    lock_models(meta, out)
    raw = read_json(out / f"multiscale_raw_{split}.json")
    verify_files(
        out, {f"raw/{split}.npz": raw["raw_sha256"], f"{split}_rows.json": raw["rows_file_sha256"]}
    )
    return {
        "model_lock": sha256(out / "model_selection_lock.json"),
        "source": meta["source_identity"],
        "raw_receipt": sha256(out / f"multiscale_raw_{split}.json"),
    }


def remaining_epochs(out):
    total = 0
    for job in jobs():
        path = out / job["name"] / "checkpoint.json"
        epoch = read_json(path)["epoch"] if path.exists() else 0
        if type(epoch) is not int or not 0 <= epoch <= EPOCHS:
            raise ValueError("Invalid committed epoch")
        total += EPOCHS - epoch
    return total


def disk_requirement(parameter_bytes, research_parents):
    """Uncompressed byte accounting for every declared binary report family.

    The schema owns the22 position and111 parent metrics; parent context counts
    exist for66 regular metrics, while45 F03 metrics carry error/support only.
    Source weights are read in place. No compression benefit is assumed.
    """
    models, cohorts = len(evaluation_jobs()), 2
    banks = models * cohorts
    checkpoint = parameter_bytes * 3 + 8 * 2**20
    full = models * research_parents * 128 * 127 * 7 * 4
    fixed = models * research_parents * len(evaluation.SAMPLE_POSITIONS) * 127 * 7 * (4 + 8 + 1)
    position_arrays = len(score_schema.POSITION_METRICS) * 2  # float64 error + int64 support
    parent_arrays = (
        len(score_schema.REGULAR_METRICS) + len(score_schema.OVERLAP_METRICS)
    ) * 2 + len(score_schema.REGULAR_METRICS)  # plus int64 supported contexts
    position_rows = models * research_parents * 128 * position_arrays * 8
    parent_rows = models * research_parents * parent_arrays * 8
    fixed_raw = models * research_parents * 255 * 8
    fixed_positions = banks * len(evaluation.SAMPLE_POSITIONS) * 8
    # Five arrays per fixed prediction archive;1024 bytes per NPY/ZIP member
    # bounds headers/directory entries without relying on deflation. Each cohort
    # can add one short final shard beyond the combined-parent quotient.
    report_headers = banks * (position_arrays + parent_arrays + 5) * 1024
    full_headers = models * (math.ceil(research_parents / 8) + cohorts - 1) * 1024
    text = 256 * 2**20
    report_parts = {
        "fixed_predictions_targets_masks": fixed,
        "position_error_support": position_rows,
        "parent_error_support_contexts": parent_rows,
        "fixed_raw_closes": fixed_raw,
        "fixed_position_indexes": fixed_positions,
        "array_container_headers": report_headers,
        "text_and_json_allowance": text,
    }
    reports = sum(report_parts.values())
    raw_cache = sum(data.COUNTS.values()) * 255 * 8 + len(data.COUNTS) * 1024
    # Eight Adam checkpoints + one atomic replacement; report base + private
    # snapshot + new archive + previous-archive backup; raw labels remain remote.
    total = 9 * checkpoint + full + full_headers + raw_cache + 4 * reports + 2 * 2**30
    return {
        "checkpoint_bytes": checkpoint,
        "checkpoint_total_bytes": 9 * checkpoint,
        "full128_predictions_bytes": full,
        "full128_headers_bytes": full_headers,
        "prepared_raw_price_cache_bytes": raw_cache,
        "score_schema": {
            "position_metrics": len(score_schema.POSITION_METRICS),
            "regular_parent_metrics": len(score_schema.REGULAR_METRICS),
            "overlap_parent_metrics": len(score_schema.OVERLAP_METRICS),
            "position_arrays_per_bank": position_arrays,
            "parent_arrays_per_bank": parent_arrays,
            "banks": banks,
        },
        "report_parts_bytes": report_parts,
        "report_payload_estimate_bytes": reports,
        "report_and_export_bytes": 4 * reports,
        "safety_bytes": 2 * 2**30,
        "required_bytes": total,
    }


def allocated_disk_credit(meta, out, budget):
    """Credit only already allocated, receipt-verified artifacts of this run.

    Never count directory size: source/raw/cache/foreign/temporary/archive bytes
    cannot stand in for planned checkpoints, prediction shards or base reports.
    A complete bank is credited only when its frozen matrix and exact inventory
    are verified. Export snapshot/new tar/old tar and safety space stay reserved.
    """
    credited = {"checkpoints": {}, "full_predictions": {}, "base_reports": {}}
    for job in jobs():
        folder = out / job["name"]
        receipt_path = folder / "checkpoint.json"
        if not receipt_path.exists():
            continue
        if any(
            (folder / name).exists()
            for name in ("pending_epoch.json", "commit_intent.json", "last.pt.tmp")
        ):
            raise ValueError("Uncommitted checkpoint cannot receive disk credit")
        receipt = read_json(data.safe_path(receipt_path))
        if (
            receipt.get("binding") != binding(out, job)
            or type(receipt.get("epoch")) is not int
            or not 0 <= receipt["epoch"] <= EPOCHS
        ):
            raise ValueError("Checkpoint disk receipt binding differs")
        verify_files(folder, {"last.pt": receipt["sha256"]})
        relative = f"{job['name']}/last.pt"
        credited["checkpoints"][relative] = min(
            (out / relative).stat().st_size, budget["checkpoint_bytes"]
        )
    lock_path = out / "model_selection_lock.json"
    if lock_path.exists():
        lock = read_json(data.safe_path(lock_path))
        if (
            lock.get("schema") != SCHEMA
            or lock.get("manifest") != sha256(out / "manifest.json")
            or set(lock.get("models", {})) != {j["name"] for j in jobs()}
            or set(lock.get("references", {}))
            != {j["name"] for j in evaluation_jobs() if j.get("reference")}
        ):
            raise ValueError("Research disk credit requires complete current model lock")
        for job in evaluation_jobs():
            name = job["name"]
            selected = lock["references" if job.get("reference") else "models"][name]
            for split in ("test", "cross_research"):
                prefix = f"evaluation/{name}/{split}"
                receipt_path = out / prefix / "complete.json"
                if not receipt_path.exists():
                    continue
                receipt = read_json(data.safe_path(receipt_path))
                bound = receipt.get("binding", {})
                parents = meta["counts"][split]
                if (
                    receipt.get("schema") != evaluation.SCHEMA
                    or receipt.get("status") != "complete"
                    or bound.get("name") != name
                    or bound.get("split") != split
                    or bound.get("parents") != parents
                    or bound.get("model_signature") != selected["signature"]
                    or bound.get("lock", {}).get("model_lock") != sha256(lock_path)
                ):
                    raise ValueError("Completed bank disk receipt binding differs")
                full = {
                    f"predictions/{name}/{split}/parents_{left:05d}_{min(left + 8, parents):05d}.npy"
                    for left in range(0, parents, 8)
                }
                reports = {
                    prefix + "/parent_rows.npz",
                    prefix + "/position_rows.npz",
                    f"results/{split}_{name}_predictions.npz",
                }
                expected = full | reports | {prefix + "/rows.json", prefix + "/metrics.json"}
                if set(receipt.get("files", {})) != expected:
                    raise ValueError("Completed bank disk inventory differs")
                verify_files(out, receipt["files"])
                for family, paths in (("full_predictions", full), ("base_reports", reports)):
                    credited[family].update(
                        {name: (out / name).stat().st_size for name in sorted(paths)}
                    )
    limits = {
        "checkpoints": len(jobs()) * budget["checkpoint_bytes"],
        "full_predictions": budget["full128_predictions_bytes"],
        "base_reports": budget["report_payload_estimate_bytes"],
    }
    totals = {key: min(sum(files.values()), limits[key]) for key, files in credited.items()}
    credit = sum(totals.values())
    return {
        "verified_files": credited,
        "category_limits": limits,
        "credited_by_category": totals,
        "credited_bytes": credit,
        "gross_required_bytes": budget["required_bytes"],
        "remaining_required_bytes": budget["required_bytes"] - credit,
        "excluded": "source, raw/cache, unknown files, uncommitted files, archives; no snapshot/tar/safety credit",
    }


def preflight(meta, out, x, raw, device, evaluate_only=False):
    """Measure every actual arm, no parameter updates; charge all work to clock."""
    features = torch.as_tensor(x[:MICRO], device=device)
    closes = torch.as_tensor(raw[:MICRO], device=device)
    ps = torch.as_tensor(training.schedule(len(x), 42, 1)[1][:MICRO], device=device)
    arms, disk_credit = {}, None
    for arm in model.ARMS:
        progress(f"Zero-update preflight: {arm}; backward/validation9/full128")
        encoder, query = load_pair(meta, {"arm": arm, "seed": 42}, device)
        before = training.parameter_signature(encoder, query)
        params = list(encoder.parameters()) + list(query.parameters())
        bytes_ = sum(p.numel() * p.element_size() for p in params)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for repeat in range(3):
            encoder.train()
            query.train()
            encoder.zero_grad(set_to_none=True)
            query.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            terms = training.training_terms(
                encoder, query, features, closes, ps, meta["statistics"], arm
            )
            objective = terms["objective"].mean()
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite preflight Q")
            objective.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True).detach().cpu()
            )
            torch.cuda.synchronize()
            if repeat:
                durations.append(time.perf_counter() - started)
            del terms, objective
        times = {"micro_backward_seconds": max(durations)}
        encoder.zero_grad(set_to_none=True)
        query.zero_grad(set_to_none=True)
        encoder.eval()
        query.eval()
        with torch.no_grad():
            for label, positions in (
                ("validation9", model.VAL_POSITIONS),
                ("evaluation128", tuple(range(1, 129))),
            ):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for left in range(0, len(positions), 5):
                    points = torch.tensor(positions[left : left + 5], device=device).expand(
                        len(features), -1
                    )
                    training.training_terms(
                        encoder, query, features, closes, points, meta["statistics"], arm
                    )
                torch.cuda.synchronize()
                times[label + "_micro_seconds"] = time.perf_counter() - started
        torch.cuda.synchronize()
        diagnostic_started = time.perf_counter()
        gradient = training.gradient_diagnostic(
            encoder, query, x[:8], raw[:8], meta["statistics"], 42, device
        )
        torch.cuda.synchronize()
        times["gradient_diagnostic_seconds"] = time.perf_counter() - diagnostic_started
        peak = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        if training.parameter_signature(encoder, query) != before:
            raise ValueError("Preflight changed parameters")
        disk = disk_requirement(bytes_, meta["counts"]["test"] + meta["counts"]["cross_research"])
        if disk_credit is None:
            disk_credit = allocated_disk_credit(meta, out, disk)
        disk_free = shutil.disk_usage(out).free
        save_seconds = 0.0
        if disk_free >= disk_credit["remaining_required_bytes"]:
            probe = out / f"preflight_{arm}_save.tmp"
            if probe.exists():
                raise ValueError("Unresolved preflight save probe")
            started = time.perf_counter()
            torch.save({"encoder": _cpu(encoder), "query": _cpu(query)}, probe)
            with probe.open("rb") as handle:
                os.fsync(handle.fileno())
            save_seconds = time.perf_counter() - started
            probe.unlink()
        arms[arm] = {
            "timing": times,
            "parameter_bytes": bytes_,
            "parameter_count": sum(p.numel() for p in params),
            "model_contract": model.model_contract(encoder),
            "preclip_norm": norm,
            "gradient_diagnostic": gradient,
            "weight_save_seconds": save_seconds,
            "peak_cuda_bytes": peak,
            "gpu_required_bytes": (peak + 3 * bytes_) * 1.2 + 512 * 2**20,
            "gpu_available_bytes": free + reserved,
            "gpu_total_bytes": total,
            "disk": disk,
            "disk_credit": disk_credit,
            "disk_free_bytes": disk_free,
            "zero_updates": True,
        }
        del encoder, query, params
        gc.collect()
        torch.cuda.empty_cache()
    epochs_left = remaining_epochs(out)
    if evaluate_only and epochs_left:
        raise ValueError("Evaluate-only requires all eight fixed120 workers complete")

    def maximum(key):
        return max(a["timing"][key] for a in arms.values())

    pending_epochs = {
        j["name"]: EPOCHS
        - (
            read_json(out / j["name"] / "checkpoint.json")["epoch"]
            if (out / j["name"] / "checkpoint.json").exists()
            else 0
        )
        for j in jobs()
    }
    training_seconds = sum(
        pending_epochs[j["name"]]
        * math.ceil(len(x) / MICRO)
        * arms[j["arm"]]["timing"]["micro_backward_seconds"]
        for j in jobs()
    )
    validation_seconds = sum(
        (1 + math.ceil(pending_epochs[j["name"]] / 5))
        * math.ceil(meta["counts"]["val"] / MICRO)
        * arms[j["arm"]]["timing"]["validation9_micro_seconds"]
        for j in jobs()
    )
    # Frozen bar references are conservatively charged at the slowest measured
    # actual arm. New workers use their own measured token/gradient path.
    evaluation_seconds = sum(
        math.ceil(meta["counts"][s] / MICRO) for s in ("test", "cross_research")
    ) * (
        sum(arms[j["arm"]]["timing"]["evaluation128_micro_seconds"] for j in jobs())
        + 2 * maximum("evaluation128_micro_seconds")
    )
    diagnostic_seconds = sum(
        len(DIAGNOSTIC_EPOCHS) * arms[j["arm"]]["timing"]["gradient_diagnostic_seconds"]
        for j in jobs()
    )
    saving_seconds = sum(
        arms[j["arm"]]["weight_save_seconds"] * 4 * (pending_epochs[j["name"]] + 1) for j in jobs()
    )
    estimate = (
        1.3
        * (
            training_seconds
            + validation_seconds
            + evaluation_seconds
            + diagnostic_seconds
            + saving_seconds
        )
        + 1800
    )
    checks = {
        "gradient_diagnostics_pass": all(
            a["gradient_diagnostic"].get("passed") is True for a in arms.values()
        ),
        "equal_parameter_count": len({a["parameter_count"] for a in arms.values()}) == 1,
        "gpu_room": all(a["gpu_required_bytes"] <= a["gpu_available_bytes"] for a in arms.values()),
        "disk_room": all(
            a["disk_credit"]["remaining_required_bytes"] <= a["disk_free_bytes"]
            for a in arms.values()
        ),
        "time_room": estimate <= remaining_seconds(out),
    }
    result = {
        "schema": SCHEMA,
        "checks": checks,
        "passed": all(checks.values()),
        "arms": arms,
        "zero_updates": True,
        "remaining_epochs": epochs_left,
        "estimated_seconds": estimate,
        "remaining_budget_seconds": remaining_seconds(out),
        "runtime": runtime.runtime_state(),
        "time_estimate": {
            "training": training_seconds,
            "validation": validation_seconds,
            "evaluation": evaluation_seconds,
            "gradient_diagnostics": diagnostic_seconds,
            "saving": saving_seconds,
            "multiplicative_margin": 1.3,
            "additional_seconds": 1800,
            "pending_epochs": pending_epochs,
        },
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    directory = out / "preflight"
    directory.mkdir(exist_ok=True)
    atomic_json(
        result, directory / f"attempt_{len(list(directory.glob('attempt_*.json'))) + 1}.json"
    )
    progress(
        f"Preflight estimated {estimate / 3600:.2f}h; disk need {max(a['disk_credit']['remaining_required_bytes'] for a in arms.values()) / 2**30:.2f}GiB; checks={checks}"
    )
    if not result["passed"]:
        raise ValueError(f"Resource/numeric preflight stopped without updates: {checks}")
    return result


def cost_summary(meta, out):
    workers = {}
    for job in jobs():
        summary = read_json(out / job["name"] / "training_summary.json")
        initial = read_json(out / f"initialization_s{job['seed']}.json")["arms"][job["arm"]]
        workers[job["name"]] = {
            "updates": summary["updates"],
            "training_seconds": sum(r["seconds"] for r in summary["history"]),
            "parent_exposures": summary["examples"],
            "state_exposures": summary["supervised_states"],
            "parameter_count": initial["parameter_count"],
            "parameter_bytes": initial["parameter_bytes"],
        }
    return {
        "task_id": "F11",
        "workers": workers,
        "public_state_float32_bytes": 3072,
        "readout_fits": 0,
        "general_reuse": "unknown_not_tested",
        "reference_updates": 0,
        "scope": "Content and measured cost; matched updates are not equal FLOPs; historical bar anchors are not concurrent factorial controls",
    }


def execute(model_source, out, evaluate_only=False):
    with (out / "worker.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not torch.cuda.is_available():
                raise ValueError("Real execution requires AutoDL CUDA")
            remaining_seconds(out)
            progress(
                "Verifying reviewed input-patch bar references and original data; no source weights loaded for training"
            )
            verified = source.verify(model_source, out)
            meta = {
                **verified["verified_data"],
                "schema": SCHEMA,
                "model_source": str(model_source),
                "source": verified["original_meta"]["source"],
                "source_identity": verified["identity"],
                "encoder": dict(model.CONFIG),
                "counts": dict(data.COUNTS),
                "epochs": EPOCHS,
                "batch": BATCH,
                "micro": MICRO,
                "protocol": protocol(),
                "implementation": implementation(),
            }
            if read_json(out / "request.json") != request(model_source, out):
                raise ValueError("Supervisor request/code changed")
            immutable_json(meta, out / "manifest.json")
            immutable_json(
                {
                    "identity": verified["identity"],
                    "source_weights_loaded": False,
                    "optimizer_updates": 0,
                },
                out / "source_qualification.json",
            )
            with runtime.replay_runtime("context"):
                data.prepare(meta, out, ("train", "val"))
                train, _ = data.cache(meta, "train")
                val, _ = data.cache(meta, "val")
                raw, val_raw = data.closes(out, "train"), data.closes(out, "val")
                prepare_initializations(meta, out, "cuda")
                for job in jobs():
                    folder = out / job["name"]
                    if folder.exists():
                        ck = recover_checkpoint(folder, binding(out, job))
                        if ck is not None:
                            history_check(ck["history"], job, len(train), config=meta["encoder"])
                            del ck
                if evaluate_only:
                    lock_models(meta, out)
                preflight(meta, out, train, raw, "cuda", evaluate_only)
                if not evaluate_only:
                    for job in jobs():
                        train_job(meta, out, job, train, raw, val, val_raw, "cuda")
                lock_models(meta, out)
                del train, raw, val, val_raw
                gc.collect()
                progress(
                    "All eight fixed-last models locked; preparing 20 research banks including frozen historical bar references"
                )
                data.prepare(meta, out, ("test",))
                data.prepare(meta, out, ("cross_research",))
                banks, row_bank = {}, {}
                for split in ("test", "cross_research"):
                    x, rows = data.cache(meta, split)
                    close = data.closes(out, split)
                    banks[split], row_bank[split] = {}, rows
                    for job in evaluation_jobs():
                        verify_lock(meta, out, job["name"], split)
                        encoder, query = load_pair(meta, job, "cuda")
                        ck = (
                            source.checkpoint(meta, job["seed"])
                            if job.get("reference")
                            else recover_checkpoint(out / job["name"], binding(out, job))
                        )
                        validator = (
                            source.validate_reference_state
                            if job.get("reference")
                            else validate_saved_state
                        )
                        validator(
                            ck,
                            encoder,
                            query,
                            EPOCHS,
                            EPOCHS * math.ceil(meta["counts"]["train"] / BATCH),
                        )
                        encoder.load_state_dict(ck["encoder"], strict=True)
                        query.load_state_dict(ck["query"], strict=True)
                        if training.parameter_signature(encoder, query) != ck["signature"]:
                            raise ValueError("Selected inference state changed")
                        del ck
                        banks[split][job["name"]] = evaluation.extract(
                            meta,
                            out,
                            job["name"],
                            encoder,
                            query,
                            x,
                            close,
                            rows,
                            split,
                            "cuda",
                            verify_lock=verify_lock,
                        )
                        del encoder, query
                        gc.collect()
                        torch.cuda.empty_cache()
                    del x, close
                atomic_json(cost_summary(meta, out), out / "f11_cost.json")
                decision = evaluation.decisions(banks, row_bank)
                atomic_json(decision, out / "input_factorial_metrics.json")
                immutable_json(
                    {
                        "source_identity": meta["source_identity"],
                        "training_source_weights_loaded": False,
                        "source_optimizer_loaded": False,
                        "optimizer_updates": 0,
                        "reference_weights_loaded": True,
                        "reference_adam_deserialized_for_integrity": True,
                        "source_optimizer_restored_for_training": False,
                        "optimizer_scope": "source_optimizer_loaded=False means no source optimizer restored for training; frozen reference checkpoint Adam tensors are deserialized solely for integrity validation",
                        "reference_use": "historical_frozen_anchor_not_concurrent_factorial_control",
                        "references": reference_records(meta),
                    },
                    out / "source_validation.json",
                )
            if (
                source.verify(model_source, out) != verified
                or implementation() != meta["implementation"]
                or protocol() != meta["protocol"]
            ):
                raise ValueError("Source/data/protocol/implementation changed during execution")
            files = {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and p.suffix in (".json", ".jsonl", ".npy", ".npz", ".pt")
                and p.name not in ("budget.json", "status.json", "completion.json", "failure.json")
            }
            files.update(
                {
                    str(p.relative_to(out)): sha256(p)
                    for p in out.rglob("completion.json")
                    if p.parent != out
                }
            )
            atomic_json(
                {
                    "schema": SCHEMA,
                    "status": "complete_requires_review",
                    "promoted": False,
                    "limited_hypothesis_supported": decision["limited_hypothesis_supported"],
                    "files": files,
                },
                out / "completion.json",
            )
            atomic_json(
                {"schema": SCHEMA, "status": "complete_requires_review", "promoted": False},
                out / "status.json",
            )
            return 0
        except Exception as error:
            failure = {
                "schema": SCHEMA,
                "status": "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            directory = out / "failures"
            directory.mkdir(exist_ok=True)
            atomic_json(failure, directory / f"attempt_{time.time_ns()}.json")
            atomic_json(failure, out / "failure.json")
            atomic_json(
                {"schema": SCHEMA, "status": "failed", "promoted": False}, out / "status.json"
            )
            traceback.print_exc()
            return 1


def request(model_source, out):
    return {
        "schema": SCHEMA,
        "source": str(model_source),
        "output": str(out),
        "protocol": protocol(),
        "implementation": implementation(),
    }


def supervise(model_source, out, evaluate_only=False):
    model_source, out = data.safe_path(model_source), data.safe_path(out)
    if model_source == out or model_source in out.parents or out in model_source.parents:
        raise ValueError("Separate source/output required")
    startup_seconds = 0.0
    if not (out / "request.json").exists():
        started = time.monotonic()
        source.verify(model_source, out)
        startup_seconds = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=True)
    with (out / "session.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out / "request.json").exists() and any(
            p.name != "session.lock" for p in out.iterdir()
        ):
            raise ValueError("Existing output lacks this run request")
        immutable_json(request(model_source, out), out / "request.json")
        if (out / "completion.json").exists():
            verify_files(out, read_json(out / "completion.json")["files"])
            progress("Completed run verified; export only")
            return 0
        budget, used, remaining = reserve_budget(out, startup_seconds)
        command = [
            sys.executable,
            "-m",
            MODULE,
            "--source",
            str(model_source),
            "--out",
            str(out),
            "--worker",
        ]
        if evaluate_only:
            command.append("--evaluate-only")
        started = time.monotonic()
        child = subprocess.Popen(command, start_new_session=True)

        def stop(signum, frame):
            raise InterruptedError(f"signal{signum}")

        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            code = child.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            atomic_json(
                {
                    "schema": SCHEMA,
                    "status": "timeout" if code == 124 else "stopped",
                    "promoted": False,
                },
                out / "status.json",
            )
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        elapsed = min(remaining, time.monotonic() - started)
        budget["used_seconds"] = used + elapsed
        budget["attempts"][-1].update(
            elapsed_seconds=elapsed, total_elapsed_seconds=startup_seconds + elapsed, exit_code=code
        )
        atomic_json(budget, out / "budget.json")
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    model_source, out = data.safe_path(args.source), data.safe_path(args.out)
    if args.worker:
        if read_json(out / "budget.json")["supervisor_pid"] != os.getppid():
            raise ValueError("Worker must be started by budget-owning supervisor")
        with worker_deadline(out):
            raise SystemExit(execute(model_source, out, args.evaluate_only))
    raise SystemExit(supervise(model_source, out, args.evaluate_only))


if __name__ == "__main__":
    main()
