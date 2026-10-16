"""F01/F03 fixed scratch comparison; real execution requires AutoDL CUDA.

Only committed epoch boundaries resume. A pending, uncommitted update is an
uncertain execution and stops the run instead of silently repeating exposure.
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
from contextlib import contextmanager, suppress
from pathlib import Path

import numpy as np
import torch

from . import multiscale_content as content
from . import multiscale_data as data
from . import multiscale_evaluate as evaluation
from . import multiscale_training as training
from . import recovery_audit as runtime
from . import state_coverage as coverage
from .ae_extend import atomic_json, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-multiscale768-v1"
MODULE = "obson.babel.multiscale_run"
EPOCHS = 120
BATCH = 128
MICRO = 8
SESSION_CAP = 36 * 3600
REPO = Path(__file__).resolve().parents[3]


def jobs():
    return [
        {"name": f"{arm}_s{seed}", "arm": arm, "seed": seed}
        for arm, seed in (("control", 42), ("multiscale", 42), ("multiscale", 43), ("control", 43))
    ]


def protocol():
    return {
        "task_ids": ["F01", "F03"],
        "jobs": jobs(),
        "epochs": EPOCHS,
        "batch": BATCH,
        "micro": MICRO,
        "workers": 1,
        "session_cap_seconds": SESSION_CAP,
        "encoder": coverage.CONFIG,
        "source": "reviewed overlap-content data lineage; no source model/Adam loaded",
        "source_completion": data.COMPLETION,
        "counts": data.COUNTS,
        "sampling": "five uniform endpoints without replacement; each sees full128",
        "labels": "raw float64 log-percent prices for both arms; original body/activity masks",
        "scale_fit": "train all4789x128; per-Haar-band RMS; delta_scale[0]*1e-3 floor",
        "objectives": {
            "control": ".4path+.2change1+.1body+.3activity",
            "multiscale": ".6mean8HaarHuber+.1body+.3activity",
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "peak_lr": 3e-4,
            "warmup": 10,
            "clip": 1.0,
        },
        "runtime": runtime.RUNTIME_PROFILES["context"],
        "precision": "FP32 parameters/forward; float64 original price labels and losses; no AMP/TF32",
        "selection": "last120 only; epoch0 and every5 validation diagnostics; zero readout fits",
        "validation_positions": list(coverage.VAL_POSITIONS),
        "evaluation_positions": list(range(1, 129)),
        "prediction_export_positions": list(evaluation.SAMPLE_POSITIONS),
        "acceptance": "both seeds/cohorts: candidate-.95control composite price MSE paired95% upper<=0; all listed band/field/population/overlap protections candidate-1.10control upper<=0; P95 point<=1.10",
        "bootstrap": {
            "draws": 2000,
            "seed": 20261012,
            "groups": ["month", "contract_period_month"],
            "min_parents": 100,
            "min_groups": 6,
        },
        "scope": "reused research data; no Macro promotion, generic transfer or forecasting acceptance",
        "stopping": "fixed budget; source/nonfinite/causality/resource failure stops; no uncertain-update replay",
    }


def implementation():
    # Include the inherited source verifier and model/target code, not mutable docs.
    from . import (
        history_query,
        multiscale_export,
        multiscale_metrics,
        overlap_content_audit,
        uniform_context,
    )

    return {
        "inherited": data.original.diagnostic.implementation(),
        "data": data.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (
                data,
                content,
                training,
                evaluation,
                coverage,
                uniform_context,
                multiscale_metrics,
                multiscale_export,
                runtime,
                history_query,
                overlap_content_audit,
            )
        }
        | {Path(__file__).name: sha256(__file__)},
        "script": sha256(REPO / "scripts/babel_multiscale768_autodl.sh"),
    }


def immutable_json(value, path):
    data.safe_path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable binding changed: {path}")
    else:
        atomic_json(value, path)


def verify_files(root, files):
    for relative, digest in files.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe receipt member")
        path = data.safe_path(root / relative)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Committed artifact changed: {relative}")


def binding(out, job):
    return {
        "manifest": sha256(out / "manifest.json"),
        "scales": sha256(out / "multiscale_scales_receipt.json"),
        "job": job,
    }


def history_check(history, job, n, batch=BATCH):
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
        }
        if any(item.get(k) != v for k, v in expected.items()):
            raise ValueError(f"History recipe/exposure mismatch at epoch{epoch}")
        # Every numeric reported value must serialize as finite JSON.
        json.dumps(item, allow_nan=False)
        if bool(item.get("validation") is not None) != (epoch % 5 == 0 or epoch == EPOCHS):
            raise ValueError("Validation cadence changed")
    return len(history) * math.ceil(n / batch)


def validate_optimizer(opt, encoder, query, epoch, steps):
    parameters = list(encoder.parameters()) + list(query.parameters())
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("Nonfinite model parameters")
    if type(opt) is not torch.optim.AdamW or len(opt.param_groups) != 1:
        raise ValueError("Fixed single-group AdamW required")
    group = opt.param_groups[0]
    expected_lr = training.learning_rate(epoch) if epoch else training.PEAK_LR
    expected = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "lr": expected_lr,
        "amsgrad": False,
        "maximize": False,
    }
    if any(group[k] != v for k, v in expected.items()):
        raise ValueError("Restored optimizer recipe differs")
    if list(map(id, group["params"])) != list(map(id, parameters)):
        raise ValueError("Optimizer parameter identity/order differs")
    if steps == 0:
        if opt.state:
            raise ValueError("Epoch0 must have fresh Adam state")
        return
    if set(opt.state) != set(parameters):
        raise ValueError("Adam state parameter support differs")
    for parameter in parameters:
        state = opt.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != steps:
            raise ValueError("Adam steps differ from committed exposure")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (
                value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid Adam state tensor")
            if key == "exp_avg_sq" and (value < 0).any():
                raise ValueError("Negative Adam second moment")


def commit_checkpoint(folder, state):
    """Two-phase file commit; receipt recovery requires the saved commit intent."""
    folder.mkdir(exist_ok=True, parents=True)
    last, temporary = folder / "last.pt", folder / "last.pt.tmp"
    old_receipt = folder / "checkpoint.json"
    previous = read_json(old_receipt) if old_receipt.exists() else None
    torch.save(state, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    receipt = {"epoch": state["epoch"], "sha256": sha256(temporary), "binding": state["binding"]}
    intent = {"previous": previous, "next": receipt}
    atomic_json(intent, folder / "commit_intent.json")
    temporary.replace(last)
    atomic_json(receipt, old_receipt)
    (folder / "commit_intent.json").unlink()


def recover_checkpoint(folder, expected_binding):
    last, receipt_path = folder / "last.pt", folder / "checkpoint.json"
    intent_path, pending_path = folder / "commit_intent.json", folder / "pending_epoch.json"
    if not last.exists():
        if any(
            p.exists() for p in (receipt_path, intent_path, pending_path, folder / "last.pt.tmp")
        ):
            raise ValueError("Incomplete initial checkpoint; no silent reinitialization")
        return None
    actual = sha256(last)
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    intent = read_json(intent_path) if intent_path.exists() else None
    if intent is not None:
        desired = intent["next"]
        if desired["sha256"] != actual or desired["binding"] != expected_binding:
            raise ValueError("Uncommitted checkpoint intent; do not replay uncertain updates")
        if receipt not in (intent["previous"], desired):
            raise ValueError("Checkpoint receipt has no matching commit chain")
        receipt = desired
        atomic_json(receipt, receipt_path)
        intent_path.unlink()
    if not receipt or receipt["sha256"] != actual or receipt["binding"] != expected_binding:
        raise ValueError("Checkpoint fingerprint/binding changed without a commit intent")
    ck = torch.load(last, map_location="cpu", weights_only=True)
    if ck["binding"] != expected_binding or ck["epoch"] != receipt["epoch"]:
        raise ValueError("Checkpoint content differs from commit receipt")
    if pending_path.exists():
        pending = read_json(pending_path)
        if pending.get("binding") != expected_binding or pending.get("epoch") != ck["epoch"]:
            raise ValueError("Uncommitted epoch may contain updates; automatic replay forbidden")
        # Same epoch is an atomically committed transaction, regardless of callback count.
        pending_path.unlink()
    return ck


def publish_history(folder, history):
    path = folder / "history.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in history))
    tmp.replace(path)


@torch.no_grad()
def validation(meta, encoder, query, x, raw, scales, arm, device):
    encoder.eval()
    query.eval()
    totals, contexts = {}, 0
    for start in range(0, len(x), MICRO):
        features = torch.as_tensor(x[start : start + MICRO], device=device)
        closes = torch.as_tensor(raw[start : start + MICRO], device=device)
        for left in range(0, len(coverage.VAL_POSITIONS), 5):
            positions = torch.tensor(coverage.VAL_POSITIONS[left : left + 5], device=device).expand(
                len(features), -1
            )
            terms = training.training_terms(
                encoder, query, features, closes, positions, meta["statistics"], scales, arm
            )
            weight = len(features) * positions.shape[1]
            for key, value in content.detached_summary(terms).items():
                totals[key] = totals.get(key, 0.0) + np.asarray(value) * weight
            contexts += weight
    result = {key: (value / contexts).tolist() for key, value in totals.items()}
    json.dumps(result, allow_nan=False)
    return {
        "positions": list(coverage.VAL_POSITIONS),
        "parents": len(x),
        "contexts": contexts,
        "components": result,
    }


def load_pair(meta, job, device):
    return coverage.make_models("rolling", job["seed"], device, meta["encoder"])


def _cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def check_job(meta, out, job):
    folder = out / job["name"]
    done = read_json(folder / "completion.json")
    if done["binding"] != binding(out, job) or done["epoch"] != EPOCHS:
        raise ValueError("Worker completion is not fixed last120")
    verify_files(folder, done["files"])
    if (folder / "pending_epoch.json").exists() or (folder / "commit_intent.json").exists():
        raise ValueError("Completed worker has an unresolved transaction")
    required = {
        "last.pt",
        "checkpoint.json",
        "history.jsonl",
        "training_summary.json",
        "causal_audit.json",
        "final_causal_audit.json",
    }
    if set(done["files"]) != required:
        raise ValueError("Incomplete fixed worker evidence")
    summary = read_json(folder / "training_summary.json")
    n = meta["counts"]["train"]
    steps = history_check(summary["history"], job, n, BATCH)
    if (
        len(summary["history"]) != EPOCHS
        or summary["epoch"] != EPOCHS
        or summary["updates"] != steps
        or done["updates"] != steps
        or summary["examples"] != n * EPOCHS
        or summary["supervised_states"] != n * EPOCHS * 5
        or any(summary[k] != done[k] for k in ("binding", "initial_signature", "signature"))
    ):
        raise ValueError("Completed worker exposure/signature mismatch")
    lines = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    receipt = read_json(folder / "checkpoint.json")
    if (
        lines != summary["history"]
        or receipt["epoch"] != EPOCHS
        or receipt["binding"] != done["binding"]
        or receipt["sha256"] != done["files"]["last.pt"]
    ):
        raise ValueError("Completed checkpoint/history receipts differ")
    for name in ("causal_audit.json", "final_causal_audit.json"):
        audit = read_json(folder / name)
        if not audit["passed"] or not all(audit["checks"].values()):
            raise ValueError("Completed model failed causal audit")
    return done


def train_job(meta, out, job, x, raw, val, val_raw, scales, device):
    folder = data.safe_path(out / job["name"])
    folder.mkdir(exist_ok=True)
    if (folder / "completion.json").exists():
        return check_job(meta, out, job)
    encoder, query = load_pair(meta, job, device)
    initial = training.parameter_signature(encoder, query)
    opt = training.new_optimizer(encoder, query)
    bind = binding(out, job)
    ck = recover_checkpoint(folder, bind)
    if ck is None:
        if any(p.name not in ("worker.lock",) for p in folder.iterdir()):
            raise ValueError("Unbound worker output; do not overwrite")
        audit = coverage.audit(
            encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"]
        )
        atomic_json(audit, folder / "causal_audit.json")
        if not audit["passed"]:
            raise ValueError("Scratch causality/target audit failed")
        baseline = validation(meta, encoder, query, val, val_raw, scales, job["arm"], device)
        history, epoch = [], 0
        ck = {
            "binding": bind,
            "initial_signature": initial,
            "initial_validation": baseline,
            "epoch": epoch,
            "history": history,
            "encoder": _cpu(encoder),
            "query": _cpu(query),
            "optimizer": opt.state_dict(),
            "rng": rng_state(),
            "signature": initial,
        }
        commit_checkpoint(folder, ck)
    else:
        if ck["initial_signature"] != initial or ck["epoch"] != len(ck["history"]):
            raise ValueError("Scratch initialization or history length changed")
        history, epoch = ck["history"], ck["epoch"]
        encoder.load_state_dict(ck["encoder"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
        if training.parameter_signature(encoder, query) != ck["signature"]:
            raise ValueError("Committed parameter signature changed")
        opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
    initial_validation = ck["initial_validation"]
    steps = history_check(history, job, len(x), BATCH)
    validate_optimizer(opt, encoder, query, epoch, steps)
    publish_history(folder, history)
    del ck
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
            scales,
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
            validation(meta, encoder, query, val, val_raw, scales, job["arm"], device)
            if current % 5 == 0 or current == EPOCHS
            else None
        )
        history.append(report)
        steps = history_check(history, job, len(x), BATCH)
        validate_optimizer(opt, encoder, query, current, steps)
        signature = training.parameter_signature(encoder, query)
        state = {
            "binding": bind,
            "epoch": current,
            "history": history,
            "initial_signature": initial,
            "initial_validation": initial_validation,
            "encoder": _cpu(encoder),
            "query": _cpu(query),
            "optimizer": opt.state_dict(),
            "rng": rng_state(),
            "signature": signature,
        }
        commit_checkpoint(folder, state)
        del state
        publish_history(folder, history)
        (folder / "pending_epoch.json").unlink()
        progress(
            f"{job['name']}: {current}/{EPOCHS}, loss={report['loss']:.6f}, epoch={report['seconds']:.1f}s; fixed last selection"
        )
    final_audit = coverage.audit(
        encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"]
    )
    atomic_json(final_audit, folder / "final_causal_audit.json")
    if not final_audit["passed"]:
        raise ValueError("Trained-model causality/target audit failed")
    summary = {
        "binding": bind,
        "epoch": EPOCHS,
        "initial_signature": initial,
        "signature": training.parameter_signature(encoder, query),
        "history": history,
        "initial_validation": initial_validation,
        "updates": steps,
        "examples": len(x) * EPOCHS,
        "supervised_states": len(x) * EPOCHS * 5,
    }
    atomic_json(summary, folder / "training_summary.json")
    files = {
        name: sha256(folder / name)
        for name in (
            "last.pt",
            "checkpoint.json",
            "history.jsonl",
            "training_summary.json",
            "causal_audit.json",
            "final_causal_audit.json",
        )
    }
    done = {
        "binding": bind,
        "epoch": EPOCHS,
        "initial_signature": initial,
        "signature": summary["signature"],
        "updates": steps,
        "files": files,
    }
    atomic_json(done, folder / "completion.json")
    del encoder, query, opt
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def lock_models(meta, out):
    matrix = {j["name"]: check_job(meta, out, j) for j in jobs()}
    for seed in (42, 43):
        if (
            matrix[f"control_s{seed}"]["initial_signature"]
            != matrix[f"multiscale_s{seed}"]["initial_signature"]
        ):
            raise ValueError("Matched-seed scratch tensors differ")
    value = {
        "schema": SCHEMA,
        "selection": "fixed_last120",
        "manifest": sha256(out / "manifest.json"),
        "scales": sha256(out / "multiscale_scales_receipt.json"),
        "models": matrix,
        "source": meta["source_identity"],
        "readout_fits": 0,
    }
    immutable_json(value, out / "model_selection_lock.json")
    return value


def verify_lock(meta, out, name, split):
    if name not in {j["name"] for j in jobs()} or split not in ("test", "cross_research"):
        raise ValueError("Unknown locked model/cohort")
    base = read_json(out / "manifest.json")
    if base["implementation"] != implementation() or base["protocol"] != protocol():
        raise ValueError("Bound implementation/protocol changed during execution")
    # Scales are a separate immutable artifact; never rewrite the base manifest.
    if {k: v for k, v in meta.items() if k != "scales"} != base:
        raise ValueError("Evaluation context differs from immutable manifest")
    if meta["scales"] != read_json(out / "multiscale_scales.json")["scales"]:
        raise ValueError("Evaluation scales changed")
    lock_models(base, out)
    raw = read_json(out / f"multiscale_raw_{split}.json")
    verify_files(
        out, {f"raw/{split}.npz": raw["raw_sha256"], f"{split}_rows.json": raw["rows_file_sha256"]}
    )
    return {
        "model_lock": sha256(out / "model_selection_lock.json"),
        "source": base["source_identity"],
        "raw_receipt": sha256(out / f"multiscale_raw_{split}.json"),
        "scales": sha256(out / "multiscale_scales_receipt.json"),
    }


def remaining_epochs(out):
    result = 0
    for job in jobs():
        path = out / job["name"] / "checkpoint.json"
        epoch = read_json(path)["epoch"] if path.exists() else 0
        if type(epoch) is not int or not 0 <= epoch <= EPOCHS:
            raise ValueError("Invalid committed epoch")
        result += EPOCHS - epoch
    return result


def remaining_seconds(out):
    budget = read_json(out / "budget.json")
    attempts = budget.get("attempts", [])
    if (
        budget.get("used_seconds") != SESSION_CAP
        or not attempts
        or "elapsed_seconds" in attempts[-1]
        or "exit_code" in attempts[-1]
    ):
        raise ValueError("Worker requires an active reserved budget attempt")
    attempt = attempts[-1]
    reserved = attempt.get("reserved_seconds", float("nan"))
    started = attempt.get("started_unix", float("nan"))
    if (
        not math.isfinite(reserved)
        or not 0 < reserved <= SESSION_CAP
        or not math.isfinite(started)
        or budget.get("deadline_unix") != started + reserved
    ):
        raise ValueError("Worker budget reservation/deadline mismatch")
    seconds = budget["deadline_unix"] - time.time()
    if not math.isfinite(seconds) or not 0 < seconds <= SESSION_CAP:
        raise ValueError("Cumulative36h budget exhausted")
    return seconds


@contextmanager
def worker_deadline(out):
    """Independent deadline even if the supervising process disappears."""
    seconds = remaining_seconds(out)
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("Worker cumulative36h deadline expired")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def preflight(meta, out, x, raw, scales, device, evaluate_only=False):
    """Zero optimizer updates: time both actual losses, validation and full128 reads."""
    encoder, query = load_pair(meta, jobs()[0], device)
    signature = training.parameter_signature(encoder, query)
    torch.cuda.reset_peak_memory_stats()
    features = torch.as_tensor(x[:MICRO], device=device)
    closes = torch.as_tensor(raw[:MICRO], device=device)
    positions = torch.as_tensor(training.schedule(len(x), 42, 1)[1][:MICRO], device=device)
    parameters = list(encoder.parameters()) + list(query.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
    measurements = {}
    for arm in content.ARMS:
        durations = []
        for repeat in range(3):
            encoder.train()
            query.train()
            encoder.zero_grad(set_to_none=True)
            query.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            terms = training.training_terms(
                encoder, query, features, closes, positions, meta["statistics"], scales, arm
            )
            terms["objective"].mean().backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            torch.cuda.synchronize()
            if repeat:
                durations.append(time.perf_counter() - started)
            del terms
        measurements[arm + "_micro_backward_seconds"] = max(durations)
    encoder.zero_grad(set_to_none=True)
    query.zero_grad(set_to_none=True)
    with torch.no_grad():
        encoder.eval()
        query.eval()
        for label, ps in (
            ("validation9", coverage.VAL_POSITIONS),
            ("evaluation128", tuple(range(1, 129))),
        ):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for left in range(0, len(ps), 5):
                pp = torch.tensor(ps[left : left + 5], device=device).expand(len(features), -1)
                training.training_terms(
                    encoder, query, features, closes, pp, meta["statistics"], scales, "control"
                )
            torch.cuda.synchronize()
            measurements[label + "_micro_seconds"] = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    free, total = torch.cuda.mem_get_info()
    resident = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    # Reserved-but-idle allocator blocks are reusable by this worker; driver
    # free alone would count the measured warmup peak against it twice.
    available = free + reserved
    # Peak already includes parameters+gradients; add Adam moments and a full
    # parameter-sized step temporary, plus20% and512MiB backend headroom.
    gpu_required = (peak + 3 * parameter_bytes) * 1.2 + 512 * 2**20
    research = meta["counts"]["test"] + meta["counts"]["cross_research"]
    full_predictions = 4 * research * 128 * 127 * 7 * 4
    selected = 4 * research * 8 * 127 * 7 * (4 + 8 + 1)
    checkpoint = parameter_bytes * 3 + 8 * 2**20  # weights + two Adam moments
    disk_required = 5 * checkpoint + full_predictions + selected + 2 * 2**30
    # Deliberately conservative: do not credit incomplete or unrelated files.
    disk_free = shutil.disk_usage(out).free
    # Probe actual serialization/rename on this disk, without allocating Adam
    # or updating weights. Include this measured I/O in the conservative ETA.
    save_seconds = None
    if disk_free >= disk_required:
        probe = out / "preflight_save.tmp"
        if probe.exists():
            raise ValueError("Unresolved preflight save probe; preserve failed attempt")
        state = _cpu(encoder) | {"query/" + key: value for key, value in _cpu(query).items()}
        started = time.perf_counter()
        torch.save(state, probe)
        with probe.open("rb") as handle:
            os.fsync(handle.fileno())
        save_seconds = time.perf_counter() - started
        probe.unlink()
        del state
    epochs_left = remaining_epochs(out)
    if evaluate_only and epochs_left:
        raise ValueError("Evaluate-only requires all four fixed120 workers complete")
    train_seconds = (
        epochs_left
        * math.ceil(len(x) / MICRO)
        * max(measurements[a + "_micro_backward_seconds"] for a in content.ARMS)
    )
    validation_calls = sum(
        1
        + math.ceil(
            (
                EPOCHS
                - (
                    read_json(out / j["name"] / "checkpoint.json")["epoch"]
                    if (out / j["name"] / "checkpoint.json").exists()
                    else 0
                )
            )
            / 5
        )
        for j in jobs()
    )
    val_seconds = (
        validation_calls
        * math.ceil(meta["counts"]["val"] / MICRO)
        * measurements["validation9_micro_seconds"]
    )
    eval_seconds = (
        4
        * sum(math.ceil(meta["counts"][s] / MICRO) for s in ("test", "cross_research"))
        * measurements["evaluation128_micro_seconds"]
    )
    # Three parameter-sized states per checkpoint (weights + Adam moments),
    # plus checksum/receipt/atomic replacement allowance.
    save_total = (save_seconds or 0.0) * 4 * (epochs_left + 4)
    estimate = (train_seconds + val_seconds + eval_seconds + save_total) * 1.3 + 1800
    checks = {
        "zero_updates_same_parameters": training.parameter_signature(encoder, query) == signature,
        "gpu_room": gpu_required <= available,
        "disk_room": disk_required <= disk_free,
        "time_room": estimate <= remaining_seconds(out),
    }
    result = {
        "schema": SCHEMA,
        "checks": checks,
        "passed": all(checks.values()),
        "zero_updates": True,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "measurements": measurements,
        "weight_save_fsync_seconds": save_seconds,
        "parameter_bytes": parameter_bytes,
        "peak_cuda_bytes": peak,
        "gpu_required_bytes": gpu_required,
        "gpu_available_bytes": available,
        "gpu_resident_bytes": resident,
        "gpu_reserved_bytes": reserved,
        "gpu_total_bytes": total,
        "disk_required_bytes": disk_required,
        "disk_free_bytes": disk_free,
        "disk_policy": "conservative complete artifact set plus one atomic replacement; no existing-file credit",
        "remaining_epochs": epochs_left,
        "estimated_seconds": estimate,
        "remaining_budget_seconds": remaining_seconds(out),
        "runtime": runtime.runtime_state(),
    }
    attempts = out / "preflight"
    attempts.mkdir(exist_ok=True)
    atomic_json(result, attempts / f"attempt_{len(list(attempts.glob('attempt_*.json'))) + 1}.json")
    progress(
        f"Zero-update preflight: estimated {estimate / 3600:.2f}h, disk need {disk_required / 2**30:.2f}GiB/free {disk_free / 2**30:.2f}GiB, GPU need {gpu_required / 2**30:.2f}GiB"
    )
    del encoder, query, parameters, features, closes
    gc.collect()
    torch.cuda.empty_cache()
    if not result["passed"]:
        raise ValueError(f"Resource/numeric preflight stopped without updates: {checks}")
    return result


def source_identity(source):
    return {
        "completion": sha256(source / "completion.json"),
        "manifest": sha256(source / "manifest.json"),
    }


def execute(source, out, evaluate_only=False):
    with (out / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not torch.cuda.is_available():
                raise ValueError("Real execution requires AutoDL CUDA")
            remaining_seconds(out)
            progress("F01/F03: verifying immutable data lineage; no source model weights loaded")
            verified = data.verify(source, out)
            meta = {
                **verified,
                "schema": SCHEMA,
                "source": str(source),
                "source_identity": source_identity(source),
                "encoder": dict(coverage.CONFIG),
                "counts": dict(data.COUNTS),
                "epochs": EPOCHS,
                "batch": BATCH,
                "micro": MICRO,
                "protocol": protocol(),
                "implementation": implementation(),
            }
            request = read_json(out / "request.json")
            if request != {
                "schema": SCHEMA,
                "source": str(source),
                "output": str(out),
                "protocol": meta["protocol"],
                "implementation": meta["implementation"],
            }:
                raise ValueError("Supervisor request/code changed before worker execution")
            immutable_json(meta, out / "manifest.json")
            with runtime.replay_runtime("context"):
                data.prepare(meta, out, ("train", "val"))
                train, train_rows = data.cache(meta, "train")
                val, _ = data.cache(meta, "val")
                raw, val_raw = data.closes(out, "train"), data.closes(out, "val")
                scale_report = data.fit_scales(meta, out, raw, train_rows)
                scales = scale_report["scales"]
                # Reject uncertain epochs before doing another preflight. A
                # committed save can repair its receipt without more updates.
                for job in jobs():
                    folder = out / job["name"]
                    if folder.exists():
                        recovered = recover_checkpoint(folder, binding(out, job))
                        if recovered is not None:
                            if recovered["epoch"] != len(recovered["history"]):
                                raise ValueError("Resume epoch/history mismatch")
                            history_check(recovered["history"], job, len(train), BATCH)
                            del recovered
                if evaluate_only:
                    lock_models(meta, out)
                preflight(meta, out, train, raw, scales, "cuda", evaluate_only)
                if not evaluate_only:
                    for job in jobs():
                        train_job(meta, out, job, train, raw, val, val_raw, scales, "cuda")
                lock_models(meta, out)
                del train, raw, val, val_raw
                gc.collect()
                progress("All four fixed-last models locked; preparing research cohorts")
                data.prepare(meta, out, ("test",))
                data.prepare(meta, out, ("cross_research",))
                banks, rows_bank = {}, {}
                eval_meta = {**meta, "scales": scales}
                for split in ("test", "cross_research"):
                    x, rows = data.cache(meta, split)
                    close = data.closes(out, split)
                    banks[split], rows_bank[split] = {}, rows
                    for job in jobs():
                        verify_lock(eval_meta, out, job["name"], split)
                        encoder, query = load_pair(meta, job, "cuda")
                        ck = recover_checkpoint(out / job["name"], binding(out, job))
                        encoder.load_state_dict(ck["encoder"], strict=True)
                        query.load_state_dict(ck["query"], strict=True)
                        if training.parameter_signature(encoder, query) != ck["signature"]:
                            raise ValueError("Selected inference state changed")
                        del ck
                        banks[split][job["name"]] = evaluation.extract(
                            eval_meta,
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
                decision = evaluation.decisions(banks, rows_bank)
                atomic_json(decision, out / "multiscale_metrics.json")
            if (
                data.verify(source, out) != verified
                or source_identity(source) != meta["source_identity"]
                or implementation() != meta["implementation"]
                or protocol() != meta["protocol"]
            ):
                raise ValueError(
                    "Source changed or implementation/protocol changed during execution"
                )
            files = {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and p.suffix in (".json", ".jsonl", ".npy", ".npz", ".pt")
                and p.name not in ("budget.json", "status.json", "completion.json", "failure.json")
            }
            # Per-job/extraction completion receipts are also immutable evidence.
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
            failures = out / "failures"
            failures.mkdir(exist_ok=True)
            atomic_json(failure, failures / f"attempt_{time.time_ns()}.json")
            atomic_json(failure, out / "failure.json")
            atomic_json(
                {"schema": SCHEMA, "status": "failed", "promoted": False}, out / "status.json"
            )
            traceback.print_exc()
            return 1


def reserve_budget(out):
    budget = (
        read_json(out / "budget.json")
        if (out / "budget.json").exists()
        else {"used_seconds": 0.0, "attempts": []}
    )
    used = budget["used_seconds"]
    if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
        raise ValueError(
            "Cumulative36h budget exhausted or unresolved supervisor attempt; export only"
        )
    remaining = SESSION_CAP - used
    attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
    budget["attempts"].append(attempt)
    budget.update(
        used_seconds=SESSION_CAP,
        deadline_unix=attempt["started_unix"] + remaining,
        supervisor_pid=os.getpid(),
    )
    atomic_json(budget, out / "budget.json")
    return budget, used, remaining


def supervise(source, out, evaluate_only=False):
    source, out = data.safe_path(source), data.safe_path(out)
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Separate source/output required")
    if not (out / "request.json").exists():
        # The data adapter confines legacy identity writes to private scratch.
        # Reject actual raw/ancestor outputs before creating any run files.
        data.verify(source, out)
    out.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": SCHEMA,
        "source": str(source),
        "output": str(out),
        "protocol": protocol(),
        "implementation": implementation(),
    }
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out / "request.json").exists() and any(
            p.name != "session.lock" for p in out.iterdir()
        ):
            raise ValueError("Existing output lacks this run request")
        immutable_json(request, out / "request.json")
        if (out / "completion.json").exists():
            verify_files(out, read_json(out / "completion.json")["files"])
            progress("Completed run verified; export only, no additional execution")
            return 0
        budget, used, remaining = reserve_budget(out)
        command = [
            sys.executable,
            "-m",
            MODULE,
            "--source",
            str(source),
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

        previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
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
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        elapsed = min(remaining, time.monotonic() - started)
        budget["used_seconds"] = used + elapsed
        budget["attempts"][-1].update(elapsed_seconds=elapsed, exit_code=code)
        atomic_json(budget, out / "budget.json")
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source, out = data.safe_path(args.source), data.safe_path(args.out)
    if args.worker:
        budget = read_json(out / "budget.json")
        if budget["supervisor_pid"] != os.getppid():
            raise ValueError("Worker must be started by its budget-owning supervisor")
        with worker_deadline(out):
            raise SystemExit(execute(source, out, args.evaluate_only))
    raise SystemExit(supervise(source, out, args.evaluate_only))


if __name__ == "__main__":
    main()
