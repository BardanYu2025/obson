"""F05/F01/F11 scratch input-granularity comparison; real execution requires AutoDL CUDA.

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

from . import input_patch as model
from . import input_patch_evaluate as evaluation
from . import input_patch_training as training
from . import multiscale_content as content
from . import multiscale_data as data
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

SCHEMA = "babel-input-patch768-v1"
MODULE = "obson.babel.input_patch_run"
EPOCHS = 120
BATCH = 128
MICRO = 8
SESSION_CAP = 24 * 3600
REPO = Path(__file__).resolve().parents[3]


def jobs():
    return [
        {"name": f"{arm}_s{seed}", "arm": arm, "seed": seed}
        for arm, seed in (
            ("bar", 42),
            ("local4", 42),
            ("patch4", 42),
            ("patch4", 43),
            ("local4", 43),
            ("bar", 43),
        )
    ]


def model_contract(arm=None, config=None):
    if arm is None:
        return {value: model.contract(value, config) for value in model.ARMS}
    return model.contract(arm, config)


def protocol():
    return {
        "task_ids": ["F05", "F01", "F11"],
        "jobs": jobs(),
        "epochs": EPOCHS,
        "batch": BATCH,
        "micro": MICRO,
        "workers": 1,
        "session_cap_seconds": SESSION_CAP,
        "session_accounting": "successful first read-only source qualification is charged before worker reservation; resumed worker wall time includes preparation, preflight, training and evaluation; final export is outside the execution cap",
        "encoder": model.CONFIG,
        "patch_length": 4,
        "source": "reviewed overlap-content DATA lineage only; no source neural weights or Adam loaded",
        "source_completion": data.COMPLETION,
        "counts": data.COUNTS,
        "initialization": "scratch; same-seed trunk/coordinates/query exact across all arms; local4/patch4 all tensors exact; different bar input shape declared",
        "sampling": "five uniform endpoints without replacement at epochs1..120; each independently sees full128",
        "labels": "unchanged raw float64 relative close, body/activity masks and frozen train statistics",
        "objectives": dict.fromkeys(model.ARMS, ".4path+.2change1+.1body+.3activity; original Q"),
        "model_contract": model_contract(),
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "peak_lr": 3e-4,
            "warmup": 10,
            "clip": 1.0,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "restore": "fresh zero moments; only own committed checkpoints resume",
        },
        "runtime": runtime.RUNTIME_PROFILES["context"],
        "precision": "FP32 parameters/forward; float64 truth/loss; no AMP/TF32",
        "selection": "fixed last120 only; epoch0/every5 validation diagnostics; no best/LR/patch-size search",
        "validation_positions": list(model.VAL_POSITIONS),
        "evaluation_positions": list(range(1, 129)),
        "prediction_export_positions": list(evaluation.SAMPLE_POSITIONS),
        "comparisons": [
            "patch4_vs_bar primary",
            "patch4_vs_local4 required",
            "local4_vs_bar secondary",
        ],
        "acceptance": "patch4 must pass BOTH bar and local4 for both seeds/cohorts: paired95% upper(candidate-.95control)<=0 for composite price; inherited all10% physical/F03/P95 guards; no generic promotion",
        "bootstrap": {
            "draws": 2000,
            "seed": 20261012,
            "groups": ["month", "contract_period_month"],
            "min_parents": 100,
            "min_groups": 6,
        },
        "budget": "six120 workers=720epochs/27360updates/17240400 supervised-state exposures; equal updates, not equal FLOPs/time",
        "F11": "zero readout fits; public768=3072float32 bytes; actual parameters/time reported; generic reuse unknown",
        "stopping": "fixed120/cumulative24h; numeric/source/causality/resource failure stops; uncertain updates never replay; no automatic extension/grid",
        "source_dependency_limit": "legacy data provenance still hashes ancestral weight files; no old model/Adam is deserialized for initialization",
        "scope": "reused research cohorts only; Macro unchanged; no forecasting or universal representation acceptance",
    }


def implementation():
    # Include the inherited source verifier and model/target code, not mutable docs.
    from . import (
        ae_extend,
        history_autoencoder,
        history_price_truth,
        history_query,
        input_patch_export,
        multiscale_evaluate,
        multiscale_metrics,
        multiscale_training,
        observed_history,
        observed_history_export,
        observed_history_export_v2,
        observed_history_training,
        overlap_content_audit,
        state_coverage,
        uniform_context,
    )

    return {
        "inherited": data.original.diagnostic.implementation(),
        "data": data.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (
                data,
                transactions,
                content,
                training,
                evaluation,
                model,
                uniform_context,
                multiscale_evaluate,
                multiscale_metrics,
                input_patch_export,
                runtime,
                history_query,
                history_price_truth,
                multiscale_training,
                state_coverage,
                overlap_content_audit,
                observed_history,
                observed_history_training,
                observed_history_export,
                observed_history_export_v2,
                history_autoencoder,
                ae_extend,
            )
        }
        | {Path(__file__).name: sha256(__file__)},
        "script": sha256(REPO / "scripts/babel_input_patch768_autodl.sh"),
    }


def binding(out, job):
    return {
        "manifest": sha256(out / "manifest.json"),
        "model_contract": model_contract(job["arm"], read_json(out / "manifest.json")["encoder"]),
        "job": job,
        "initialization_sha256": sha256(out / f"initialization_s{job['seed']}.json"),
    }


def prepare_initializations(meta, out, device):
    """Both seeded comparisons qualify before any training update."""
    for seed in (42, 43):
        receipt = training.initialization_receipt(seed, device, meta["encoder"])
        if (
            receipt.get("seed") != seed
            or receipt.get("passed") is not True
            or not all(receipt.get("checks", {}).values())
        ):
            raise ValueError("Scratch common-parameter initialization check failed")
        arms = receipt.get("arms", {})
        if set(arms) != set(model.ARMS):
            raise ValueError("Initialization receipt requires all three input arms")
        if (
            len({arms[a]["common_signature"] for a in model.ARMS}) != 1
            or arms["local4"]["signature"] != arms["patch4"]["signature"]
        ):
            raise ValueError("Matched trunk/query or local4/patch4 initialization differs")
        immutable_json(receipt, out / f"initialization_s{seed}.json")
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def expected_initial(out, job):
    receipt = read_json(out / f"initialization_s{job['seed']}.json")
    if (
        receipt.get("seed") != job["seed"]
        or receipt.get("passed") is not True
        or not all(receipt.get("checks", {}).values())
    ):
        raise ValueError("Invalid bound scratch initialization receipt")
    return receipt["arms"][job["arm"]]["signature"]


def history_check(history, job, n, batch=BATCH, *, config=None):
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
        if item.get("model_contract") != model.contract(job["arm"], config):
            raise ValueError("History complete model input/attention contract differs")
        # Every numeric reported value must serialize as finite JSON.
        json.dumps(item, allow_nan=False)
        if bool(item.get("validation") is not None) != (epoch % 5 == 0 or epoch == EPOCHS):
            raise ValueError("Validation cadence changed")
    return len(history) * math.ceil(n / batch)


def validate_optimizer(opt, encoder, query, epoch, steps):
    parameters = list(encoder.parameters()) + list(query.parameters())
    if any(
        p.dtype != torch.float32 or not p.requires_grad or not torch.isfinite(p).all()
        for p in parameters
    ):
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
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    if not set(expected).issubset(group) or any(group[k] != v for k, v in expected.items()):
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
                or value.device != parameter.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid Adam state tensor")
            if key == "exp_avg_sq" and (value < 0).any():
                raise ValueError("Negative Adam second moment")


@torch.no_grad()
def validation(meta, encoder, query, x, raw, arm, device):
    encoder.eval()
    query.eval()
    totals, contexts = {}, 0
    for start in range(0, len(x), MICRO):
        features = torch.as_tensor(x[start : start + MICRO], device=device)
        closes = torch.as_tensor(raw[start : start + MICRO], device=device)
        for left in range(0, len(model.VAL_POSITIONS), 5):
            positions = torch.tensor(model.VAL_POSITIONS[left : left + 5], device=device).expand(
                len(features), -1
            )
            terms = training.training_terms(
                encoder, query, features, closes, positions, meta["statistics"], arm
            )
            weight = len(features) * positions.shape[1]
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
    }


def load_pair(meta, job, device):
    encoder, query = model.make_models(job["arm"], job["seed"], device, meta["encoder"])
    contract = model.model_contract(encoder)
    if (
        contract.get("arm") != job["arm"]
        or contract.get("attention_mode") != "causal"
        or contract.get("input_length") != 128
        or contract.get("encoder") != meta["encoder"]
    ):
        raise ValueError("Constructed model differs from bound input/attention arm")
    return encoder, query


def validate_saved_state(ck, encoder, query, epoch, steps):
    """Reject casting, malformed Adam and RNG before restoring any state."""
    import random
    from collections.abc import Mapping

    required = {
        "binding",
        "initial_signature",
        "initial_validation",
        "epoch",
        "history",
        "encoder",
        "query",
        "optimizer",
        "rng",
        "signature",
    }
    if (
        not isinstance(ck, Mapping)
        or set(ck) != required
        or type(ck["epoch"]) is not int
        or ck["epoch"] != epoch
    ):
        raise ValueError("Committed scratch checkpoint schema differs")
    for family, module in (("encoder", encoder), ("query", query)):
        current, saved = module.state_dict(), ck[family]
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise ValueError("Saved model tensor keys differ")
        for name, ref in current.items():
            value = saved[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != ref.shape
                or value.dtype != ref.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Saved {family}/{name} shape/dtype/value differs")
    opt = ck["optimizer"]
    parameters = list(encoder.parameters()) + list(query.parameters())
    ids = list(range(len(parameters)))
    if (
        not isinstance(opt, Mapping)
        or set(opt) != {"state", "param_groups"}
        or len(opt["param_groups"]) != 1
    ):
        raise ValueError("Saved optimizer schema differs")
    group = opt["param_groups"][0]
    expected = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "lr": training.learning_rate(epoch) if epoch else training.PEAK_LR,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    if (
        group.get("params") != ids
        or not set(expected).issubset(group)
        or any(group.get(k) != v for k, v in expected.items())
        or set(opt["state"]) != (set(ids) if steps else set())
    ):
        raise ValueError("Saved optimizer recipe/order/support differs")
    for index, parameter in enumerate(parameters):
        if not steps:
            break
        moment = opt["state"][index]
        if not isinstance(moment, Mapping) or set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Saved Adam moment schema differs")
        step = moment["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.ndim != 0
            or step.dtype != torch.float32
            or not torch.isfinite(step).all()
            or float(step) != steps
        ):
            raise ValueError("Saved Adam steps differ")
        for key in ("exp_avg", "exp_avg_sq"):
            value = moment[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()
                or (key == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError("Saved Adam moment shape/dtype/value differs")
    rng = ck["rng"]
    if not isinstance(rng, Mapping) or set(rng) != {"torch", "cuda", "python", "numpy"}:
        raise ValueError("Saved RNG schema differs")
    cpu = rng["torch"]
    if (
        not isinstance(cpu, torch.Tensor)
        or cpu.dtype != torch.uint8
        or cpu.shape != torch.get_rng_state().shape
    ):
        raise ValueError("Saved CPU RNG dtype/shape differs")
    local = torch.Generator()
    local.set_state(cpu)
    if not isinstance(rng["cuda"], list) or len(rng["cuda"]) != (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    ):
        raise ValueError("Resume requires matching CUDA RNG device count")
    for value in rng["cuda"]:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or not value.numel()
        ):
            raise ValueError("Saved CUDA RNG dtype/shape differs")
    random.Random().setstate(rng["python"])
    n = rng["numpy"]
    if not isinstance(n, (tuple, list)) or len(n) != 5 or n[0] != "MT19937":
        raise ValueError("Saved NumPy RNG schema differs")
    nums = np.asarray(n[1])
    if (
        nums.shape != (624,)
        or nums.dtype.kind not in "iu"
        or (nums < 0).any()
        or (nums > 2**32 - 1).any()
    ):
        raise ValueError("Saved NumPy RNG words differ")
    np.random.RandomState().set_state((n[0], nums.astype(np.uint32), n[2], n[3], n[4]))


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
    steps = history_check(summary["history"], job, n, BATCH, config=meta["encoder"])
    if done["initial_signature"] != expected_initial(out, job):
        raise ValueError("Completed model initial signature differs from qualification")
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


def train_job(meta, out, job, x, raw, val, val_raw, device):
    folder = data.safe_path(out / job["name"])
    folder.mkdir(exist_ok=True)
    if (folder / "completion.json").exists():
        return check_job(meta, out, job)
    encoder, query = load_pair(meta, job, device)
    initial = training.parameter_signature(encoder, query)
    if initial != expected_initial(out, job):
        raise ValueError("Scratch initialization differs from both-seed qualification")
    opt = training.new_optimizer(encoder, query)
    bind = binding(out, job)
    ck = recover_checkpoint(folder, bind)
    if ck is None:
        if any(p.name not in ("worker.lock",) for p in folder.iterdir()):
            raise ValueError("Unbound worker output; do not overwrite")
        audit = model.audit(
            encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"]
        )
        atomic_json(audit, folder / "causal_audit.json")
        if not audit["passed"]:
            raise ValueError("Scratch causality/target audit failed")
        baseline = validation(meta, encoder, query, val, val_raw, job["arm"], device)
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
        steps = history_check(history, job, len(x), BATCH, config=meta["encoder"])
        validate_saved_state(ck, encoder, query, epoch, steps)
        encoder.load_state_dict(ck["encoder"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
        if training.parameter_signature(encoder, query) != ck["signature"]:
            raise ValueError("Committed parameter signature changed")
        opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
    initial_validation = ck["initial_validation"]
    steps = history_check(history, job, len(x), BATCH, config=meta["encoder"])
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
        history.append(report)
        steps = history_check(history, job, len(x), BATCH, config=meta["encoder"])
        validate_optimizer(opt, encoder, query, current, steps)
        signature = training.parameter_signature(encoder, query)
        report["parameter_signature"] = signature
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
    final_audit = model.audit(
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
    initialization = {}
    for seed in (42, 43):
        receipt = read_json(out / f"initialization_s{seed}.json")
        if not receipt["passed"] or not all(receipt["checks"].values()):
            raise ValueError("Incomplete scratch initialization qualification")
        if (
            matrix[f"local4_s{seed}"]["initial_signature"]
            != matrix[f"patch4_s{seed}"]["initial_signature"]
        ):
            raise ValueError("Local4/patch4 starting tensors differ")
        initialization[str(seed)] = sha256(out / f"initialization_s{seed}.json")
    value = {
        "schema": SCHEMA,
        "selection": "fixed_last120",
        "manifest": sha256(out / "manifest.json"),
        "models": matrix,
        "source": meta["source_identity"],
        "readout_fits": 0,
        "initialization": initialization,
    }
    immutable_json(value, out / "model_selection_lock.json")
    return value


def verify_lock(meta, out, name, split):
    if name not in {j["name"] for j in jobs()} or split not in ("test", "cross_research"):
        raise ValueError("Unknown locked model/cohort")
    base = read_json(out / "manifest.json")
    if base["implementation"] != implementation() or base["protocol"] != protocol():
        raise ValueError("Bound implementation/protocol changed during execution")
    if meta != base:
        raise ValueError("Evaluation context differs from immutable manifest")
    lock_models(base, out)
    raw = read_json(out / f"multiscale_raw_{split}.json")
    verify_files(
        out, {f"raw/{split}.npz": raw["raw_sha256"], f"{split}_rows.json": raw["rows_file_sha256"]}
    )
    return {
        "model_lock": sha256(out / "model_selection_lock.json"),
        "source": base["source_identity"],
        "raw_receipt": sha256(out / f"multiscale_raw_{split}.json"),
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
        raise ValueError("Cumulative24h budget exhausted")
    return seconds


@contextmanager
def worker_deadline(out):
    """Independent deadline even if the supervising process disappears."""
    seconds = remaining_seconds(out)
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("Worker cumulative24h deadline expired")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def preflight(meta, out, x, raw, device, evaluate_only=False):
    """Measure each actual attention arm with zero optimizer updates.

    All timings use the arm-specific encoder, including validation9 and full128
    queries. The conservative estimate charges every remaining epoch at the
    slower measured arm; equal updates are not claimed as equal wall time.
    """
    features = torch.as_tensor(x[:MICRO], device=device)
    closes = torch.as_tensor(raw[:MICRO], device=device)
    positions = torch.as_tensor(training.schedule(len(x), 42, 1)[1][:MICRO], device=device)
    research = meta["counts"]["test"] + meta["counts"]["cross_research"]
    full_predictions = 6 * research * 128 * 127 * 7 * 4
    selected = 6 * research * 8 * 127 * 7 * (4 + 8 + 1)
    disk_free = shutil.disk_usage(out).free
    measurements, arms = {}, {}
    initialization = {
        str(seed): read_json(out / f"initialization_s{seed}.json") for seed in (42, 43)
    }
    for arm in model.ARMS:
        progress(f"Zero-update preflight: {arm}, backward and validation9/full128 timing")
        encoder, query = load_pair(meta, {"arm": arm, "seed": 42}, device)
        signature = training.parameter_signature(encoder, query)
        parameters = list(encoder.parameters()) + list(query.parameters())
        if any(
            p.dtype != torch.float32 or p.device.type != torch.device(device).type
            for p in parameters
        ):
            raise ValueError("Preflight requires FP32 parameters on the requested device")
        parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        contribution_check = None
        initial_components = None
        norm = None
        for repeat in range(3):
            encoder.train()
            query.train()
            encoder.zero_grad(set_to_none=True)
            query.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            terms = training.training_terms(
                encoder, query, features, closes, positions, meta["statistics"], arm
            )
            objective = terms["objective"].mean()
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite actual Q before any update")
            initial_components = content.detached_summary(terms)
            contributions = torch.stack(
                [
                    terms["objective"][i].sum() / terms["objective"].numel()
                    for i in range(len(features))
                ]
            ).sum()
            contribution_check = {
                "full": float(objective.detach().cpu()),
                "sum_contributions": float(contributions.detach().cpu()),
                "passed": bool(torch.allclose(objective, contributions, atol=1e-10, rtol=1e-8)),
            }
            if not contribution_check["passed"]:
                raise ValueError("Effective-batch Q accumulation mismatch before updates")
            objective.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                .detach()
                .cpu()
            )
            del objective, contributions
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
                ("validation9", model.VAL_POSITIONS),
                ("evaluation128", tuple(range(1, 129))),
            ):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for left in range(0, len(ps), 5):
                    pp = torch.tensor(ps[left : left + 5], device=device).expand(len(features), -1)
                    training.training_terms(
                        encoder, query, features, closes, pp, meta["statistics"], arm
                    )
                torch.cuda.synchronize()
                measurements[arm + "_" + label + "_micro_seconds"] = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        resident = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        checkpoint = parameter_bytes * 3 + 8 * 2**20
        disk_required = 7 * checkpoint + full_predictions + selected + 2 * 2**30
        save_seconds = None
        if disk_free >= disk_required:
            probe = out / f"preflight_{arm}_save.tmp"
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
        arms[arm] = {
            "initial_Q_components": initial_components,
            "preclip_norm": norm,
            "effective_batch_contribution_check": contribution_check,
            "model_contract": model.model_contract(encoder),
            "initial_signature": signature,
            "final_signature": training.parameter_signature(encoder, query),
            "parameter_count": sum(p.numel() for p in parameters),
            "parameter_bytes": parameter_bytes,
            "attention_mode": encoder.attention_mode,
            "peak_cuda_bytes": peak,
            "gpu_required_bytes": (peak + 3 * parameter_bytes) * 1.2 + 512 * 2**20,
            # Allocator-reserved blocks are reusable; do not charge them twice.
            "gpu_available_bytes": free + reserved,
            "gpu_resident_bytes": resident,
            "gpu_reserved_bytes": reserved,
            "gpu_total_bytes": total,
            "disk_required_bytes": disk_required,
            "weight_save_fsync_seconds": save_seconds,
        }
        del encoder, query, parameters
        gc.collect()
        torch.cuda.empty_cache()
    epochs_left = remaining_epochs(out)
    if evaluate_only and epochs_left:
        raise ValueError("Evaluate-only requires all six fixed120 workers complete")
    train_seconds = (
        epochs_left
        * math.ceil(len(x) / MICRO)
        * max(measurements[a + "_micro_backward_seconds"] for a in model.ARMS)
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
        * max(measurements[a + "_validation9_micro_seconds"] for a in model.ARMS)
    )
    eval_seconds = (
        6
        * sum(math.ceil(meta["counts"][s] / MICRO) for s in ("test", "cross_research"))
        * max(measurements[a + "_evaluation128_micro_seconds"] for a in model.ARMS)
    )
    save_seconds = max(v["weight_save_fsync_seconds"] or 0.0 for v in arms.values())
    save_total = save_seconds * 4 * (epochs_left + 6)
    estimate = (train_seconds + val_seconds + eval_seconds + save_total) * 1.3 + 1800
    checks = {
        "zero_updates_same_parameters": all(
            v["initial_signature"] == v["final_signature"] for v in arms.values()
        ),
        "both_seed_initialization": all(
            r.get("passed") is True and all(r.get("checks", {}).values())
            for r in initialization.values()
        ),
        "matched_common_initialization": all(
            len({r["arms"][a]["common_signature"] for a in model.ARMS}) == 1
            for r in initialization.values()
        ),
        "local_patch_complete_initialization": all(
            r["arms"]["local4"]["signature"] == r["arms"]["patch4"]["signature"]
            for r in initialization.values()
        ),
        "declared_parameters": all(
            v["parameter_count"] == initialization["42"]["arms"][a]["parameter_count"]
            and v["parameter_bytes"] == initialization["42"]["arms"][a]["parameter_bytes"]
            and v["initial_signature"] == initialization["42"]["arms"][a]["signature"]
            for a, v in arms.items()
        ),
        "gpu_room": all(v["gpu_required_bytes"] <= v["gpu_available_bytes"] for v in arms.values()),
        "disk_room": all(v["disk_required_bytes"] <= disk_free for v in arms.values()),
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
        "arms": arms,
        "initialization_sha256": {
            str(seed): sha256(out / f"initialization_s{seed}.json") for seed in (42, 43)
        },
        "weight_save_fsync_seconds": save_seconds,
        "parameter_bytes": max(v["parameter_bytes"] for v in arms.values()),
        "peak_cuda_bytes": max(v["peak_cuda_bytes"] for v in arms.values()),
        "gpu_required_bytes": max(v["gpu_required_bytes"] for v in arms.values()),
        "gpu_available_bytes": min(v["gpu_available_bytes"] for v in arms.values()),
        "gpu_resident_bytes": max(v["gpu_resident_bytes"] for v in arms.values()),
        "gpu_reserved_bytes": max(v["gpu_reserved_bytes"] for v in arms.values()),
        "gpu_total_bytes": min(v["gpu_total_bytes"] for v in arms.values()),
        "disk_required_bytes": max(v["disk_required_bytes"] for v in arms.values()),
        "disk_free_bytes": disk_free,
        "disk_policy": "complete artifacts plus one atomic replacement; no existing-file credit",
        "remaining_epochs": epochs_left,
        "estimated_seconds": estimate,
        "remaining_budget_seconds": remaining_seconds(out),
        "runtime": runtime.runtime_state(),
    }
    attempts = out / "preflight"
    attempts.mkdir(exist_ok=True)
    atomic_json(result, attempts / f"attempt_{len(list(attempts.glob('attempt_*.json'))) + 1}.json")
    progress(
        f"Zero-update three-arm input preflight: estimated {estimate / 3600:.2f}h, "
        f"disk need {result['disk_required_bytes'] / 2**30:.2f}GiB/free {disk_free / 2**30:.2f}GiB, "
        f"GPU need {result['gpu_required_bytes'] / 2**30:.2f}GiB"
    )
    del features, closes
    gc.collect()
    torch.cuda.empty_cache()
    if not result["passed"]:
        raise ValueError(f"Resource/numeric preflight stopped without updates: {checks}")
    return result


def cost_summary(meta, out):
    workers = {}
    for job in jobs():
        summary = read_json(out / job["name"] / "training_summary.json")
        init = read_json(out / f"initialization_s{job['seed']}.json")["arms"][job["arm"]]
        workers[job["name"]] = {
            "updates": summary["updates"],
            "training_seconds": sum(r["seconds"] for r in summary["history"]),
            "parent_exposures": summary["examples"],
            "state_exposures": summary["supervised_states"],
            "parameter_count": init["parameter_count"],
            "parameter_bytes": init["parameter_bytes"],
        }
    return {
        "task_id": "F11",
        "workers": workers,
        "public_state_float32_bytes": 3072,
        "private_memory_float32_bytes_per_endpoint": 0,
        "private_memory_scope": "No extra public/persistent latent; ordinary token activations differ by arm",
        "raw_128_close_float32_bytes": 512,
        "prepared_128x28_float32_bytes": 14336,
        "readout_fits": 0,
        "reuse_status": "not_tested_by_new_task",
        "scope": "Measured parameters/time and content retention; equal updates are not equal FLOPs; no storage or generic utility claim",
    }


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
            progress("F01/F05: verifying immutable data lineage; no source model weights loaded")
            verified = data.verify(source, out)
            meta = {
                **verified,
                "schema": SCHEMA,
                "source": str(source),
                "source_identity": source_identity(source),
                "encoder": dict(model.CONFIG),
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
                train, _ = data.cache(meta, "train")
                val, _ = data.cache(meta, "val")
                raw, val_raw = data.closes(out, "train"), data.closes(out, "val")
                immutable_json(
                    {
                        "schema": SCHEMA,
                        "source_identity": meta["source_identity"],
                        "data_completion": data.COMPLETION,
                        "verified_data": verified,
                        "raw_train": sha256(out / "multiscale_raw_train.json"),
                        "raw_val": sha256(out / "multiscale_raw_val.json"),
                        "source_neural_weights_loaded": False,
                        "source_optimizer_loaded": False,
                        "optimizer_updates": 0,
                        "scope": "Reviewed data cache/raw/statistics provenance; ancestry weight hashes remain dependencies",
                    },
                    out / "source_validation.json",
                )
                prepare_initializations(meta, out, "cuda")
                # Reject uncertain epochs before doing another preflight. A
                # committed save can repair its receipt without more updates.
                for job in jobs():
                    folder = out / job["name"]
                    if folder.exists():
                        recovered = recover_checkpoint(folder, binding(out, job))
                        if recovered is not None:
                            if recovered["epoch"] != len(recovered["history"]):
                                raise ValueError("Resume epoch/history mismatch")
                            history_check(
                                recovered["history"], job, len(train), BATCH, config=meta["encoder"]
                            )
                            del recovered
                if evaluate_only:
                    lock_models(meta, out)
                preflight(meta, out, train, raw, "cuda", evaluate_only)
                if not evaluate_only:
                    for job in jobs():
                        train_job(meta, out, job, train, raw, val, val_raw, "cuda")
                lock_models(meta, out)
                del train, raw, val, val_raw
                gc.collect()
                progress("All six fixed-last models locked; preparing research cohorts")
                data.prepare(meta, out, ("test",))
                data.prepare(meta, out, ("cross_research",))
                banks, rows_bank = {}, {}
                eval_meta = meta
                for split in ("test", "cross_research"):
                    x, rows = data.cache(meta, split)
                    close = data.closes(out, split)
                    banks[split], rows_bank[split] = {}, rows
                    for job in jobs():
                        verify_lock(eval_meta, out, job["name"], split)
                        encoder, query = load_pair(meta, job, "cuda")
                        ck = recover_checkpoint(out / job["name"], binding(out, job))
                        validate_saved_state(
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
                atomic_json(cost_summary(meta, out), out / "f11_cost.json")
                decision = evaluation.decisions(banks, rows_bank)
                atomic_json(decision, out / "input_patch_metrics.json")
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


def reserve_budget(out, startup_seconds=0.0):
    budget = (
        read_json(out / "budget.json")
        if (out / "budget.json").exists()
        else {"used_seconds": 0.0, "attempts": []}
    )
    used = budget["used_seconds"]
    if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
        raise ValueError(
            "Cumulative24h budget exhausted or unresolved supervisor attempt; export only"
        )
    if not math.isfinite(startup_seconds) or startup_seconds < 0:
        raise ValueError("Invalid pre-reservation source qualification time")
    used += startup_seconds
    if used >= SESSION_CAP:
        budget["used_seconds"] = SESSION_CAP
        budget["attempts"].append(
            {
                "started_unix": time.time(),
                "source_qualification_seconds": startup_seconds,
                "elapsed_seconds": 0.0,
                "total_elapsed_seconds": startup_seconds,
                "exit_code": 124,
                "phase": "read_only_source_qualification_exhausted_budget",
            }
        )
        atomic_json(budget, out / "budget.json")
        raise ValueError("Read-only source qualification exhausted cumulative24h budget")
    remaining = SESSION_CAP - used
    attempt = {
        "started_unix": time.time(),
        "reserved_seconds": remaining,
        "source_qualification_seconds": startup_seconds,
    }
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
    startup_seconds = 0.0
    if not (out / "request.json").exists():
        # The data adapter confines legacy identity writes to private scratch.
        # Reject actual raw/ancestor outputs before creating any run files.
        qualified_at = time.monotonic()
        data.verify(source, out)
        startup_seconds = time.monotonic() - qualified_at
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
        budget, used, remaining = reserve_budget(out, startup_seconds)
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
        budget["attempts"][-1].update(
            elapsed_seconds=elapsed,
            total_elapsed_seconds=startup_seconds + elapsed,
            exit_code=code,
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
