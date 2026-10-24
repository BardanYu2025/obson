"""F01/F05 content protection with fixed plain architecture; real execution requires AutoDL CUDA.

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

from . import content_guard_evaluate as evaluation
from . import content_guard_training as training
from . import layer_supervision as parent_model
from . import layer_supervision_run as parent
from . import layer_supervision_training as parent_training
from . import multiscale_content as content
from . import multiscale_data as data
from . import recovery_audit as runtime
from . import state_organization as model
from . import state_organization_source as sources
from .ae_extend import atomic_json, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-content-guard768-v1"
MODULE = "obson.babel.content_guard_run"
EPOCHS = 60
BATCH = 128
MICRO = 8
SESSION_CAP = 24 * 3600
REPO = Path(__file__).resolve().parents[3]


def jobs():
    return [
        {"name": f"{arm}_s{seed}", "arm": arm, "seed": seed}
        for arm, seed in (
            ("control", 42),
            ("fixed", 42),
            ("feedback", 42),
            ("feedback", 43),
            ("fixed", 43),
            ("control", 43),
        )
    ]


def model_contract(arm=None):
    if arm is not None and arm not in training.ARMS:
        raise ValueError("Unknown content-protection objective arm")
    return {
        "objective_arm": arm,
        "encoder_arm": "plain",
        "attention_mode": "causal",
        "public_state": "one768 endpoint vector only",
        "input": "independent observed128 x28 per endpoint; causal EMA prehistory retained",
        "architecture": "unchanged4layer768/8head encoder and original HistoryQuery",
        "decoder": "seven-field age1..127 from final768 only; no raw/target bypass",
    }


def protocol():
    return {
        "task_ids": ["F01", "F05", "F11"],
        "jobs": jobs(),
        "epochs": EPOCHS,
        "source_epochs": 180,
        "source_run_epoch": 60,
        "source_updates": 6840,
        "batch": BATCH,
        "micro": MICRO,
        "workers": 1,
        "session_cap_seconds": SESSION_CAP,
        "encoder": model.CONFIG,
        "source": "reviewed layer_supervision control local last60 cumulative180; original Adam retained",
        "source_pins": {
            "manifest": sources.MANIFEST_SHA256,
            "completion": sources.COMPLETION_SHA256,
            "checkpoints": {str(k): v for k, v in sources.CHECKPOINT_SHA256.items()},
        },
        "counts": data.COUNTS,
        "sampling": "uniform5 without replacement; absolute181..240; each endpoint own128",
        "labels": "unchanged rawfloat64 relative close, body/activity masks and statistics",
        "Q": ".4path+.2change1+.1body+.3activity; SmoothL1; equal available age bands",
        "objectives": {
            "control": "original Q",
            "fixed": "Q + Q0/21 sum_j (risk_j / reference_j - 1); fixed lambda1",
            "feedback": "same objective with epoch-frozen lambda_j; initialize1; projected ascent eta0.1 cap10 toward risk/reference1",
        },
        "protection": {
            "group_names": list(training.GROUP_NAMES),
            "bound_ratio": 1.0,
            "initial_multipliers": {
                arm: training.initial_multipliers(arm) for arm in training.ARMS
            },
            "dual_rate": training.DUAL_RATE,
            "dual_cap": training.DUAL_CAP,
            "objective_multiplier": "Q0/21",
            "feasibility_numeric_tolerance": 1e-6,
            "source_calibration_states": data.COUNTS["train"] * 5 * 2,
            "final_train_panel_states": data.COUNTS["train"] * 5 * len(jobs()),
            "final_model_feasibility": "actual frozen last60 panel, not online epoch mean",
            "first_epoch_parity": "fixed vs feedback physical risks/components numeric replay; signatures retained as exact diagnostic",
        },
        "reference": "one immutable per-seed train-only source180 panel: all4789 parents, fixed5 uniform positions at absolute180; Q0 plus21physical field/age-band MSE and supports",
        "controller": "online epoch risks update only next epoch; exact persisted trajectory; fixed final model re-evaluated on original train calibration panel",
        "model_contract": model_contract(),
        "optimizer": {
            "name": "AdamW",
            "restore": "all6840 moments/steps and RNG retained; no reset",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "inherited_lr": 1.5e-5,
            "clip": 1.0,
            "clip_scope": "global inherited encoder and query",
        },
        "runtime": runtime.RUNTIME_PROFILES["context"],
        "precision": "FP32 forward/parameters, float64 truth/risks; noAMP/TF32",
        "selection": "fixed additional last60 only; source180 descriptive reference; no best, LR or multiplier search",
        "validation_positions": list(model.VAL_POSITIONS),
        "evaluation_positions": list(range(1, 129)),
        "prediction_export_positions": list(evaluation.SAMPLE_POSITIONS),
        "evaluation_models": [f"{a}_s{s}" for a in ("initial", *training.ARMS) for s in (42, 43)],
        "comparisons": [
            "feedback_vs_fixed primary",
            "feedback_vs_control required",
            "feedback_vs_initial protection only",
        ],
        "acceptance": "feedback must pass BOTH fixed and control: both seeds/cohorts5%price paired confidence benefit, inherited10%physical/F03/P95 guards; initial content protection and final train-panel feasibility also required; no generic promotion",
        "bootstrap": {
            "draws": 2000,
            "seed": 20261012,
            "groups": ["month", "contract_period_month"],
            "min_parents": 100,
            "min_groups": 6,
        },
        "F11": "zero readout fits; content retention and actual compute costs; generic reuse unknown",
        "budget_interpretation": "six workers60=360worker epochs/13680updates; same data/parameters/exposure, measured cost not exact FLOPs",
        "attribution_limit": "soft feedback constraint recipe, not guaranteed hard feasibility or proof of gradient conflict; encoder and original query adapt jointly",
        "zero_update_retry_policy": "An interrupted final train panel may rerun within the same cumulative24h budget; attempt receipts retained and planned panel counts exclude retries. Calibration and research extraction keep their own bounded refusal policy; uncertain updates never repeat.",
        "stopping": "fixed60/24h cumulative; source/nonfinite/causality/resource failure stops and exports; uncertain updates never replay; no automatic extension/grid",
        "scope": "existing research data only; no future prediction or Macro promotion",
    }


def implementation():
    # Include the inherited source verifier and model/target code, not mutable docs.
    from . import (
        content_guard_export,
        history_price_truth,
        history_query,
        multiscale_evaluate,
        multiscale_metrics,
        multiscale_training,
        observed_history_export,
        observed_history_export_v2,
        overlap_content_audit,
        state_coverage,
        state_organization_training,
        uniform_context,
    )

    return {
        "inherited": data.original.diagnostic.implementation(),
        "source_implementation": parent.implementation(),
        "source_restore": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (sources, parent, parent_model, parent_training)
        },
        "data": data.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (
                data,
                content,
                training,
                evaluation,
                model,
                uniform_context,
                multiscale_evaluate,
                multiscale_metrics,
                content_guard_export,
                observed_history_export,
                observed_history_export_v2,
                runtime,
                history_query,
                history_price_truth,
                multiscale_training,
                state_coverage,
                state_organization_training,
                overlap_content_audit,
            )
        }
        | {Path(__file__).name: sha256(__file__)},
        "script": sha256(REPO / "scripts/babel_content_guard768_autodl.sh"),
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
        "model_contract": model_contract(job["arm"]),
        "job": job,
        "reference_sha256": sha256(out / f"training_reference_s{job['seed']}.json"),
    }


def reference_binding(meta, out, seed):
    raw = read_json(out / "multiscale_raw_train.json")
    return {
        "seed": seed,
        "source_identity": meta["source_identity"],
        "source_signature": meta["source_identity"]["controls"][str(seed)]["signature"],
        "raw_receipt_sha256": sha256(out / "multiscale_raw_train.json"),
        "raw_sha256": raw["raw_sha256"],
        "rows_sha256": raw["rows_file_sha256"],
        "statistics": meta["statistics"],
    }


def load_reference(meta, out, seed):
    path = out / f"training_reference_s{seed}.json"
    reference = read_json(path)
    lock_path = out / "training_reference_lock.json"
    if lock_path.exists() and read_json(lock_path).get("files", {}).get(path.name) != sha256(path):
        raise ValueError("Frozen train reference fingerprint changed")
    training.validate_reference(reference, seed=seed, parents=meta["counts"]["train"])
    if reference.get("binding") != reference_binding(meta, out, seed):
        raise ValueError("Train-only reference lineage changed")
    if reference["source_signature"] != reference["binding"]["source_signature"]:
        raise ValueError("Train reference does not use selected source signature")
    return reference


@torch.no_grad()
def prepare_references(meta, out, x, raw, device):
    """Freeze train-only source risks, with no weight updates and no val/research fitting."""
    for seed in (42, 43):
        path = out / f"training_reference_s{seed}.json"
        pending = out / f"training_reference_s{seed}.pending.json"
        if path.exists():
            load_reference(meta, out, seed)
            if pending.exists():
                pending.unlink()
            continue
        if pending.exists():
            raise ValueError(
                "Interrupted calibration without committed reference; preserve attempt"
            )
        atomic_json(reference_binding(meta, out, seed), pending)
        encoder, query = load_pair(meta, {"arm": "initial", "seed": seed}, device)
        opt = training.new_optimizer(encoder, query)
        restore_source(meta, {"arm": "initial", "seed": seed}, encoder, query, opt)
        del opt
        before = training.parameter_signature(encoder, query)
        progress(
            f"Train-only source reference s{seed}: fixed5 positions over all{len(x)} parents, zero updates"
        )
        reference = training.fit_reference(
            encoder, query, x, raw, meta["statistics"], seed=seed, device=device, micro_size=MICRO
        )
        if (
            training.parameter_signature(encoder, query) != before
            or reference["source_signature"] != before
        ):
            raise ValueError("Source changed while fitting train-only reference")
        reference["binding"] = reference_binding(meta, out, seed)
        immutable_json(reference, path)
        load_reference(meta, out, seed)
        pending.unlink()
        del encoder, query
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    immutable_json(
        {
            "source_identity": meta["source_identity"],
            "files": {
                f"training_reference_s{seed}.json": sha256(out / f"training_reference_s{seed}.json")
                for seed in (42, 43)
            },
        },
        out / "training_reference_lock.json",
    )


def history_check(history, job, n, batch=BATCH, *, reference=None):
    if len(history) > EPOCHS:
        raise ValueError("History exceeds fixed epoch budget")
    multipliers = training.initial_multipliers(job["arm"])
    for epoch, item in enumerate(history, 1):
        if reference is None:
            raise ValueError("Bound reference required for controller history")
        support = np.asarray(item["group_support"], dtype=np.float64)
        if (
            support.shape != (21,)
            or not np.isfinite(support).all()
            or (support <= 0).any()
            or (support > n * 5).any()
            or not np.array_equal(support, support.astype(np.int64))
        ):
            raise ValueError("Epoch physical-risk support is invalid")
        risks = np.asarray(item["group_risks"], dtype=np.float64)
        if risks.shape != (21,) or not np.isfinite(risks).all() or (risks < 0).any():
            raise ValueError("Epoch physical risks are invalid")
        ratios = risks / np.asarray(reference["baseline"], dtype=np.float64)
        if not np.allclose(ratios, item["group_ratios"], atol=1e-12, rtol=1e-12):
            raise ValueError("Controller risk ratios differ from bound reference")
        if item["multipliers_used"] != multipliers:
            raise ValueError("Controller multipliers changed between committed epochs")
        multipliers = training.update_multipliers(job["arm"], multipliers, item["group_ratios"])
        if item["multipliers_next"] != multipliers:
            raise ValueError("Controller trajectory differs from fixed recurrence")
        order, ps = training.schedule(n, job["seed"], epoch)
        expected = {
            "arm": job["arm"],
            "epoch": epoch,
            "global_epoch": 180 + epoch,
            "head_passes": 1,
            "steps": math.ceil(n / batch),
            "examples": n,
            "supervised_states": n * 5,
            "query_state_reads": n * 5,
            "lr": training.learning_rate(epoch),
            "batch_size": batch,
            "micro_size": MICRO,
            "order_hash": training.array_hash(order),
            "prefix_hash": training.array_hash(ps),
            "position_counts": np.bincount(ps.ravel() - 1, minlength=128).tolist(),
        }
        if any(item.get(k) != v for k, v in expected.items()):
            raise ValueError(f"History recipe/exposure mismatch at epoch{epoch}")
        contract = item.get("model_contract", {})
        if (
            contract.get("arm") != "plain"
            or contract.get("attention_mode") != "causal"
            or contract.get("endpoint_only") is not True
            or contract.get("input_length") != 128
        ):
            raise ValueError("History model attention contract differs")
        # Every numeric reported value must serialize as finite JSON.
        json.dumps(item, allow_nan=False)
        if (
            not isinstance(item.get("parameter_signature"), str)
            or len(item["parameter_signature"]) != 64
        ):
            raise ValueError("Epoch parameter signature is missing")
        if bool(item.get("validation") is not None) != (epoch % 5 == 0 or epoch == EPOCHS):
            raise ValueError("Validation cadence changed")
    return len(history) * math.ceil(n / batch)


def validate_optimizer(opt, encoder, query, epoch, steps):
    if not 0 <= epoch <= EPOCHS or steps != epoch * math.ceil(data.COUNTS["train"] / BATCH):
        raise ValueError("Additional optimizer accounting differs")
    # Preserve every inherited moment/order/dtype and6840+new updates.
    training.validate_optimizer(opt, encoder, query, epoch, steps)


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
def validation(meta, encoder, query, x, raw, arm, device, epoch=0):
    return training.validation(meta, encoder, query, x, raw, arm, device, epoch=epoch)


def load_pair(meta, job, device):
    if job["arm"] not in ("initial", *training.ARMS):
        raise ValueError("Unknown objective arm")
    encoder, query = model.make_models("plain", job["seed"], device, meta["encoder"])
    if encoder.arm != "plain" or encoder.attention_mode != "causal":
        raise ValueError("All objective arms require the original plain causal architecture")
    return encoder, query


def _cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def check_job(meta, out, job):
    folder = out / job["name"]
    done = read_json(folder / "completion.json")
    if done["binding"] != binding(out, job) or done["epoch"] != EPOCHS:
        raise ValueError("Worker completion is not fixed last60")
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
        "source_restore.json",
        "final_train_risk.json",
        "final_train_risk_attempts.json",
    }
    if set(done["files"]) != required:
        raise ValueError("Incomplete fixed worker evidence")
    summary = read_json(folder / "training_summary.json")
    n = meta["counts"]["train"]
    reference = load_reference(meta, out, job["seed"])
    steps = history_check(summary["history"], job, n, BATCH, reference=reference)
    if (
        len(summary["history"]) != EPOCHS
        or summary["epoch"] != EPOCHS
        or summary["updates"] != steps
        or done["updates"] != steps
        or summary.get("cumulative_updates") != 6840 + steps
        or done.get("cumulative_updates") != 6840 + steps
        or summary["examples"] != n * EPOCHS
        or summary["supervised_states"] != n * EPOCHS * 5
        or summary.get("query_state_reads") != n * EPOCHS * 5
        or any(summary[k] != done[k] for k in ("binding", "initial_signature", "signature"))
    ):
        raise ValueError("Completed worker exposure/signature mismatch")
    restored = read_json(folder / "source_restore.json")
    if (
        restored.get("source_identity") != meta["source_identity"]
        or restored.get("source_seed") != job["seed"]
        or restored.get("source_updates") != 6840
        or restored.get("signature") != summary["initial_signature"]
        or restored.get("signature")
        != meta["source_identity"]["controls"][str(job["seed"])]["signature"]
        or restored.get("complete_inherited_adam_restored") is not True
        or restored.get("optimizer_reset") is not False
    ):
        raise ValueError("Source optimizer restoration receipt differs")
    lines = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    receipt = read_json(folder / "checkpoint.json")
    if (
        lines != summary["history"]
        or receipt["epoch"] != EPOCHS
        or receipt["binding"] != done["binding"]
        or receipt["sha256"] != done["files"]["last.pt"]
    ):
        raise ValueError("Completed checkpoint/history receipts differ")
    final_risk = read_json(folder / "final_train_risk.json")
    if (
        final_risk.get("checkpoint_sha256") != done["files"]["last.pt"]
        or final_risk.get("reference_sha256") != done["binding"]["reference_sha256"]
        or final_risk.get("model_signature") != done["signature"]
        or final_risk.get("binding") != done["binding"]
    ):
        raise ValueError("Final frozen-model risk evidence is bound to another model/reference")
    attempts = read_json(folder / "final_train_risk_attempts.json")
    if (
        not attempts
        or final_risk.get("panel_attempt") != len(attempts)
        or attempts[-1].get("status") != "complete"
    ):
        raise ValueError("Final train-panel attempt receipt is incomplete")
    expected_panel_binding = {
        key: final_risk[key]
        for key in ("binding", "checkpoint_sha256", "reference_sha256", "model_signature")
    }
    if any(attempt.get("binding") != expected_panel_binding for attempt in attempts):
        raise ValueError("Final train-panel retry changed model/reference")
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
    opt = training.new_optimizer(encoder, query)
    restored = restore_source(meta, job, encoder, query, opt)
    initial = restored["signature"]
    bind = binding(out, job)
    reference = load_reference(meta, out, job["seed"])
    multipliers = training.initial_multipliers(job["arm"])
    ck = recover_checkpoint(folder, bind)
    if ck is None:
        if any(p.name not in ("worker.lock",) for p in folder.iterdir()):
            raise ValueError("Unbound worker output; do not overwrite")
        atomic_json(restored, folder / "source_restore.json")
        audit = model.audit(
            encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"]
        )
        atomic_json(audit, folder / "causal_audit.json")
        if not audit["passed"]:
            raise ValueError("Restored-source causality/target audit failed")
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
            "reference_sha256": bind["reference_sha256"],
            "multipliers": multipliers,
        }
        commit_checkpoint(folder, ck)
    else:
        if ck["initial_signature"] != initial or ck["epoch"] != len(ck["history"]):
            raise ValueError("Source initialization or history length changed")
        history, epoch = ck["history"], ck["epoch"]
        encoder.load_state_dict(ck["encoder"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
        if training.parameter_signature(encoder, query) != ck["signature"]:
            raise ValueError("Committed parameter signature changed")
        opt.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
    initial_validation = ck["initial_validation"]
    expected_multipliers = (
        history[-1]["multipliers_next"] if history else training.initial_multipliers(job["arm"])
    )
    if (
        ck.get("reference_sha256") != bind["reference_sha256"]
        or ck.get("multipliers") != expected_multipliers
    ):
        raise ValueError("Committed controller/reference changed")
    multipliers = ck["multipliers"]
    steps = history_check(history, job, len(x), BATCH, reference=reference)
    validate_optimizer(opt, encoder, query, epoch, steps)
    publish_history(folder, history)
    if history:
        first_epoch_parity(out, job["seed"])
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
            reference=reference,
            multipliers=multipliers,
        )
        report["validation"] = (
            validation(meta, encoder, query, val, val_raw, job["arm"], device, current)
            if current % 5 == 0 or current == EPOCHS
            else None
        )
        report["parameter_signature"] = training.parameter_signature(encoder, query)
        history.append(report)
        multipliers = report["multipliers_next"]
        steps = history_check(history, job, len(x), BATCH, reference=reference)
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
            "reference_sha256": bind["reference_sha256"],
            "multipliers": multipliers,
        }
        commit_checkpoint(folder, state)
        del state
        publish_history(folder, history)
        if current == 1:
            first_epoch_parity(out, job["seed"])
        (folder / "pending_epoch.json").unlink()
        progress(
            f"{job['name']}: {current}/{EPOCHS}, Q={report['components']['query']:.6f}, "
            f"objective={report['loss']:.6f}, online risk max={max(report['group_ratios']):.3f}, "
            f"online above1={sum(v > 1 for v in report['group_ratios'])}/21, "
            f"lambda max={max(report['multipliers_next']):.3f}, epoch={report['seconds']:.1f}s; fixed last selection"
        )
    final_audit = model.audit(
        encoder, query, torch.as_tensor(x[:2], device=device), meta["statistics"]
    )
    atomic_json(final_audit, folder / "final_causal_audit.json")
    if not final_audit["passed"]:
        raise ValueError("Trained-model causality/target audit failed")
    final_path = folder / "final_train_risk.json"
    final_binding = {
        "binding": bind,
        "checkpoint_sha256": sha256(folder / "last.pt"),
        "reference_sha256": bind["reference_sha256"],
        "model_signature": training.parameter_signature(encoder, query),
    }
    if final_path.exists():
        final_risk = read_json(final_path)
        if any(final_risk.get(k) != v for k, v in final_binding.items()):
            raise ValueError("Final frozen-model train-risk binding differs")
    else:
        progress(f"{job['name']}: fixed final60 train-panel feasibility, zero updates")
        attempts_path = folder / "final_train_risk_attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        attempts.append(
            {"binding": final_binding, "started_unix": time.time(), "status": "running"}
        )
        atomic_json(attempts, attempts_path)
        try:
            final_risk = training.evaluate_reference(
                encoder,
                query,
                x,
                raw,
                meta["statistics"],
                reference,
                seed=job["seed"],
                device=device,
                micro_size=MICRO,
            )
        except Exception as error:
            attempts[-1].update(status="failed", error=str(error), finished_unix=time.time())
            atomic_json(attempts, attempts_path)
            raise
        attempts[-1].update(status="complete", finished_unix=time.time())
        atomic_json(attempts, attempts_path)
        final_risk["panel_attempt"] = len(attempts)
        if training.parameter_signature(encoder, query) != final_binding["model_signature"]:
            raise ValueError("Final feasibility changed model weights")
        atomic_json(final_risk | final_binding, final_path)
    progress(
        f"{job['name']}: final frozen train panel max risk={max(final_risk['group_ratios']):.3f}, within source={sum(v <= 1 + 1e-6 for v in final_risk['group_ratios'])}/21"
    )
    summary = {
        "binding": bind,
        "epoch": EPOCHS,
        "initial_signature": initial,
        "signature": training.parameter_signature(encoder, query),
        "history": history,
        "initial_validation": initial_validation,
        "cumulative_updates": 6840 + steps,
        "updates": steps,
        "examples": len(x) * EPOCHS,
        "supervised_states": len(x) * EPOCHS * 5,
        "query_state_reads": len(x) * EPOCHS * 5,
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
            "source_restore.json",
            "final_train_risk.json",
            "final_train_risk_attempts.json",
        )
    }
    done = {
        "binding": bind,
        "epoch": EPOCHS,
        "initial_signature": initial,
        "signature": summary["signature"],
        "updates": steps,
        "cumulative_updates": 6840 + steps,
        "files": files,
    }
    atomic_json(done, folder / "completion.json")
    del encoder, query, opt
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def first_epoch_parity(out, seed, *, required=False):
    """Check matched first updates before feedback can change the next epoch."""
    paths = [out / f"{arm}_s{seed}" / "history.jsonl" for arm in ("fixed", "feedback")]
    if not all(path.exists() and path.stat().st_size for path in paths):
        if required:
            raise ValueError("Both first epochs required for matched feedback parity")
        return None
    fixed, feedback = [json.loads(path.read_text().splitlines()[0]) for path in paths]
    checks = compare_replay(
        {key: feedback[key] for key in ("group_risks", "components")},
        {key: fixed[key] for key in ("group_risks", "components")},
    )
    exact = {
        key: fixed[key] == feedback[key]
        for key in ("multipliers_used", "group_support", "prefix_hash", "order_hash")
    }
    passed = all(check["passed"] for check in checks.values()) and all(exact.values())
    result = {
        "seed": seed,
        "passed": passed,
        "checks": checks,
        "exact": exact,
        "atol": 1e-6,
        "rtol": 2e-5,
        "signature_exact_equal": fixed["parameter_signature"] == feedback["parameter_signature"],
        "fixed_signature": fixed["parameter_signature"],
        "feedback_signature": feedback["parameter_signature"],
        "scope": "same first-epoch objective/source/order; numeric engineering check, not scientific benefit",
    }
    immutable_json(result, out / f"first_epoch_parity_s{seed}.json")
    if not passed:
        raise ValueError(
            "Fixed/feedback first epoch differs before feedback can act; numeric evidence saved"
        )
    return result


def lock_models(meta, out):
    first_epoch_checks = {}
    matrix = {j["name"]: check_job(meta, out, j) for j in jobs()}
    for seed in (42, 43):
        matrix[f"initial_s{seed}"] = check_initial(meta, out, seed)
    for seed in (42, 43):
        if len({matrix[f"{arm}_s{seed}"]["initial_signature"] for arm in training.ARMS}) != 1:
            raise ValueError("Matched-seed source tensors differ")
        if (
            matrix[f"control_s{seed}"]["initial_signature"]
            != matrix[f"initial_s{seed}"]["signature"]
        ):
            raise ValueError("Matched objective arms differ from selected initial reference")
        first_epoch_checks[str(seed)] = first_epoch_parity(out, seed, required=True)
    value = {
        "schema": SCHEMA,
        "selection": "fixed_last60",
        "manifest": sha256(out / "manifest.json"),
        "models": matrix,
        "source": meta["source_identity"],
        "readout_fits": 0,
        "references": {
            str(seed): sha256(out / f"training_reference_s{seed}.json") for seed in (42, 43)
        },
        "reference_lock": sha256(out / "training_reference_lock.json"),
        "fixed_feedback_first_epoch_equal": True,
        "first_epoch_parity": first_epoch_checks,
    }
    immutable_json(value, out / "model_selection_lock.json")
    return value


def verify_lock(meta, out, name, split):
    if name not in {j["name"] for j in evaluation_jobs()} or split not in (
        "test",
        "cross_research",
    ):
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
    """Measure each objective arm with zero optimizer updates.

    All timings use the arm-specific encoder, including validation9 and full128
    queries. The conservative estimate charges every remaining epoch at the
    slower measured arm; equal updates are not claimed as equal wall time.
    """
    features = torch.as_tensor(x[:MICRO], device=device)
    closes = torch.as_tensor(raw[:MICRO], device=device)
    positions = torch.as_tensor(training.schedule(len(x), 42, 1)[1][:MICRO], device=device)
    research = meta["counts"]["test"] + meta["counts"]["cross_research"]
    full_predictions = 8 * research * 128 * 127 * 7 * 4
    selected = 8 * research * 8 * 127 * 7 * (4 + 8 + 1)
    disk_free = shutil.disk_usage(out).free
    measurements, arms = {}, {}
    reference = load_reference(meta, out, 42)
    for arm in training.ARMS:
        progress(f"Zero-update preflight: {arm}, backward and validation9/full128 timing")
        encoder, query = load_pair(meta, {"arm": arm, "seed": 42}, device)
        opt = training.new_optimizer(encoder, query)
        restore_source(meta, {"arm": arm, "seed": 42}, encoder, query, opt)
        del opt
        signature = training.parameter_signature(encoder, query)
        parameters = list(encoder.parameters()) + list(query.parameters())
        parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        initial_components = None
        preclip = None
        contribution_check = None
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
            initial_components = content.detached_summary(
                {k: v for k, v in terms.items() if k not in ("group_rows", "group_support")}
            )
            objective = training.batch_objective(
                terms, reference, training.initial_multipliers(arm), arm
            )
            counts = terms["group_support"].sum((0, 1))
            contributions = [
                training.batch_objective(
                    {key: value[start : start + 1] for key, value in terms.items()},
                    reference,
                    training.initial_multipliers(arm),
                    arm,
                    counts=counts,
                    contexts=terms["query"].numel(),
                )
                for start in range(len(features))
            ]
            accumulated = torch.stack(contributions).sum()
            contribution_check = {
                "full": float(objective.detach().cpu()),
                "sum_contributions": float(accumulated.detach().cpu()),
                "passed": bool(torch.allclose(objective, accumulated, atol=1e-10, rtol=1e-8)),
            }
            if not contribution_check["passed"]:
                raise ValueError("Effective-batch support/normalization differs before updates")
            objective.backward()
            del objective, accumulated, contributions, counts
            preclip = float(
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                .detach()
                .cpu()
            )
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
            "source_final_Q_components": initial_components,
            "source_preclip_norm": preclip,
            "effective_batch_contribution_check": contribution_check,
            "objective_arm": arm,
            "reference_sha256": sha256(out / "training_reference_s42.json"),
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
        raise ValueError("Evaluate-only requires all six fixed60 workers complete")
    train_seconds = (
        epochs_left
        * math.ceil(len(x) / MICRO)
        * max(measurements[a + "_micro_backward_seconds"] for a in training.ARMS)
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
        (validation_calls + 2)
        * math.ceil(meta["counts"]["val"] / MICRO)
        * max(measurements[a + "_validation9_micro_seconds"] for a in training.ARMS)
    )
    eval_seconds = (
        8
        * sum(math.ceil(meta["counts"][s] / MICRO) for s in ("test", "cross_research"))
        * max(measurements[a + "_evaluation128_micro_seconds"] for a in training.ARMS)
    )
    save_seconds = max(v["weight_save_fsync_seconds"] or 0.0 for v in arms.values())
    save_total = save_seconds * 4 * (epochs_left + 6)
    # Six final fixed5 train-panel passes are additional zero-update evidence.
    final_panel_seconds = (
        6
        * math.ceil(len(x) / MICRO)
        * max(measurements[a + "_validation9_micro_seconds"] for a in training.ARMS)
    )
    estimate = (
        train_seconds + val_seconds + eval_seconds + final_panel_seconds + save_total
    ) * 1.3 + 1800
    checks = {
        "zero_updates_same_parameters": all(
            v["initial_signature"] == v["final_signature"] for v in arms.values()
        ),
        "matched_plain_initialization": len({v["initial_signature"] for v in arms.values()}) == 1,
        "matched_parameter_count": len({v["parameter_count"] for v in arms.values()}) == 1,
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
        f"Zero-update content-protection preflight: estimated {estimate / 3600:.2f}h, "
        f"disk need {result['disk_required_bytes'] / 2**30:.2f}GiB/free {disk_free / 2**30:.2f}GiB, "
        f"GPU need {result['gpu_required_bytes'] / 2**30:.2f}GiB"
    )
    del features, closes
    gc.collect()
    torch.cuda.empty_cache()
    if not result["passed"]:
        raise ValueError(f"Resource/numeric preflight stopped without updates: {checks}")
    return result


def evaluation_jobs():
    return [{"name": f"initial_s{s}", "arm": "initial", "seed": s} for s in (42, 43)] + jobs()


def restore_source(meta, job, encoder, query, opt):
    ck = sources.checkpoint(meta, job["seed"])
    migrated = training.restore_source(encoder, query, opt, ck)
    restore_rng(ck["rng"])
    return {
        **migrated,
        "source_identity": meta["source_identity"],
        "source_seed": job["seed"],
        "source_epoch": 180,
        "source_run_epoch": 60,
        "source_updates": 6840,
        "signature": training.parameter_signature(encoder, query),
        "optimizer_reset": False,
        "lr": opt.param_groups[0]["lr"],
    }


def compare_replay(actual, expected, path=""):
    checks = {}
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected) <= set(actual):
            raise ValueError("Source replay keys differ: " + path)
        for key, value in expected.items():
            checks.update(compare_replay(actual[key], value, path + "/" + key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Source replay sequence differs: " + path)
        for i, value in enumerate(expected):
            checks.update(compare_replay(actual[i], value, path + f"/{i}"))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        checks[path] = {
            "actual": actual,
            "expected": expected,
            "passed": bool(np.isclose(actual, expected, atol=1e-6, rtol=2e-5)),
        }
    else:
        checks[path] = {"actual": actual, "expected": expected, "passed": actual == expected}
    return checks


def check_initial(meta, out, seed):
    folder = out / f"initial_s{seed}"
    receipt = read_json(folder / "checkpoint.json")
    if (
        receipt["source_identity"] != meta["source_identity"]
        or receipt["seed"] != seed
        or receipt["epoch"] != 180
    ):
        raise ValueError("Initial reference identity changed")
    audit = read_json(folder / "causal_audit.json")
    if not audit["passed"] or not all(audit["checks"].values()):
        raise ValueError("Initial source causal audit failed")
    qualification = read_json(out / "source_validation.json")
    verify_files(out, qualification["files"])
    if not qualification["passed"] or qualification["source_identity"] != meta["source_identity"]:
        raise ValueError("Initial source qualification changed")
    return receipt | {
        "files": {name: sha256(folder / name) for name in ("checkpoint.json", "causal_audit.json")},
        "updates": 0,
    }


@torch.no_grad()
def qualify_source(meta, out, train, val, raw, device):
    path = out / "source_validation.json"
    if path.exists():
        receipt = read_json(path)
        if receipt["source_identity"] != meta["source_identity"] or not receipt["passed"]:
            raise ValueError("Saved source validation did not pass; preserve failed attempt")
        verify_files(out, receipt["files"])
        return receipt
    results = {}
    for seed in (42, 43):
        folder = out / f"initial_s{seed}"
        folder.mkdir(exist_ok=True)
        if any(folder.iterdir()):
            raise ValueError("Incomplete source qualification; no silent repeated replay")
        ck = sources.checkpoint(meta, seed)
        old_e, old_q = parent_model.make_models("control", seed, device, meta["encoder"])
        old_e.load_state_dict(ck["encoder"], strict=True)
        old_q.load_state_dict(ck["query"], strict=True)
        opt = parent_training.new_optimizer(old_e, old_q)
        opt.load_state_dict(ck["optimizer"])
        parent.validate_optimizer(opt, old_e, old_q, 60, 2280)
        old_e.eval()
        old_q.eval()
        replay = parent.validation(meta, old_e, old_q, val, raw, "control", device, epoch=60)
        expected = ck["history"][-1]["validation"]
        checks = compare_replay(replay, expected)
        x = torch.as_tensor(train[:2], device=device)
        ps = torch.tensor([1, 32, 64, 128], device=device).expand(len(x), -1)
        before, pred = parent_model.predict(old_e, old_q, x, ps)
        for arm in training.ARMS:
            e, q = model.make_models("plain", seed, device, meta["encoder"])
            new_opt = training.new_optimizer(e, q)
            training.restore_source(e, q, new_opt, ck)
            del new_opt
            e.eval()
            q.eval()
            after, output = model.predict(e, q, x, ps)
            checks[arm + "/final_state_exact"] = {"passed": bool(torch.equal(before, after))}
            checks[arm + "/final_prediction_exact"] = {"passed": bool(torch.equal(pred, output))}
            del e, q
        passed = all(c["passed"] for c in checks.values())
        atomic_json({"passed": passed, "checks": checks}, folder / "source_replay.json")
        if not passed:
            raise ValueError(
                "Source validation/parity replay differs; saved exact numeric evidence"
            )
        e, q = model.make_models("plain", seed, device, meta["encoder"])
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
        audit = model.audit(e, q, x, meta["statistics"])
        atomic_json(audit, folder / "causal_audit.json")
        if not audit["passed"]:
            raise ValueError("Restored source causal audit failed")
        receipt = {
            "seed": seed,
            "epoch": 180,
            "source_run_epoch": 60,
            "signature": ck["signature"],
            "source_identity": meta["source_identity"],
            "source_path": str(Path(meta["model_source"]) / f"control_s{seed}" / "last.pt"),
            "optimizer_restored": True,
        }
        atomic_json(receipt, folder / "checkpoint.json")
        results[str(seed)] = {
            "passed": True,
            "source_signature": ck["signature"],
            "validation": replay,
        }
        del ck, old_e, old_q, opt, e, q, x, before, pred, after, output
        gc.collect()
        torch.cuda.empty_cache()
    files = {
        str(p.relative_to(out)): sha256(p)
        for seed in (42, 43)
        for p in (out / f"initial_s{seed}").glob("*.json")
    }
    result = {
        "passed": True,
        "source_identity": meta["source_identity"],
        "results": results,
        "files": files,
        "updates": 0,
    }
    immutable_json(result, path)
    return result


def cost_summary(meta, out):
    measurements = {}
    for job in jobs():
        history = read_json(out / job["name"] / "training_summary.json")["history"]
        measurements[job["name"]] = {
            "updates": sum(x["steps"] for x in history),
            "training_seconds": sum(x["seconds"] for x in history),
            "parent_exposures": sum(x["examples"] for x in history),
            "state_exposures": sum(x["supervised_states"] for x in history),
        }
    return {
        "task_id": "F11",
        "workers": measurements,
        "public_state_float32_bytes": 768 * 4,
        "private_memory_float32_bytes_per_endpoint": 0,
        "private_memory_scope": "no new private memory; original token activations retained",
        "raw_128_close_float32_bytes": 128 * 4,
        "prepared_128x28_float32_bytes": 128 * 28 * 4,
        "raw_comparison_scope": "close-only history is not all28 model features; no storage superiority asserted",
        "readout_fits": 0,
        "reuse_status": "not_tested_by_new_task",
        "scope": "retention and actual compute costs; generic utility unknown, no promotion",
    }


def source_identity(source):
    return sources._identity(source)[1]


def execute(source, out, evaluate_only=False):
    with (out / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if not torch.cuda.is_available():
                raise ValueError("Real execution requires AutoDL CUDA")
            remaining_seconds(out)
            progress("F05/F01: verifying reviewed control180 model and immutable data lineage")
            source_checked = sources.verify(source, out)
            verified = source_checked["verified_data"]
            meta = {
                **verified,
                "schema": SCHEMA,
                "source": source_checked["original_meta"]["source"],
                "model_source": str(source),
                "source_identity": source_checked["identity"],
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
                qualify_source(meta, out, train, val, val_raw, "cuda")
                prepare_references(meta, out, train, raw, "cuda")
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
                                recovered["history"],
                                job,
                                len(train),
                                BATCH,
                                reference=load_reference(meta, out, job["seed"]),
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
                progress(
                    "Six continued models and two immutable initial references locked; preparing research cohorts"
                )
                data.prepare(meta, out, ("test",))
                data.prepare(meta, out, ("cross_research",))
                banks, rows_bank = {}, {}
                eval_meta = meta
                for split in ("test", "cross_research"):
                    x, rows = data.cache(meta, split)
                    close = data.closes(out, split)
                    banks[split], rows_bank[split] = {}, rows
                    for job in evaluation_jobs():
                        verify_lock(eval_meta, out, job["name"], split)
                        encoder, query = load_pair(meta, job, "cuda")
                        ck = (
                            sources.checkpoint(meta, job["seed"])
                            if job["arm"] == "initial"
                            else recover_checkpoint(out / job["name"], binding(out, job))
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
                decision = evaluation.decisions(banks, rows_bank, evaluation.training_evidence(out))
                atomic_json(decision, out / "content_guard_metrics.json")
                atomic_json(cost_summary(meta, out), out / "f11_cost.json")
            if (
                sources.verify(source, out)["verified_data"] != verified
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
            "Cumulative24h budget exhausted or unresolved supervisor attempt; export only"
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
        sources.verify(source, out)
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
