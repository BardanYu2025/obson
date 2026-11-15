"""Standalone F05/F01/F11 study: eight factorial runs plus two fresh bar controls.

Reuses the tested epoch/transaction/evaluation components explicitly. No old
artifact verification, fabricated historical metadata, or monkey-patched globals.
"""

import argparse
import fcntl
import gc
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import input_factorial as model
from . import input_factorial_evaluate as evaluation
from . import input_factorial_fresh_data as data
from . import input_factorial_run as factorial
from . import input_factorial_training as training
from . import input_patch_run as bar
from . import raw_snapshot
from . import recovery_audit as runtime
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .observed_history_run import immutable_json
from .progress import progress

SCHEMA = "babel-input-factorial-fresh768-v1"
REPO = Path(__file__).resolve().parents[3]
MODULE = "obson.babel.input_factorial_fresh_run"
CAP_SECONDS = 36 * 3600
DATA_BUNDLE = REPO / "artifacts/babel_fresh_20261115"
SNAPSHOT_SHA256 = "adc34d786321f25cf3e501e265ba54b37cad8076c9e25b22a6b02d8ae9b672bd"
DATASET_SHA256 = "b5d401b5e9fc45116f8d8ce3a34125c7c23859db778a7124c0c76512ec0658c3"


def jobs():
    arms = ("reference_bar", *model.ARMS)
    return [
        {"name": f"{arm}_s{seed}", "arm": arm, "seed": seed}
        for seed in (42, 43)
        for arm in (arms if seed == 42 else tuple(reversed(arms)))
    ]


def protocol():
    return {
        "schema": SCHEMA,
        "task_ids": ["F05", "F01", "F11"],
        "jobs": jobs(),
        "epochs": 120,
        "batch": 128,
        "micro": 8,
        "workers": 1,
        "worker_epochs": 1200,
        "updates": 45600,
        "parent_exposures": 5746800,
        "supervised_states": 28734000,
        "cap_seconds": CAP_SECONDS,
        "selection": "fixed_last120_no_best_or_seed_selection",
        "starting_point": "all10 scratch; originalQ, matched sampling/LR/Adam; no historical weight restored",
        "bar_role": "concurrent freshly trained quality baseline; fewer86784 parameters than factorial arms; not an equal-parameter structure contrast",
        "data": data.recipe(),
        "data_revision_reason": "User confirmed previous AutoDL erased and no complete backup; original fixed indexes/scalers and bar weights unavailable",
        "reference_revision_reason": "Replaces2 missing frozen bar anchors with2 concurrent scratch120 controls, +240worker_epochs; no reconstruction of old history",
        "scope": "New cohort/scales: only within-run paired conclusions, not historical numeric replay or Macro promotion",
        "acceptance": evaluation.metric_contract(),
        "runtime": runtime.RUNTIME_PROFILES["context"],
        "diagnostics": "factorial gradient panels epoch0/10/120 unchanged; bar controls originalQ validation/causality, no additional fitted heads",
        "research": "all10 last120 locked before any research model scoring; all128 endpoints; original1580 interval and20 P95 gates",
        "backup": "automatic report and complete resume backup; raw data available in git snapshot; user must download backup before instance deletion",
    }


def implementation():
    paths = sorted((REPO / "src/obson/babel").glob("*.py"))
    paths += [
        REPO / "scripts/babel_input_factorial_fresh_autodl.sh",
        REPO / "requirements-babel-fresh.txt",
    ]
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def source_identity(bundle, target):
    if (
        sha256(bundle / "manifest.json") != SNAPSHOT_SHA256
        or sha256(target / "factorial_fresh/manifest.json") != DATASET_SHA256
    ):
        raise ValueError("Release snapshot/dataset differs from the predeclared cohort")
    raw_snapshot.verify(bundle, target)
    dm = data.verify(target / "factorial_fresh")
    return {
        "snapshot": sha256(bundle / "manifest.json"),
        "dataset": sha256(target / "factorial_fresh/manifest.json"),
        "dataset_files": dm["files"],
        "historical_artifacts_required": False,
    }


def make_manifest(bundle, target):
    identity = source_identity(bundle, target)
    dm = read_json(target / "factorial_fresh/manifest.json")
    return {
        "schema": SCHEMA,
        "protocol": protocol(),
        "implementation": implementation(),
        "source_identity": identity,
        "bundle": str(bundle),
        "data_root": str(target),
        "source": str(target / "factorial_fresh"),
        "raw_root": str(target / "contracts"),
        "statistics": dm["statistics"],
        "encoder": model.CONFIG,
        "counts": data.COUNTS,
        "epochs": 120,
        "batch": 128,
        "micro": 8,
        "historical_replay": False,
    }


def worker_location(meta, out, job):
    if job["arm"] == "reference_bar":
        root = out / "bar_control"
        return (
            bar,
            read_json(root / "manifest.json"),
            root,
            {"name": f"bar_s{job['seed']}", "arm": "bar", "seed": job["seed"]},
        )
    return factorial, meta, out, job


def check_job(meta, out, job):
    module, mm, root, actual = worker_location(meta, out, job)
    done = module.check_job(mm, root, actual)
    return {
        "job": job,
        "role": "concurrent_scratch_control" if job["arm"] == "reference_bar" else "factorial_arm",
        "file": str((root / actual["name"] / "last.pt").relative_to(out)),
        "sha256": done["files"]["last.pt"],
        "signature": done["signature"],
        "epoch": 120,
        "updates": done["updates"],
        "completion": done,
    }


def prepare_initializations(meta, out, device):
    factorial.prepare_initializations(meta, out, device)
    root = out / "bar_control"
    root.mkdir(exist_ok=True)
    immutable_json(meta | {"role": "concurrent_scratch_bar_control"}, root / "manifest.json")
    for seed in (42, 43):
        encoder, query = bar.model.make_models("bar", seed, device, meta["encoder"])
        sig = training.parameter_signature(encoder, query)
        matched = read_json(out / f"initialization_s{seed}.json")
        common = {v["common_signature"] for v in matched["arms"].values()}
        # The factorial helper's hash includes named tensors, unlike parameter_signature.
        reference_hash = model._tensor_signature((encoder, query))
        if common != {reference_hash}:
            raise ValueError("Fresh bar common parameters differ from factorial arms")
        receipt = {
            "seed": seed,
            "passed": True,
            "checks": {"same_factorial_common_tensors": True},
            "arms": {"bar": {"signature": sig}},
            "role": "fresh_bar_only_not_old_three_arm_experiment",
        }
        immutable_json(receipt, root / f"initialization_s{seed}.json")
        del encoder, query
    gc.collect()
    torch.cuda.empty_cache()


def lock_models(meta, out):
    value = {
        "schema": SCHEMA,
        "manifest": sha256(out / "manifest.json"),
        "selection": "fixed_last120",
        "source": meta["source_identity"],
        "models": {j["name"]: check_job(meta, out, j) for j in jobs()},
        "readout_fits": 0,
        "reference_role": "concurrent_scratch_bar_no_historical_weights",
    }
    immutable_json(value, out / "model_selection_lock.json")
    return value


def verify_lock(meta, out, name, split):
    if name not in {j["name"] for j in jobs()} or split not in evaluation.COHORTS:
        raise ValueError("Unknown research model/cohort")
    if meta != read_json(out / "manifest.json") or meta["implementation"] != implementation():
        raise ValueError("Bound fresh source code/manifest changed")
    lock = read_json(out / "model_selection_lock.json")
    if (
        lock.get("schema") != SCHEMA
        or lock.get("manifest") != sha256(out / "manifest.json")
        or set(lock.get("models", {})) != {j["name"] for j in jobs()}
        or any(v["epoch"] != 120 or v["updates"] != 4560 for v in lock["models"].values())
    ):
        raise ValueError("All10 fixed-last120 workers must be locked")
    verify_files(out, {v["file"]: v["sha256"] for v in lock["models"].values()})
    dataset = Path(meta["source"])
    for suffix in (".npz", "_rows.json"):
        relative = split + suffix
        if sha256(dataset / relative) != meta["source_identity"]["dataset_files"][relative]:
            raise ValueError("Research raw/cache identity changed")
    return {
        "model_lock": sha256(out / "model_selection_lock.json"),
        "source": meta["source_identity"],
        "raw_receipt": meta["source_identity"]["dataset_files"][split + ".npz"],
    }


def decisions(banks, rows):
    # Reuse the exact paired tests; explicitly replace the former anchor role.
    result = evaluation.decisions(banks, rows)
    groups = result["comparison_groups"]
    anchor = groups.pop("historical_bar_anchor")
    anchor["role"] = "concurrent_scratch_bar_quality_control_not_equal_parameter_factorial_contrast"
    groups["concurrent_bar_anchor"] = anchor
    result["concurrent_anchor_passed"] = result.pop("historical_anchor_passed")
    result.update(
        schema=SCHEMA,
        scope=protocol()["scope"],
        historical_replay=False,
        reference_role=protocol()["bar_role"],
        data_revision=protocol()["data_revision_reason"],
    )
    return result


def remaining_epochs(meta, out):
    result = {}
    for job in jobs():
        _, _, root, actual = worker_location(meta, out, job)
        receipt = root / actual["name"] / "checkpoint.json"
        epoch = read_json(receipt)["epoch"] if receipt.exists() else 0
        if type(epoch) is not int or not 0 <= epoch <= 120:
            raise ValueError("Invalid committed epoch")
        result[job["name"]] = 120 - epoch
    return result


def disk_required():
    # Original20-bank gross + two additional Adam checkpoints + portable backup
    # and its atomically replaced prior version. Raw/cache package exists outside.
    original = factorial.disk_requirement(30274183 * 4, 1423)["required_bytes"]
    return original + 2 * 30274183 * 12 + 24 * 1024**3


def preflight(meta, out):
    x, close, _ = data.cache(Path(meta["source"]), "train")
    timing = {}
    for arm in ("reference_bar", *model.ARMS):
        is_bar = arm == "reference_bar"
        component = bar if is_bar else factorial
        actual = "bar" if is_bar else arm
        encoder, query = component.model.make_models(actual, 42, "cuda", meta["encoder"])
        encoder.checkpointing = True
        torch.cuda.reset_peak_memory_stats()
        xx = torch.as_tensor(x[:8], device="cuda")
        cc = torch.as_tensor(close[:8], device="cuda")
        ps = torch.as_tensor(training.schedule(8, 42, 1)[1], device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = component.training.training_terms(
            encoder, query, xx, cc, ps, meta["statistics"], actual
        )["objective"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*encoder.parameters(), *query.parameters()], 1.0, error_if_nonfinite=True
        )
        # Reserve two FP32 Adam moments and a conservative step temporary without
        # performing an optimizer update or retaining a warmed-up model.
        moment_reserve = torch.empty(
            sum(p.numel() for p in (*encoder.parameters(), *query.parameters())) * 3,
            device="cuda",
            dtype=torch.float32,
        )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        # Actual128-position extraction microgeometry:8 parents x5 windows.
        encoder.eval()
        query.eval()
        with torch.no_grad():
            for begin in range(1, 129, 5):
                positions = torch.arange(begin, min(begin + 5, 129), device="cuda").expand(8, -1)
                component.model.predict(encoder, query, xx, positions)
        torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - started - train_seconds
        timing[arm] = {
            "train_micro_seconds": train_seconds,
            "eval128_eight_parents_seconds": evaluation_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_updates": 0,
        }
        if not is_bar:
            timing[arm]["gradient_diagnostic"] = training.gradient_diagnostic(
                encoder, query, x[:8], close[:8], meta["statistics"], 42, "cuda"
            )["passed"]
            if not timing[arm]["gradient_diagnostic"]:
                raise ValueError("Preflight gradient diagnostic failed")
        timing[arm]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        progress(f"preflight {arm}: {timing[arm]}")
        del encoder, query, loss, xx, cc, ps, moment_reserve
        gc.collect()
        torch.cuda.empty_cache()
    remaining = remaining_epochs(meta, out)
    estimate = sum(
        remaining[j["name"]] * timing[j["arm"]]["train_micro_seconds"] * np.ceil(4789 / 8)
        for j in jobs()
    )
    estimate += sum(
        timing[j["arm"]]["eval128_eight_parents_seconds"] * np.ceil(1423 / 8) for j in jobs()
    )
    estimate *= 1.4  # Includes validation/Adam/save overhead; not a completion promise.
    estimate += 3600
    saved = {
        "schema": SCHEMA,
        "timing": timing,
        "remaining_epoch_estimate_seconds": float(estimate),
        "cap_seconds": CAP_SECONDS,
        "gpu": torch.cuda.get_device_name(),
        "free_disk": shutil.disk_usage(out).free,
        "initial_disk_requirement": disk_required(),
        "scope": "zero optimizer updates; disposable real CUDA networks; stop on resource failure",
    }
    budget = read_json(out / "budget.json")
    saved["passed"] = bool(estimate <= CAP_SECONDS - budget["elapsed_seconds"])
    atomic_json(saved, out / "preflight.json")
    if not saved["passed"]:
        raise ValueError(
            "Measured remaining workload exceeds36h cap; evidence exported, no training started"
        )


def evaluate(meta, out):
    lock_models(meta, out)
    banks, rows_bank = {}, {}
    for split in evaluation.COHORTS:
        x, close, rows = data.cache(Path(meta["source"]), split)
        banks[split], rows_bank[split] = {}, rows
        for job in jobs():
            verify_lock(meta, out, job["name"], split)
            module, mm, root, actual = worker_location(meta, out, job)
            done = module.check_job(mm, root, actual)
            encoder, query = module.load_pair(mm, actual, "cuda")
            ck = torch.load(
                root / actual["name"] / "last.pt", map_location="cpu", weights_only=True
            )
            module.validate_saved_state(ck, encoder, query, 120, 4560)
            if ck["binding"] != done["binding"] or ck["signature"] != done["signature"]:
                raise ValueError("Locked checkpoint payload differs")
            encoder.load_state_dict(ck["encoder"], strict=True)
            query.load_state_dict(ck["query"], strict=True)
            if training.parameter_signature(encoder, query) != done["signature"]:
                raise ValueError("Restored model signature differs")
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
    metrics = decisions(banks, rows_bank)
    atomic_json(metrics, out / "input_factorial_fresh_metrics.json")
    names = ["manifest.json", "model_selection_lock.json", "input_factorial_fresh_metrics.json"]
    names += [
        f"evaluation/{j['name']}/{s}/complete.json" for j in jobs() for s in evaluation.COHORTS
    ]
    atomic_json(
        {
            "schema": SCHEMA,
            "status": "complete_requires_review",
            "promoted": False,
            "limited_hypothesis_supported": metrics["limited_hypothesis_supported"],
            "files": {n: sha256(out / n) for n in names},
        },
        out / "completion.json",
    )


def check_completion(meta, out):
    """No model execution: verify every committed research shard, not only scores."""
    done = read_json(out / "completion.json")
    names = {"manifest.json", "model_selection_lock.json", "input_factorial_fresh_metrics.json"}
    names.update(
        f"evaluation/{j['name']}/{s}/complete.json" for j in jobs() for s in evaluation.COHORTS
    )
    if (
        done.get("schema") != SCHEMA
        or done.get("status") != "complete_requires_review"
        or set(done.get("files", {})) != names
    ):
        raise ValueError("Incomplete fresh completion inventory")
    verify_files(out, done["files"])
    lock = lock_models(meta, out)
    for split in evaluation.COHORTS:
        x, close, rows = data.cache(Path(meta["source"]), split)
        for job in jobs():
            receipt = out / f"evaluation/{job['name']}/{split}/complete.json"
            binding = read_json(receipt)["binding"]
            expected = {
                "name": job["name"],
                "split": split,
                "parents": len(rows),
                "model_signature": lock["models"][job["name"]]["signature"],
                "lock": verify_lock(meta, out, job["name"], split),
                "rows_digest": evaluation._digest(rows),
                "x_sha256": training.array_hash(x),
                "raw_sha256": training.array_hash(close),
                "statistics_digest": evaluation._digest(meta["statistics"]),
            }
            if any(binding.get(k) != v for k, v in expected.items()):
                raise ValueError("Completed bank source binding differs")
            evaluation._load_complete(out, receipt, binding)
    return done


def worker(out, name):
    meta = read_json(out / "manifest.json")
    if meta != make_manifest(Path(meta["bundle"]), Path(meta["data_root"])):
        raise ValueError("Fresh protocol/code/data changed")
    if not torch.cuda.is_available():
        raise ValueError("Real model execution requires AutoDL CUDA")
    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError(
            "This release requires the predeclared AutoDL PyTorch2.8.0 CUDA environment"
        )
    active = read_json(out / "budget.json")["active"]
    if not isinstance(active, dict) or active.get("stage") != name:
        raise ValueError("Worker requires an active supervisor budget reservation")
    seconds = active["deadline_unix"] - time.time()
    if not 0 < seconds <= CAP_SECONDS:
        raise ValueError("Worker budget exhausted or invalid")

    def expired(signum, frame):
        raise TimeoutError("Worker cumulative36h deadline expired")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    with runtime.replay_runtime("context"):
        if name == "initialize":
            prepare_initializations(meta, out, "cuda")
        elif name == "preflight":
            preflight(meta, out)
        elif name == "evaluate":
            evaluate(meta, out)
        else:
            job = next((j for j in jobs() if j["name"] == name), None)
            if job is None:
                raise ValueError("Undeclared worker")
            module, mm, root, actual = worker_location(meta, out, job)
            x, close, _ = data.cache(Path(meta["source"]), "train")
            val, val_close, _ = data.cache(Path(meta["source"]), "val")
            module.train_job(mm, root, actual, x, close, val, val_close, "cuda")


def supervise(bundle, target, out):
    started = time.monotonic()
    bundle, target, out = map(raw_snapshot.no_links, (bundle, target, out))
    out = evaluation._output_directory(out)
    if out.is_relative_to(bundle) or bundle.is_relative_to(out):
        raise ValueError("Snapshot and output must be separate")
    if out == target or out.is_relative_to(target) or target.is_relative_to(out):
        raise ValueError("Data and output must be separate")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = make_manifest(bundle, target)
        immutable_json(meta, out / "manifest.json")
        budget_path = out / "budget.json"
        budget = (
            read_json(budget_path)
            if budget_path.exists()
            else {"schema": SCHEMA, "elapsed_seconds": 0.0, "active": None, "stages": []}
        )
        if budget["active"] is not None:
            raise ValueError(
                "Previous supervisor stopped without settling its budget; preserve/export evidence"
            )
        accounted = time.monotonic()
        budget["elapsed_seconds"] += accounted - started
        atomic_json(budget, budget_path)
        if (out / "completion.json").exists():
            check_completion(meta, out)
            return
        if not (out / "preflight.json").exists() and shutil.disk_usage(out).free < disk_required():
            raise ValueError(
                f"Need {disk_required() / 1024**3:.1f}GiB free including report/full backup; no old data removed"
            )
        stages = ["initialize", "preflight", *[j["name"] for j in jobs()], "evaluate"]
        for stage in stages:
            if stage == "preflight" and (out / "preflight.json").exists():
                if read_json(out / "preflight.json").get("passed") is not True:
                    raise ValueError(
                        "Previous preflight failed; export its evidence, do not bypass it"
                    )
                continue
            if stage not in ("initialize", "preflight", "evaluate"):
                j = next(j for j in jobs() if j["name"] == stage)
                _, _, root, actual = worker_location(meta, out, j)
                if (root / actual["name"] / "completion.json").exists():
                    check_job(meta, out, j)
                    continue
            now = time.monotonic()
            budget["elapsed_seconds"] += now - accounted
            accounted = now
            remaining = CAP_SECONDS - budget["elapsed_seconds"]
            if remaining <= 0:
                raise ValueError("Cumulative36h budget exhausted")
            budget["active"] = {"stage": stage, "deadline_unix": time.time() + remaining}
            atomic_json(budget, budget_path)
            atomic_json(
                {"schema": SCHEMA, "status": "running", "stage": stage}, out / "status.json"
            )
            progress(
                f"fresh stage={stage}, cumulative elapsed={budget['elapsed_seconds'] / 3600:.2f}h; no old checkpoints required"
            )
            began = time.monotonic()
            proc = subprocess.Popen(
                [sys.executable, "-m", MODULE, "--out", str(out), "--worker", stage],
                start_new_session=True,
            )
            previous_handlers = {}

            def stop(signum, frame):
                raise KeyboardInterrupt(f"signal {signum}")

            try:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    previous_handlers[sig] = signal.signal(sig, stop)
                code = proc.wait(timeout=remaining)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
                finished = time.monotonic()
                elapsed = finished - began
                budget["elapsed_seconds"] += finished - accounted
                accounted = finished
                budget["stages"].append(
                    {"stage": stage, "seconds": elapsed, "exit_code": proc.returncode}
                )
                budget["active"] = None
                atomic_json(budget, budget_path)
            if code:
                raise ValueError(
                    f"Stage {stage} exited {code}; complete report and checkpoint backup will be exported"
                )
        atomic_json(
            {
                "schema": SCHEMA,
                "status": "complete_requires_review",
                "scientific_acceptance": "pending",
            },
            out / "status.json",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=DATA_BUNDLE)
    p.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker")
    a = p.parse_args()
    a.out = evaluation._output_directory(a.out)
    try:
        if a.worker:
            worker(raw_snapshot.no_links(a.out), a.worker)
        else:
            supervise(a.bundle, a.data_root, a.out)
    except BaseException as error:
        if a.worker and a.out.is_dir():
            atomic_json(
                {"worker": a.worker, "error": str(error), "traceback": traceback.format_exc()},
                a.out / f"worker_failure_{time.time_ns()}.json",
            )
        if not a.worker and a.out.is_dir():
            failure = {
                "schema": SCHEMA,
                "status": "stopped"
                if isinstance(error, (KeyboardInterrupt, subprocess.TimeoutExpired))
                else "failed",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "scientific_acceptance": "not_established",
            }
            # A second launcher must never overwrite the live owner's status.
            with (a.out / "run.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    atomic_json(failure, a.out / "status.json")
        raise


if __name__ == "__main__":
    main()
