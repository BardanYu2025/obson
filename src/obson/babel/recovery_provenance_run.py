"""V14: original scaler refits plus two-step real CUDA process-resume audit.

Calls the bound V06 train_job unchanged. Harness overrides ONLY its epoch cap
(two), supplied training subset(first128), and diagnostic RNG/exit hooks. No
candidate promotion, research scoring, new heads, or continuation of source runs.
"""

import argparse
import fcntl
import gc
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import recovery_context_run as production
from . import recovery_provenance as core
from . import research_state as research
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-recovery-provenance-v1"
CAP_SECONDS = 7200
ARCH_INDEX = "c24db61e390cdff81ea8522c81c3cda59d3983cb39f7d2b9e3be7ad82d8629b0"
LOCAL_SHA = "2fe0bccddda3d8980401d2c6d2d6f4f2565f6dc20ff6cdfc4d047c47aa6dbc3b"
r0, lr = production.r0, production.lr


def protocol():
    return {
        "task_ids": ["V14"],
        "requirements": ["N01", "N03", "N04", "N07", "N08", "N09"],
        "purpose": "Close missing scaler provenance and committed-checkpoint recovery evidence; no model improvement claim",
        "state_refit": "Original control600 s42/s43 selected control best, train4789 endpoint states; target0 support checked; original extraction batch64 and fastpath profile. FP64 population moments, std floor1e-6. Compare to bound fit and bundle, then exact inherited float32 Macro query buffers",
        "local_refit": "Original Architecture train4789 y/mask; Prefix32/64/96/128 past16 before queried bar; close reanchored to predecessor; original float32 physical target/difference rounding, masked FP64 moments, std floor1e-5, delta RMS floor.01",
        "reused": "Coverage raw input/target refit with identical statistics and Architecture index SHA; not rerun",
        "resume": "Original Macro97/94 768/4/8; production recovery_context_run.train_job B_full_common; source/eval/optimizer/commit/load checks unchanged; fresh optimizer for audit clones",
        "changes_to_harness": "EPOCHS=2, fixed first128 bound full255 train rows; batch128 micro8; RNG sentinel before each epoch and SystemExit73 after epoch1 commit only in interrupted child; each resume is a fresh process",
        "loss": production.protocol()["loss"],
        "rates": production.protocol()["rates"],
        "optimizer": production.protocol()["optimizer"],
        "budget": {
            "seeds": [42, 43],
            "paths_per_seed": 2,
            "child_processes_total": 6,
            "updates_per_path": 2,
            "total_updates": 8,
            "window_exposures": 1024,
            "state_exposures": 5120,
            "seconds_max": CAP_SECONDS,
            "parallel_gpu_workers": 1,
        },
        "validation": "Unchanged original978 validation for source replay and production journal only; no held-out/test/research reads and no scientific selection. Compare committed checkpoint+best tensors+optimizer+RNG+history(except seconds), both epoch RNG probes and subsequent updates exactly",
        "numeric": {
            "state_atol": r0.ATOL,
            "state_rtol": r0.RTOL,
            "local_atol": 1e-10,
            "local_rtol": 1e-10,
            "resume": "bit exact",
        },
        "backend": "Require reviewed R1 GPU/Torch; FP32 no AMP/TF32; restore historical MHA profile per stage; resume additionally deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "acceptance": "All specified comparisons pass for both seeds; evidence then reviewed, never automatic V14 closure or model promotion",
        "stop": "Missing source/hash, shape/support, numerical/refit/restore mismatch, nonfinite or timeout stops and exports. One attempt only; no automatic rerun or threshold relaxation",
        "scope_limit": "Recovery qualifies this production path on this device at committed epoch boundary; not arbitrary mid-step crash or all historical workers. Refit covers these three scaler families only",
    }


def locate(node, schema):
    if isinstance(node, dict):
        if node.get("schema") == schema:
            return node
        for value in node.values():
            found = locate(value, schema)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = locate(value, schema)
            if found is not None:
                return found
    return None


def bound(path, digest, receipt):
    path = Path(path).resolve()
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"Missing or changed required V14 source: {path}; expected={digest}")
    receipt[str(path)] = digest
    return path


def scalar_checks(actual, expected, atol, rtol):
    if actual.keys() != expected.keys():
        return {"keys": {"passed": False}}
    return {
        k: (
            {"passed": actual[k] == v}
            if isinstance(v, str)
            else r0.compare(actual[k], v, atol, rtol)
        )
        for k, v in expected.items()
    }


def preflight(source, bundle_source, out):
    meta, r1, context, rm = production.verify_source(source, out)
    if torch.__version__ != rm["torch"] or torch.cuda.get_device_name() != rm["gpu"]:
        raise ValueError("Use reviewed R1 CUDA GPU/PyTorch environment for exact recovery audit")
    receipt = {}
    # Current Context manifest carries the original delivery identity directly.
    identity = meta["identity"]
    for name in (
        "manifest.json",
        "bundle/index.json",
        "bundle/control_s42_best.pt",
        "bundle/control_s43_best.pt",
    ):
        bound(bundle_source / name, identity["files"][name], receipt)
    if read_json(bundle_source / "manifest.json") != identity["manifest"]:
        raise ValueError("Original control600 delivery differs")
    delivery = identity["manifest"]
    train_root = Path(delivery["identity"]["training_source"])
    bound(train_root / "fit.json", delivery["identity"]["training_files"]["fit.json"], receipt)
    alignment = locate(meta, "babel-bar-alignment-v1")
    if alignment is None:
        raise ValueError("Missing original alignment lineage")
    original = Path(alignment["original_source"]) / "cache"
    bound(original / "index.json", ARCH_INDEX, receipt)
    idx = read_json(original / "index.json")
    for name in ("train_x.npy", "train_y.npy", "train_mask.npy", "statistics.json"):
        bound(original / name, idx["files"][name], receipt)
    stats = read_json(original / "statistics.json")
    if sha256(original / "statistics.json") != production.STATISTICS_SHA256:
        raise ValueError("Reviewed Coverage statistics differ")
    inherited, local = production.old.parent.stats(meta)
    if stats != inherited:
        raise ValueError("Macro inherited input/target statistics differ")
    # Need only the retained canonical local file; do not require deleted probe caches.
    alignment_parent = locate(meta, "babel-causal-path768-v1")
    local_path = Path(alignment_parent["source"]) / "cache/local_scales.json"
    bound(local_path, LOCAL_SHA, receipt)
    if read_json(local_path) != local:
        raise ValueError("Macro inherited local scales differ")
    data = {
        k: np.load(original / f"train_{k}.npy", mmap_mode="r", allow_pickle=False)
        for k in ("x", "y", "mask")
    }
    if data["x"].shape != (4789, 128, 28):
        raise ValueError("Original fixed train shape differs")
    atomic_json(receipt, out / "scaler_source_files.json")
    atomic_json(
        {
            "input_target_refit": "reused Coverage report, not recomputed",
            "coverage_archive_sha256": "c6a1ce420ba6463794d807eec6819e3bc69db6f1a6328558d52932eaa69bd5f5",
            "architecture_index_sha256": ARCH_INDEX,
            "statistics_sha256": production.STATISTICS_SHA256,
        },
        out / "reused_evidence.json",
    )
    return meta, r1, context, data, stats, local, train_root, receipt


@torch.inference_mode()
def scaler_audit(meta, bundle_source, data, stats, local, train_root, out):
    fitted = core.local_moments(data, stats)
    r0.save_checks(
        out,
        "local_refit",
        scalar_checks(fitted, local, 1e-10, 1e-10),
        recomputed=fitted,
        original=local,
        rows=4789,
        optimizer_updates=0,
    )
    fit = read_json(train_root / "fit.json")
    tm = meta["identity"]["manifest"]["identity"]["training_manifest"]
    batch = tm["evaluation_batch"]
    if batch != 64:
        raise ValueError("Original state extraction batch changed")
    for seed in (42, 43):
        progress(f"V14 s{seed}: refitting ORIGINAL control600 train states; zero updates")
        checkpoint = torch.load(
            bundle_source / f"bundle/control_s{seed}_best.pt", map_location="cpu", weights_only=True
        )
        head = fit["heads"][f"control_s{seed}_best"]["targets"][0]
        if (
            head != checkpoint["utility"]["heads"][0]
            or head["training_support"] != 4789
            or head["target"] != research.up.NAMES[0]
        ):
            raise ValueError("Original target0 scaler/support differs")
        # Utility target0 is a historical descriptor; independently check full support.
        _, mask = research.up.targets(research.up.restore_raw(data["x"], stats))
        if not mask[:, 0].all():
            raise ValueError("State scaler population differs")
        model = research.restore_model(checkpoint).to("cuda")
        states = []
        with r0.replay_runtime("macro_shared"):
            for start in range(0, len(data["x"]), batch):
                b = torch.tensor(np.asarray(data["x"][start : start + batch]), device="cuda")
                states.append(model.core.encoder(b)[:, -1].cpu().numpy())
        states = np.concatenate(states)
        if states.shape != (4789, 768):
            raise ValueError("Original state dimension changed")
        np.savez_compressed(out / f"state_train_s{seed}.npz", states=states)
        refit = core.state_moments(states)
        r0.save_checks(
            out,
            f"state_refit_s{seed}",
            scalar_checks(refit, head["statistics"], r0.ATOL, r0.RTOL),
            recomputed=refit,
            original=head["statistics"],
            rows=4789,
            optimizer_updates=0,
        )
        del model, checkpoint, states
        gc.collect()
        torch.cuda.empty_cache()
        encoder, query = production.construct(meta, seed, "cuda")
        checks = {
            key: {
                "passed": torch.equal(
                    getattr(query, f"state_{key}").cpu(),
                    torch.tensor(head["statistics"][key], dtype=torch.float32),
                )
            }
            for key in ("mean", "scale")
        }
        r0.save_checks(out, f"inherited_query_s{seed}", checks, optimizer_updates=0)
        del encoder, query
        gc.collect()
        torch.cuda.empty_cache()


def rng_probe():
    return {
        "python": random.random(),
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4).tolist(),
        "torch_cuda": torch.rand(4, device="cuda").cpu().tolist(),
    }


def install_hooks(folder, mode):
    """Test instrumentation around, never replacing, production updates/restore."""
    train_epoch, pending_epoch = production.train_epoch, lr.pending_epoch

    def observed_epoch(*args, **kwargs):
        epoch = args[6]
        atomic_json(rng_probe(), folder / f"rng_before_epoch{epoch}.json")
        return train_epoch(*args, **kwargs)

    def committed(where, epoch):
        result = pending_epoch(where, epoch)
        if mode == "interrupt" and epoch == 1:
            ck = torch.load(where / "last.pt", map_location="cpu", weights_only=True)
            if (
                ck["epoch"] != 1
                or len(ck["history"]) != 1
                or (where / "pending_epoch.json").exists()
                or read_json(where / "checkpoint.json")["sha256"] != sha256(where / "last.pt")
                or read_json(where / "history.json") != ck["history"]
            ):
                raise ValueError("Interruption is not at a committed checkpoint")
            atomic_json(
                {
                    "epoch": 1,
                    "exit_code": 73,
                    "pid": os.getpid(),
                    "checkpoint_sha256": sha256(where / "last.pt"),
                    "updates": 1,
                },
                where / "intentional_exit.json",
            )
            raise SystemExit(73)
        return result

    production.train_epoch, lr.pending_epoch = observed_epoch, committed
    return train_epoch, pending_epoch


def worker(source, out, seed, mode):
    request = read_json(out.parent / "request.json")
    if request["source"] != str(source) or request["implementation"] != implementation():
        raise ValueError("Worker request/code changed")
    meta, r1, context, rm = production.verify_source(source, out)
    if torch.__version__ != rm["torch"] or torch.cuda.get_device_name() != rm["gpu"]:
        raise ValueError("Worker CUDA environment differs")
    production.EPOCHS = 2
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train, rows = production.old.data.cache(context, "train")
    val, _ = production.old.data.cache(context, "val")
    if len(train) != 4789 or len(val) != 978 or meta["batch"] != 128 or meta["micro"] != 8:
        raise ValueError("Production data/batch changed")
    atomic_json(rows[:128], out / "audit_train_rows.json")
    job = {"arm": "B_full_common", "seed": seed, "name": f"resume_s{seed}"}
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    if mode == "resume":
        proof = read_json(folder / "intentional_exit.json")
        if proof["pid"] == os.getpid() or proof["checkpoint_sha256"] != sha256(folder / "last.pt"):
            raise ValueError("Not a fresh-process restart of committed checkpoint")
        atomic_json(
            {
                "pid": os.getpid(),
                "previous_pid": proof["pid"],
                "checkpoint_sha256": proof["checkpoint_sha256"],
            },
            folder / "restart.json",
        )
    elif (folder / "last.pt").exists():
        raise ValueError("Refuse unbudgeted repeated initial worker")
    old_hooks = install_hooks(folder, mode)
    try:
        with r0.replay_runtime("context"):
            atomic_json(
                {
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "runtime": r0.runtime_state(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                },
                folder / f"runtime_{mode}.json",
            )
            done = production.train_job(meta, context, r1, out, job, train[:128], val)
        if done["optimizer_updates"] != 2:
            raise ValueError("Unexpected audit updates")
    finally:
        production.train_epoch, lr.pending_epoch = old_hooks


def compare_paths(out, seed):
    left = out / f"continuous_s{seed}" / f"resume_s{seed}"
    right = out / f"restarted_s{seed}" / f"resume_s{seed}"
    checks = {}
    for name in ("last.pt", "best.pt"):
        a, b = (torch.load(p / name, map_location="cpu", weights_only=True) for p in (left, right))
        if name == "last.pt":
            if any(
                v["epoch"] != 2 or sum(h["train"]["steps"] for h in v["history"]) != 2
                for v in (a, b)
            ):
                raise ValueError("Audit checkpoint has wrong update budget")
            a, b = core.checkpoint_semantics(a), core.checkpoint_semantics(b)
        mismatch = core.compare_tree(a, b)
        checks[name] = {"passed": not mismatch, "mismatches": mismatch}
        del a, b
    for epoch in (1, 2):
        name = f"rng_before_epoch{epoch}.json"
        mismatch = core.compare_tree(read_json(left / name), read_json(right / name))
        checks[name] = {"passed": not mismatch, "mismatches": mismatch}
    checks["train_rows"] = {
        "passed": read_json(left.parent / "audit_train_rows.json")
        == read_json(right.parent / "audit_train_rows.json")
    }
    r0.save_checks(
        out,
        f"resume_comparison_s{seed}",
        checks,
        actual_updates=4,
        ignored=[
            "training elapsed seconds",
            "best.pt serialization hash (tensors compared separately)",
        ],
        scope="Two real production updates on first128 train rows, committed boundary, fresh process",
    )
    return checks


def implementation():
    return {
        "production": production.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__) for m in (core, sys.modules[__name__])
        },
        "script": sha256(lr.REPO / "scripts/babel_recovery_provenance_autodl.sh"),
    }


def run(source, bundle_source, out):
    out = production.interface.separate(out, source, bundle_source)
    if out.exists():
        raise ValueError("V14 audit is one bounded attempt; existing output is export-only")
    out.mkdir(parents=True)
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"task_ids": ["V14"], "status": "running", "promoted": False}
        atomic_json(status, out / "status.json")
        start = time.monotonic()
        try:
            if not torch.cuda.is_available():
                raise ValueError("V14 real-weight execution requires AutoDL CUDA")
            request = {
                "schema": SCHEMA,
                "source": str(source),
                "bundle_source": str(bundle_source),
                "implementation": implementation(),
                "protocol": protocol(),
            }
            atomic_json(request, out / "request.json")
            atomic_json(protocol(), out / "protocol.json")
            atomic_json(
                {
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "scaler_profile": r0.RUNTIME_PROFILES["macro_shared"],
                    "resume_profile": r0.RUNTIME_PROFILES["context"],
                    "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                },
                out / "runtime.json",
            )
            progress("V14: checking original scaler sources and qualified Macro lineage")
            meta, r1, context, data, stats, local, train_root, receipt = preflight(
                source, bundle_source, out
            )
            largest = max((r1 / f"s{s}/last.pt").stat().st_size for s in (42, 43))
            required = 7 * largest + 512 * 2**20
            atomic_json(
                {"required_free_bytes": required, "free_bytes": shutil.disk_usage(out).free},
                out / "disk_preflight.json",
            )
            if shutil.disk_usage(out).free < required:
                raise ValueError(f"Need {required / 2**30:.2f} GiB free; no source deleted")
            scaler_audit(meta, bundle_source, data, stats, local, train_root, out)
            del data
            gc.collect()
            torch.cuda.empty_cache()
            for seed in (42, 43):
                for name in ("continuous", "restarted"):
                    folder = out / f"{name}_s{seed}"
                    folder.mkdir()
                    # Same binding bytes in both paths, not path-dependent metadata.
                    atomic_json(request, folder / "manifest.json")
                for name, mode, expected in (
                    ("continuous", "continuous", 0),
                    ("restarted", "interrupt", 73),
                    ("restarted", "resume", 0),
                ):
                    folder = out / f"{name}_s{seed}"
                    progress(f"V14 s{seed}: {mode}; two-step production checkpoint audit")
                    remaining = CAP_SECONDS - (time.monotonic() - start)
                    if remaining <= 0:
                        raise TimeoutError("V14 time budget exhausted")
                    command = [
                        sys.executable,
                        "-m",
                        "obson.babel.recovery_provenance_run",
                        "--worker",
                        mode,
                        "--source",
                        str(source),
                        "--out",
                        str(folder),
                        "--seed",
                        str(seed),
                    ]
                    code = subprocess.run(command, timeout=remaining).returncode
                    atomic_json({"code": code, "expected": expected}, folder / f"{mode}_exit.json")
                    if code != expected:
                        raise ValueError(f"{mode} child exited {code}, expected {expected}")
                compare_paths(out, seed)
            for path, digest in receipt.items():
                bound(path, digest, {})
            production.verify_source(source, out)
            if implementation() != request["implementation"]:
                raise ValueError("Implementation changed during audit")
            status.update(
                status="v14_checks_complete_requires_review",
                actual_optimizer_updates=8,
                scaler_optimizer_updates=0,
                source_unchanged=True,
                seconds=time.monotonic() - start,
            )
            files = {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file() and p.suffix in (".json", ".npz", ".pt") and p != out / "status.json"
            }
            atomic_json({**status, "files": files}, out / "completion.json")
            atomic_json(status, out / "status.json")
        except BaseException as error:
            status.update(status="failed", error=str(error), traceback=traceback.format_exc())
            atomic_json(status, out / "status.json")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--bundle-source", type=Path, default=Path("checkpoints/babel_control600_v2")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", choices=("continuous", "interrupt", "resume"))
    parser.add_argument("--seed", type=int, choices=(42, 43))
    args = parser.parse_args()
    if args.worker:
        if args.seed is None:
            parser.error("--worker requires --seed")
        worker(args.source.resolve(), args.out.resolve(), args.seed, args.worker)
    else:
        run(args.source.resolve(), args.bundle_source.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
