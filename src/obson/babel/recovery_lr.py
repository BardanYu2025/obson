"""V03/R1: two fixed five-epoch low-LR controls using the immutable Context step."""

import argparse
import fcntl
import gc
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import recovery_qualification as qualification
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

old = qualification.previous
r0 = qualification.r0
SCHEMA = "babel-recovery-r1-v1"
QUAL_COMPLETION = "2e0fe27289d2000db90c0ac4e80d67cc175545f07370c85975373a859cc47c86"
EPOCHS = 5
SESSION_CAP = 3600  # Explicit R1 share of the existing 10h recovery budget.
RATES = {
    42: (3.066380815642346e-6, 3.0663808156423456e-5),
    43: (3.2648704606900738e-6, 3.2648704606900735e-5),
}
FIELDS = {
    f"{v}/{c}/{m}"
    for v in ("full", "native")
    for c in ("endpoint", "interior")
    for m in old.core.METRICS
}
REPO = Path(__file__).resolve().parents[3]


def implementation():
    return {
        "modules": old.code_identity()
        | {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (r0, qualification, qualification.objectives)
        }
        | {Path(__file__).name: sha256(__file__)},
        "script": sha256(REPO / "scripts/babel_recovery_r1_autodl.sh"),
        "protocol": sha256(REPO / "docs/BABEL_RECOVERY_R1.md"),
    }


def verify_qualification(source, qual, out):
    if sha256(qual / "completion.json") != QUAL_COMPLETION:
        raise ValueError("Expected reviewed V02/V14 qualification, not another/latest run")
    done = read_json(qual / "completion.json")
    if not (
        done["r1_preflight_passed"] and done["source_unchanged"] and done["optimizer_updates"] == 0
    ):
        raise ValueError("Reviewed common qualification required")
    for name, digest in done["files"].items():
        r0.required_file(qual, name, digest)
    qm = read_json(qual / "manifest.json")
    if Path(qm["source"]).resolve() != source:
        raise ValueError("Qualification belongs to another source")
    modules = old.code_identity() | {
        Path(m.__file__).name: sha256(m.__file__)
        for m in (r0, qualification, qualification.objectives)
    }
    if modules != qm["historical_code"] | qm["code_sha256"]:
        raise ValueError("Qualified historical implementation changed")
    meta, _, _, _ = r0.identities(source, out)
    identity = read_json(out / "identity.json")
    if identity != read_json(qual / "identity.json"):
        raise ValueError("Source identity changed since qualification")
    if (meta["batch"], meta["micro"], meta["validation_prefixes"]) != (
        128,
        8,
        list(old.core.PREFIXES),
    ):
        raise ValueError("Original batch/micro/validation contract changed")
    for seed, rates in RATES.items():
        inherited = identity["learning_rate_bridge"][str(seed)]["source_rates"]
        if rates != (inherited["encoder_lr"], inherited["query_lr"]):
            raise ValueError("Predeclared LR differs from reviewed source")
    return meta, qm


def retention(values, initial):
    if set(values) != FIELDS or set(initial) != FIELDS:
        raise ValueError("Exactly24 original validation fields required")
    if not all(np.isfinite(v) and v >= 0 for v in list(values.values()) + list(initial.values())):
        raise ValueError("Nonfinite or negative validation values")
    ratios = {k: float(values[k]) / max(float(initial[k]), 1e-12) for k in sorted(FIELDS)}
    return {
        "passed": all(v <= 1.10 for v in ratios.values()),
        "ratios": ratios,
        "failed_fields": [k for k, v in ratios.items() if v > 1.10],
        "worst_ratio": max(ratios.values()),
    }


def optimizer(encoder, query, rates):
    # Same fresh AdamW defaults, groups and decay as the historical worker.
    return torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": rates[0], "weight_decay": 0.01},
            {"params": query.parameters(), "lr": rates[1], "weight_decay": 1e-4},
        ]
    )


def optimizer_steps(opt, expected):
    steps = [int(v["step"].item()) for v in opt.state.values() if "step" in v]
    if (expected and not steps) or any(s != expected for s in steps):
        raise ValueError("Optimizer state update count differs from committed epochs")
    return {"state_entries": len(steps), "steps": expected}


def model_signature(e, q):
    fn = old.parent.ur.bb.state_signature
    return {"encoder": fn(e), "query": fn(q)}


def check_history(history, references, initial, rates):
    if len(history) > EPOCHS:
        raise ValueError("Five-epoch budget exceeded")
    for epoch, row in enumerate(history, 1):
        if (row["encoder_lr"], row["query_lr"]) != rates:
            raise ValueError("Recorded learning rates differ")
        if (
            not np.isfinite(row["train"]["loss"])
            or not np.isfinite(row["train"]["seconds"])
            or row["train"]["seconds"] < 0
        ):
            raise ValueError("Invalid training accounting")
        if row["epoch"] != epoch:
            raise ValueError("Noncontiguous epoch journal")
        for name in ("order_hash", "prefix_hash", "steps", "examples", "supervised_states"):
            if row["train"][name] != references[epoch - 1]["train"][name]:
                raise ValueError("Sampling/update reference differs: " + name)
        if row["observed_optimizer_steps"] != row["train"]["steps"]:
            raise ValueError("Actual optimizer step count differs")
        if row["retention"] != retention(row["validation"], initial):
            raise ValueError("Retention journal differs")


@torch.no_grad()
def export_validation(meta, e, q, x, device, path):
    """Export per-window/per-prefix errors, independently aggregate original24 means."""
    arrays = {}
    stats = old.parent.stats(meta)[0]
    for start in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[start : start + meta["micro"]], device=device)
        ps = torch.tensor(old.core.PREFIXES, device=device).expand(len(b), -1)
        for view, native in (("full", False), ("native", True)):
            _, pred = old.core.predict(e, q, b, ps, native)
            y, mask = old.core.original.targets(b, ps, stats, native)
            for k, value in old.core.original.band_rows(pred, y, mask, stats).items():
                if k in old.core.METRICS:
                    arrays.setdefault(view + "/" + k, []).append(value.cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix(".tmp").replace(path)
    result = {}
    for key, value in arrays.items():
        view, metric = key.split("/")
        for context, sl in (("endpoint", slice(-1, None)), ("interior", slice(None, -1))):
            result[f"{view}/{context}/{metric}"] = float(value[:, sl].mean(1).mean())
    return result


def replay_check(actual, expected, path):
    checks = r0.nested_compare(actual, expected)
    atomic_json({"actual": actual, "expected": expected, "checks": checks}, path)
    r0.require(checks, path.stem)


def pending_epoch(folder, committed):
    p = folder / "pending_epoch.json"
    if p.exists():
        pending = read_json(p)["epoch"]
        if pending > committed:
            raise ValueError(
                "Uncommitted epoch detected; refuse silent replay/extra updates. Export for review."
            )
        if pending != committed:
            raise ValueError("Stale pending epoch journal")
        p.unlink()  # The complete atomic checkpoint already contains this epoch.


def train_seed(meta, source, out, seed, train, val, reference, device="cuda"):
    folder = out / f"s{seed}"
    folder.mkdir(exist_ok=True)
    bind = {"manifest_sha256": sha256(out / "manifest.json"), "seed": seed}
    checkpoint = folder / "last.pt"
    receipt = folder / "checkpoint.json"
    if (folder / "completion.json").exists():
        done = read_json(folder / "completion.json")
        if done["binding"] != bind:
            raise ValueError("Completed worker binding changed")
        for name, digest in done["files"].items():
            r0.required_file(folder, name, digest)
        return done
    e, q = old.construct(meta, seed, device)
    initial_signature = model_signature(e, q)
    original = read_json(source / f"prefix_s{seed}/completion.json")
    if initial_signature != {
        "encoder": original["initial_encoder_hash"],
        "query": original["initial_query_hash"],
    }:
        raise ValueError("Original source initialization differs")
    rates = RATES[seed]
    opt = optimizer(e, q, rates)
    expected_groups = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    if checkpoint.exists():
        # Receipt is redundant; the atomic checkpoint journal wins only at the known pending boundary.
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            ck["binding"] != bind
            or ck["initial_signature"] != initial_signature
            or tuple(ck["rates"]) != rates
        ):
            raise ValueError("Resume checkpoint source/LR binding changed")
        if ck["initial_validation"] != original["initial_validation"]:
            raise ValueError("Resume source validation differs")
        check_history(ck["history"], reference, ck["initial_validation"], rates)
        if ck["epoch"] != len(ck["history"]):
            raise ValueError("Checkpoint epoch/history mismatch")
        if receipt.exists() and read_json(receipt)["sha256"] != sha256(checkpoint):
            p = folder / "pending_epoch.json"
            if not p.exists() or read_json(p)["epoch"] != ck["epoch"]:
                raise ValueError("Checkpoint fingerprint mismatch")
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        if [
            {k: v for k, v in g.items() if k != "params"} for g in opt.param_groups
        ] != expected_groups:
            raise ValueError("Resumed optimizer configuration changed")
        if model_signature(e, q) != ck["current_signature"]:
            raise ValueError("Checkpoint tensor signatures differ")
        optimizer_steps(opt, sum(h["train"]["steps"] for h in ck["history"]))
        pending_epoch(folder, ck["epoch"])
        restore_rng(ck["rng"])
    else:
        pending_epoch(folder, 0)
        initial = old.validation(meta, e, q, val, device)
        replay_check(initial, original["initial_validation"], folder / "initial_replay.json")
        # Use the archived exact initial scalars as the fixed denominator after replay.
        initial = original["initial_validation"]
        exported = export_validation(meta, e, q, val, device, folder / "initial_errors.npz")
        replay_check(exported, initial, folder / "initial_export_replay.json")
        ck = {
            "binding": bind,
            "epoch": 0,
            "history": [],
            "initial_signature": initial_signature,
            "current_signature": initial_signature,
            "initial_validation": initial,
            "rates": list(rates),
            "encoder": old.cpu_state(e),
            "query": old.cpu_state(q),
            "optimizer": opt.state_dict(),
            "rng": rng_state(),
        }
        atomic_save(ck, checkpoint)
    # Restore/export the committed journal before doing more work.
    atomic_json({"epoch": ck["epoch"], "sha256": sha256(checkpoint)}, receipt)
    atomic_json(ck["history"], folder / "history.json")
    for group, rate, decay in zip(opt.param_groups, rates, (0.01, 1e-4), strict=True):
        if (
            group["lr"] != rate
            or group["weight_decay"] != decay
            or group["betas"] != (0.9, 0.999)
            or group["eps"] != 1e-8
        ):
            raise ValueError("Resumed optimizer recipe changed")
    job = {"name": f"r1_s{seed}", "mode": "prefix", "seed": seed}
    while ck["epoch"] < EPOCHS:
        epoch = ck["epoch"] + 1
        atomic_json(
            {"epoch": epoch, "prior_committed_epoch": ck["epoch"]}, folder / "pending_epoch.json"
        )
        observed = [0]

        def stepped(*_, counter=observed):
            counter[0] += 1

        hook = opt.register_step_post_hook(stepped)
        try:
            trained = old.train_epoch(meta, job, e, q, train, opt, epoch, device)
        finally:
            hook.remove()
        validation = old.validation(meta, e, q, val, device)
        row = {
            "epoch": epoch,
            "train": trained,
            "validation": validation,
            "retention": retention(validation, ck["initial_validation"]),
            "observed_optimizer_steps": observed[0],
            "encoder_lr": rates[0],
            "query_lr": rates[1],
        }
        history = ck["history"] + [row]
        check_history(history, reference, ck["initial_validation"], rates)
        for p in list(e.state_dict().values()) + list(q.state_dict().values()):
            if not torch.isfinite(p).all():
                raise ValueError("Nonfinite trained model state")
        optimizer_steps(opt, sum(h["train"]["steps"] for h in history))
        ck.update(
            epoch=epoch,
            history=history,
            encoder=old.cpu_state(e),
            query=old.cpu_state(q),
            optimizer=opt.state_dict(),
            current_signature=model_signature(e, q),
            rng=rng_state(),
        )
        atomic_save(ck, checkpoint)
        atomic_json({"epoch": epoch, "sha256": sha256(checkpoint)}, receipt)
        atomic_json(history, folder / "history.json")
        pending_epoch(folder, epoch)
        progress(
            f"V03 s{seed}: {epoch}/5, actual_steps={observed[0]}, retained={row['retention']['passed']}, worst_ratio={row['retention']['worst_ratio']:.4f}"
        )
    # Final means and rowwise errors come from the actually updated epoch5.
    final = export_validation(meta, e, q, val, device, folder / "epoch5_errors.npz")
    replay_check(final, ck["history"][-1]["validation"], folder / "final_export_replay.json")
    r0.causality(e, q, torch.as_tensor(train[:2], device=device), folder, "epoch5")
    changed = all(ck["current_signature"][k] != initial_signature[k] for k in initial_signature)
    if not changed:
        raise ValueError(
            "Epoch5 must contain actual encoder and query updates, not epoch0 fallback"
        )
    gate = retention(ck["history"][-1]["validation"], ck["initial_validation"])
    comparison = {
        "initial": ck["initial_validation"],
        "low_lr_epoch5": ck["history"][-1]["validation"],
        "old_high_lr_epoch5": reference[-1]["validation"],
        "low_lr_retention": gate,
        "high_lr_retention": retention(reference[-1]["validation"], ck["initial_validation"]),
        "low_to_high_ratios": {
            k: ck["history"][-1]["validation"][k] / max(reference[-1]["validation"][k], 1e-12)
            for k in sorted(FIELDS)
        },
    }
    atomic_json(comparison, folder / "comparison.json")
    done = {
        "binding": bind,
        "epochs": ck["epoch"],
        "optimizer_updates": sum(h["train"]["steps"] for h in ck["history"]),
        "initial_signature": initial_signature,
        "final_signature": ck["current_signature"],
        "actual_updated_epoch5": changed,
        "gate": gate,
        "files": {
            p.name: sha256(p)
            for p in sorted(folder.iterdir())
            if p.suffix in (".pt", ".json", ".npz")
        },
    }
    atomic_json(done, folder / "completion.json")
    del e, q, opt, ck
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def execute(source, qual, out):
    status = {"schema": SCHEMA, "task_ids": ["V03"], "status": "running", "rs_authorized": False}
    atomic_json(status, out / "status.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Real experiment requires AutoDL CUDA; local synthetic tests only")
        progress(
            "V03: verifying reviewed qualification, source weights, caches and original five-epoch reference"
        )
        meta, qm = verify_qualification(source, qual, out)
        if torch.__version__ != qm["torch"] or torch.cuda.get_device_name() != qm["gpu"]:
            raise ValueError(
                "Use the qualified CUDA/PyTorch environment for this single-factor study"
            )
        identity = read_json(out / "identity.json")
        manifest = {
            "schema": SCHEMA,
            "task_ids": ["V03"],
            "source": str(source),
            "qualification": str(qual),
            "qualification_completion_sha256": QUAL_COMPLETION,
            "implementation": implementation(),
            "identity_sha256": sha256(out / "identity.json"),
            "epochs": EPOCHS,
            "rates": {str(k): list(v) for k, v in RATES.items()},
            "batch": 128,
            "micro": 8,
            "seeds": [42, 43],
            "optimizer_updates_max": 380,
            "session_seconds_cap": SESSION_CAP,
            "gpu": qm["gpu"],
            "torch": qm["torch"],
            "runtime": r0.RUNTIME_PROFILES["context"],
        }
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json") != manifest:
                raise ValueError("Run manifest changed; cannot resume")
        else:
            atomic_json(manifest, out / "manifest.json")
            shutil.copyfile(REPO / "docs/BABEL_RECOVERY_R1.md", out / "protocol.md")
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            for n, h in done["files"].items():
                r0.required_file(out, n, h)
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0 if done["r1_passed"] else 3
        if shutil.disk_usage(out).free < 2 * 2**30:
            raise ValueError(
                "Need2GiB free for two optimizer checkpoints and atomic replacement; source stays intact"
            )
        train, _ = old.data.cache(source, "train")
        val, rows = old.data.cache(source, "val")
        if len(train) != 4789 or len(val) != 978:
            raise ValueError("Bound train/val counts differ")
        r0.schedule_audit(source, meta, len(train), out)
        if read_json(out / "reference_schedule.json") != read_json(
            qual / "reference_schedule.json"
        ):
            raise ValueError("Qualified sampling reference changed")
        references = {str(s): read_json(source / f"prefix_s{s}/history.json")[:5] for s in (42, 43)}
        atomic_json(references, out / "high_lr_reference.json")
        atomic_json(rows, out / "validation_rows.json")
        results = {}
        with r0.replay_runtime("context"):
            for seed in (42, 43):
                results[str(seed)] = train_seed(
                    meta, source, out, seed, train, val, references[str(seed)]
                )
        verify_qualification(source, qual, out)
        if (
            read_json(out / "identity.json") != identity
            or implementation() != manifest["implementation"]
        ):
            raise ValueError("Source or implementation changed during R1")
        passed = all(
            d["gate"]["passed"] and d["epochs"] == 5 and d["optimizer_updates"] == 190
            for d in results.values()
        )
        status.update(
            status="r1_complete_requires_review",
            r1_passed=passed,
            optimizer_updates=380,
            seeds={s: {k: v for k, v in d.items() if k != "files"} for s, d in results.items()},
            source_unchanged=True,
        )
        atomic_json(status, out / "status.json")
        # Mutable session accounting/status are deliberately outside immutable evidence closure.
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and p.suffix in (".json", ".npz", ".pt", ".md")
            and p.name not in ("status.json", "budget.json", "request.json")
        }
        atomic_json({**status, "files": files}, out / "completion.json")
        return 0 if passed else 3
    except Exception as error:
        status.update(status="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(status, out / "status.json")
        raise


def paths(source, qual, out):
    source, qual, out = (Path(v).resolve() for v in (source, qual, out))
    if any(out == p or out in p.parents or p in out.parents for p in (source, qual)):
        raise ValueError("Output must be separate from source and qualification")
    return source, qual, out


def supervise(source, qual, out):
    source, qual, out = paths(source, qual, out)
    request = {
        "schema": SCHEMA,
        "source": str(source),
        "qualification": str(qual),
        "output": str(out),
    }
    if (
        out.exists()
        and any(out.iterdir())
        and (not (out / "request.json").exists() or read_json(out / "request.json") != request)
    ):
        raise ValueError("Refuse existing unbound output directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(request, out / "request.json")
        budget = (
            read_json(out / "budget.json")
            if (out / "budget.json").exists()
            else {"used_seconds": 0.0, "attempts": []}
        )
        used = float(budget["used_seconds"])
        if not math.isfinite(used) or used < 0 or used >= SESSION_CAP:
            raise ValueError("R1 cumulative session budget exhausted; no silent extension")
        remaining = SESSION_CAP - used
        # Reserve remaining time before spawning. If supervisor itself dies, no free budget reset.
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget["used_seconds"] = SESSION_CAP
        atomic_json(budget, out / "budget.json")
        started = time.monotonic()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "obson.babel.recovery_lr",
                    "--worker",
                    "--source",
                    str(source),
                    "--qualification",
                    str(qual),
                    "--out",
                    str(out),
                ],
                timeout=remaining,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            state = read_json(out / "status.json") if (out / "status.json").exists() else {}
            state.update(status="timeout", rs_authorized=False, task_ids=["V03"])
            atomic_json(state, out / "status.json")
        elapsed = min(remaining, time.monotonic() - started)
        budget["used_seconds"] = used + elapsed
        attempt.update(elapsed_seconds=elapsed, exit_code=code)
        atomic_json(budget, out / "budget.json")
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    source, qual, out = paths(a.source, a.qualification, a.out)
    if a.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(source, qual, out))
    raise SystemExit(supervise(source, qual, out))


if __name__ == "__main__":
    main()
