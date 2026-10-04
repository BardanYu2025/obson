"""V04/V05 RS-v2: fixed 2x2 loss interventions, bounded matched adaptation."""

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

from . import recovery_interface as interface
from . import recovery_lr as lr
from . import recovery_objectives as objective
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

old, r0, qual = lr.old, lr.r0, lr.qualification
SCHEMA = "babel-recovery-rs-v2"
INTERFACE_COMPLETION = "3e4b4624238d31b5ace4d1a9787d3dbe3aa28342a596a2a62c05d992fb36a6c7"
EPOCHS = 30
SESSION_CAP = 35241
ARMS = tuple(objective.ARMS)


def implementation():
    from . import recovery_rs_evaluate as ev

    return {
        "interface": interface.implementation(),
        "modules": {Path(m.__file__).name: sha256(m.__file__) for m in (sys.modules[__name__], ev)},
        "script": sha256(lr.REPO / "scripts/babel_recovery_rs_autodl.sh"),
        "protocol": sha256(lr.REPO / "docs/BABEL_RECOVERY_RS.md"),
    }


def jobs():
    return [
        {"arm": arm, "seed": seed, "name": f"{arm}_s{seed}"} for arm in ARMS for seed in (42, 43)
    ]


def verify_source(source, out):
    if sha256(source / "completion.json") != INTERFACE_COMPLETION:
        raise ValueError("Expected reviewed V07 completion")
    done = read_json(source / "completion.json")
    for n, h in done["files"].items():
        r0.required_file(source, n, h)
    im = read_json(source / "manifest.json")
    if im["implementation"] != interface.implementation():
        raise ValueError("V07 historical implementation changed")
    r1 = Path(im["source"]).resolve()
    interface.separate(out, source, r1)
    meta, rm, _ = interface.verify_source(r1, out)
    context, qualification = Path(rm["source"]), Path(rm["qualification"])
    q = read_json(qualification / "completion.json")
    if not q["rs_reader_qualified"]:
        raise ValueError("Qualified source reader, targets and routes required")
    return meta, r1, context, rm


def construct(meta, seed, device):
    model, query = old.parent.construct(meta, seed, device)
    model.eval().requires_grad_(False)
    encoder = model.core.encoder.eval().requires_grad_(True)
    query.eval().requires_grad_(True)
    local = model.local_head.eval().requires_grad_(False)
    return encoder, query, local


def local_signature(local):
    return old.parent.ur.bb.state_signature(local)


def terms(encoder, query, local, x, batch, prefixes, stats, local_stats, arm):
    if (
        arm not in objective.ARMS
        or any(m.training for m in local.modules())
        or any(p.requires_grad for p in local.parameters())
    ):
        raise ValueError("Declared arm and frozen eval local reader required")
    z, pred = old.core.predict(encoder, query, x, prefixes, True)
    y, mask = old.core.original.targets(x, prefixes, stats, True)
    parts = old.core.original.loss_rows(pred, y, mask, stats, True)
    mode = objective.ARMS[arm]
    lp = local(z if mode.local_to_encoder else z.detach()).reshape(len(x), prefixes.shape[1], 16, 7)
    ly, lm = objective.ba.local_targets(batch["y"], batch["mask"], prefixes, stats, local_stats)
    loc = objective.ba.local_rows(lp, ly, lm, True)["primary"]
    # Preserve the historical baseline's addition/mean order, not only algebraic equality.
    total = parts["objective" if mode.remote_structure else "query"].mean(1)
    if mode.local_to_encoder:
        total = total + objective.ba.LOCAL_WEIGHT * loc
    return {
        "query": parts["query"].mean(1),
        "structure": parts["structure"].mean(1),
        "local": loc,
        "objective": total,
    }


def train_epoch(meta, job, encoder, query, local, x, native, opt, epoch, device, on_step=None):
    encoder.eval()
    query.eval()
    local.eval()
    stats, local_stats = old.parent.stats(meta)
    order, ps = old.core.schedule(len(x), job["seed"], epoch)
    sums = dict.fromkeys(("query", "structure", "local", "objective"), 0.0)
    norms, factors = [], []
    old.sync(device)
    started = time.perf_counter()
    for left in range(0, len(x), meta["batch"]):
        size = min(meta["batch"], len(x) - left)
        opt.zero_grad(set_to_none=True)
        for start in range(left, left + size, meta["micro"]):
            stop = min(start + meta["micro"], left + size)
            ids = order[start:stop]
            b = torch.as_tensor(x[ids], device=device)
            p = torch.as_tensor(ps[start:stop], device=device)
            batch = {k: torch.as_tensor(v[ids], device=device) for k, v in native.items()}
            values = terms(encoder, query, local, b, batch, p, stats, local_stats, job["arm"])
            loss = values["objective"].sum() / size
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite RS loss")
            loss.backward()
            for k in sums:
                sums[k] += float(values[k].detach().sum())
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(query.parameters()), 1.0, error_if_nonfinite=True
            )
        )
        norms.append(norm)
        factors.append(min(1.0, 1.0 / (norm + 1e-6)))
        opt.step()
        if on_step:
            on_step(len(norms))
    old.sync(device)
    return {
        "loss": sums["objective"] / len(x),
        "components": {k: v / len(x) for k, v in sums.items()},
        "seconds": time.perf_counter() - started,
        "steps": len(norms),
        "examples": len(x),
        "supervised_states": 5 * len(x),
        "order_hash": old.parent.ur.bb.cov.ndarray_hash(order),
        "prefix_hash": old.parent.ur.bb.cov.ndarray_hash(ps),
        "preclip_norm": {"mean": float(np.mean(norms)), "max": max(norms)},
        "clip_factor": {"mean": float(np.mean(factors)), "min": min(factors)},
        "steps_clipped": sum(f < 1 for f in factors),
    }


def binding(out, job):
    return {"manifest_sha256": sha256(out / "manifest.json"), "job": job}


def snapshot(ck, encoder, query, epoch, validation):
    return {
        "binding": ck["binding"],
        "epoch": epoch,
        "encoder": old.cpu_state(encoder),
        "query": old.cpu_state(query),
        "validation": validation,
    }


def history_check(history, seed, n, initial):
    for epoch, row in enumerate(history, 1):
        order, ps = old.core.schedule(n, seed, epoch)
        if row["epoch"] != epoch or row["rates"] != list(lr.RATES[seed]):
            raise ValueError("Noncontiguous epochs or changed rates")
        t = row["train"]
        if (t["steps"], t["examples"], t["supervised_states"]) != (math.ceil(n / 128), n, n * 5):
            raise ValueError("Exposure or update accounting differs")
        for k, a in (("order_hash", order), ("prefix_hash", ps)):
            if t[k] != old.parent.ur.bb.cov.ndarray_hash(a):
                raise ValueError("Sampling differs")
        if row["retention"] != lr.retention(row["validation"], initial):
            raise ValueError("Recorded retention differs")


def train_job(meta, context, r1, out, job, train, native, val, device="cuda"):
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    bind = binding(out, job)
    if (folder / "completion.json").exists():
        done = read_json(folder / "completion.json")
        if done["binding"] != bind:
            raise ValueError("Worker binding changed")
        for n, h in done["files"].items():
            r0.required_file(folder, n, h)
        return done
    e, q, local = construct(meta, job["seed"], device)
    sig = lr.model_signature(e, q)
    lsig = local_signature(local)
    original = read_json(context / f"prefix_s{job['seed']}/completion.json")
    if sig != {
        "encoder": original["initial_encoder_hash"],
        "query": original["initial_query_hash"],
    }:
        raise ValueError("Original Macro signatures differ")
    opt = lr.optimizer(e, q, lr.RATES[job["seed"]])
    expected_options = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    checkpoint = folder / "last.pt"
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ck["binding"] != bind or ck["initial_signature"] != sig or ck["local_signature"] != lsig:
            raise ValueError("Resume source changed")
        if (folder / "checkpoint.json").exists() and read_json(folder / "checkpoint.json")[
            "sha256"
        ] != sha256(checkpoint):
            pending = folder / "pending_epoch.json"
            if not pending.exists() or read_json(pending)["epoch"] != ck["epoch"]:
                raise ValueError("Checkpoint receipt differs")
            atomic_json(
                {"epoch": ck["epoch"], "sha256": sha256(checkpoint)}, folder / "checkpoint.json"
            )
        lr.pending_epoch(folder, ck["epoch"])
        r0.required_file(folder, "best.pt", ck["best_sha256"])
        if not 0 <= ck["epoch"] <= EPOCHS or ck["epoch"] != len(ck["history"]):
            raise ValueError("Epoch bound differs")
        history_check(ck["history"], job["seed"], len(train), ck["initial_validation"])
        if ck["initial_validation"] != original["initial_validation"]:
            # Replays may differ within the original numeric tolerance, so recheck it.
            lr.replay_check(
                ck["initial_validation"],
                original["initial_validation"],
                folder / "resume_initial_replay.json",
            )
        best_epoch, best_score, best_validation = 0, 1.0, ck["initial_validation"]
        for row in ck["history"]:
            selected = old.core.score(row["validation"], ck["initial_validation"])
            if selected["eligible"] and selected["score"] < best_score:
                best_epoch, best_score, best_validation = (
                    row["epoch"],
                    selected["score"],
                    row["validation"],
                )
        if (ck["best_epoch"], ck["best_score"], ck["best_validation"]) != (
            best_epoch,
            best_score,
            best_validation,
        ):
            raise ValueError("Resume selection journal differs")
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
        if lr.model_signature(e, q) != ck["current_signature"]:
            raise ValueError("Resume model signature differs")
        opt.load_state_dict(ck["optimizer"])
        lr.optimizer_steps(opt, sum(h["train"]["steps"] for h in ck["history"]))
        if [
            {k: v for k, v in g.items() if k != "params"} for g in opt.param_groups
        ] != expected_options:
            raise ValueError("Resume optimizer configuration changed")
        restore_rng(ck["rng"])
    else:
        if (folder / "pending_epoch.json").exists():
            raise ValueError("Uncommitted worker without checkpoint")
        initial = lr.export_validation(meta, e, q, val, device, folder / "initial_errors.npz")
        lr.replay_check(initial, original["initial_validation"], folder / "initial_replay.json")
        ck = {
            "binding": bind,
            "epoch": 0,
            "history": [],
            "initial_signature": sig,
            "local_signature": lsig,
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_score": 1.0,
        }
        atomic_save(snapshot(ck, e, q, 0, initial), folder / "best.pt")
        ck.update(
            encoder=old.cpu_state(e),
            query=old.cpu_state(q),
            optimizer=opt.state_dict(),
            rng=rng_state(),
            best_sha256=sha256(folder / "best.pt"),
            current_signature=lr.model_signature(e, q),
        )
        atomic_save(ck, checkpoint)
        atomic_json({"epoch": 0, "sha256": sha256(checkpoint)}, folder / "checkpoint.json")
    refs = read_json(r1 / f"s{job['seed']}/history.json")
    for epoch in range(ck["epoch"] + 1, EPOCHS + 1):
        pending = {"epoch": epoch, "observed_steps": 0}
        atomic_json(pending, folder / "pending_epoch.json")

        def observed(n, pending=pending):
            pending["observed_steps"] = n
            atomic_json(pending, folder / "pending_epoch.json")

        trained = train_epoch(meta, job, e, q, local, train, native, opt, epoch, device, observed)
        if pending["observed_steps"] != trained["steps"]:
            raise ValueError("Observed updates differ")
        validation = old.validation(meta, e, q, val, device)
        if job["arm"] == "baseline" and epoch <= 5:
            lr.replay_check(
                validation, refs[epoch - 1]["validation"], folder / f"r1_epoch{epoch}_replay.json"
            )
            for k in ("order_hash", "prefix_hash", "steps", "examples", "supervised_states"):
                if trained[k] != refs[epoch - 1]["train"][k]:
                    raise ValueError("R1 baseline exposure differs")
        history = ck["history"] + [
            {
                "epoch": epoch,
                "train": trained,
                "validation": validation,
                "retention": lr.retention(validation, ck["initial_validation"]),
                "rates": list(lr.RATES[job["seed"]]),
            }
        ]
        history_check(history, job["seed"], len(train), ck["initial_validation"])
        lr.optimizer_steps(opt, sum(h["train"]["steps"] for h in history))
        if local_signature(local) != lsig:
            raise ValueError("Frozen local reader changed")
        if any(not torch.isfinite(v).all() for m in (e, q) for v in m.state_dict().values()):
            raise ValueError("Nonfinite model")
        selection = old.core.score(validation, ck["initial_validation"])
        if selection["eligible"] and selection["score"] < ck["best_score"]:
            atomic_save(snapshot(ck, e, q, epoch, validation), folder / "best.pt")
            ck.update(
                best_epoch=epoch,
                best_score=selection["score"],
                best_validation=validation,
                best_sha256=sha256(folder / "best.pt"),
            )
        ck.update(
            epoch=epoch,
            current_signature=lr.model_signature(e, q),
            history=history,
            encoder=old.cpu_state(e),
            query=old.cpu_state(q),
            optimizer=opt.state_dict(),
            rng=rng_state(),
        )
        atomic_save(ck, checkpoint)
        atomic_json({"epoch": epoch, "sha256": sha256(checkpoint)}, folder / "checkpoint.json")
        atomic_json(history, folder / "history.json")
        lr.pending_epoch(folder, epoch)
        progress(
            f"RS {job['name']}: {epoch}/{EPOCHS}, best={ck['best_epoch']}, near={validation['full/endpoint/near']:.6f}, worst={history[-1]['retention']['worst_ratio']:.4f}, {trained['seconds']:.1f}s/train_epoch"
        )
    r0.causality(e, q, torch.as_tensor(train[:2], device=device), folder, "last")
    final = lr.export_validation(meta, e, q, val, device, folder / "last_errors.npz")
    lr.replay_check(final, ck["history"][-1]["validation"], folder / "last_replay.json")
    done = {
        "binding": bind,
        "epochs": ck["epoch"],
        "best_epoch": ck["best_epoch"],
        "optimizer_updates": sum(h["train"]["steps"] for h in ck["history"]),
        "initial_signature": sig,
        "local_signature": lsig,
        "files": {
            p.name: sha256(p)
            for p in folder.iterdir()
            if p.is_file() and p.suffix in (".json", ".pt", ".npz") and p.name != "completion.json"
        },
    }
    atomic_json(done, folder / "completion.json")
    del e, q, local, opt, ck
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def check_disk(out, r1):
    largest = max((r1 / f"s{s}/last.pt").stat().st_size for s in (42, 43))
    total = math.ceil(13 * largest + 1.2 * 2**30)
    owned = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    need = max(512 * 2**20, total - owned)
    if shutil.disk_usage(out).free < need:
        raise ValueError(
            f"Need {need / 2**30:.2f}GiB free for remaining RS outputs; no source deleted"
        )
    return {
        "estimated_total_bytes": total,
        "existing_output_bytes": owned,
        "required_new_bytes": need,
    }


def lock_models(out, results):
    lock = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "workers": results,
        "files": {
            f"{j['name']}/{k}.pt": sha256(out / j["name"] / f"{k}.pt")
            for j in jobs()
            for k in ("best", "last")
        },
    }
    if (out / "model_selection_lock.json").exists() and read_json(
        out / "model_selection_lock.json"
    ) != lock:
        raise ValueError("Model lock changed")
    atomic_json(lock, out / "model_selection_lock.json")


def execute(source, out):
    from . import recovery_rs_evaluate as ev

    state = {"schema": SCHEMA, "task_ids": ["V04", "V05"], "status": "running", "promoted": False}
    atomic_json(state, out / "status.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Real experiment requires AutoDL CUDA")
        progress("RS-v2: verifying reviewed sources; R1 failure remains recorded")
        meta, r1, context, rm = verify_source(source, out)
        if torch.__version__ != rm["torch"] or torch.cuda.get_device_name() != rm["gpu"]:
            raise ValueError("Use reviewed R1/V07 CUDA environment")
        manifest = {
            "schema": SCHEMA,
            "source": str(source),
            "context": str(context),
            "r1": str(r1),
            "implementation": implementation(),
            "epochs": EPOCHS,
            "jobs": jobs(),
            "session_cap": SESSION_CAP,
            "runtime": r0.RUNTIME_PROFILES["context"],
            "gpu": rm["gpu"],
            "torch": rm["torch"],
        }
        if (out / "manifest.json").exists() and read_json(out / "manifest.json") != manifest:
            raise ValueError("RS manifest changed")
        atomic_json(manifest, out / "manifest.json")
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            for n, h in done["files"].items():
                r0.required_file(out, n, h)
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        atomic_json(check_disk(out, r1), out / "disk_preflight.json")
        shutil.copyfile(lr.REPO / "docs/BABEL_RECOVERY_RS.md", out / "protocol.md")
        train, native, rows = qual.aligned_data(context, meta, "train")
        val, vrows = old.data.cache(context, "val")
        if len(train) != 4789 or len(val) != 978:
            raise ValueError("Training/validation sizes changed")
        atomic_json(rows, out / "train_rows.json")
        atomic_json(vrows, out / "val_rows.json")
        with r0.replay_runtime("context"):
            results = {
                j["name"]: train_job(meta, context, r1, out, j, train, native, val) for j in jobs()
            }
            if any(d["epochs"] != 30 or d["optimizer_updates"] != 1140 for d in results.values()):
                raise ValueError("Incomplete equal-budget matrix")
            lock_models(out, results)
            del train, native, val
            gc.collect()
            ev.run(meta, context, out, "cuda")
        verify_source(source, out)
        if implementation() != manifest["implementation"]:
            raise ValueError("Implementation changed during RS")
        state.update(
            status="rs_complete_requires_review",
            optimizer_updates=9120,
            worker_epochs=240,
            source_unchanged=True,
        )
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".npz", ".pt", ".md")
            and p.name not in ("status.json", "budget.json", "request.json")
            and p != out / "completion.json"
        }
        atomic_json({**state, "files": files}, out / "completion.json")
        atomic_json(state, out / "status.json")
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(state, out / "status.json")
        raise


def supervise(source, out):
    source = Path(source).resolve()
    out = interface.separate(out, source)
    request = {"schema": SCHEMA, "source": str(source), "output": str(out)}
    if (
        out.exists()
        and any(out.iterdir())
        and (not (out / "request.json").exists() or read_json(out / "request.json") != request)
    ):
        raise ValueError("Refuse existing unbound output")
    if (source / "manifest.json").exists():
        im = read_json(source / "manifest.json")
        r1 = Path(im["source"])
        interface.separate(out, r1)
        if (r1 / "manifest.json").exists():
            rm = read_json(r1 / "manifest.json")
            interface.separate(out, rm["source"], rm["qualification"])
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
        if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
            raise ValueError("RS cumulative time budget exhausted")
        remaining = SESSION_CAP - used
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget["used_seconds"] = SESSION_CAP
        atomic_json(budget, out / "budget.json")
        started = time.monotonic()
        try:
            code = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "obson.babel.recovery_rs",
                    "--worker",
                    "--source",
                    str(source),
                    "--out",
                    str(out),
                ],
                timeout=remaining,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
            atomic_json(
                {
                    "schema": SCHEMA,
                    "task_ids": ["V04", "V05"],
                    "status": "timeout",
                    "promoted": False,
                },
                out / "status.json",
            )
        elapsed = min(remaining, time.monotonic() - started)
        budget["used_seconds"] = used + elapsed
        attempt.update(elapsed_seconds=elapsed, exit_code=code)
        atomic_json(budget, out / "budget.json")
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    source, out = a.source.resolve(), interface.separate(a.out, a.source)
    if a.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(source, out))
    raise SystemExit(supervise(source, out))


if __name__ == "__main__":
    main()
