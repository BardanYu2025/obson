"""Bounded F01/F11 2x2 scratch study. Real execution is AutoDL CUDA only."""

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

from . import state_coverage as core
from . import state_coverage_data as data
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

old = data.source.old
lr = data.source.lr
SCHEMA = "babel-state-coverage768-v1"
EPOCHS = 120
SESSION_CAP = 129600


def jobs():
    return [{"name": f"{a}_s{s}", "arm": a, "seed": s} for a in core.ARMS for s in (42, 43)]


def protocol():
    return {
        "task_ids": ["F01", "F11"],
        "arms": {a: {"sampling": v[0], "extra_structure": v[1]} for a, v in core.ARMS.items()},
        "epochs": EPOCHS,
        "seeds": [42, 43],
        "batch": 128,
        "micro": 8,
        "updates": 8 * EPOCHS * 38,
        "session_seconds": SESSION_CAP,
        "source": "V06 qualified data; random rolling768/4/8; original Macro97/94 frozen reference only",
        "targets": "all endpoints1..128 independently see128; past ages1..127; inherited Q with optional S; no new local head",
        "uniform_sampling": "5 without replacement estimates equal128 objective; not dense128 per step",
        "selection": "fixed last120 primary; best validation Q diagnostic only; no research checkpoint selection",
        "readouts": "13 representations x13 targets x5 alphas=845; train fit/val choose; no encoder gradient",
        "primary": "D vs A utility gain5%; family and endpoint/interior age-band price/activity and window priceP95 retention10%; both seeds/cohorts/month and contract-period-month intervals",
        "candidate": "primary plus old absolute utility/current criteria, raw/PCA/current baselines and Macro utility5%/history10% retention; no promotion",
        "inference": "paired2000 draws seed20261009; >=100 parents/6groups; reused research cohorts; V13 open",
        "stop": "fixed budget; no automatic extension; committed-boundary resume only; failed/stopped exports",
    }


def implementation():
    from . import state_coverage_evaluate as ev
    from . import utility_probe_run

    return {
        "inherited": data.source.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (core, data, ev, ev.up, ev.up.probe, utility_probe_run, sys.modules[__name__])
        },
        "script": sha256(lr.REPO / "scripts/babel_state_coverage768_autodl.sh"),
    }


def binding(out, job):
    return {"manifest_sha256": sha256(out / "manifest.json"), "job": job}


def signature(e, q):
    return lr.model_signature(e, q)


def cpu_state(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def optimizer(e, q):
    return torch.optim.AdamW(
        list(e.parameters()) + list(q.parameters()),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )


@torch.no_grad()
def validation(meta, e, q, x, device):
    e.eval()
    q.eval()
    bank = {}
    for left in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[left : left + meta["micro"]], device=device)
        ps = torch.tensor(core.VAL_POSITIONS, device=device).expand(len(b), -1)
        parts = {}
        for start in range(0, ps.shape[1], 5):
            pp = ps[:, start : start + 5]
            _, pred = core.predict(e, q, b, pp)
            y, m = core.targets(b, pp, meta["statistics"])
            for k, v in core.band_rows(pred, y, m, meta["statistics"]).items():
                parts.setdefault(k, []).append(v.cpu().numpy())
        for k, v in parts.items():
            bank.setdefault(k, []).append(np.concatenate(v, axis=1))
    return {k: np.concatenate(v).mean(0).tolist() for k, v in bank.items()}


def score(values):
    return float(np.mean(values["query"]))


def train_epoch(meta, job, e, q, x, opt, epoch, device, on_step=None):
    e.train()
    q.train()
    order, ps = core.schedule(len(x), job["seed"], epoch, job["arm"])
    sums = dict.fromkeys(("query", "structure", "objective"), 0.0)
    norms = []
    counts = np.zeros(128, dtype=np.int64)
    sync(device)
    start_time = time.perf_counter()
    for left in range(0, len(x), meta["batch"]):
        size = min(meta["batch"], len(x) - left)
        opt.zero_grad(set_to_none=True)
        for start in range(left, left + size, meta["micro"]):
            stop = min(start + meta["micro"], left + size)
            ids = order[start:stop]
            b = torch.as_tensor(x[ids], device=device)
            p = torch.as_tensor(ps[start:stop], device=device)
            terms = core.terms(e, q, b, p, meta["statistics"], job["arm"])
            loss = terms["objective"].sum() / size
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            for k in sums:
                sums[k] += float(terms[k].detach().sum())
            counts += np.bincount(ps[start:stop].ravel() - 1, minlength=128)
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(e.parameters()) + list(q.parameters()), 1.0, error_if_nonfinite=True
            )
        )
        norms.append(norm)
        opt.step()
        if on_step:
            on_step(len(norms))
    sync(device)
    return {
        "steps": len(norms),
        "examples": len(x),
        "supervised_states": len(x) * 5,
        "seconds": time.perf_counter() - start_time,
        "components": {k: v / len(x) for k, v in sums.items()},
        "order_hash": old.parent.ur.bb.cov.ndarray_hash(order),
        "prefix_hash": old.parent.ur.bb.cov.ndarray_hash(ps),
        "position_counts": counts.tolist(),
        "preclip_mean": float(np.mean(norms)),
        "preclip_max": max(norms),
        "steps_clipped": sum(v > 1 for v in norms),
    }


def history_check(meta, job, ck, n):
    if ck["epoch"] != len(ck["history"]) or not 0 <= ck["epoch"] <= EPOCHS:
        raise ValueError("Invalid committed epoch")
    best_epoch = 0
    best_score = score(ck["initial_validation"])
    for epoch, row in enumerate(ck["history"], 1):
        order, ps = core.schedule(n, job["seed"], epoch, job["arm"])
        t = row["train"]
        expected = {
            "order_hash": old.parent.ur.bb.cov.ndarray_hash(order),
            "prefix_hash": old.parent.ur.bb.cov.ndarray_hash(ps),
            "position_counts": np.bincount(ps.ravel() - 1, minlength=128).tolist(),
            "steps": math.ceil(n / meta["batch"]),
            "examples": n,
            "supervised_states": n * 5,
        }
        if (
            row["epoch"] != epoch
            or row["lr"] != core.learning_rate(epoch, EPOCHS, 3e-4)
            or any(t[k] != v for k, v in expected.items())
        ):
            raise ValueError("History sampling/update/LR accounting differs")
        value = score(row["validation"])
        if value < best_score:
            best_epoch, best_score = epoch, value
    if (ck["best_epoch"], ck["best_score"]) != (best_epoch, best_score):
        raise ValueError("Validation selection journal changed")


def materialize(ck, folder):
    atomic_save(
        {
            "binding": ck["binding"],
            "epoch": ck["best_epoch"],
            "encoder": ck["best_encoder"],
            "query": ck["best_query"],
        },
        folder / "best.pt",
    )
    atomic_json(ck["history"], folder / "history.json")


def train_job(meta, out, job, train, val, device="cuda"):
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    bind = binding(out, job)
    if (folder / "completion.json").exists():
        done = read_json(folder / "completion.json")
        if done["binding"] != bind:
            raise ValueError("Worker identity changed")
        old.verify_files(folder, done["files"])
        return done
    e, q = core.make_models("rolling", job["seed"], device, meta["encoder"])
    initial_sig = signature(e, q)
    opt = optimizer(e, q)
    checkpoint = folder / "last.pt"
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ck["binding"] != bind or ck["initial_signature"] != initial_sig:
            raise ValueError("Resume initialization/manifest changed")
        receipt = (
            read_json(folder / "checkpoint.json") if (folder / "checkpoint.json").exists() else None
        )
        pending = (
            read_json(folder / "pending_epoch.json")
            if (folder / "pending_epoch.json").exists()
            else None
        )
        if (receipt is None or receipt["sha256"] != sha256(checkpoint)) and not (
            (pending and pending["epoch"] == ck["epoch"]) or (receipt is None and ck["epoch"] == 0)
        ):
            raise ValueError("Checkpoint receipt mismatch without atomic commit evidence")
        lr.pending_epoch(folder, ck["epoch"])
        history_check(meta, job, ck, len(train))
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
        if signature(e, q) != ck["current_signature"]:
            raise ValueError("Resumed model changed")
        opt.load_state_dict(ck["optimizer"])
        if any(
            not torch.isfinite(v).all()
            for state in opt.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ):
            raise ValueError("Nonfinite optimizer state")
        lr.optimizer_steps(opt, sum(h["train"]["steps"] for h in ck["history"]))
        expected_lr = core.learning_rate(ck["epoch"], EPOCHS, 3e-4) if ck["epoch"] else 3e-4
        for group in opt.param_groups:
            if (group["lr"], group["betas"], group["eps"], group["weight_decay"]) != (
                expected_lr,
                (0.9, 0.999),
                1e-8,
                0.01,
            ):
                raise ValueError("Optimizer configuration changed")
        restore_rng(ck["rng"])
    else:
        if (folder / "pending_epoch.json").exists():
            raise ValueError("Uncommitted updates without checkpoint")
        audit = core.audit(e, q, torch.as_tensor(train[:2], device=device), meta["statistics"])
        atomic_json(audit, folder / "initial_audit.json")
        if not audit["passed"]:
            raise ValueError("Initial full-position qualification failed")
        initial = validation(meta, e, q, val, device)
        ck = {
            "binding": bind,
            "epoch": 0,
            "history": [],
            "initial_signature": initial_sig,
            "initial_validation": initial,
            "best_epoch": 0,
            "best_score": score(initial),
            "best_encoder": cpu_state(e),
            "best_query": cpu_state(q),
        }
        ck.update(
            encoder=cpu_state(e),
            query=cpu_state(q),
            optimizer=opt.state_dict(),
            rng=rng_state(),
            current_signature=signature(e, q),
        )
        atomic_save(ck, checkpoint)
    atomic_json({"epoch": ck["epoch"], "sha256": sha256(checkpoint)}, folder / "checkpoint.json")
    materialize(ck, folder)
    for epoch in range(ck["epoch"] + 1, EPOCHS + 1):
        pending = {"epoch": epoch, "observed_steps": 0}
        atomic_json(pending, folder / "pending_epoch.json")

        def observed(steps, pending=pending):
            pending["observed_steps"] = steps
            atomic_json(pending, folder / "pending_epoch.json")

        rate = core.learning_rate(epoch, EPOCHS, 3e-4)
        for g in opt.param_groups:
            g["lr"] = rate
        tick = time.perf_counter()
        trained = train_epoch(meta, job, e, q, train, opt, epoch, device, observed)
        values = validation(meta, e, q, val, device)
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite validation")
        if score(values) < ck["best_score"]:
            ck.update(
                best_epoch=epoch,
                best_score=score(values),
                best_encoder=cpu_state(e),
                best_query=cpu_state(q),
            )
        ck["history"].append(
            {
                "epoch": epoch,
                "lr": rate,
                "train": trained,
                "validation": values,
                "epoch_seconds": time.perf_counter() - tick,
            }
        )
        ck.update(
            epoch=epoch,
            encoder=cpu_state(e),
            query=cpu_state(q),
            optimizer=opt.state_dict(),
            rng=rng_state(),
            current_signature=signature(e, q),
        )
        lr.optimizer_steps(opt, epoch * math.ceil(len(train) / meta["batch"]))
        if any(not torch.isfinite(v).all() for m in (e, q) for v in m.state_dict().values()):
            raise ValueError("Nonfinite weights")
        atomic_save(ck, checkpoint)
        atomic_json({"epoch": epoch, "sha256": sha256(checkpoint)}, folder / "checkpoint.json")
        materialize(ck, folder)
        lr.pending_epoch(folder, epoch)
        progress(
            f"{job['name']}: {epoch}/{EPOCHS}, Q={score(values):.6f}, best(diagnostic)={ck['best_epoch']}, {ck['history'][-1]['epoch_seconds']:.1f}s/epoch"
        )
    history_check(meta, job, ck, len(train))
    audit = core.audit(e, q, torch.as_tensor(train[:2], device=device), meta["statistics"])
    atomic_json(audit, folder / "last_audit.json")
    if not audit["passed"]:
        raise ValueError("Trained full-position qualification failed")
    final = validation(meta, e, q, val, device)
    if not all(
        np.allclose(v, ck["history"][-1]["validation"][k], atol=1e-6, rtol=2e-5)
        for k, v in final.items()
    ):
        raise ValueError("Last validation replay mismatch")
    done = {
        "binding": bind,
        "epochs": EPOCHS,
        "optimizer_updates": sum(h["train"]["steps"] for h in ck["history"]),
        "initial_signature": initial_sig,
        "best_epoch": ck["best_epoch"],
        "files": {
            p.name: sha256(p)
            for p in folder.iterdir()
            if p.suffix in (".pt", ".json") and p.name != "completion.json"
        },
    }
    atomic_json(done, folder / "completion.json")
    del e, q, opt, ck
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return done


def lock_models(out, results):
    if set(results) != {j["name"] for j in jobs()}:
        raise ValueError("Incomplete eight-arm matrix")
    for s in (42, 43):
        ds = [results[f"{a}_s{s}"] for a in core.ARMS]
        if any(
            d["initial_signature"] != ds[0]["initial_signature"]
            or d["epochs"] != EPOCHS
            or d["optimizer_updates"] != 38 * EPOCHS
            for d in ds
        ):
            raise ValueError("Unmatched start or budget")
    lock = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "workers": results,
        "files": {f"{j['name']}/last.pt": sha256(out / j["name"] / "last.pt") for j in jobs()},
    }
    if (out / "model_selection_lock.json").exists() and read_json(
        out / "model_selection_lock.json"
    ) != lock:
        raise ValueError("Locked models changed")
    atomic_json(lock, out / "model_selection_lock.json")


def source_replay(meta, out, val, device):
    for seed in (42, 43):
        e, q = old.construct(meta["source_meta"], seed, device)
        expected = read_json(Path(meta["context"]) / f"prefix_s{seed}/completion.json")
        expected_sig = {
            "encoder": expected["initial_encoder_hash"],
            "query": expected["initial_query_hash"],
        }
        if signature(e, q) != expected_sig:
            raise ValueError("Macro reference initialization changed")
        values = old.validation(meta["source_meta"], e, q, val, device)
        data.source.r0.save_checks(
            out,
            f"macro_s{seed}_original_replay",
            data.source.r0.nested_compare(values, expected["initial_validation"]),
        )
        if signature(e, q) != expected_sig:
            raise ValueError("Macro qualification changed parameters")
        del e, q
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def resource_preflight(meta, out, train, val, remaining, device="cuda"):
    """No optimization; time forward/backward, nine-position validation and full128 evaluation separately."""
    e, q = core.make_models("rolling", 42, device, meta["encoder"])
    b = torch.as_tensor(train[: meta["micro"]], device=device)
    ps = torch.as_tensor(core.schedule(len(b), 42, 1, "B")[1], device=device)
    parameters = sum(p.numel() for m in (e, q) for p in m.parameters())

    def measure(count, backward, e=e, q=q, b=b):
        p = ps if backward else torch.arange(1, count + 1, device=device).expand(len(b), -1)
        if count == 9:
            p = torch.tensor(core.VAL_POSITIONS, device=device).expand(len(b), -1)
        sync(device)
        start = time.perf_counter()
        if backward:
            e.zero_grad(set_to_none=True)
            q.zero_grad(set_to_none=True)
            core.terms(e, q, b, p, meta["statistics"], "B")["objective"].mean().backward()
        else:
            with torch.no_grad():
                for first in range(0, count, 5):
                    pp = p[:, first : first + 5]
                    _, pred = core.predict(e, q, b, pp)
                    y, m = core.targets(b, pp, meta["statistics"])
                    core.band_rows(pred, y, m, meta["statistics"])
        sync(device)
        return time.perf_counter() - start

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    measure(5, True)  # warm-up outside throughput sample
    gradient_groups = {
        label: float(
            sum(
                p.grad.detach().double().square().sum()
                for p in model.parameters()
                if p.grad is not None
            ).sqrt()
        )
        for label, model in (("encoder", e), ("query", q))
    }
    if any(not math.isfinite(v) or v <= 0 for v in gradient_groups.values()):
        raise ValueError("Actual loss failed to reach encoder/query")
    train_seconds = max(measure(5, True) for _ in range(2))
    e.eval()
    q.eval()
    val_seconds = measure(9, False)
    research_seconds = measure(128, False)
    peak = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0
    free, total = torch.cuda.mem_get_info() if str(device).startswith("cuda") else (2**60, 2**60)
    reserved = torch.cuda.memory_reserved() if str(device).startswith("cuda") else 0
    estimated_gpu = peak + parameters * 12 + 512 * 2**20
    last_bytes = parameters * 20  # parameters+Adam+best copy; runtime serialization reserve below
    best_bytes = parameters * 4
    cache_bytes = (4789 + 978 + 984 + 453) * 10 * 768 * 4
    total_disk = int((8 * (last_bytes + best_bytes) + last_bytes) * 1.15 + cache_bytes + 2 * 2**30)
    owned = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    disk_need = max(512 * 2**20, total_disk - owned)
    completed = sum(
        read_json(out / j["name"] / "completion.json")["epochs"]
        if (out / j["name"] / "completion.json").exists()
        else read_json(out / j["name"] / "checkpoint.json")["epoch"]
        if (out / j["name"] / "checkpoint.json").exists()
        else 0
        for j in jobs()
    )
    estimate = (8 * EPOCHS - completed) * (
        math.ceil(len(train) / len(b)) * train_seconds + math.ceil(len(val) / len(b)) * val_seconds
    )
    estimate += (
        10 * math.ceil((984 + 453) / len(b)) * research_seconds + 3600
    )  # extraction/Ridge/io margin
    report = {
        "parameters": parameters,
        "initial_loss_gradient_norms": gradient_groups,
        "peak_bytes": peak,
        "estimated_with_optimizer_bytes": estimated_gpu,
        "gpu_total_bytes": total,
        "gpu_available_including_own_allocator_bytes": free + reserved,
        "required_new_disk_bytes": disk_need,
        "disk_free_bytes": shutil.disk_usage(out).free,
        "seconds_per_micro": {
            "train5": train_seconds,
            "val9": val_seconds,
            "research128": research_seconds,
        },
        "estimated_remaining_seconds_with_margin": estimate * 1.25,
        "session_remaining_seconds": remaining,
    }
    atomic_json(report, out / "resource_preflight.json")
    del measure, e, q, b
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    # Cached blocks owned by this process are reusable, not competing allocations.
    if estimated_gpu > free + reserved:
        raise ValueError("Insufficient measured GPU memory; no protocol change")
    if disk_need > report["disk_free_bytes"]:
        raise ValueError(f"Need {disk_need / 2**30:.2f}GiB free; sources retained")
    if estimate * 1.25 > remaining:
        raise ValueError("Measured workload exceeds remaining36h budget; zero additional updates")
    return report


def execute(root, out):
    from . import state_coverage_evaluate as ev

    state = {"schema": SCHEMA, "task_ids": ["F01", "F11"], "status": "running", "promoted": False}
    atomic_json(state, out / "status.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Real training requires AutoDL CUDA")
        progress("F01/F11: verifying reviewed lineage and immutable full128-context data")
        source_info = data.verify(root, out)
        meta = {
            **source_info,
            "schema": SCHEMA,
            "source": str(root),
            "implementation": implementation(),
            "protocol": protocol(),
            "encoder": core.CONFIG,
            "micro": 8,
            "batch": 128,
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
        }
        if (out / "manifest.json").exists() and read_json(out / "manifest.json") != meta:
            raise ValueError("Bound run manifest changed")
        atomic_json(meta, out / "manifest.json")
        atomic_json(protocol(), out / "protocol.json")
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            old.verify_files(out, done["files"])
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        train, rows = data.cache(meta, out, "train")
        val, vrows = data.cache(meta, out, "val")
        atomic_json(data.row_audit(rows, vrows), out / "partition_audit.json")
        with data.source.r0.replay_runtime("context") as runtime:
            atomic_json(runtime, out / "runtime.json")
            data.raw_audit(meta, out, ("train", "val"))
            source_replay(meta, out, val, "cuda")
            if not (out / "model_selection_lock.json").exists():
                remaining = read_json(out / "budget.json")["deadline_unix"] - time.time()
                resource_preflight(meta, out, train, val, remaining)
                results = {j["name"]: train_job(meta, out, j, train, val) for j in jobs()}
                lock_models(out, results)
            ev.check_selection(out)
            del train, val
            gc.collect()
            ev.fit(meta, out, "cuda")
            data.raw_audit(meta, out, ("test",))
            data.raw_audit(meta, out, ("cross_research",))
            ev.evaluate(meta, out, "cuda")
        if data.verify(root, out) != source_info or implementation() != meta["implementation"]:
            raise ValueError("Source/code changed during run")
        state.update(
            status="f01_f11_complete_requires_review",
            optimizer_updates=8 * EPOCHS * 38,
            source_unchanged=True,
        )
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".npz", ".pt", ".html")
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


def supervise(root, out):
    root = root.resolve()
    out = data.source.interface.separate(out, root)
    request = {"schema": SCHEMA, "source": str(root), "output": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unexpected = [
            p.name for p in out.iterdir() if p.name not in ("session.lock", "request.json")
        ]
        if ((out / "request.json").exists() and read_json(out / "request.json") != request) or (
            unexpected and not (out / "request.json").exists()
        ):
            raise ValueError("Existing unbound output; no overwrite")
        atomic_json(request, out / "request.json")
        budget = (
            read_json(out / "budget.json")
            if (out / "budget.json").exists()
            else {"used_seconds": 0.0, "attempts": []}
        )
        used = budget["used_seconds"]
        if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
            raise ValueError("Cumulative36h budget exhausted; export only")
        remaining = SESSION_CAP - used
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget.update(used_seconds=SESSION_CAP, deadline_unix=time.time() + remaining)
        atomic_json(budget, out / "budget.json")
        start = time.monotonic()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "obson.babel.state_coverage_run",
                    "--worker",
                    "--source",
                    str(root),
                    "--out",
                    str(out),
                ],
                timeout=remaining,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            atomic_json(
                {"schema": SCHEMA, "status": "timeout", "promoted": False}, out / "status.json"
            )
        elapsed = min(remaining, time.monotonic() - start)
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
    root = a.source.resolve()
    out = data.source.interface.separate(a.out, root)
    if a.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(root, out))
    raise SystemExit(supervise(root, out))


if __name__ == "__main__":
    main()
