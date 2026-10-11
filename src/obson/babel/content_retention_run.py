"""F03/F11 bounded matched continuation; real weights execute on AutoDL only."""

import argparse
import fcntl
import gc
import math
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

from . import content_retention as core
from . import content_retention_data as data
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-content-retention768-v1"
MODULE = "obson.babel.content_retention_run"
EPOCHS, BATCH, MICRO, SESSION_CAP = 60, 128, 8, 86400
RATE = 1.5e-5
base = data.diagnostic.prior.source_run
r0 = data.diagnostic.prior.r0


def jobs():
    return [
        {"name": f"{arm}_s{seed}", "arm": arm, "seed": seed}
        for seed in core.SEEDS
        for arm in core.ARMS
    ]


def protocol():
    return {
        "schema": SCHEMA,
        "task_ids": ["F03", "F11"],
        "source_completion": data.COMPLETION,
        "start": "D_s42/D_s43 fixed last120; Macro97/94 remains reference, not trained",
        "architecture": "unchanged768/4/8 encoder; original4x192,2layer4head seven-field query; both trainable, fixed scaling buffers",
        "arms": {"control": "Q", "content": "Q+0.10H", "aligned": "Q+0.10H+0.01G"},
        "Q": "original band-balanced seven-field SmoothL1 loss, original cached labels and masks; no S",
        "H": "mean of both raw-truth common-content losses; path centered per shared segment, price in one train delta scale, all common bars equal, original body/activity train scales and masks; 0.4path+0.2change1+0.1body+0.3activity",
        "G": "same family loss between own centered shared predictions; no whole-state match or old prediction as truth",
        "data": dict(data.COUNTS),
        "sampling": "one shared order/pair schedule per seed/epoch; uniform shift1or16 then uniform later endpoint[d+1,128]; both states independently see128 causal prepared features in255block; pair exposure not uniform128 and not the old D five-position schedule",
        "raw_price": "original verified float64 CSV close labels; never encoder/query inputs; original Q coordinate retained separately",
        "held_shift": 64,
        "epochs": EPOCHS,
        "seeds": list(core.SEEDS),
        "batch": BATCH,
        "micro": MICRO,
        "worker_epochs": 6 * EPOCHS,
        "optimizer_updates": 6 * EPOCHS * 38,
        "train_state_exposures": 6 * EPOCHS * 4789 * 2,
        "optimizer": {
            "kind": "fresh AdamW both encoder/query",
            "lr": RATE,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "clip": 1.0,
            "scheduler": "constant; original D terminal lr, not a resumed Adam trajectory",
        },
        "runtime": "context FP32, no AMP/TF32, original dropout0; train mode during training, eval during validation",
        "selection": "fixed last60 primary; epoch0 frozen D baseline; val at0/every5/60 monitor only, no best or early-stop selection",
        "evaluation_positions": list(core.POSITIONS),
        "overlap_shifts": [1, 16, 64],
        "primary": "content vs matched control AND own frozen D: later common shape MSE gain>=5% at shifts16/64, both seeds/cohorts and both cluster intervals; no promotion",
        "protection": "<=10% relative MSE cost vs both control and own D: original price/activity near/mid/far each8positions; physical common anchored/shape, entered/leaving, disagreement all3shifts; each body/activity field entered/leaving; decoded16/64 descriptors; parent price/activity P95 point ratios<=1.10",
        "incremental_alignment": "aligned vs content: >=10% disagreement MSE gain all3shifts, both seeds/cohorts/cluster intervals; also same>=5% physical primary gain vs control/D and<=10% protection vs content/control/D; content arm primary must pass first",
        "acceptance_scope": "limited paired historical-content candidate only; not all128-position/current-field utility or independent market/prediction acceptance; failure/insufficient ends fixed recipe without search",
        "inference": "2000 paired month and contract_period_month draws seed20261010; >=100parents/6groups; reused research, no multiplicity-adjusted independent test; P95 descriptive",
        "session_seconds": SESSION_CAP,
        "jobs_allowed": [1, 2],
        "default_jobs": 2,
        "resume": "atomic completed epochs only; pending uncommitted epoch stops, no repeat updates; unfinished evaluation chunks export-only; completed outputs reused",
        "fits": 0,
        "preflight": "one joint forward and four gradient extractions checking Q+0.1H+0.01G; three zero-update backward timings; source signatures unchanged",
        "extra_grid": False,
        "promoted": False,
    }


def implementation():
    from . import content_retention_evaluate as evaluate

    return {
        "inherited": data.diagnostic.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (core, data, evaluate, sys.modules[__name__])
        },
        "script": sha256(base.lr.REPO / "scripts/babel_content_retention768_autodl.sh"),
    }


def binding(out, job):
    return {"manifest": sha256(out / "manifest.json"), "job": job}


def optimizer(e, q):
    return torch.optim.AdamW(
        list(e.parameters()) + list(q.parameters()),
        lr=RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )


def train_epoch(meta, job, e, q, x, closes, opt, epoch, device, on_step=None):
    e.train()
    q.train()
    order, pairs = core.schedule(len(x), job["seed"], epoch)
    total = dict.fromkeys(("query", "content", "alignment", "objective"), 0.0)
    norms = []
    start_time = time.monotonic()
    for left in range(0, len(x), meta["batch"]):
        size = min(meta["batch"], len(x) - left)
        opt.zero_grad(set_to_none=True)
        for first in range(left, left + size, meta["micro"]):
            stop = min(first + meta["micro"], left + size)
            idx = order[first:stop]
            values = core.terms(
                e,
                q,
                torch.as_tensor(x[idx], device=device),
                torch.as_tensor(closes[idx], device=device),
                torch.as_tensor(pairs[first:stop], device=device),
                meta["statistics"],
                job["arm"],
            )
            loss = values["objective"].sum() / size
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            for k in total:
                total[k] += float(values[k].detach().sum())
        norm = torch.nn.utils.clip_grad_norm_(
            list(e.parameters()) + list(q.parameters()), 1.0, error_if_nonfinite=True
        )
        norms.append(float(norm))
        opt.step()
        if on_step:
            on_step(len(norms))
    base.sync(device)
    hash_array = base.old.parent.ur.bb.cov.ndarray_hash
    return {
        "steps": len(norms),
        "parents": len(x),
        "states": len(x) * 2,
        "order_hash": hash_array(order),
        "pairs_hash": hash_array(pairs),
        "position_counts": np.bincount(pairs.ravel() - 1, minlength=128).tolist(),
        "shift_counts": {
            str(d): int(((pairs[:, 1] - pairs[:, 0]) == d).sum()) for d in core.TRAIN_SHIFTS
        },
        "components": {k: v / len(x) for k, v in total.items()},
        "seconds": time.monotonic() - start_time,
        "preclip_mean": float(np.mean(norms)),
        "preclip_max": max(norms),
        "clipped_steps": sum(v > 1 for v in norms),
    }


@torch.no_grad()
def validation(meta, e, q, x, device):
    e.eval()
    q.eval()
    bank = {}
    for left in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[left : left + meta["micro"]], device=device)
        ps = torch.tensor((64, 112, 127, 128), device=device).expand(len(b), -1)
        _, pred = core.source.predict(e, q, b, ps)
        y, mask = core.source.targets(b, ps, meta["statistics"])
        for k, v in core.source.band_rows(pred, y, mask, meta["statistics"]).items():
            bank.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v).mean(0).tolist() for k, v in bank.items()}


def history_check(meta, job, ck, n):
    if ck["epoch"] != len(ck["history"]) or not 0 <= ck["epoch"] <= EPOCHS:
        raise ValueError("Changed epoch journal")
    for epoch, h in enumerate(ck["history"], 1):
        order, pairs = core.schedule(n, job["seed"], epoch)
        fn = base.old.parent.ur.bb.cov.ndarray_hash
        expected = {
            "steps": math.ceil(n / meta["batch"]),
            "parents": n,
            "states": n * 2,
            "order_hash": fn(order),
            "pairs_hash": fn(pairs),
            "position_counts": np.bincount(pairs.ravel() - 1, minlength=128).tolist(),
            "shift_counts": {
                str(d): int(((pairs[:, 1] - pairs[:, 0]) == d).sum()) for d in core.TRAIN_SHIFTS
            },
        }
        if (
            h["epoch"] != epoch
            or h["lr"] != RATE
            or any(h["train"][k] != v for k, v in expected.items())
        ):
            raise ValueError("Sampling/update/LR budget changed")
        if (h["validation"] is not None) != (epoch % 5 == 0 or epoch == EPOCHS):
            raise ValueError("Validation opportunities changed")


@torch.no_grad()
def qualify(meta, out, job, e, q, x, initial, device):
    audit = core.source.audit(e, q, torch.as_tensor(x[:2], device=device), meta["statistics"])
    atomic_json(audit, out / job["name"] / ("initial_audit.json" if initial else "last_audit.json"))
    if not audit["passed"]:
        raise ValueError("Causality/batch/target qualification failed")
    if initial:
        with np.load(
            Path(meta["source"]) / f"val_D_s{job['seed']}.npz", allow_pickle=False
        ) as saved:
            expected = saved["query"][:2]
        b = torch.as_tensor(x[:2], device=device)
        ps = torch.tensor((64, 112, 127, 128), device=device).expand(len(b), -1)
        _, pred = core.source.predict(e, q, b, ps)
        r0.save_checks(
            out / job["name"],
            "source_replay",
            {"query": r0.compare(pred.cpu().numpy(), expected, 1e-4, 2e-4)},
        )


def train_job(meta, out, job, train, raw, val, device="cuda"):
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    bind = binding(out, job)
    if (folder / "completion.json").exists():
        done = read_json(folder / "completion.json")
        if done["binding"] != bind:
            raise ValueError("Completed worker identity changed")
        base.old.verify_files(folder, done["files"])
        return done
    e, q = data.load_source(meta, job["seed"], device)
    initial_sig = base.signature(e, q)
    e.requires_grad_(True)
    q.requires_grad_(True)
    opt = optimizer(e, q)
    path = folder / "last.pt"
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["binding"] != bind or ck["initial_signature"] != initial_sig:
            raise ValueError("Resume source/binding changed")
        pending = (
            read_json(folder / "pending_epoch.json")
            if (folder / "pending_epoch.json").exists()
            else None
        )
        receipt = (
            read_json(folder / "checkpoint.json") if (folder / "checkpoint.json").exists() else None
        )
        if (receipt is None or receipt["sha256"] != sha256(path)) and not (
            (pending and pending["epoch"] == ck["epoch"]) or (receipt is None and ck["epoch"] == 0)
        ):
            raise ValueError("Checkpoint fingerprint differs without atomic commit evidence")
        base.lr.pending_epoch(folder, ck["epoch"])
        history_check(meta, job, ck, len(train))
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
        if base.signature(e, q) != ck["current_signature"]:
            raise ValueError("Restored model signature changed")
        opt.load_state_dict(ck["optimizer"])
        base.lr.optimizer_steps(opt, ck["epoch"] * math.ceil(len(train) / meta["batch"]))
        for g in opt.param_groups:
            if (g["lr"], g["betas"], g["eps"], g["weight_decay"]) != (
                RATE,
                (0.9, 0.999),
                1e-8,
                0.01,
            ):
                raise ValueError("Optimizer settings changed")
        if any(
            not torch.isfinite(v).all()
            for state in opt.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ):
            raise ValueError("Nonfinite restored optimizer")
        restore_rng(ck["rng"])
    else:
        if (folder / "pending_epoch.json").exists():
            raise ValueError("Uncommitted training attempt; export only")
        qualify(meta, out, job, e, q, val, True, device)
        initial = validation(meta, e, q, val, device)
        ck = {
            "binding": bind,
            "epoch": 0,
            "history": [],
            "initial_signature": initial_sig,
            "initial_validation": initial,
        }
        ck.update(
            encoder=base.cpu_state(e),
            query=base.cpu_state(q),
            optimizer=opt.state_dict(),
            rng=rng_state(),
            current_signature=base.signature(e, q),
        )
        atomic_save(ck, path)
    atomic_json({"epoch": ck["epoch"], "sha256": sha256(path)}, folder / "checkpoint.json")
    for epoch in range(ck["epoch"] + 1, EPOCHS + 1):
        pending = {"epoch": epoch, "observed_steps": 0, "attempt_reserved_before_training": True}
        atomic_json(pending, folder / "pending_epoch.json")

        def observed(steps, pending=pending):
            pending["observed_steps"] = steps
            atomic_json(pending, folder / "pending_epoch.json")

        tick = time.monotonic()
        trained = train_epoch(meta, job, e, q, train, raw, opt, epoch, device, observed)
        values = validation(meta, e, q, val, device) if epoch % 5 == 0 or epoch == EPOCHS else None
        if values and any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite validation")
        ck["history"].append(
            {
                "epoch": epoch,
                "lr": RATE,
                "train": trained,
                "validation": values,
                "epoch_seconds": time.monotonic() - tick,
            }
        )
        ck.update(
            epoch=epoch,
            encoder=base.cpu_state(e),
            query=base.cpu_state(q),
            optimizer=opt.state_dict(),
            rng=rng_state(),
            current_signature=base.signature(e, q),
        )
        base.lr.optimizer_steps(opt, epoch * math.ceil(len(train) / meta["batch"]))
        if any(
            not torch.isfinite(v).all() for model in (e, q) for v in model.state_dict().values()
        ):
            raise ValueError("Nonfinite trained parameters")
        atomic_save(ck, path)
        atomic_json({"epoch": epoch, "sha256": sha256(path)}, folder / "checkpoint.json")
        atomic_json(ck["history"], folder / "history.json")
        base.lr.pending_epoch(folder, epoch)
        progress(
            f"{job['name']}: {epoch}/{EPOCHS}, Q={trained['components']['query']:.6f}, H={trained['components']['content']:.6f}, G={trained['components']['alignment']:.6f}, {trained['seconds']:.1f}s train; fixed-last, no best selection"
        )
    history_check(meta, job, ck, len(train))
    qualify(meta, out, job, e, q, val, False, device)
    final = validation(meta, e, q, val, device)
    r0.save_checks(folder, "last_replay", r0.nested_compare(final, ck["history"][-1]["validation"]))
    done = {
        "binding": bind,
        "epochs": EPOCHS,
        "updates": EPOCHS * math.ceil(len(train) / meta["batch"]),
        "initial_signature": initial_sig,
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


def lock_models(meta, out):
    records = {}
    for job in jobs():
        folder = out / job["name"]
        done = read_json(folder / "completion.json")
        if (
            done["binding"] != binding(out, job)
            or done["epochs"] != EPOCHS
            or done["updates"] != EPOCHS * math.ceil(data.COUNTS["train"] / meta["batch"])
        ):
            raise ValueError("Incomplete matched update budget")
        base.old.verify_files(folder, done["files"])
        records[job["name"]] = done
    for seed in core.SEEDS:
        names = [f"{arm}_s{seed}" for arm in core.ARMS]
        if any(
            records[n]["initial_signature"] != records[names[0]]["initial_signature"] for n in names
        ):
            raise ValueError("Different initial model weights")
        histories = [read_json(out / n / "history.json") for n in names]
        for i in range(EPOCHS):
            for key in (
                "order_hash",
                "pairs_hash",
                "states",
                "steps",
                "position_counts",
                "shift_counts",
            ):
                if any(h[i]["train"][key] != histories[0][i]["train"][key] for h in histories):
                    raise ValueError("Unmatched training exposure")
    lock = {
        "manifest": sha256(out / "manifest.json"),
        "records": records,
        "files": {f"{j['name']}/last.pt": sha256(out / j["name"] / "last.pt") for j in jobs()},
    }
    path = out / "model_lock.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("Model selection lock changed")
    atomic_json(lock, path)
    return lock


def gradient_audit(e, q, x, raw, ps, stats):
    """Check the actual loaded encoder AND reader, without any optimizer step."""
    parameters = list(e.parameters()) + list(q.parameters())
    split = len(list(e.parameters()))
    values = core.terms(e, q, x, raw, ps, stats, "aligned")
    gradients = {
        k: torch.autograd.grad(values[k].mean(), parameters, retain_graph=k != "objective")
        for k in ("query", "content", "alignment", "objective")
    }
    norms = {}
    for k, gs in gradients.items():
        norms[k] = {
            name: float(sum(g.detach().double().square().sum() for g in part).sqrt())
            for name, part in (("encoder", gs[:split]), ("query", gs[split:]))
        }
    worst = 0.0
    for total, qg, hg, gg in zip(
        gradients["objective"],
        gradients["query"],
        gradients["content"],
        gradients["alignment"],
        strict=True,
    ):
        expected = qg + core.CONTENT_WEIGHT * hg + core.ALIGN_WEIGHT * gg
        worst = max(worst, float((total - expected).abs().max()))
        torch.testing.assert_close(total, expected, atol=2e-6, rtol=2e-4)
    if any(not math.isfinite(v) or v <= 0 for pair in norms.values() for v in pair.values()):
        raise ValueError("Individual objective gradient does not reach encoder/query")
    return {"norms": norms, "decomposition_max_abs": worst, "passed": True, "updates": 0}


def preflight(meta, out, x, raw, val, remaining, device="cuda"):
    e, q = data.load_source(meta, 42, device)
    e.requires_grad_(True)
    q.requires_grad_(True)
    b = torch.as_tensor(x[: meta["micro"]], device=device)
    cl = torch.as_tensor(raw[: len(b)], device=device)
    ps = torch.as_tensor(core.schedule(len(b), 42, 1)[1], device=device)
    parameters = list(e.parameters()) + list(q.parameters())
    param_count = sum(p.numel() for p in parameters)
    before = base.signature(e, q)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    gradients = gradient_audit(e, q, b, cl, ps, meta["statistics"])
    atomic_json(gradients, out / "gradient_audit.json")
    measures = []
    for _ in range(3):
        e.zero_grad(set_to_none=True)
        q.zero_grad(set_to_none=True)
        base.sync(device)
        tick = time.monotonic()
        core.terms(e, q, b, cl, ps, meta["statistics"], "aligned")["objective"].mean().backward()
        base.sync(device)
        measures.append(time.monotonic() - tick)
    # This is gradient reachability, not an estimate of loss/capacity percentages.
    norms = {
        name: float(
            sum(
                p.grad.detach().double().square().sum()
                for p in model.parameters()
                if p.grad is not None
            ).sqrt()
        )
        for name, model in (("encoder", e), ("query", q))
    }
    if any(not math.isfinite(v) or v <= 0 for v in norms.values()) or before != base.signature(
        e, q
    ):
        raise ValueError("Preflight failed gradient reachability or changed source weights")
    base.sync(device)
    tick = time.monotonic()
    validation(meta, e, q, val[: meta["micro"]], device)
    base.sync(device)
    val_seconds = time.monotonic() - tick
    completed = sum(
        read_json(out / j["name"] / "checkpoint.json")["epoch"]
        if (out / j["name"] / "checkpoint.json").exists()
        else 0
        for j in jobs()
    )
    train_seconds = max(measures[1:])
    remaining_epochs = 6 * EPOCHS - completed
    estimate = remaining_epochs * (
        math.ceil(len(x) / len(b)) * train_seconds + math.ceil(len(val) / len(b)) * val_seconds / 5
    )
    estimate += 8 * 2 * math.ceil(2401 / len(b)) * val_seconds + 1800
    # Conservative serialized-GPU estimate: jobs2 does NOT imply a2x speedup.
    estimate *= 1.3
    peak = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0
    free, total = torch.cuda.mem_get_info() if str(device).startswith("cuda") else (2**60, 2**60)
    reserved = torch.cuda.memory_reserved() if str(device).startswith("cuda") else 0
    gpu_need = meta["jobs"] * (peak + param_count * 12 + 512 * 2**20)
    total_disk = (6 + meta["jobs"]) * param_count * 12 * 1.15 + 2 * 2**30
    owned = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    disk_need = max(512 * 2**20, total_disk - owned)
    report = {
        "train_micro_seconds": train_seconds,
        "val4_micro_seconds": val_seconds,
        "zero_update_measurements": 3,
        "gradient_decomposition": gradients,
        "parameters": param_count,
        "gradient_norms": norms,
        "unchanged_source_signature": before,
        "peak_bytes": peak,
        "estimated_concurrent_gpu_bytes": gpu_need,
        "gpu_total": total,
        "gpu_available": free + reserved,
        "disk_need": disk_need,
        "disk_free": shutil.disk_usage(out).free,
        "estimated_remaining_seconds": estimate,
        "available_seconds": remaining,
        "jobs": meta["jobs"],
        "estimate_is_not_promise": True,
    }
    report["passed"] = (
        gpu_need <= free + reserved and disk_need <= report["disk_free"] and estimate <= remaining
    )
    atomic_json(report, out / "preflight.json")
    del e, q, parameters, b, cl
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if not report["passed"]:
        raise ValueError(
            "Measured time/GPU/disk insufficient; see preflight.json. No budget/micro changes or source deletion."
        )
    progress(
        f"F03 measured remaining estimate~{estimate / 3600:.1f}h (conservative, not jobs2/2); disk~{disk_need / 2**30:.1f}GiB"
    )


def run_workers(out, concurrency):
    pending = list(jobs())
    active = []
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                job = pending.pop(0)
                if (out / job["name"] / "completion.json").exists():
                    continue
                p = subprocess.Popen(
                    [sys.executable, "-m", MODULE, "--out", str(out), "--job", job["name"]]
                )
                active.append((job, p))
            for job, p in list(active):
                code = p.poll()
                if code is None:
                    continue
                active.remove((job, p))
                if code:
                    raise RuntimeError(
                        f"Worker {job['name']} failed with exit{code}; other workers stopped; export all journals"
                    )
            time.sleep(0.25)
    finally:
        for _, p in active:
            if p.poll() is None:
                p.terminate()
        for _, p in active:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def execute(root, out, concurrency):
    from . import content_retention_evaluate as evaluate

    state = {"schema": SCHEMA, "task_ids": ["F03", "F11"], "status": "running", "promoted": False}
    atomic_json(state, out / "status.json")
    try:
        require_session(out)
        if not torch.cuda.is_available():
            raise ValueError("Real-model execution requires AutoDL CUDA")
        progress("F03/F11: checking reviewed overlap diagnostic and fixed D_last120 sources")
        info = data.verify(root, out)
        meta = {
            **info,
            "schema": SCHEMA,
            "source": str(root),
            "implementation": implementation(),
            "protocol": protocol(),
            "batch": BATCH,
            "micro": MICRO,
            "jobs": concurrency,
        }
        path = out / "manifest.json"
        if path.exists() and read_json(path) != meta:
            raise ValueError("Run source/protocol/implementation/resources changed")
        atomic_json(meta, path)
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            base.old.verify_files(out, done["files"])
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        train, rows = data.cache(meta, "train")
        val, vrows = data.cache(meta, "val")
        atomic_json(data.source.row_audit(rows, vrows), out / "partition_audit.json")
        with r0.replay_runtime("context") as runtime:
            atomic_json(
                {
                    "runtime": runtime,
                    "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(),
                },
                out / "runtime.json",
            )
            data.prepare(meta, out, ("train", "val"))
            raw = data.closes(out, "train")
            if not (out / "model_lock.json").exists():
                remaining = read_json(out / "budget.json")["deadline_unix"] - time.time()
                preflight(meta, out, train, raw, val, remaining)
                del train, raw, val
                gc.collect()
                run_workers(out, concurrency)
            lock_models(meta, out)
            data.prepare(meta, out, ("test",))
            data.prepare(meta, out, ("cross_research",))
            evaluate.evaluate(meta, out, "cuda")
        if data.verify(root, out) != info or implementation() != meta["implementation"]:
            raise ValueError("Source or code changed during execution")
        state.update(
            status="f03_content_retention_complete_requires_review",
            optimizer_updates=6 * EPOCHS * 38,
            readout_fits=0,
            source_unchanged=True,
        )
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".npz", ".pt")
            and p.name not in ("status.json", "request.json", "budget.json")
            and p != out / "completion.json"
        }
        atomic_json({**state, "files": files}, out / "completion.json")
        atomic_json(state, out / "status.json")
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(state, out / "status.json")
        raise


def require_session(out):
    budget = read_json(out / "budget.json")
    if budget["deadline_unix"] <= time.time() or budget["used_seconds"] != SESSION_CAP:
        raise ValueError("No active bounded supervisor session")
    if budget.get("supervisor_pid", 0) <= 0:
        raise ValueError("Missing supervisor identity")
    os.kill(budget["supervisor_pid"], 0)


def supervise(root, out, concurrency):
    root = root.resolve()
    out = data.source.source.interface.separate(out, root)
    out.mkdir(parents=True, exist_ok=True)
    request = {"schema": SCHEMA, "source": str(root), "output": str(out), "jobs": concurrency}
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ((out / "request.json").exists() and read_json(out / "request.json") != request) or (
            not (out / "request.json").exists()
            and any(p.name != "session.lock" for p in out.iterdir())
        ):
            raise ValueError("Existing output not bound to this request")
        atomic_json(request, out / "request.json")
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            base.old.verify_files(out, done["files"])
            progress("Completed fixed run verified; export only, no new execution")
            return 0
        budget = (
            read_json(out / "budget.json")
            if (out / "budget.json").exists()
            else {"used_seconds": 0.0, "attempts": []}
        )
        used = budget["used_seconds"]
        if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
            raise ValueError("Cumulative24h budget exhausted; export only")
        remaining = SESSION_CAP - used
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget.update(
            used_seconds=SESSION_CAP,
            deadline_unix=time.time() + remaining,
            supervisor_pid=os.getpid(),
        )
        atomic_json(budget, out / "budget.json")
        started = time.monotonic()
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                MODULE,
                "--source",
                str(root),
                "--out",
                str(out),
                "--jobs",
                str(concurrency),
                "--coordinator",
            ],
            start_new_session=True,
        )

        def stop(signum, frame):
            raise InterruptedError(f"signal{signum}")

        previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            code = child.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
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
        attempt.update(elapsed_seconds=elapsed, exit_code=code)
        atomic_json(budget, out / "budget.json")
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    p.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--job", choices=[j["name"] for j in jobs()], help=argparse.SUPPRESS)
    a = p.parse_args()
    out = a.out.resolve()
    if a.job:
        require_session(out)
        if not torch.cuda.is_available():
            raise ValueError("Training requires AutoDL CUDA")
        meta = read_json(out / "manifest.json")
        if meta["protocol"] != protocol() or meta["implementation"] != implementation():
            raise ValueError("Worker protocol/code changed")
        job = next(j for j in jobs() if j["name"] == a.job)
        folder = out / a.job
        folder.mkdir(exist_ok=True)
        with (folder / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with r0.replay_runtime("context"):
                train = data.cache(meta, "train")[0]
                val = data.cache(meta, "val")[0]
                train_job(meta, out, job, train, data.closes(out, "train"), val)
        return
    if a.source is None:
        p.error("--source is required")
    if a.coordinator:
        raise SystemExit(execute(a.source.resolve(), out, a.jobs))
    raise SystemExit(supervise(a.source, out, a.jobs))


if __name__ == "__main__":
    main()
