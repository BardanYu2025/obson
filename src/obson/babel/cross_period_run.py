"""Four fixed-depth continuation arms with equal cross-period triplet exposure."""

import argparse
import fcntl
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import capacity_growth as growth
from . import capacity_growth_evaluate as ge
from . import capacity_growth_run as gr
from . import consistency_readout as rr
from . import cross_period_consistency as xp
from . import cross_period_data as xd
from . import local_readapt_run as lr
from .ae_extend import atomic_json, restore_rng, rng_state
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

cr = rr.cr
ur = rr.ur
ce = rr.ce
SCHEMA = "babel-cross-period768-v1"


def code_identity():
    from . import cross_period_evaluate as ev

    return (
        lr.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (xp, xd, xd.raw_audit, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


def source_identity(source):
    return lr.source_identity(source)


def parent_meta(meta):
    return gr.parent_meta(meta["identity"]["manifest"])


def parent_root(meta):
    return gr.parent_root(meta["identity"]["manifest"])


def statistics(meta):
    return gr.statistics(meta["identity"]["manifest"])


def train_plan(meta):
    return read_json(Path(meta["source"]) / "train_plan.json")


def make_manifest(source, identity, raw_root, audit, audit_id, micro=16):
    if micro not in (8, 16, 32, 64):
        raise ValueError("Microbatch must be8/16/32/64 triplets")
    gm = identity["manifest"]
    if gm["state_width"] != 768 or gm["epochs"] != 100:
        raise ValueError("Completed growth768 required")
    for seed in (42, 43):
        job = next(j for j in gm["experiments"] if j["name"] == f"deep_s{seed}")
        if (job["layers"], job["ff"]) != (4, 3072):
            raise ValueError("Fixed4-layer FFN3072 required")
        if read_json(source / job["name"] / "training_summary.json")["selected_epoch"] != 100:
            raise ValueError("Expected locked deep best500")
    return {
        "schema": SCHEMA,
        "source": str(source),
        "identity": identity,
        "code_sha256": code_identity(),
        "raw_root": str(raw_root),
        "raw_audit": str(audit),
        "raw_audit_identity": audit_id,
        "epochs": 100,
        "micro": micro,
        "batch": 128,
        "budget": 4789,
        "evaluation_batch": 64,
        "state_width": 768,
        "encoder_lr": 1e-5,
        "head_lr": 3e-5,
        "warmup": 5,
        "experiments": [
            {
                "name": f"{mode}_s{s}",
                "variant": mode,
                "seed": s,
                "layers": 4,
                "ff": 3072,
                "weight": w,
            }
            for s in (42, 43)
            for mode, w in [("control", 0.0), ("cross", 0.10)]
        ],
        "initialization": "Both same-seed arms use growth deep validation best500, including ORIGINAL local head. All model tensors replay; BOTH AdamW reset with identical5-epoch warmup and cosine. Fixed PCA decoder remains frozen.",
        "sampling": "4789 triplets/epoch; original eligible15m dense fine anchor B, earlier fine A shift1/16, same-closed coarse C30/60. Both arms identical triplets, four nonheld prefixes and38 updates. Seeded shuffled cycles without replacement within each cycle; if pool<budget repeats are explicitly counted.30->60 held from auxiliary training.",
        "objective": "Mean original global+0.25local losses on all3 views +0.10 same-period A/B consistency. Candidate alone adds0.10 common historical close-change SmoothL1 B/C, scaled by original train delta1 scale. Both arms compute identical graphs. No complete-state equality or truth-anchor correction.",
        "pairs": "Raw strict OHLCV/OI aggregation; same contract/session and exact close timestamp; both main eligible and>=512 bars within original partition; same-period shifted view also retains512 partition bars. At least8 adjacent fully-covered historical coarse intervals; both current bars excluded. No normalized activity/EMA aggregation.",
        "evaluation_cohort": "Original capability cohorts unchanged. NEW cross-period metrics use fixed stride16 main-eligible512-warmup endpoints inside the original test packed partitions, same for parent/control/candidate. All three pair groups require50 pairs/5 weeks before any training; no model-score-dependent filtering.",
        "selection": "Original full validation global MSE+0.25local only; epoch0 eligible. All4 fixed budgets and8 best/last selections locked before any research model scoring. Research raw pair geometry may be checked beforehand but never selects weights.",
        "readouts": "8 checkpoints x13 targets x5 original Ridge alphas=520 candidates; train4789/val978; all104 heads lock before research extraction. Frozen deep500 paired heads and exact raw/PCA/current controls retained.",
        "decision": {
            "gap_gain": 0.10,
            "retention": 0.05,
            "family_retention": 0.10,
            "current_r2": 0.8,
            "current_nmse": 0.1,
            "rule": "Both seeds/research sets/best AND last: >=10% cross-period gap reduction vs same-budget control on each15->30 and15->60 group, weekly upper CI<=0; both true-history branch errors retain5% vs control AND frozen500. Hold30->60 gap/true errors within5%. Original global/local/overlap/utility/current protection checked; no automatic deployment.",
        },
        "stop": "One100-epoch fixed-weight trial; no architecture/LR/weight sweep or automatic extension. Same data exposure and updates, not convergence proof. Old full-utility/capacity failures remain; reused research sets are not independent final holdout.",
    }


def parent_job(meta, job):
    return next(
        j
        for j in meta["identity"]["manifest"]["experiments"]
        if j["name"] == f"deep_s{job['seed']}"
    )


def construct(meta, job, device, optimizers=False):
    model, expected = ge.load(
        meta["identity"]["manifest"], Path(meta["source"]), parent_job(meta, job), "best", device
    )
    model.core.encoder.requires_grad_(True)
    model.local_head.requires_grad_(True)
    model.core.decoder.requires_grad_(False)
    if not optimizers:
        return model
    enc = torch.optim.AdamW(
        model.core.encoder.parameters(), lr=meta["encoder_lr"], weight_decay=0.01
    )
    head = torch.optim.AdamW(model.local_head.parameters(), lr=meta["head_lr"], weight_decay=1e-4)
    info = {
        "width": 768,
        "layers": 4,
        "heads": 8,
        "ff": 3072,
        "encoder_parameters": sum(p.numel() for p in model.core.encoder.parameters()),
        "local_parameters": sum(p.numel() for p in model.local_head.parameters()),
        "decoder_trainable_parameters": 0,
    }
    return model, enc, head, expected, info


def validation(meta, job, model, device):
    return gr.validation(meta["identity"]["manifest"], parent_job(meta, job), model, device)


def plan(meta, job, epoch, n):
    if n < 1:
        raise ValueError("Empty paired pool")
    rng = np.random.default_rng(np.random.SeedSequence([job["seed"], 500 + epoch, 20260926]))
    ids = np.concatenate([rng.permutation(n) for _ in range((meta["budget"] + n - 1) // n)])[
        : meta["budget"]
    ]
    ds = np.array([1, 16])[rng.permutation(np.arange(len(ids)) % 2)]
    ps = ur.bb.ba.position_plan(len(ids), job["seed"], 500 + epoch)
    return (
        ids,
        ds,
        ps,
        {
            "absolute_epoch": 500 + epoch,
            "ids": ur.bb.cov.ndarray_hash(ids),
            "shifts": ur.bb.cov.ndarray_hash(ds),
            "prefixes": ur.bb.cov.ndarray_hash(ps),
            "unique_triplets": len(np.unique(ids)),
        },
    )


def verify_history(meta, job, state, n):
    if [v["epoch"] for v in state["history"]] != list(range(1, state["epoch"] + 1)) or state[
        "epoch"
    ] > meta["epochs"]:
        raise ValueError("Incomplete epoch history")
    steps = (meta["budget"] + meta["batch"] - 1) // meta["batch"]
    for v in state["history"]:
        if (
            v["sampling"] != plan(meta, job, v["epoch"], n)[3]
            or v["pairs"] != meta["budget"]
            or v["views"] != 3 * meta["budget"]
            or v["encoder_steps"] != steps
            or v["head_steps"] != steps
        ):
            raise ValueError("Exposure/position schedule changed")
        for key in ("encoder_lr", "head_lr"):
            if not np.isclose(
                v[key],
                growth.learning_rate(v["epoch"], meta["epochs"], meta[key], meta["warmup"]),
                rtol=1e-14,
                atol=0,
            ):
                raise ValueError("Learning rate changed")
    best = min(
        [(0, state["initial_validation"])]
        + [(v["epoch"], v["validation"]) for v in state["history"]],
        key=lambda v: v[1]["selection"],
    )
    if best != (state["best_epoch"], state["best_validation"]):
        raise ValueError("Validation selection changed")


@torch.no_grad()
def preflight(meta, out, device):
    data = ur.bb.load_data(cr.alignment(parent_meta(meta)), "val")
    x = torch.tensor(np.asarray(data["x"][:4]), device=device)
    ps = torch.tensor([[32, 64, 96, 128]] * len(x), device=device)
    report = {}
    for seed in (42, 43):
        job = next(j for j in meta["experiments"] if j["seed"] == seed)
        parent, _ = ge.load(
            meta["identity"]["manifest"],
            Path(meta["source"]),
            parent_job(meta, job),
            "best",
            device,
        )
        model, enc, head, expected, info = construct(meta, job, device, True)
        model.eval()
        z = parent.core.encoder(x)
        g, local_prediction = parent(x, ps, True)
        gg, ll = model(x, ps, True)
        checks = {
            "state": growth.compare(model.core.encoder(x), z),
            "global_output": growth.compare(gg, g),
            "local_output": growth.compare(ll, local_prediction),
        }
        if not all(v["passed"] for v in checks.values()):
            raise ValueError("Original deep/local-head initialization changed")
        actual = validation(meta, job, model, device)
        ur.pf.ea.require_replay(actual, expected, "parent500 replay")
        causal = ur.pf.er.trained_causality(model, x)
        if causal["status"] == "failed":
            raise ValueError("Noncausal initialization")
        report[str(seed)] = {
            "initial_function": checks,
            "validation": actual,
            "causality": causal,
            "structure": info,
            "both_optimizers_reset": not enc.state and not head.state,
        }
        atomic_json(report, out / "preflight.json")
        del parent, model, enc, head


def worker(out, name, device="cuda"):
    meta = read_json(out / "manifest.json")
    if meta["schema"] != SCHEMA or meta["code_sha256"] != code_identity():
        raise ValueError("Worker code/configuration changed")
    xd.verify_data(meta, out)
    job = next(j for j in meta["experiments"] if j["name"] == name)
    path = out / name
    path.mkdir(exist_ok=True)
    metadata = {"manifest": meta, "job": job}
    if (path / "completion.json").exists():
        done = read_json(path / "completion.json")
        if done["status"] != "complete" or done["metadata"] != metadata:
            raise ValueError("Worker metadata changed")
        verify_files(path, done["files"])
        return
    model, enc, head, expected, info = construct(meta, job, device, True)
    builder = xd.Builder(meta, out)
    stats, local = statistics(meta)
    if (path / "last.pt").exists():
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != metadata:
            raise ValueError("Resume changed")
        verify_history(meta, job, state, len(builder.plan))
        model.load_state_dict(state["model"])
        enc.load_state_dict(state["encoder_optimizer"])
        head.load_state_dict(state["head_optimizer"])
        restore_rng(state["rng"])
        expected = (
            state["history"][-1]["validation"] if state["history"] else state["initial_validation"]
        )
        ur.pf.ea.require_replay(
            validation(meta, job, model, device), expected, "cross-period resume"
        )
    else:
        initial = validation(meta, job, model, device)
        ur.pf.ea.require_replay(initial, expected, "original deep500 replay")
        torch.manual_seed(job["seed"] + 20261001)
        state = {
            "metadata": metadata,
            "epoch": 0,
            "history": [],
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_model": ur.bb.ab.cpu_state(model),
            "structure": info,
        }
        atomic_json(
            {
                "validation": initial,
                "parent_epoch": 500,
                "optimizer_reset": True,
                "structure": info,
            },
            path / "initial_validation.json",
        )
    state.update(
        model=ur.bb.ab.cpu_state(model),
        encoder_optimizer=enc.state_dict(),
        head_optimizer=head.state_dict(),
        rng=rng_state(),
    )
    ur.pf.publish(state, path)
    decoder_signature = {
        k: v.detach().cpu().clone() for k, v in model.core.decoder.state_dict().items()
    }
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(state["epoch"] + 1, meta["epochs"] + 1):
        started = time.monotonic()
        ids, ds, ps, sampling = plan(meta, job, epoch, len(builder.plan))
        rates = {
            k: growth.learning_rate(epoch, meta["epochs"], meta[k], meta["warmup"])
            for k in ("encoder_lr", "head_lr")
        }
        for g in enc.param_groups:
            g["lr"] = rates["encoder_lr"]
        for g in head.param_groups:
            g["lr"] = rates["head_lr"]
        train, steps = xp.run_epoch(
            model,
            builder,
            ids,
            ds,
            ps,
            stats,
            local,
            meta["batch"],
            meta["micro"],
            device,
            job["weight"],
            enc,
            head,
        )
        result = validation(meta, job, model, device)
        if result["selection"] < state["best_validation"]["selection"]:
            state.update(
                best_epoch=epoch, best_validation=result, best_model=ur.bb.ab.cpu_state(model)
            )
        state["history"].append(
            dict(
                epoch=epoch,
                sampling=sampling,
                train=train,
                validation=result,
                **rates,
                pairs=len(ids),
                views=3 * len(ids),
                encoder_steps=steps,
                head_steps=steps,
                seconds=time.monotonic() - started,
            )
        )
        state.update(
            epoch=epoch,
            model=ur.bb.ab.cpu_state(model),
            encoder_optimizer=enc.state_dict(),
            head_optimizer=head.state_dict(),
            rng=rng_state(),
        )
        ur.pf.publish(state, path)
        progress(
            f"{name}: epoch={epoch}/{meta['epochs']}, val={result['selection']:.6f}, cross_gap={train['cross_period']:.6f}, best={state['best_epoch']}, seconds={state['history'][-1]['seconds']:.1f}"
        )
    verify_history(meta, job, state, len(builder.plan))
    if any(
        not torch.equal(v, model.core.decoder.state_dict()[k].cpu())
        for k, v in decoder_signature.items()
    ):
        raise ValueError("Fixed decoder changed")
    summary = dict(
        selected_epoch=state["best_epoch"],
        epochs=meta["epochs"],
        validation=state["best_validation"],
        structure=info,
        **{
            k: sum(v[k] for v in state["history"])
            for k in ("pairs", "views", "encoder_steps", "head_steps")
        },
        training_seconds=sum(v["seconds"] for v in state["history"]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
        decoder_unchanged=True,
    )
    tail = state["history"][-min(20, len(state["history"])) :]
    summary["validation_tail"] = {
        "epochs": [v["epoch"] for v in tail],
        "selection": [v["validation"]["selection"] for v in tail],
        "slope_per_epoch": float(
            np.polyfit([v["epoch"] for v in tail], [v["validation"]["selection"] for v in tail], 1)[
                0
            ]
        ),
    }
    atomic_json(summary, path / "training_summary.json")
    files = {
        p.name: sha256(p)
        for p in path.iterdir()
        if p.is_file()
        and p.name not in ("completion.json", "progress.json", "run.log")
        and not p.name.endswith(".tmp")
    }
    atomic_json(
        {"status": "complete", "metadata": metadata, "files": files}, path / "completion.json"
    )


def lock_selection(meta, out):
    trials = {}
    weights = {}
    n = read_json(out / "train_cross_plan.json")["eligible"]
    for job in meta["experiments"]:
        root = out / job["name"]
        done = read_json(root / "completion.json")
        if done["status"] != "complete" or done["metadata"] != {"manifest": meta, "job": job}:
            raise ValueError("All four complete budgets required")
        verify_files(root, done["files"])
        last = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        verify_history(meta, job, last, n)
        summary = read_json(root / "training_summary.json")
        if (
            last["metadata"] != done["metadata"]
            or best["metadata"] != done["metadata"]
            or last["epoch"] != meta["epochs"]
            or best["epoch"] != last["best_epoch"]
            or best["validation"] != last["best_validation"]
            or read_json(root / "history.json") != last["history"]
        ):
            raise ValueError("Checkpoint/selection mismatch")
        if (
            summary["selected_epoch"] != best["epoch"]
            or summary["validation"] != best["validation"]
            or not summary["decoder_unchanged"]
        ):
            raise ValueError("Summary mismatch")
        if any(not torch.equal(v, best["model"][k]) for k, v in last["best_model"].items()):
            raise ValueError("Best model differs from selected state")
        for k in ("pairs", "views", "encoder_steps", "head_steps"):
            if summary[k] != sum(v[k] for v in last["history"]):
                raise ValueError("Budget mismatch")
        trials[job["name"]] = summary
        weights[job["name"]] = {k: sha256(root / f"{k}.pt") for k in ("best", "last")}
    result = {"manifest": meta, "trials": trials, "weights": weights}
    if (out / "model_selection_lock.json").exists() and read_json(
        out / "model_selection_lock.json"
    ) != result:
        raise ValueError("Model selection changed")
    atomic_json(result, out / "model_selection_lock.json")
    return result


def run_jobs(out, jobs):
    pending = list(read_json(out / "manifest.json")["experiments"])
    active = {}
    seen = {}
    try:
        while pending or active:
            while pending and len(active) < jobs:
                j = pending.pop(0)
                p = out / j["name"]
                p.mkdir(exist_ok=True)
                log = (p / "run.log").open("a")
                try:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "obson.babel.cross_period_run",
                            "worker",
                            "--out",
                            str(out),
                            "--name",
                            j["name"],
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                active[j["name"]] = (child, log)
                progress(f"Started {j['name']}, pid={child.pid}")
            for name, (p, log) in list(active.items()):
                progress_file = out / name / "progress.json"
                if progress_file.exists():
                    row = read_json(progress_file)
                    snapshot = (row["epoch"], row["best_epoch"])
                    if seen.get(name) != snapshot:
                        seen[name] = snapshot
                        progress(
                            f"{name}: epoch={row['epoch']}/{row['epochs']}, best={row['best_epoch']}, seconds={row.get('seconds')}"
                        )
                if p.poll() is not None:
                    log.close()
                    del active[name]
                    if p.returncode:
                        path = out / name / "run.log"
                        with path.open("rb") as failure_log:
                            failure_log.seek(0, 2)
                            failure_log.seek(max(0, failure_log.tell() - 65536))
                            tail = "\n".join(
                                failure_log.read()
                                .decode("utf-8", errors="replace")
                                .splitlines()[-80:]
                            )
                        raise RuntimeError(
                            f"{name} failed with exit{p.returncode}; worker log: {path}\n{tail}"
                        )
                    progress(f"Completed {name}")
            if active:
                time.sleep(1)
    finally:
        for p, log in active.values():
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            log.close()


def release_preflight_cuda(out, jobs, micro, device):
    """Release unused parent allocations before child CUDA contexts start."""
    if not str(device).startswith("cuda"):
        return
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report = {
            "gpu": torch.cuda.get_device_name(),
            "total_bytes": total,
            "free_bytes": free,
            "manager_allocated_bytes": torch.cuda.memory_allocated(),
            "manager_reserved_bytes": torch.cuda.memory_reserved(),
            "jobs": jobs,
            "micro_triplets": micro,
            "windows_per_forward": 3 * micro,
            "effective_batch": 128,
        }
    atomic_json(report, out / "gpu_launch.json")
    progress(
        f"GPU {report['gpu']}: total={total / 2**30:.2f} GiB, free={free / 2**30:.2f} GiB; "
        f"workers={jobs}, micro={micro} triplets ({3 * micro} windows), effective batch=128. "
        "Unused preflight cache released; available memory does not guarantee peak fit."
    )


def check_output(source, out, identity=None):
    lr.check_output(source, out, identity)


def run(source, out, raw_root, audit, jobs=1, micro=16, device="cuda"):
    from . import cross_period_evaluate as ev

    source, out, raw_root, audit = (p.resolve() for p in (source, out, raw_root, audit))
    check_output(source, out)
    for ref in (raw_root, audit):
        rr.check_output(ref, out)
    if jobs not in (1, 2):
        raise ValueError("One or two GPU workers supported")
    progress("Checking original deep500 checkpoints, source lineage and remote raw audit")
    identity = source_identity(source)
    check_output(source, out, identity)
    raw_id = xd.audit_identity(audit, raw_root)
    meta = make_manifest(source, identity, raw_root, audit, raw_id, micro)
    runtime = read_json(source / "runtime.json")
    if runtime["torch"] != str(torch.__version__) or runtime["numpy"] != np.__version__:
        raise ValueError("Restore source runtime")
    if (out / "manifest.json").exists():
        if read_json(out / "manifest.json") != meta:
            raise ValueError("Configuration/source changed; new output required")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Empty output required")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        ur.bb.ab.sc.verify_worker_files(out, done["files"])
        lock_selection(meta, out)
        ev.check_readouts(out)
        xd.verify_data(meta, out)
        if done["status"] != "complete":
            raise ValueError("Invalid completion")
        progress("Already complete; no retraining")
        return
    started = time.time()
    try:
        for split in rr.old.SPLITS:
            shutil.copyfile(source / f"{split}_inventory.json", out / f"{split}_inventory.json")
        atomic_json(train_plan(meta), out / "train_plan.json")
        atomic_json(
            rr.up.probe.inventory_audit(
                {s: read_json(out / f"{s}_inventory.json") for s in rr.old.SPLITS}
            ),
            out / "data_audit.json",
        )
        xd.prepare(meta, out)
        preflight(meta, out, device)
        release_preflight_cuda(out, jobs, micro, device)
        run_jobs(out, jobs)
        lock_selection(meta, out)
        for split in ("train", "val"):
            ev.prepare(meta, out, split, device)
        ev.fit(meta, out, device)
        for split in ("test", "cross_research"):
            ev.prepare(meta, out, split, device)
        ev.evaluate(meta, out, device)
        if source_identity(source) != identity or xd.audit_identity(audit, raw_root) != raw_id:
            raise ValueError("Frozen source changed")
    except Exception as exc:
        atomic_json(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, out / "failure.json"
        )
        raise
    atomic_json(
        {
            "seconds": time.time() - started,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "jobs": jobs,
            "device": str(device),
            "ridge_candidates": 520,
        },
        out / "runtime.json",
    )
    files = {
        str(p.relative_to(out)): sha256(p)
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".md", ".npz")
        and p.name not in ("completion.json", "failure.json", "progress.json")
    }
    atomic_json(
        {"status": "complete", "source_unchanged": True, "cells": 4, "files": files},
        out / "completion.json",
    )


def main():
    import signal

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("all", "worker"))
    p.add_argument("--source", type=Path, default=Path("checkpoints/babel_growth768"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/babel_cross_period768"))
    p.add_argument("--raw-root", type=Path, default=Path("/root/autodl-tmp/data/contracts"))
    p.add_argument("--audit", type=Path, default=Path("checkpoints/babel_cross_period_audit"))
    p.add_argument("--name")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--micro", type=int, default=16)
    a = p.parse_args()
    ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Formal training/inference/reader fitting only on AutoDL CUDA")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if a.action == "worker":
        worker(a.out, a.name)
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with (a.out.parent / f".{a.out.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Output already running") from None
        run(a.source, a.out, a.raw_root, a.audit, a.jobs, a.micro)


if __name__ == "__main__":
    main()
