"""Warm-start the original control600 local heads; frozen causal encoders."""

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

from . import cross_period_evaluate as xe
from . import cross_period_recheck as recheck
from . import cross_period_run as xr
from . import local_readapt as lr
from . import local_readapt_run as legacy
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-local-warmstart768-v1"
ur, ba, gr = legacy.ur, legacy.ba, legacy.gr
cache_positions = legacy.cache_positions
arrays, device_arrays, publish, plan = (
    legacy.arrays,
    legacy.device_arrays,
    legacy.publish,
    legacy.plan,
)


def code_identity():
    return xr.code_identity() | {
        Path(recheck.__file__).name: sha256(recheck.__file__),
        Path(__file__).name: sha256(__file__),
    }


def source_identity(source):
    report = read_json(source / "manifest.json")
    done = read_json(source / "completion.json")
    if (
        report["schema"] != "babel-cross-period-evaluation-recheck-v1"
        or report["evaluator_sha256"] != sha256(recheck.__file__)
        or done["status"] != "complete"
        or not done["source_unchanged"]
    ):
        raise ValueError("Completed unchanged cross-period reevaluation required")
    verify_files(source, done["files"])
    root = Path(report["source"])
    meta, files = recheck.verify_inputs(root)
    if report["original_manifest"] != meta or report["source_files"] != files:
        raise ValueError("Reevaluation/training lineage changed")
    locked = xe.check_models(root)
    for seed in (42, 43):
        job = next(j for j in meta["experiments"] if j["name"] == f"control_s{seed}")
        if job["weight"] != 0 or locked["trials"][job["name"]]["selected_epoch"] != 100:
            raise ValueError("Fixed control600 validation-best required")
    return {
        "report_manifest": report,
        "training_source": str(root),
        "training_manifest": meta,
        "training_files": files,
        "files": done["files"] | {"completion.json": sha256(source / "completion.json")},
    }


def training_meta(meta):
    return meta["identity"]["training_manifest"]


def training_root(meta):
    return Path(meta["identity"]["training_source"])


def statistics(meta):
    return xr.statistics(training_meta(meta))


def check_output(source, out, identity=None):
    xr.check_output(source, out)
    if identity is not None:
        tm = identity["training_manifest"]
        xr.check_output(Path(identity["training_source"]), out)
        xr.check_output(Path(tm["source"]), out, tm["identity"])
        for key in ("raw_root", "raw_audit"):
            xr.check_output(Path(tm[key]), out)


def make_manifest(source, identity):
    tm = identity["training_manifest"]
    if tm["state_width"] != 768 or tm["evaluation_batch"] != 64:
        raise ValueError("Expected completed control768 study with scoring batch64")
    return {
        "schema": SCHEMA,
        "source": str(source),
        "identity": identity,
        "code_sha256": code_identity(),
        "epochs": 100,
        "batch": 128,
        "extract_batch": 64,
        "evaluation_batch": 64,
        "hidden": 256,
        "lr": 3e-5,
        "width": 768,
        "experiments": [{"name": f"control_s{s}", "seed": s} for s in (42, 43)],
        "train_prefixes": list(ba.TRAIN_PREFIXES),
        "val_prefixes": list(ba.VAL_PREFIXES),
        "held_prefixes": list(ba.HELD_PREFIXES),
        "initialization": "Restore each original control600 local head exactly. Raw cached states, identity normalization, no fitted state statistics. AdamW reset; original function is epoch0 fallback.",
        "objective": "Only past16 local SmoothL1; same original targets, masks and scales. Encoder/global decoder and calibrated utility readouts frozen.",
        "budget": "4789 fixed train windows once per epoch,4 distinct nonheld positions each;100 epochs,3800 head updates,1915600 prefix exposures per seed. This coverage differs from parent triplet training; no causal comparison to old random-head study.",
        "optimizer": "AdamW peak3e-5,5-epoch warmup,cosine to0.1peak,weight_decay1e-4,clip_norm1; no LR/head grid.",
        "selection": "Original978 validation windows,32/64/96/128 local MSE only; epoch0 eligible. Both full budgets and best/last locked before research extraction.48/80/112 never train or select.",
        "decision": "Both seeds, both reused research sets, best AND last: held primary >=5% improvement over own original head with weekly upper CI<=0; trained primary retain5%, held path/body/activity/change1 retain10%, each evaluated prefix primary retain10%. No automatic model promotion.",
        "stop": "One warm-start diagnostic, no extension/grid. Failure cannot prove absent information. Original global/utility/cross-period conclusions unchanged.",
    }


def original_model(meta, job, device):
    tm = training_meta(meta)
    oldjob = next(j for j in tm["experiments"] if j["name"] == job["name"])
    return xe.load(tm, training_root(meta), oldjob, "best", device)


def identity_scales(width):
    return {"mean": [0.0] * width, "scale": [1.0] * width, "fitted_on": "none_identity_raw_states"}


def original_head(meta, out, job, device):
    verify_cache(meta, out, "train")
    info = read_json(out / f"{job['name']}_frozen_audit.json")
    filename = f"{job['name']}_original_head.pt"
    expected = verify_cache(meta, out, "train")["files"][filename]
    if sha256(out / "cache" / filename) != expected:
        raise ValueError("Original local head changed")
    head = lr.reader(meta["width"], meta["hidden"], job["seed"], device)
    head.load_state_dict(
        torch.load(out / "cache" / filename, map_location=device, weights_only=True)
    )
    if ur.bb.state_signature(head) != info["original_head_signature"]:
        raise ValueError("Warm-start function changed")
    return head


def verify_cache(meta, out, split):
    result = legacy.verify_cache(meta, out, split)
    if split == "train":
        audits = result["frozen_audits"]
        expected = {f"{j['name']}_frozen_audit.json" for j in meta["experiments"]}
        if set(audits) != expected:
            raise ValueError("Missing frozen model audits")
        verify_files(out, audits)
    return result


def release_cache_cuda(out, jobs, device):
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
            "jobs": jobs,
            "stage": "Only small local heads train; frozen state caches are read-only.",
        }
    atomic_json(report, out / "gpu_launch.json")
    progress(
        f"Frozen state caching complete; GPU free={free / 2**30:.2f}/{total / 2**30:.2f}GiB, head workers={jobs}"
    )


def decide(records, rows):
    checks = []
    for split in ("test", "cross_research"):
        if split not in records:
            raise ValueError("Both research sets required")
        bank = records[split]
        for seed in (42, 43):
            reference = f"control_s{seed}_original"
            for kind in ("best", "last"):
                name = f"control_s{seed}_{kind}"
                rules = [("held", "primary", 0.95), ("trained", "primary", 1.05)]
                rules += [("held", key, 1.10) for key in ("path", "body", "activity", "change1")]
                rules += [
                    (f"p{p}", "primary", 1.10) for p in sorted(ba.VAL_PREFIXES + ba.HELD_PREFIXES)
                ]
                for task, metric, factor in rules:
                    a = np.asarray(bank[name]["errors"][task][metric])
                    b = np.asarray(bank[reference]["errors"][task][metric])
                    if (
                        a.shape != b.shape
                        or a.shape != (len(rows[split]),)
                        or not np.isfinite(a).all()
                        or not np.isfinite(b).all()
                    ):
                        raise ValueError("Invalid paired local errors")
                    ci = ur.interval(a, factor * b, rows[split])
                    checks.append(
                        {
                            "dataset": split,
                            "seed": seed,
                            "checkpoint": kind,
                            "reference": reference,
                            "task": task,
                            "metric": metric,
                            "factor": factor,
                            "candidate_mean": float(a.mean()),
                            "reference_mean": float(b.mean()),
                            "interval": ci,
                            "passed": bool(
                                ci["supported"] and ci["high"] is not None and ci["high"] <= 0
                            ),
                        }
                    )
    passed = bool(checks) and all(v["passed"] for v in checks)
    return {
        "status": "warm_local_readout_gain" if passed else "no_warm_local_readout_gain",
        "checks": checks,
        "encoder_updates": 0,
        "automatic_promotion": False,
        "scope": "Own original local head is fixed epoch0 reference. Both seeds/sets and best+last required. Held-prefix gain tests bounded readout adaptability, not absent information, full utility, or future prediction.",
    }


@torch.no_grad()
def prepare(meta, out, split, device):
    if split in ("test", "cross_research"):
        check_selection(meta, out)
    positions = cache_positions(meta, split)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    if (cache / f"{split}_index.json").exists():
        verify_cache(meta, out, split)
        return
    tm = training_meta(meta)
    cm = xr.parent_meta(tm)
    data = ur.bb.load_data(xr.cr.alignment(cm), split)
    stats, local = statistics(meta)
    n = len(data["x"])
    batch = meta["extract_batch"]
    files = {}
    rows = read_json(out / f"{split}_inventory.json")
    if len(rows) != n or rows != read_json(training_root(meta) / f"{split}_inventory.json"):
        raise ValueError("Source rows changed")

    def mmap(name, shape, dtype):
        return np.lib.format.open_memmap(cache / name, mode="w+", dtype=dtype, shape=shape)

    y = mmap(f"{split}_y.npy", (n, len(positions), 16, 7), np.float32)
    mask = mmap(f"{split}_mask.npy", y.shape, bool)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        ps = torch.tensor([positions] * (end - start))
        target, valid = ba.local_targets(
            torch.tensor(np.asarray(data["y"][start:end])),
            torch.tensor(np.asarray(data["mask"][start:end])),
            ps,
            stats,
            local,
        )
        y[start:end] = target.numpy()
        mask[start:end] = valid.numpy()
    y.flush()
    mask.flush()
    del y, mask
    for suffix in ("y", "mask"):
        files[f"{split}_{suffix}.npy"] = sha256(cache / f"{split}_{suffix}.npy")
    source = training_root(meta)
    xr.rr.old.verify_cache(tm, source, split)
    endpoint = xr.rr.old.arrays(source, split)
    for job in meta["experiments"]:
        model, expected = original_model(meta, job, device)
        model.eval().requires_grad_(False)
        before = ur.bb.state_signature(model)
        if split == "train":
            original = next(j for j in tm["experiments"] if j["name"] == job["name"])
            actual = xr.validation(tm, original, model, device)
            ur.pf.ea.require_replay(actual, expected, "frozen control600 validation")
            val = ur.bb.load_data(gr.cr.alignment(cm), "val")
            causal = ur.pf.er.trained_causality(
                model, torch.tensor(np.asarray(val["x"][:4]), device=device)
            )
            if causal["status"] == "failed":
                raise ValueError("Frozen model noncausal")
            atomic_json(
                {
                    "validation": actual,
                    "causality": causal,
                    "original_head_signature": ur.bb.state_signature(model.local_head),
                    "frozen_model_signature": before,
                },
                out / f"{job['name']}_frozen_audit.json",
            )
            filename = f"{job['name']}_original_head.pt"
            atomic_save(ur.bb.ab.cpu_state(model.local_head), cache / filename)
            files[filename] = sha256(cache / filename)
        name = f"{split}_{job['name']}_x.npy"
        x = mmap(name, (n, len(positions), meta["width"]), np.float32)
        for start in range(0, n, batch):
            z = model.core.encoder(
                torch.tensor(np.asarray(data["x"][start : start + batch]), device=device)
            )
            reference = torch.tensor(
                endpoint[job["name"] + "_best"][start : start + batch], device=device
            )
            if not gr.growth.compare(z[:, -1], reference)["passed"]:
                raise ValueError("Endpoint state differs from control600 cache")
            x[start : start + batch] = z[:, np.array(positions) - 1].cpu().numpy()
            if start and start % (batch * 8) == 0:
                progress(f"Caching {split}/{job['name']}: {min(start + batch, n)}/{n}")
        if not np.isfinite(x).all() or before != ur.bb.state_signature(model):
            raise ValueError("Nonfinite cache or mutated frozen model")
        x.flush()
        del x
        files[name] = sha256(cache / name)
        del model
        progress(f"Cached {split}/{job['name']}, {n} windows x{len(positions)} positions")
    index = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "positions": positions,
        "windows": n,
        "files": files,
    }
    if split == "train":
        index["frozen_audits"] = {
            f"{j['name']}_frozen_audit.json": sha256(out / f"{j['name']}_frozen_audit.json")
            for j in meta["experiments"]
        }
    atomic_json(index, cache / f"{split}_index.json")


def verify_history(meta, job, state, n):
    if [v["epoch"] for v in state["history"]] != list(range(1, state["epoch"] + 1)) or state[
        "epoch"
    ] > meta["epochs"]:
        raise ValueError("Incomplete reader history")
    steps = (n + meta["batch"] - 1) // meta["batch"]
    for v in state["history"]:
        if (
            v["sampling"] != plan(n, job["seed"], v["epoch"])[2]
            or v["windows"] != n
            or v["prefix_exposures"] != 4 * n
            or v["steps"] != steps
        ):
            raise ValueError("Reader exposure mismatch")
        if not np.isclose(
            v["lr"],
            gr.growth.learning_rate(v["epoch"], meta["epochs"], meta["lr"]),
            rtol=1e-14,
            atol=0,
        ):
            raise ValueError("Reader LR changed")
    best = min(
        [(0, state["initial_validation"])]
        + [(v["epoch"], v["validation"]) for v in state["history"]],
        key=lambda v: v[1]["primary"],
    )
    if best != (state["best_epoch"], state["best_validation"]):
        raise ValueError("Reader selection changed")


def worker(out, name, device="cuda"):
    meta = read_json(out / "manifest.json")
    if meta["code_sha256"] != code_identity() or meta["schema"] != SCHEMA:
        raise ValueError("Worker code changed")
    job = next(j for j in meta["experiments"] if j["name"] == name)
    path = out / name
    path.mkdir(exist_ok=True)
    indexes = {s: verify_cache(meta, out, s) for s in ("train", "val")}
    metadata = {
        "manifest": meta,
        "job": job,
        "cache": {s: sha256(out / f"cache/{s}_index.json") for s in indexes},
    }
    if (path / "completion.json").exists():
        done = read_json(path / "completion.json")
        if done["metadata"] != metadata or done["status"] != "complete":
            raise ValueError("Reader completion changed")
        verify_files(path, done["files"])
        return
    host = arrays(out, "train", name)
    stats = identity_scales(meta["width"])
    n = len(host["x"])
    data = device_arrays(host, device)
    val = device_arrays(arrays(out, "val", name), device)
    head = original_head(meta, out, job, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=meta["lr"], weight_decay=1e-4)
    initial_signature = ur.bb.state_signature(head)
    if (path / "last.pt").exists():
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if (
            state["metadata"] != metadata
            or state["scales"] != stats
            or state["initial_signature"] != initial_signature
        ):
            raise ValueError("Resume configuration changed")
        verify_history(meta, job, state, n)
        head.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        expected = (
            state["history"][-1]["validation"] if state["history"] else state["initial_validation"]
        )
        ur.pf.ea.require_replay(
            lr.validate(head, val, stats, meta["evaluation_batch"]), expected, "reader resume"
        )
    else:
        initial = lr.validate(head, val, stats, meta["evaluation_batch"])
        expected = read_json(out / f"{name}_frozen_audit.json")["validation"]
        ur.pf.ea.require_replay(
            {"local_primary": initial["primary"]},
            {"local_primary": expected["local_primary"]},
            "original cached local validation",
        )
        if optimizer.state:
            raise ValueError("Warm-start optimizer must start empty")
        state = {
            "metadata": metadata,
            "scales": stats,
            "initial_signature": initial_signature,
            "epoch": 0,
            "history": [],
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_model": ur.bb.ab.cpu_state(head),
        }
        atomic_json(
            {
                "validation": initial,
                "initial_signature": initial_signature,
                "reader_parameters": sum(p.numel() for p in head.parameters()),
            },
            path / "initial_validation.json",
        )

    def save():
        state.update(
            model=ur.bb.ab.cpu_state(head), optimizer=optimizer.state_dict(), rng=rng_state()
        )
        publish(state, path)

    save()
    for epoch in range(state["epoch"] + 1, meta["epochs"] + 1):
        started = time.monotonic()
        order, ps, sampling = plan(n, job["seed"], epoch)
        rate = gr.growth.learning_rate(epoch, meta["epochs"], meta["lr"])
        for g in optimizer.param_groups:
            g["lr"] = rate
        trained = lr.epoch(
            head, data, stats, meta["train_prefixes"], order, ps, meta["batch"], optimizer
        )
        score = lr.validate(head, val, stats, meta["evaluation_batch"])
        if score["primary"] < state["best_validation"]["primary"]:
            state.update(
                best_epoch=epoch, best_validation=score, best_model=ur.bb.ab.cpu_state(head)
            )
        state["history"].append(
            dict(
                epoch=epoch,
                lr=rate,
                sampling=sampling,
                validation=score,
                **trained,
                seconds=time.monotonic() - started,
            )
        )
        state["epoch"] = epoch
        save()
        progress(
            f"{name}: {epoch}/{meta['epochs']}, validation={score['primary']:.6f}, best={state['best_epoch']}"
        )
    verify_history(meta, job, state, n)
    atomic_json(
        dict(
            selected_epoch=state["best_epoch"],
            epochs=state["epoch"],
            validation=state["best_validation"],
            encoder_updates=0,
            **{
                k: sum(v[k] for v in state["history"])
                for k in ("windows", "prefix_exposures", "steps")
            },
            seconds=sum(v["seconds"] for v in state["history"]),
            tail_validation=[v["validation"]["primary"] for v in state["history"][-20:]],
        ),
        path / "training_summary.json",
    )
    files = {
        n: sha256(path / n)
        for n in (
            "best.pt",
            "last.pt",
            "history.json",
            "initial_validation.json",
            "training_summary.json",
        )
    }
    atomic_json(
        {"status": "complete", "metadata": metadata, "files": files}, path / "completion.json"
    )


def lock_selection(meta, out):
    trials = {}
    weights = {}
    n = verify_cache(meta, out, "train")["windows"]
    for j in meta["experiments"]:
        path = out / j["name"]
        d = read_json(path / "completion.json")
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        summary = read_json(path / "training_summary.json")
        expected = {
            "manifest": meta,
            "job": j,
            "cache": {s: sha256(out / f"cache/{s}_index.json") for s in ("train", "val")},
        }
        if (
            d["status"] != "complete"
            or d["metadata"] != expected
            or last["metadata"] != expected
            or best["metadata"] != expected
        ):
            raise ValueError("All reader completions required")
        verify_files(path, d["files"])
        verify_history(meta, j, last, n)
        if (
            last["epoch"] != meta["epochs"]
            or best["epoch"] != last["best_epoch"]
            or best["validation"] != last["best_validation"]
            or summary["selected_epoch"] != best["epoch"]
            or summary["validation"] != best["validation"]
            or read_json(path / "history.json") != last["history"]
        ):
            raise ValueError("Reader checkpoint selection mismatch")
        if any(not torch.equal(v, best["model"][k]) for k, v in last["best_model"].items()):
            raise ValueError("Selected reader payload mismatch")
        for k in ("windows", "prefix_exposures", "steps"):
            if summary[k] != sum(v[k] for v in last["history"]):
                raise ValueError("Reader budget mismatch")
        trials[j["name"]] = summary
        weights[j["name"]] = {k: sha256(path / f"{k}.pt") for k in ("best", "last")}
    lock = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "trials": trials,
        "weights": weights,
        "cache_indexes": {s: sha256(out / f"cache/{s}_index.json") for s in ("train", "val")},
    }
    path = out / "selection_lock.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("Selections changed")
    atomic_json(lock, path)
    return lock


def check_selection(meta, out):
    lock = read_json(out / "selection_lock.json")
    names = {j["name"] for j in meta["experiments"]}
    if (
        lock["manifest_sha256"] != sha256(out / "manifest.json")
        or set(lock["trials"]) != names
        or set(lock["weights"]) != names
        or set(lock["cache_indexes"]) != {"train", "val"}
    ):
        raise ValueError("Both warm-start readers must be locked")
    for n, w in lock["weights"].items():
        if set(w) != {"best", "last"}:
            raise ValueError("Both reader checkpoints required")
        for k, h in w.items():
            if sha256(out / n / f"{k}.pt") != h:
                raise ValueError("Locked reader changed")
    for s, h in lock["cache_indexes"].items():
        if sha256(out / f"cache/{s}_index.json") != h:
            raise ValueError("Fitting cache changed")
        verify_cache(meta, out, s)
    return lock


def run_jobs(out, jobs):
    pending = list(read_json(out / "manifest.json")["experiments"])
    active = {}
    seen = {}
    try:
        while pending or active:
            while pending and len(active) < jobs:
                j = pending.pop(0)
                path = out / j["name"]
                path.mkdir(exist_ok=True)
                log = (path / "run.log").open("a")
                try:
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "obson.babel.local_warmstart_run",
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
                active[j["name"]] = (p, log)
                progress(f"Started {j['name']}, pid={p.pid}")
            for name, (p, log) in list(active.items()):
                f = out / name / "progress.json"
                if f.exists():
                    row = read_json(f)
                    if seen.get(name) != row["epoch"]:
                        seen[name] = row["epoch"]
                        progress(f"{name}: {row}")
                if p.poll() is not None:
                    log.close()
                    del active[name]
                    if p.returncode:
                        file = out / name / "run.log"
                        with file.open("rb") as f:
                            f.seek(0, 2)
                            f.seek(max(0, f.tell() - 65536))
                            tail = "\n".join(
                                f.read().decode("utf-8", errors="replace").splitlines()[-80:]
                            )
                        raise RuntimeError(f"{name} failed ({p.returncode}); {file}\n{tail}")
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


@torch.no_grad()
def evaluate(meta, out, device):
    check_selection(meta, out)
    _, local = statistics(meta)
    source = Path(meta["source"])
    records = {}
    inventories = {}
    replay = {}
    for split in ("test", "cross_research"):
        index = verify_cache(meta, out, split)
        positions = index["positions"]
        rows = read_json(out / f"{split}_inventory.json")
        inventories[split] = rows
        records[split] = {}
        for job in meta["experiments"]:
            name = job["name"]
            data = arrays(out, split, name)
            original, _ = original_model(meta, job, device)
            original.eval().requires_grad_(False)
            before = ur.bb.state_signature(original)
            old = read_json(source / f"{split}_{name}_best.json")
            last = torch.load(out / name / "last.pt", map_location="cpu", weights_only=True)
            for kind in ("original", "best", "last"):
                if kind == "original":
                    head = original.local_head
                    stats = None
                else:
                    ck = torch.load(
                        out / name / f"{kind}.pt", map_location="cpu", weights_only=True
                    )
                    head = lr.reader(meta["width"], meta["hidden"], job["seed"], device)
                    head.load_state_dict(ck["model"])
                    head.eval().requires_grad_(False)
                    stats = identity_scales(meta["width"])
                    if last["scales"] != stats:
                        raise ValueError("Raw state identity normalization changed")
                    expected = (
                        ck["validation"] if kind == "best" else ck["history"][-1]["validation"]
                    )
                    actual = lr.validate(
                        head,
                        device_arrays(arrays(out, "val", name), device),
                        stats,
                        meta["evaluation_batch"],
                    )
                    ur.pf.ea.require_replay(actual, expected, "selected reader validation")
                pred = []
                for start in range(0, len(data["x"]), meta["evaluation_batch"]):
                    x = torch.tensor(
                        np.asarray(data["x"][start : start + meta["evaluation_batch"]]),
                        device=device,
                    )
                    p = (
                        head(x).reshape(*x.shape[:-1], 16, 7)
                        if stats is None
                        else lr.predict(head, x, stats)
                    )
                    pred.append(p.cpu().numpy())
                pred = np.concatenate(pred)
                scores = {}
                errors = {}
                for i, prefix in enumerate(positions):
                    scores[f"p{prefix}"], errors[f"p{prefix}"] = ur.bb.pr.measure(
                        pred[:, i],
                        np.asarray(data["y"][:, i]),
                        np.asarray(data["mask"][:, i]),
                        local,
                    )
                for group, ps in [
                    ("trained", meta["val_prefixes"]),
                    ("held", meta["held_prefixes"]),
                ]:
                    errors[group] = {
                        k: np.mean([errors[f"p{p}"][k] for p in ps], axis=0)
                        for k in ur.bb.pr.METRICS
                    }
                    scores[group] = {
                        "metrics": {k: float(v.mean()) for k, v in errors[group].items()}
                    }
                if kind == "original":
                    differences = recheck.mismatches(
                        scores, {k: old["reconstruction"]["scores"][k] for k in scores}
                    )
                    report = {
                        "passed": not differences,
                        "differences": differences,
                        "scoring_batch": meta["evaluation_batch"],
                    }
                    atomic_json(report, out / f"{split}_{name}_original_numeric_replay.json")
                    if differences:
                        raise ValueError(f"Original local replay differs: {differences[:3]}")
                    replay[split + "/" + name] = {
                        "original_head_replayed": True,
                        "frozen_model_signature": before,
                        "endpoint_certificate_sha256": sha256(source / f"{split}_{name}_best.json"),
                    }
                value = {
                    "scores": scores,
                    "errors": {p: {k: v.tolist() for k, v in e.items()} for p, e in errors.items()},
                    "examples": [
                        {
                            "index": int(i),
                            "positions": positions,
                            "prediction": pred[i].tolist(),
                            "target": data["y"][i].tolist(),
                            "mask": data["mask"][i].tolist(),
                        }
                        for i in np.linspace(0, len(rows) - 1, 4, dtype=int)
                    ],
                }
                atomic_json(value, out / f"{split}_{name}_{kind}.json")
                records[split][name + "_" + kind] = value
            if before != ur.bb.state_signature(original):
                raise ValueError("Frozen encoder/decoder/head mutated")
            del original
            progress(f"Local readout evaluation: {split}/{name}, original/best/last")
    decision = decide(records, inventories)
    atomic_json(decision, out / "decision.json")
    atomic_json(replay, out / "frozen_replay.json")
    inherited = read_json(source / "decision.json")
    atomic_json(
        {
            "source_status": inherited["status"],
            "original_utility_protocol": inherited["original_utility_protocol"],
            "source_decision_sha256": sha256(source / "decision.json"),
            "encoder_updates": 0,
            "global_decoder_updates": 0,
            "utility_readout_updates": 0,
            "interpretation": "Original control600 encoder, global decoder and utility files unchanged. Local-head adaptation does not improve or requalify those capabilities; full utility remains unpassed.",
        },
        out / "unchanged_capabilities.json",
    )
    summary = {
        s: {
            n: {task: v["scores"][task]["metrics"]["primary"] for task in ("trained", "held")}
            for n, v in bank.items()
        }
        for s, bank in records.items()
    }
    atomic_json({"summary": summary, "decision": decision}, out / "local_warmstart_metrics.json")
    lines = [
        "# Original-head warm-start on frozen control600",
        "",
        f"Decision: {decision['status']}; original capacity verdict unchanged.",
        "",
        "| dataset/model/head | trained-prefix error | held-prefix error |",
        "|---|---:|---:|",
    ]
    for s, bank in summary.items():
        for n, v in bank.items():
            lines.append(f"| {s}/{n} | {v['trained']:.6f} | {v['held']:.6f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def run(source, out, jobs=1, device="cuda"):
    source, out = source.resolve(), out.resolve()
    check_output(source, out)
    if jobs not in (1, 2):
        raise ValueError("One or two reader workers supported")
    progress("Verifying control600, completed reevaluation and original local heads")
    identity = source_identity(source)
    check_output(source, out, identity)
    meta = make_manifest(source, identity)
    runtime = read_json(source / "runtime.json")
    if runtime["torch"] != str(torch.__version__) or runtime["numpy"] != np.__version__:
        raise ValueError("Restore source runtime")
    if (out / "manifest.json").exists():
        if read_json(out / "manifest.json") != meta:
            raise ValueError("Source/configuration changed; use new output")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Empty output required")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out / "manifest.json")
    if (out / "completion.json").exists():
        d = read_json(out / "completion.json")
        if d["status"] != "complete" or d["encoder_updates"] != 0:
            raise ValueError("Invalid completion")
        ur.bb.ab.sc.verify_worker_files(out, d["files"])
        check_selection(meta, out)
        progress("Already complete; no repeated training")
        return
    started = time.monotonic()
    try:
        for s in ("train", "val", "test", "cross_research"):
            shutil.copyfile(
                training_root(meta) / f"{s}_inventory.json", out / f"{s}_inventory.json"
            )
        atomic_json(
            gr.rr.up.probe.inventory_audit(
                {s: read_json(out / f"{s}_inventory.json") for s in gr.rr.old.SPLITS}
            ),
            out / "data_audit.json",
        )
        for s in ("train", "val"):
            prepare(meta, out, s, device)
        release_cache_cuda(out, jobs, device)
        run_jobs(out, jobs)
        lock_selection(meta, out)
        for s in ("test", "cross_research"):
            prepare(meta, out, s, device)
        evaluate(meta, out, device)
        if source_identity(source) != identity:
            raise ValueError("Frozen source changed")
    except Exception as exc:
        atomic_json(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, out / "failure.json"
        )
        raise
    atomic_json(
        {
            "seconds": time.monotonic() - started,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "jobs": jobs,
            "device": str(device),
        },
        out / "runtime.json",
    )
    files = {
        str(p.relative_to(out)): sha256(p)
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".md")
        and p.name not in ("completion.json", "failure.json", "progress.json")
    }
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "encoder_updates": 0,
            "cells": 2,
            "files": files,
        },
        out / "completion.json",
    )


def main():
    import signal

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("all", "worker"))
    p.add_argument(
        "--source", type=Path, default=Path("checkpoints/babel_cross_period768_v2_recheck")
    )
    p.add_argument("--out", type=Path, default=Path("checkpoints/babel_local_warmstart768"))
    p.add_argument("--name")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Formal inference and reader training run on AutoDL CUDA only")
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
        run(a.source, a.out, a.jobs)


if __name__ == "__main__":
    main()
