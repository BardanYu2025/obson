"""Controlled sparse, expectation-matched dense and all-prefix supervision."""

import argparse
import fcntl
import gc
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import depth_scaling_evaluate as de
from . import depth_scaling_run as parent
from . import linear_history_run as source_run
from . import macro_history_run as base
from . import position_supervision as core
from .ae_extend import atomic_json, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-position-supervision768-v1"
KEYS = (*core.FAMILIES, "query_fixed", "structure", "local", "optimization_value")
tm, root, stats, data = parent.tm, parent.root, parent.stats, parent.data
xr, ur = parent.xr, parent.ur


def code_identity():
    from . import position_supervision_evaluate as ev

    return (
        source_run.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (core, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


source_identity = source_run.source_identity


def make_manifest(source, identity, micro=8):
    meta = source_run.make_manifest(source, identity, micro)
    meta.update(
        schema=SCHEMA,
        code_sha256=code_identity(),
        epochs=60,
        position_chunk=4,
        sampling_stream_start=1591,
        experiments=[
            {"name": f"{mode}_s{seed}", "mode": mode, "seed": seed}
            for seed in (42, 43)
            for mode in core.MODES
        ],
        initialization="Identical Macro control best97/94 model and original nonlinear query; no calibration or architecture change.",
        objective="Original historical query0.3+structure1+detached local0.25. sampled: four original interior positions + endpoint. dense_matched: exact expectation of sampled weights. dense_all: uniform prefixes2..128 including legacy held positions; position1 has no past target.",
        training="Six60-epoch arms, identical4800 triplets/14400 views and38 updates per epoch. Decoder query counts/compute differ; no equal-FLOPs claim.",
        selection="Original validation protected near/far minimax, same epoch0 fallback. No research position profile used for selection.",
        decision="Dense matched versus sampled tests estimator density; dense all versus both tests additional short prefixes and reweighting, not a single isolated causal factor. Historical price/activity/utility protection, no automatic promotion.",
        stop="One finite six-arm matrix; not from-scratch pretraining or future prediction. All-prefix result does not establish equal full128 contexts.",
        position_semantics="Prefix2..128 has p observed window bars; features may contain causal prehistory. Legacy held48/80/112 are TRAINED in dense_all, not position holdouts in comparisons.",
    )
    meta.pop("calibration_epochs", None)
    return meta


construct = source_run.construct


@lru_cache(maxsize=8)
def read_plan(path):
    return read_json(Path(path) / "train_cross_plan.json")["rows"]


def plan_rows(meta):
    return read_plan(str(root(meta)))


@lru_cache(maxsize=8)
def training_ids(path):
    return tuple(base.macro_data.eligible(read_plan(path)))


def plan(meta, seed, epoch, n):
    rows = plan_rows(meta)
    if len(rows) != n:
        raise ValueError("Training pool changed")
    stream = meta["sampling_stream_start"] + epoch
    ids, shifts = base.macro_data.schedule(
        rows, seed, stream, meta["budget"], training_ids(str(root(meta)))
    )
    ps = core.hq.ba.position_plan(len(ids), seed, stream)
    qs = core.hq.sample_prefixes(len(ids), seed, stream)
    info = {
        "phase": "main",
        "stream": stream,
        "ids": ur.bb.cov.ndarray_hash(ids),
        "shifts": ur.bb.cov.ndarray_hash(shifts),
        "prefixes": ur.bb.cov.ndarray_hash(ps),
        "query_prefixes": ur.bb.cov.ndarray_hash(qs),
        "unique_triplets": len(np.unique(ids)),
    }
    return ids, shifts, ps, qs, info


def rates(meta, epoch):
    return {
        k: xr.growth.learning_rate(epoch, meta["epochs"], meta[k])
        for k in ("encoder_lr", "head_lr", "query_lr")
    }


def make_batch(builder, schedule, start, stop, device):
    ids, shifts, prefixes, qps, _ = schedule
    a, b, c, _, _ = builder(ids[start:stop], shifts[start:stop])
    batch = parent.batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device)
    ps = torch.tensor(np.tile(prefixes[start:stop], (3, 1)), device=device)
    qs = torch.tensor(np.tile(qps[start:stop], (3, 1)), device=device)
    return batch, ps, qs, torch.tensor(shifts[start:stop], device=device)


def train_epoch(meta, job, model, query, builder, schedule, optimizers, device, max_steps=None):
    model.eval()
    query.eval()
    totals = dict.fromkeys(KEYS, 0.0)
    norms, steps, seen = [], 0, 0
    statistics, local = stats(meta)
    groups = core.configure(model, query, True)
    for left in range(0, len(schedule[0]), meta["batch"]):
        size = min(meta["batch"], len(schedule[0]) - left)
        model.zero_grad(set_to_none=True)
        query.zero_grad(set_to_none=True)
        for start in range(left, left + size, meta["micro"]):
            stop = min(start + meta["micro"], left + size)
            b, ps, qs, sh = make_batch(builder, schedule, start, stop, device)
            parts = core.backward_batch(
                model,
                query,
                b,
                ps,
                qs,
                sh,
                statistics,
                local,
                job["mode"],
                size,
                meta["position_chunk"],
            )
            for k in KEYS:
                totals[k] += float(parts[k].detach().sum())
        norms.append(
            [
                float(torch.nn.utils.clip_grad_norm_(g, 1.0, error_if_nonfinite=True))
                for _, g, _ in groups
            ]
        )
        if optimizers is not None:
            for opt in optimizers:
                opt.step()
        steps += 1
        seen += size
        if max_steps is not None and steps >= max_steps:
            break
    return dict(
        **{k: v / seen for k, v in totals.items()},
        gradient_norm_mean=np.mean(norms, axis=0).tolist(),
        gradient_clip_fraction=np.mean(np.asarray(norms) > 1, axis=0).tolist(),
    ), steps


@torch.inference_mode()
def validation(meta, model, query, device):
    model.eval()
    query.eval()
    d = data(meta, "val")
    statistics, _ = stats(meta)
    parts = dict.fromkeys((*core.FAMILIES, "primary"), 0.0)
    far = 0.0
    for start in range(0, len(d["x"]), meta["evaluation_batch"]):
        b = parent.batch_tensors(
            {
                k: v[start : start + meta["evaluation_batch"]]
                for k, v in d.items()
                if k in ("x", "y", "mask")
            },
            device,
        )
        ps = torch.tensor([core.hq.ba.VAL_PREFIXES] * len(b["x"]), device=device)
        _, pred = core.hq.predict_states(model, query, b["x"], ps)
        y, m = core.hq.targets(b, ps, statistics)
        sc = core.hq.metrics(pred, y, m, statistics)
        for k in parts:
            parts[k] += float(sc[k].mean(1).sum())
        far += float(
            core.structure.metrics(pred[:, -1:], y[:, -1:], m[:, -1:], statistics, lo=65)[
                "primary"
            ].sum()
        )
    parts = {k: v / len(d["x"]) for k, v in parts.items()}
    matched = de.matched_scores(meta, model, query, d, device, lengths=(64, 128), pair=(64, 128))
    pair = matched["paired64_128"]
    near = float(np.mean(np.asarray(pair["actual_error"]["path"])))
    gap = float(np.mean(np.asarray(pair["gap"]["path"])))
    result = {
        "query_price": float(core.price(parts)),
        "near_price": near,
        "far": far / len(d["x"]),
        "gap_price": gap,
        "query_fixed": parts["primary"],
        "activity": parts["activity"],
        "families": parts,
    }
    if not np.isfinite([v for k, v in result.items() if k != "families"]).all():
        raise ValueError("Nonfinite validation")
    return result


def selected_value(v, initial):
    for k in ("query_price", "near_price", "far", "gap_price"):
        if initial[k] <= 0 or not np.isfinite(initial[k]) or not np.isfinite(v[k]) or v[k] < 0:
            raise ValueError("Invalid selection metric")
        if v[k] > initial[k] * (1.10 if k == "gap_price" else 1.05):
            return (float("inf"), float("inf"))
    a, b = v["near_price"] / initial["near_price"], v["far"] / initial["far"]
    return max(a, b), (a + b) / 2


def verify_history(meta, job, state, n):
    h = state["history"]
    if state["epoch"] != len(h) or len(h) > meta["epochs"]:
        raise ValueError("Invalid history length")
    steps = (meta["budget"] + meta["batch"] - 1) // meta["batch"]
    for epoch, row in enumerate(h, 1):
        if (
            row["epoch"] != epoch
            or row["sampling"] != plan(meta, job["seed"], epoch, n)[4]
            or row["views"] != 3 * meta["budget"]
            or row["steps"] != steps
            or row["exposure"] != core.exposure(job["mode"], 3 * meta["budget"])
        ):
            raise ValueError("Sampling or update budget changed")
        if any(
            not np.isclose(row[k], v, rtol=1e-14, atol=0) for k, v in rates(meta, epoch).items()
        ):
            raise ValueError("LR changed")
    best = min(
        [(0, state["initial_validation"])] + [(x["epoch"], x["validation"]) for x in h],
        key=lambda x: selected_value(x[1], state["initial_validation"]),
    )
    if (state["best_epoch"], state["best_validation"]) != best:
        raise ValueError("Validation selection changed")


def gradient_diagnostic(meta, job, model, query, builder, device):
    before = (ur.bb.state_signature(model), ur.bb.state_signature(query))
    result, _ = train_epoch(
        meta,
        job,
        model,
        query,
        builder,
        plan(meta, job["seed"], 1, len(builder.plan)),
        None,
        device,
        max_steps=1,
    )
    if before != (ur.bb.state_signature(model), ur.bb.state_signature(query)):
        raise ValueError("Diagnostic updated weights")
    return {
        "optimizer_steps": 0,
        "effective_batch": meta["batch"],
        "mode": job["mode"],
        "metrics": result,
    }


def worker(out, name, device="cuda"):
    meta = read_json(out / "manifest.json")
    job = next(j for j in meta["experiments"] if j["name"] == name)
    if meta["code_sha256"] != code_identity():
        raise ValueError("Bound code changed")
    path = out / name
    path.mkdir(exist_ok=True)
    metadata = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "job": job,
        "phase": "main",
    }
    if (path / "completion.json").exists():
        done = read_json(path / "completion.json")
        if done["status"] != "complete" or done["metadata"] != metadata:
            raise ValueError("Worker identity changed")
        parent.verify_files(path, done["files"])
        parent.cleanup_completed(path)
        return
    model, query = construct(meta, job["seed"], device)
    fork = meta["task_source"]["chosen"][str(job["seed"])]["sha256"]
    initial_encoder = ur.bb.state_signature(model.core.encoder)
    initial_query = ur.bb.state_signature(query)
    groups = core.configure(model, query, True)
    fixed = parent.cpu(model.core.decoder)
    opts = [torch.optim.AdamW(g, lr=meta[k], weight_decay=wd) for k, g, wd in groups]
    builder = xr.xd.Builder(tm(meta), root(meta))
    if (path / "last.pt").exists():
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != metadata or state["fork_sha256"] != fork:
            raise ValueError("Resume identity changed")
        verify_history(meta, job, state, len(builder.plan))
        model.load_state_dict(state["model"], strict=True)
        query.load_state_dict(state["query"], strict=True)
        for o, v in zip(opts, state["optimizers"], strict=True):
            o.load_state_dict(v)
        restore_rng(state["rng"])
        expected = (
            state["history"][-1]["validation"] if state["history"] else state["initial_validation"]
        )
        xr.ce.require_nested(
            validation(meta, model, query, device), expected, "position-supervision resume"
        )
    else:
        torch.manual_seed(job["seed"] + 20260930)
        initial = validation(meta, model, query, device)
        state = {
            "metadata": metadata,
            "fork_sha256": fork,
            "epoch": 0,
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_model": parent.cpu(model),
            "best_query": parent.cpu(query),
            "history": [],
        }
        atomic_json(
            gradient_diagnostic(meta, job, model, query, builder, device),
            path / "initial_gradient.json",
        )

    def save():
        state.update(
            model=parent.cpu(model),
            query=parent.cpu(query),
            optimizers=[o.state_dict() for o in opts],
            rng=rng_state(),
        )
        parent.publish(state, path)

    save()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(state["epoch"] + 1, meta["epochs"] + 1):
        started = time.monotonic()
        schedule = plan(meta, job["seed"], epoch, len(builder.plan))
        lr = rates(meta, epoch)
        for o, (k, _, _) in zip(opts, groups, strict=True):
            for g in o.param_groups:
                g["lr"] = lr[k]
        tr, steps = train_epoch(meta, job, model, query, builder, schedule, opts, device)
        val = validation(meta, model, query, device)
        if selected_value(val, state["initial_validation"]) < selected_value(
            state["best_validation"], state["initial_validation"]
        ):
            state.update(
                best_epoch=epoch,
                best_validation=val,
                best_model=parent.cpu(model),
                best_query=parent.cpu(query),
            )
        seconds = time.monotonic() - started
        state["history"].append(
            dict(
                epoch=epoch,
                sampling=schedule[4],
                views=3 * len(schedule[0]),
                steps=steps,
                exposure=core.exposure(job["mode"], 3 * len(schedule[0])),
                train=tr,
                validation=val,
                seconds=seconds,
                **lr,
            )
        )
        state["epoch"] = epoch
        save()
        progress(
            f"{name}: {epoch}/{meta['epochs']}, price_val={val['query_price']:.6f}, activity_val={val['activity']:.6f}, best={state['best_epoch']}, {seconds:.1f}s/epoch, this_worker_eta~{seconds * (meta['epochs'] - epoch) / 60:.0f}min"
        )
    verify_history(meta, job, state, len(builder.plan))
    if any(not torch.equal(v, model.core.decoder.state_dict()[k].cpu()) for k, v in fixed.items()):
        raise ValueError("Frozen PCA decoder changed")
    atomic_json(
        gradient_diagnostic(meta, job, model, query, builder, device),
        path / "last_gradient.json",
    )
    parent.snapshot(state, path, meta["epochs"])
    atomic_json(
        {
            k: state[k]
            for k in (
                "metadata",
                "fork_sha256",
                "epoch",
                "initial_validation",
                "best_epoch",
                "best_validation",
                "history",
            )
        },
        path / "training_state.json",
    )
    summary = {
        "selected_epoch": state["best_epoch"],
        "epochs": meta["epochs"],
        "validation": state["best_validation"],
        "initial_validation": state["initial_validation"],
        "fork_sha256": fork,
        "encoder_frozen": False,
        "exposure_per_epoch": core.exposure(job["mode"], 3 * meta["budget"]),
        "query_start_sha256": initial_query,
        "encoder_start_sha256": initial_encoder,
        "encoder_end_sha256": ur.bb.state_signature(model.core.encoder),
        "seconds": sum(h["seconds"] for h in state["history"]),
        "views": sum(h["views"] for h in state["history"]),
        "steps": sum(h["steps"] for h in state["history"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    }
    atomic_json(summary, path / "training_summary.json")
    files = [
        "history.json",
        "training_state.json",
        "training_summary.json",
        "initial_gradient.json",
        "last_gradient.json",
    ] + [f"e{meta['epochs']}_{k}.pt" for k in ("best", "last")]
    atomic_json(
        {"status": "complete", "metadata": metadata, "files": {n: sha256(path / n) for n in files}},
        path / "completion.json",
    )
    parent.cleanup_completed(path)


def verify_summary(meta, state, summary):
    expected = {
        "selected_epoch": state["best_epoch"],
        "epochs": meta["epochs"],
        "validation": state["best_validation"],
        "initial_validation": state["initial_validation"],
        "fork_sha256": state["fork_sha256"],
        **{k: sum(h[k] for h in state["history"]) for k in ("seconds", "views", "steps")},
    }
    if any(summary[k] != v for k, v in expected.items()):
        raise ValueError("Training summary differs from history")


def lock_models(meta, out):
    trials = {}
    weights = {}
    mh = sha256(out / "manifest.json")
    for job in meta["experiments"]:
        path = out / job["name"]
        done = read_json(path / "completion.json")
        parent.verify_files(path, done["files"])
        expected = {"manifest_sha256": mh, "job": job, "phase": "main"}
        state = read_json(path / "training_state.json")
        if (
            done["metadata"] != expected
            or state["metadata"] != expected
            or done["status"] != "complete"
            or state["epoch"] != meta["epochs"]
        ):
            raise ValueError("Incomplete or foreign worker")
        verify_history(
            meta, job, state, read_json(root(meta) / "train_cross_plan.json")["eligible"]
        )
        if state["history"] != read_json(path / "history.json"):
            raise ValueError("History mismatch")
        if state["fork_sha256"] != meta["task_source"]["chosen"][str(job["seed"])]["sha256"]:
            raise ValueError("Wrong fork")
        summary = read_json(path / "training_summary.json")
        verify_summary(meta, state, summary)
        trials[job["name"]] = summary
        weights[job["name"]] = {}
        for kind in ("best", "last"):
            rel = f"{job['name']}/e{meta['epochs']}_{kind}.pt"
            ck = torch.load(out / rel, map_location="cpu", weights_only=True)
            epoch = state["best_epoch"] if kind == "best" else meta["epochs"]
            val = state["best_validation"] if kind == "best" else state["history"][-1]["validation"]
            if (
                ck["metadata"] != expected
                or ck["budget_epoch"] != meta["epochs"]
                or ck["epoch"] != epoch
                or ck["validation"] != val
            ):
                raise ValueError("Snapshot selection mismatch")
            weights[job["name"]][kind] = {"path": rel, "sha256": sha256(out / rel), "epoch": epoch}
    for seed in (42, 43):
        paired = [trials[f"{mode}_s{seed}"] for mode in core.MODES]
        if any(
            len({r[k] for r in paired}) != 1
            for k in ("encoder_start_sha256", "query_start_sha256", "fork_sha256")
        ):
            raise ValueError("Paired arms did not start from identical checkpoints")
        for r in paired[1:]:
            xr.ce.require_nested(
                r["initial_validation"], paired[0]["initial_validation"], "identical source replay"
            )
    atomic_json(
        {"manifest_sha256": mh, "trials": trials, "weights": weights},
        out / "model_selection_lock.json",
    )


def dispatch(meta, out, jobs):
    pending = list(meta["experiments"])
    running = []
    try:
        while pending or running:
            while pending and len(running) < jobs:
                j = pending.pop(0)
                if (out / j["name"] / "completion.json").exists():
                    worker(out, j["name"])
                    continue
                progress(f"Starting {j['name']}; {len(pending)} queued, {len(running) + 1} active")
                running.append(
                    (
                        j["name"],
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "obson.babel.position_supervision_run",
                                "--out",
                                str(out),
                                "--worker",
                                j["name"],
                            ]
                        ),
                    )
                )
            for name, p in list(running):
                rc = p.poll()
                if rc is not None:
                    running.remove((name, p))
                    if rc:
                        raise RuntimeError(
                            f"{name} failed with exit {rc}; original traceback is in log"
                        )
            if running:
                time.sleep(0.5)
    finally:
        for _, p in running:
            p.terminate()
        for _, p in running:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def preflight(meta, out, device, jobs):
    source_lock = read_json(Path(meta["task_source"]["source"]) / "model_selection_lock.json")
    records = []
    for seed in (42, 43):
        model, query = construct(meta, seed, device)
        with torch.inference_mode():
            causal = ur.pf.er.trained_causality(
                model, torch.tensor(data(meta, "val")["x"][:4], device=device)
            )
            if causal["status"] == "failed":
                raise ValueError("Source causality failed")
            val = validation(meta, model, query, device)
            xr.ce.require_nested(
                val,
                source_lock["trials"][f"control_s{seed}"]["validation"],
                "source validation replay",
            )
        records.append({"seed": seed, "causality": causal, "validation": val})
        del model, query
    atomic_json(records, out / "preflight.json")
    builder = xr.xd.Builder(tm(meta), root(meta))
    checks = []
    for mode in core.MODES:
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model, query = construct(meta, 42, device)
        diagnostic = gradient_diagnostic(
            meta, {"seed": 42, "mode": mode}, model, query, builder, device
        )
        groups = core.configure(model, query, True)
        moments = 8 * sum(p.numel() for _, g, _ in groups for p in g)
        peak = torch.cuda.max_memory_allocated() - baseline
        checks.append(
            {
                "mode": mode,
                "diagnostic": diagnostic,
                "estimated_worker_bytes": int(1.3 * (peak + moments)) + 384 * 1024**2,
            }
        )
        del model, query, groups
    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    per = max(c["estimated_worker_bytes"] for c in checks)
    effective = min(jobs, int(free // per))
    atomic_json(
        {
            "checks": checks,
            "requested_jobs": jobs,
            "effective_jobs": effective,
            "available_bytes": free,
            "estimated_worker_bytes": per,
        },
        out / "gradient_preflight.json",
    )
    if effective < 1:
        raise ValueError(
            "Insufficient GPU memory even for one worker; use smaller MICRO and a new output"
        )
    progress(f"Resource preflight: {effective} workers (requested {jobs})")
    return effective


def check_disk(meta, out, jobs):
    # Size actual model/optimizer tensors on CPU; no inference or training.
    model, query = construct(meta, 42, "cpu")
    state = sum(
        v.numel() * v.element_size() for m in (model, query) for v in m.state_dict().values()
    )
    trained = sum(p.numel() for _, g, _ in core.configure(model, query, True) for p in g)
    retained = 2 * len(meta["experiments"]) * state
    temp = jobs * (2 * (2 * state + 8 * trained) + state)
    required = int(1.2 * (retained + temp)) + 2 * 1024**3
    free = shutil.disk_usage(out).free
    atomic_json(
        {
            "required_bytes": required,
            "available_bytes": free,
            "retained_bytes": retained,
            "atomic_temporary_bytes": temp,
        },
        out / "disk_preflight.json",
    )
    if free < required:
        raise ValueError(
            f"Need {required / 2**30:.1f}GiB free, have {free / 2**30:.1f}GiB. No source files deleted."
        )
    progress(f"Disk: need {required / 2**30:.1f}GiB, available {free / 2**30:.1f}GiB")


def run(source, out, jobs=1, micro=8, evaluate_only=False):
    from . import position_supervision_evaluate as ev

    source = source.resolve()
    out = out.resolve()
    parent.warm.check_output(source, out)
    progress("Verifying completed Macro History control checkpoints; position-supervision study")
    identity = source_identity(source)
    meta = make_manifest(source, identity, micro)
    meta["run_directory"] = str(out)
    for p in (
        meta["source"],
        meta["warmup"]["source"],
        meta["start"]["source"],
        root(meta),
        identity["source"],
        Path(read_json(source / "manifest.json")["task_source"]["source"]),
    ):
        parent.warm.check_output(Path(p), out)
    rt = read_json(Path(identity["source"]) / "runtime.json")
    if rt["torch"] != str(torch.__version__) or rt["numpy"] != np.__version__:
        raise ValueError("Keep source Torch/NumPy versions")
    if out.exists() and any(out.iterdir()) and read_json(out / "manifest.json") != meta:
        raise ValueError("Different source/config: use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        if done["status"] != "complete" or not done["source_unchanged"]:
            raise ValueError("Invalid completion")
        parent.verify_files(out, done["files"])
        return
    complete = all((out / j["name"] / "completion.json").exists() for j in meta["experiments"])
    if evaluate_only and not complete:
        raise ValueError("All six completed workers required for evaluation-only")
    started = time.monotonic()
    atomic_json(
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(),
            "jobs": jobs,
            "micro": micro,
        },
        out / "runtime.json",
    )
    if not complete:
        check_disk(meta, out, jobs)
    xr.xd.verify_data(tm(meta), root(meta))
    ev.audit_sources(meta, out, "cuda")
    if not complete:
        jobs = preflight(meta, out, "cuda", jobs)
        progress(f"Six paired position-supervision arms, {meta['epochs']} epochs each")
        dispatch(meta, out, jobs)
    lock_models(meta, out)
    progress("Stage3/3: locked original-state readouts and all information/utility comparisons")
    ev.fit_readouts(meta, out, "cuda")
    ev.evaluate(meta, out, "cuda")
    if source_identity(source) != identity:
        raise ValueError("Source changed during experiment")
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "seconds_this_invocation": time.monotonic() - started,
            "files": {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and p != out / "completion.json"
                and p.name != "run_status.txt"
                and p.suffix not in (".tmp", ".log")
            },
        },
        out / "completion.json",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="checkpoints/babel_macro_history768")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    p.add_argument("--micro", type=int, choices=(1, 2, 4, 8, 16, 32), default=8)
    p.add_argument("--worker")
    p.add_argument("--evaluate-only", action="store_true")
    a = p.parse_args()
    ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Real training/evaluation requires AutoDL CUDA")
    out = Path(a.out).resolve()
    if a.worker:
        worker(out, a.worker)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Position-supervision experiment already running") from None
        run(Path(a.source), out, a.jobs, a.micro, a.evaluate_only)


if __name__ == "__main__":
    main()
