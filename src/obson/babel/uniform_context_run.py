"""Finite matched scratch training; CUDA only, resumable and selection-locked."""

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

from . import linear_history_run as parent
from . import uniform_context as core
from . import uniform_context_data as data
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-uniform-context768-v1"
VAL_PREFIXES = (32, 64, 96, 128)


def code_identity():
    from . import uniform_context_evaluate as ev

    return (
        parent.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (core, data, ev, data.original)}
        | {Path(__file__).name: sha256(__file__)}
    )


def jobs(meta):
    return [
        {"name": f"{mode}_s{seed}", "mode": mode, "seed": seed}
        for seed in meta["seeds"]
        for mode in core.MODES
    ]


def make_manifest(source, identity, micro):
    old = read_json(source / "manifest.json")
    return {
        "schema": SCHEMA,
        "identity": old["identity"],
        "architecture": old["architecture"],
        "task_source": identity,
        "code_sha256": code_identity(),
        "encoder": core.CONFIG,
        "windows": list(core.WINDOWS),
        "seeds": [42, 43],
        "epochs": 120,
        "batch": 128,
        "micro": micro,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "validation_prefixes": list(VAL_PREFIXES),
        "protocol": {
            "initialization": "Scratch, paired seeds and identical learned parameter initialization for all three arms; no neural source weights or PCA teacher",
            "input": "255 same-contract/same-partition causal feature rows; last128 legacy prefixes versus complete128 rolling contexts; EMA prehistory permitted",
            "objective": "Equal five sampled physical endpoints; query0.4path+0.2change1+0.1body+0.3activity plus remote structure1. Source detached local head omitted uniformly",
            "selection": "Minimum same-full-context query+structure MSE on validation endpoints32/64/96/128, epoch0 included, strict improvement; no research selection",
            "fairness": "Same examples/endpoints/updates/parameter count/LR; unequal compute. Prefix vs rolling changes training context AND target support; relative vs rolling bundles locality and relative bias. Old Macro has substantially more pretraining: reference only",
            "primary": "Relative vs rolling: at least5% balanced price gain with supported upper monthly paired interval<=0; each age band and activity no worse than10%; utility no worse than5%. Both seeds/cohorts required",
            "stop": "Six120-epoch workers, no early stop, no automatic LR grid or promotion. Report best/last and tails; budget limits convergence claims. Existing research cohorts reused",
            "branches": "Dense all128 equal-position supervision from scratch, cross-resolution common history, and stochastic innovation remain untested; not silently closed",
        },
    }


def binding(out, job):
    return {"manifest_sha256": sha256(out / "manifest.json"), "job": job}


def schedule(n, seed, epoch):
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261004])).permutation(n)
    return order, core.prefixes(n, seed, epoch)


def predict(encoder, query, x, ps, native=False):
    z = core.read_states(encoder, x, ps, native)
    pred = query(z.flatten(0, 1)).reshape(len(x), ps.shape[1], 127, 7)
    return z, pred


@torch.no_grad()
def validation(meta, encoder, query, x, device):
    encoder.eval()
    query.eval()
    stats = parent.stats(meta)[0]
    rows = {}
    for left in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[left : left + meta["micro"]], device=device)
        ps = torch.tensor(meta["validation_prefixes"], device=device).expand(len(b), -1)
        _, pred = predict(encoder, query, b, ps)
        y, mask = core.targets(b, ps, stats)
        for k, v in core.band_rows(pred, y, mask, stats).items():
            rows.setdefault(k, []).append(v.cpu().numpy())
    return {k: float(np.concatenate(v).mean()) for k, v in rows.items()}


def train_epoch(meta, job, encoder, query, x, opt, epoch, device):
    encoder.train()
    query.train()
    stats = parent.stats(meta)[0]
    order, ps = schedule(len(x), job["seed"], epoch)
    total, steps = 0.0, 0
    for left in range(0, len(x), meta["batch"]):
        size = min(meta["batch"], len(x) - left)
        opt.zero_grad(set_to_none=True)
        for start in range(left, left + size, meta["micro"]):
            stop = min(start + meta["micro"], left + size)
            b = torch.as_tensor(x[order[start:stop]], device=device)
            p = torch.as_tensor(ps[start:stop], device=device)
            _, pred = predict(encoder, query, b, p, native=True)
            y, mask = core.targets(b, p, stats, native=job["mode"] == "prefix")
            values = core.loss_rows(pred, y, mask, stats, True)["objective"].mean(1)
            loss = values.sum() / size
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            loss.backward()
            total += float(values.detach().sum())
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(query.parameters()), 1.0, error_if_nonfinite=True
        )
        opt.step()
        steps += 1
    return {
        "loss": total / len(x),
        "steps": steps,
        "examples": len(x),
        "supervised_states": 5 * len(x),
        "order_hash": parent.ur.bb.cov.ndarray_hash(order),
        "prefix_hash": parent.ur.bb.cov.ndarray_hash(ps),
    }


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def materialize_best(ck, folder):
    atomic_save(
        {
            "encoder": ck["best_encoder"],
            "query": ck["best_query"],
            "epoch": ck["best_epoch"],
            "metadata": ck["metadata"],
            "validation": ck["best_validation"],
        },
        folder / "best.pt",
    )


def worker(out, job):
    meta = read_json(out / "manifest.json")
    if meta["code_sha256"] != code_identity():
        raise ValueError("Worker code differs from manifest")
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "completion.json").exists():
            done = read_json(folder / "completion.json")
            if done["metadata"] != binding(out, job):
                raise ValueError("Worker identity changed")
            verify_files(folder, done["files"])
            return
        train, _ = data.cache(out, "train")
        val, _ = data.cache(out, "val")
        device = "cuda"
        encoder, query = core.make_models(job["mode"], job["seed"], device, meta["encoder"])
        opt = torch.optim.AdamW(
            list(encoder.parameters()) + list(query.parameters()),
            lr=meta["lr"],
            weight_decay=meta["weight_decay"],
        )
        if (folder / "last.pt").exists():
            ck = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
            if ck["metadata"] != binding(out, job):
                raise ValueError("Checkpoint identity changed")
            encoder.load_state_dict(ck["encoder"], strict=True)
            query.load_state_dict(ck["query"], strict=True)
            opt.load_state_dict(ck["optimizer"])
            # AdamW casts optimizer tensors to their parameter device on load.
            materialize_best(ck, folder)
        else:
            initial = validation(meta, encoder, query, val, device)
            ck = {
                "epoch": 0,
                "best_epoch": 0,
                "best_score": core.validation_score(initial),
                "best_validation": initial,
                "best_encoder": cpu_state(encoder),
                "best_query": cpu_state(query),
                "metadata": binding(out, job),
                "history": [],
                "initial_validation": initial,
                "initial_parameter_hash": parent.ur.bb.state_signature(encoder),
                "initial_query_hash": parent.ur.bb.state_signature(query),
                "parameters": sum(p.numel() for p in encoder.parameters())
                + sum(p.numel() for p in query.parameters()),
            }
        for epoch in range(ck["epoch"] + 1, meta["epochs"] + 1):
            started = time.monotonic()
            lr = core.learning_rate(epoch, meta["epochs"], meta["lr"])
            for group in opt.param_groups:
                group["lr"] = lr
            train_values = train_epoch(meta, job, encoder, query, train, opt, epoch, device)
            scores = validation(meta, encoder, query, val, device)
            score = core.validation_score(scores)
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation score")
            if score < ck["best_score"]:
                ck.update(
                    best_score=score,
                    best_epoch=epoch,
                    best_validation=scores,
                    best_encoder=cpu_state(encoder),
                    best_query=cpu_state(query),
                )
            seconds = time.monotonic() - started
            ck["history"].append(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train": train_values,
                    "validation": scores,
                    "best_epoch": ck["best_epoch"],
                    "seconds": seconds,
                }
            )
            ck.update(
                epoch=epoch,
                encoder=cpu_state(encoder),
                query=cpu_state(query),
                optimizer=opt.state_dict(),
            )
            # last is the atomic journal; best/history are derived, recoverable views.
            atomic_save(ck, folder / "last.pt")
            materialize_best(ck, folder)
            atomic_json(ck["history"], folder / "history.json")
            progress(
                f"{job['name']}: {epoch}/{meta['epochs']}, val={score:.6f}, best={ck['best_epoch']}, {seconds:.1f}s/epoch, worker ETA~{seconds * (meta['epochs'] - epoch) / 60:.0f}min"
            )
        atomic_json(ck["history"], folder / "history.json")
        atomic_json(
            {
                "metadata": binding(out, job),
                "epochs": ck["epoch"],
                "best_epoch": ck["best_epoch"],
                "parameters": ck["parameters"],
                "initial_parameter_hash": ck["initial_parameter_hash"],
                "initial_query_hash": ck["initial_query_hash"],
                "initial_validation": ck["initial_validation"],
                "files": {n: sha256(folder / n) for n in ("best.pt", "last.pt", "history.json")},
            },
            folder / "completion.json",
        )


def check_selection(out):
    lock = read_json(out / "model_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Model selection manifest changed")
    verify_files(out, lock["files"])
    for split, h in lock["cache_indexes"].items():
        if sha256(out / f"cache/{split}_index.json") != h:
            raise ValueError("Training/selection data changed")
    return lock


def lock_selection(meta, out):
    if (out / "model_selection_lock.json").exists():
        return check_selection(out)
    files = {}
    initial = {}
    schedules = {}
    for job in jobs(meta):
        folder = out / job["name"]
        done = read_json(folder / "completion.json")
        if done["metadata"] != binding(out, job) or done["epochs"] != meta["epochs"]:
            raise ValueError("Incomplete matrix")
        verify_files(folder, done["files"])
        initial.setdefault(job["seed"], set()).add(
            (done["parameters"], done["initial_parameter_hash"], done["initial_query_hash"])
        )
        history = read_json(folder / "history.json")
        if [h["epoch"] for h in history] != list(range(1, meta["epochs"] + 1)):
            raise ValueError("Incomplete epoch accounting")
        chosen = min(
            [(0, core.validation_score(done["initial_validation"]))]
            + [(h["epoch"], core.validation_score(h["validation"])) for h in history],
            key=lambda x: (x[1], x[0]),
        )[0]
        if chosen != done["best_epoch"]:
            raise ValueError("Best epoch does not match validation-only selection")
        schedule_rows = [
            (
                h["train"]["order_hash"],
                h["train"]["prefix_hash"],
                h["train"]["steps"],
                h["train"]["examples"],
                h["train"]["supervised_states"],
                h["lr"],
            )
            for h in history
        ]
        if job["seed"] in schedules and schedules[job["seed"]] != schedule_rows:
            raise ValueError("Paired budgets/schedules differ")
        schedules[job["seed"]] = schedule_rows
        for name, h in done["files"].items():
            files[f"{job['name']}/{name}"] = h
    if any(len(v) != 1 for v in initial.values()):
        raise ValueError("Paired initialization/parameter counts differ")
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": files,
            "cache_indexes": {s: sha256(out / f"cache/{s}_index.json") for s in ("train", "val")},
        },
        out / "model_selection_lock.json",
    )
    return check_selection(out)


def preflight(meta, out):
    x, _ = data.cache(out, "train")
    stats = parent.stats(meta)[0]
    report = {}
    for mode in core.MODES:
        encoder, query = core.make_models(mode, 42, "cuda", meta["encoder"])
        b = torch.tensor(x[: meta["micro"]], device="cuda")
        ps = torch.tensor(core.prefixes(len(b), 42, 1), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        z, pred = predict(encoder, query, b, ps, True)
        y, mask = core.targets(b, ps, stats, mode == "prefix")
        loss = core.loss_rows(pred, y, mask, stats, True)["objective"].mean()
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in encoder.parameters()
        ):
            raise ValueError("Preflight gradient nonfinite")
        parameters = sum(p.numel() for p in encoder.parameters()) + sum(
            p.numel() for p in query.parameters()
        )
        report[mode] = {
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_extra_bytes": parameters * 8,
            "parameters": parameters,
            "loss": float(loss),
        }
        if mode == "relative":
            from .uniform_context_evaluate import chunk_audit

            report[mode]["chunk_audit"] = chunk_audit(encoder.eval(), b[:2])
            atomic_json(report, out / "preflight.json")
            if not report[mode]["chunk_audit"]["passed"]:
                raise ValueError("GPU preflight chunk/future invariance failed")
        del encoder, query, b, ps, z, pred, y, mask, loss
        gc.collect()
        torch.cuda.empty_cache()
    for seed in meta["seeds"]:
        model, query = parent.construct(meta, seed, "cuda")
        with torch.no_grad():
            z = model.core.encoder(torch.tensor(x[:1, -128:], device="cuda"))[:, -1]
            prediction = query(z)
            if z.shape != (1, 768) or not torch.isfinite(prediction).all():
                raise ValueError("Frozen Macro reference preflight failed")
        del model, query, z, prediction
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(report, out / "preflight.json")
    return report


def launch(meta, out, count):
    queue = jobs(meta)
    active = []
    try:
        while queue or active:
            while queue and len(active) < count:
                job = queue.pop(0)
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        __package__ + ".uniform_context_run",
                        "--out",
                        str(out),
                        "--worker",
                        job["name"],
                    ]
                )
                active.append((job, proc))
            time.sleep(2)
            for job, proc in active[:]:
                if proc.poll() is not None:
                    if proc.returncode:
                        raise RuntimeError(
                            f"Worker {job['name']} failed ({proc.returncode}); rerun same command to resume"
                        )
                    active.remove((job, proc))
    finally:
        for _, proc in active:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in active:
            proc.wait()


def run(source, out, count, micro):
    from . import uniform_context_evaluate as ev

    if count not in (1, 2) or micro < 1 or micro > 128:
        raise ValueError("jobs1..2 and micro1..128 required")
    if not torch.cuda.is_available():
        raise RuntimeError("Real optimization is AutoDL CUDA-only")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    progress("Verifying frozen Macro lineage; three scratch arms, two paired seeds")
    identity = parent.source_identity(source)
    meta = make_manifest(source, identity, micro)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json") != meta:
                raise ValueError("Existing protocol differs; choose a new output directory")
        else:
            atomic_json(meta, out / "manifest.json")
        if (out / "completion.json").exists():
            verify_files(out, read_json(out / "completion.json")["files"])
            return
        statistics = parent.stats(meta)[0]
        if (out / "statistics.json").exists() and read_json(out / "statistics.json") != statistics:
            raise ValueError("Exported source statistics changed")
        atomic_json(statistics, out / "statistics.json")
        training_done = all((out / j["name"] / "completion.json").exists() for j in jobs(meta))
        need = (2 if training_done else 8) * 2**30
        if shutil.disk_usage(out).free < need:
            raise ValueError(
                f"Need {need / 2**30:.0f}GiB free for journal/checkpoints/reports; source is protected"
            )
        atomic_json(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "jobs": count,
                "micro": micro,
                "free_bytes": shutil.disk_usage(out).free,
            },
            out / "environment.json",
        )
        data.prepare(meta, out, ("train", "val"))
        from . import utility_probe as up

        for split in ("train", "val"):
            values, _ = data.cache(out, split)
            y, valid = up.targets(up.restore_raw(values[:, -128:], parent.stats(meta)[0]))
            if (
                (valid.sum(0) < 2).any()
                or not np.isfinite(y).all()
                or (split == "train" and len(values) <= 768)
            ):
                raise ValueError(
                    "Insufficient data for declared PCA/readout controls; stopped before training"
                )
            del values, y, valid
        if not training_done:
            probes = preflight(meta, out)
            required = (
                count
                * max(v["peak_bytes"] + v["optimizer_extra_bytes"] for v in probes.values())
                * 1.25
            )
            free, _ = torch.cuda.mem_get_info()
            if required > free:
                raise ValueError(
                    "Requested concurrency/micro exceeds measured memory budget; reduce jobs or micro using a new output directory"
                )
            launch(meta, out, count)
        lock_selection(meta, out)
        ev.fit(meta, out, "cuda")
        data.prepare(meta, out, ("test",))
        data.prepare(meta, out, ("cross_research",), True)
        ev.evaluate(meta, out, "cuda")
        if parent.source_identity(source) != identity:
            raise ValueError("Frozen source changed during run")
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".npz", ".png", ".html", ".pt")
            and p.name not in ("completion.json", "audit_status.json")
        }
        atomic_json(
            {"status": "complete", "source_unchanged": True, "files": files},
            out / "completion.json",
        )
        atomic_json({"status": "complete"}, out / "audit_status.json")
        progress("Uniform-context study complete; source unchanged, no automatic promotion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("checkpoints/babel_macro_history768"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--worker")
    a = p.parse_args()
    a.out = a.out.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Native eval fastpath has different mask handling; use one consistent attention path.
    torch.backends.mha.set_fastpath_enabled(False)
    if a.worker:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        meta = read_json(a.out / "manifest.json")
        job = next(j for j in jobs(meta) if j["name"] == a.worker)
        worker(a.out, job)
    else:
        try:
            run(a.source.resolve(), a.out, a.jobs, a.micro)
        except Exception as e:
            if a.out.is_dir():
                atomic_json({"status": "failed", "error": repr(e)}, a.out / "audit_status.json")
            raise


if __name__ == "__main__":
    main()
