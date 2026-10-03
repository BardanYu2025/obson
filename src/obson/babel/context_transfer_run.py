"""Same-source warm study: matched updates and bounded matched training wall time."""

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

from . import context_transfer as core
from . import context_transfer_data as data
from . import linear_history_run as parent
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-context-transfer768-v1"


def code_identity():
    from . import context_transfer_evaluate as ev

    return (
        data.previous.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (core, data, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


def jobs(meta):
    # Serial same-GPU timing; rolling must finish before its paired prefix worker.
    return [
        {"name": f"{mode}_s{s}", "mode": mode, "seed": s}
        for s in meta["seeds"]
        for mode in ("rolling", "prefix")
    ]


def make_manifest(source, identity, cache_source, micro):
    old = read_json(source / "manifest.json")
    return {
        "schema": SCHEMA,
        "identity": old["identity"],
        "architecture": old["architecture"],
        "task_source": identity,
        "cache_source": cache_source,
        "code_sha256": code_identity(),
        "seeds": [42, 43],
        "base_epochs": 60,
        "prefix_cap": 300,
        "batch": 128,
        "micro": micro,
        "encoder_lr": 3e-5,
        "query_lr": 3e-4,
        "validation_prefixes": list(core.PREFIXES),
        "protocol": {
            "source": "Macro control97/94, original trained input adapter, global encoder and query reader; fresh matched AdamW; no scratch/Uniform neural weights",
            "objective": "Same sampled five physical endpoints, query0.4path+0.2change1+0.1body+0.3activity plus structure1. Historical targets only; old detached local diagnostic omitted uniformly",
            "arms": "Rolling60, prefix first60 on the same trajectory, prefix extended until paired rolling actual training seconds reached (min60 max300). Source frozen reference",
            "timing": "One worker, synchronized training-stage wall clock; CPU feed/targets included, validation and saving excluded. Budget independent of scores; allow5% overshoot, otherwise comparison inconclusive. Not FLOPs",
            "selection": "Protect source full/native end/interior price, age bands, activity and structure within10%; minimize full-context end/interior price minimax; epoch0 fallback. Locks before research",
            "fairness": "Paired common sampling, first60 identical schedules/updates/LRs; constant LR throughout makes prefix60 an exact trajectory prefix of extended control. Old Macro multi-view curriculum not reproduced",
            "stop": "Finite60 rolling and at most300 prefix epochs per seed. No LR/depth sweep or automatic promotion; shared time cap may yield unmatched control, never silently call it matched",
            "future": "Sequential real-bar refresh and free future rollout remain distinct planned branches; neither trained/scored in this historical representation study",
        },
    }


def construct(meta, seed, device):
    model, query = parent.construct(meta, seed, device)
    encoder = model.core.encoder
    encoder.eval().requires_grad_(True)
    query.eval().requires_grad_(True)
    return encoder, query


def binding(out, job):
    return {"manifest_sha256": sha256(out / "manifest.json"), "job": job}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def validation(meta, encoder, query, x, device):
    encoder.eval()
    query.eval()
    stats = parent.stats(meta)[0]
    values = {}
    for start in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[start : start + meta["micro"]], device=device)
        ps = torch.tensor(meta["validation_prefixes"], device=device).expand(len(b), -1)
        for view, native in (("full", False), ("native", True)):
            _, pred = core.predict(encoder, query, b, ps, native)
            y, mask = core.original.targets(b, ps, stats, native)
            for k, v in core.original.band_rows(pred, y, mask, stats).items():
                if k not in core.METRICS:
                    continue
                for context, index in (
                    ("endpoint", slice(-1, None)),
                    ("interior", slice(None, -1)),
                ):
                    values.setdefault(f"{view}/{context}/{k}", []).append(
                        v[:, index].mean(1).cpu().numpy()
                    )
    return {k: float(np.concatenate(v).mean()) for k, v in values.items()}


def train_epoch(meta, job, encoder, query, x, opt, epoch, device):
    encoder.eval()
    query.eval()
    stats = parent.stats(meta)[0]
    order, ps = core.schedule(len(x), job["seed"], epoch)
    sync(device)
    start_time = time.perf_counter()
    total = 0.0
    steps = 0
    for left in range(0, len(x), meta["batch"]):
        size = min(meta["batch"], len(x) - left)
        opt.zero_grad(set_to_none=True)
        for start in range(left, left + size, meta["micro"]):
            stop = min(start + meta["micro"], left + size)
            b = torch.as_tensor(x[order[start:stop]], device=device)
            p = torch.as_tensor(ps[start:stop], device=device)
            native = job["mode"] == "prefix"
            _, pred = core.predict(encoder, query, b, p, native)
            y, mask = core.original.targets(b, p, stats, native)
            values = core.original.loss_rows(pred, y, mask, stats, True)["objective"].mean(1)
            loss = values.sum() / size
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite warm reconstruction loss")
            loss.backward()
            total += float(values.detach().sum())
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(query.parameters()), 1.0, error_if_nonfinite=True
        )
        opt.step()
        steps += 1
    sync(device)
    return {
        "loss": total / len(x),
        "seconds": time.perf_counter() - start_time,
        "steps": steps,
        "examples": len(x),
        "supervised_states": 5 * len(x),
        "order_hash": parent.ur.bb.cov.ndarray_hash(order),
        "prefix_hash": parent.ur.bb.cov.ndarray_hash(ps),
    }


def snapshot(ck, best=False):
    return {
        "encoder": ck["best_encoder"] if best else ck["encoder"],
        "query": ck["best_query"] if best else ck["query"],
        "epoch": ck["best_epoch"] if best else ck["epoch"],
        "metadata": ck["metadata"],
        "validation": ck["best_validation"] if best else ck["history"][-1]["validation"],
    }


def materialize(ck, folder, refresh=False):
    if "update_best" in ck:
        for key in ("update_best", "update_last"):
            if refresh or not (folder / (key + ".pt")).exists():
                atomic_save(ck[key], folder / (key + ".pt"))
    if ck["metadata"]["job"]["mode"] == "prefix":
        atomic_save(snapshot(ck, True), folder / "time_best.pt")
        atomic_save(snapshot(ck), folder / "time_last.pt")
    atomic_json(ck["history"], folder / "history.json")


def target_time(meta, out, job):
    if job["mode"] == "rolling":
        return None
    folder = out / f"rolling_s{job['seed']}"
    done = read_json(folder / "completion.json")
    if (
        done["metadata"]
        != binding(out, {"name": folder.name, "mode": "rolling", "seed": job["seed"]})
        or done["epochs"] != meta["base_epochs"]
    ):
        raise ValueError("Completed paired rolling baseline required")
    verify_files(folder, done["files"])
    return done["training_seconds"]


def worker(out, job):
    meta = read_json(out / "manifest.json")
    if meta["code_sha256"] != code_identity():
        raise ValueError("Worker code identity changed")
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "completion.json").exists():
            done = read_json(folder / "completion.json")
            if done["metadata"] != binding(out, job):
                raise ValueError("Worker protocol changed")
            verify_files(folder, done["files"])
            return
        target = target_time(meta, out, job)
        train, _ = data.cache(out, "train")
        val, _ = data.cache(out, "val")
        device = "cuda"
        encoder, query = construct(meta, job["seed"], device)
        opt = torch.optim.AdamW(
            [
                {"params": encoder.parameters(), "lr": meta["encoder_lr"], "weight_decay": 0.01},
                {"params": query.parameters(), "lr": meta["query_lr"], "weight_decay": 1e-4},
            ]
        )
        if (folder / "last.pt").exists():
            ck = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
            if ck["metadata"] != binding(out, job) or ck["target_seconds"] != target:
                raise ValueError("Checkpoint/budget binding changed")
            encoder.load_state_dict(ck["encoder"], strict=True)
            query.load_state_dict(ck["query"], strict=True)
            opt.load_state_dict(ck["optimizer"])
            materialize(ck, folder, refresh=True)
        else:
            initial = validation(meta, encoder, query, val, device)
            ck = {
                "epoch": 0,
                "metadata": binding(out, job),
                "initial_validation": initial,
                "best_epoch": 0,
                "best_score": 1.0,
                "best_validation": initial,
                "best_encoder": cpu_state(encoder),
                "best_query": cpu_state(query),
                "initial_encoder_hash": parent.ur.bb.state_signature(encoder),
                "initial_query_hash": parent.ur.bb.state_signature(query),
                "parameters": sum(p.numel() for p in encoder.parameters())
                + sum(p.numel() for p in query.parameters()),
                "history": [],
                "training_seconds": 0.0,
                "target_seconds": target,
            }
        while not core.stop(
            job["mode"],
            ck["epoch"],
            ck["training_seconds"],
            target,
            meta["base_epochs"],
            meta["prefix_cap"],
        ):
            epoch = ck["epoch"] + 1
            start = time.monotonic()
            training = train_epoch(meta, job, encoder, query, train, opt, epoch, device)
            scores = validation(meta, encoder, query, val, device)
            selection = core.score(scores, ck["initial_validation"])
            if selection["eligible"] and selection["score"] < ck["best_score"]:
                ck.update(
                    best_epoch=epoch,
                    best_score=selection["score"],
                    best_validation=scores,
                    best_encoder=cpu_state(encoder),
                    best_query=cpu_state(query),
                )
            ck["history"].append(
                {
                    "epoch": epoch,
                    "train": training,
                    "validation": scores,
                    "selection": selection,
                    "best_epoch": ck["best_epoch"],
                    "epoch_seconds": time.monotonic() - start,
                    "encoder_lr": meta["encoder_lr"],
                    "query_lr": meta["query_lr"],
                }
            )
            ck.update(
                epoch=epoch,
                encoder=cpu_state(encoder),
                query=cpu_state(query),
                optimizer=opt.state_dict(),
                training_seconds=ck["training_seconds"] + training["seconds"],
            )
            if epoch == meta["base_epochs"]:
                ck["update_best"] = snapshot(ck, True)
                ck["update_last"] = snapshot(ck)
            # Complete journal is committed before any derived snapshot; handles boundary60 interruption.
            atomic_save(ck, folder / "last.pt")
            materialize(ck, folder)
            budget = (
                f"{ck['training_seconds'] / 60:.1f}/{target / 60:.1f}min"
                if target
                else f"{epoch}/{meta['base_epochs']}epochs"
            )
            progress(
                f"{job['name']}: epoch={epoch}, protected_best={ck['best_epoch']}, eligible={selection['eligible']}, score={selection['score']:.6f}, training_budget={budget}, epoch={ck['history'][-1]['epoch_seconds']:.1f}s"
            )
        names = ["update_best.pt", "update_last.pt", "history.json", "last.pt"] + (
            ["time_best.pt", "time_last.pt"] if job["mode"] == "prefix" else []
        )
        atomic_json(
            {
                "metadata": binding(out, job),
                "epochs": ck["epoch"],
                "best_epoch": ck["best_epoch"],
                "training_seconds": ck["training_seconds"],
                "target_seconds": target,
                "time_match": core.time_match(ck["training_seconds"], target) if target else None,
                "parameters": ck["parameters"],
                "initial_encoder_hash": ck["initial_encoder_hash"],
                "initial_query_hash": ck["initial_query_hash"],
                "initial_validation": ck["initial_validation"],
                "files": {n: sha256(folder / n) for n in names},
            },
            folder / "completion.json",
        )


def check_selection(out):
    return data.previous.check_selection(out)


def lock_selection(meta, out):
    if (out / "model_selection_lock.json").exists():
        return check_selection(out)
    train, _ = data.cache(out, "train")
    n = len(train)
    files = {}
    paired = {}
    timing = {}
    for job in jobs(meta):
        folder = out / job["name"]
        done = read_json(folder / "completion.json")
        history = read_json(folder / "history.json")
        if done["metadata"] != binding(out, job) or [h["epoch"] for h in history] != list(
            range(1, done["epochs"] + 1)
        ):
            raise ValueError("Worker accounting differs")
        verify_files(folder, done["files"])
        if not core.stop(
            job["mode"],
            done["epochs"],
            done["training_seconds"],
            done["target_seconds"],
            meta["base_epochs"],
            meta["prefix_cap"],
        ):
            raise ValueError("Budget not completed")
        seconds = 0.0
        best = 0
        score = 1.0
        for h in history:
            if core.stop(
                job["mode"],
                h["epoch"] - 1,
                seconds,
                done["target_seconds"],
                meta["base_epochs"],
                meta["prefix_cap"],
            ):
                raise ValueError("Training continued beyond predeclared budget")
            order, ps = core.schedule(n, job["seed"], h["epoch"])
            expected = {
                "order_hash": parent.ur.bb.cov.ndarray_hash(order),
                "prefix_hash": parent.ur.bb.cov.ndarray_hash(ps),
                "steps": (n + meta["batch"] - 1) // meta["batch"],
                "examples": n,
                "supervised_states": 5 * n,
            }
            if any(h["train"][k] != v for k, v in expected.items()) or any(
                h[k] != meta[k] for k in ("encoder_lr", "query_lr")
            ):
                raise ValueError("Predeclared sampling/update schedule differs")
            if not np.isfinite(h["train"]["seconds"]) or h["train"]["seconds"] <= 0:
                raise ValueError("Invalid measured training time")
            seconds += h["train"]["seconds"]
            choice = core.score(h["validation"], done["initial_validation"])
            if choice != h["selection"]:
                raise ValueError("Validation protection changed")
            if choice["eligible"] and choice["score"] < score:
                best, score = h["epoch"], choice["score"]
            if best != h["best_epoch"]:
                raise ValueError("Wrong validation-selected epoch")
        if not np.isclose(seconds, done["training_seconds"], rtol=1e-12):
            raise ValueError("Timing accounting differs")
        sig = (done["initial_encoder_hash"], done["initial_query_hash"], done["parameters"])
        schedule = [
            tuple(
                h["train"][k]
                for k in ("order_hash", "prefix_hash", "steps", "examples", "supervised_states")
            )
            for h in history[: meta["base_epochs"]]
        ]
        if job["seed"] in paired and paired[job["seed"]] != (sig, schedule):
            raise ValueError("Initialization/update matching failed")
        paired[job["seed"]] = (sig, schedule)
        if job["mode"] == "prefix":
            expected_target = target_time(meta, out, job)
            if done["target_seconds"] != expected_target or done["time_match"] != core.time_match(
                done["training_seconds"], expected_target
            ):
                raise ValueError("Paired measured-time budget differs")
            timing[str(job["seed"])] = done["time_match"]
        files.update({f"{job['name']}/{n}": h for n, h in done["files"].items()})
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": files,
            "time_controls": timing,
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
        e, q = construct(meta, 42, "cuda")
        b = torch.tensor(x[: meta["micro"]], device="cuda")
        ps = torch.tensor(core.original.prefixes(len(b), 42, 1), device="cuda")
        torch.cuda.reset_peak_memory_stats()
        z, p = core.predict(e, q, b, ps, mode == "prefix")
        y, m = core.original.targets(b, ps, stats, mode == "prefix")
        loss = core.original.loss_rows(p, y, m, stats, True)["objective"].mean()
        loss.backward()
        if not torch.isfinite(loss) or any(
            v.grad is not None and not torch.isfinite(v.grad).all()
            for v in list(e.parameters()) + list(q.parameters())
        ):
            raise ValueError("Preflight nonfinite gradient")
        n = sum(v.numel() for v in e.parameters()) + sum(v.numel() for v in q.parameters())
        report[mode] = {
            "loss": float(loss.detach()),
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_bytes": n * 8,
            "parameters": n,
        }
        with torch.no_grad():
            expected = e(b[:, -128:])[:, -1]
            actual = core.states(e, b, torch.full((len(b), 1), 128, device="cuda"))[:, 0]
            if not torch.allclose(actual, expected, atol=1e-4, rtol=2e-4):
                raise ValueError("Original source endpoint replay differs")
        del e, q, b, ps, z, p, y, m, loss, expected, actual
        gc.collect()
        torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    required = max(v["peak_bytes"] + v["optimizer_bytes"] for v in report.values()) * 1.3
    # Check both trained seeds before launching the first long-running trajectory.
    for seed in meta["seeds"]:
        e, q = construct(meta, seed, "cuda")
        with torch.no_grad():
            b = torch.tensor(x[:1], device="cuda")
            _, pred = core.predict(e, q, b, torch.tensor([[128]], device="cuda"))
            if not torch.isfinite(pred).all():
                raise ValueError(f"Source seed{seed} has nonfinite endpoint prediction")
        del e, q, b, pred
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(report, out / "preflight.json")
    if required > free:
        raise ValueError(
            "Insufficient measured GPU memory; use smaller micro in a new output directory"
        )


def disk_reserve(meta, out, trained=False):
    """10GiB fresh estimate; resume credits only known allocated experiment artifacts."""
    headroom = 2 * 2**30
    if trained:
        return headroom
    names = {"last.pt", "update_best.pt", "update_last.pt", "time_best.pt", "time_last.pt"}
    paths = [out / job["name"] / name for job in jobs(meta) for name in names]
    paths += [out / "cache" / (split + ".npz") for split in data.SPLITS]
    allocated = sum(p.stat().st_size for p in paths if p.is_file() and not p.is_symlink())
    return headroom + max(0, 8 * 2**30 - allocated)


def run(source, cache_source, out, micro):
    from . import context_transfer_evaluate as ev

    if not torch.cuda.is_available():
        raise RuntimeError("Real training is AutoDL CUDA-only")
    if not 1 <= micro <= 128:
        raise ValueError("micro1..128 required")
    identity = parent.source_identity(source)
    cached = data.source_identity(cache_source, identity)
    meta = make_manifest(source, identity, cached, micro)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json") != meta:
                raise ValueError("Protocol changed; choose new output directory")
        else:
            atomic_json(meta, out / "manifest.json")
        if (out / "completion.json").exists():
            verify_files(out, read_json(out / "completion.json")["files"])
            return
        trained = all((out / j["name"] / "completion.json").exists() for j in jobs(meta))
        required = disk_reserve(meta, out, trained)
        free = shutil.disk_usage(out).free
        if free < required:
            raise ValueError(
                f"Need {required / 2**30:.1f}GiB free for remaining journals/cache and atomic-save headroom; "
                f"have {free / 2**30:.1f}GiB. Source files protected"
            )
        stats = parent.stats(meta)[0]
        if (out / "statistics.json").exists() and read_json(out / "statistics.json") != stats:
            raise ValueError("Frozen source statistics changed")
        atomic_json(stats, out / "statistics.json")
        atomic_json(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "micro": micro,
                "workers": 1,
            },
            out / "environment.json",
        )
        data.prepare(meta, out, ("train", "val"))
        if not trained:
            preflight(meta, out)
            for job in jobs(meta):
                progress(f"Starting {job['name']} (serial GPU timing control)")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        __package__ + ".context_transfer_run",
                        "--out",
                        str(out),
                        "--worker",
                        job["name"],
                    ],
                    check=True,
                )
        lock_selection(meta, out)
        ev.fit(meta, out, "cuda")
        data.prepare(meta, out, ("test", "cross_research"))
        ev.evaluate(meta, out, "cuda")
        if (
            parent.source_identity(source) != identity
            or data.source_identity(cache_source, identity) != cached
        ):
            raise ValueError("Immutable input/source lineage changed")
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
        progress("Context transfer complete; reports ready; no automatic promotion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("checkpoints/babel_macro_history768"))
    p.add_argument(
        "--cache-source", type=Path, default=Path("checkpoints/babel_uniform_context768")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--worker")
    a = p.parse_args()
    a.out = a.out.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    if a.worker:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        worker(
            a.out,
            next(j for j in jobs(read_json(a.out / "manifest.json")) if j["name"] == a.worker),
        )
    else:
        try:
            run(a.source.resolve(), a.cache_source.resolve(), a.out, a.micro)
        except Exception as e:
            if a.out.is_dir():
                atomic_json({"status": "failed", "error": repr(e)}, a.out / "audit_status.json")
            raise


if __name__ == "__main__":
    main()
