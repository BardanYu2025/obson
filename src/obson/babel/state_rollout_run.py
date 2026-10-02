"""Finite frozen-state rollout matrix; real optimization runs only on AutoDL CUDA."""

import argparse
import copy
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
from . import state_rollout as core
from . import state_rollout_data as data
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-state-rollout768-v1"


def code_identity():
    from . import state_rollout_evaluate as ev

    return (
        parent.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (core, data, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


def make_manifest(source, identity, batch):
    old = read_json(source / "manifest.json")
    return {
        "schema": SCHEMA,
        "identity": old["identity"],
        "architecture": old["architecture"],
        "task_source": identity,
        "code_sha256": code_identity(),
        "extraction_batch": batch,
        "epochs": 60,
        "batch": 256,
        "seeds": [42, 43],
        "lrs": list(core.LRS),
        "arms": list(core.ARMS),
        "train_horizon": 8,
        "evaluation_horizon": 32,
        "protocol": {
            "source": "Macro control validation best97/94; E and original D frozen; no source promotion",
            "selection": "Free1..8 price normalized MSE; epoch0 included; ties earlier epoch then smaller LR; locks before research extraction",
            "labels": "Independent rolling128 states.128input+32future within original contract/partition; past causal EMA prehistory allowed",
            "primary": "N8 fixed; >=5% price gain vs val-selected simple baseline, <=10% increment MSE and tail95 degradation, both seeds/cohorts with paired monthly intervals",
            "loss": "State increment-scaled MSE; full8-step gradient for N8; TF8 matched labels for L1/N1. Direct normalized price MSE. No encoder gradients.",
            "fairness": "Same origins/update budgets/LR search; unequal parameters/compute. Two paired pretrained encoders, not independent repeated pretraining.",
            "stop": "20 fits x60epochs; no automatic extra training, grid, source switch or promotion; reused research cohorts, not new holdout",
        },
    }


def jobs(meta):
    return [
        {"name": f"{arm}_s{s}_lr{i}", "arm": arm, "seed": s, "lr": lr}
        for s in meta["seeds"]
        for arm in meta["arms"]
        for i, lr in enumerate(meta["lrs"])
    ]


def prepared(bank, fit, seed):
    return {
        "u": core.normalize(bank[f"z{seed}"], fit["states"][str(seed)]),
        "mapping": bank["mapping"],
        "raw": core.normalize(bank["x"].reshape(len(bank["x"]), -1), fit["raw"]),
        "target": bank["targets"],
        "scale": core.price_scale(bank, fit["price"]),
    }


@torch.no_grad()
def forecast(
    model,
    arm,
    prepared_bank,
    fit,
    seed,
    query,
    delta_scale,
    device,
    horizon=8,
    batch=64,
    diagnostics=False,
):
    """Only start state/raw observed input reaches predictor; labels remain outside."""
    p = prepared_bank
    preds, zs, integrated = [], [], []
    scaling = fit["states"][str(seed)]
    for start in range(0, len(p["mapping"]), batch):
        mapping = p["mapping"][start : start + batch]
        initial = torch.as_tensor(p["u"][mapping[:, 0]], device=device)
        if arm in core.ARMS[:3] or arm == "identity":
            u = (
                initial[:, None].expand(-1, horizon, -1)
                if arm == "identity"
                else model.rollout(initial, horizon)
            )
            z = u * u.new_tensor(scaling["scale"]) + u.new_tensor(scaling["mean"])
            pred = core.read_prices(query, z, delta_scale)
            if diagnostics:
                integrated.append(
                    core.read_prices(query, z, delta_scale, [1] * horizon).cumsum(1).cpu().numpy()
                )
                zs.append(u.cpu().numpy())
        else:
            if horizon != core.TRAIN_H:
                raise ValueError(
                    "Direct control supports exactly8 outputs; no invented extrapolation"
                )
            x = (
                torch.as_tensor(p["raw"][start : start + batch], device=device)
                if arm == "Raw-direct"
                else initial
            )
            pred = model(x)
        preds.append(pred.cpu().numpy())
    pred = np.concatenate(preds)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite forecast; run cannot pass")
    result = {"price": pred}
    if diagnostics and zs:
        result.update(states=np.concatenate(zs), integrated=np.concatenate(integrated))
    return result


def binding(out, job):
    return {
        "manifest_sha256": sha256(out / "manifest.json"),
        "job": job,
        "train": sha256(out / "cache/train_index.json"),
        "val": sha256(out / "cache/val_index.json"),
        "transforms": sha256(out / "transforms_lock.json"),
        "resources": sha256(out / "resources.json"),
    }


def fit_job(meta, out, job, train, val, fit, query, delta_scale, device):
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    bind = binding(out, job)
    if (folder / "completion.json").exists():
        done = read_json(folder / "completion.json")
        if done["binding"] != bind:
            raise ValueError("Completed fit binding changed")
        verify_files(folder, done["files"])
        return read_json(folder / "summary.json")
    torch.manual_seed(job["seed"])
    width, raw_width = train["u"].shape[1], train["raw"].shape[1]
    model = core.make_model(job["arm"], width, raw_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=0.01)
    micro = read_json(out / "resources.json")["micro"]
    increment = torch.tensor(fit["states"][str(job["seed"])]["increment"], device=device)
    signature = parent.ur.bb.state_signature(query)
    query.eval().requires_grad_(False)

    def validation():
        model.eval()
        prediction = forecast(model, job["arm"], val, fit, job["seed"], query, delta_scale, device)[
            "price"
        ]
        return float(core.errors(prediction, val["target"], val["scale"])["price"].mean())

    if (folder / "resume.pt").exists():
        state = torch.load(folder / "resume.pt", map_location=device, weights_only=True)
        if state["binding"] != bind:
            raise ValueError("Resume lineage or resource batch changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(state["rng_cuda"].cpu())
    else:
        initial = validation()
        state = {
            "binding": bind,
            "epoch": 0,
            "best_epoch": 0,
            "best_value": initial,
            "initial_validation": initial,
            "best": copy.deepcopy(model.state_dict()),
            "history": [],
            "steps": 0,
            "initial_signature": parent.ur.bb.state_signature(model),
        }
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n = len(train["mapping"])
    for epoch in range(state["epoch"] + 1, meta["epochs"] + 1):
        started = time.monotonic()
        order = np.random.default_rng(job["seed"] * 100000 + epoch).permutation(n)
        total, steps = 0.0, 0
        model.train()
        for left in range(0, n, meta["batch"]):
            ids = order[left : left + meta["batch"]]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(ids), micro):
                ix = ids[offset : offset + micro]
                # TF receives true previous training states; free forecast never does.
                states = torch.as_tensor(train["u"][train["mapping"][ix, :9]], device=device)
                raw = (
                    torch.as_tensor(train["raw"][ix], device=device)
                    if job["arm"] == "Raw-direct"
                    else None
                )
                target = torch.as_tensor(train["target"][ix], device=device, dtype=torch.float32)
                scale = torch.as_tensor(train["scale"][ix], device=device, dtype=torch.float32)
                loss = core.objective(model, states, increment, raw, target, scale)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite optimization loss")
                (loss * (len(ix) / len(ids))).backward()
                total += float(loss.detach()) * len(ix)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        value = validation()
        if value < state["best_value"]:
            state.update(best_value=value, best_epoch=epoch, best=copy.deepcopy(model.state_dict()))
        state["steps"] += steps
        seconds = time.monotonic() - started
        state["history"].append(
            {
                "epoch": epoch,
                "train_loss": total / n,
                "validation": value,
                "samples": n,
                "supervised_targets": n * 8,
                "steps": steps,
                "seconds": seconds,
                "order_sha256": parent.ur.bb.cov.ndarray_hash(order),
            }
        )
        state.update(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state() if device == "cuda" else None,
        )
        atomic_save(state, folder / "resume.pt")
        atomic_json(state["history"], folder / "history.json")
        if epoch <= 2 or epoch % 5 == 0:
            progress(
                f"{job['name']}: {epoch}/{meta['epochs']}, val={value:.6f}, best={state['best_epoch']}, {seconds:.1f}s/epoch, worker remaining~{seconds * (meta['epochs'] - epoch) / 60:.1f}min"
            )
    if parent.ur.bb.state_signature(query) != signature or any(
        p.grad is not None for p in query.parameters()
    ):
        raise ValueError("Frozen decoder mutated or received gradients")
    summary = {
        "job": job,
        "selected_epoch": state["best_epoch"],
        "validation": state["best_value"],
        "initial_validation": state["initial_validation"],
        "epochs": state["epoch"],
        "updates": state["steps"],
        "origins": n,
        "exposures": n * state["epoch"],
        "supervised_targets": n * 8 * state["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "width": width,
        "raw_width": raw_width,
        "micro": micro,
        "initial_signature": state["initial_signature"],
        "final_signature": parent.ur.bb.state_signature(model),
        "epoch0_fallback": state["best_epoch"] == 0,
        "decoder_unchanged": True,
        "seconds": sum(r["seconds"] for r in state["history"]),
        "peak_cuda_allocated": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "transition_applications_per_origin": 8 if job["arm"] in core.ARMS[:3] else 0,
        "note": "Equal labels/updates, not equal parameters or FLOPs. N8 retains8-step graph; TF vectorizes8 one-step pairs.",
    }
    for tag, weights in (("best", state["best"]), ("last", model.state_dict())):
        atomic_save({"binding": bind, "model": weights, "summary": summary}, folder / f"{tag}.pt")
    atomic_json(summary, folder / "summary.json")
    files = {n: sha256(folder / n) for n in ("best.pt", "last.pt", "summary.json", "history.json")}
    atomic_json({"binding": bind, "files": files}, folder / "completion.json")
    return summary


def worker(out, name, device="cuda"):
    meta = read_json(out / "manifest.json")
    if meta["code_sha256"] != code_identity():
        raise ValueError("Worker code changed")
    job = next(j for j in jobs(meta) if j["name"] == name)
    fit = data.transformations(out)
    train, _ = data.cache(out, "train")
    val, _ = data.cache(out, "val")
    tr, va = prepared(train, fit, job["seed"]), prepared(val, fit, job["seed"])
    del train, val
    model, query = parent.construct(meta, job["seed"], device)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    with (out / f".worker_{name}.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fit_job(
            meta, out, job, tr, va, fit, query, parent.stats(meta)[0]["delta_scale"][0], device
        )


def choose(candidates):
    chosen = {}
    for seed in (42, 43):
        for arm in core.ARMS:
            options = [
                v
                for v in candidates.values()
                if v["job"]["seed"] == seed and v["job"]["arm"] == arm
            ]
            best = min(
                options, key=lambda v: (v["validation"], v["selected_epoch"], v["job"]["lr"])
            )
            chosen[f"{arm}_s{seed}"] = best["job"]["name"]
    return chosen


def check_selection(meta, out):
    lock = read_json(out / "selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Selection manifest changed")
    verify_files(out, lock["files"])
    if set(lock["candidates"]) != {j["name"] for j in jobs(meta)}:
        raise ValueError("Incomplete20-fit selection")
    for job in jobs(meta):
        folder = out / job["name"]
        done, summary = read_json(folder / "completion.json"), read_json(folder / "summary.json")
        verify_files(folder, done["files"])
        history = read_json(folder / "history.json")
        if done["binding"] != binding(out, job) or summary != lock["candidates"][job["name"]]:
            raise ValueError("Fit binding or summary changed")
        n = summary["origins"]
        steps = int(np.ceil(n / meta["batch"]))
        if (
            len(history) != meta["epochs"]
            or summary["updates"] != steps * meta["epochs"]
            or summary["epochs"] != meta["epochs"]
        ):
            raise ValueError("Incomplete training budget")
        for epoch, r in enumerate(history, 1):
            expected = np.random.default_rng(job["seed"] * 100000 + epoch).permutation(n)
            if (
                r["epoch"] != epoch
                or r["samples"] != n
                or r["steps"] != steps
                or r["supervised_targets"] != n * 8
                or r["order_sha256"] != parent.ur.bb.cov.ndarray_hash(expected)
            ):
                raise ValueError("Sampling or update budget differs")
        best = min(
            [(0, summary["initial_validation"])] + [(r["epoch"], r["validation"]) for r in history],
            key=lambda v: (v[1], v[0]),
        )
        if best != (summary["selected_epoch"], summary["validation"]):
            raise ValueError("Selected epoch differs")
    if lock["chosen"] != choose(lock["candidates"]):
        raise ValueError("LR selection differs")
    if lock["baseline"] != min(
        lock["baseline_validation"], key=lambda k: (lock["baseline_validation"][k], k)
    ):
        raise ValueError("Baseline selection differs")
    return lock


def fit_all(meta, out):
    if (out / "selection_lock.json").exists():
        return check_selection(meta, out)
    capacity = read_json(out / "resources.json")["jobs"]
    pending, completed = [], []
    for job in jobs(meta):
        folder = out / job["name"]
        if (folder / "completion.json").exists():
            done = read_json(folder / "completion.json")
            if done["binding"] != binding(out, job):
                raise ValueError("Completed worker lineage changed")
            verify_files(folder, done["files"])
            completed.append(job["name"])
        else:
            pending.append(job)
    skipped = len(completed)
    active = []
    started = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < capacity:
                j = pending.pop(0)
                p = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "obson.babel.state_rollout_run",
                        "--out",
                        str(out),
                        "--worker",
                        j["name"],
                    ]
                )
                active.append((p, j))
            remaining = []
            for p, j in active:
                status = p.poll()
                if status is None:
                    remaining.append((p, j))
                elif status:
                    raise RuntimeError(
                        f"Worker {j['name']} failed ({status}); completed fits preserved"
                    )
                else:
                    completed.append(j["name"])
                    elapsed = time.monotonic() - started
                    eta = elapsed / max(1, len(completed) - skipped) * (20 - len(completed))
                    progress(
                        f"Matrix completed {len(completed)}/20; rough training ETA~{eta / 60:.1f}min (cache/evaluation separate)"
                    )
            active = remaining
            if active:
                time.sleep(2)
    finally:
        for p, _ in active:
            if p.poll() is None:
                p.terminate()
        for p, _ in active:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    candidates = {j["name"]: read_json(out / j["name"] / "summary.json") for j in jobs(meta)}
    val, _ = data.cache(out, "val")
    fit = data.transformations(out)
    scale = core.price_scale(val, fit["price"])
    scores = {
        k: float(core.errors(v[:, :8], val["targets"], scale)["price"].mean())
        for k, v in core.baselines(val, fit["price"]).items()
    }
    files = {
        str(p.relative_to(out)): sha256(p)
        for j in jobs(meta)
        for p in (out / j["name"]).iterdir()
        if p.name in ("best.pt", "last.pt", "summary.json", "history.json", "completion.json")
    }
    lock = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "files": files,
        "candidates": candidates,
        "chosen": choose(candidates),
        "baseline_validation": scores,
        "baseline": min(scores, key=lambda k: (scores[k], k)),
    }
    atomic_json(lock, out / "selection_lock.json")
    return check_selection(meta, out)


def disk_budget(meta, out):
    counts = {s: len(read_json(parent.root(meta) / f"{s}_inventory.json")) for s in data.SPLITS}
    # Upper bound: no endpoint dedup, two float32 state banks, no expanded rolling inputs.
    states = sum(n * (9 if s == "train" else 33) * 768 * 4 * 2 for s, n in counts.items())
    origins = sum(counts.values()) * 128 * 28 * 4
    parameters = sum(
        sum(p.numel() for p in core.make_model(j["arm"]).parameters()) for j in jobs(meta)
    )
    checkpoints = parameters * 4 * 7  # best,last, resume model+best+Adam moments, atomic temporary
    reports = 512 * 1024**2
    required = int(2 * (states + origins) + checkpoints + reports + 1024**3)
    free = shutil.disk_usage(out).free
    result = {
        "original_origins": counts,
        "state_upper_bytes": states,
        "origin_input_bytes": origins,
        "checkpoint_upper_bytes": checkpoints,
        "report_allowance_bytes": reports,
        "additional_free_required_bytes": required,
        "free_bytes": free,
        "note": "Conservative no-dedup upper bound plus atomic writes/export headroom; no source deletion",
    }
    if free < required:
        raise ValueError(
            f"Need {required / 1024**3:.2f}GiB free; have {free / 1024**3:.2f}GiB; source unchanged"
        )
    return result


def preflight(meta, out, requested_jobs, device):
    if (out / "resources.json").exists():
        return read_json(out / "resources.json")
    disk = disk_budget(meta, out)
    model, query = parent.construct(meta, 42, device)
    model.eval().requires_grad_(False)
    query.eval().requires_grad_(False)
    original = parent.data(meta, "train")["x"][: meta["extraction_batch"]]
    with torch.no_grad():
        z = model.core.encoder(torch.as_tensor(np.array(original), device=device))[:, -1]
        core.read_prices(
            query, z[:, None].expand(-1, 32, -1), parent.stats(meta)[0]["delta_scale"][0]
        )
    del model, z
    gc.collect()
    torch.cuda.empty_cache()
    micro = 128
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            transition = core.Transition("N8").to(device)
            states = torch.zeros(micro, 9, 768, device=device)
            core.objective(transition, states, torch.ones(768, device=device)).backward()
            peak = torch.cuda.max_memory_allocated()
            break
        except torch.OutOfMemoryError:
            if micro <= 8:
                raise
            micro //= 2
        finally:
            if "transition" in locals():
                del transition
            if "states" in locals():
                del states
            gc.collect()
            torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    selected_jobs = min(requested_jobs, 2) if free > 2 * (peak + 1024**3) else 1
    resources = {
        "jobs": selected_jobs,
        "requested_jobs": requested_jobs,
        "micro": micro,
        "peak_preflight_bytes": peak,
        "disk": disk,
        "optimizer_steps": 0,
        "extraction_batch": meta["extraction_batch"],
    }
    atomic_json(resources, out / "resources.json")
    progress(
        f"Preflight passed: jobs={selected_jobs}, micro={micro}, batch256; zero optimization updates"
    )
    return resources


def run(source, out, batch=16, requested_jobs=1):
    if not torch.cuda.is_available():
        raise ValueError("Real run requires AutoDL CUDA; no local training")
    if batch < 1 or requested_jobs not in (1, 2):
        raise ValueError("Positive extraction batch and jobs1/2 required")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress("Verifying immutable Macro control97/94 and original data lineage")
        identity = parent.source_identity(source)
        meta = make_manifest(source, identity, batch)
        meta["runtime"] = {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name(),
        }
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json") != meta:
                raise ValueError("Manifest differs; use a new output directory")
        else:
            if any(p.name != ".run.lock" for p in out.iterdir()):
                raise ValueError("Nonempty output without manifest")
            atomic_json(meta, out / "manifest.json")
        if (out / "completion.json").exists():
            verify_files(out, read_json(out / "completion.json")["files"])
            progress("Completed run verified; nothing retrained")
            return
        tm = parent.tm(meta)
        if (
            parent.xr.xd.audit_identity(Path(tm["raw_audit"]), Path(tm["raw_root"]))
            != tm["raw_audit_identity"]
        ):
            raise ValueError("Raw source lineage differs")
        preflight(meta, out, requested_jobs, "cuda")
        data.prepare(meta, out, ("train", "val"), False, "cuda")
        data.transformations(out)
        from . import state_rollout_evaluate as ev

        ev.pretrain_audit(meta, out, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        fit_all(meta, out)
        data.prepare(meta, out, ("test",), False, "cuda")
        data.prepare(meta, out, ("cross_research",), True, "cuda")
        ev.evaluate(meta, out, "cuda")
        if parent.source_identity(source) != identity:
            raise ValueError("Frozen source mutated")
        if (
            parent.xr.xd.audit_identity(Path(tm["raw_audit"]), Path(tm["raw_root"]))
            != tm["raw_audit_identity"]
        ):
            raise ValueError("Raw data changed during run")
        check_selection(meta, out)
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.name
            not in ("completion.json", "resume.pt", "run_status.txt", "audit_status.json")
            and not p.name.startswith(".")
        }
        atomic_json(
            {
                "status": "complete",
                "source_unchanged": True,
                "encoder_updates": 0,
                "fits": 20,
                "files": files,
            },
            out / "completion.json",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("checkpoints/babel_macro_history768"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--worker")
    a = parser.parse_args()
    try:
        if a.worker:
            worker(a.out, a.worker)
        else:
            run(a.source, a.out, a.batch, a.jobs)
    except Exception as error:
        if (
            not a.worker
            and not isinstance(error, BlockingIOError)
            and (a.out / "manifest.json").exists()
            and not (a.out / "completion.json").exists()
        ):
            atomic_json({"status": "failed", "error": str(error)}, a.out / "audit_status.json")
        raise


if __name__ == "__main__":
    main()
