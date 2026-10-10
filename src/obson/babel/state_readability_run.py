"""F11 frozen readout-route diagnostic. No optimizer, fitting or checkpoint selection."""

import argparse
import fcntl
import gc
import math
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import state_coverage as core
from . import state_coverage_data as data
from . import state_coverage_evaluate as ev
from . import state_coverage_run as source_run
from . import state_readability as study
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-state-readability768-v1"
COMPLETION = "eadd412ea5534a966d35094fb10d3a3dda61089d775991b505828693cd0388f5"
SESSION_CAP = 7200
BATCH = 64
SPLITS = {"val": 978, "test": 984, "cross_research": 439}
r0 = data.source.r0


def protocol():
    return {
        "schema": SCHEMA,
        "task_ids": ["F11"],
        "source_completion": COMPLETION,
        "models": ev.variants(),
        "splits": SPLITS,
        "position": 128,
        "optimizer_updates": 0,
        "readout_fits": 0,
        "query_endpoints": 24010,
        "endpoint_budget_meaning": "unique planned endpoints; uncommitted inference may repeat within cumulative time cap",
        "encoder_replays": 60,
        "query_batch": BATCH,
        "session_seconds": SESSION_CAP,
        "primary": "D existing decoder versus same D existing Ridge: utility gain5%, families retain10%; both seeds/cohorts/groupings",
        "recoverability": "four R2>=0.5; utility gain5% versus train mean and within-contract half-rotation",
        "reference": "D versus A decoded gain5%/families10%; versus Macro decoded retain5%/families10%",
        "inference": "2000 bootstrap draws seed20261009;>=100 parents/6groups; exploratory repeated cohorts",
        "no_amplitude_calibration": True,
        "no_new_selection": True,
        "old_exit_qualification_unchanged": True,
        "promoted": False,
    }


def implementation():
    return {
        "inherited": source_run.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__) for m in (study, sys.modules[__name__])
        },
        "script": sha256(source_run.lr.REPO / "scripts/babel_state_readability768_autodl.sh"),
    }


def verify_source(root, out):
    if sha256(root / "completion.json") != COMPLETION:
        raise ValueError("Expected reviewed state-coverage completion; no source substitution")
    done = read_json(root / "completion.json")
    data.source.old.verify_files(root, done["files"])
    meta = read_json(root / "manifest.json")
    if (
        meta["implementation"] != source_run.implementation()
        or meta["protocol"] != source_run.protocol()
    ):
        raise ValueError("Bound historical implementation or protocol changed")
    verified = data.verify(Path(meta["source"]), out)
    if any(meta[k] != v for k, v in verified.items()):
        raise ValueError("Source lineage/scalers/indexes differ")
    ev.check_readouts(root)
    return meta


def cache_states(root, meta, split):
    folder = root / "cache"
    receipt = read_json(folder / f"{split}_states_index.json")
    binding = {
        "model_lock_sha256": sha256(root / "model_selection_lock.json"),
        "input_index_sha256": meta["source_indexes"][split],
    }
    if receipt["binding"] != binding:
        raise ValueError("State cache binding mismatch")
    data.source.old.verify_files(folder, receipt["files"])
    with np.load(folder / f"{split}_states.npz", allow_pickle=False) as f:
        bank = {name: f[name] for name in ev.variants()}
    return bank


def locked_prediction(root, split):
    name = "validation_predictions.npz" if split == "val" else f"{split}_predictions.npz"
    with np.load(root / name, allow_pickle=False) as f:
        return dict(f)


def direct_prediction(fit, weights, name, z):
    return ev.up.predict(
        fit["heads"][name],
        weights[name + "/weights"],
        weights[name + "/intercepts"],
        z,
        fit["target_stats"],
    )


@torch.no_grad()
def decode(e, q, x, z, statistics, device):
    """Fresh encoder sampling is a cache check; primary path consumes cached states only."""
    if any(p.requires_grad for m in (e, q) for p in m.parameters()) or e.training or q.training:
        raise ValueError("Both existing models must be eval and frozen")
    before = source_run.signature(e, q)
    b = torch.as_tensor(x[:2], device=device)
    ps = torch.full((len(b), 1), 128, device=device)
    fresh = core.read_states(e, b, ps)[:, 0].cpu().numpy()
    checks = {"sampled_encoder_state": r0.compare(fresh, z[:2], r0.STATE_ATOL, r0.STATE_RTOL)}
    predictions, errors = [], {}
    for left in range(0, len(x), BATCH):
        state = torch.as_tensor(z[left : left + BATCH], device=device)
        pred = q(state)
        predictions.append(pred.cpu().numpy())
        b = torch.as_tensor(x[left : left + BATCH], device=device)
        ps = torch.full((len(b), 1), 128, device=device)
        y, mask = core.targets(b, ps, statistics)
        for metric, values in core.band_rows(pred[:, None], y, mask, statistics).items():
            errors.setdefault(metric, []).append(values[:, 0].cpu().numpy())
    pred = np.concatenate(predictions)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite decoder output")
    if before != source_run.signature(e, q):
        raise ValueError("Frozen model tensors changed")
    return pred, {k: np.concatenate(v) for k, v in errors.items()}, checks


def replay_model(root, out, meta, split, name, x, z, fit, weights, locked, device):
    receipt = out / f"{split}_{name}.json"
    path = out / f"{split}_{name}.npz"
    binding = {"manifest_sha256": sha256(out / "manifest.json"), "split": split, "name": name}
    if receipt.exists():
        saved = read_json(receipt)
        if saved["binding"] != binding or sha256(path) != saved["sha256"]:
            raise ValueError("Committed result changed")
        with np.load(path, allow_pickle=False) as f:
            return dict(f)
    direct = direct_prediction(fit, weights, name, z)
    checks = {
        "fixed_Ridge_replay": r0.compare(
            direct, locked[name if split == "val" else name + "/predictions"]
        )
    }
    # Fail cheaply before any real forward if a selected readout changed.
    r0.save_checks(out, f"{split}_{name}_direct", checks)
    e, q = ev.load(meta, root, name, device)
    journal = out / "forward_attempts.json"
    attempts = read_json(journal) if journal.exists() else []
    attempts.append(
        {
            "split": split,
            "model": name,
            "reserved_query_endpoints": len(x),
            "reserved_encoder_replays": min(2, len(x)),
            "started_unix": time.time(),
        }
    )
    atomic_json(attempts, journal)
    pred, errors, neural = decode(e, q, x, z, meta["statistics"], device)
    checks.update(neural)
    if split != "val":
        with np.load(root / f"{split}_examples.npz", allow_pickle=False) as f:
            checks["original_query_examples"] = r0.compare(
                pred[:2], f[name + "/prediction"], r0.STATE_ATOL, r0.STATE_RTOL
            )
        summary = read_json(root / f"{split}_reconstruction.json")[name]
        for k, v in errors.items():
            checks["endpoint/" + k] = r0.compare(float(v.mean()), summary[k]["endpoint"])
    r0.save_checks(
        out,
        f"{split}_{name}_qualification",
        checks,
        frozen_tensors_unchanged=True,
        optimizer_updates=0,
        readout_fits=0,
    )
    path_values = study.close_path(pred, meta["statistics"])
    bank = {
        "query": pred,
        "path": path_values,
        "decoded": study.descriptors(path_values),
        "direct": direct[:, :6],
        **{"error/" + k: v for k, v in errors.items()},
    }
    ev.save_arrays(path, bank)
    atomic_json({"binding": binding, "sha256": sha256(path)}, receipt)
    del e, q
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return bank


def evaluate_split(root, out, meta, split, device):
    x, rows = data.cache(meta, root, split)
    if len(rows) != SPLITS[split] or rows != read_json(root / f"{split}_rows.json"):
        raise ValueError("Reviewed row inventory changed")
    atomic_json(rows, out / f"{split}_rows.json")
    fit = read_json(root / "readout_fit.json")
    with np.load(root / "readout_weights.npz", allow_pickle=False) as f:
        weights = dict(f)
    locked = locked_prediction(root, split)
    raw = ev.up.restore_raw(x[:, -128:], meta["statistics"])
    truth, mask = ev.up.targets(raw)
    true_path = study.true_path(raw)
    r0.save_checks(
        out,
        f"{split}_targets",
        {
            "original_targets": r0.compare(truth, locked["targets"]),
            "original_mask": {"passed": bool(np.array_equal(mask, locked["mask"]))},
            "independent_close_descriptors": r0.compare(
                study.descriptors(true_path), truth[:, :6], 1e-9, 1e-8
            ),
        },
    )
    truth = truth[:, :6]
    target = fit["target_stats"]
    predictions = {
        "train_mean": np.broadcast_to(np.asarray(target["mean"])[:6], truth.shape),
        "flat": study.descriptors(np.zeros((len(x), 128))),
    }
    for name in ("raw3584", "pca768", "current28"):
        predictions[name] = locked[name if split == "val" else name + "/predictions"][:, :6]
    bank = cache_states(root, meta, split)
    decoded = {}
    for name in ev.variants():
        z = bank[name]
        if z.shape != (len(rows), 768) or not np.isfinite(z).all():
            raise ValueError("Invalid frozen state cache")
        part = replay_model(root, out, meta, split, name, x, z, fit, weights, locked, device)
        for route in ("direct", "decoded"):
            predictions[name + "/" + route] = part[route]
        decoded[name] = part["decoded"]
        progress(f"F11 {split}/{name}: {len(rows)} fixed endpoints; zero updates, zero fits")
    scores, errors = {}, {}
    for name, pred in predictions.items():
        scores[name], errors[name] = study.measure(pred, truth, target)
    indices = study.mismatch_indices(rows)
    ev.save_arrays(
        out / f"{split}_predictions.npz",
        {"targets": truth, "true_path": true_path, "mismatch_indices": indices, **predictions},
    )
    atomic_json(scores, out / f"{split}_scores.json")
    if split == "val":
        return {"role": "qualification_and_descriptive_only_no_selection", "windows": len(rows)}
    decision = study.decide(
        scores, errors, rows, indices, {"target": target, "decoded": decoded}, truth
    )
    atomic_json(decision, out / f"{split}_decision.json")
    return decision


def execute(root, out):
    state = {
        "schema": SCHEMA,
        "task_ids": ["F11"],
        "status": "running",
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
    }
    atomic_json(state, out / "status.json")
    try:
        progress(
            "F11: checking reviewed source, fixed models/readouts and input lineage; zero training"
        )
        if not torch.cuda.is_available():
            raise ValueError("Real execution requires AutoDL CUDA")
        meta = verify_source(root, out)
        spec = {
            "source": str(root),
            "protocol": protocol(),
            "implementation": implementation(),
            "source_manifest_sha256": sha256(root / "manifest.json"),
            "model_lock_sha256": sha256(root / "model_selection_lock.json"),
            "readout_lock_sha256": sha256(root / "readout_lock.json"),
            "statistics": meta["statistics"],
            "target_stats": read_json(root / "readout_fit.json")["target_stats"],
            "input_index_sha256": meta["source_indexes"],
        }
        manifest = out / "manifest.json"
        if manifest.exists() and read_json(manifest) != spec:
            raise ValueError("Frozen protocol/implementation changed; no overwrite")
        atomic_json(spec, manifest)
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            data.source.old.verify_files(out, done["files"])
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise ValueError("Need 2GiB output/export headroom; source files not removed")
        with r0.replay_runtime("context") as runtime:
            atomic_json(
                {
                    "policy": runtime,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "precision": "FP32_no_AMP",
                },
                out / "runtime.json",
            )
            reports = {split: evaluate_split(root, out, meta, split, "cuda") for split in SPLITS}
        # Rehash all source-bound files and lineage; never claim immutability from names alone.
        verify_source(root, out)
        summary = {
            "cohorts": reports,
            "common_decoder_route_advantage": all(
                reports[s]["seeds"][str(seed)]["decoder_route_advantage"]
                for s in ("test", "cross_research")
                for seed in (42, 43)
            ),
            "common_recoverability_qualification": all(
                reports[s]["seeds"][str(seed)]["recoverability_qualification"]
                for s in ("test", "cross_research")
                for seed in (42, 43)
            ),
            "original_exit_qualification_changed": False,
            "promoted": False,
            "limits": "Different readout capacities; repeated research cohorts; p128 only; old activity/history/current gates not replaced",
        }
        atomic_json(summary, out / "decision.json")
        state.update(status="f11_readability_complete_requires_review", source_unchanged=True)
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.iterdir()
            if p.suffix in (".json", ".npz")
            and p.name not in ("completion.json", "status.json", "budget.json", "request.json")
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
    out.mkdir(parents=True, exist_ok=True)
    request = {"schema": SCHEMA, "source": str(root), "output": str(out)}
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        req = out / "request.json"
        if (req.exists() and read_json(req) != request) or (
            not req.exists() and any(p.name != "session.lock" for p in out.iterdir())
        ):
            raise ValueError("Existing unbound output; no overwrite")
        atomic_json(request, req)
        budget = (
            read_json(out / "budget.json")
            if (out / "budget.json").exists()
            else {"used_seconds": 0.0, "attempts": []}
        )
        used = budget["used_seconds"]
        if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
            raise ValueError("Cumulative2h budget exhausted; export only")
        remaining = SESSION_CAP - used
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget["used_seconds"] = SESSION_CAP  # crash conservatively consumes reservation
        atomic_json(budget, out / "budget.json")
        started = time.monotonic()
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "obson.babel.state_readability_run",
                "--worker",
                "--source",
                str(root),
                "--out",
                str(out),
            ]
        )

        def stop(signum, frame):
            raise InterruptedError(f"signal {signum}")

        previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = child.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            atomic_json(
                {
                    "schema": SCHEMA,
                    "status": "timeout" if code == 124 else "stopped",
                    "optimizer_updates": 0,
                    "readout_fits": 0,
                    "promoted": False,
                },
                out / "status.json",
            )
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
        elapsed = min(remaining, time.monotonic() - started)
        budget["used_seconds"] = used + elapsed
        attempt.update(elapsed_seconds=elapsed, exit_code=code)
        # Preserve failure evidence across an authorized retry; mutable status is not history.
        snapshot = {
            "attempt": attempt,
            "status": read_json(out / "status.json")
            if (out / "status.json").exists()
            else {"status": "worker_did_not_write_status"},
            "numeric_diagnostics": {
                p.name: read_json(p)
                for p in out.glob("*.json")
                if p.stem.endswith(("_qualification", "_direct", "_targets"))
            },
        }
        atomic_json(snapshot, out / f"attempt_{len(budget['attempts']):03}.json")
        atomic_json(budget, out / "budget.json")
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.source.resolve()
    out = data.source.interface.separate(args.out, root)
    if args.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(root, out))
    raise SystemExit(supervise(root, out))


if __name__ == "__main__":
    main()
