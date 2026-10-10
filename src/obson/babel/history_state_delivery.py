"""Bounded zero-training F11 public-interface parity and low-motion audit."""

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

from . import history_state_api as api
from .ae_extend import atomic_json
from .holdout_audit import read_json
from .progress import progress

prior = api.prior
data = prior.data
SCHEMA = "babel-history-state-delivery-v1"
SESSION_CAP = 7200
MICRO = 8


def protocol():
    return {
        "schema": SCHEMA,
        "task_ids": ["F11"],
        "source_completion": api.SOURCE_COMPLETION,
        "models": list(api.MODELS),
        "splits": dict(prior.SPLITS),
        "position": 128,
        "planned_unique_endpoints": 9604,
        "single_example_checks": 24,
        "micro": MICRO,
        "session_seconds": SESSION_CAP,
        "optimizer_updates": 0,
        "readout_fits": 0,
        "strata_RMS_log_return": [api.FLOOR, api.LOW_RMS],
        "direction_threshold": 0.2,
        "source_tensor_tolerance": [prior.r0.STATE_ATOL, prior.r0.STATE_RTOL],
        "acceptance": "all numerical/source/frozen identity checks; interface parity only, not reliability certification",
        "selection": "none; both Macro seeds as references, both D seeds as research comparison",
        "strata": "observed-only audit, no replacement/filter/calibration; >=100 rows/6months for support, descriptive not new hypothesis test",
        "stop": "qualification/nonfinite/resource failure or cumulative2h; uncommitted inference may repeat within cap",
        "promoted": False,
        "quality_assured": False,
    }


def implementation():
    return {
        "inherited": prior.implementation(),
        "modules": {
            Path(m.__file__).name: prior.sha256(m.__file__) for m in (api, sys.modules[__name__])
        },
        "script": prior.sha256(
            prior.source_run.lr.REPO / "scripts/babel_history_state768_autodl.sh"
        ),
    }


def endpoint_metadata(rows, origin):
    return [
        {
            "key": r["key"],
            "period": r["period"],
            "end": r["end"],
            "closed": True,
            "feature_history_origin": origin,
        }
        for r in rows
    ]


def independent_descriptors(path):
    # Re-express relative closes as asinh log-percent returns, use the source raw-feature oracle.
    raw = np.zeros((len(path), 128, 28), dtype=np.float64)
    raw[:, 1:, 0] = np.arcsinh(np.diff(path, axis=1) * 100)
    return prior.ev.up.probe.descriptors(raw)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value


def extract(root, coverage, out, meta, split, model, x, rows, z, device):
    name = f"{split}_{model}"
    path, receipt = out / (name + ".npz"), out / (name + ".json")
    binding = {
        "manifest_sha256": prior.sha256(out / "manifest.json"),
        "split": split,
        "model": model,
    }
    if receipt.exists():
        saved = read_json(receipt)
        if saved["binding"] != binding or saved["sha256"] != prior.sha256(path):
            raise ValueError("Committed interface result changed")
        return saved
    with np.load(root / (name + ".npz"), allow_pickle=False) as f:
        expected = dict(f)
    e, q = prior.ev.load(meta, coverage, model, device)
    reader = api.HistoryStateReader(
        e,
        q,
        meta["statistics"],
        {"source_completion": api.SOURCE_COMPLETION, "model": model, "width": z.shape[1]},
        device,
    )
    before = prior.source_run.signature(e, q)
    journal = out / "forward_attempts.json"
    attempts = read_json(journal) if journal.exists() else []
    attempts.append(
        {
            "split": split,
            "model": model,
            "reserved_endpoints": len(x),
            "single_checks": 2,
            "started_unix": time.time(),
        }
    )
    atomic_json(attempts, journal)
    bank = {}
    checks = {}
    first_result = None
    for start in range(0, len(x), MICRO):
        values = x[start : start + MICRO, -128:]
        ends = endpoint_metadata(
            rows[start : start + MICRO],
            f"qualified_causal_cache:{meta['source_indexes'][split]}; EMA prehistory retained",
        )
        result = reader.read_prepared(
            values, ends, input_signature=reader.input_signature, audit_observed=True
        )
        if start == 0:
            first_result = result
            for i in range(2):
                single = reader.read_prepared(
                    values[i : i + 1], ends[i : i + 1], input_signature=reader.input_signature
                )
                for field in ("embedding", "query"):
                    checks[f"single_{i}/{field}"] = prior.r0.compare(
                        single[field],
                        result[field][i : i + 1],
                        prior.r0.STATE_ATOL,
                        prior.r0.STATE_RTOL,
                    )
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                bank.setdefault(k, []).append(v)
        for k, v in result["observed_audit"].items():
            if isinstance(v, np.ndarray):
                bank.setdefault("observed/" + k, []).append(v)
    bank = {k: np.concatenate(v) for k, v in bank.items()}
    for field, old in (("embedding", z), ("query", expected["query"])):
        checks["source/" + field] = prior.r0.compare(
            bank[field], old, prior.r0.STATE_ATOL, prior.r0.STATE_RTOL
        )
    checks["independent_descriptors"] = prior.r0.compare(
        bank["descriptors"], independent_descriptors(bank["relative_log_close"]), 1e-9, 1e-8
    )
    with np.load(root / f"{split}_predictions.npz", allow_pickle=False) as f:
        checks["observed_targets"] = prior.r0.compare(
            bank["observed/true_descriptors"], f["targets"], 1e-9, 1e-8
        )
        checks["observed_path"] = prior.r0.compare(
            bank["observed/true_path"], f["true_path"], 1e-9, 1e-8
        )
    checks["frozen_tensors"] = {"passed": before == prior.source_run.signature(e, q)}
    prior.r0.save_checks(out, name + "_qualification", checks)
    prior.ev.save_arrays(path, bank)
    audit = {k[9:]: v for k, v in bank.items() if k.startswith("observed/")}
    boundary = api.boundary_report(bank["relative_log_close"], audit, rows)
    old_direction = api.path_outputs(expected["path"])["direction"]
    delta = bank["descriptors"] - expected["decoded"]
    report = {
        "binding": binding,
        "sha256": prior.sha256(path),
        "rows": len(rows),
        "boundary": boundary,
        "interface_numerical_qualification": True,
        "source_descriptor_max_abs_by_field": np.max(np.abs(delta), axis=0).tolist(),
        "source_descriptor_rms_by_field": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
        "source_direction_changes_by_horizon": np.sum(
            old_direction != bank["direction"], axis=0
        ).tolist(),
        "promoted": False,
        "quality_assured": False,
    }
    # Examples retain both model estimates and explicitly separate observed audits.
    atomic_json(serial(first_result), out / (name + "_examples.json"))
    atomic_json(report, receipt)
    del reader, e, q
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def execute(root, out):
    state = {
        "schema": SCHEMA,
        "status": "running",
        "task_ids": ["F11"],
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
        "quality_assured": False,
    }
    atomic_json(state, out / "status.json")
    try:
        if not torch.cuda.is_available():
            raise ValueError("Real-model execution requires AutoDL CUDA")
        progress(
            "F11: verifying reviewed readability and original model/data lineage; no training or fitting"
        )
        coverage, meta = api.verify_source(root)
        data.source.interface.separate(out, root, coverage)
        if prior.verify_source(coverage, out) != meta:
            raise ValueError("Original source ancestry changed")
        spec = {
            "source": str(root),
            "coverage": str(coverage),
            "protocol": protocol(),
            "implementation": implementation(),
            "statistics": meta["statistics"],
        }
        manifest = out / "manifest.json"
        if manifest.exists() and read_json(manifest) != spec:
            raise ValueError("Bound protocol/implementation differs; no overwrite")
        atomic_json(spec, manifest)
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            data.source.old.verify_files(out, done["files"])
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise ValueError("Need 2GiB headroom; no source files deleted")
        reports = {}
        with prior.r0.replay_runtime("context") as runtime:
            atomic_json(
                {
                    "runtime": runtime,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                },
                out / "runtime.json",
            )
            for split, count in prior.SPLITS.items():
                x, rows = data.cache(meta, coverage, split)
                if len(rows) != count or rows != read_json(root / f"{split}_rows.json"):
                    raise ValueError("Original cohort identities changed")
                atomic_json(rows, out / f"{split}_rows.json")
                states = prior.cache_states(coverage, meta, split)
                for model in api.MODELS:
                    reports[f"{split}/{model}"] = extract(
                        root, coverage, out, meta, split, model, x, rows, states[model], "cuda"
                    )
                    progress(
                        f"F11 {split}/{model}: {len(rows)} fresh prepared-window API results; frozen, no fitting"
                    )
        api.verify_source(root)
        atomic_json(
            {
                "models": list(api.MODELS),
                "reports": reports,
                "interface_numerical_qualification": True,
                "scope": "source-specific prepared features; repeated endpoint cohorts; not raw-CSV/live-stream certification",
                "promoted": False,
                "quality_assured": False,
            },
            out / "interface_report.json",
        )
        state.update(status="f11_interface_complete_requires_review", source_unchanged=True)
        files = {
            str(p.relative_to(out)): prior.sha256(p)
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
                "obson.babel.history_state_delivery",
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
