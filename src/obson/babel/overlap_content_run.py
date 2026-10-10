"""F03: one bounded, frozen four-model physical-overlap diagnostic on AutoDL."""

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

from . import history_price_truth as truth
from . import history_state_api as api
from . import history_state_delivery as delivery
from . import overlap_content_audit as geometry
from . import overlap_content_metrics as metrics
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

prior, data = api.prior, api.prior.data
SCHEMA = "babel-overlap-content768-v1"
SOURCE_COMPLETION = "a192bf5f0e733375bc36bb2af26e250561f1ddb664202b2dcb1107156d25b7d0"
INVENTORY_SHA256 = "007e788425787f60c115ff559796a36448908d0800e53c23986e7a04b5b2b0f1"
INVENTORY = Path(__file__).with_name("overlap_content_inventory.json")
MICRO, SESSION_CAP, MAIN_CAP, SINGLE_CAP = 8, 7200, 38416, 24
MODULE = "obson.babel.overlap_content_run"


def protocol():
    return {
        "schema": SCHEMA,
        "task_ids": ["F03"],
        "source_completion": SOURCE_COMPLETION,
        "models": list(api.MODELS),
        "splits": dict(prior.SPLITS),
        "positions": list(metrics.POSITIONS),
        "shifts": list(geometry.SHIFTS),
        "common_predicted_counts": [126, 111, 63],
        "inventory_sha256": INVENTORY_SHA256,
        "asof": "2026-09-12 03:00:00",
        "micro": MICRO,
        "jobs": 1,
        "session_seconds": SESSION_CAP,
        "main_endpoints_max": MAIN_CAP,
        "single_endpoints_max": SINGLE_CAP,
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
        "raw_price_truth": "original float64 closes; physical price and source-cache scoring separate",
        "activity": "original train scales and causal feature masks; no raw-activity certification",
        "inference": "2000 paired month/contract-period-month cluster bootstrap; seed20261010; >=100 parents/6groups",
        "tails": "P50/P95 across parent-window errors descriptive; intervals for paired mean contrasts",
        "decision": "no winner/quality gate; distinguish shared error, disagreement and later correction; review then select one core change",
        "resume": "committed chunks only; attempted forwards reserved before execution; interrupted uncommitted inference is export-only",
        "resource": "first main chunk reused for timing/memory; projected remaining main cost*1.5+600s must fit remaining time; minimum2GiB disk",
        "single_checks": "first parent p64 and p127, each model/cohort; no additional uncounted warmups",
    }


def implementation():
    return {
        "inherited": delivery.implementation(),
        "modules": {
            Path(m.__file__).name: sha256(m.__file__)
            for m in (truth, geometry, metrics, sys.modules[__name__])
        },
        "inventory": sha256(INVENTORY),
        "script": sha256(prior.source_run.lr.REPO / "scripts/babel_overlap_content768_autodl.sh"),
    }


def verify_source(root, out):
    if sha256(root / "completion.json") != SOURCE_COMPLETION:
        raise ValueError(
            "Expected reviewed history-state interface completion; no source substitution"
        )
    done = read_json(root / "completion.json")
    data.source.old.verify_files(root, done["files"])
    spec = read_json(root / "manifest.json")
    if (
        spec["implementation"] != delivery.implementation()
        or spec["protocol"] != delivery.protocol()
    ):
        raise ValueError("Original interface implementation/protocol changed")
    readability = Path(spec["source"]).resolve()
    coverage, meta = api.verify_source(readability)
    data.source.interface.separate(out, root, readability, coverage)
    if prior.verify_source(coverage, out) != meta or spec["statistics"] != meta["statistics"]:
        raise ValueError("Original model/data/statistics lineage changed")
    return coverage, meta


def inventory():
    if sha256(INVENTORY) != INVENTORY_SHA256:
        raise ValueError("Pinned original raw inventory changed")
    return read_json(INVENTORY)


def prepare_raw(loader, rows, x, statistics):
    """CPU only. Keep raw truth out of the inference function and feature tensors."""
    closes, identities = [], []
    for r in rows:
        loader.window(r, asof=protocol()["asof"])
        times, _ = loader.cache[r["key"]]
        if r["row"] < 254 or str(times[r["row"] - 254]) != r["input_start"]:
            raise ValueError("Original full255 start differs from raw rows")
        ends, windows = [], []
        for p in metrics.POSITIONS:
            end = r["row"] - (128 - p)
            endpoint = {**r, "row": end, "end": str(times[end])}
            windows.append(loader.window(endpoint, asof=protocol()["asof"]))
            ends.append(
                {
                    "position": p,
                    "key": r["key"],
                    "row": end,
                    "end": str(times[end]),
                    "input_start": str(times[end - 127]),
                }
            )
        for d in geometry.SHIFTS:
            a = metrics.POSITIONS.index(128 - d)
            if not np.array_equal(windows[a][d:-1], windows[3][: 127 - d]):
                raise ValueError("Raw common closes not identical")
        closes.append(windows)
        identities.append(ends)
    closes = np.asarray(closes, dtype=np.float64)
    path = truth.from_raw_closes(closes.reshape(-1, 128))["relative_log_close"].reshape(
        len(rows), 4, 128
    )
    ps = torch.tensor(metrics.POSITIONS, dtype=torch.long).expand(len(x), -1)
    y, mask = prior.core.targets(torch.as_tensor(x), ps, statistics)
    raw_target = (
        path[:, :, -2::-1] * 100 / (np.sqrt(np.arange(1, 128)) * statistics["delta_scale"][0])
    )
    check = prior.r0.compare(raw_target, y.numpy()[..., 0], prior.r0.TARGET_ATOL, prior.r0.RTOL)
    return (
        {"raw_close": closes, "raw_path": path, "source_target": y.numpy(), "mask": mask.numpy()},
        identities,
        check,
    )


class ForwardLedger:
    """Crash-safe attempted-forward accounting; no invisible retries beyond the card."""

    def __init__(self, out):
        self.path = out / "forward_attempts.json"
        self.value = (
            read_json(self.path) if self.path.exists() else {"main": 0, "single": 0, "chunks": {}}
        )
        entries = list(self.value["chunks"].values())
        for kind, cap in (("main", MAIN_CAP), ("single", SINGLE_CAP)):
            if (
                self.value[kind] != sum(v[kind] for v in entries)
                or not 0 <= self.value[kind] <= cap
            ):
                raise ValueError("Invalid attempted-forward ledger")

    def reserve(self, key, main, single):
        if key in self.value["chunks"]:
            raise ValueError("Uncommitted inference attempt exists; export only, no repeat forward")
        if self.value["main"] + main > MAIN_CAP or self.value["single"] + single > SINGLE_CAP:
            raise ValueError("Fixed attempted-forward budget exhausted; export only")
        self.value["main"] += main
        self.value["single"] += single
        self.value["chunks"][key] = {"main": main, "single": single}
        atomic_json(self.value, self.path)


@torch.no_grad()
def forward(e, q, x, device):
    if any(m.training or any(p.requires_grad for p in m.parameters()) for m in (e, q)):
        raise ValueError("Frozen eval models required")
    z = e(torch.as_tensor(x, device=device))[:, -1]
    pred = q(z)
    if not torch.isfinite(z).all() or not torch.isfinite(pred).all():
        raise ValueError("Nonfinite frozen output")
    return z.cpu().numpy(), pred.cpu().numpy()


def committed(path, binding):
    receipt = path.with_suffix(".json")
    if not receipt.exists():
        return None
    r = read_json(receipt)
    if r["binding"] != binding or r["sha256"] != sha256(path):
        raise ValueError("Committed output binding/content changed")
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def save_committed(path, bank, binding):
    prior.ev.save_arrays(path, bank)
    atomic_json({"binding": binding, "sha256": sha256(path)}, path.with_suffix(".json"))


def resource_check(out, model, main, seconds, deadline, device, ledger):
    remaining = max(0, MAIN_CAP - ledger.value["main"])
    projected = seconds / main * remaining * 1.5 + 600
    available = deadline - time.monotonic()
    report = {
        "model": model,
        "main_endpoints_reused": main,
        "measured_seconds": seconds,
        "projected_remaining_seconds": projected,
        "available_seconds": available,
        "projection_is_estimate": True,
        "extra_warmup_forwards": 0,
    }
    if str(device).startswith("cuda"):
        report.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_device_bytes=torch.cuda.mem_get_info()[0],
        )
    report["passed"] = projected <= available and shutil.disk_usage(out).free >= 2 * 1024**3
    atomic_json(report, out / f"resource_{model}.json")
    if not report["passed"]:
        raise ValueError(
            "Resource projection exceeds fixed remaining budget; stopped, no automatic micro/budget change"
        )


def extract(root, coverage, out, meta, split, model, x, device, deadline):
    resource = out / f"resource_{model}.json"
    if resource.exists() and not read_json(resource)["passed"]:
        raise ValueError("Previous resource preflight failed; export only")
    manifest_hash = sha256(out / "manifest.json")
    name = f"{split}_{model}"
    final_path = out / f"{name}.npz"
    binding = {"manifest": manifest_hash, "split": split, "model": model}
    saved = committed(final_path, binding)
    if saved is not None:
        return saved
    with np.load(root / f"{name}.npz", allow_pickle=False) as f:
        expected = {k: f[k] for k in ("query", "embedding")}
    e = q = None
    before = None
    ledger = ForwardLedger(out)
    folder = out / "chunks"
    folder.mkdir(exist_ok=True)
    bank = []
    for left in range(0, len(x), MICRO):
        key = f"{name}_{left:05}"
        path = folder / f"{key}.npz"
        chunk_binding = {**binding, "left": left, "right": min(left + MICRO, len(x))}
        chunk = committed(path, chunk_binding)
        if chunk is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Cumulative time limit reached before forward")
            if e is None:
                e, q = prior.ev.load(meta, coverage, model, device)
                before = prior.source_run.signature(e, q)
            b = x[left : left + MICRO]
            ledger.reserve(key, len(b) * 4, 2 if left == 0 else 0)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            prior.source_run.sync(device)
            started = time.monotonic()
            queries = []
            states = []
            for p in metrics.POSITIONS:
                z, pred = forward(e, q, b[:, p - 1 : p + 127], device)
                queries.append(pred)
                states.append(z)
            prior.source_run.sync(device)
            chunk = {"query": np.stack(queries, 1), "endpoint_state": states[-1]}
            checks = {}
            if left == 0:
                for index in (0, 2):
                    p = metrics.POSITIONS[index]
                    z, pred = forward(e, q, b[:1, p - 1 : p + 127], device)
                    for field, actual, target in (
                        ("state", z, states[index][:1]),
                        ("query", pred, queries[index][:1]),
                    ):
                        checks[f"single_p{p}/{field}"] = prior.r0.compare(
                            actual, target, prior.r0.STATE_ATOL, prior.r0.STATE_RTOL
                        )
            for field, actual, target in (
                ("source_query", queries[-1], expected["query"][left : left + len(b)]),
                ("source_state", states[-1], expected["embedding"][left : left + len(b)]),
            ):
                checks[field] = prior.r0.compare(
                    actual, target, prior.r0.STATE_ATOL, prior.r0.STATE_RTOL
                )
            checks["frozen_tensors"] = {"passed": before == prior.source_run.signature(e, q)}
            prior.r0.save_checks(folder, key + "_qualification", checks)
            save_committed(path, chunk, chunk_binding)
            seconds = time.monotonic() - started
            if not (out / f"resource_{model}.json").exists():
                resource_check(out, model, len(b) * 4, seconds, deadline, device, ledger)
            elif not read_json(out / f"resource_{model}.json")["passed"]:
                raise ValueError("Previous resource preflight failed; export only")
        bank.append(chunk)
        if left == 0 or (left // MICRO) % 25 == 0:
            progress(
                f"F03 {name}: {min(left + MICRO, len(x))}/{len(x)} parents; four causal endpoints each; zero updates"
            )
    if e is not None:
        frozen = before == prior.source_run.signature(e, q)
        prior.r0.save_checks(out, name + "_frozen", {"frozen_tensors": {"passed": frozen}})
    result = {k: np.concatenate([chunk[k] for chunk in bank]) for k in ("query", "endpoint_state")}
    save_committed(final_path, result, binding)
    del e, q
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def check_complete(out):
    if read_json(out / "manifest.json")["implementation"] != implementation():
        raise ValueError("Completed run implementation changed; use export, not overwrite")
    done = read_json(out / "completion.json")
    data.source.old.verify_files(out, done["files"])
    return done


def execute(root, raw_root, out, seconds):
    state = {
        "schema": SCHEMA,
        "status": "running",
        "task_ids": ["F03"],
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
    }
    atomic_json(state, out / "status.json")
    deadline = time.monotonic() + seconds
    try:
        if not torch.cuda.is_available():
            raise ValueError("Real-model execution requires AutoDL CUDA")
        progress(
            "F03: fixed Macro/D overlap content; checking original source chain, zero training/fitting"
        )
        coverage, meta = verify_source(root, out)
        data.source.interface.separate(out, raw_root)
        spec = {
            "source": str(root),
            "raw_root": str(raw_root),
            "coverage": str(coverage),
            "protocol": protocol(),
            "implementation": implementation(),
            "statistics": meta["statistics"],
        }
        manifest = out / "manifest.json"
        if manifest.exists() and read_json(manifest) != spec:
            raise ValueError("Bound protocol/source/implementation changed; no overwrite")
        atomic_json(spec, manifest)
        if (out / "completion.json").exists():
            done = check_complete(out)
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise ValueError("Need 2GiB disk headroom; source untouched")
        inv = inventory()
        loader = truth.RawCloseWindows(raw_root, inv["files"])
        report = {}
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
                raw, endpoints, check = prepare_raw(loader, rows, x, meta["statistics"])
                prior.r0.save_checks(out, split + "_raw_coordinate", {"raw_vs_cache_price": check})
                atomic_json(endpoints, out / f"{split}_endpoints.json")
                prior.ev.save_arrays(out / f"{split}_truth.npz", raw)
                progress(
                    f"F03 {split}: {count} parent rows and three physical overlaps verified on CPU"
                )
                for model in api.MODELS:
                    decoded = extract(root, coverage, out, meta, split, model, x, "cuda", deadline)
                    per = metrics.paired_rows(
                        decoded["query"],
                        raw["source_target"],
                        raw["mask"],
                        raw["raw_close"],
                        meta["statistics"],
                    )
                    # Original combined query/near/mid/far metrics retain original source truth.
                    original = prior.core.band_rows(
                        torch.from_numpy(decoded["query"]),
                        torch.from_numpy(raw["source_target"]),
                        torch.from_numpy(raw["mask"]),
                        meta["statistics"],
                    )
                    for field, vals in original.items():
                        for j, p in enumerate(metrics.POSITIONS):
                            per[f"original/p{p}/{field}"] = vals[:, j].numpy()
                    prior.ev.save_arrays(out / f"{split}_{model}_errors.npz", per)
                    result = metrics.summarize(per, rows)
                    atomic_json(result, out / f"{split}_{model}_metrics.json")
                    report[f"{split}/{model}"] = {
                        "parents": len(rows),
                        "metrics": f"{split}_{model}_metrics.json",
                    }
                    progress(f"F03 {split}/{model}: paired content scores saved; no selection")
        verify_source(root, out)
        ledger = ForwardLedger(out)
        if ledger.value["main"] != MAIN_CAP or ledger.value["single"] != SINGLE_CAP:
            raise ValueError("Incomplete or changed fixed forward exposure")
        atomic_json(
            {
                "files": {k: inv["files"][k] for k in sorted(loader.cache)},
                "count": len(loader.cache),
                "asof": inv["asof"],
            },
            out / "raw_inventory_verified.json",
        )
        atomic_json(
            {
                "reports": report,
                "decision": "diagnostic_complete_requires_review",
                "promoted": False,
                "quality_assured": False,
                "optimizer_updates": 0,
                "readout_fits": 0,
            },
            out / "overlap_report.json",
        )
        state.update(status="f03_diagnostic_complete_requires_review", source_unchanged=True)
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.suffix in (".json", ".npz")
            and p.name not in ("completion.json", "status.json", "budget.json", "request.json")
            and not p.name.startswith("attempt_")
        }
        atomic_json({**state, "files": files}, out / "completion.json")
        atomic_json(state, out / "status.json")
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(state, out / "status.json")
        raise


def supervise(root, raw_root, out):
    root, raw_root = root.resolve(), raw_root.resolve()
    out = data.source.interface.separate(out, root, raw_root)
    out.mkdir(parents=True, exist_ok=True)
    request = {"schema": SCHEMA, "source": str(root), "raw_root": str(raw_root), "output": str(out)}
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        req = out / "request.json"
        if (req.exists() and read_json(req) != request) or (
            not req.exists() and any(p.name != "session.lock" for p in out.iterdir())
        ):
            raise ValueError("Existing unbound output; no overwrite")
        atomic_json(request, req)
        if (out / "completion.json").exists():
            check_complete(out)
            return 0
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
        budget["used_seconds"] = SESSION_CAP
        atomic_json(budget, out / "budget.json")
        started = time.monotonic()
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                MODULE,
                "--worker",
                "--source",
                str(root),
                "--raw-root",
                str(raw_root),
                "--out",
                str(out),
                "--seconds",
                str(remaining),
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
        atomic_json(
            {
                "attempt": attempt,
                "status": read_json(out / "status.json")
                if (out / "status.json").exists()
                else {"status": "worker_no_status"},
            },
            out / f"attempt_{len(budget['attempts']):03}.json",
        )
        atomic_json(budget, out / "budget.json")
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seconds", type=float, default=SESSION_CAP, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= SESSION_CAP:
        parser.error("Invalid bounded duration")
    root, raw_root = args.source.resolve(), args.raw_root.resolve()
    out = data.source.interface.separate(args.out, root, raw_root)
    if args.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(root, raw_root, out, args.seconds))
    raise SystemExit(supervise(root, raw_root, out))


if __name__ == "__main__":
    main()
