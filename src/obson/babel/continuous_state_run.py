"""F11: bounded closed-bar raw replay; fixed Macro, no updates or fitted rules."""

import argparse
import fcntl
import gc
import math
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import closed_bar_state as stream
from . import continuous_state_launch as launch
from . import continuous_state_metrics as metrics
from . import continuous_state_selection as selection
from . import overlap_content_run as parent
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

api, prior, data = parent.api, parent.prior, parent.data
SCHEMA = launch.SCHEMA
MODELS = ("macro_s42", "macro_s43")
SPLITS = ("test", "cross_research")
SEGMENTS, LENGTH, MICRO = 13, 512, 8
OFFSETS = (0, 31, 63, 95, 127, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 511)
PER_SEGMENT = LENGTH + len(OFFSETS)
NOMINAL = SEGMENTS * len(MODELS) * PER_SEGMENT
MAX_ATTEMPTS = 2
SELECTION_SHA256 = "1c8094152d477260ddbea11141ebb59b2b4bb00b90d060358e3087d9210c1394"


class ResourceLimitError(RuntimeError):
    """Bounded projection stop; preserves unused wall-time for manual diagnosis."""


def protocol():
    return {
        "schema": SCHEMA,
        "task_ids": ["F11"],
        "source_completion": parent.SOURCE_COMPLETION,
        "models": list(MODELS),
        "width_layers_heads": [768, 4, 8],
        "selected_epochs": [97, 94],
        "asof": "2026-09-12 03:00:00",
        "inventory_sha256": parent.INVENTORY_SHA256,
        "segments": SEGMENTS,
        "selection_sha256": SELECTION_SHA256,
        "selection_protocol": selection.protocol(),
        "length": LENGTH,
        "selection": "two distinct symbols per split/period; earliest eligible saved endpoint per key; sha256(split/key) order; plus separately labelled known-flat stress test:171",
        "input": "single-contract full-prefix causal28; rolling128; only closed bars; full input range in original research partition; EMA retains earlier prefix",
        "oracle_offsets": list(OFFSETS),
        "online_batch": 1,
        "oracle_micro": MICRO,
        "snapshot_restore_after_offset": 255,
        "jobs": 1,
        "session_seconds": launch.SESSION_CAP,
        "planned_endpoint_forwards": NOMINAL,
        "maximum_reserved_endpoint_forwards": NOMINAL * MAX_ATTEMPTS,
        "attempts_per_segment_model": MAX_ATTEMPTS,
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
        "selection_rule": "both fixed seeds; no checkpoint, rule, smoothing or threshold selection",
        "numeric_contract": {
            "features_atol": 1e-6,
            "features_rtol": 2e-5,
            "state_query_atol": prior.r0.STATE_ATOL,
            "state_query_rtol": prior.r0.STATE_RTOL,
            "relative_log_path_atol": 1e-6,
            "relative_log_path_rtol": prior.r0.STATE_RTOL,
        },
        "evaluation": "raw float64 close truth; exclude current anchor; per-segment 16/64/128 price;16/64 direction, volatility, error runs, switching and lag; stress separate",
        "decision": "engineering qualification plus descriptive use-boundaries, not a scientific promotion gate or independent V13 acceptance",
        "stopping": "source/numeric/nonfinite/resource failure saves evidence then stops; no tolerance relaxation or automatic retry; committed segments reused",
        "resource": "minimum2GiB disk; one model resident; first32 real online outputs reused for projected remaining cost*1.5+600 within remaining cap; no extra warmup",
    }


def implementation():
    return {
        "inherited": parent.implementation(),
        "modules": launch.implementation(),
        "selection": sha256(selection.__file__),
    }


def bind_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable binding changed: {path.name}")
    else:
        atomic_json(value, path)


def comparison(actual, expected, *, features=False):
    return prior.r0.compare(
        actual,
        expected,
        1e-6 if features else prior.r0.STATE_ATOL,
        2e-5 if features else prior.r0.STATE_RTOL,
    )


def save_checks(path, checks):
    atomic_json(checks, path)
    bad = [key for key, value in checks.items() if not value["passed"]]
    if bad:
        raise ValueError(f"{path.name}: failed {bad}; numeric evidence saved")


def source_file(raw_root, segment, inventory):
    symbol, period, contract = segment["key"].split("/")
    path = raw_root / symbol / f"{contract}_{period}m.csv"
    expected = inventory["files"][segment["key"]]
    if sha256(path) != expected:
        raise ValueError(f"Original raw fingerprint changed: {segment['key']}")
    return path, expected


def reserve(out, name, binding):
    journal = out / "forward_attempts.json"
    entries = read_json(journal) if journal.exists() else []
    previous = [entry for entry in entries if entry["name"] == name]
    if any(entry["binding"] != binding for entry in previous):
        raise ValueError("Attempt binding changed")
    if (
        len(previous) >= MAX_ATTEMPTS
        or sum(e["reserved_endpoints"] for e in entries) + PER_SEGMENT > NOMINAL * MAX_ATTEMPTS
    ):
        raise ValueError("Fixed forward-attempt budget exhausted; export only")
    attempt = len(previous) + 1
    entries.append(
        {
            "name": name,
            "binding": binding,
            "attempt": attempt,
            "reserved_endpoints": PER_SEGMENT,
            "started_unix": time.time(),
        }
    )
    atomic_json(entries, journal)
    return attempt


def committed(out, name, binding):
    path = out / f"{name}.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt["binding"] != binding:
        raise ValueError("Committed segment binding changed")
    verify_files(out, receipt["files"])
    return receipt


def raw_preparation(s, segment, stats, cached_window, origin):
    """Independent vectorized oracle, only from prefix ending at final closed bar."""
    first, last = segment["first"], segment["last"]
    frame = s.frame.iloc[: last + 1]
    features = data.original.original.encode(frame, s.period)
    normalized = ((features - stats["x_mean"]) / stats["x_scale"]).astype(np.float32)
    original = segment["source_row"]
    source_check = comparison(
        normalized[original - 127 : original + 1], cached_window, features=True
    )
    bars = list(stream.frame_to_bars(frame, s.key, s.period))
    adapter = stream.PreparedBarStream(s.key, s.period, stats, origin)
    # Prior history prepares EMAs; warmup never calls the model or adds scores.
    for bar in bars[:first]:
        adapter.push(bar, pd.Timestamp(bar.datetime) + pd.Timedelta(minutes=s.period))
    truth = np.stack(
        [frame.close.to_numpy(np.float64)[e - 127 : e + 1] for e in range(first, last + 1)]
    )
    times = [str(t) for t in frame.datetime.iloc[first : last + 1]]
    return bars, normalized, adapter.snapshot(), truth, times, source_check


def run_segment(
    reader,
    s,
    segment,
    stats,
    cached_window,
    expected,
    origin,
    out,
    model,
    binding,
    deadline,
    remaining_endpoints,
    sync=lambda: None,
    runtime=None,
    check_frozen=lambda: True,
):
    """One batch1 streaming trajectory; tests inject synthetic readers only."""
    name = f"{segment['id']}_{model}"
    saved = committed(out, name, binding)
    if saved is not None:
        return saved
    bars, vector, snapshot, raw_closes, times, source_check = raw_preparation(
        s, segment, stats, cached_window, origin
    )
    attempt = reserve(out, name, binding)
    tag = f"{name}_attempt{attempt}"
    atomic_json(runtime or {"kind": "synthetic_injected_reader"}, out / f"{tag}_runtime.json")
    save_checks(out / f"{tag}_input.json", {"original_cached_input": source_check})
    wrapper = stream.ClosedBarHistoryReader(reader, s.key, s.period, origin)
    wrapper.stream = stream.PreparedBarStream.restore(snapshot, stats, origin)
    bank, oracle_windows, endpoints, durations = {}, [], [], []
    feature_max = 0.0
    feature_passed = True
    first = segment["first"]
    for offset, bar in enumerate(bars[first : segment["last"] + 1]):
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed session time exhausted")
        sync()
        started = time.monotonic()
        result = wrapper.push(bar, pd.Timestamp(bar.datetime) + pd.Timedelta(minutes=s.period))
        sync()
        durations.append(time.monotonic() - started)
        prepared, estimate = result["prepared"], result["estimate"]
        end = first + offset
        numeric = comparison(prepared["windows"][0], vector[end - 127 : end + 1], features=True)
        if not numeric["passed"]:
            save_checks(
                out / f"{tag}_numeric.json", {f"incremental_feature_offset{offset}": numeric}
            )
        feature_max = max(feature_max, numeric["max_abs"])
        feature_passed = feature_passed and numeric["passed"]
        for field in ("embedding", "query", "relative_log_close"):
            value = np.asarray(estimate[field])
            if not np.isfinite(value).all():
                save_checks(
                    out / f"{tag}_numeric.json", {field: {"passed": False, "reason": "nonfinite"}}
                )
            bank.setdefault(field, []).append(value)
        if offset in OFFSETS:
            oracle_windows.append(vector[end - 127 : end + 1])
            endpoints.extend(prepared["endpoints"])
        if offset == 255:
            restored = wrapper.stream.snapshot()
            atomic_json(restored, out / f"{tag}_snapshot.json")
            wrapper.stream = stream.PreparedBarStream.restore(
                read_json(out / f"{tag}_snapshot.json"), stats, origin
            )
        if offset == 31:
            projected = np.mean(durations) * max(0, remaining_endpoints - 32) * 1.5 + 600
            enough = projected <= deadline - time.monotonic()
            atomic_json(
                {
                    "observed_endpoints": 32,
                    "mean_online_seconds": float(np.mean(durations)),
                    "projected_remaining_seconds_including_margin": float(projected),
                    "remaining_seconds": deadline - time.monotonic(),
                    "passed": bool(enough),
                },
                out / f"{tag}_resource.json",
            )
            if not enough:
                raise ResourceLimitError("Measured projection exceeds remaining two-hour budget")
        if (offset + 1) % 128 == 0:
            progress(f"F11 {name}: {offset + 1}/{LENGTH} closed bars, no updates")
    bank = {key: np.concatenate(values) for key, values in bank.items()}
    reference = {}
    for start in range(0, len(OFFSETS), MICRO):
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed session time exhausted before oracle")
        result = reader.read_prepared(
            np.stack(oracle_windows[start : start + MICRO]),
            endpoints[start : start + MICRO],
            input_signature=reader.input_signature,
            audit_observed=False,
        )
        for field in ("embedding", "query", "relative_log_close"):
            reference.setdefault(field, []).append(result[field])
    reference = {key: np.concatenate(values) for key, values in reference.items()}
    checks = {
        "all_incremental_features": {
            "passed": feature_passed,
            "max_abs": feature_max,
            "atol": 1e-6,
            "rtol": 2e-5,
        }
    }
    for field in ("embedding", "query"):
        checks[f"batch_oracle/{field}"] = comparison(bank[field][list(OFFSETS)], reference[field])
        at = segment["source_offset"]
        checks[f"original_source/{field}"] = comparison(bank[field][at : at + 1], expected[field])
    checks["batch_oracle/relative_log_close"] = prior.r0.compare(
        bank["relative_log_close"][list(OFFSETS)],
        reference["relative_log_close"],
        1e-6,
        prior.r0.STATE_RTOL,
    )
    checks["frozen_tensors"] = {"passed": bool(check_frozen())}
    save_checks(out / f"{tag}_numeric.json", checks)
    scored = metrics.summarize_segment(bank["relative_log_close"], raw_closes, times, s.period)
    payload = {
        **bank,
        "raw_closes": raw_closes,
        "bar_times": np.asarray(times),
        "online_seconds": np.asarray(durations, np.float64),
        "oracle_offsets": np.asarray(OFFSETS, np.int64),
        **{f"oracle/{key}": value for key, value in reference.items()},
    }
    npz = out / f"{tag}.npz"
    prior.ev.save_arrays(npz, payload)
    report = {
        "binding": binding,
        "segment": segment,
        "model": model,
        "attempt": attempt,
        "online_endpoints": LENGTH,
        "oracle_endpoints": len(OFFSETS),
        "metrics": scored,
        "latency_seconds": {
            "mean": float(np.mean(durations)),
            "p50": float(np.median(durations)),
            "p95": float(np.quantile(durations, 0.95)),
        },
        "files": {
            p.name: sha256(p)
            for p in (
                npz,
                out / f"{tag}_numeric.json",
                out / f"{tag}_resource.json",
                out / f"{tag}_snapshot.json",
                out / f"{tag}_input.json",
                out / f"{tag}_runtime.json",
            )
        },
    }
    atomic_json(report, out / f"{name}.json")
    return report


def check_complete(out):
    meta = read_json(out / "manifest.json")
    done = read_json(out / "completion.json")
    if meta["implementation"] != implementation() or meta["protocol"] != protocol():
        raise ValueError("Completion implementation/protocol changed")
    verify_files(out, done["files"])
    journal = read_json(out / "forward_attempts.json")
    matrix = read_json(out / "continuous_report.json")["reports"]
    names = {
        f"{segment['id']}_{model}"
        for segment in read_json(out / "segments.json")
        for model in MODELS
    }
    counts = Counter(entry["name"] for entry in journal)
    reserved = sum(entry["reserved_endpoints"] for entry in journal)
    if (
        set(matrix) != names
        or set(counts) != names
        or len(names) != SEGMENTS * len(MODELS)
        or any(count > MAX_ATTEMPTS for count in counts.values())
        or any(entry["reserved_endpoints"] != PER_SEGMENT for entry in journal)
        or reserved != done["reserved_endpoint_forwards"]
        or reserved > NOMINAL * MAX_ATTEMPTS
    ):
        raise ValueError("Completion forward ledger/matrix changed")
    if done["status"] != "f11_continuous_complete_requires_review" or done[
        "completed_segment_models"
    ] != SEGMENTS * len(MODELS):
        raise ValueError("Incomplete trajectory matrix")
    if sha256(Path(meta["source"]) / "completion.json") != parent.SOURCE_COMPLETION:
        raise ValueError("Original source completion changed")
    parent.verify_source(Path(meta["source"]), out)
    for path, fingerprint in read_json(out / "raw_identity.json").items():
        if sha256(path) != fingerprint:
            raise ValueError("Completed source raw file changed")
    return done


def execute(source, raw_root, out, seconds):
    source, raw_root, out = launch.separate(source, raw_root, out)
    if not math.isfinite(seconds) or not 0 < seconds <= launch.SESSION_CAP:
        raise ValueError("Invalid remaining time")
    request = read_json(out / "request.json")
    if request["implementation"] != launch.implementation() or any(
        request[k] != str(v)
        for k, v in (("source", source), ("raw_root", raw_root), ("output", out))
    ):
        raise ValueError("Worker requires matching bounded supervisor request")
    started = time.monotonic()
    deadline = started + seconds
    if not torch.cuda.is_available():
        raise ValueError("Real-weight execution belongs on AutoDL CUDA")
    if shutil.disk_usage(out).free < 2 * 1024**3:
        raise ValueError("Need2GiB free; source files untouched")
    progress("F11: verifying fixed Macro lineage; continuous raw-bar inference only, no training")
    coverage, meta = parent.verify_source(source, out)
    stats = meta["statistics"]
    spec = {
        "source": str(source),
        "coverage": str(coverage),
        "raw_root": str(raw_root),
        "protocol": protocol(),
        "implementation": implementation(),
        "statistics": stats,
    }
    bind_json(out / "manifest.json", spec)
    inv = parent.inventory()
    series, masks, rows, cache = {}, {}, {}, {}
    tm = dict(data.source.old.parent.tm(meta["source_meta"]))
    tm["raw_root"] = str(raw_root)
    for split in SPLITS:
        cache[split], rows[split] = data.cache(meta, coverage, split)
        if rows[split] != read_json(source / f"{split}_rows.json"):
            raise ValueError("Original source endpoint identities changed")
        loaded, bounds = data.source.old.parent.xr.xd.load_raw(tm, cross=split == "cross_research")
        series[split] = {s.key: s for s in loaded}
        masks[split] = {s.key: data.original.rp.time_mask(s, bounds, "test") for s in loaded}
    chosen = selection.select_segments(series, rows, masks)
    atomic_json(
        {
            "expected_sha256": SELECTION_SHA256,
            "actual_sha256": api.token(chosen),
            "segments": chosen,
        },
        out / "selection_check.json",
    )
    if len(chosen) != SEGMENTS or api.token(chosen) != SELECTION_SHA256:
        raise ValueError("Fixed13-segment inventory required")
    bind_json(out / "segments.json", chosen)
    raw_identity = {}
    for seg in chosen:
        path, fingerprint = source_file(raw_root, seg, inv)
        raw_identity[str(path)] = fingerprint
    bind_json(out / "raw_identity.json", raw_identity)
    manifest_hash = sha256(out / "manifest.json")
    reports = {}
    with prior.r0.replay_runtime("context") as runtime:
        atomic_json(
            {
                "runtime": runtime,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
            out / "runtime.json",
        )
        for model in MODELS:
            e, q = prior.ev.load(meta, coverage, model, "cuda")
            before = prior.source_run.signature(e, q)
            reader = api.HistoryStateReader(
                e,
                q,
                stats,
                {"source_completion": api.SOURCE_COMPLETION, "model": model, "width": 768},
                "cuda",
            )
            torch.cuda.reset_peak_memory_stats()
            for seg in chosen:
                split = seg["split"]
                s = series[split][seg["key"]]
                index = seg["source_index"]
                with np.load(source / f"{split}_{model}.npz", allow_pickle=False) as f:
                    expected = {k: f[k][index : index + 1] for k in ("embedding", "query")}
                path, fingerprint = source_file(raw_root, seg, inv)
                origin = f"raw:{fingerprint};first:{s.frame.datetime.iloc[0]};complete_prefix"
                binding = {
                    "manifest_sha256": manifest_hash,
                    "segment": seg,
                    "model": model,
                    "raw_sha256": fingerprint,
                }
                name = f"{seg['id']}_{model}"
                remaining = (SEGMENTS * len(MODELS) - len(reports)) * PER_SEGMENT
                reports[name] = run_segment(
                    reader,
                    s,
                    seg,
                    stats,
                    cache[split][index, -128:],
                    expected,
                    origin,
                    out,
                    model,
                    binding,
                    deadline,
                    remaining,
                    torch.cuda.synchronize,
                    read_json(out / "runtime.json"),
                    lambda before=before, e=e, q=q: before == prior.source_run.signature(e, q),
                )
                if sha256(path) != fingerprint:
                    raise ValueError("Raw file changed during streaming")
            save_checks(
                out / f"{model}_frozen.json",
                {"frozen_tensors": {"passed": before == prior.source_run.signature(e, q)}},
            )
            atomic_json(
                {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
                out / f"{model}_memory.json",
            )
            del reader, e, q
            gc.collect()
            torch.cuda.empty_cache()
    for name, fingerprint in raw_identity.items():
        if sha256(name) != fingerprint:
            raise ValueError("Raw identity changed before completion")
    parent.verify_source(source, out)
    atomic_json(
        {
            "schema": SCHEMA,
            "segments": chosen,
            "reports": reports,
            "scope": "descriptive continuous historical research; fixed rules, no smoothing/selection; stress separate",
            "engineering_qualification": True,
            "quality_assured": False,
            "promoted": False,
            "independent_validation": False,
        },
        out / "continuous_report.json",
    )
    # Mutable supervisor accounting/status are deliberately outside the immutable receipt.
    names = [
        "manifest.json",
        "segments.json",
        "selection_check.json",
        "raw_identity.json",
        "runtime.json",
        "continuous_report.json",
        "forward_attempts.json",
    ]
    for name, receipt in reports.items():
        names.extend([name + ".json", *receipt["files"]])
    for model in MODELS:
        names.extend([f"{model}_frozen.json", f"{model}_memory.json"])
    done = {
        "schema": SCHEMA,
        "status": "f11_continuous_complete_requires_review",
        "task_ids": ["F11"],
        "completed_segment_models": len(reports),
        "planned_endpoint_forwards": NOMINAL,
        "reserved_endpoint_forwards": sum(
            x["reserved_endpoints"] for x in read_json(out / "forward_attempts.json")
        ),
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
        "quality_assured": False,
        "files": {name: sha256(out / name) for name in sorted(set(names))},
    }
    atomic_json(done, out / "completion.json")
    check_complete(out)
    launch.status(out, done["status"], completed_segment_models=len(reports), quality_assured=False)
    progress("F11 continuous closed-bar study complete; descriptive results require review")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("checkpoints/babel_history_state768"))
    parser.add_argument("--raw-root", type=Path, default=Path("/root/autodl-tmp/data/contracts"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/babel_continuous_state768"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seconds", type=float, default=launch.SESSION_CAP)
    args = parser.parse_args()
    source, raw, out = args.source.resolve(), args.raw_root.resolve(), args.out.resolve()
    if args.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                execute(source, raw, out, args.seconds)
            except Exception as error:
                label = (
                    "timeout"
                    if isinstance(error, TimeoutError)
                    else "stopped_resource"
                    if isinstance(error, ResourceLimitError)
                    else "failed"
                )
                launch.status(out, label, error=str(error), traceback=traceback.format_exc())
                if isinstance(error, TimeoutError):
                    raise SystemExit(124) from error
                raise
    else:
        raise SystemExit(launch.supervise(source, raw, out, check_complete))


if __name__ == "__main__":
    main()
