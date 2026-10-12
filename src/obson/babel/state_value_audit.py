"""F11 purpose/cost settlement from an already reviewed report, CPU only.

No neural imports, model execution, fitting, new scores or winner selection.
Payload accounting is not a compression benchmark: information sets differ.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import history_price_truth as truth

SCHEMA = "babel-state-value-audit-v1"
COMPLETION = "674efdd5690123022c925a1dcfda1e83a7e76ef3cf40e32f4ab8b88b767ba6ab"
MODELS = ("macro_s42", "macro_s43")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_files(root, files):
    """Only consume bound regular files inside the supplied report directory."""
    root = Path(root).resolve()
    for name, expected in files.items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(
                (root.joinpath(*relative.parts[:i])).is_symlink()
                for i in range(1, len(relative.parts) + 1)
            )
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError(f"Invalid report member: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Report fingerprint changed: {name}")


def payload_ledger(endpoints):
    """Numeric payload only; distinct physical endpoints, not seed replicas.

    Consecutive raw windows share127 bars. States are stored as dense float32
    vectors, without an assumed lossless compressor or latent delta codec.
    """
    if type(endpoints) is not int or endpoints < 1:
        raise ValueError("Positive integer endpoint count required")
    rows = endpoints + 127
    return {
        "endpoints": endpoints,
        "single_window_bytes": {
            "close128_float64": 128 * 8,
            "close64_float64_for_fixed16_64_descriptors": 64 * 8,
            "latent768_float32": 768 * 4,
            "latent_with_absolute_price_anchor_float64": 768 * 4 + 8,
            "derived128x28_float32": 128 * 28 * 4,
            "raw128x6_float32_reference_only": 128 * 6 * 4,
            "six_history_descriptors_float64": 6 * 8,
            "four_direction_volatility_descriptors_float64": 4 * 8,
        },
        "consecutive_numeric_archive_bytes": {
            "unique_close_rows": rows,
            "close_float64": rows * 8,
            "one_seed_dense_states_float32": endpoints * 768 * 4,
            "one_seed_states_plus_endpoint_anchors": endpoints * (768 * 4 + 8),
            "two_seed_dense_states_float32": endpoints * 768 * 4 * 2,
            "derived_rows_float32": rows * 28 * 4,
        },
        "incremental_numeric_archive_bytes_per_bar_after_initial_window": {
            "close_float64": 8,
            "one_seed_dense_state_float32": 3072,
            "one_seed_state_plus_anchor": 3080,
            "derived_row_float32": 112,
        },
        "excluded": [
            "timestamps, contract/period identifiers, masks and file/container overhead",
            "shared encoder/query weights, normalization statistics and model identity",
            "live feature rings, EMA prefix state and original raw data for refresh",
            "lossless file compression, quantization and any hypothetical latent delta codec",
        ],
        "information_limits": {
            "close": "Exact original close history only; no OHLC/activity or EMA prefix.",
            "latent": "Approximate multi-field history; also receives full-prefix EMA features.",
            "raw6": "OHLCV/OI arithmetic reference, not a complete equivalent input. Opening OI, availability, timestamps and prefix state are additional.",
            "descriptors": "Only six fixed historical summaries; cannot reconstruct arbitrary history.",
            "derived": "Same prepared encoder input rows;28 channels contain derived redundancy.",
        },
        "interpretation": "No equal-information whole-model compression advantage follows from these counts.",
    }


def compact_closes(windows):
    """Losslessly deduplicate consecutive raw windows; never downcast precision."""
    x = np.asarray(windows)
    if x.dtype != np.dtype("float64") or x.ndim != 2 or x.shape[1] != 128 or not len(x):
        raise ValueError("Nonempty original float64 Nx128 closes required")
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Positive finite closes required")
    if not np.array_equal(x[1:, :-1], x[:-1, 1:]):
        raise ValueError("Windows must share127 identical closes without joining segments")
    compact = np.concatenate((x[0], x[1:, -1]))
    if not np.array_equal(np.lib.stride_tricks.sliding_window_view(compact, 128), x):
        raise AssertionError("Lossless close-history round trip failed")
    return compact


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Finite nonnegative measurements required")
    return {"count": int(a.size), "p50": float(np.median(a)), "p95": float(np.quantile(a, 0.95))}


def benchmark_descriptors(raw):
    """Fixed416 endpoints x3 passes over13 segments; no neural execution.

    This measures validated in-memory single-window raw price summaries only,
    not CSV ingestion, the entire application, or GPU-relative acceleration.
    """
    offsets = np.linspace(0, len(raw) - 1, 32, dtype=int)
    truth.from_raw_closes(raw[:1])  # one recorded, untimed warm-up per segment
    durations = []
    for _ in range(3):
        for i in offsets:
            start = time.perf_counter_ns()
            result = truth.from_raw_closes(raw[i : i + 1])
            durations.append((time.perf_counter_ns() - start) / 1e9)
            if not np.isfinite(result["descriptors"]).all():
                raise ValueError("Nonfinite direct descriptor")
    return durations


def audit(source):
    source = Path(source).resolve()
    if sha256(source / "completion.json") != COMPLETION:
        raise ValueError("Expected the reviewed F11 continuous-state completion")
    done = read_json(source / "completion.json")
    verify_files(source, done["files"])
    manifest = read_json(source / "manifest.json")
    segments = read_json(source / "segments.json")
    if len(segments) != 13 or done["optimizer_updates"] or done["readout_fits"]:
        raise ValueError("Expected13 frozen source segments")
    details, durations, imported, profiles = [], [], {m: [] for m in MODELS}, []
    for segment in segments:
        previous = None
        for model in MODELS:
            name = f"{segment['id']}_{model}"
            receipt = read_json(source / f"{name}.json")
            if receipt["segment"] != segment or receipt["model"] != model:
                raise ValueError("Segment identity changed")
            tag = f"{name}_attempt{receipt['attempt']}"
            array_name = f"{tag}.npz"
            if receipt["files"][array_name] != done["files"][array_name]:
                raise ValueError("Receipt and completion disagree")
            with np.load(source / array_name, allow_pickle=False) as saved:
                raw, z = saved["raw_closes"], saved["embedding"]
                times, seconds = saved["bar_times"], saved["online_seconds"]
                if (
                    raw.shape != (512, 128)
                    or raw.dtype != np.float64
                    or z.shape != (512, 768)
                    or z.dtype != np.float32
                ):
                    raise ValueError("Expected fixed512 raw windows and float32 embeddings")
                if not np.isfinite(z).all() or seconds.shape != (512,) or times.shape != (512,):
                    raise ValueError("Invalid archived output")
                distribution(seconds)
                imported[model].extend(seconds.tolist())
                profiles.append(read_json(source / f"{tag}_runtime.json"))
                if previous is not None:
                    if not np.array_equal(raw, previous[0]) or not np.array_equal(
                        times, previous[1]
                    ):
                        raise ValueError("Seed replicas do not share identical physical history")
                    continue
                compact = compact_closes(raw)
                previous = (raw.copy(), times.copy())
                direct = truth.from_raw_closes(raw)
                durations.extend(benchmark_descriptors(raw))
                ledger = payload_ledger(len(raw))
                if compact.nbytes != ledger["consecutive_numeric_archive_bytes"]["close_float64"]:
                    raise AssertionError("Actual close payload disagrees with ledger")
                details.append(
                    {
                        "segment": segment["id"],
                        "key": segment["key"],
                        "role": segment["role"],
                        "endpoints": len(raw),
                        "close_stream_bytes": compact.nbytes,
                        "one_seed_state_bytes": z.nbytes,
                        "source_npz": array_name,
                        "source_npz_sha256": done["files"][array_name],
                        "close_roundtrip_exact": True,
                        "direct_descriptor_semantics": "Definition computed on original observed prices; not a learned zero-error score.",
                        "direct_descriptors_finite": bool(np.isfinite(direct["descriptors"]).all()),
                        "ledger": ledger,
                    }
                )
    return {
        "schema": SCHEMA,
        "task_ids": ["F11"],
        "status": "cost_accounting_complete_not_model_acceptance",
        "source": {
            "completion_sha256": COMPLETION,
            "verified_members": len(done["files"]),
            "protocol": manifest["protocol"],
            "directory": str(source),
        },
        "implementation": {
            "state_value_audit.py": sha256(__file__),
            "history_price_truth.py": sha256(truth.__file__),
        },
        "budget": {
            "neural_forwards": 0,
            "optimizer_updates": 0,
            "readout_fits": 0,
            "measured_cpu_calls": len(durations),
            "untimed_cpu_warmups": len(segments),
            "cpu_full_segment_summary_calls": len(segments),
        },
        "segments": details,
        "totals": {
            "physical_endpoints": sum(d["endpoints"] for d in details),
            "raw_close_stream_bytes_separate_segments": sum(
                d["close_stream_bytes"] for d in details
            ),
            "one_seed_state_bytes": sum(d["one_seed_state_bytes"] for d in details),
        },
        "direct_cpu_seconds": {
            **distribution(durations),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scope": "in-memory batch1 raw price path plus six past descriptors; excludes ingestion/UI/feature preparation",
        },
        "imported_autodl_online_seconds": {m: distribution(v) for m, v in imported.items()},
        "imported_runtime_profiles": [
            json.loads(s) for s in sorted({json.dumps(p, sort_keys=True) for p in profiles})
        ],
        "cost_caveats": [
            "CPU and AutoDL timings are different hardware and different operations; no speedup ratio.",
            "AutoDL timing includes incremental feature preparation, encoder, query and synchronized transfers; excludes startup and source checks.",
            "Weight files are omitted; shared deployable tensor bytes are unknown, not zero. GPU peak is not a weight-file size.",
            "Current runtime still maintains feature/raw rings and EMA prefix state; a latent is not a self-updating replacement for the stream.",
            "Existing PCA/linear-utility results remain archived; no refitting or repeat utility grid is authorized by this audit.",
        ],
        "decision": {
            "current_history_descriptor_use": "Prefer direct raw calculation as factual reference; model is an optional research view.",
            "embedding_reuse_advantage": "Not established by this cost ledger; historical multi-field evidence remains.",
            "scientific_acceptance": False,
            "promoted": False,
            "next_gpu_jobs": [],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.is_relative_to(source) or out.exists():
        raise ValueError("Use a new output file outside the immutable source directory")
    result = audit(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous evidence, including concurrent runs.
    with out.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"F11 CPU accounting complete: {out}; zero neural execution, no model promotion")


if __name__ == "__main__":
    main()
