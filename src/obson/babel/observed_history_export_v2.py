"""Read-only CPU export repair for the immutable observed-history v1 run.

The experiment, scientific gates and historical exporter stay unchanged. Only
identity_residual is a signed roundoff diagnostic; actual MSE remains
nonnegative. Payload integrity is reported separately from the producer status.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
import zipfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import observed_history_export as original

POSITIONS = original.POSITIONS
SCHEMA = "babel-observed-history-export-v2"
RUNS = tuple(f"{arm}_s{seed}" for arm in ("control", "full_history") for seed in (42, 43))
COHORTS = ("test", "cross_research")
BANDS = {"near": 16, "mid": 48, "far": 63}
FIELDS = (
    "body",
    "volume_vs_prior_ema32",
    "delta_log_volume",
    "oi_change_percent_asinh",
    "oi_per_volume_asinh",
    "turnover_asinh",
)
POSITION_METRICS = frozenset(
    [f"price/{b}" for b in (*BANDS, "composite")]
    + [f"nonprice/{f}/{b}" for f in FIELDS for b in BANDS]
)
REGULAR_METRICS = frozenset(
    f"{population}/{metric}"
    for population in ("full", "interior", "endpoint")
    for metric in POSITION_METRICS
)
OVERLAP_FIELDS = (
    "shape_a_mse",
    "shape_b_mse",
    "mean_shape_mse",
    "consensus_shape_mse",
    "disagreement_mse",
    "identity_residual",
    "common_anchored_a_mse",
    "common_anchored_b_mse",
    "common_offset_a_mse",
    "common_offset_b_mse",
    "leaving_a_mse",
    "entered_b_mse",
    "flat_shape_baseline_mse",
    "flat_leaving_baseline_mse",
    "flat_entered_baseline_mse",
)
OVERLAP_METRICS = frozenset(
    f"overlap/shift{shift}/{field}" for shift in (1, 16, 64) for field in OVERLAP_FIELDS
)
SIGNED_METRICS = frozenset(f"overlap/shift{s}/identity_residual" for s in (1, 16, 64))
# These rows were computed and serialized as float64. This is a roundoff bound,
# not a configurable numerical-replay or scientific acceptance threshold.
ROUNDING_ULPS = 64


def _score_array(path, per_position):
    """Validate the exact locked row schema, supports and roundoff identity."""
    metrics = POSITION_METRICS if per_position else REGULAR_METRICS | OVERLAP_METRICS
    expected = {f"{family}::{metric}" for family in ("errors", "support") for metric in metrics}
    if not per_position:
        expected.update(f"contexts::{metric}" for metric in REGULAR_METRICS)
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(expected) or set(saved.files) != expected:
            raise ValueError("Exact locked score schema required (22 position /111 parent metrics)")
        arrays = {key: saved[key] for key in expected}
    count = None
    for key, value in arrays.items():
        family, metric = key.split("::")
        if value.ndim != (2 if per_position else 1) or not len(value):
            raise ValueError("Nonempty parent or parent-by128 score arrays required")
        if per_position and value.shape[1] != 128:
            raise ValueError("Score evidence must cover128 positions")
        if count is not None and len(value) != count:
            raise ValueError("Score parent counts differ")
        count = len(value)
        if family == "errors":
            if value.dtype != np.float64 or np.isinf(value).any():
                raise ValueError("Float64 finite-or-missing score values required")
            if metric not in SIGNED_METRICS and (value < 0).any():
                raise ValueError("MSE scores must be nonnegative")
        elif value.dtype.kind not in "iu" or (value < 0).any():
            raise ValueError("Nonnegative integer score supports required")
    for metric in metrics:
        value, support = arrays[f"errors::{metric}"], arrays[f"support::{metric}"]
        if not np.array_equal(np.isfinite(value), support > 0):
            raise ValueError("Score finite/missing values must match exact positive supports")
        if metric in OVERLAP_METRICS:
            _, shift, field = metric.split("/")
            shift = int(shift[5:])
            width = (
                shift
                if field.startswith(("entered", "leaving", "flat_entered", "flat_leaving"))
                else 127 - shift
            )
            if not np.all(support == width):
                raise ValueError("Overlap supports must equal the declared common/edge width")
            continue
        width = 127 if metric.endswith("/composite") else BANDS[metric.rsplit("/", 1)[1]]
        if per_position:
            contexts = np.ones(value.shape, dtype=np.int64)
        else:
            contexts = arrays[f"contexts::{metric}"]
            maximum = {"full": 128, "interior": 127, "endpoint": 1}[metric.split("/", 1)[0]]
            if (contexts > maximum).any() or not np.array_equal(contexts > 0, support > 0):
                raise ValueError("Parent context counts do not match supported states")
        if (support > width * contexts).any() or (not per_position and (support < contexts).any()):
            raise ValueError("Supports exceed declared age/context bounds")
        if metric.startswith("price/") or "/price/" in metric:
            maximum = (
                1
                if per_position
                else {"full": 128, "interior": 127, "endpoint": 1}[metric.split("/", 1)[0]]
            )
            if not np.all(support == width * maximum) or not np.all(contexts == maximum):
                raise ValueError("All price ages and contexts must be present")
    if not per_position:
        for shift in (1, 16, 64):
            prefix = f"errors::overlap/shift{shift}/"
            a, b, mean, consensus, disagreement, residual = (
                arrays[prefix + field]
                for field in (
                    "shape_a_mse",
                    "shape_b_mse",
                    "mean_shape_mse",
                    "consensus_shape_mse",
                    "disagreement_mse",
                    "identity_residual",
                )
            )
            scale = np.maximum.reduce((a, b, mean, consensus, disagreement / 4))
            tolerance = ROUNDING_ULPS * np.finfo(np.float64).eps * scale
            # Retain a representable bound at zero/subnormal scales, without
            # granting a unit-sized absolute tolerance to tiny financial errors.
            tolerance = np.maximum(
                tolerance, np.nextafter(np.float64(0), np.float64(1)) * ROUNDING_ULPS
            )
            expected_residual = mean - consensus - disagreement / 4
            if (
                (np.abs(mean - (a / 2 + b / 2)) > tolerance).any()
                or (np.abs(residual) > tolerance).any()
                or (np.abs(residual - expected_residual) > tolerance).any()
            ):
                raise ValueError(
                    "Overlap identity_residual exceeds scale-aware roundoff or formula differs"
                )
    return count


def _disposition(path, relative):
    # All old path and fixed-eight payload rules are reused. Score handling is
    # explicitly replaced here; no mutation of the historical module occurs.
    parts = Path(relative).parts
    if path.suffix == ".npz" and parts[0] == "evaluation":
        if (
            len(parts) != 4
            or parts[1] not in RUNS
            or parts[2] not in COHORTS
            or path.name not in ("position_rows.npz", "parent_rows.npz")
        ):
            return False, "undeclared_evaluation_array"
        try:
            _score_array(path, path.name == "position_rows.npz")
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
            return False, f"invalid_score_evidence: {error}"
        return True, "report_array"
    return original._disposition(path, relative)


def _required_payload():
    required = {
        "status.json",
        "completion.json",
        "manifest.json",
        "request.json",
        "model_selection_lock.json",
        "observed_history_metrics.json",
        "budget.json",
        "data_partition_audit.json",
    }
    for run in RUNS:
        required.update(
            f"{run}/{name}"
            for name in (
                "completion.json",
                "checkpoint.json",
                "history.jsonl",
                "training_summary.json",
                "causal_audit.json",
                "final_causal_audit.json",
            )
        )
        for cohort in COHORTS:
            required.update(
                f"evaluation/{run}/{cohort}/{name}"
                for name in (
                    "complete.json",
                    "metrics.json",
                    "rows.json",
                    "parent_rows.npz",
                    "position_rows.npz",
                )
            )
            required.add(f"results/{cohort}_{run}_predictions.npz")
    return required


def _report_integrity(root, included, omitted):
    """Presence, score schema and recorded byte bindings; no scientific acceptance."""
    present = {item["path"]: item["sha256"] for item in included}
    inventory = present | {item["path"]: item["sha256"] for item in omitted}
    required = _required_payload()
    missing = sorted(required - set(present))
    invalid = [item for item in omitted if item["reason"].startswith("invalid_")]
    issues = []
    receipts = [("completion.json", "")]
    receipts += [(f"{run}/completion.json", f"{run}/") for run in RUNS]
    receipts += [
        (f"evaluation/{run}/{cohort}/complete.json", "") for run in RUNS for cohort in COHORTS
    ]
    for receipt, prefix in receipts:
        if receipt not in present:
            continue
        try:
            data = json.loads((root / receipt).read_text())
            bindings = data["files"]
            if not isinstance(bindings, dict) or not bindings:
                raise ValueError("nonempty files mapping required")
            for relative, expected in bindings.items():
                if (
                    not isinstance(relative, str)
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or not isinstance(expected, str)
                    or re.fullmatch(r"[a-f0-9]{64}", expected) is None
                ):
                    raise ValueError("invalid relative file binding")
                full = prefix + relative
                if inventory.get(full) != expected:
                    issues.append(
                        {
                            "receipt": receipt,
                            "path": full,
                            "reason": "source_file_missing_or_fingerprint_changed",
                        }
                    )
            if receipt == "completion.json":
                # Status and budget remain mutable operational receipts. Every
                # scientific required file must still be in the completion map.
                unbound = (
                    required - {"status.json", "completion.json", "budget.json"} - set(bindings)
                )
                if unbound:
                    issues.append(
                        {
                            "receipt": receipt,
                            "reason": "required_payload_unbound",
                            "paths": sorted(unbound),
                        }
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(
                {"receipt": receipt, "reason": "invalid_completion_receipt", "detail": str(error)}
            )
    for run in RUNS:
        for cohort in COHORTS:
            prefix = f"evaluation/{run}/{cohort}/"
            paths = [
                prefix + name
                for name in ("parent_rows.npz", "position_rows.npz", "rows.json", "complete.json")
            ]
            paths.append(f"results/{cohort}_{run}_predictions.npz")
            if not all(path in present for path in paths):
                continue
            try:
                counts = []
                for relative in paths[:2]:
                    with np.load(root / relative, allow_pickle=False) as saved:
                        counts.append(len(saved[saved.files[0]]))
                rows = json.loads((root / paths[2]).read_text())
                if not isinstance(rows, list):
                    raise ValueError("parent rows must be a list")
                counts.append(len(rows))
                receipt = json.loads((root / paths[3]).read_text())
                counts.append(receipt["binding"]["parents"])
                with np.load(root / paths[4], allow_pickle=False) as saved:
                    counts.append(len(saved["predictions"]))
                if (
                    any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts)
                    or len(set(counts)) != 1
                ):
                    raise ValueError("score/identity/prediction/receipt parent counts differ")
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append(
                    {"receipt": paths[3], "reason": "parent_count_mismatch", "detail": str(error)}
                )
    complete = not (missing or invalid or issues)
    return {
        "status": "complete" if complete else "incomplete",
        "required_payload_count": len(required),
        "missing_or_rejected": missing,
        "invalid_payloads": invalid,
        "binding_issues": issues,
        "scope": "Download payload schemas/presence and source receipt hashes only; training status and scientific acceptance are separate. Omitted full128 predictions and weights are not replayed.",
    }


def export_reports(out, download, *, command_exit_code=0, mode="export", log=None):
    """Acquire producer locks, copy a private snapshot and atomically publish it.

    A previous archive is preserved with a unique timestamped backup. Omitted
    raw caches, weights and full128 arrays remain on AutoDL with sizes/hashes in
    export_manifest.json. No source experiment directories are ever deleted.
    """
    out, download = original._path(out), original._path(download)
    if (
        out == download
        or out in download.parents
        or download in out.parents
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", out.name)
    ):
        raise ValueError("Separate, named output/download directories required")
    if (
        mode not in ("all", "evaluate", "export")
        or isinstance(command_exit_code, bool)
        or not isinstance(command_exit_code, int)
    ):
        raise ValueError("Known export mode and integer command exit code required")
    if out.exists() and not out.is_dir():
        raise ValueError("Output path is not a directory")
    if log is not None:
        log = original._path(log)
        if log.exists() and not log.is_file():
            raise ValueError("Run log must be a regular file")
    download.mkdir(parents=True, exist_ok=True)
    archive = original._path(download / f"{out.name}_reports_v2.tar.gz")
    with ExitStack() as stack:
        locked = []
        if out.exists():
            # The parent holds session.lock; its serial child holds worker.lock.
            # Both are necessary when an interrupted parent has not reaped child.
            locks = {out / "session.lock", out / "worker.lock"}
            locks.update(out.glob("**/worker.lock"))
            locks.update(out.glob("**/evaluation.lock"))
            for path in sorted(locks):
                if not path.exists():
                    continue
                stream = stack.enter_context(original._path(path).open("rb"))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked.append(path.relative_to(out).as_posix())
                except BlockingIOError as error:
                    raise ValueError(
                        "Active worker/session: refuse an inconsistent export"
                    ) from error
        files = original._inventory(out) if out.exists() else []
        temporary = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=".observed_history_export_v2_", dir=download)
        )
        stage = Path(temporary)
        root = stage / out.name
        root.mkdir()
        included, omitted = [], []
        for path in files:
            relative = path.relative_to(out).as_posix()
            before = path.stat()
            digest = original._sha(path)
            entry = {"path": relative, "sha256": digest, "bytes": before.st_size}
            allowed, reason = _disposition(path, relative)
            if allowed:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if original._sha(target) != digest:
                    raise ValueError(f"Report changed during export: {relative}")
                included.append(entry)
            else:
                omitted.append(entry | {"reason": reason})
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"Report changed during export: {relative}")
        if log is not None and log.is_file():
            target = root / "wrapper.log"
            if target.exists():
                raise ValueError("Run log would overwrite report member wrapper.log")
            before_log = log.stat()
            log_digest = original._sha(log)
            shutil.copyfile(log, target)
            after_log = log.stat()
            if (
                original._sha(target) != log_digest
                or original._sha(log) != log_digest
                or (before_log.st_ino, before_log.st_size, before_log.st_mtime_ns)
                != (after_log.st_ino, after_log.st_size, after_log.st_mtime_ns)
            ):
                raise ValueError("Run log changed during export")
            included.append(
                {
                    "path": "wrapper.log",
                    "sha256": original._sha(target),
                    "bytes": target.stat().st_size,
                }
            )
        # Locks coordinate live producers; verify the whole source inventory too.
        # No lock files or other files are created inside the experiment tree.
        if (original._inventory(out) if out.exists() else []) != files:
            raise ValueError("Source inventory changed during export")
        for item in included + omitted:
            if item["path"] == "wrapper.log" and log is not None:
                continue
            if original._sha(out / item["path"]) != item["sha256"]:
                raise ValueError("Source bytes changed during export: " + item["path"])
        run_status = original._status(root, command_exit_code)
        integrity = _report_integrity(root, included, omitted)
        consistency = (
            "existing_producer_locks_held_and_source_rechecked"
            if locked
            else "source_rechecked_no_producer_locks"
            if out.exists()
            else "no_output_directory"
        )
        export_status = (
            f"command={mode}\ncommand_exit_code={command_exit_code}\nrun_status={run_status}\n"
            f"report_integrity={integrity['status']}\n"
            "task_ids=F01,F03,F05\nweights_exported=false\nfull128_predictions_exported=false\n"
            f"snapshot_consistency={consistency}\n"
        )
        (root / "export_status.txt").write_text(export_status)
        report = {
            "schema": SCHEMA,
            "run": out.name,
            "source_directory": str(out),
            "created_utc": datetime.now(UTC).isoformat(),
            "command": mode,
            "command_exit_code": command_exit_code,
            "run_status": run_status,
            "snapshot_consistency": consistency,
            "held_producer_locks": locked,
            "source_files_created_or_rewritten": 0,
            "included": included,
            "omitted": omitted,
            "fixed_prediction_positions": list(POSITIONS),
            "task_ids": ["F01", "F03", "F05"],
            "report_integrity": integrity,
            "exporter_sha256": original._sha(Path(__file__)),
            "historical_exporter_sha256": original._sha(Path(original.__file__)),
            "metrics_file": "observed_history_metrics.json",
            "export_training_updates": 0,
            "export_readout_fits": 0,
            "export_neural_forwards": 0,
            "evaluation_semantics": "Public states are independent closed128 endpoints; internal full-attention tokens are not historical bar states",
            "scope": "Packaging only; no assertion of scientific acceptance or omitted-prediction replay",
        }
        original._json(root / "export_manifest.json", report)
        packed = stage / "reports.tar.gz"
        with tarfile.open(packed, "w:gz") as tar:
            tar.add(root, arcname=out.name, recursive=True)
        backup = None
        if archive.exists():
            backup = original._path(
                download
                / f"{out.name}_reports_v2.previous_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}.tar.gz"
            )
            shutil.copyfile(archive, backup)
        os.replace(packed, archive)
    result = {
        "archive": str(archive),
        "run_status": run_status,
        "command_exit_code": command_exit_code,
        "report_integrity": integrity["status"],
    }
    if backup is not None:
        result["previous_archive"] = str(backup)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--command-exit-code", type=int, default=0)
    parser.add_argument("--mode", choices=("all", "evaluate", "export"), default="export")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    result = export_reports(
        args.out,
        args.download,
        command_exit_code=args.command_exit_code,
        mode=args.mode,
        log=args.log,
    )
    print(
        f"Download archive: {result['archive']} (command_exit_code={result['command_exit_code']}, run_status={result['run_status']}, report_integrity={result['report_integrity']})",
        flush=True,
    )
    if "previous_archive" in result:
        print(f"Previous archive preserved: {result['previous_archive']}", flush=True)


if __name__ == "__main__":
    main()
