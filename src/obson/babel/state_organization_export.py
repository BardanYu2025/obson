"""Immutable, weights-free F05/F01/F11 eight-model report snapshots.

All initial and final endpoint evidence is retained. Signed identity residual
validation is reused from the corrected exporter, without changing historical
modules. Producer completion and download integrity remain separate states.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import uuid
import zipfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import observed_history_export as original
from .observed_history_export_v2 import (
    BANDS as BANDS,
)
from .observed_history_export_v2 import (
    FIELDS as FIELDS,
)
from .observed_history_export_v2 import (
    OVERLAP_METRICS as OVERLAP_METRICS,
)
from .observed_history_export_v2 import (
    POSITION_METRICS as POSITION_METRICS,
)
from .observed_history_export_v2 import (
    REGULAR_METRICS as REGULAR_METRICS,
)
from .observed_history_export_v2 import (
    SIGNED_METRICS as SIGNED_METRICS,
)
from .observed_history_export_v2 import (
    _score_array,
)

POSITIONS = original.POSITIONS
SCHEMA = "babel-state-organization-export-v1"
RUNS = tuple(
    f"{arm}_s{seed}"
    for arm in ("initial", "plain", "late_memory", "staged_memory")
    for seed in (42, 43)
)
TRAINING_RUNS = tuple(name for name in RUNS if not name.startswith("initial_"))
COHORTS = ("test", "cross_research")
WRITE_SOURCES = {"plain": [], "late_memory": [4, 4], "staged_memory": [2, 4]}


def _disposition(path, relative):
    parts = Path(relative).parts
    if relative in ("export_manifest.json", "export_status.txt"):
        return False, "replaced_by_current_export_metadata"
    if path.suffix in (".pt", ".pth", ".bin", ".safetensors"):
        return False, "weights_or_optimizer_state"
    if path.suffix == ".npy":
        return False, "full128_prediction_shard" if parts[0] == "predictions" else "binary_array"
    if path.suffix == ".lock" or ".tmp" in path.name:
        return False, "lock_or_uncommitted_temporary"
    if path.suffix == ".npz":
        if parts[0] in ("raw", "cache", "predictions"):
            return False, "raw_or_feature_cache_or_full_predictions"
        if parts[0] == "results":
            if (
                not re.fullmatch(
                    r"(?:test|cross_research)_(?:initial|plain|late_memory|staged_memory)_s(?:42|43)_predictions\.npz",
                    path.name,
                )
                or len(parts) != 2
            ):
                return False, "undeclared_prediction_file"
            try:
                original._fixed_predictions(path)
            except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
                return False, f"invalid_fixed_prediction_evidence: {error}"
        elif parts[0] == "evaluation":
            if (
                len(parts) != 4
                or re.fullmatch(r"(?:initial|plain|late_memory|staged_memory)_s(?:42|43)", parts[1])
                is None
                or parts[2] not in ("test", "cross_research")
                or path.name not in ("position_rows.npz", "parent_rows.npz")
            ):
                return False, "undeclared_evaluation_array"
            try:
                _score_array(path, path.name == "position_rows.npz")
            except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
                return False, f"invalid_score_evidence: {error}"
        else:
            return False, "undeclared_binary_report"
        return True, "report_array"
    if path.suffix in (".json", ".jsonl", ".txt", ".log", ".md"):
        return True, "report_text"
    return False, "not_report_payload"


def _required_payload():
    required = {
        "status.json",
        "completion.json",
        "manifest.json",
        "request.json",
        "model_selection_lock.json",
        "state_organization_metrics.json",
        "budget.json",
        "data_partition_audit.json",
        "source_validation.json",
        "f11_cost.json",
    }
    for run in RUNS:
        names = ("checkpoint.json", "causal_audit.json")
        if run.startswith("initial_"):
            names += ("source_replay.json",)
        if run in TRAINING_RUNS:
            names += (
                "completion.json",
                "history.jsonl",
                "training_summary.json",
                "final_causal_audit.json",
                "source_restore.json",
            )
        required.update(f"{run}/{name}" for name in names)
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
    lock = None
    if "model_selection_lock.json" in present:
        try:
            lock = json.loads((root / "model_selection_lock.json").read_text())
            if (
                lock.get("schema") != "babel-state-organization768-v1"
                or lock.get("selection") != "fixed_last60"
                or set(lock.get("models", {})) != set(RUNS)
            ):
                raise ValueError("Complete fixed initial180/final60 eight-model lock required")
        except (OSError, ValueError, TypeError) as error:
            issues.append(
                {
                    "receipt": "model_selection_lock.json",
                    "reason": "invalid_model_lock",
                    "detail": str(error),
                }
            )
    receipts = [("completion.json", "")]
    receipts += [(f"{run}/completion.json", f"{run}/") for run in TRAINING_RUNS]
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
                binding = receipt["binding"]
                counts.append(binding["parents"])
                arm = "plain" if run.startswith("initial_") else run.rsplit("_s", 1)[0]
                contract = binding.get("model_contract", {})
                if (
                    receipt.get("schema") != "babel-state-organization-evaluation-v1"
                    or binding.get("name") != run
                    or binding.get("split") != cohort
                    or contract.get("arm") != arm
                    or contract.get("attention_mode") != "causal"
                    or contract.get("public_output") != "organized_final_state_only"
                    or contract.get("private_memory_exposed") is not False
                    or contract.get("memory_write_sources") != WRITE_SOURCES[arm]
                    or binding.get("lock", {}).get("model_lock")
                    != present.get("model_selection_lock.json")
                    or lock is None
                    or binding.get("lock", {}).get("source") != lock.get("source")
                ):
                    raise ValueError("Endpoint evidence and complete model lock identity differ")
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
    archive = original._path(download / f"{out.name}_reports.tar.gz")
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
            tempfile.TemporaryDirectory(prefix=".state_organization_export_", dir=download)
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
            "task_ids=F05,F01,F11\nweights_exported=false\nfull128_predictions_exported=false\n"
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
            "task_ids": ["F05", "F01", "F11"],
            "report_integrity": integrity,
            "exporter_sha256": original._sha(Path(__file__)),
            "path_inventory_helper_sha256": original._sha(Path(original.__file__)),
            "score_validator_sha256": original._sha(
                Path(__file__).with_name("observed_history_export_v2.py")
            ),
            "metrics_file": "state_organization_metrics.json",
            "export_training_updates": 0,
            "export_readout_fits": 0,
            "export_neural_forwards": 0,
            "evaluation_semantics": "Only final768 independent closed128 endpoint states; private cross-depth memory is never passed directly to the decoder",
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
                / f"{out.name}_reports.previous_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}.tar.gz"
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
    if (
        result["run_status"] == "complete_requires_review"
        and result["report_integrity"] != "complete"
    ):
        print(
            "Producer completed, but the saved report is incomplete; inspect export_manifest.json before review.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
