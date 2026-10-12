"""Consistent, weights-free F01/F03 report export after the worker has stopped."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import uuid
import zipfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

POSITIONS = (1, 32, 64, 80, 96, 112, 127, 128)
SCHEMA = "babel-multiscale-export-v1"


def _path(value):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden in report paths")
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlinks are forbidden in report paths")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _inventory(out):
    found = []
    for directory, dirs, files in os.walk(out, followlinks=False):
        for name in dirs:
            _path(Path(directory) / name)
        for name in files:
            path = _path(Path(directory) / name)
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Only regular files may enter a report snapshot")
            found.append(path)
    return sorted(found)


def _fixed_predictions(path):
    """Check export scope only; this is not a scientific prediction audit."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"predictions", "targets", "masks", "raw_closes", "positions"}:
            raise ValueError("Fixed-position prediction fields differ")
        positions = saved["positions"]
        if positions.dtype.kind not in "iu" or not np.array_equal(positions, POSITIONS):
            raise ValueError("Only the fixed eight prediction positions are downloadable")
        shape = None
        for name, dtype in (
            ("predictions", np.float32),
            ("targets", np.float64),
            ("masks", np.bool_),
        ):
            value = saved[name]
            if value.ndim != 4 or value.shape[1:] != (8, 127, 7) or value.dtype != dtype:
                raise ValueError(f"Invalid fixed-position {name} shape/dtype")
            if shape is not None and value.shape != shape:
                raise ValueError("Fixed-position prediction/target/mask identities differ")
            shape = value.shape
        raw = saved["raw_closes"]
        if raw.dtype != np.float64 or raw.shape != (shape[0], 255):
            raise ValueError("Original float64 raw255 labels must accompany predictions")


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
                    r"(?:test|cross_research)_(?:control|multiscale)_s(?:42|43)_predictions\.npz",
                    path.name,
                )
                or len(parts) != 2
            ):
                return False, "undeclared_prediction_file"
            try:
                _fixed_predictions(path)
            except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
                return False, f"invalid_fixed_prediction_evidence: {error}"
        elif parts[0] == "evaluation" and (
            len(parts) != 4 or path.name not in ("position_rows.npz", "parent_rows.npz")
        ):
            return False, "undeclared_evaluation_array"
        return True, "report_array"
    if path.suffix in (".json", ".jsonl", ".txt", ".log", ".md"):
        return True, "report_text"
    return False, "not_report_payload"


def _status(out, code):
    if code == 124:
        return "timeout"
    if code in (130, 143):
        return "stopped"
    if code:
        return "failed"
    path = out / "status.json"
    if path.exists():
        try:
            return str(json.loads(path.read_text())["status"])
        except (OSError, ValueError, KeyError):
            return "unreadable_status"
    return "not_started"


def export_reports(out, download, *, command_exit_code=0, mode="export", log=None):
    """Acquire producer locks, copy a private snapshot and atomically publish it.

    A previous archive is preserved with a unique timestamped backup. Omitted
    raw caches, weights and full128 arrays remain on AutoDL with sizes/hashes in
    export_manifest.json. No source experiment directories are ever deleted.
    """
    out, download = _path(out), _path(download)
    if (
        out == download
        or out in download.parents
        or download in out.parents
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", out.name)
    ):
        raise ValueError("Separate, named output/download directories required")
    if mode not in ("all", "evaluate", "export") or not isinstance(command_exit_code, int):
        raise ValueError("Known export mode and integer command exit code required")
    if out.exists() and not out.is_dir():
        raise ValueError("Output path is not a directory")
    if log is not None:
        log = _path(log)
        if log.exists() and not log.is_file():
            raise ValueError("Run log must be a regular file")
    download.mkdir(parents=True, exist_ok=True)
    archive = _path(download / f"{out.name}_reports.tar.gz")
    with ExitStack() as stack:
        if out.exists():
            # The parent holds session.lock; its serial child holds worker.lock.
            # Both are necessary when an interrupted parent has not reaped child.
            locks = {out / "session.lock", out / "worker.lock"}
            locks.update(out.glob("**/worker.lock"))
            locks.update(out.glob("**/evaluation.lock"))
            for path in sorted(locks):
                stream = stack.enter_context(_path(path).open("a"))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(
                        "Active worker/session: refuse an inconsistent export"
                    ) from error
        files = _inventory(out) if out.exists() else []
        run_status = _status(out, command_exit_code)
        temporary = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=".multiscale_export_", dir=download)
        )
        stage = Path(temporary)
        root = stage / out.name
        root.mkdir()
        included, omitted = [], []
        for path in files:
            relative = path.relative_to(out).as_posix()
            before = path.stat()
            digest = _sha(path)
            entry = {"path": relative, "sha256": digest, "bytes": before.st_size}
            allowed, reason = _disposition(path, relative)
            if allowed:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if _sha(target) != digest:
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
            shutil.copyfile(log, target)
            included.append(
                {"path": "wrapper.log", "sha256": _sha(target), "bytes": target.stat().st_size}
            )
        consistency = "producer_locks_held" if out.exists() else "no_output_directory"
        export_status = (
            f"command={mode}\ncommand_exit_code={command_exit_code}\nrun_status={run_status}\n"
            "task_ids=F01,F03\nweights_exported=false\nfull128_predictions_exported=false\n"
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
            "included": included,
            "omitted": omitted,
            "fixed_prediction_positions": list(POSITIONS),
            "scope": "Packaging only; no assertion of scientific acceptance or omitted-prediction replay",
        }
        _json(root / "export_manifest.json", report)
        packed = stage / "reports.tar.gz"
        with tarfile.open(packed, "w:gz") as tar:
            tar.add(root, arcname=out.name, recursive=True)
        backup = None
        if archive.exists():
            backup = _path(
                download
                / f"{out.name}_reports.previous_{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}.tar.gz"
            )
            shutil.copyfile(archive, backup)
        os.replace(packed, archive)
    result = {
        "archive": str(archive),
        "run_status": run_status,
        "command_exit_code": command_exit_code,
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
        f"Download archive: {result['archive']} (command_exit_code={result['command_exit_code']}, run_status={result['run_status']})",
        flush=True,
    )
    if "previous_archive" in result:
        print(f"Previous archive preserved: {result['previous_archive']}", flush=True)


if __name__ == "__main__":
    main()
