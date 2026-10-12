"""Bounded F11 launcher/exporter; no neural imports or model selection."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from contextlib import suppress
from pathlib import Path

SCHEMA = "babel-continuous-state768-v1"
SESSION_CAP = 7200.0
REPOSITORY = Path(__file__).resolve().parents[3]
THREAD_ENV = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "4")


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def separate(source, raw_root, out):
    """Protect requested inputs and source code in both ancestor directions.

    The worker additionally checks restored ancestry, which is not known here.
    This check is also used before export, so a failed unsafe run cannot cause a
    broad scan/copy of an input or repository directory from the EXIT trap.
    """
    source, raw_root, out = (Path(value).resolve() for value in (source, raw_root, out))
    for protected in (source, raw_root, REPOSITORY / "src"):
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError(f"Output must be separate from source/raw/code: {protected}")
    if out.exists() and not out.is_dir():
        raise ValueError("Output must be a directory")
    return source, raw_root, out


def implementation():
    files = [
        "src/obson/babel/continuous_state_launch.py",
        "src/obson/babel/continuous_state_run.py",
        "src/obson/babel/continuous_state_metrics.py",
        "src/obson/babel/continuous_state_selection.py",
        "src/obson/babel/closed_bar_state.py",
        "scripts/babel_continuous_state768_autodl.sh",
    ]
    return {name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest() for name in files}


def status(out, label, **extra):
    result = {
        "schema": SCHEMA,
        "status": label,
        "optimizer_updates": 0,
        "readout_fits": 0,
        "promoted": False,
        **extra,
    }
    atomic_json(result, out / "status.json")
    return result


def _terminate(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def supervise(source, raw_root, out, check_complete):
    """One manual attempt, consuming a cumulative two-hour wall-time budget.

    check_complete(out) must raise on missing/invalid immutable completion. It is
    injected by the worker module to keep this launcher free of neural imports.
    A crash leaves the reservation spent; a normal failure/stop keeps remaining
    time available for a same-request manual retry. There is no auto-retry loop.
    """
    source, raw_root, out = separate(source, raw_root, out)
    if not source.is_dir() or not raw_root.is_dir():
        raise ValueError("Source and raw root directories must exist")
    request = {
        "schema": SCHEMA,
        "source": str(source),
        "raw_root": str(raw_root),
        "output": str(out),
        "session_cap_seconds": SESSION_CAP,
        "implementation": implementation(),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_path = out / "request.json"
        if request_path.exists():
            if read_json(request_path) != request:
                raise ValueError("Bound request/source/implementation differs; export only")
        elif any(item.name != "session.lock" for item in out.iterdir()):
            raise ValueError("Existing output has no matching request; no overwrite")
        atomic_json(request, request_path)
        if (out / "completion.json").exists():
            try:
                completed = check_complete(out)
                if isinstance(completed, dict) and "status" in completed:
                    atomic_json(
                        {k: v for k, v in completed.items() if k != "files"}, out / "status.json"
                    )
            except Exception as error:
                status(
                    out,
                    "failed",
                    error=str(error),
                    traceback=traceback.format_exc(),
                    completion_recheck=True,
                )
                return 1
            # Preserve the worker's scientific scope/status; no budget consumed.
            return 0
        budget_path = out / "budget.json"
        budget = (
            read_json(budget_path)
            if budget_path.exists()
            else {"used_seconds": 0.0, "attempts": []}
        )
        used = budget["used_seconds"]
        if (
            isinstance(used, bool)
            or not isinstance(used, (int, float))
            or not math.isfinite(used)
            or not 0 <= used <= SESSION_CAP
        ):
            raise ValueError("Invalid cumulative wall-time ledger; export only")
        if not isinstance(budget["attempts"], list):
            raise ValueError("Invalid attempt ledger")
        if used >= SESSION_CAP:
            status(out, "timeout", error="Cumulative two-hour budget exhausted; export only")
            return 124
        remaining = SESSION_CAP - used
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget["used_seconds"] = SESSION_CAP
        atomic_json(budget, budget_path)  # A supervisor crash consumes its reservation.
        started = time.monotonic()
        child = None
        code = 1
        old_handlers = {}

        def stopped(signum, frame):
            raise InterruptedError(f"signal {signum}")

        try:
            old_handlers = {
                sig: signal.signal(sig, stopped) for sig in (signal.SIGTERM, signal.SIGINT)
            }
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "obson.babel.continuous_state_run",
                    "--worker",
                    "--source",
                    str(source),
                    "--raw-root",
                    str(raw_root),
                    "--out",
                    str(out),
                    "--seconds",
                    str(remaining),
                ],
                env={**os.environ, **THREAD_ENV},
                start_new_session=True,
            )
            code = child.wait(timeout=max(0.001, remaining - (time.monotonic() - started)))
            if code == 0:
                if not (out / "completion.json").is_file():
                    raise ValueError("Worker exited zero without immutable completion")
                check_complete(out)
            elif code in (124, 130):
                prior = read_json(out / "status.json") if (out / "status.json").exists() else None
                status(
                    out,
                    "timeout" if code == 124 else "stopped",
                    worker_exit_code=code,
                    worker_status=prior,
                )
            else:
                prior = read_json(out / "status.json") if (out / "status.json").exists() else None
                status(out, "failed", worker_exit_code=code, worker_status=prior)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
            if child is not None:
                _terminate(child)
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            prior = read_json(out / "status.json") if (out / "status.json").exists() else None
            status(
                out, "timeout" if code == 124 else "stopped", error=str(error), worker_status=prior
            )
        except Exception as error:
            if child is not None:
                _terminate(child)
            code = 1
            status(out, "failed", error=str(error), traceback=traceback.format_exc())
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        elapsed = max(0.0, min(remaining, time.monotonic() - started))
        # A timeout may come from the child enforcing the same deadline.
        elapsed = remaining if code == 124 else elapsed
        budget["used_seconds"] = used + elapsed
        attempt.update(elapsed_seconds=elapsed, exit_code=code, finished_unix=time.time())
        snapshot = {
            "attempt": attempt,
            "status": read_json(out / "status.json")
            if (out / "status.json").exists()
            else {"status": "missing_worker_status"},
        }
        atomic_json(snapshot, out / f"attempt_{len(budget['attempts']):03}.json")
        atomic_json(budget, budget_path)
        return code


def export_report(source, raw_root, out, log, download, mode, command_exit_code):
    """Export lightweight evidence even when run startup failed; never weights."""
    try:
        _, _, safe_out = separate(source, raw_root, out)
        separation_error = None
    except ValueError as error:
        safe_out = None
        separation_error = str(error)
    download = Path(download).resolve()
    if safe_out is not None and download.is_relative_to(safe_out):
        raise ValueError("Download directory must be outside the output being exported")
    for protected in (Path(source).resolve(), Path(raw_root).resolve(), REPOSITORY / "src"):
        if download.is_relative_to(protected):
            raise ValueError("Download directory must not write into source/raw/code")
    download.mkdir(parents=True, exist_ok=True)
    name = Path(out).name
    if not name or name in (".", ".."):
        name = "babel_continuous_state768_invalid_output"
    run_status = "not_started"
    with tempfile.TemporaryDirectory(prefix=".continuous_state_export.", dir=download) as directory:
        stage = Path(directory)
        report = stage / name
        report.mkdir()
        skipped = []
        if safe_out is not None and safe_out.is_dir():
            for file in sorted(safe_out.rglob("*")):
                relative = file.relative_to(safe_out)
                if file.is_symlink() or any(
                    part in ("cache", "weights", "readouts") for part in relative.parts
                ):
                    skipped.append(str(relative))
                    continue
                if file.is_file() and file.suffix in (".json", ".npz", ".log"):
                    target = report / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, target)
            if (safe_out / "status.json").is_file() and not (safe_out / "status.json").is_symlink():
                try:
                    run_status = read_json(safe_out / "status.json")["status"]
                except (ValueError, KeyError, OSError):
                    run_status = "unreadable_status"
        log = Path(log)
        if log.is_file() and not log.is_symlink():
            shutil.copyfile(log, report / "run.log")
        for suffix in ("md", "json"):
            shutil.copyfile(
                REPOSITORY / "docs" / f"BABEL_ROADMAP.{suffix}",
                report / f"roadmap_at_export.{suffix}",
            )
        if command_exit_code in (124, 130):
            run_status = "timeout" if command_exit_code == 124 else "stopped"
        elif command_exit_code != 0 or separation_error:
            run_status = "failed"
        atomic_json(
            {
                "schema": SCHEMA,
                "command": mode,
                "command_exit_code": command_exit_code,
                "run_status": run_status,
                "optimizer_updates": 0,
                "readout_fits": 0,
                "task_ids": ["F11"],
                "promoted": False,
                "output_separation_error": separation_error,
                "skipped_paths": skipped,
            },
            report / "export_status.json",
        )
        archive = download / f"{name}_reports.tar.gz"
        temporary = stage / "reports.tar.gz"
        with tarfile.open(temporary, "w:gz") as tar:
            tar.add(report, arcname=name)
        temporary.replace(archive)
    print(
        f"Download archive: {archive} (command_exit_code={command_exit_code}, run_status={run_status})",
        flush=True,
    )
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "export"), required=True)
    parser.add_argument("--command-exit-code", type=int, required=True)
    args = parser.parse_args()
    export_report(
        args.source,
        args.raw_root,
        args.out,
        args.log,
        args.download,
        args.mode,
        args.command_exit_code,
    )


if __name__ == "__main__":
    main()
