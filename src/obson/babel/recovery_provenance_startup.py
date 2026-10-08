"""V14 startup repair only. Standard library, before importing CUDA/OpenMP.

A single replacement of an empty or demonstrably preflight-only failed run is
allowed. Numerical audit failures and any worker evidence stay export-only.
The original audit implementation and eight-update protocol remain unchanged.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

PREFLIGHT_FILES = frozenset(
    {
        "session.lock",
        "status.json",
        "request.json",
        "protocol.json",
        "runtime.json",
        "identity.json",
        "scaler_source_files.json",
        "reused_evidence.json",
        "disk_preflight.json",
    }
)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def inspect_existing(out, source, bundle):
    if out.is_symlink() or not out.is_dir():
        raise ValueError("Existing output must be an ordinary directory")
    files = list(out.iterdir())
    unsafe = [
        p.name for p in files if p.is_symlink() or not p.is_file() or p.name not in PREFLIGHT_FILES
    ]
    if unsafe:
        raise ValueError(
            f"Existing audit may have progressed or is unrecognized; export for review: {unsafe}"
        )
    if not files or {p.name for p in files} == {"session.lock"}:
        return {
            "reason": "empty_output_no_worker_evidence",
            "files": {p.name: digest(p) for p in files},
            "optimizer_updates": 0,
        }
    if not (out / "status.json").is_file():
        raise ValueError("Nonempty output without failed status; export for review")
    status = read(out / "status.json")
    if status.get("status") != "failed" or status.get("task_ids") != ["V14"]:
        raise ValueError("Output is active, completed or unrecognized; export for review")
    if any(status.get(k, 0) != 0 for k in ("actual_optimizer_updates", "optimizer_updates")):
        raise ValueError("Status reports updates; export for review")
    request_path = out / "request.json"
    if request_path.exists():
        request = read(request_path)
        if (
            request.get("schema") != "babel-recovery-provenance-v1"
            or request.get("source") != str(source)
            or request.get("bundle_source") != str(bundle)
        ):
            raise ValueError("Old request source/bundle identity differs; export for review")
        modules = request.get("implementation", {}).get("modules", {})
        for name in ("recovery_provenance_run.py", "recovery_provenance.py"):
            if modules.get(name) != digest(Path(__file__).with_name(name)):
                raise ValueError(
                    "Unknown audit implementation; cannot establish preflight-only failure"
                )
    elif {p.name for p in files} - {"session.lock", "status.json"} or status.get(
        "error"
    ) != "V14 real-weight execution requires AutoDL CUDA":
        raise ValueError("Missing original request; zero-update failure cannot be established")
    return {
        "reason": "failed_before_scaler_results_and_worker_creation",
        "files": {p.name: digest(p) for p in files},
        "original_status": status,
        "optimizer_updates": 0,
    }


def launch(source, bundle, out, runner=subprocess.run):
    # Check the unresolved final component before resolve can conceal a symlink.
    if out.is_symlink():
        raise ValueError("Symlink output not supported")
    source, bundle, out = (p.resolve() for p in (source, bundle, out))
    for p in (source, bundle):
        if out == p or out in p.parents or p in out.parents:
            raise ValueError("Audit output must be separate from source and bundle")
    out.parent.mkdir(parents=True, exist_ok=True)
    backup = out.with_name(out.name + ".startup_failure_v1")
    receipt_path = out.with_name(out.name + ".startup_receipt.json")
    lock_path = out.with_name("." + out.name + ".startup.lock")
    with ExitStack() as stack:
        lock = stack.enter_context(lock_path.open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {
            "task_ids": ["V14"],
            "source": str(source),
            "bundle": str(bundle),
            "output": str(out),
            "launcher_sha256": digest(__file__),
            "started_unix": time.time(),
            "status": "checking",
            "threads": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        }
        if receipt_path.exists():
            receipt["previous_receipt"] = read(receipt_path)
        try:
            if backup.exists() or backup.is_symlink():
                raise ValueError(
                    "One startup replacement already used; export for review, no further retry"
                )
            if out.exists():
                existing_lock = out / "session.lock"
                if existing_lock.is_symlink():
                    raise ValueError("Symlink session lock not supported")
                if existing_lock.exists():
                    handle = stack.enter_context(existing_lock.open("r+"))
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                evidence = inspect_existing(out, source, bundle)
                receipt["previous_startup"] = dict(evidence, path=str(backup))
                # Reserve a single deterministic backup name, never delete or overwrite it.
                out.rename(backup)
                print(f"V14: preserved zero-update startup files at {backup}", flush=True)
            receipt["status"] = "launching_original_audit"
            write(receipt_path, receipt)
            command = [
                sys.executable,
                "-m",
                "obson.babel.recovery_provenance_run",
                "--source",
                str(source),
                "--bundle-source",
                str(bundle),
                "--out",
                str(out),
            ]
            code = runner(command).returncode
            receipt.update(status="audit_exited", command_exit_code=code)
            return code
        except BaseException as error:
            receipt.update(status="startup_blocked", error=str(error))
            raise
        finally:
            receipt["elapsed_seconds"] = time.time() - receipt["started_unix"]
            # Keep this attempt receipt separate from the old immutable audit directory.
            write(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.source, args.bundle_source, args.out))


if __name__ == "__main__":
    main()
