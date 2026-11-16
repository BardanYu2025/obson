"""Bounded Macro recovery bootstrap: exact data/PCA qualification, then stage01 only."""

import argparse
import fcntl
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import macro_restore_data as data
from . import macro_restore_stage as stage
from .ae_extend import atomic_json
from .progress import progress
from .raw_snapshot import digest, no_links


def binding():
    paths = list((data.REPO / "src/obson").rglob("*.py"))
    paths += [
        data.REPO / "scripts/babel_macro_restore_autodl.sh",
        data.REPO / "requirements-babel-restore.txt",
    ]
    paths += [p for folder in (data.ASSETS, data.SNAPSHOT) for p in folder.iterdir() if p.is_file()]
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=data.REPO, text=True
        ).strip(),
        "files": {str(p.relative_to(data.REPO)): digest(p) for p in sorted(paths)},
    }


def verify_binding(meta):
    for name, checksum in meta["binding"]["files"].items():
        if digest(no_links(data.REPO / name)) != checksum:
            raise ValueError(f"Frozen recovery code/data changed: {name}")


def usage_start(out, cap):
    path = out / "usage.json"
    value = data.read(path) if path.exists() else {"used_seconds": 0.0, "active_since": None}
    if value["active_since"] is not None:
        # A killed process cannot silently erase consumed walltime. Downtime is conservatively charged.
        value["used_seconds"] += max(0.0, time.time() - value["active_since"])
    if value["used_seconds"] >= cap:
        raise ValueError("Stage01 cumulative12h budget exhausted; no automatic extension")
    value["active_since"] = time.time()
    atomic_json(value, path)
    return time.monotonic() + cap - value["used_seconds"], value


def usage_end(out, value):
    value["used_seconds"] += max(0.0, time.time() - value["active_since"])
    value["active_since"] = None
    atomic_json(value, out / "usage.json")


def run_child(out, seed, deadline, check_only=False):
    cmd = [
        sys.executable,
        "-m",
        "obson.babel.macro_restore_stage",
        "--out",
        str(out),
        "--seed",
        str(seed),
    ]
    if check_only:
        cmd.append("--check-only")
    process = subprocess.Popen(cmd)
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Stage01 cumulative runtime cap reached")
            time.sleep(1)
        if process.returncode:
            raise RuntimeError(f"Stage01 seed{seed} process exited{process.returncode}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run(raw, out, download, micro=16, prepare_only=False):
    raw, out, download = map(no_links, (raw, out, download))
    if out == raw or raw.is_relative_to(out) or out.is_relative_to(raw):
        raise ValueError("Use a separate recovery run directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = None
        previous_alarm = None
        try:
            contract = data.source_contract()
            previous = (
                data.read(out / "manifest.json") if (out / "manifest.json").exists() else None
            )
            if previous is not None:
                # A complete snapshot includes source bytes, not a .git directory.
                verify_binding(previous)
            meta = {
                "schema": "babel-macro-recovery-bootstrap-v1",
                "binding": previous["binding"] if previous is not None else binding(),
                "stage": stage.recipe(contract, micro),
                "final_macro_restored": False,
                "remaining_stages": 12,
                "four_arm_experiment": "retained_not_started",
            }
            if previous is not None:
                if previous != meta:
                    raise ValueError("Recovery source/configuration changed; use a new run")
            elif any(p.name != "run.lock" for p in out.iterdir()):
                raise ValueError("Nonempty output has no recovery manifest")
            atomic_json(meta, out / "manifest.json")
            if shutil.disk_usage(out).free < meta["stage"]["minimum_free_bytes"]:
                raise ValueError(
                    "Stage01 needs at least12GiB free for data, checkpoints and full snapshots"
                )
            if not (out / "runtime.json").exists():
                atomic_json(
                    {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "torch": str(torch.__version__),
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                        "environment": "recorded_actual_runtime",
                        "source_commit": meta["binding"]["git_commit"],
                    },
                    out / "runtime.json",
                )
            deadline, usage = usage_start(out, meta["stage"]["session_cap_seconds"])

            def timeout(signum, frame):
                raise TimeoutError("Stage01 cumulative runtime cap reached")

            previous_alarm = signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
            atomic_json(
                {"status": "preparing_original_data", "neural_updates": 0}, out / "status.json"
            )
            prepared = data.prepare(raw, out / "data", meta["binding"])
            if not prepared["training_allowed"]:
                raise ValueError(
                    "Original input hashes differ. Data/PCA evidence saved; no model training. Send reports archive for source diagnosis."
                )
            if prepare_only:
                atomic_json(
                    {
                        "status": "original_data_qualified",
                        "neural_updates": 0,
                        "final_macro_restored": False,
                    },
                    out / "status.json",
                )
                return
            if not torch.cuda.is_available() or str(torch.__version__).split("+")[0] != "2.8.0":
                raise ValueError(
                    "Stage01 requires AutoDL CUDA PyTorch2.8.0; no CPU training fallback"
                )
            current = data.read(out / "runtime.json")
            if (
                current["torch"] != str(torch.__version__)
                or current["numpy"] != np.__version__
                or current["gpu"] != torch.cuda.get_device_name()
            ):
                raise ValueError(
                    "Runtime changed within recovery run; do not mix checkpoints silently"
                )
            root = out / "stages" / stage.STAGE
            root.mkdir(parents=True, exist_ok=True)
            (root / "stage.lock").touch(exist_ok=True)
            for seed in meta["stage"]["seeds"]:
                progress(
                    f"Stage01 seed{seed}: original scratch768 joint200; later12 stages remain queued"
                )
                atomic_json(
                    {"status": "training_stage01", "seed": seed, "final_macro_restored": False},
                    out / "status.json",
                )
                done_path = root / f"s{seed}/completion.json"
                if done_path.exists():
                    done = data.read(done_path)
                    if done["binding"]["run_manifest"] != digest(out / "manifest.json"):
                        raise ValueError("Completed worker belongs to a different recovery run")
                    for name, checksum in done["files"].items():
                        if digest(done_path.parent / name) != checksum:
                            raise ValueError("Completed worker changed; refusing to overwrite it")
                else:
                    run_child(out, seed, deadline)
                run_child(out, seed, deadline, check_only=True)
                # Do not wait for the second seed to preserve the first completed checkpoint.
                from .macro_restore_backup import stage_snapshot

                stage_snapshot(
                    out,
                    stage.STAGE,
                    download,
                    "stopped" if seed == meta["stage"]["seeds"][0] else "complete",
                )
            atomic_json(
                {
                    "status": "stage01_complete",
                    "stages_complete": 1,
                    "stages_remaining": 12,
                    "final_macro_restored": False,
                    "off_instance_backup": "pending_local_verification",
                },
                out / "status.json",
            )
        except BaseException as exc:
            atomic_json(
                {
                    "status": "stopped"
                    if isinstance(exc, (KeyboardInterrupt, TimeoutError))
                    else "failed",
                    "error": str(exc),
                    "final_macro_restored": False,
                },
                out / "status.json",
            )
            raise
        finally:
            if previous_alarm is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
            if usage is not None:
                usage_end(out, usage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--download", type=Path, required=True)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()

    def stopped(signum, frame):
        raise KeyboardInterrupt(f"signal{signum}")

    signal.signal(signal.SIGTERM, stopped)
    run(a.raw_root, a.out, a.download, a.micro, a.prepare_only)


if __name__ == "__main__":
    main()
