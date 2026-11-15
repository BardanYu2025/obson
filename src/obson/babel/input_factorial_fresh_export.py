"""Export review reports AND a full recovery backup, without mutating run evidence."""

import argparse
import fcntl
import io
import json
import tarfile
import tempfile
from contextlib import ExitStack
from pathlib import Path

from . import input_factorial_export as filtering
from . import input_factorial_fresh_run as run
from .raw_snapshot import digest, no_links, safe_relative


def collect(out, *, backup):
    selected = {}
    if not out.is_dir():
        return selected
    prefix = f"obson/checkpoints/{out.name}" if backup else out.name
    for p in sorted(out.rglob("*")):
        no_links(p)
        if not p.is_file() or p.suffix == ".lock" or ".tmp" in p.name:
            continue
        relative = str(p.relative_to(out))
        if backup or filtering._disposition(p, relative)[0]:
            selected[f"{prefix}/{relative}"] = p
    if backup and (out / "manifest.json").is_file():
        meta = json.loads((out / "manifest.json").read_text())
        source = no_links(meta["source"])
        for p in sorted(source.rglob("*")):
            no_links(p)
            if p.is_file():
                selected["data/factorial_fresh/" + str(p.relative_to(source))] = p
        for name in meta["implementation"]:
            selected["source_code/" + name] = no_links(run.REPO / safe_relative(name))
    return selected


def archive(paths, destination, metadata):
    inventory = {
        name: {"sha256": digest(p), "bytes": p.stat().st_size} for name, p in paths.items()
    }
    evidence = dict(metadata, files=inventory)
    payload = (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        with tarfile.open(temporary, "w:gz") as tar:
            for name, p in paths.items():
                tar.add(p, arcname=name, recursive=False)
                if digest(p) != inventory[name]["sha256"]:
                    raise ValueError("Run changed while exporting; original archive retained")
            info = tarfile.TarInfo(
                "backup_manifest.json"
                if metadata["kind"] == "full_backup"
                else "export_manifest.json"
            )
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = digest(destination)
    destination.with_name(destination.name + ".sha256").write_text(
        f"{checksum}  {destination.name}\n"
    )
    print(
        f"Download {metadata['kind']}: {destination} (status={metadata['run_status']}, sha256={checksum})",
        flush=True,
    )


def export(out, download, code=0):
    out, download = no_links(out), no_links(download)
    if out == download or out.is_relative_to(download) or download.is_relative_to(out):
        raise ValueError("Download must be separate from the experiment")
    download.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Same lock held by the supervisor. Never publish a live-changing model.
        if out.is_dir():
            lock = stack.enter_context((out / "run.lock").open("a+"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = (
            json.loads((out / "status.json").read_text())
            if (out / "status.json").is_file()
            else {"status": "failed_before_run"}
        )
        if code:
            status = status | {"status": "failed_or_stopped", "command_exit_code": code}
        complete = False
        if (out / "completion.json").is_file() and code == 0:
            meta = json.loads((out / "manifest.json").read_text())
            try:
                run.check_completion(meta, out)
                complete = True
                status["status"] = "complete_requires_review"
            except (ValueError, KeyError, OSError) as error:
                status = {"status": "completion_verification_failed", "error": str(error)}
        metadata = {
            "schema": "babel-input-factorial-fresh-export-v1",
            "run": str(out),
            "run_status": status["status"],
            "command_exit_code": code,
            "execution_complete": complete,
            "scientific_acceptance": "pending_review" if complete else "not_established",
            "historical_replay": False,
            "status_detail": status,
            "recovery": "Full backup includes all committed weights/Adam/RNG, evidence, prediction shards, pending journals, fresh dataset and bound code. Raw CSVs are in the git data snapshot. Download this file off AutoDL before deletion.",
        }
        archive(
            collect(out, backup=False),
            download / f"{out.name}_reports.tar.gz",
            dict(metadata, kind="reports"),
        )
        archive(
            collect(out, backup=True),
            download / f"{out.name}_backup.tar.gz",
            dict(metadata, kind="full_backup"),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--download", type=Path, required=True)
    p.add_argument("--command-exit-code", type=int, default=0)
    a = p.parse_args()
    export(a.out, a.download, a.command_exit_code)


if __name__ == "__main__":
    main()
