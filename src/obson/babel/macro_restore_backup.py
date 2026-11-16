"""Automatic reports and complete per-stage snapshots for Macro recovery.

No checkpoint loading here. Completed snapshots require real CUDA receipts;
interrupted snapshots preserve their failed/stopped status and available seeds.
"""

import argparse
import fcntl
import hashlib
import io
import json
import tarfile
import tempfile
import time
from pathlib import Path

from . import macro_restore_data as data
from . import model_release
from .ae_extend import atomic_json
from .raw_snapshot import digest, no_links


def add(files, name, path, roles):
    path = no_links(path)
    files[name] = {
        "source": str(path),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "roles": roles,
    }


def stage_snapshot(out, stage, download, status):
    """Every stage calls the same backup contract; only s01 currently has a trainer."""
    meta = data.read(out / "manifest.json")
    stage_root = out / "stages" / stage
    files, models = {}, []
    for seed in meta["stage"]["seeds"]:
        root = stage_root / f"s{seed}"
        if not (root / "last.pt").exists():
            continue
        for name in ("last.pt", "best.pt", "history.json", "example_inputs.npz"):
            if not (root / name).is_file():
                raise ValueError(f"Cannot publish partial checkpoint transaction: {root / name}")
        prefix = f"run/stages/{stage}/s{seed}/"
        for name, roles in (
            ("last.pt", ["weights", "resume_state"]),
            ("best.pt", ["weights"]),
            ("history.json", ["evaluation"]),
            ("example_inputs.npz", ["example"]),
        ):
            add(files, prefix + name, root / name, roles)
        if (root / "completion.json").exists():
            done = data.read(root / "completion.json")
            if done["status"] != "passed" or done["binding"]["run_manifest"] != digest(
                out / "manifest.json"
            ):
                raise ValueError("Stage CUDA restoration receipt does not bind this run")
            for name, checksum in done["files"].items():
                if digest(root / name) != checksum:
                    raise ValueError("Completed stage artifact changed")
            add(files, prefix + "completion.json", root / "completion.json", ["evaluation"])
            add(files, prefix + "examples.npz", root / "examples.npz", ["example"])
            add(
                files,
                prefix + "continuation.pt",
                root / "continuation.pt",
                ["example", "resume_state"],
            )
        elif status == "complete":
            raise ValueError("Completed snapshot requires clean CUDA restore receipt")
        models.append(
            {
                "id": f"{stage}_s{seed}",
                "seed": seed,
                "best": prefix + "best.pt",
                "last": prefix + "last.pt",
                "resume": prefix + "last.pt",
            }
        )
    if not models:
        return {"status": "no_model_weights_yet", "stage": stage}
    if status == "complete" and len(models) != len(meta["stage"]["seeds"]):
        raise ValueError("Complete stage must preserve both declared seeds")
    add(files, "run/manifest.json", out / "manifest.json", ["architecture", "recipe"])
    add(files, "run/runtime.json", out / "runtime.json", ["environment"])
    # Preserve consumed budget without treating time in offline storage as training.
    # The live journal is never modified here; this is a checkpoint of its usage.
    usage = data.read(out / "usage.json")
    if usage["active_since"] is not None:
        usage["used_seconds"] += max(0.0, time.time() - usage["active_since"])
    usage["active_since"] = None
    state_root = out / "backup_state"
    state_root.mkdir(exist_ok=True)
    usage_path = state_root / "usage.json"
    atomic_json(usage, usage_path)
    add(files, "run/usage.json", usage_path, ["resume_state"])
    if (out / "status.json").exists():
        add(files, "run/status.json", out / "status.json", ["evaluation"])
    receipt = data.verify_prepared(out / "data")
    add(files, "run/data/data_receipt.json", out / "data/data_receipt.json", ["data"])
    for name in receipt["files"]:
        add(files, "run/data/" + name, out / "data" / name, ["preprocessing", "data"])
    # All consumed source code and raw evidence ship with the model; no old absolute paths.
    for name, checksum in meta["binding"]["files"].items():
        path = data.REPO / name
        if digest(path) != checksum:
            raise ValueError(f"Bound code/data changed before backup: {name}")
        add(
            files,
            "source_code/" + name,
            path,
            ["code"] if name.startswith(("src/", "scripts/")) else ["data", "environment"],
        )
    fingerprint = hashlib.sha256(
        json.dumps({"files": files, "status": status}, sort_keys=True).encode()
    ).hexdigest()
    release_id = f"macro-{stage}-{fingerprint[:20]}"
    specification = {
        "schema": model_release.SCHEMA,
        "release_id": release_id,
        "source_commit": meta["binding"]["git_commit"],
        "run_status": status,
        "limitations": "Selected-recipe ancestor recovery only. Final Macro is NOT restored by stage01. PCA is rederived. Runtime receipt covers strict load, full validation and optimizer/RNG restoration; independent storage still requires local receipt review.",
        "models": models,
        "files": files,
        "lock_files": [str(stage_root / "stage.lock")],
    }
    spec_dir = out / "backup_specs"
    spec_dir.mkdir(exist_ok=True)
    spec = spec_dir / f"{release_id}.json"
    if spec.exists() and data.read(spec) != specification:
        raise ValueError("Immutable backup specification changed")
    atomic_json(specification, spec)
    destination = download / out.name / release_id
    if destination.exists() and not (destination / "release_receipt.json").exists():
        # A killed publication is not a version. Preserve its bytes for diagnosis;
        # never overwrite it or mistake an unfinished archive for a verified copy.
        no_links(destination).rename(
            destination.with_name(destination.name + f".incomplete-{time.time_ns()}")
        )
    if destination.exists():
        saved = data.read(destination / "release_receipt.json")
        checked = model_release.scan(destination / "model.tar.gz", saved["archive_sha256"])
        if checked["release_id"] != release_id:
            raise ValueError("Existing snapshot identity differs")
    else:
        checked = model_release.pack(spec, destination)
    atomic_json(
        {
            "stage": stage,
            "location": str(destination),
            "archive_sha256": checked["archive_sha256"],
            "off_instance": "pending_download_and_local_verification",
        },
        out / "backup_receipt.json",
    )
    print(
        f"Full snapshot: {destination} (snapshot_status={status}; off-instance verification pending)",
        flush=True,
    )
    return checked


def reports(out, download, code, backup):
    download.mkdir(parents=True, exist_ok=True)
    selected = {}
    if out.exists():
        for path in out.rglob("*"):
            if (
                path.is_file()
                and path.suffix in (".json", ".jsonl", ".log")
                and "backup_specs" not in path.parts
            ):
                no_links(path)
                selected[str(path.relative_to(out))] = path
    metadata = {
        "schema": "babel-macro-recovery-reports-v1",
        "run": out.name,
        "command_exit_code": code,
        "status": data.read(out / "status.json")
        if (out / "status.json").exists()
        else {"status": "failed_before_initialization"},
        "backup": backup,
        "model_weights_in_reports": False,
        "final_macro_restored": False,
        "files": {n: {"sha256": digest(p), "bytes": p.stat().st_size} for n, p in selected.items()},
    }
    destination = download / f"{out.name}_reports.tar.gz"
    with tempfile.NamedTemporaryFile(dir=download, suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        with tarfile.open(temp, "w:gz") as tar:
            for name, path in selected.items():
                tar.add(path, arcname=f"{out.name}/{name}", recursive=False)
                if digest(path) != metadata["files"][name]["sha256"]:
                    raise ValueError("Report changed while exporting")
            content = model_release.json_bytes(metadata)
            member = tarfile.TarInfo("export_manifest.json")
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    destination.with_name(destination.name + ".sha256").write_text(
        f"{digest(destination)}  {destination.name}\n"
    )
    print(
        f"Download reports: {destination} (exit={code}; final Macro not yet restored)", flush=True
    )


def export(out, download, code):
    out, download = no_links(out), no_links(download)
    if out == download or download.is_relative_to(out) or out.is_relative_to(download):
        raise ValueError("Download directory must be separate from run")
    backup, error = None, None
    if out.is_dir():
        with (out / "run.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if (out / "manifest.json").exists():
                    meta = data.read(out / "manifest.json")
                    status = (
                        data.read(out / "status.json").get("status")
                        if (out / "status.json").exists()
                        else "failed"
                    )
                    backup = stage_snapshot(
                        out,
                        meta["stage"]["stage"],
                        download,
                        "complete"
                        if status == "stage01_complete" and code == 0
                        else "stopped"
                        if status == "stopped" or code in (130, 143)
                        else "failed",
                    )
            except Exception as exc:
                error = exc
                backup = {"status": "backup_failed", "error": str(exc)}
            reports(out, download, code, backup)
    else:
        reports(out, download, code, backup)
    if error is not None:
        raise error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--download", type=Path, required=True)
    p.add_argument("--exit-code", type=int, default=0)
    a = p.parse_args()
    export(a.out, a.download, a.exit_code)


if __name__ == "__main__":
    main()
