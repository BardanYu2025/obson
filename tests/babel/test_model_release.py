"""Storage failures must not masquerade as recoverable or scientifically accepted models."""

import fcntl
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from obson.babel import model_release as release


def fixture_spec(tmp):
    source = tmp / "source"
    source.mkdir()
    (source / "run.lock").touch()
    files, models = {}, []

    def asset(name, roles, value=b"deliberately not a torch pickle"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        files[name] = {
            "source": str(path),
            "roles": roles,
            "sha256": release.digest(path),
            "bytes": path.stat().st_size,
        }

    for seed in (42, 43):
        best, last = f"models/{seed}/best.pt", f"models/{seed}/last.pt"
        asset(best, ["weights"])
        asset(last, ["weights", "resume_state"])
        models.append(
            {"id": f"macro_s{seed}", "seed": seed, "best": best, "last": last, "resume": last}
        )
    for role in sorted(release.REQUIRED_ROLES - {"weights", "resume_state"}):
        asset(f"assets/{role}.json", [role], b'{"synthetic_fixture_only":true}\n')
    spec = {
        "schema": release.SCHEMA,
        "release_id": "macro-synthetic-v1",
        "source_commit": "a" * 40,
        "run_status": "complete",
        "limitations": "Synthetic byte fixture; no real model or scientific acceptance.",
        "files": files,
        "models": models,
        "lock_files": [str(source / "run.lock")],
    }
    path = tmp / "spec.json"
    path.write_bytes(release.json_bytes(spec))
    return path, spec


def published(tmp):
    spec, value = fixture_spec(tmp)
    out = tmp / "release"
    result = release.pack(spec, out)
    return out, result, value


def rewrite_archive(source, target, change):
    with tarfile.open(source, "r:gz") as tar:
        contents = [(m, tar.extractfile(m).read()) for m in tar.getmembers()]
    contents = change(contents)
    with tarfile.open(target, "w:gz") as tar:
        for member, value in contents:
            if member.isfile():
                member.size = len(value)
            tar.addfile(member, io.BytesIO(value) if member.isfile() else None)
    return release.digest(target)


def test_roundtrip_both_seeds_exact_bytes_and_no_model_claim(tmp_path):
    out, receipt, spec = published(tmp_path)
    target = tmp_path / "restored"
    restored = release.scan(out / "model.tar.gz", receipt["archive_sha256"], target)
    assert restored["integrity"] == "passed"
    assert restored["runtime_restore"] == "not_tested_by_archiver"
    assert restored["scientific_acceptance"] == "not_assessed_by_archiver"
    assert "requires_review" in restored["off_instance_backup"]
    for name, info in spec["files"].items():
        assert (target / name).read_bytes() == Path(info["source"]).read_bytes()
    meta = json.loads((target / "snapshot_manifest.json").read_text())
    assert [m["seed"] for m in meta["models"]] == [42, 43]
    assert all("source" not in f for f in meta["files"].values())
    assert (out / "restore.py").read_bytes() == (target / release.TOOL).read_bytes()
    assert receipt["archive_sha256"] == (out / "model.tar.gz.sha256").read_text().split()[0]


def test_offline_standalone_clean_process_needs_no_repo_or_third_party(tmp_path):
    out, receipt, _ = published(tmp_path)
    target, proof = tmp_path / "empty-machine", tmp_path / "download_verified.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(out / "restore.py"),
            "restore",
            "--archive",
            str(out / "model.tar.gz"),
            "--sha256",
            receipt["archive_sha256"],
            "--target",
            str(target),
            "--receipt",
            str(proof),
        ],
        cwd=tmp_path,
        env={"PATH": os.defpath},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == json.loads(proof.read_text())
    assert (target / "models/43/last.pt").is_file()


def test_deterministic_archive_and_immutable_publication(tmp_path):
    out, receipt, _ = published(tmp_path)
    second = release.pack(tmp_path / "spec.json", tmp_path / "second")
    assert second["archive_sha256"] == receipt["archive_sha256"]
    with pytest.raises(FileExistsError):
        release.pack(tmp_path / "spec.json", out)
    assert release.digest(out / "model.tar.gz") == receipt["archive_sha256"]
    target = tmp_path / "existing"
    target.mkdir()
    (target / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        release.scan(out / "model.tar.gz", receipt["archive_sha256"], target)
    assert (target / "user.txt").read_text() == "keep"


@pytest.mark.parametrize(
    "kind", ["missing_role", "missing_seed_weight", "source_changed", "source_missing", "unlocked"]
)
def test_incomplete_or_stale_spec_cannot_publish(tmp_path, kind):
    path, spec = fixture_spec(tmp_path)
    if kind == "missing_role":
        del spec["files"]["assets/preprocessing.json"]
    elif kind == "missing_seed_weight":
        del spec["files"]["models/43/best.pt"]
    elif kind == "source_changed":
        Path(spec["files"]["models/42/best.pt"]["source"]).write_bytes(b"changed")
    elif kind == "source_missing":
        Path(spec["files"]["models/42/best.pt"]["source"]).unlink()
    else:
        spec["lock_files"] = []
    path.write_bytes(release.json_bytes(spec))
    with pytest.raises((ValueError, FileNotFoundError)):
        release.pack(path, tmp_path / "release")
    assert not (tmp_path / "release/release_receipt.json").exists()


def test_live_worker_lock_rejects_export(tmp_path):
    path, spec = fixture_spec(tmp_path)
    with Path(spec["lock_files"][0]).open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            release.pack(path, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_source_mutation_mid_export_never_publishes_receipt(tmp_path, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    original = release.add_file

    def mutate(tar, name, size, stream):
        original(tar, name, size, stream)
        if name == "payload/models/42/best.pt":
            Path(spec["files"]["models/42/best.pt"]["source"]).write_bytes(b"new run")

    monkeypatch.setattr(release, "add_file", mutate)
    with pytest.raises(ValueError, match="changed during"):
        release.pack(path, tmp_path / "release")
    assert not list((tmp_path / "release").iterdir())


@pytest.mark.parametrize(
    "kind",
    [
        "archive_hash",
        "file_hash",
        "missing",
        "duplicate",
        "symlink",
        "hardlink",
        "traversal",
        "extra",
        "reserved",
    ],
)
def test_corruption_and_unsafe_members_fail_without_partial_restore(tmp_path, kind):
    out, receipt, _ = published(tmp_path)
    damaged, target = tmp_path / "damaged.tar.gz", tmp_path / "target"

    def change(parts):
        member, content = parts[-1]
        if kind == "file_hash":
            parts[-1] = member, b"x" * len(content)
        elif kind == "missing":
            parts.pop()
        elif kind == "duplicate":
            parts.append(parts[-1])
        elif kind in {"symlink", "hardlink"}:
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../../outside"
        elif kind == "traversal":
            member.name = "payload/../../outside"
        elif kind == "extra":
            member.name = "payload/unlisted.txt"
        elif kind == "reserved":
            first, payload = parts[0]
            meta = json.loads(payload)
            meta["files"]["snapshot_manifest.json"] = meta["files"].pop(release.TOOL)
            parts[0] = first, release.json_bytes(meta)
        return parts

    checksum = rewrite_archive(out / "model.tar.gz", damaged, change)
    if kind == "archive_hash":
        checksum = "0" * 64
    with pytest.raises(ValueError):
        release.scan(damaged, checksum, target)
    assert not target.exists()
    assert not (tmp_path / "outside").exists()


def test_source_and_destination_symlinks_rejected(tmp_path):
    path, spec = fixture_spec(tmp_path)
    source = Path(spec["files"]["models/42/best.pt"]["source"])
    moved = source.with_suffix(".original")
    source.rename(moved)
    source.symlink_to(moved)
    with pytest.raises(ValueError, match="Symlink"):
        release.pack(path, tmp_path / "release")
    source.unlink()
    moved.rename(source)
    out = tmp_path / "release"
    receipt = release.pack(path, out)
    target = tmp_path / "linked"
    target.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        release.scan(out / "model.tar.gz", receipt["archive_sha256"], target)


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_failed_or_stopped_snapshot_keeps_status(tmp_path, status):
    path, spec = fixture_spec(tmp_path)
    spec["run_status"] = status
    path.write_bytes(release.json_bytes(spec))
    receipt = release.pack(path, tmp_path / "release")
    assert receipt["run_status"] == status
    assert receipt["scientific_acceptance"] == "not_assessed_by_archiver"


def test_receipt_collision_is_not_overwritten(tmp_path):
    out, receipt, _ = published(tmp_path)
    proof = tmp_path / "proof.json"
    proof.write_text("prior evidence")
    run = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(out / "restore.py"),
            "verify",
            "--archive",
            str(out / "model.tar.gz"),
            "--sha256",
            receipt["archive_sha256"],
            "--receipt",
            str(proof),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0
    assert proof.read_text() == "prior evidence"


def test_empty_python_package_file_is_preserved(tmp_path):
    path, spec = fixture_spec(tmp_path)
    init = tmp_path / "source/__init__.py"
    init.touch()
    spec["files"]["source/__init__.py"] = {
        "source": str(init),
        "roles": ["code"],
        "bytes": 0,
        "sha256": release.digest(init),
    }
    path.write_bytes(release.json_bytes(spec))
    out, target = tmp_path / "release", tmp_path / "restored"
    receipt = release.pack(path, out)
    release.scan(out / "model.tar.gz", receipt["archive_sha256"], target)
    assert (target / "source/__init__.py").read_bytes() == b""


@pytest.mark.parametrize("kind", ["active", "empty_weight", "collision", "noncanonical_path"])
def test_invalid_release_contract_rejected(tmp_path, kind):
    path, spec = fixture_spec(tmp_path)
    if kind == "active":
        spec["run_status"] = "running"
    elif kind == "empty_weight":
        spec["files"]["models/42/best.pt"]["bytes"] = 0
    elif kind == "collision":
        spec["files"]["models"] = spec["files"]["models/42/best.pt"]
    else:
        spec["files"]["x/./y"] = spec["files"]["models/42/best.pt"]
    path.write_bytes(release.json_bytes(spec))
    with pytest.raises(ValueError):
        release.pack(path, tmp_path / "release")


def test_truncated_archive_has_no_successful_restore(tmp_path):
    out, _, _ = published(tmp_path)
    broken = tmp_path / "truncated.tar.gz"
    payload = (out / "model.tar.gz").read_bytes()
    broken.write_bytes(payload[: len(payload) // 2])
    with pytest.raises((ValueError, EOFError, tarfile.TarError)):
        release.scan(broken, release.digest(broken), tmp_path / "target")
    assert not (tmp_path / "target").exists()
