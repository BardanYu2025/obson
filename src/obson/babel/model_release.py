"""Freeze and restore a fingerprinted model snapshot; Python standard library only.

This checks bytes, not model quality, checkpoint contents or CUDA reproducibility.
The same file is shipped as restore.py so recovery does not depend on this repo.
"""

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import socket
import tarfile
import tempfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SCHEMA = "babel-model-snapshot-v1"
REQUIRED_ROLES = {
    "weights",
    "resume_state",
    "architecture",
    "preprocessing",
    "data",
    "code",
    "environment",
    "recipe",
    "evaluation",
    "example",
}
TOOL = "tools/restore.py"
MAX_MANIFEST = 16 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def relative(name):
    if not isinstance(name, str):
        raise ValueError("File name must be a string")
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or ".." in p.parts
        or str(p) != name
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 for c in name)
        or name == "."
    ):
        raise ValueError(f"Unsafe file name: {name!r}")
    return p


def no_links(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink or parent traversal: {path}")
    return path


def valid_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a full lowercase SHA256")
    return value


def validate(meta):
    if not isinstance(meta, dict):
        raise ValueError("Invalid snapshot manifest")
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unsupported model snapshot schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", meta.get("release_id", "")):
        raise ValueError("Invalid release_id")
    if not re.fullmatch(r"[0-9a-f]{40}", meta.get("source_commit", "")):
        raise ValueError("Record the full source Git commit")
    if meta.get("run_status") not in {"complete", "failed", "stopped"}:
        raise ValueError("Record complete/failed/stopped; active runs cannot be frozen")
    if not isinstance(meta.get("limitations"), str) or not meta["limitations"].strip():
        raise ValueError("Record the model's scope and known limitations")
    files, roles = meta.get("files"), set()
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing file inventory")
    for name, info in files.items():
        p = relative(name)
        if p.parts[0] == "snapshot_manifest.json":
            raise ValueError("Reserved snapshot file name")
        if any(str(parent) in files for parent in p.parents):
            raise ValueError("File/directory name collision")
        if not isinstance(info, dict):
            raise ValueError("Invalid file record")
        valid_sha(info.get("sha256"))
        if type(info.get("bytes")) is not int or info["bytes"] < 0:
            raise ValueError(f"Invalid asset size: {name}")
        if (
            not isinstance(info.get("roles"), list)
            or not info["roles"]
            or any(not isinstance(role, str) for role in info["roles"])
        ):
            raise ValueError("Each file needs explicit roles")
        roles.update(info["roles"])
    if missing := REQUIRED_ROLES - roles:
        raise ValueError(f"Missing asset categories: {sorted(missing)}")
    models = meta.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Declare every model/seed being preserved")
    ids = set()
    for model in models:
        name = model.get("id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Missing or duplicate model identity")
        ids.add(name)
        if type(model.get("seed")) is not int:
            raise ValueError("Missing model seed")
        for field, role in (("best", "weights"), ("last", "weights"), ("resume", "resume_state")):
            entry = files.get(model.get(field))
            if entry is None or role not in entry["roles"] or entry["bytes"] == 0:
                raise ValueError(f"Missing {field} asset for {name}")


def receipt(meta, archive, checksum, target=None):
    return {
        "schema": SCHEMA + "-receipt",
        "release_id": meta["release_id"],
        "source_commit": meta["source_commit"],
        "run_status": meta["run_status"],
        "archive": str(archive),
        "archive_sha256": checksum,
        "manifest_sha256": hashlib.sha256(json_bytes(meta)).hexdigest(),
        "file_count": len(meta["files"]),
        "payload_bytes": sum(f["bytes"] for f in meta["files"].values()),
        "integrity": "passed",
        "restore_directory": str(target) if target else None,
        "verified_host": socket.gethostname(),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "runtime_restore": "not_tested_by_archiver",
        "scientific_acceptance": "not_assessed_by_archiver",
        "off_instance_backup": "location_requires_review; a hostname or download path is not proof",
    }


def scan(archive, checksum, target=None):
    """Read every member without extractall, pickle, torch, or executing packed code."""
    archive = no_links(archive)
    valid_sha(checksum)
    if digest(archive) != checksum:
        raise ValueError("Archive SHA256 mismatch")
    target = no_links(target) if target is not None else None
    created = False
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            first = tar.next()
            if (
                first is None
                or first.name != "manifest.json"
                or not first.isfile()
                or first.sparse is not None
                or not 0 < first.size <= MAX_MANIFEST
            ):
                raise ValueError("Manifest must be the first regular member")
            meta = json.loads(tar.extractfile(first).read())
            validate(meta)
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir()  # Never overlay even an empty existing directory.
                created = True
            seen = set()
            while member := tar.next():
                relative(member.name)
                if not member.name.startswith("payload/"):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                name = member.name.removeprefix("payload/")
                entry = meta["files"].get(name)
                if (
                    entry is None
                    or name in seen
                    or not member.isfile()
                    or member.sparse is not None
                    or member.size != entry["bytes"]
                ):
                    raise ValueError(f"Unexpected/duplicate/nonregular/size mismatch: {name}")
                h = hashlib.sha256()
                with ExitStack() as stack:
                    source = stack.enter_context(tar.extractfile(member))
                    output = None
                    if target is not None:
                        dest = no_links(target / name)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        output = stack.enter_context(dest.open("xb"))
                    for block in iter(lambda source=source: source.read(1024 * 1024), b""):
                        h.update(block)
                        if output is not None:
                            output.write(block)
                if h.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Asset SHA256 mismatch: {name}")
                if target is not None and digest(dest) != entry["sha256"]:
                    raise ValueError(f"Restored file SHA256 mismatch: {name}")
                seen.add(name)
            if seen != set(meta["files"]):
                raise ValueError("Missing archive assets")
        if digest(archive) != checksum:
            raise ValueError("Archive changed during verification")
        if target is not None:
            # Retain the binding in the extracted directory as well as the archive.
            (target / "snapshot_manifest.json").write_bytes(json_bytes(meta))
        return receipt(meta, archive, checksum, target)
    except BaseException:
        if created:
            shutil.rmtree(target)
        raise


def add_file(tar, name, size, stream):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o644, 0
    tar.addfile(info, stream)


def pack(spec_path, out):
    """Spec is an explicit allowlist with pre-existing hashes, never a folder glob."""
    spec_path, out = no_links(spec_path), no_links(out)
    spec = json.loads(spec_path.read_text())
    validate(spec)
    if TOOL in spec["files"] or "snapshot_manifest.json" in spec["files"]:
        raise ValueError("Reserved snapshot file name")
    if not isinstance(spec.get("lock_files"), list) or not spec["lock_files"]:
        raise ValueError("Declare the source run locks; unlocked export is forbidden")
    files = {}
    for name, info in spec["files"].items():
        if not isinstance(info.get("source"), str) or not info["source"]:
            raise ValueError(f"Missing source file: {name}")
        files[name] = no_links(spec_path.parent / info["source"])
    tool = no_links(Path(__file__))
    meta = {
        k: spec[k]
        for k in ("schema", "release_id", "source_commit", "run_status", "limitations", "models")
    }
    meta["files"] = {
        name: {key: info[key] for key in ("sha256", "bytes", "roles")}
        for name, info in spec["files"].items()
    }
    meta["files"][TOOL] = {
        "sha256": digest(tool),
        "bytes": tool.stat().st_size,
        "roles": ["verification_tool"],
    }
    files[TOOL] = tool
    validate(meta)
    payload = json_bytes(meta)
    if len(payload) > MAX_MANIFEST:
        raise ValueError("Manifest too large")
    with ExitStack() as stack:
        for name in spec["lock_files"]:
            lock = stack.enter_context(no_links(spec_path.parent / name).open("rb"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, path in files.items():
            info = meta["files"][name]
            if (
                not path.is_file()
                or path.stat().st_size != info["bytes"]
                or digest(path) != info["sha256"]
            ):
                raise ValueError(f"Source fingerprint mismatch: {name}")
            if path == out or path.is_relative_to(out):
                raise ValueError("Snapshot output must be separate from input files")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir()  # Immutable publication: a second invocation never replaces it.
        final = out / "model.tar.gz"
        with tempfile.NamedTemporaryFile(dir=out, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
        try:
            with (
                temporary.open("wb") as f,
                gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as compressed,
                tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar,
            ):
                add_file(tar, "manifest.json", len(payload), io.BytesIO(payload))
                for name, path in sorted(files.items()):
                    with path.open("rb") as source:
                        add_file(tar, "payload/" + name, meta["files"][name]["bytes"], source)
                    if digest(path) != meta["files"][name]["sha256"]:
                        raise ValueError(f"Source changed during publication: {name}")
            checksum = digest(temporary)
            checked = scan(temporary, checksum)
            os.replace(temporary, final)
            checked["archive"] = str(final)
            (out / "model.tar.gz.sha256").write_text(f"{checksum}  model.tar.gz\n")
            (out / "restore.py").write_bytes(tool.read_bytes())
            if digest(out / "restore.py") != meta["files"][TOOL]["sha256"]:
                raise ValueError("Verification tool changed during publication")
            # Last file is the publication receipt; incomplete exports have no receipt.
            (out / "release_receipt.json").write_bytes(json_bytes(checked))
            return checked
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack", help="Freeze a pre-fingerprinted spec (no training)")
    packing.add_argument("--spec", type=Path, required=True)
    packing.add_argument("--out", type=Path, required=True)
    for name in ("verify", "restore"):
        p = commands.add_parser(
            name, help="Check trusted SHA256 and every asset (no model loading)"
        )
        p.add_argument("--archive", type=Path, required=True)
        p.add_argument("--sha256", required=True, help="Expected SHA256 retained from publication")
        p.add_argument("--receipt", type=Path)
        if name == "restore":
            p.add_argument("--target", type=Path, required=True, help="Must be a new directory")
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(args.spec, args.out)
    else:
        destination = no_links(args.receipt) if args.receipt else None
        if destination is not None and destination.exists():
            raise ValueError("Refusing to overwrite a verification receipt")
        result = scan(args.archive, args.sha256, getattr(args, "target", None))
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as f:
                f.write(json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
