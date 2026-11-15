"""Portable, lossless raw/cache snapshots. Standard library only; no pickle/model use."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

SCHEMA = "babel-portable-data-v1"
CHUNK = 32 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = Path(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name or "\\" in name:
        raise ValueError(f"Unsafe snapshot member: {name}")
    return p


def no_links(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink snapshot path: {path}")
    return path


def manifest(bundle):
    bundle = no_links(bundle)
    value = json.loads((bundle / "manifest.json").read_text())
    if value.get("schema") != SCHEMA or not value.get("files") or not value.get("parts"):
        raise ValueError("Invalid snapshot manifest")
    for name, info in value["files"].items():
        safe_relative(name)
        if type(info.get("bytes")) is not int or info["bytes"] < 0:
            raise ValueError("Invalid snapshot size")
    names = [p["name"] for p in value["parts"]]
    if names != [f"payload.part{i:03d}" for i in range(len(names))]:
        raise ValueError("Snapshot parts are missing or reordered")
    for info in value["parts"]:
        p = no_links(bundle / info["name"])
        if not p.is_file() or p.stat().st_size != info["bytes"] or digest(p) != info["sha256"]:
            raise ValueError(f"Snapshot part hash/size differs: {p.name}")
    return value


def pack(raw_root, dataset, out):
    raw_root, dataset, out = map(no_links, (raw_root, dataset, out))
    if out.exists():
        raise ValueError("Snapshot output must be new")
    members = {
        "contracts/" + str(p.relative_to(raw_root)): p for p in sorted(raw_root.glob("*/*m.csv"))
    }
    members.update(
        {
            "factorial_fresh/" + str(p.relative_to(dataset)): p
            for p in sorted(dataset.rglob("*"))
            if p.is_file()
        }
    )
    if not members:
        raise ValueError("No snapshot data")
    out.mkdir(parents=True)
    files = {}
    for name, path in members.items():
        safe_relative(name)
        no_links(path)
        files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    # Fixed tar/gzip headers make the publication reproducible across hosts.
    with tempfile.TemporaryFile() as payload:
        with (
            gzip.GzipFile(fileobj=payload, mode="wb", filename="", mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar,
        ):
            for name, path in members.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = files[name]["bytes"], 0o644, 0
                with path.open("rb") as f:
                    tar.addfile(info, f)
                if digest(path) != files[name]["sha256"]:
                    raise ValueError("Source changed during publication")
        payload.seek(0)
        parts, combined = [], hashlib.sha256()
        while block := payload.read(CHUNK):
            name = f"payload.part{len(parts):03d}"
            (out / name).write_bytes(block)
            combined.update(block)
            parts.append(
                {"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()}
            )
    value = {
        "schema": SCHEMA,
        "files": files,
        "parts": parts,
        "payload_sha256": combined.hexdigest(),
        "raw_files": len([n for n in files if n.startswith("contracts/")]),
        "scope": "Single-contract raw CSVs plus freshly locked data/statistics. No old weights or fabricated lineage.",
    }
    (out / "manifest.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value


def verify(bundle, target):
    value, target = manifest(bundle), no_links(target)
    for name, info in value["files"].items():
        path = no_links(target / safe_relative(name))
        if (
            not path.is_file()
            or path.stat().st_size != info["bytes"]
            or digest(path) != info["sha256"]
        ):
            raise ValueError(f"Restored data mismatch: {name}")
    # Reject additional contract CSVs; the loader must not silently see new data.
    observed = {
        "contracts/" + str(p.relative_to(target / "contracts"))
        for p in (target / "contracts").glob("*/*m.csv")
    }
    expected = {n for n in value["files"] if n.startswith("contracts/")}
    if observed != expected:
        raise ValueError("Additional or missing raw contracts")
    return value


def install(bundle, target):
    bundle, target = no_links(bundle), no_links(target)
    value = manifest(bundle)
    target.mkdir(parents=True, exist_ok=True)
    for name, info in value["files"].items():
        path = no_links(target / safe_relative(name))
        if path.exists() and (not path.is_file() or digest(path) != info["sha256"]):
            raise ValueError(f"Refusing to overwrite different existing data: {name}")
    with tempfile.TemporaryFile(dir=target) as payload:
        combined = hashlib.sha256()
        for info in value["parts"]:
            with (bundle / info["name"]).open("rb") as part:
                for block in iter(lambda: part.read(1024 * 1024), b""):
                    payload.write(block)
                    combined.update(block)
        if combined.hexdigest() != value["payload_sha256"]:
            raise ValueError("Combined payload hash differs")
        payload.seek(0)
        seen = set()
        with tarfile.open(fileobj=payload, mode="r|gz") as tar:
            for member in tar:
                name = str(safe_relative(member.name))
                if (
                    name in seen
                    or name not in value["files"]
                    or not member.isfile()
                    or member.size != value["files"][name]["bytes"]
                ):
                    raise ValueError("Unexpected, duplicate, linked or oversized archive member")
                seen.add(name)
                path = no_links(target / name)
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                    temporary = Path(f.name)
                    try:
                        shutil.copyfileobj(stream, f)
                        f.flush()
                        if digest(temporary) != value["files"][name]["sha256"]:
                            raise ValueError("Extracted member hash differs")
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
        if seen != set(value["files"]):
            raise ValueError("Incomplete archive payload")
    return verify(bundle, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("pack", "install", "verify"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--target", type=Path)
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--dataset", type=Path)
    a = p.parse_args()
    if a.mode == "pack":
        if a.raw_root is None or a.dataset is None:
            p.error("pack needs --raw-root and --dataset")
        result = pack(a.raw_root, a.dataset, a.bundle)
    else:
        if a.target is None:
            p.error("install/verify needs --target")
        result = (install if a.mode == "install" else verify)(a.bundle, a.target)
    print(
        f"Snapshot {a.mode}: {result['raw_files']} raw files, {len(result['files'])} total files verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
