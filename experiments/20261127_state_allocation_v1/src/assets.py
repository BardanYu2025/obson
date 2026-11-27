"""Release, checkpoint and off-machine package identities."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic, digest, json_file, read_json


def release_id(published=True):
    seal = read_json(ROOT / "release.json")
    for name, sha in seal["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Release mismatch: {name}")
    tracked = {
        p.relative_to(ROOT).as_posix()
        for top in ("src", "scripts", "configs")
        for p in (ROOT / top).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if tracked - set(seal["files"]):
        raise ValueError("Unsealed code or configuration in release")
    result = {"release_sha256": digest(ROOT / "release.json")}
    if published:

        def git(*a):
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *a], text=True
            ).strip()

        commit = git("rev-parse", "HEAD")
        rel = Path(git("rev-parse", "--show-prefix")) / "release.json"
        recorded = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:{rel.as_posix()}"]
        )
        if hashlib.sha256(recorded).hexdigest() != result["release_sha256"]:
            raise ValueError("Checked out commit does not contain this release")
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                commit,
                "refs/remotes/origin/codex/babel-next",
            ],
            check=True,
        )
        result["commit"] = commit
    return result


def environment():
    import pyarrow

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "device_capability": list(torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else [],
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
    }


def rng_state():
    kind, keys, pos, gauss, value = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (kind, torch.from_numpy(keys.astype("int64")), pos, gauss, value),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda"]])
    random.setstate(state["python"])
    kind, keys, pos, gauss, value = state["numpy"]
    np.random.set_state((kind, keys.cpu().numpy().astype("uint32"), pos, gauss, value))


def save_checkpoint(folder, model, opt, step, trace, run_id, is_best=False):
    folder = Path(folder)
    path = folder / "weights" / (f"{step:07d}-" + uuid.uuid4().hex + ".pt")
    index_path = folder / "checkpoints.json"
    old = read_json(index_path) if index_path.exists() else {}
    with atomic(path) as tmp:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "trace": trace,
                "rng": rng_state(),
                "run": run_id,
            },
            tmp,
        )
    new = {
        **old,
        "last": {
            "path": path.relative_to(folder).as_posix(),
            "sha256": digest(path),
            "step": step,
        },
    }
    if is_best:
        new["best"] = new["last"]
    json_file(index_path, new)
    kept = {v["path"] for v in new.values()}
    for v in old.values():
        if v["path"] not in kept:
            (folder / v["path"]).unlink(missing_ok=True)


def load_checkpoint(folder, tag, model, opt, run_id):
    entry = read_json(Path(folder) / "checkpoints.json")[tag]
    path = Path(folder) / entry["path"]
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint SHA256 mismatch: {tag}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["run"] != run_id or state["step"] != entry["step"]:
        raise ValueError("Checkpoint identity/step mismatch")
    model.load_state_dict(state["model"], strict=True)
    if opt is not None:
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    return state["step"], state["trace"]


def export_run(folder, dest):
    folder, dest = Path(folder), Path(dest)
    if dest.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Download directory must be outside the run")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / (folder.name + "-" + uuid.uuid4().hex[:12] + ".tar.gz")
    archive = {
        f"experiment/{k}": ROOT / k
        for k in [*read_json(ROOT / "release.json")["files"], "release.json"]
    }
    weights = set()
    for index in folder.rglob("checkpoints.json"):
        for row in read_json(index).values():
            target = index.parent / row["path"]
            if digest(target) != row["sha256"]:
                raise ValueError("Damaged checkpoint at export")
            weights.add(target.relative_to(folder).as_posix())
    for file in sorted(folder.rglob("*")):
        if file.is_file() and not file.is_symlink():
            rel = file.relative_to(folder).as_posix()
            if rel.split("/")[0].startswith(".") or (
                "weights/" in rel and rel not in weights
            ):
                continue
            archive["run/" + rel] = file
    manifest = {
        k: {"bytes": p.stat().st_size, "sha256": digest(p)} for k, p in archive.items()
    }
    with atomic(out) as tmp:
        with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
            for k, p in archive.items():
                tar.add(p, k, recursive=False)
            data = json.dumps(manifest, sort_keys=True).encode()
            entry = tarfile.TarInfo("manifest.json")
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    receipt = {"name": out.name, "bytes": out.stat().st_size, "sha256": digest(out)}
    json_file(Path(str(out) + ".json"), receipt)
    print(json.dumps(receipt), flush=True)
    return out


def verify_package(path):
    path = Path(path)
    outer = read_json(Path(str(path) + ".json"))
    if digest(path) != outer["sha256"] or path.stat().st_size != outer["bytes"]:
        raise ValueError("Outer package digest/size mismatch")
    observed = {}
    with tarfile.open(path, "r|gz") as tar:
        manifest = None
        for member in tar:
            name = member.name
            if (
                not member.isfile()
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or name in observed
            ):
                raise ValueError(f"Unsafe/duplicate archive member: {name}")
            reader = tar.extractfile(member)
            if name == "manifest.json":
                if manifest is not None:
                    raise ValueError("Duplicate package inventory")
                manifest = json.load(reader)
                continue
            h = hashlib.sha256()
            for block in iter(lambda: reader.read(1 << 20), b""):
                h.update(block)
            observed[name] = {"bytes": member.size, "sha256": h.hexdigest()}
    if observed != manifest:
        raise ValueError("Inner package inventory or digest mismatch")
    return {"verified_sha256": outer["sha256"], "files": len(observed)}


def snapshot_context(folder, working, release, stats, meta):
    folder.mkdir(parents=True, exist_ok=True)
    json_file(
        folder / "run.json", {"release": release, "environment": environment(), **meta}
    )
    json_file(folder / "scales.json", {k: v.tolist() for k, v in stats.items()})
    shutil.copyfile(working / "catalogue.json", folder / "catalogue.json")


def restore_package(path, to):
    """Restore into a new sibling staging directory, after checking every member."""
    import tempfile

    verify_package(path)
    to = Path(to).resolve()
    if to.exists():
        raise FileExistsError(to)
    to.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".babel-restore-", dir=to.parent))
    try:
        with tarfile.open(path, "r|gz") as tar:
            for m in tar:
                if m.name == "manifest.json":
                    continue
                dest = (tmp / m.name).resolve()
                if not m.isfile() or not dest.is_relative_to(tmp):
                    raise ValueError("Unsafe restore member")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as out:
                    shutil.copyfileobj(tar.extractfile(m), out, 1 << 20)
        # Verify after extraction as well, before publishing the restored directory.
        with tarfile.open(path, "r:gz") as tar:
            manifest = json.load(tar.extractfile("manifest.json"))
        for name, item in manifest.items():
            if digest(tmp / name) != item["sha256"]:
                raise ValueError("Restore mismatch: " + name)
        tmp.rename(to)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return to
