#!/usr/bin/env python3
"""Check experiment isolation and immutable releases. Never executes experiment code.

Static checks cover common accidental imports, not arbitrary dynamic Python or
shell behavior. A clean isolated restore remains part of every release review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

IGNORED = {"runs", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN_IMPORTS = {"obson", "experiments", "tools"}
ID = re.compile(r"\d{8}_[a-z][a-z0-9_]*_v[1-9]\d*\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_relative(name):
    require(isinstance(name, str) and bool(name), f"Unsafe path: {name}")
    p = Path(name)
    require(not p.is_absolute()
            and ".." not in p.parts and p.as_posix() == name, f"Unsafe path: {name}")
    return p


def files(folder):
    result = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if any(part in IGNORED for part in rel.parts) or p.name == ".DS_Store":
            continue
        require(not p.is_symlink(), f"Symlink not allowed: {p}")
        if p.is_file():
            result[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def contents(folder):
    result = files(folder)
    source = {k: v for k, v in result.items()
              if k != "release.json" and not k.startswith("reports/")}
    reports = {k: v for k, v in result.items() if k.startswith("reports/")}
    return source, reports


def check_source(folder):
    for name in files(folder):
        if name.startswith("reports/") or not name.endswith(".py"):
            continue
        file = folder / name
        tree = ast.parse(file.read_text(), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                require(not any(m.split('.')[0] in FORBIDDEN_IMPORTS for m in modules),
                        f"Cross-experiment/legacy import: {file}:{node.lineno}")
                if isinstance(node, ast.ImportFrom) and node.level:
                    require(node.level <= len(Path(name).parent.parts),
                            f"Relative import escapes experiment: {file}:{node.lineno}")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                require(not (node.value.id == "sys" and node.attr == "path"),
                        f"sys.path modification/access requires removal: {file}:{node.lineno}")


def check_experiment(folder, allow_unsealed=False):
    require(not folder.is_symlink(), f"Symlink experiment: {folder}")
    meta_path = folder / "experiment.json"
    require(meta_path.is_file(), f"Missing experiment.json: {folder}")
    meta = json.loads(meta_path.read_text())
    require(meta.get("schema") == "babel.experiment.v1", f"Bad schema: {folder}")
    require(meta.get("id") == folder.name, f"ID/directory mismatch: {folder}")
    require(folder.name == "_template" or ID.fullmatch(folder.name), f"Bad experiment ID: {folder.name}")
    require(meta.get("kind") in {"neural", "data_audit"}, f"Bad kind: {folder}")
    require(meta.get("status") in {"template", "draft", "released"}, f"Bad status: {folder}")
    require((folder.name == "_template") == (meta["status"] == "template"), f"Template status misuse: {folder}")
    for required in ["README.md", "requirements.lock"]:
        require((folder / required).is_file(), f"Missing {required}: {folder}")
    for name in [*meta.get("entrypoints", {}).values(), *meta.get("source_files", []), *meta.get("data_manifests", [])]:
        require((folder / safe_relative(name)).is_file(), f"Missing declared file {name}: {folder}")
    check_source(folder)
    if meta["status"] == "released":
        require(meta.get("ready_for_execution") is True, f"Released but not ready: {folder}")
        require(bool(meta.get("entrypoints")) and bool(meta.get("source_files")), f"Release has no implementation: {folder}")
        if meta["kind"] == "neural":
            require({"train", "evaluate", "export"} <= set(meta["entrypoints"]), f"Missing neural entrypoints: {folder}")
        release = folder / "release.json"
        if allow_unsealed and not release.exists():
            return meta
        require(release.is_file(), f"Missing release seal: {folder}")
        seal = json.loads(release.read_text())
        require(seal.get("schema") == "babel.experiment.release.v1" and seal.get("id") == folder.name,
                f"Bad release identity: {folder}")
        source, reports = contents(folder)
        require(seal.get("files") == source, f"Frozen source/config/docs changed: {folder}")
        for name, digest in seal.get("reports_at_release", {}).items():
            require(reports.get(name) == digest, f"Original report changed: {folder / name}")
    else:
        require(not (folder / "release.json").exists(), f"Sealed experiment cannot become draft: {folder}")
        require(meta.get("ready_for_execution") is False, f"Draft/template cannot be ready: {folder}")
    return meta


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def check_base(root, base):
    require(re.fullmatch(r"[0-9A-Za-z_./~^+-]+", base) and not base.startswith('-'), "Unsafe base ref")
    names = git(root, "ls-tree", "-r", "--name-only", base, "--", "experiments").decode().splitlines()
    released = {name.rsplit('/', 1)[0] for name in names if name.endswith('/release.json')}
    for folder in released:
        for name in names:
            if name.startswith(folder + '/'):
                p = root / name
                require(p.is_file() and not p.is_symlink(), f"Frozen file deleted/moved: {name}")
                require(p.read_bytes() == git(root, "show", f"{base}:{name}"), f"Published file overwritten: {name}")


def check(root, base=None):
    experiments = root / "experiments"
    require(experiments.is_dir(), "Missing experiments directory")
    count = 0
    for folder in sorted(experiments.iterdir()):
        if folder.is_dir() or folder.is_symlink():
            check_experiment(folder)
            count += folder.name != "_template"
    if base:
        check_base(root, base)
    return count


def seal(root, name):
    require(ID.fullmatch(name), "Invalid experiment ID")
    folder = root / "experiments" / name
    require(not (folder / 'release.json').exists(), "Already sealed; create a new experiment version")
    # Do not permit deleting a tracked seal and re-sealing the same ID.
    old_seal = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"HEAD:experiments/{name}/release.json"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(old_seal.returncode != 0, "Previously published; create a new experiment version")
    meta = check_experiment(folder, allow_unsealed=True)
    require(meta['status'] == 'released', "Set reviewed experiment status to released before sealing")
    source, reports = contents(folder)
    value = {"schema": "babel.experiment.release.v1", "id": name,
             "files": source, "reports_at_release": reports}
    (folder / 'release.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    check_experiment(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--base', help='Git baseline used to reject edits/deletion of previously released files')
    parser.add_argument('--seal', help='Seal one reviewed experiment once; never reseals an existing release')
    args = parser.parse_args()
    try:
        if args.seal:
            seal(args.root, args.seal)
        n = check(args.root, args.base)
    except (ValueError, OSError, KeyError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Experiment guard failed: {error}\n')
    print(f'Experiment guard passed: {n} experiment(s); no experiment code executed.')


if __name__ == '__main__':
    main()
