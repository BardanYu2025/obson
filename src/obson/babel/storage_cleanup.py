"""Conservative AutoDL cleanup with protected lineage and downloaded-report hashes."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic_json(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def run_root(path, checkpoints):
    """Only direct checkpoint children are experimental roots; reject symlink escapes."""
    try:
        relative = Path(path).resolve().relative_to(checkpoints.resolve())
    except (ValueError, TypeError):
        return None
    return checkpoints.resolve() / relative.parts[0] if relative.parts else None


def referenced_roots(identity, checkpoints, repo):
    roots, pending = set(), [identity]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and (value.startswith("/") or value.startswith("checkpoints/")):
            p = Path(value) if value.startswith("/") else repo / value
            root = run_root(p, checkpoints)
            if root:
                roots.add(root)
    return roots


def verified_lineage(source, checkpoints, repo):
    # Imported lazily: dry fixture tests require no model loading or Torch runtime.
    from .history_query_run import source_identity

    opened = set()
    enabled = True

    def trace(event, args):
        if enabled and event == "open" and isinstance(args[0], (str, bytes)):
            root = run_root(os.fsdecode(args[0]), checkpoints)
            if root:
                opened.add(root)

    # Hooks cannot be removed. Disable immediately after this bounded read-only check.
    sys.addaudithook(trace)
    try:
        identity = source_identity(source)
    finally:
        enabled = False
    roots = opened | referenced_roots(identity, checkpoints, repo) | {source.resolve()}
    return identity, roots


def regular(path, root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        return False
    return all(not p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))


def file_record(path, kind):
    st = path.stat()
    return {
        "path": str(path),
        "kind": kind,
        "size": st.st_size,
        "allocated": st.st_blocks * 512,
        "links": st.st_nlink,
        "device": st.st_dev,
        "inode": st.st_ino,
        "mtime_ns": st.st_mtime_ns,
        "sha256": digest(path),
    }


def plan(checkpoints, download, protected, catalog):
    candidates, skipped = [], []
    allowed_reports = {entry["sha256"] for entry in catalog["archives"]}
    # Closed leaves only. Preserve best, all caches, readers, JSON and metadata.
    for root in sorted(checkpoints.glob("babel_*")):
        if root.is_symlink() or not root.is_dir():
            continue
        if root.resolve() in protected:
            continue
        completion, manifest = root / "completion.json", root / "manifest.json"
        if not regular(completion, checkpoints) or not regular(manifest, checkpoints):
            continue
        done = read(completion)
        if done.get("status") != "complete" or not isinstance(done.get("files"), dict):
            continue
        files = done["files"]
        for path in sorted(root.rglob("last.pt")):
            best = path.with_name("best.pt")
            if not regular(path, checkpoints) or not regular(best, checkpoints):
                continue
            rel, best_rel = str(path.relative_to(root)), str(best.relative_to(root))
            if rel not in files or best_rel not in files:
                skipped.append({"path": str(path), "reason": "best/last not bound in completion"})
                continue
            record = file_record(path, "closed_leaf_resume_checkpoint")
            if record["sha256"] != files[rel] or digest(best) != files[best_rel]:
                raise ValueError(f"Closed experiment checkpoint changed: {path}")
            record["retained_best"] = {"path": str(best), "sha256": files[best_rel]}
            record["completion_sha256"] = digest(completion)
            record["completion_path"] = str(completion)
            candidates.append(record)
    if download.exists():
        for path in sorted(download.glob("babel*_reports.tar.gz")):
            if not regular(path, download):
                continue
            record = file_record(path, "report_archive_verified_in_local_downloads")
            if record["sha256"] in allowed_reports:
                candidates.append(record)
            else:
                skipped.append({"path": str(path), "reason": "no matching downloaded archive hash"})
    return {
        "schema": "babel-storage-cleanup-v1",
        "protected_roots": sorted(map(str, protected)),
        "files": candidates,
        "skipped": skipped,
        "logical_bytes": sum(r["size"] for r in candidates),
        "estimated_reclaim_bytes": sum(r["allocated"] for r in candidates if r["links"] == 1),
        "scope": "Delete only bound last.pt of completed non-ancestor experiments while retaining best.pt; these leaves cannot exactly resume or pass their old full-file completion check. Delete only report archives with verified local copies. Preserve raw data, caches, metadata, best weights and the full current source lineage.",
    }


def assert_no_active_jobs(selected, proc=Path("/proc")):
    if not proc.is_dir():
        raise ValueError("Cleanup execution requires Linux /proc; this is an AutoDL command")
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int((proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    selected_paths = {r["path"] for r in selected}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            args = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except FileNotFoundError:
            continue
        if "obson.babel" in args or "/obson/babel/" in args:
            raise ValueError(f"Babel process still running, pid={entry.name}: {args[:220]}")
        for fd in (entry / "fd").glob("*"):
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            if target in selected_paths:
                raise ValueError(f"Planned file is open in pid={entry.name}: {target}")


def verify_record(record):
    path = Path(record["path"])
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError(f"Cleanup target path changed: {path}")
    current = file_record(path, record["kind"])
    if any(current[k] != record[k] for k in current):
        raise ValueError(f"Cleanup target changed: {path}")
    if "retained_best" in record:
        best = record["retained_best"]
        if digest(Path(best["path"])) != best["sha256"]:
            raise ValueError(f"Retained best changed: {best['path']}")
        if digest(Path(record["completion_path"])) != record["completion_sha256"]:
            raise ValueError("Completion changed during cleanup")


def apply_plan(report, receipt, check_processes, verify_source):
    check_processes()
    for record in report["files"]:
        verify_record(record)
    verify_source()
    journal = dict(
        report,
        status="deleting",
        deleted=[],
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    atomic_json(journal, receipt)  # If even this write cannot finish, delete nothing.
    try:
        check_processes()
        for record in report["files"]:
            verify_record(record)
            Path(record["path"]).unlink()
            journal["deleted"].append(record["path"])
            atomic_json(journal, receipt)
        verify_source()
        journal["status"] = "complete"
        atomic_json(journal, receipt)
    except Exception as exc:
        journal.update(status="failed", error=str(exc))
        atomic_json(journal, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Actually delete the listed verified files"
    )
    parser.add_argument("--download", default="/root/autodl-tmp/download")
    parser.add_argument("--source", default="checkpoints/babel_control600_v2")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    checkpoints, download = repo / "checkpoints", Path(args.download).resolve()
    source = Path(args.source)
    source = (repo / source).resolve() if not source.is_absolute() else source.resolve()
    if source.parent != checkpoints or not re.fullmatch("babel_[A-Za-z0-9_-]+", source.name):
        raise ValueError("Source must be a Babel checkpoint in this repository")
    catalog = read(repo / "docs/BABEL_DOWNLOADED_REPORTS.json")
    with (repo / ".babel-storage-cleanup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.apply:
            assert_no_active_jobs([])
        print("Verifying current model and all existing source dependencies...", flush=True)
        identity, protected = verified_lineage(source, checkpoints, repo)
        report = plan(checkpoints, download, protected, catalog)
        report["source"] = str(source)
        report["free_bytes_before"] = shutil.disk_usage(checkpoints).free
        report["catalog_sha256"] = digest(repo / "docs/BABEL_DOWNLOADED_REPORTS.json")
        for r in report["files"]:
            print(f"{r['size'] / 1024**3:.3f} GiB  {r['kind']}  {r['path']}", flush=True)
        print(
            f"Protected {len(protected)} experiment roots; {len(report['files'])} deletable files; estimated reclaim {report['estimated_reclaim_bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )
        if not args.apply:
            print("Preview only. Add --apply to delete these verified files.")
            return
        logs = repo / "logs"
        logs.mkdir(exist_ok=True)
        receipt = logs / f"babel_cleanup_{time.time_ns()}.json"

        def same_source():
            current, roots = verified_lineage(source, checkpoints, repo)
            if current != identity or roots != protected:
                raise ValueError("Current model lineage changed")

        apply_plan(
            report,
            receipt,
            lambda: assert_no_active_jobs(report["files"]),
            same_source,
        )
        print(
            f"Filesystem free: {shutil.disk_usage(checkpoints).free / 1024**3:.2f} GiB", flush=True
        )
        print(f"Cleanup complete; current source verified. Receipt: {receipt}", flush=True)


if __name__ == "__main__":
    main()
