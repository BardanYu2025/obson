import json
from pathlib import Path

import pytest

from obson.babel import storage_cleanup as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def completed(root):
    write(root / "manifest.json", "{}")
    best = write(root / "control_s42/best.pt", "selected model")
    last = write(root / "control_s42/last.pt", "resume optimizer and last model")
    files = {str(p.relative_to(root)): c.digest(p) for p in (best, last)}
    write(root / "completion.json", json.dumps({"status": "complete", "files": files}))
    return best, last


def fixture(tmp):
    checkpoints, download = tmp / "checkpoints", tmp / "download"
    protected = checkpoints / "babel_control600_v2"
    current_best, current_last = completed(protected)
    best, last = completed(checkpoints / "babel_history_query768")
    archive = write(download / "babel_old_reports.tar.gz", "verified downloaded bytes")
    unknown = write(download / "babel_unknown_reports.tar.gz", "not downloaded")
    catalog = {"archives": [{"sha256": c.digest(archive)}]}
    return (
        checkpoints,
        download,
        protected,
        best,
        last,
        archive,
        unknown,
        catalog,
        current_best,
        current_last,
    )


def test_plan_and_apply_preserve_best_lineage_and_unknown_reports(tmp_path):
    cp, down, root, best, last, archive, unknown, cat, current_best, current_last = fixture(
        tmp_path
    )
    report = c.plan(cp, down, {root}, cat)
    assert {r["path"] for r in report["files"]} == {str(last), str(archive)}
    assert last.exists() and archive.exists()  # Planning never deletes.
    events = []
    receipt = tmp_path / "receipt.json"
    c.apply_plan(report, receipt, lambda: events.append("idle"), lambda: events.append("source"))
    assert not last.exists() and not archive.exists()
    assert all(p.exists() for p in (best, unknown, current_best, current_last))
    assert events == ["idle", "source", "idle", "source"]
    assert c.read(receipt)["status"] == "complete"
    assert len(c.read(receipt)["deleted"]) == 2
    assert not c.plan(cp, down, {root}, cat)["files"]  # Idempotent.


@pytest.mark.parametrize("change", ["last", "best", "completion", "parent_symlink"])
def test_changes_after_plan_delete_nothing(tmp_path, change):
    cp, down, root, best, last, archive, _, cat, *_ = fixture(tmp_path)
    report = c.plan(cp, down, {root}, cat)
    if change == "parent_symlink":
        old = last.parent
        target = old.with_name("moved")
        old.rename(target)
        old.symlink_to(target, target_is_directory=True)
    else:
        path = {"last": last, "best": best, "completion": last.parent.parent / "completion.json"}[
            change
        ]
        path.write_text("changed")
    with pytest.raises(ValueError):
        c.apply_plan(report, tmp_path / "receipt.json", lambda: None, lambda: None)
    assert archive.exists()
    assert not (tmp_path / "receipt.json").exists()


def test_incomplete_unbound_symlinks_and_nonbabel_models_untouched(tmp_path):
    cp, down, root, best, last, _, _, cat, *_ = fixture(tmp_path)
    unfinished = cp / "babel_running"
    completed(unfinished)
    (unfinished / "completion.json").unlink()
    completed(cp / "other_model")
    alias = cp / "babel_alias"
    alias.symlink_to(unfinished, target_is_directory=True)
    outside = tmp_path / "outside"
    outside_best, outside_last = completed(outside)
    (last.parent / "linked").symlink_to(outside / "control_s42", target_is_directory=True)
    (down / "babel_link_reports.tar.gz").symlink_to(outside_last)
    # An unbound last file is not deleted merely because of its filename.
    write(best.parent / "extra/last.pt", "other")
    write(best.parent / "extra/best.pt", "other")
    report = c.plan(cp, down, {root}, cat)
    assert len(report["files"]) == 2
    c.apply_plan(report, tmp_path / "receipt.json", lambda: None, lambda: None)
    assert (unfinished / "control_s42/last.pt").exists()
    assert (cp / "other_model/control_s42/last.pt").exists()
    assert outside_best.exists() and outside_last.exists()
    assert (best.parent / "extra/last.pt").exists()


def test_failed_source_or_active_process_deletes_nothing(tmp_path):
    cp, down, root, _, last, archive, _, cat, *_ = fixture(tmp_path)
    report = c.plan(cp, down, {root}, cat)

    def fail():
        raise ValueError("blocked")

    for check, verify in ((fail, lambda: None), (lambda: None, fail)):
        with pytest.raises(ValueError, match="blocked"):
            c.apply_plan(report, tmp_path / "receipt.json", check, verify)
        assert last.exists() and archive.exists()


def test_process_guard_and_open_file_guard(tmp_path):
    proc = tmp_path / "proc"
    entry = proc / "99999999"
    write(entry / "cmdline", "python\0-m\0obson.babel.history_query_run\0")
    with pytest.raises(ValueError, match="Babel process"):
        c.assert_no_active_jobs([], proc)
    write(entry / "cmdline", "unrelated")
    target = write(tmp_path / "last.pt", "data")
    (entry / "fd").mkdir()
    (entry / "fd/3").symlink_to(target)
    with pytest.raises(ValueError, match="file is open"):
        c.assert_no_active_jobs([{"path": str(target)}], proc)


def test_initial_process_check_does_not_inspect_descriptors(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    write(proc / "99999999/cmdline", "unrelated")

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight has no files to check")

    monkeypatch.setattr(c.os, "readlink", forbidden)
    result = c.assert_no_active_jobs([], proc)
    assert result["command_lines_checked"] == 1 and not result["fd_scan_requested"]


@pytest.mark.parametrize("denied", ["descriptor", "directory"])
def test_container_fd_permissions_are_recorded_without_deleting_unknown_files(
    tmp_path, monkeypatch, denied
):
    proc = tmp_path / "proc"
    entry = proc / "99999999"
    write(entry / "cmdline", "unrelated")
    (entry / "fd").mkdir()
    restricted = entry / "fd/0"
    restricted.symlink_to("/dev/null")
    original_readlink, original_iterdir = c.os.readlink, Path.iterdir

    def readlink(path, *args, **kwargs):
        if Path(path) == restricted:
            raise PermissionError("container denied descriptor")
        return original_readlink(path, *args, **kwargs)

    def iterdir(path):
        if path == entry / "fd":
            raise PermissionError("container denied fd directory")
        return original_iterdir(path)

    if denied == "descriptor":
        monkeypatch.setattr(c.os, "readlink", readlink)
    else:
        monkeypatch.setattr(Path, "iterdir", iterdir)
    selected = [{"path": str(tmp_path / "last.pt")}]
    result = c.assert_no_active_jobs(selected, proc)
    assert result["fd_visibility"] == "partial"
    assert result["fd_permission_denied_pids"] == [entry.name]
    if denied == "descriptor":
        (entry / "fd/3").symlink_to(selected[0]["path"])
        with pytest.raises(ValueError, match="file is open"):
            c.assert_no_active_jobs(selected, proc)
    write(entry / "cmdline", "python -m obson.babel.history_query_run")
    with pytest.raises(ValueError, match="Babel process"):
        c.assert_no_active_jobs(selected, proc)


def test_unreadable_command_line_still_blocks_cleanup(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    entry = proc / "99999999"
    command = write(entry / "cmdline", "unreadable")
    original = Path.read_bytes

    def denied(path):
        if path == command:
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(ValueError, match="Cannot inspect process command line"):
        c.assert_no_active_jobs([], proc)


def test_process_visibility_is_saved_in_cleanup_receipt(tmp_path):
    cp, down, root, _, _, _, _, cat, *_ = fixture(tmp_path)
    report = c.plan(cp, down, {root}, cat)
    visibility = {"fd_visibility": "partial", "fd_permission_denied_pids": ["99999999"]}
    receipt = tmp_path / "receipt.json"
    c.apply_plan(report, receipt, lambda: visibility, lambda: None)
    assert c.read(receipt)["process_checks"] == [visibility, visibility]


def test_dependency_paths_protect_whole_root_without_escape(tmp_path):
    cp = tmp_path / "checkpoints"
    identity = {
        "source": str(cp / "babel_parent"),
        "nested": [{"cache": "checkpoints/babel_ancestor/cache/x.npy"}, "/outside/data"],
    }
    assert c.referenced_roots(identity, cp, tmp_path) == {
        cp / "babel_parent",
        cp / "babel_ancestor",
    }
    assert c.run_root(tmp_path / "other", cp) is None


def test_source_open_tracking_protects_dependencies_absent_from_identity(tmp_path, monkeypatch):
    from obson.babel import history_query_run

    cp = tmp_path / "checkpoints"
    source = cp / "babel_control600_v2"
    hidden = write(cp / "babel_hidden/cache.npy", "data")

    def source_identity(path):
        assert path == source
        hidden.read_bytes()
        return {"source": str(source)}

    monkeypatch.setattr(history_query_run, "source_identity", source_identity)
    _, protected = c.verified_lineage(source, cp, tmp_path)
    assert hidden.parent in protected and source in protected


def test_interrupted_delete_journal_records_progress(tmp_path, monkeypatch):
    cp, down, root, _, last, archive, _, cat, *_ = fixture(tmp_path)
    report = c.plan(cp, down, {root}, cat)
    original = Path.unlink

    def interrupted(path, *args, **kwargs):
        if path == archive:
            raise OSError("synthetic interruption")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(OSError):
        c.apply_plan(report, receipt, lambda: None, lambda: None)
    journal = c.read(receipt)
    assert journal["status"] == "failed" and journal["deleted"] == [str(last)]
    assert not last.exists() and archive.exists()
