"""Startup repair tests never import/load a real model or invoke CUDA."""

import fcntl
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from obson.babel import recovery_provenance_startup as start


def failed(out, source, bundle):
    out.mkdir()
    start.write(
        out / "status.json", {"task_ids": ["V14"], "status": "failed", "error": "Missing source"}
    )
    start.write(
        out / "request.json",
        {
            "schema": "babel-recovery-provenance-v1",
            "source": str(source),
            "bundle_source": str(bundle),
            "implementation": {
                "modules": {
                    name: start.digest(Path(start.__file__).with_name(name))
                    for name in ("recovery_provenance_run.py", "recovery_provenance.py")
                }
            },
        },
    )
    (out / "session.lock").touch()


def test_empty_directory_preserved_and_original_entrypoint_launched(tmp_path):
    out, source, bundle = [tmp_path / k for k in ("out", "source", "bundle")]
    out.mkdir()
    calls = []

    def runner(cmd):
        calls.append(cmd)
        assert not out.exists()
        assert out.with_name("out.startup_failure_v1").is_dir()
        out.mkdir()
        return SimpleNamespace(returncode=0)

    assert start.launch(source, bundle, out, runner) == 0
    assert len(calls) == 1 and calls[0][1:3] == ["-m", "obson.babel.recovery_provenance_run"]
    receipt = start.read(out.with_name("out.startup_receipt.json"))
    assert receipt["previous_startup"]["reason"] == "empty_output_no_worker_evidence"


def test_failed_preflight_backed_up_byte_exact_once(tmp_path):
    out, source, bundle = [tmp_path / k for k in ("out", "source", "bundle")]
    failed(out, source, bundle)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def runner(cmd):
        failed(out, source, bundle)
        return SimpleNamespace(returncode=1)

    assert start.launch(source, bundle, out, runner) == 1
    saved = out.with_name("out.startup_failure_v1")
    assert {p.name: p.read_bytes() for p in saved.iterdir()} == before
    with pytest.raises(ValueError, match="already used"):
        start.launch(source, bundle, out, lambda cmd: pytest.fail("extra retry"))
    assert (
        start.read(out.with_name("out.startup_receipt.json"))["previous_receipt"][
            "command_exit_code"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "name", ["local_refit.json", "last.pt", "continuous_s42", "completion.json", "unknown.txt"]
)
def test_numerical_or_worker_or_unknown_evidence_never_restarted(tmp_path, name):
    out, source, bundle = [tmp_path / k for k in ("out", "source", "bundle")]
    failed(out, source, bundle)
    if name == "continuous_s42":
        (out / name).mkdir()
    else:
        (out / name).write_text("{}")
    with pytest.raises(ValueError, match="progressed"):
        start.launch(source, bundle, out, lambda cmd: pytest.fail("unexpected training"))
    assert (out / name).exists() and not out.with_name("out.startup_failure_v1").exists()


def test_live_old_session_lock_prevents_rename(tmp_path):
    out, source, bundle = [tmp_path / k for k in ("out", "source", "bundle")]
    failed(out, source, bundle)
    with (out / "session.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            start.launch(source, bundle, out, lambda cmd: pytest.fail("unexpected launch"))
    assert out.is_dir() and not out.with_name("out.startup_failure_v1").exists()


@pytest.mark.parametrize("condition", ["running", "wrong_source", "missing_status", "symlink"])
def test_unproven_zero_update_or_source_change_blocked(tmp_path, condition):
    out, source, bundle = [tmp_path / k for k in ("out", "source", "bundle")]
    failed(out, source, bundle)
    if condition == "running":
        start.write(out / "status.json", {"task_ids": ["V14"], "status": "running"})
    elif condition == "wrong_source":
        source = tmp_path / "other"
    elif condition == "missing_status":
        (out / "status.json").unlink()
    else:
        (out / "elsewhere").symlink_to(source)
    with pytest.raises(ValueError):
        start.launch(source, bundle, out, lambda cmd: pytest.fail("unexpected launch"))
    assert out.is_dir()


def test_invalid_thread_environment_overridden_before_python_and_archives_preserved(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    env_path = tmp_path / "environment.json"
    fake_python = fakebin / "capture-python"
    fake_python.write_text(
        '#!/bin/sh\nprintf "%s %s %s" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" > "$V14_CAPTURE"\nexit 1\n'
    )
    fake_python.chmod(0o755)
    timer = fakebin / "timeout"
    timer.write_text('#!/bin/sh\nshift 3\nexec "$@"\n')
    timer.chmod(0o755)
    out = tmp_path / "run"
    download = tmp_path / "download"
    download.mkdir()
    original_archive = download / "run_reports.tar.gz"
    original_archive.write_bytes(b"prior failed run archive")
    env = dict(
        os.environ,
        PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
        OMP_NUM_THREADS="invalid",
        OPENBLAS_NUM_THREADS="-7",
        MKL_NUM_THREADS="0",
        BABEL_V14_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(download),
        PYTHON_BIN=str(fake_python),
        V14_CAPTURE=str(env_path),
    )
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_provenance_autodl.sh", "all"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert env_path.read_text() == "4 4 4"
    backups = list(download.glob("run_before_startup_fix.*.tar.gz"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"prior failed run archive"
    with tarfile.open(original_archive) as t:
        assert b"run_status=failed" in t.extractfile("run/export_status.txt").read()


def test_export_includes_startup_diagnostics_and_preserved_failure(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    previous = tmp_path / "run.startup_failure_v1"
    previous.mkdir()
    start.write(previous / "status.json", {"status": "failed", "error": "original"})
    start.write(
        tmp_path / "run.startup_receipt.json", {"status": "audit_exited", "command_exit_code": 1}
    )
    env = dict(os.environ, BABEL_V14_RUN=str(out), BABEL_DOWNLOAD_DIR=str(tmp_path / "download"))
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_provenance_autodl.sh", "export"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as t:
        assert json.load(t.extractfile("run/previous_startup/status.json"))["error"] == "original"
        assert "run/startup_receipt.json" in t.getnames()


@pytest.mark.parametrize("reason", ["reported_updates", "unknown_code"])
def test_claimed_updates_or_unknown_control_flow_never_restarted(tmp_path, reason):
    out, source, bundle = [tmp_path / k for k in ("out", "source", "bundle")]
    failed(out, source, bundle)
    if reason == "reported_updates":
        path = out / "status.json"
        value = start.read(path)
        value["actual_optimizer_updates"] = 1
    else:
        path = out / "request.json"
        value = start.read(path)
        value["implementation"]["modules"]["recovery_provenance_run.py"] = "unknown"
    start.write(path, value)
    with pytest.raises(ValueError):
        start.launch(source, bundle, out, lambda cmd: pytest.fail("unexpected retry"))
    assert not out.with_name("out.startup_failure_v1").exists()
