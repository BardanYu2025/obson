"""Failure-safe report packaging and shell wrappers on synthetic artifacts only."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest

from obson.babel import multiscale_export as export


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixed_arrays(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=np.zeros((2, 8, 127, 7), np.float32),
        targets=np.zeros((2, 8, 127, 7), np.float64),
        masks=np.ones((2, 8, 127, 7), bool),
        raw_closes=np.full((2, 255), 100.0, np.float64),
        positions=np.array(export.POSITIONS),
    )


def read_archive(path, name):
    with tarfile.open(path) as tar:
        assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in tar)
        return json.load(tar.extractfile(f"{name}/export_manifest.json")), tar.getnames()


def test_include_fixed_evidence_and_list_omitted_weights_raw_full128(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    put_json(out / "status.json", {"status": "complete"})
    (out / "last.pt").write_bytes(b"synthetic weights")
    (out / "raw").mkdir()
    np.savez(out / "raw/train.npz", close=np.ones((2, 255)))
    (out / "predictions/control_s42/test").mkdir(parents=True)
    np.save(
        out / "predictions/control_s42/test/parents_00000_00002.npy",
        np.zeros((2, 128, 127, 7), np.float32),
    )
    fixed_arrays(out / "results/test_control_s42_predictions.npz")
    evidence = out / "evaluation/control_s42/test"
    evidence.mkdir(parents=True)
    np.savez(evidence / "parent_rows.npz", error=np.ones(2))
    np.savez(evidence / "position_rows.npz", error=np.ones((2, 128)))
    (out / "epoch.jsonl").write_text('{"epoch":120}\n')
    result = export.export_reports(out, download)
    manifest, names = read_archive(result["archive"], "run")
    assert "run/results/test_control_s42_predictions.npz" in names
    assert "run/evaluation/control_s42/test/position_rows.npz" in names
    assert not any(
        name.endswith((".pt", ".npy")) or name.endswith("raw/train.npz") for name in names
    )
    omitted = {row["path"]: row for row in manifest["omitted"]}
    assert omitted["last.pt"]["reason"] == "weights_or_optimizer_state"
    shard = "predictions/control_s42/test/parents_00000_00002.npy"
    assert omitted[shard]["reason"] == "full128_prediction_shard"
    assert omitted[shard]["bytes"] == (out / shard).stat().st_size
    assert omitted[shard]["sha256"] == hashlib.sha256((out / shard).read_bytes()).hexdigest()
    with tarfile.open(result["archive"]) as tar:
        for item in manifest["included"]:
            assert (
                hashlib.sha256(tar.extractfile("run/" + item["path"]).read()).hexdigest()
                == item["sha256"]
            )
    assert manifest["snapshot_consistency"] == "producer_locks_held"
    assert not list(download.glob(".multiscale_export_*"))


def test_new_archive_preserves_previous_failure_or_success_bytes(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    put_json(out / "status.json", {"status": "failed"})
    first = export.export_reports(out, download, command_exit_code=1, mode="all")
    old = Path(first["archive"]).read_bytes()
    put_json(out / "status.json", {"status": "complete"})
    second = export.export_reports(out, download, mode="evaluate")
    assert Path(second["previous_archive"]).read_bytes() == old
    manifest, _ = read_archive(second["archive"], "run")
    assert manifest["run_status"] == "complete"


@pytest.mark.parametrize(
    "lockname",
    ("session.lock", "worker.lock", "jobs/control_s42/worker.lock", "evaluation/evaluation.lock"),
)
def test_active_producer_refuses_inconsistent_snapshot(tmp_path, lockname):
    out = tmp_path / "run"
    path = out / lockname
    path.parent.mkdir(parents=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active"):
            export.export_reports(out, tmp_path / "download")
    assert not list((tmp_path / "download").glob("*.tar.gz"))


def test_traversal_symlink_special_files_and_nested_destinations_rejected(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    out.mkdir()
    with pytest.raises(ValueError, match="traversal"):
        export.export_reports(out / ".." / "run", download)
    with pytest.raises(ValueError, match="Separate"):
        export.export_reports(out, out / "download")
    secret = tmp_path / "external.json"
    secret.write_text("sensitive outside artifact")
    (out / "leak.json").symlink_to(secret)
    with pytest.raises(ValueError, match="Symlink"):
        export.export_reports(out, download)
    (out / "leak.json").unlink()
    os.mkfifo(out / "pipe.txt")
    with pytest.raises(ValueError, match="regular"):
        export.export_reports(out, download)
    assert secret.read_text() == "sensitive outside artifact"


def test_invalid_fixed_prediction_is_recorded_not_silently_exported(tmp_path):
    out = tmp_path / "run"
    put_json(out / "status.json", {"status": "failed"})
    path = out / "results/test_control_s42_predictions.npz"
    path.parent.mkdir()
    np.savez(path, positions=np.arange(1, 129), predictions=np.zeros(2))
    result = export.export_reports(out, tmp_path / "download", command_exit_code=1)
    report, names = read_archive(result["archive"], "run")
    assert "run/results/test_control_s42_predictions.npz" not in names
    entry = next(row for row in report["omitted"] if row["path"].startswith("results/"))
    assert entry["reason"].startswith("invalid_fixed_prediction_evidence")


@pytest.fixture
def fake_environment(tmp_path):
    executable = tmp_path / "fake python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, signal, sys, time\n"
        "if sys.argv[2] == 'obson.babel.multiscale_export':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "assert sys.argv[2] == 'obson.babel.multiscale_run'\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'arguments.json').write_text(json.dumps({'argv':sys.argv[3:], 'omp':os.environ['OMP_NUM_THREADS']}))\n"
        "def stopped(*args):\n"
        "    (out/'status.json').write_text(json.dumps({'status':'stopped'})); sys.exit(130)\n"
        "signal.signal(signal.SIGTERM, stopped)\n"
        "if os.environ.get('FAKE_WAIT') == '1':\n"
        "    (out/'ready.txt').write_text('ready')\n"
        "    while True: time.sleep(.05)\n"
        "code = int(os.environ.get('FAKE_EXIT', '0'))\n"
        "(out/'status.json').write_text(json.dumps({'status':'complete' if code == 0 else 'failed'}))\n"
        "sys.exit(code)\n"
    )
    executable.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(executable),
        BABEL_MULTISCALE_RUN=str(tmp_path / "run"),
        BABEL_MULTISCALE_SOURCE=str(tmp_path / "source"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        BABEL_MULTISCALE_LOG=str(tmp_path / "log.txt"),
        OMP_NUM_THREADS="invalid",
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_multiscale768_autodl.sh"
    return script, env


@pytest.mark.parametrize(
    "mode,code,status",
    (
        ("all", 0, "complete"),
        ("all", 1, "failed"),
        ("all", 124, "timeout"),
        ("evaluate", 0, "complete"),
    ),
)
def test_shell_success_failure_timeout_and_evaluate_flags(fake_environment, mode, code, status):
    script, env = fake_environment
    env["FAKE_EXIT"] = str(code)
    result = subprocess.run(
        ["bash", str(script), mode], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == code, result.stderr
    out = Path(env["BABEL_MULTISCALE_RUN"])
    arguments = json.loads((out / "arguments.json").read_text())
    assert arguments["omp"] == "4"
    assert ("--evaluate-only" in arguments["argv"]) == (mode == "evaluate")
    assert "--jobs" not in arguments["argv"] and "--micro" not in arguments["argv"]
    archive = Path(env["BABEL_DOWNLOAD_DIR"]) / "run_reports.tar.gz"
    report, _ = read_archive(archive, "run")
    assert report["command_exit_code"] == code and report["run_status"] == status


def test_shell_signal_waits_for_child_then_exports_stopped(fake_environment):
    script, env = fake_environment
    env["FAKE_WAIT"] = "1"
    process = subprocess.Popen(
        ["bash", str(script), "all"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = Path(env["BABEL_MULTISCALE_RUN"]) / "ready.txt"
    deadline = time.monotonic() + 5
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (stdout, stderr)
        report, _ = read_archive(Path(env["BABEL_DOWNLOAD_DIR"]) / "run_reports.tar.gz", "run")
        assert report["command_exit_code"] == 130 and report["run_status"] == "stopped"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_shell_export_only_never_runs_training(fake_environment):
    script, env = fake_environment
    out = Path(env["BABEL_MULTISCALE_RUN"])
    put_json(out / "status.json", {"status": "stopped"})
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert not (out / "arguments.json").exists()
    report, _ = read_archive(Path(env["BABEL_DOWNLOAD_DIR"]) / "run_reports.tar.gz", "run")
    assert report["run_status"] == "stopped"
