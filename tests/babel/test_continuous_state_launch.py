"""CPU/mocked process tests for bounded execution and unconditional exports."""

import fcntl
import json
import os
import subprocess
import sys
import tarfile

import pytest

from obson.babel import continuous_state_launch as launch


@pytest.fixture
def paths(tmp_path, monkeypatch):
    source, raw, out = (tmp_path / name for name in ("source", "raw", "out"))
    source.mkdir()
    raw.mkdir()
    monkeypatch.setattr(launch, "implementation", lambda: {"fixture": "v1"})
    return source, raw, out


class Worker:
    pid = 999999

    def __init__(self, out, code=0, failure=None):
        self.out, self.code, self.failure = out, code, failure
        self.returncode = None
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        if self.failure is not None and self.waits == 1:
            raise self.failure
        self.returncode = self.code
        if self.code == 0:
            launch.atomic_json({"status": "complete_requires_review"}, self.out / "completion.json")
            launch.status(self.out, "complete_requires_review")
        else:
            launch.status(self.out, "failed", numeric_diagnostic="retained")
        return self.code

    def poll(self):
        return self.returncode


def check(out):
    return launch.read_json(out / "completion.json")


def fake_worker(monkeypatch, paths, code=0, failure=None):
    worker = Worker(paths[2], code, failure)
    calls = []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        # Reservation must be persisted before actual process creation.
        assert launch.read_json(paths[2] / "budget.json")["used_seconds"] == launch.SESSION_CAP
        return worker

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    return worker, calls


def test_success_completed_recheck_does_not_consume_budget_or_spawn(paths, monkeypatch):
    _, calls = fake_worker(monkeypatch, paths)
    assert launch.supervise(*paths, check) == 0
    first_budget = launch.read_json(paths[2] / "budget.json")
    assert len(calls) == 1
    assert first_budget["attempts"][0]["exit_code"] == 0
    # Completion is checked before budget even when a previous crash spent it.
    first_budget["used_seconds"] = launch.SESSION_CAP
    launch.atomic_json(first_budget, paths[2] / "budget.json")
    assert launch.supervise(*paths, check) == 0
    assert len(calls) == 1
    assert launch.read_json(paths[2] / "budget.json") == first_budget


def test_worker_thread_env_corrected_and_deadline_passed(paths, monkeypatch):
    for key in launch.THREAD_ENV:
        monkeypatch.setenv(key, "bad")
    _, calls = fake_worker(monkeypatch, paths)
    assert launch.supervise(*paths, check) == 0
    args, kwargs = calls[0]
    assert args[1:4] == ["-m", "obson.babel.continuous_state_run", "--worker"]
    assert args[-2:] == ["--seconds", str(launch.SESSION_CAP)]
    assert kwargs["start_new_session"] is True
    assert all(kwargs["env"][key] == "4" for key in launch.THREAD_ENV)


def test_failure_keeps_status_snapshot_and_allows_manual_remaining_time(paths, monkeypatch):
    _, calls = fake_worker(monkeypatch, paths, code=7)
    assert launch.supervise(*paths, check) == 7
    budget = launch.read_json(paths[2] / "budget.json")
    assert 0 <= budget["used_seconds"] < launch.SESSION_CAP
    state = launch.read_json(paths[2] / "attempt_001.json")["status"]
    assert state["status"] == "failed"
    assert state["worker_status"]["numeric_diagnostic"] == "retained"
    first = (paths[2] / "attempt_001.json").read_bytes()
    assert launch.supervise(*paths, check) == 7
    assert len(calls) == 2
    assert (paths[2] / "attempt_001.json").read_bytes() == first
    assert (paths[2] / "attempt_002.json").is_file()


def test_request_identity_cannot_change_on_retry(paths, monkeypatch):
    fake_worker(monkeypatch, paths, code=7)
    launch.supervise(*paths, check)
    before = (paths[2] / "request.json").read_bytes()
    monkeypatch.setattr(launch, "implementation", lambda: {"fixture": "v2"})
    with pytest.raises(ValueError, match="differs"):
        launch.supervise(*paths, check)
    assert (paths[2] / "request.json").read_bytes() == before


def test_request_source_cannot_change(paths, monkeypatch, tmp_path):
    fake_worker(monkeypatch, paths, code=7)
    launch.supervise(*paths, check)
    changed = tmp_path / "other_source"
    changed.mkdir()
    with pytest.raises(ValueError, match="differs"):
        launch.supervise(changed, paths[1], paths[2], check)


def test_unbound_existing_output_is_not_overwritten(paths):
    paths[2].mkdir()
    (paths[2] / "keep.txt").write_text("original")
    with pytest.raises(ValueError, match="no matching request"):
        launch.supervise(*paths, check)
    assert not (paths[2] / "request.json").exists()


@pytest.mark.parametrize(
    "unsafe", ["source", "inside_source", "raw", "inside_raw", "ancestor", "code"]
)
def test_unsafe_output_rejected_before_modification(paths, unsafe):
    source, raw, out = paths
    output = {
        "source": source,
        "inside_source": source / "output",
        "raw": raw,
        "inside_raw": raw / "output",
        "ancestor": source.parent,
        "code": launch.REPOSITORY / "src" / "unsafe",
    }[unsafe]
    with pytest.raises(ValueError, match="separate"):
        launch.supervise(source, raw, output, check)
    assert not (source / "session.lock").exists()
    assert not out.exists()


def test_symlink_into_input_rejected(paths):
    source, raw, out = paths
    out.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        launch.supervise(source, raw, out, check)
    assert not (raw / "session.lock").exists()


def test_timeout_terminates_process_group_and_spends_reserved_budget(paths, monkeypatch):
    worker, calls = fake_worker(
        monkeypatch, paths, code=9, failure=subprocess.TimeoutExpired("fixture", 7200)
    )
    killed = []
    monkeypatch.setattr(launch.os, "killpg", lambda *args: killed.append(args))
    assert launch.supervise(*paths, check) == 124
    assert killed[0][0] == worker.pid
    assert launch.read_json(paths[2] / "budget.json")["used_seconds"] == launch.SESSION_CAP
    assert launch.read_json(paths[2] / "status.json")["status"] == "timeout"
    assert launch.supervise(*paths, check) == 124
    assert len(calls) == 1


def test_interruption_preserves_remaining_budget(paths, monkeypatch):
    fake_worker(monkeypatch, paths, code=9, failure=InterruptedError("signal 15"))
    monkeypatch.setattr(launch.os, "killpg", lambda *args: None)
    assert launch.supervise(*paths, check) == 130
    assert launch.read_json(paths[2] / "status.json")["status"] == "stopped"
    assert launch.read_json(paths[2] / "budget.json")["used_seconds"] < launch.SESSION_CAP


def test_invalid_completion_never_launches_more_work(paths, monkeypatch):
    _, calls = fake_worker(monkeypatch, paths)
    launch.supervise(*paths, check)

    def bad_check(out):
        raise ValueError("completion hash mismatch")

    assert launch.supervise(*paths, bad_check) == 1
    assert len(calls) == 1
    assert "completion hash mismatch" in launch.read_json(paths[2] / "status.json")["error"]


def test_spawn_failure_is_recorded(paths, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(launch.subprocess, "Popen", fail)
    assert launch.supervise(*paths, check) == 1
    assert launch.read_json(paths[2] / "attempt_001.json")["status"]["error"] == "cannot spawn"


def test_export_failure_omits_cache_weights_and_keeps_evidence(paths, tmp_path):
    source, raw, out = paths
    out.mkdir()
    for name in ("metrics.json", "predictions.npz", "worker.log", "weights.pt", "cache/inputs.npz"):
        file = out / name
        file.parent.mkdir(exist_ok=True)
        file.write_text("payload")
    launch.status(out, "failed", error="numeric mismatch")
    log = tmp_path / "run.log"
    log.write_text("failure log")
    archive = launch.export_report(source, raw, out, log, tmp_path / "download", "all", 7)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        exported = json.load(tar.extractfile("out/export_status.json"))
        assert "out/metrics.json" in names and "out/predictions.npz" in names
        assert "out/worker.log" in names and "out/run.log" in names
        assert "out/roadmap_at_export.md" in names and "out/roadmap_at_export.json" in names
        assert not any(name.endswith(".pt") or "/cache/" in name for name in names)
        assert exported["command_exit_code"] == 7 and exported["run_status"] == "failed"
        assert exported["optimizer_updates"] == exported["readout_fits"] == 0


def test_unsafe_output_exports_only_error_receipt_and_roadmap(paths, tmp_path):
    source, raw, _ = paths
    (source / "must_not_copy.json").write_text("private input")
    archive = launch.export_report(
        source, raw, source, tmp_path / "missing.log", tmp_path / "download", "all", 1
    )
    with tarfile.open(archive) as tar:
        assert "source/must_not_copy.json" not in tar.getnames()
        exported = json.load(tar.extractfile("source/export_status.json"))
        assert exported["output_separation_error"]


def test_shell_export_on_worker_failure_and_bad_thread_environment(paths, tmp_path):
    source, raw, out = paths
    shim = tmp_path / "python-shim"
    shim.write_text(f"""#!{sys.executable}
import json, os, sys
if sys.argv[1:3] == ['-m', 'obson.babel.continuous_state_run']:
    from pathlib import Path
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir()
    (out / 'thread_env.json').write_text(json.dumps({{key: os.environ[key] for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}))
    raise SystemExit(7)
os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])
""")
    shim.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON_BIN": str(shim),
        "BABEL_CONTINUOUS_STATE_SOURCE": str(source),
        "BABEL_RAW_ROOT": str(raw),
        "BABEL_CONTINUOUS_STATE_RUN": str(out),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "OMP_NUM_THREADS": "bad",
        "OPENBLAS_NUM_THREADS": "bad",
        "MKL_NUM_THREADS": "bad",
    }
    result = subprocess.run(
        ["bash", str(launch.REPOSITORY / "scripts/babel_continuous_state768_autodl.sh"), "all"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7, result.stderr
    assert "command_exit_code=7, run_status=failed" in result.stdout
    assert launch.read_json(out / "thread_env.json") == launch.THREAD_ENV
    assert (tmp_path / "download/out_reports.tar.gz").is_file()


def test_lock_prevents_concurrent_supervision(paths):
    paths[2].mkdir()
    with (paths[2] / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            launch.supervise(*paths, check)
    assert not (paths[2] / "request.json").exists()


def test_zero_without_completion_is_failure(paths, monkeypatch):
    worker, _ = fake_worker(monkeypatch, paths)
    monkeypatch.setattr(worker, "wait", lambda timeout=None: 0)
    monkeypatch.setattr(worker, "poll", lambda: 0)
    assert launch.supervise(*paths, check) == 1
    assert "without immutable completion" in launch.read_json(paths[2] / "status.json")["error"]


def test_crash_reservation_prevents_unbounded_new_attempt(paths, monkeypatch):
    worker, calls = fake_worker(monkeypatch, paths)
    monkeypatch.setattr(worker, "wait", lambda timeout=None: (_ for _ in ()).throw(SystemExit(99)))
    with pytest.raises(SystemExit):
        launch.supervise(*paths, check)
    assert launch.read_json(paths[2] / "budget.json")["used_seconds"] == launch.SESSION_CAP
    assert launch.supervise(*paths, check) == 124
    assert len(calls) == 1


def test_export_cannot_recurse_into_its_own_stage(paths, tmp_path):
    source, raw, out = paths
    out.mkdir()
    with pytest.raises(ValueError, match="outside the output"):
        launch.export_report(
            source, raw, out, tmp_path / "missing.log", out / "download", "export", 0
        )
    assert not (out / "download").exists()
