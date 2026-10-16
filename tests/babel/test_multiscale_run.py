"""CPU synthetic checks of bounded execution, commits and immutable selection."""

import copy
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obson.babel import multiscale_run as run
from obson.babel import multiscale_training as training

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def synthetic_runtime(monkeypatch):
    threads = torch.get_num_threads()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    monkeypatch.setattr(run, "EPOCHS", 2)
    monkeypatch.setattr(run, "BATCH", 2)
    monkeypatch.setattr(run, "MICRO", 1)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def synthetic(n=3):
    rng = np.random.default_rng(564)
    x = rng.normal(0, 0.003, (n, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    raw = (100 * np.exp(rng.normal(0, 0.001, (n, 255)).cumsum(1))).astype(np.float64)
    return x, raw


def experiment(out):
    out.mkdir()
    meta = {
        "encoder": CONFIG,
        "statistics": STATS,
        "counts": {"train": 3},
        "source_identity": {"synthetic": True},
    }
    write(out / "manifest.json", meta)
    write(out / "multiscale_scales_receipt.json", {"synthetic": True})
    x, raw = synthetic()
    scales = training.fit_train_scales(raw, STATS, split="train")["scales"]
    return meta, x, raw, scales


def worker(out, meta, x, raw, scales, job=None):
    return run.train_job(meta, out, job or run.jobs()[0], x, raw, x[:2], raw[:2], scales, "cpu")


def test_two_phase_commit_recovers_only_matching_intent(tmp_path, monkeypatch):
    folder = tmp_path / "worker"
    binding = {"seed": 42}
    run.commit_checkpoint(folder, {"epoch": 0, "binding": binding, "value": torch.tensor([1.0])})
    write(folder / "pending_epoch.json", {"epoch": 1, "binding": binding, "observed_steps": 2})
    actual_atomic = run.atomic_json

    def crash_after_replace(value, path):
        if path.name == "checkpoint.json":
            raise RuntimeError("power loss after last.pt replacement")
        actual_atomic(value, path)

    monkeypatch.setattr(run, "atomic_json", crash_after_replace)
    with pytest.raises(RuntimeError, match="power loss"):
        run.commit_checkpoint(
            folder, {"epoch": 1, "binding": binding, "value": torch.tensor([2.0])}
        )
    assert (folder / "commit_intent.json").exists()
    monkeypatch.setattr(run, "atomic_json", actual_atomic)
    recovered = run.recover_checkpoint(folder, binding)
    assert recovered["epoch"] == 1 and recovered["value"].item() == 2
    assert not (folder / "commit_intent.json").exists()
    assert not (folder / "pending_epoch.json").exists()
    assert run.read_json(folder / "checkpoint.json")["sha256"] == run.sha256(folder / "last.pt")


@pytest.mark.parametrize("observed", [0, 1])
def test_uncommitted_future_epoch_never_repeats_even_zero_logged_steps(tmp_path, observed):
    binding = {"seed": 42}
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": binding})
    pending = {"epoch": 1, "binding": binding, "observed_steps": observed}
    write(tmp_path / "pending_epoch.json", pending)
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        run.recover_checkpoint(tmp_path, binding)
    assert run.read_json(tmp_path / "pending_epoch.json") == pending


def test_unreplaced_intent_and_corrupted_checkpoint_fail_before_loading(tmp_path, monkeypatch):
    binding = {"seed": 42}
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": binding})
    receipt = run.read_json(tmp_path / "checkpoint.json")
    write(
        tmp_path / "commit_intent.json",
        {"previous": receipt, "next": {"epoch": 1, "sha256": "0" * 64, "binding": binding}},
    )
    with pytest.raises(ValueError, match="Uncommitted checkpoint intent"):
        run.recover_checkpoint(tmp_path, binding)
    (tmp_path / "commit_intent.json").unlink()
    with (tmp_path / "last.pt").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Corruption must fail before unpickling")
    )
    with pytest.raises(ValueError, match="fingerprint"):
        run.recover_checkpoint(tmp_path, binding)


def test_initial_partial_commit_or_changed_binding_never_reinitializes(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "last.pt.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Incomplete initial"):
        run.recover_checkpoint(partial, {"seed": 42})
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": {"seed": 42}})
    with pytest.raises(ValueError, match="binding"):
        run.recover_checkpoint(tmp_path, {"seed": 43})


def stepped_optimizer():
    e, q = torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)
    opt = training.new_optimizer(e, q)
    opt.param_groups[0]["lr"] = training.learning_rate(1)
    sum(p.square().sum() for m in (e, q) for p in m.parameters()).backward()
    opt.step()
    return e, q, opt


@pytest.mark.parametrize(
    "damage",
    [
        "fractional_step",
        "negative_second_moment",
        "nan_moment",
        "missing_state",
        "wrong_lr",
        "parameter_order",
    ],
)
def test_optimizer_exact_recipe_support_and_numeric_checks(damage):
    e, q, opt = stepped_optimizer()
    run.validate_optimizer(opt, e, q, 1, 1)
    parameter = next(e.parameters())
    if damage == "fractional_step":
        opt.state[parameter]["step"].fill_(1.5)
    elif damage == "negative_second_moment":
        opt.state[parameter]["exp_avg_sq"].fill_(-0.1)
    elif damage == "nan_moment":
        opt.state[parameter]["exp_avg"].fill_(torch.nan)
    elif damage == "missing_state":
        del opt.state[parameter]
    elif damage == "wrong_lr":
        opt.param_groups[0]["lr"] *= 2
    else:
        opt.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError):
        run.validate_optimizer(opt, e, q, 1, 1)


def test_actual_tiny_worker_fixed_last_and_completed_reentry_has_no_model_load(
    tmp_path, monkeypatch
):
    out = tmp_path / "run"
    meta, x, raw, scales = experiment(out)
    done = worker(out, meta, x, raw, scales)
    assert done["epoch"] == 2 and done["updates"] == 4
    folder = out / run.jobs()[0]["name"]
    saved = torch.load(folder / "last.pt", weights_only=True)
    assert len(saved["history"]) == 2
    assert saved["history"][0]["validation"] is None
    assert saved["history"][1]["validation"]["contexts"] == 18
    assert all(state["step"].item() == 4 for state in saved["optimizer"]["state"].values())
    assert not (folder / "pending_epoch.json").exists()
    monkeypatch.setattr(
        run, "load_pair", lambda *a: pytest.fail("Completed worker must not restore a model")
    )
    assert worker(out, meta, x, raw, scales) == done
    with (folder / "history.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        worker(out, meta, x, raw, scales)


def test_committed_epoch_resume_matches_uninterrupted_parameters_and_adam(tmp_path, monkeypatch):
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    meta, x, raw, scales = experiment(baseline)
    experiment(resumed)
    worker(baseline, meta, x, raw, scales)
    publish = run.publish_history
    calls = []

    def interrupted(folder, history):
        calls.append(len(history))
        if len(history) == 1:
            raise RuntimeError("crash after committed epoch1")
        publish(folder, history)

    monkeypatch.setattr(run, "publish_history", interrupted)
    with pytest.raises(RuntimeError, match="committed epoch1"):
        worker(resumed, meta, x, raw, scales)
    assert calls == [0, 1]
    folder = resumed / run.jobs()[0]["name"]
    assert run.read_json(folder / "pending_epoch.json")["epoch"] == 1
    monkeypatch.setattr(run, "publish_history", publish)
    worker(resumed, meta, x, raw, scales)
    a = torch.load(baseline / run.jobs()[0]["name"] / "last.pt", weights_only=True)
    b = torch.load(folder / "last.pt", weights_only=True)
    assert a["signature"] == b["signature"]
    for key in ("encoder", "query"):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], atol=0, rtol=0)
    for key, state in a["optimizer"]["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, b["optimizer"]["state"][key][name], atol=0, rtol=0)


def test_post_step_failure_preserves_uncertainty_and_blocks_second_attempt(tmp_path, monkeypatch):
    out = tmp_path / "run"
    meta, x, raw, scales = experiment(out)
    original = training.train_epoch

    def crash(*args, **kwargs):
        record = kwargs["on_step"]

        def observed(receipt):
            record(receipt)
            raise RuntimeError("process died after optimizer step")

        return original(*args, **(kwargs | {"on_step": observed}))

    monkeypatch.setattr(training, "train_epoch", crash)
    with pytest.raises(RuntimeError, match="optimizer step"):
        worker(out, meta, x, raw, scales)
    folder = out / run.jobs()[0]["name"]
    assert run.read_json(folder / "pending_epoch.json")["observed_steps"] == 1
    assert run.read_json(folder / "checkpoint.json")["epoch"] == 0
    monkeypatch.setattr(
        training, "train_epoch", lambda *a, **k: pytest.fail("Uncertain updates must not repeat")
    )
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        worker(out, meta, x, raw, scales)


def test_matrix_lock_requires_all_four_matched_seed_starts_and_full_exposure(tmp_path):
    out = tmp_path / "run"
    meta, x, raw, scales = experiment(out)
    for job in run.jobs()[:3]:
        worker(out, meta, x, raw, scales, job)
    with pytest.raises(FileNotFoundError):
        run.lock_models(meta, out)
    worker(out, meta, x, raw, scales, run.jobs()[3])
    lock = run.lock_models(meta, out)
    assert set(lock["models"]) == {j["name"] for j in run.jobs()}
    for seed in (42, 43):
        assert (
            lock["models"][f"control_s{seed}"]["initial_signature"]
            == lock["models"][f"multiscale_s{seed}"]["initial_signature"]
        )
    bad = out / run.jobs()[0]["name"] / "completion.json"
    done = run.read_json(bad)
    done["updates"] -= 1
    write(bad, done)
    with pytest.raises(ValueError):
        run.lock_models(meta, out)


def test_budget_reservation_is_fail_closed_until_supervisor_settles(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SESSION_CAP", 10)
    write(tmp_path / "budget.json", {"used_seconds": 3.0, "attempts": []})
    saved, used, remaining = run.reserve_budget(tmp_path)
    assert used == 3 and remaining == 7 and saved["used_seconds"] == 10
    assert saved["attempts"][0]["reserved_seconds"] == 7
    with pytest.raises(ValueError, match="unresolved supervisor"):
        run.reserve_budget(tmp_path)


def test_independent_worker_deadline_stops_orphan_and_restores_handler(tmp_path, monkeypatch):
    handlers, timers = {}, []

    def previous(signum, frame):
        pass

    monkeypatch.setattr(run, "remaining_seconds", lambda *_: 12.5)
    monkeypatch.setattr(run.signal, "getsignal", lambda *_: previous)
    monkeypatch.setattr(
        run.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler)
    )
    monkeypatch.setattr(
        run.signal, "setitimer", lambda timer, seconds: timers.append((timer, seconds))
    )
    with pytest.raises(TimeoutError, match="deadline expired"), run.worker_deadline(tmp_path):
        assert timers == [(run.signal.ITIMER_REAL, 12.5)]
        handlers[run.signal.SIGALRM](run.signal.SIGALRM, None)
    assert handlers[run.signal.SIGALRM] is previous
    assert timers[-1] == (run.signal.ITIMER_REAL, 0)


def test_supervisor_timeout_kills_group_records_budget_and_keeps_evidence(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    monkeypatch.setattr(run, "SESSION_CAP", 10)
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    monkeypatch.setattr(run.data, "verify", lambda *a: {"synthetic": True})
    killed, commands = [], []

    class Child:
        pid = 99001

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -15

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return Child()

    monkeypatch.setattr(run.subprocess, "Popen", popen)
    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    ticks = iter((100.0, 103.0))
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))
    assert run.supervise(source, out) == 124
    assert killed == [(99001, run.signal.SIGTERM)]
    assert commands[0][1]["start_new_session"] is True
    assert "--worker" in commands[0][0]
    assert run.read_json(out / "status.json")["status"] == "timeout"
    budget = run.read_json(out / "budget.json")
    assert budget["used_seconds"] == 3 and budget["attempts"][0]["exit_code"] == 124


def test_completed_supervisor_checks_evidence_without_new_budget_or_process(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    out.mkdir()
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    request = {
        "schema": run.SCHEMA,
        "source": str(source),
        "output": str(out),
        "protocol": run.protocol(),
        "implementation": run.implementation(),
    }
    write(out / "request.json", request)
    write(out / "result.json", {"evidence": True})
    write(out / "completion.json", {"files": {"result.json": run.sha256(out / "result.json")}})
    monkeypatch.setattr(
        run, "reserve_budget", lambda *a: pytest.fail("Completed runs reserve no time")
    )
    monkeypatch.setattr(
        run.subprocess, "Popen", lambda *a, **k: pytest.fail("Completed runs launch no process")
    )
    assert run.supervise(source, out) == 0
    write(out / "result.json", {"evidence": False})
    with pytest.raises(ValueError, match="artifact changed"):
        run.supervise(source, out)


def test_supervisor_changed_protocol_rejects_before_launch(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    out.mkdir()
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    request = {
        "schema": run.SCHEMA,
        "source": str(source),
        "output": str(out),
        "protocol": run.protocol(),
        "implementation": run.implementation(),
    }
    request["protocol"]["batch"] = 123
    write(out / "request.json", request)
    monkeypatch.setattr(
        run, "reserve_budget", lambda *_: pytest.fail("Changed protocol must not reserve budget")
    )
    with pytest.raises(ValueError, match="Immutable binding changed"):
        run.supervise(source, out)


@pytest.fixture
def writing_source_verifier(tmp_path, monkeypatch):
    """Keep the production adapter; emulate legacy verification's real file contract."""
    names = (
        "source",
        "history_state",
        "readability",
        "coverage",
        "v06",
        "v07",
        "r1",
        "context",
        "qualification",
        "raw",
    )
    roots = {name: tmp_path / name for name in names}
    for name, root in roots.items():
        root.mkdir()
        write(root / "manifest.json", {"synthetic": name})
    for current, parent in zip(names[:7], names[1:8], strict=True):
        write(roots[current] / "manifest.json", {"source": str(roots[parent])})
    write(
        roots["r1"] / "manifest.json",
        {"source": str(roots["context"]), "qualification": str(roots["qualification"])},
    )
    write(roots["source"] / "completion.json", {"synthetic": True})
    meta = {
        "coverage": str(roots["coverage"]),
        "coverage_meta": {
            "context": str(roots["context"]),
            "statistics": copy.deepcopy(STATS),
            "source_meta": {
                "identity": {
                    "manifest": {"identity": {"training_manifest": {"raw_root": str(roots["raw"])}}}
                }
            },
        },
        "statistics": copy.deepcopy(STATS),
    }
    identity = {"source": str(roots["context"]), "synthetic_identity": True}
    calls, fail_after_write = [], []

    def legacy_verify(source, directory):
        assert source == roots["source"] and directory.is_dir()
        assert directory not in roots.values()
        calls.append(directory)
        # Same deterministic sibling-temp write and subsequent read/hash as R0/R1.
        run.atomic_json(identity, directory / "identity.json")
        assert run.read_json(directory / "identity.json") == identity
        assert len(run.sha256(directory / "identity.json")) == 64
        if fail_after_write:
            raise ValueError("Synthetic qualification mismatch after identity write")
        return copy.deepcopy(meta)

    def snapshot():
        return {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for root in roots.values()
            for path in root.rglob("*")
            if path.is_file()
        }

    monkeypatch.setattr(run.data.original, "verify", legacy_verify)
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    return SimpleNamespace(
        source=roots["source"],
        roots=roots,
        meta=meta,
        identity=identity,
        calls=calls,
        fail_after_write=fail_after_write,
        snapshot=snapshot,
    )


@pytest.mark.parametrize("already_exists", [False, True])
def test_startup_real_adapter_isolates_legacy_write_and_passes_output_guard(
    tmp_path, monkeypatch, writing_source_verifier, already_exists
):
    fixture = writing_source_verifier
    out = tmp_path / "new_experiment"
    if already_exists:
        out.mkdir()
    before = fixture.snapshot()
    first = run.data.verify(fixture.source, out)
    assert out.exists() == already_exists
    assert not out.exists() or not list(out.iterdir())
    assert first["source_verification"]["identity"] == fixture.identity
    assert (
        str(fixture.roots["qualification"] / "manifest.json")
        in first["source_verification"]["source_manifests"]
    )
    launches = []

    def popen(command, **kwargs):
        request = run.read_json(out / "request.json")
        assert request["source"] == str(fixture.source)
        assert request["output"] == str(out)
        assert run.read_json(out / "budget.json")["used_seconds"] == run.SESSION_CAP
        assert not (out / "identity.json").exists()
        launches.append(command)
        return SimpleNamespace(pid=99001, wait=lambda timeout: 0)

    monkeypatch.setattr(run.subprocess, "Popen", popen)
    assert run.supervise(fixture.source, out) == 0
    assert run.supervise(fixture.source, out) == 0
    assert len(launches) == 2 and all("--worker" in command for command in launches)
    assert len(run.read_json(out / "budget.json")["attempts"]) == 2
    assert run.data.verify(fixture.source, out) == first
    assert fixture.snapshot() == before
    assert fixture.calls and all(not directory.exists() for directory in fixture.calls)
    assert not (out / "identity.json").exists()


@pytest.mark.parametrize("forbidden", ["raw", "history_state", "v06", "qualification"])
def test_startup_rejects_raw_and_intermediate_source_before_output_mutation(
    tmp_path, monkeypatch, writing_source_verifier, forbidden
):
    fixture = writing_source_verifier
    before = fixture.snapshot()
    monkeypatch.setattr(
        run.subprocess, "Popen", lambda *a, **k: pytest.fail("Forbidden output must not launch")
    )
    root = fixture.roots[forbidden]
    for out in (root, root / "child_experiment"):
        with pytest.raises(ValueError, match="separate"):
            run.supervise(fixture.source, out)
        assert fixture.snapshot() == before
        assert not (root / "child_experiment").exists()
        assert not (root / "request.json").exists()
    assert all(not directory.exists() for directory in fixture.calls)


def test_failed_legacy_audit_never_creates_output_request_or_budget(
    tmp_path, monkeypatch, writing_source_verifier
):
    fixture = writing_source_verifier
    fixture.fail_after_write.append(True)
    before = fixture.snapshot()
    out = tmp_path / "not_created"
    monkeypatch.setattr(
        run.subprocess, "Popen", lambda *a, **k: pytest.fail("Failed audit must not launch")
    )
    with pytest.raises(ValueError, match="qualification mismatch"):
        run.supervise(fixture.source, out)
    assert not out.exists() and fixture.snapshot() == before
    assert len(fixture.calls) == 1 and not fixture.calls[0].exists()


def test_unbound_existing_output_is_not_adopted_or_identity_rewritten(
    tmp_path, monkeypatch, writing_source_verifier
):
    fixture = writing_source_verifier
    out = tmp_path / "unbound"
    out.mkdir()
    identity = out / "identity.json"
    write(identity, {"unrelated_existing_evidence": True})
    protected = tmp_path / "protected.txt"
    protected.write_text("preserve this target")
    (out / "identity.json.tmp").symlink_to(protected)
    before = (identity.read_bytes(), identity.stat().st_mtime_ns, fixture.snapshot())
    monkeypatch.setattr(
        run.subprocess, "Popen", lambda *a, **k: pytest.fail("Unbound output must not launch")
    )
    with pytest.raises(ValueError, match="Existing output lacks this run request"):
        run.supervise(fixture.source, out)
    assert (identity.read_bytes(), identity.stat().st_mtime_ns, fixture.snapshot()) == before
    assert protected.read_text() == "preserve this target"
    assert (out / "identity.json.tmp").is_symlink()
    assert not (out / "request.json").exists() and not (out / "budget.json").exists()


def test_readonly_reverification_preserves_completed_evidence_and_detects_tampering(
    tmp_path, monkeypatch, writing_source_verifier
):
    fixture = writing_source_verifier
    out = tmp_path / "completed"
    out.mkdir()
    verified = run.data.verify(fixture.source, out)
    write(out / "manifest.json", verified)
    write(out / "identity.json", {"already_bound_evidence": True})
    write(
        out / "request.json",
        {
            "schema": run.SCHEMA,
            "source": str(fixture.source),
            "output": str(out),
            "protocol": run.protocol(),
            "implementation": run.implementation(),
        },
    )
    write(
        out / "completion.json",
        {
            "files": {
                name: run.sha256(out / name)
                for name in ("manifest.json", "identity.json", "request.json")
            }
        },
    )
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    assert run.data.verify(fixture.source, out) == verified
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()} == before
    monkeypatch.setattr(run, "reserve_budget", lambda *a: pytest.fail("Completion reserves none"))
    monkeypatch.setattr(
        run.subprocess, "Popen", lambda *a, **k: pytest.fail("Completion must not launch")
    )
    assert run.supervise(fixture.source, out) == 0
    write(out / "identity.json", {"tampered_evidence": True})
    with pytest.raises(ValueError, match="artifact changed"):
        run.supervise(fixture.source, out)
    assert run.read_json(out / "identity.json") == {"tampered_evidence": True}


def test_worker_cuda_guard_saves_failure_before_source_or_model_access(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        run.data, "verify", lambda *a: pytest.fail("No source work after CUDA rejection")
    )
    assert run.execute(tmp_path / "source", tmp_path) == 1
    assert run.read_json(tmp_path / "status.json")["status"] == "failed"
    assert "AutoDL CUDA" in run.read_json(tmp_path / "failure.json")["error"]


def pipeline(tmp_path, monkeypatch):
    """Keep the runner control flow real; substitute only data/GPU/report work."""
    source, out = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    out.mkdir()
    write(source / "completion.json", {"synthetic_source": True})
    write(source / "manifest.json", {"synthetic_source": True})
    counts = {"train": 3, "val": 2, "test": 2, "cross_research": 2}
    monkeypatch.setattr(run.data, "COUNTS", counts)
    monkeypatch.setattr(run.coverage, "CONFIG", CONFIG)
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run.torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(run.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(run, "worker_deadline", lambda *_: nullcontext())
    monkeypatch.setattr(run.runtime, "replay_runtime", lambda *_: nullcontext())
    verified = {
        "statistics": STATS,
        "coverage_meta": {"statistics": STATS},
        "coverage": "synthetic",
    }
    events = []
    monkeypatch.setattr(run.data, "verify", lambda *_: copy.deepcopy(verified))

    def row_cache(meta, split):
        x, _ = synthetic(counts[split])
        rows = [{"key": f"AA/15/C{i}", "month": "2026-01", "row": 254 + i} for i in range(len(x))]
        return x, rows

    def prepare(meta, directory, splits):
        events.append(("prepare", tuple(splits)))
        for split in splits:
            raw_dir = directory / "raw"
            raw_dir.mkdir(exist_ok=True)
            path = raw_dir / f"{split}.npz"
            np.savez(path, close=synthetic(counts[split])[1])
            row_path = directory / f"{split}_rows.json"
            write(row_path, row_cache(meta, split)[1])
            write(
                directory / f"multiscale_raw_{split}.json",
                {"raw_sha256": run.sha256(path), "rows_file_sha256": run.sha256(row_path)},
            )

    def fit_scales(meta, directory, raw, rows):
        report = training.fit_train_scales(raw, STATS, split="train")
        write(directory / "multiscale_scales.json", report)
        write(
            directory / "multiscale_scales_receipt.json",
            {"sha256": run.sha256(directory / "multiscale_scales.json")},
        )
        return report

    monkeypatch.setattr(run.data, "cache", row_cache)
    monkeypatch.setattr(run.data, "prepare", prepare)
    monkeypatch.setattr(run.data, "closes", lambda directory, split: synthetic(counts[split])[1])
    monkeypatch.setattr(run.data, "fit_scales", fit_scales)
    monkeypatch.setattr(run, "preflight", lambda *a, **k: events.append(("preflight",)))
    load_pair = run.load_pair
    monkeypatch.setattr(run, "load_pair", lambda meta, job, device: load_pair(meta, job, "cpu"))
    train_job = run.train_job

    def train(meta, directory, job, x, raw, val, val_raw, scales, device):
        events.append(("train", job["name"]))
        return train_job(meta, directory, job, x, raw, val, val_raw, scales, "cpu")

    def extract(meta, directory, name, encoder, query, x, raw, rows, split, device, *, verify_lock):
        assert len(run.read_json(directory / "model_selection_lock.json")["models"]) == 4
        assert verify_lock(meta, directory, name, split)["source"] == run.source_identity(source)
        assert "scales" not in run.read_json(directory / "manifest.json")
        assert raw.dtype == np.float64 and x.dtype == np.float32
        events.append(("extract", name, split))
        return {"model": name, "cohort": split}

    def decide(banks, rows):
        assert set(banks) == set(rows) == {"test", "cross_research"}
        assert all(set(bank) == {j["name"] for j in run.jobs()} for bank in banks.values())
        events.append(("decision",))
        return {"limited_hypothesis_supported": False, "synthetic": True}

    monkeypatch.setattr(run, "train_job", train)
    monkeypatch.setattr(run.evaluation, "extract", extract)
    monkeypatch.setattr(run.evaluation, "decisions", decide)
    write(
        out / "request.json",
        {
            "schema": run.SCHEMA,
            "source": str(source),
            "output": str(out),
            "protocol": run.protocol(),
            "implementation": run.implementation(),
        },
    )
    started = run.time.time()
    write(
        out / "budget.json",
        {
            "used_seconds": run.SESSION_CAP,
            "deadline_unix": started + 60,
            "supervisor_pid": run.os.getppid(),
            "attempts": [{"started_unix": started, "reserved_seconds": 60}],
        },
    )
    return source, out, events


def test_execute_completes_entire_matrix_before_research_and_binds_reports(tmp_path, monkeypatch):
    source, out, events = pipeline(tmp_path, monkeypatch)
    assert run.execute(source, out) == 0
    assert [entry[1] for entry in events if entry[0] == "train"] == [j["name"] for j in run.jobs()]
    extracts = [entry for entry in events if entry[0] == "extract"]
    assert len(extracts) == 8 and len(set(extracts)) == 8
    assert events.index(("prepare", ("test",))) > max(
        i for i, e in enumerate(events) if e[0] == "train"
    )
    done = run.read_json(out / "completion.json")
    assert done["status"] == "complete_requires_review" and not done["promoted"]
    run.verify_files(out, done["files"])
    assert "budget.json" not in done["files"] and "status.json" not in done["files"]
    assert all(f"{j['name']}/completion.json" in done["files"] for j in run.jobs())
    # A completed root run is read-only despite the previous budget being reserved.
    monkeypatch.setattr(run, "reserve_budget", lambda *_: pytest.fail("No second budget"))
    assert run.supervise(source, out) == 0


def test_worker_changed_request_fails_before_preflight_or_model_loading(tmp_path, monkeypatch):
    source, out, events = pipeline(tmp_path, monkeypatch)
    request = run.read_json(out / "request.json")
    request["implementation"] = {"changed": True}
    write(out / "request.json", request)
    monkeypatch.setattr(
        run,
        "load_pair",
        lambda *a: pytest.fail("Changed request must stop before model allocation"),
    )
    assert run.execute(source, out) == 1
    assert not events
    assert "request/code" in run.read_json(out / "failure.json")["error"]


def test_worker_source_change_after_scoring_does_not_publish_completion(tmp_path, monkeypatch):
    source, out, events = pipeline(tmp_path, monkeypatch)
    original_decisions = run.evaluation.decisions

    def changed(banks, rows):
        write(source / "manifest.json", {"changed_during_execution": True})
        return original_decisions(banks, rows)

    monkeypatch.setattr(run.evaluation, "decisions", changed)
    assert run.execute(source, out) == 1
    assert events[-1] == ("decision",)
    assert not (out / "completion.json").exists()
    assert "Source changed" in run.read_json(out / "failure.json")["error"]


def test_evaluate_only_without_full_matrix_stops_before_preflight(tmp_path, monkeypatch):
    source, out, events = pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "load_pair", lambda *a: pytest.fail("No training/evaluation models"))
    assert run.execute(source, out, evaluate_only=True) == 1
    assert not any(e[0] in ("preflight", "train", "extract") for e in events)


@pytest.mark.parametrize(
    "damage", ["not_reserved", "settled", "changed_deadline", "over_cap", "expired", "no_attempt"]
)
def test_worker_rejects_stale_or_malformed_budget_before_execution(tmp_path, monkeypatch, damage):
    now = run.time.time()
    budget = {
        "used_seconds": run.SESSION_CAP,
        "deadline_unix": now + 60,
        "supervisor_pid": run.os.getppid(),
        "attempts": [{"started_unix": now, "reserved_seconds": 60}],
    }
    if damage == "not_reserved":
        budget["used_seconds"] = 0
    elif damage == "settled":
        budget["attempts"][0]["exit_code"] = 0
    elif damage == "changed_deadline":
        budget["deadline_unix"] += 1
    elif damage == "over_cap":
        budget["attempts"][0]["reserved_seconds"] = run.SESSION_CAP + 1
        budget["deadline_unix"] = now + run.SESSION_CAP + 1
    elif damage == "expired":
        budget["attempts"][0]["started_unix"] = now - 70
        budget["deadline_unix"] = now - 10
    else:
        budget["attempts"] = []
    write(tmp_path / "budget.json", budget)
    monkeypatch.setattr(
        run.sys,
        "argv",
        [
            "multiscale_run",
            "--source",
            str(tmp_path / "source"),
            "--out",
            str(tmp_path),
            "--worker",
        ],
    )
    monkeypatch.setattr(run, "execute", lambda *a: pytest.fail("Invalid ledger must not execute"))
    with pytest.raises(ValueError):
        run.main()


@pytest.mark.parametrize("seconds", [run.SESSION_CAP, 1])
def test_preflight_measures_actual_synthetic_backward_and_stops_without_updates(
    tmp_path, monkeypatch, seconds
):
    out = tmp_path / "run"
    meta, x, raw, scales = experiment(out)
    meta["counts"].update(val=2, test=2, cross_research=2)
    created = []
    load_pair = run.load_pair

    def observed_pair(meta, job, device):
        pair = load_pair(meta, job, "cpu")
        created.append((pair, training.parameter_signature(*pair)))
        return pair

    monkeypatch.setattr(run, "load_pair", observed_pair)
    monkeypatch.setattr(
        training, "new_optimizer", lambda *a: pytest.fail("Preflight performs no optimizer updates")
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2**20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 64 * 2**20)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 2**30, 12 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Synthetic stats; CPU execution")
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    monkeypatch.setattr(run, "remaining_seconds", lambda _: seconds)
    if seconds == 1:
        with pytest.raises(ValueError, match="preflight stopped without updates"):
            run.preflight(meta, out, x, raw, scales, "cpu")
    else:
        assert run.preflight(meta, out, x, raw, scales, "cpu")["passed"]
    report = run.read_json(out / "preflight" / "attempt_1.json")
    assert report["zero_updates"] and report["checks"]["zero_updates_same_parameters"]
    assert report["checks"]["time_room"] == (seconds != 1)
    assert len(created) == 1 and training.parameter_signature(*created[0][0]) == created[0][1]
    assert report["gpu_available_bytes"] == 8 * 2**30 + 64 * 2**20
    assert report["weight_save_fsync_seconds"] > 0
    assert not (out / "preflight_save.tmp").exists()
    assert all(v > 0 for v in report["measurements"].values())
    # Independently account every final Adam checkpoint, atomic replacement,
    # full128 float32 queries and selected8 float32/float64/bool evidence.
    checkpoint_bytes = report["parameter_bytes"] * 3 + 8 * 2**20
    full_predictions = 4 * 4 * 128 * 127 * 7 * 4
    selected = 4 * 4 * 8 * 127 * 7 * (4 + 8 + 1)
    assert report["disk_required_bytes"] >= 5 * checkpoint_bytes + full_predictions + selected
