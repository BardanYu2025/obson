"""Synthetic CPU lifecycle checks; no real source weights or neural market execution."""

import copy
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from obson.babel import input_patch_run as run
from obson.babel import input_patch_training as training

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    threads = torch.get_num_threads()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    monkeypatch.setattr(run, "EPOCHS", 2)
    monkeypatch.setattr(run, "BATCH", 2)
    monkeypatch.setattr(run, "MICRO", 1)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fast)


def experiment(out):
    out.mkdir()
    meta = {
        "encoder": CONFIG,
        "statistics": STATS,
        "counts": {"train": 3},
        "source_identity": {"synthetic": True},
    }
    write(out / "manifest.json", meta)
    run.prepare_initializations(meta, out, "cpu")
    x, raw = synthetic()
    return meta, x, raw


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


def worker(out, meta, x, raw, job=None):
    return run.train_job(meta, out, job or run.jobs()[0], x, raw, x[:2], raw[:2], "cpu")


@pytest.mark.parametrize("observed", [0, 1])
def test_uncommitted_future_epoch_never_repeats_even_zero_logged_steps(tmp_path, observed):
    binding = {"seed": 42}
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": binding})
    pending = {"epoch": 1, "binding": binding, "observed_steps": observed}
    write(tmp_path / "pending_epoch.json", pending)
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        run.recover_checkpoint(tmp_path, binding)
    assert run.read_json(tmp_path / "pending_epoch.json") == pending


def test_initial_partial_commit_or_changed_binding_never_reinitializes(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "last.pt.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Incomplete initial"):
        run.recover_checkpoint(partial, {"seed": 42})
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": {"seed": 42}})
    with pytest.raises(ValueError, match="binding"):
        run.recover_checkpoint(tmp_path, {"seed": 43})


def test_actual_tiny_worker_fixed_last_and_completed_reentry_has_no_model_load(
    tmp_path, monkeypatch
):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    done = worker(out, meta, x, raw)
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
    assert worker(out, meta, x, raw) == done
    with (folder / "history.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        worker(out, meta, x, raw)


def test_committed_epoch_resume_matches_uninterrupted_parameters_and_adam(tmp_path, monkeypatch):
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    meta, x, raw = experiment(baseline)
    experiment(resumed)
    worker(baseline, meta, x, raw)
    publish = run.publish_history
    calls = []

    def interrupted(folder, history):
        calls.append(len(history))
        if len(history) == 1:
            raise RuntimeError("crash after committed epoch1")
        publish(folder, history)

    monkeypatch.setattr(run, "publish_history", interrupted)
    with pytest.raises(RuntimeError, match="committed epoch1"):
        worker(resumed, meta, x, raw)
    assert calls == [0, 1]
    folder = resumed / run.jobs()[0]["name"]
    assert run.read_json(folder / "pending_epoch.json")["epoch"] == 1
    monkeypatch.setattr(run, "publish_history", publish)
    worker(resumed, meta, x, raw)
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
    meta, x, raw = experiment(out)
    original = training.train_epoch

    def crash(*args, **kwargs):
        record = kwargs["on_step"]

        def observed(receipt):
            record(receipt)
            raise RuntimeError("process died after optimizer step")

        return original(*args, **(kwargs | {"on_step": observed}))

    monkeypatch.setattr(training, "train_epoch", crash)
    with pytest.raises(RuntimeError, match="optimizer step"):
        worker(out, meta, x, raw)
    folder = out / run.jobs()[0]["name"]
    assert run.read_json(folder / "pending_epoch.json")["observed_steps"] == 1
    assert run.read_json(folder / "checkpoint.json")["epoch"] == 0
    monkeypatch.setattr(
        training, "train_epoch", lambda *a, **k: pytest.fail("Uncertain updates must not repeat")
    )
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        worker(out, meta, x, raw)


def test_budget_reservation_is_fail_closed_until_supervisor_settles(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SESSION_CAP", 10)
    write(tmp_path / "budget.json", {"used_seconds": 3.0, "attempts": []})
    saved, used, remaining = run.reserve_budget(tmp_path)
    assert used == 3 and remaining == 7 and saved["used_seconds"] == 10
    assert saved["attempts"][0]["reserved_seconds"] == 7
    with pytest.raises(ValueError, match="unresolved supervisor"):
        run.reserve_budget(tmp_path)


def test_read_only_qualification_cost_can_exhaust_budget_before_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SESSION_CAP", 10)
    with pytest.raises(ValueError, match="source qualification exhausted"):
        run.reserve_budget(tmp_path, startup_seconds=11)
    budget = run.read_json(tmp_path / "budget.json")
    assert budget["used_seconds"] == 10
    assert budget["attempts"][0]["source_qualification_seconds"] == 11
    assert budget["attempts"][0]["exit_code"] == 124
    with pytest.raises(ValueError, match="budget exhausted"):
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
    ticks = iter((90.0, 92.0, 100.0, 103.0))
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))
    assert run.supervise(source, out) == 124
    assert killed == [(99001, run.signal.SIGTERM)]
    assert commands[0][1]["start_new_session"] is True
    assert "--worker" in commands[0][0]
    assert run.read_json(out / "status.json")["status"] == "timeout"
    budget = run.read_json(out / "budget.json")
    assert budget["used_seconds"] == 5 and budget["attempts"][0]["exit_code"] == 124
    assert budget["attempts"][0]["source_qualification_seconds"] == 2
    assert budget["attempts"][0]["reserved_seconds"] == 8
    assert budget["attempts"][0]["total_elapsed_seconds"] == 5


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


def pipeline(tmp_path, monkeypatch):
    """Keep the runner control flow real; substitute only data/GPU/report work."""
    source, out = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    out.mkdir()
    write(source / "completion.json", {"synthetic_source": True})
    write(source / "manifest.json", {"synthetic_source": True})
    counts = {"train": 3, "val": 2, "test": 2, "cross_research": 2}
    monkeypatch.setattr(run.data, "COUNTS", counts)
    monkeypatch.setattr(run.model, "CONFIG", CONFIG)
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

    monkeypatch.setattr(run.data, "cache", row_cache)
    monkeypatch.setattr(run.data, "prepare", prepare)
    monkeypatch.setattr(run.data, "closes", lambda directory, split: synthetic(counts[split])[1])
    monkeypatch.setattr(
        run.data, "fit_scales", lambda *a: pytest.fail("No Haar fitting in this experiment")
    )
    monkeypatch.setattr(run, "preflight", lambda *a, **k: events.append(("preflight",)))
    initialize = run.prepare_initializations
    monkeypatch.setattr(
        run, "prepare_initializations", lambda meta, out, device: initialize(meta, out, "cpu")
    )
    load_pair = run.load_pair
    monkeypatch.setattr(run, "load_pair", lambda meta, job, device: load_pair(meta, job, "cpu"))
    train_job = run.train_job

    def train(meta, directory, job, x, raw, val, val_raw, device):
        events.append(("train", job["name"]))
        return train_job(meta, directory, job, x, raw, val, val_raw, "cpu")

    def extract(meta, directory, name, encoder, query, x, raw, rows, split, device, *, verify_lock):
        assert len(run.read_json(directory / "model_selection_lock.json")["models"]) == 6
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
    assert len(extracts) == 12 and len(set(extracts)) == 12
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
            "input_patch_run",
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
    meta, x, raw = experiment(out)
    meta["counts"].update(val=2, test=2, cross_research=2)
    created = []
    measured_calls = []
    actual_terms = training.training_terms

    def measured_terms(encoder, query, features, closes, positions, stats, arm):
        measured_calls.append((encoder.arm, arm, positions.shape[1], torch.is_grad_enabled()))
        return actual_terms(encoder, query, features, closes, positions, stats, arm)

    monkeypatch.setattr(training, "training_terms", measured_terms)
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
            run.preflight(meta, out, x, raw, "cpu")
    else:
        assert run.preflight(meta, out, x, raw, "cpu")["passed"]
    report = run.read_json(out / "preflight" / "attempt_1.json")
    assert report["zero_updates"] and report["checks"]["zero_updates_same_parameters"]
    assert report["checks"]["time_room"] == (seconds != 1)
    assert len(created) == 3
    assert all(training.parameter_signature(*pair) == sig for pair, sig in created)
    assert {pair[0].arm for pair, _ in created} == {"bar", "local4", "patch4"}
    assert created[1][1] == created[2][1]
    assert created[0][1] != created[1][1]
    for arm in ("bar", "local4", "patch4"):
        calls = [c for c in measured_calls if c[0] == arm]
        assert all(c[1] == arm for c in calls)
        assert sum(c[2] for c in calls if c[3]) == 3 * 5
        assert sum(c[2] for c in calls if not c[3]) == 9 + 128
    assert report["gpu_available_bytes"] == 8 * 2**30 + 64 * 2**20
    assert report["weight_save_fsync_seconds"] > 0
    assert not list(out.glob("preflight_*_save.tmp"))
    assert set(report["arms"]) == {"bar", "local4", "patch4"}
    assert report["arms"]["bar"]["attention_mode"] == "causal"
    assert report["arms"]["patch4"]["attention_mode"] == "causal"
    assert all(v > 0 for v in report["measurements"].values())
    # Independently account every final Adam checkpoint, atomic replacement,
    # full128 float32 queries and selected8 float32/float64/bool evidence.
    checkpoint_bytes = report["parameter_bytes"] * 3 + 8 * 2**20
    full_predictions = 6 * 4 * 128 * 127 * 7 * 4
    selected = 6 * 4 * 8 * 127 * 7 * (4 + 8 + 1)
    assert report["disk_required_bytes"] >= 7 * checkpoint_bytes + full_predictions + selected


def test_locked_scratch_protocol_and_initialization_both_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "EPOCHS", 120)
    monkeypatch.setattr(run, "BATCH", 128)
    monkeypatch.setattr(run, "MICRO", 8)
    p = run.protocol()
    assert p["session_cap_seconds"] == 86400 and p["epochs"] == 120 and len(p["jobs"]) == 6
    assert [(j["arm"], j["seed"]) for j in p["jobs"]] == [
        ("bar", 42),
        ("local4", 42),
        ("patch4", 42),
        ("patch4", 43),
        ("local4", 43),
        ("bar", 43),
    ]
    assert (
        p["source_completion"] == "835cce0452c27173f808b530871624b59041dceb71552074bf32e4c722e4139d"
    )
    assert p["optimizer"]["peak_lr"] == 3e-4 and "27360updates" in p["budget"]
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail("Scratch qualification cannot load any source checkpoint"),
    )
    out = tmp_path / "run"
    meta, _, _ = experiment(out)
    for seed in (42, 43):
        receipt = run.read_json(out / f"initialization_s{seed}.json")
        assert receipt["passed"] and all(receipt["checks"].values())
        arms = receipt["arms"]
        assert len({a["common_signature"] for a in arms.values()}) == 1
        assert (
            arms["local4"]["signature"] == arms["patch4"]["signature"] != arms["bar"]["signature"]
        )
        assert (
            arms["local4"]["parameter_count"] - arms["bar"]["parameter_count"]
            == 84 * CONFIG["width"]
        )
        assert arms["patch4"]["model_contract"]["token_count"] == 32
    assert run.model_contract("local4", CONFIG) == run.model.contract("local4", CONFIG)


def test_full_matrix_requires_six_workers_and_declared_input_difference(tmp_path):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    for job in run.jobs()[:-1]:
        worker(out, meta, x, raw, job)
    with pytest.raises(FileNotFoundError):
        run.lock_models(meta, out)
    worker(out, meta, x, raw, run.jobs()[-1])
    lock = run.lock_models(meta, out)
    assert len(lock["models"]) == 6 and lock["selection"] == "fixed_last120"
    for seed in (42, 43):
        assert (
            lock["models"][f"local4_s{seed}"]["initial_signature"]
            == lock["models"][f"patch4_s{seed}"]["initial_signature"]
        )
        assert (
            lock["models"][f"bar_s{seed}"]["initial_signature"]
            != lock["models"][f"patch4_s{seed}"]["initial_signature"]
        )
        assert lock["initialization"][str(seed)] == run.sha256(out / f"initialization_s{seed}.json")
    cost = run.cost_summary(meta, out)
    assert (
        len(cost["workers"]) == 6
        and cost["readout_fits"] == 0
        and cost["public_state_float32_bytes"] == 3072
    )
    assert all(v["updates"] == 4 for v in cost["workers"].values())


@pytest.mark.parametrize(
    "damage",
    [
        "model_dtype",
        "model_shape",
        "optimizer_dtype",
        "optimizer_step",
        "optimizer_recipe",
        "optimizer_foreach",
        "optimizer_capturable",
        "optimizer_fused",
        "optimizer_differentiable",
        "optimizer_missing_flag",
        "optimizer_step_dtype",
        "rng_dtype",
        "rng_count",
        "rng_numpy",
        "rng_python",
    ],
)
def test_preload_checkpoint_damage_rejected_before_restore(tmp_path, monkeypatch, damage):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    job = run.jobs()[0]
    worker(out, meta, x, raw, job)
    folder = out / job["name"]
    ck = torch.load(folder / "last.pt", weights_only=True)
    if damage == "model_dtype":
        ck["encoder"]["input.weight"] = ck["encoder"]["input.weight"].half()
    elif damage == "model_shape":
        ck["encoder"]["input.weight"] = ck["encoder"]["input.weight"][:1]
    elif damage == "optimizer_dtype":
        ck["optimizer"]["state"][0]["exp_avg"] = ck["optimizer"]["state"][0]["exp_avg"].half()
    elif damage == "optimizer_step":
        ck["optimizer"]["state"][0]["step"] -= 1
    elif damage == "optimizer_recipe":
        ck["optimizer"]["param_groups"][0]["lr"] *= 2
    elif damage in (
        "optimizer_foreach",
        "optimizer_capturable",
        "optimizer_fused",
        "optimizer_differentiable",
    ):
        ck["optimizer"]["param_groups"][0][damage.removeprefix("optimizer_")] = True
    elif damage == "optimizer_missing_flag":
        del ck["optimizer"]["param_groups"][0]["foreach"]
    elif damage == "optimizer_step_dtype":
        ck["optimizer"]["state"][0]["step"] = ck["optimizer"]["state"][0]["step"].double()
    elif damage == "rng_dtype":
        ck["rng"]["torch"] = ck["rng"]["torch"].float()
    elif damage == "rng_count":
        ck["rng"]["cuda"] = [torch.zeros(5, dtype=torch.uint8)]
    elif damage == "rng_numpy":
        ck["rng"]["numpy"] = ("MT19937", [0] * 4, 0, 0, 0.0)
    elif damage == "rng_python":
        ck["rng"]["python"] = (999, (), None)
    e, q = run.load_pair(meta, job, "cpu")
    monkeypatch.setattr(
        e, "load_state_dict", lambda *a, **k: pytest.fail("Must fail before casting load")
    )
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        run.validate_saved_state(ck, e, q, 2, 4)


def test_same_shaped_local_patch_checkpoint_cannot_change_stride(tmp_path):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    local, patch = run.jobs()[1:3]
    worker(out, meta, x, raw, local)
    with pytest.raises(ValueError, match="binding"):
        run.recover_checkpoint(out / local["name"], run.binding(out, patch))
    history = run.read_json(out / local["name"] / "training_summary.json")["history"]
    history[0]["model_contract"]["token_stride"] = 4
    with pytest.raises(ValueError, match="contract"):
        run.history_check(history, local, len(x), 2, config=CONFIG)


def test_source_cuda_guard_publishes_failure_without_weights(tmp_path, monkeypatch):
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    out.mkdir()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No source checkpoint load"))
    monkeypatch.setattr(run.data, "verify", lambda *a: pytest.fail("CUDA guard first"))
    assert run.execute(source, out) == 1
    assert "AutoDL CUDA" in run.read_json(out / "failure.json")["error"]


def test_two_phase_commit_repairs_only_matching_intent(tmp_path, monkeypatch):
    binding = {"seed": 42}
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": binding, "value": torch.tensor([1.0])})
    write(tmp_path / "pending_epoch.json", {"epoch": 1, "binding": binding, "observed_steps": 1})
    atomic = run.transactions.atomic_json

    def fail(value, path):
        if path.name == "checkpoint.json":
            raise RuntimeError("crash receipt")
        return atomic(value, path)

    monkeypatch.setattr(run.transactions, "atomic_json", fail)
    with pytest.raises(RuntimeError, match="crash receipt"):
        run.commit_checkpoint(
            tmp_path, {"epoch": 1, "binding": binding, "value": torch.tensor([2.0])}
        )
    monkeypatch.setattr(run.transactions, "atomic_json", atomic)
    assert run.recover_checkpoint(tmp_path, binding)["epoch"] == 1
    assert not (tmp_path / "pending_epoch.json").exists()
