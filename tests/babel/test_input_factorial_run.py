"""Synthetic CPU lifecycle checks; no real source weights or neural market execution."""

import copy
import json
import subprocess
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from obson.babel import input_factorial_run as run
from obson.babel import input_factorial_training as training

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
    monkeypatch.setattr(run, "BATCH", 4)
    monkeypatch.setattr(run, "MICRO", 2)
    monkeypatch.setattr(run, "DIAGNOSTIC_EPOCHS", (0, 2))
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fast)


def experiment(out):
    out.mkdir()
    meta = {
        "encoder": CONFIG,
        "statistics": STATS,
        "counts": {"train": 8},
        "source_identity": {"synthetic": True},
    }
    write(out / "manifest.json", meta)
    run.prepare_initializations(meta, out, "cpu")
    x, raw = synthetic()
    return meta, x, raw


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def synthetic(n=8):
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
    monkeypatch.setattr(run.previous, "SESSION_CAP", 10)
    write(tmp_path / "budget.json", {"used_seconds": 3.0, "attempts": []})
    saved, used, remaining = run.reserve_budget(tmp_path)
    assert used == 3 and remaining == 7 and saved["used_seconds"] == 10
    assert saved["attempts"][0]["reserved_seconds"] == 7
    with pytest.raises(ValueError, match="unresolved supervisor"):
        run.reserve_budget(tmp_path)


def test_read_only_qualification_cost_can_exhaust_budget_before_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SESSION_CAP", 10)
    monkeypatch.setattr(run.previous, "SESSION_CAP", 10)
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

    monkeypatch.setattr(run.previous, "remaining_seconds", lambda *_: 12.5)
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


def test_validation_reports_original_q_and_independent_physical_price(tmp_path):
    from obson.babel import multiscale_evaluate as scoring
    from obson.babel import multiscale_metrics as price

    meta, x, raw = experiment(tmp_path / "run")
    encoder, query = run.load_pair(meta, run.jobs()[0], "cpu")
    result = run.validation(meta, encoder, query, x, raw, run.jobs()[0]["arm"], "cpu")
    ps = torch.tensor(run.model.VAL_POSITIONS).expand(len(x), -1)
    with torch.no_grad():
        _, pred = run.model.predict(encoder, query, torch.as_tensor(x), ps)
    expected = price.price_rows(
        scoring._paths(pred.numpy(), STATS), price.raw_paths(raw, ps[0].numpy())
    )
    assert result["contexts"] == len(x) * 9
    assert set(result["physical_price_mse"]) == {"near", "mid", "far", "composite"}
    for key, values in expected.items():
        assert result["physical_price_mse"][key] == pytest.approx(float(values.mean()), rel=2e-5)
    assert result["components"]["objective"] == result["components"]["query"]


def test_failed_gradient_diagnostic_preserves_numbers_and_never_repeats(tmp_path, monkeypatch):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    job = run.jobs()[0]
    (out / job["name"]).mkdir()
    encoder, query = run.load_pair(meta, job, "cpu")
    monkeypatch.setattr(
        training, "gradient_diagnostic", lambda *a: {"passed": False, "max_abs": 0.123}
    )
    with pytest.raises(ValueError, match="numeric evidence saved"):
        run.diagnostic(meta, out, job, encoder, query, x, raw, 0, "cpu")
    saved = run.read_json(out / job["name"] / "gradient_diagnostic_epoch0.json")
    assert saved["complete"] and saved["report"]["max_abs"] == 0.123
    monkeypatch.setattr(training, "gradient_diagnostic", lambda *a: pytest.fail("must not repeat"))
    with pytest.raises(ValueError, match="evidence retained"):
        run.diagnostic(meta, out, job, encoder, query, x, raw, 0, "cpu")


def test_uncommitted_gradient_diagnostic_is_export_only(tmp_path, monkeypatch):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    job = run.jobs()[0]
    folder = out / job["name"]
    folder.mkdir()
    write(folder / "gradient_diagnostic_epoch0.pending.json", {"partial": True})
    encoder, query = run.load_pair(meta, job, "cpu")
    monkeypatch.setattr(training, "gradient_diagnostic", lambda *a: pytest.fail("must not repeat"))
    with pytest.raises(ValueError, match="export-only"):
        run.diagnostic(meta, out, job, encoder, query, x, raw, 0, "cpu")


@pytest.mark.parametrize(
    "damage", ["missing_moment", "wrong_step", "wrong_dtype", "negative_variance", "missing_rng"]
)
def test_strict_checkpoint_rejects_damaged_adam_or_rng(tmp_path, damage):
    out = tmp_path / "run"
    meta, x, raw = experiment(out)
    job = run.jobs()[0]
    worker(out, meta, x, raw, job)
    ck = torch.load(out / job["name"] / "last.pt", weights_only=True)
    encoder, query = run.load_pair(meta, job, "cpu")
    first = ck["optimizer"]["state"][0]
    if damage == "missing_moment":
        del first["exp_avg"]
    elif damage == "wrong_step":
        first["step"].add_(1)
    elif damage == "wrong_dtype":
        first["exp_avg"] = first["exp_avg"].double()
    elif damage == "negative_variance":
        first["exp_avg_sq"].fill_(-1)
    else:
        del ck["rng"]["numpy"]
    with pytest.raises(ValueError):
        run.validate_saved_state(ck, encoder, query, 2, 4)


def test_disk_budget_includes_eight_adam_atomic_replacement_ten_banks_and_export_copies():
    value = run.disk_requirement(121096732, 1423)
    assert value["checkpoint_total_bytes"] == 9 * value["checkpoint_bytes"]
    assert value["full128_predictions_bytes"] == 10 * 1423 * 128 * 127 * 7 * 4
    assert value["report_and_export_bytes"] == 4 * value["report_payload_estimate_bytes"]
    assert value["required_bytes"] / 2**30 == pytest.approx(19.708588, abs=1e-6)


def mock_pipeline(tmp_path, monkeypatch):
    """Exercise complete orchestration with synthetic state dictionaries, no market execution."""
    src, out = tmp_path / "source", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    events = []
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    monkeypatch.setattr(run, "remaining_seconds", lambda *_: 60)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(run.runtime, "replay_runtime", lambda *_: nullcontext())
    monkeypatch.setattr(run.model, "CONFIG", CONFIG)
    counts = {"train": 8, "val": 2, "test": 2, "cross_research": 2}
    monkeypatch.setattr(run.data, "COUNTS", counts)

    def pair(meta, job, device):
        torch.manual_seed(job["seed"])
        return torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)

    monkeypatch.setattr(run, "load_pair", pair)
    controls = {}
    for seed in (42, 43):
        encoder, query = pair({}, {"seed": seed}, "cpu")
        controls[str(seed)] = {
            "checkpoint": f"bar_s{seed}/last.pt",
            "checkpoint_sha256": "d" * 64,
            "updates": 4560,
            "signature": training.parameter_signature(encoder, query),
            "binding": {"seed": seed},
        }
    identity = {"controls": controls, "synthetic": True}
    verified = {
        "original_meta": {"source": str(src)},
        "verified_data": {"statistics": STATS},
        "identity": identity,
    }
    monkeypatch.setattr(run.source, "verify", lambda *_: copy.deepcopy(verified))

    def initialize(meta, directory, device):
        for seed in (42, 43):
            write(directory / f"initialization_s{seed}.json", {"synthetic": True})

    monkeypatch.setattr(run, "prepare_initializations", initialize)

    def prepare(meta, directory, splits):
        events.append(("prepare", tuple(splits)))
        for split in splits:
            raw_dir = directory / "raw"
            raw_dir.mkdir(exist_ok=True)
            (raw_dir / f"{split}.npz").write_bytes(b"fake-raw")
            write(directory / f"{split}_rows.json", [])
            write(
                directory / f"multiscale_raw_{split}.json",
                {
                    "raw_sha256": run.sha256(raw_dir / f"{split}.npz"),
                    "rows_file_sha256": run.sha256(directory / f"{split}_rows.json"),
                },
            )

    monkeypatch.setattr(run.data, "prepare", prepare)
    monkeypatch.setattr(run.data, "cache", lambda _, split: (synthetic(counts[split])[0], []))
    monkeypatch.setattr(run.data, "closes", lambda _, split: synthetic(counts[split])[1])
    monkeypatch.setattr(run, "preflight", lambda *a: events.append(("preflight",)))

    def train(meta, directory, job, *args):
        events.append(("train", job["name"]))
        write(directory / job["name"] / "completion.json", {"job": job})

    monkeypatch.setattr(run, "train_job", train)
    monkeypatch.setattr(
        run,
        "check_job",
        lambda meta, directory, job: run.read_json(directory / job["name"] / "completion.json"),
    )

    def checkpoint(job):
        encoder, query = pair({}, job, "cpu")
        return {
            "encoder": encoder.state_dict(),
            "query": query.state_dict(),
            "signature": training.parameter_signature(encoder, query),
        }

    monkeypatch.setattr(
        run, "recover_checkpoint", lambda folder, binding: checkpoint(binding["job"])
    )

    def source_checkpoint(meta, seed):
        assert len(run.read_json(out / "model_selection_lock.json")["models"]) == 8
        events.append(("reference_load", seed))
        return checkpoint({"seed": seed})

    monkeypatch.setattr(run.source, "checkpoint", source_checkpoint)
    monkeypatch.setattr(
        run, "validate_saved_state", lambda *a: events.append(("new_state_validation",))
    )
    monkeypatch.setattr(
        run.source,
        "validate_reference_state",
        lambda *a: events.append(("reference_state_validation",)),
        raising=False,
    )
    monkeypatch.setattr(run, "cost_summary", lambda *a: {"synthetic": True})

    def extract(meta, directory, name, *args, verify_lock):
        split = args[-2]
        verify_lock(meta, directory, name, split)
        events.append(("extract", name, split))
        return {"name": name, "split": split}

    monkeypatch.setattr(run.evaluation, "extract", extract)

    def decisions(banks, rows):
        assert set(banks) == set(rows) == {"test", "cross_research"}
        assert all(set(v) == {j["name"] for j in run.evaluation_jobs()} for v in banks.values())
        events.append(("decide",))
        return {"limited_hypothesis_supported": False}

    monkeypatch.setattr(run.evaluation, "decisions", decisions)
    write(out / "request.json", run.request(src, out))
    return src, out, events


def test_complete_eight_before_research_and_reference_weights_then_twenty_banks(
    tmp_path, monkeypatch
):
    src, out, events = mock_pipeline(tmp_path, monkeypatch)
    assert run.execute(src, out) == 0
    assert len([e for e in events if e[0] == "train"]) == 8
    assert len([e for e in events if e[0] == "extract"]) == 20
    assert len([e for e in events if e[0] == "reference_load"]) == 4
    assert len([e for e in events if e[0] == "new_state_validation"]) == 16
    assert len([e for e in events if e[0] == "reference_state_validation"]) == 4
    last_train = max(i for i, e in enumerate(events) if e[0] == "train")
    assert all(
        i > last_train for i, e in enumerate(events) if e[0] in ("reference_load", "extract")
    )
    assert events.index(("prepare", ("test",))) > last_train
    done = run.read_json(out / "completion.json")
    assert done["status"] == "complete_requires_review" and done["promoted"] is False
    run.verify_files(out, done["files"])
    assert "source_validation.json" in done["files"]
    assert len(run.read_json(out / "source_validation.json")["references"]) == 2
    monkeypatch.setattr(
        run, "reserve_budget", lambda *a: pytest.fail("Completed run must not reserve")
    )
    assert run.supervise(src, out) == 0


def test_incomplete_matrix_never_reads_research_or_reference_weights(tmp_path, monkeypatch):
    src, out, events = mock_pipeline(tmp_path, monkeypatch)
    assert run.execute(src, out, evaluate_only=True) == 1
    assert not any(e[0] in ("preflight", "train", "extract", "reference_load") for e in events)


def test_changed_request_stops_before_model_initialization(tmp_path, monkeypatch):
    src, out, events = mock_pipeline(tmp_path, monkeypatch)
    value = run.read_json(out / "request.json")
    value["implementation"] = {"changed": True}
    write(out / "request.json", value)
    monkeypatch.setattr(
        run, "prepare_initializations", lambda *a: pytest.fail("must stop before initialize")
    )
    assert run.execute(src, out) == 1
    assert not events
    assert "request/code" in run.read_json(out / "failure.json")["error"]


def test_supervisor_timeout_kills_group_records_budget_and_keeps_evidence(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    monkeypatch.setattr(run, "SESSION_CAP", 10)
    monkeypatch.setattr(run.previous, "SESSION_CAP", 10)
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    monkeypatch.setattr(run.source, "verify", lambda *a: {"synthetic": True})
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


def disk_fixture(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    meta = {"encoder": CONFIG, "counts": {"test": 2, "cross_research": 2}}
    write(out / "manifest.json", meta)
    for seed in (42, 43):
        write(out / f"initialization_s{seed}.json", {"fixture": True})
    budget = run.disk_requirement(1000, 4)
    job = run.jobs()[0]
    folder = out / job["name"]
    folder.mkdir()
    (folder / "last.pt").write_bytes(b"synthetic committed weights")
    write(
        folder / "checkpoint.json",
        {"epoch": 1, "binding": run.binding(out, job), "sha256": run.sha256(folder / "last.pt")},
    )
    return out, meta, budget, job


def test_remaining_disk_credits_only_bound_checkpoint_not_foreign_or_raw(tmp_path):
    out, meta, budget, job = disk_fixture(tmp_path)
    (out / "foreign.pt").write_bytes(b"foreign" * 10000)
    (out / "raw").mkdir()
    (out / "raw" / "train.npz").write_bytes(b"raw" * 10000)
    credit = run.allocated_disk_credit(meta, out, budget)
    actual = (out / job["name"] / "last.pt").stat().st_size
    assert credit["credited_bytes"] == actual
    assert credit["remaining_required_bytes"] == budget["required_bytes"] - actual
    assert set(credit["verified_files"]["checkpoints"]) == {job["name"] + "/last.pt"}


@pytest.mark.parametrize("damage", ["uncommitted", "hash", "binding"])
def test_disk_credit_rejects_uncommitted_or_unbound_checkpoint(tmp_path, damage):
    out, meta, budget, job = disk_fixture(tmp_path)
    folder = out / job["name"]
    if damage == "uncommitted":
        write(folder / "pending_epoch.json", {"epoch": 2})
    elif damage == "hash":
        (folder / "last.pt").write_bytes(b"changed")
    else:
        receipt = run.read_json(folder / "checkpoint.json")
        receipt["binding"] = {"other": True}
        write(folder / "checkpoint.json", receipt)
    with pytest.raises(ValueError):
        run.allocated_disk_credit(meta, out, budget)


def bank_disk_fixture(out, meta, job):
    name, split = job["name"], "test"
    lock = {
        "schema": run.SCHEMA,
        "manifest": run.sha256(out / "manifest.json"),
        "models": {j["name"]: {"signature": j["name"]} for j in run.jobs()},
        "references": {
            j["name"]: {"signature": j["name"]} for j in run.evaluation_jobs() if j.get("reference")
        },
    }
    write(out / "model_selection_lock.json", lock)
    prefix = f"evaluation/{name}/{split}"
    full = f"predictions/{name}/{split}/parents_00000_00002.npy"
    reports = [
        prefix + "/parent_rows.npz",
        prefix + "/position_rows.npz",
        f"results/{split}_{name}_predictions.npz",
    ]
    files = [full, *reports, prefix + "/rows.json", prefix + "/metrics.json"]
    for relative in files:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic disk allocation")
    receipt = {
        "schema": run.evaluation.SCHEMA,
        "status": "complete",
        "binding": {
            "name": name,
            "split": split,
            "parents": 2,
            "model_signature": name,
            "lock": {"model_lock": run.sha256(out / "model_selection_lock.json")},
        },
        "files": {p: run.sha256(out / p) for p in files},
    }
    write(out / prefix / "complete.json", receipt)
    return out / prefix / "complete.json", full, reports


def test_completed_bank_credit_is_capped_to_original_category_and_export_reserve_kept(tmp_path):
    out, meta, budget, job = disk_fixture(tmp_path)
    _, full, reports = bank_disk_fixture(out, meta, job)
    credit = run.allocated_disk_credit(meta, out, budget)
    assert credit["credited_by_category"]["full_predictions"] == (out / full).stat().st_size
    assert credit["credited_by_category"]["base_reports"] == sum(
        (out / p).stat().st_size for p in reports
    )
    smaller = budget | {"full128_predictions_bytes": 1, "report_payload_estimate_bytes": 1}
    capped = run.allocated_disk_credit(meta, out, smaller)
    assert capped["credited_by_category"]["full_predictions"] == 1
    assert capped["credited_by_category"]["base_reports"] == 1
    assert (
        credit["remaining_required_bytes"]
        >= budget["safety_bytes"] + 3 * budget["report_payload_estimate_bytes"]
    )


@pytest.mark.parametrize("damage", ["lock", "missing", "extra", "bytes"])
def test_bank_credit_rejects_wrong_lock_partial_inventory_or_changed_bytes(tmp_path, damage):
    out, meta, budget, job = disk_fixture(tmp_path)
    path, full, _ = bank_disk_fixture(out, meta, job)
    receipt = run.read_json(path)
    if damage == "lock":
        receipt["binding"]["lock"]["model_lock"] = "wrong"
    elif damage == "missing":
        receipt["files"].pop(full)
    elif damage == "extra":
        receipt["files"]["foreign.pt"] = "irrelevant"
    else:
        (out / full).write_bytes(b"changed")
    write(path, receipt)
    with pytest.raises(ValueError):
        run.allocated_disk_credit(meta, out, budget)


def test_disk_report_budget_counts_actual_uncompressed_score_array_schema():
    """Build real CPU score banks independently of the resource-count formula."""
    parents = 2
    positions = np.arange(1, 129)
    raw = np.full((parents, 255), 100.0, np.float64)
    pred = np.zeros((parents, 128, 127, 7), np.float32)
    truth = np.zeros(pred.shape, np.float64)
    masks = np.ones(pred.shape, bool)
    scored = run.scoring.score_rows(pred, truth, masks, raw, STATS, positions)
    parent = run.scoring.aggregate_rows(scored["errors"], scored["support"])
    selected = np.asarray(run.evaluation.SAMPLE_POSITIONS) - 1
    fixed = pred[:, selected]
    overlap = run.scoring.overlap_rows(fixed, raw, STATS)
    parent["errors"].update(overlap["errors"])
    parent["support"].update(overlap["support"])
    position_bytes = sum(
        a.nbytes for family in ("errors", "support") for a in scored[family].values()
    )
    parent_bytes = sum(
        a.nbytes for family in ("errors", "support", "contexts") for a in parent[family].values()
    )
    report = run.disk_requirement(121096732, parents)
    parts = report["report_parts_bytes"]
    assert parts["position_error_support"] == position_bytes * 10
    assert parts["parent_error_support_contexts"] == parent_bytes * 10
    assert parts["fixed_predictions_targets_masks"] == 10 * sum(
        a.nbytes for a in (fixed, truth[:, selected], masks[:, selected])
    )
    assert parts["fixed_raw_closes"] == raw.nbytes * 10
    assert report["report_payload_estimate_bytes"] == sum(parts.values())
    assert parts["array_container_headers"] > 0
    assert report["prepared_raw_price_cache_bytes"] > 0
    assert report["full128_headers_bytes"] > 0
    assert report["score_schema"] == {
        "position_metrics": 22,
        "regular_parent_metrics": 66,
        "overlap_parent_metrics": 45,
        "position_arrays_per_bank": 44,
        "parent_arrays_per_bank": 288,
        "banks": 20,
    }
    assert report["required_bytes"] == (
        report["checkpoint_total_bytes"]
        + report["full128_predictions_bytes"]
        + report["full128_headers_bytes"]
        + report["prepared_raw_price_cache_bytes"]
        + 4 * report["report_payload_estimate_bytes"]
        + report["safety_bytes"]
    )
