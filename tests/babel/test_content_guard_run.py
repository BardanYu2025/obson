"""Synthetic checkpoint/continuation lifecycle; no real model or optimizer updates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from obson.babel import content_guard_run as run

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
    threads, fast = torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fast)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def synthetic_state(seed=42):
    e, q = run.model.make_models("plain", seed, "cpu", CONFIG)
    opt = run.training.new_optimizer(e, q)
    for p in opt.param_groups[0]["params"]:
        opt.state[p] = {
            "step": torch.tensor(6840.0),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.full_like(p, 0.1),
        }
    return {
        "epoch": 60,
        "history": [{"global_epoch": 121 + i} for i in range(60)],
        "encoder": run._cpu(e),
        "query": run._cpu(q),
        "optimizer": opt.state_dict(),
        "rng": run.rng_state(),
        "signature": run.training.parameter_signature(e, q),
    }


def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "EPOCHS", 2)
    monkeypatch.setattr(run, "BATCH", 2)
    monkeypatch.setattr(run, "MICRO", 1)
    monkeypatch.setitem(run.data.COUNTS, "train", 3)
    ck = synthetic_state()
    monkeypatch.setattr(run.sources, "checkpoint", lambda meta, seed: copy.deepcopy(ck))
    meta = {
        "encoder": CONFIG,
        "statistics": STATS,
        "counts": {"train": 3},
        "source_identity": {"synthetic": True},
        "model_source": "synthetic",
    }
    meta["source_identity"]["controls"] = {
        str(seed): {"signature": ck["signature"]} for seed in (42, 43)
    }
    write(tmp_path / "manifest.json", meta)
    write(
        tmp_path / "multiscale_raw_train.json",
        {"raw_sha256": "a" * 64, "rows_file_sha256": "b" * 64},
    )
    rng = np.random.default_rng(123)
    x = rng.normal(0, 0.003, (3, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    raw = (100 * np.exp(rng.normal(0, 0.001, (3, 255)).cumsum(1))).astype(np.float64)
    run.prepare_references(meta, tmp_path, x, raw, "cpu")
    return meta, x, raw, ck


def step_counter_only(self, closure=None):
    # Synthetic accounting only: do not update parameters or moments.
    for group in self.param_groups:
        for p in group["params"]:
            if p not in self.state:
                self.state[p] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
            self.state[p]["step"].add_(1)


def test_protocol_is_bounded_and_separates_source_and_continuation():
    p = run.protocol()
    assert p["epochs"] == 60 and p["source_updates"] == 6840
    assert p["source_epochs"] == 180 and p["session_cap_seconds"] == 24 * 3600
    assert len(p["jobs"]) == 6 and len(p["evaluation_models"]) == 8
    assert p["optimizer"]["inherited_lr"] == 1.5e-5
    assert "memory_lr" not in p["optimizer"]
    assert "zero readout fits" in p["F11"]
    assert run.training.schedule(3, 42, 1)[1].shape == (3, 5)


def test_restore_source_requires_all_original_moments_and_signature(tmp_path, monkeypatch):
    meta, _, _, ck = experiment(tmp_path, monkeypatch)
    e, q = run.model.make_models("plain", 42, "cpu", CONFIG)
    opt = run.training.new_optimizer(e, q)
    receipt = run.restore_source(meta, run.jobs()[1], e, q, opt)
    assert receipt["complete_inherited_adam_restored"] and not receipt["optimizer_reset"]
    run.validate_optimizer(opt, e, q, 0, 0)
    assert all(float(opt.state[p]["step"]) == 6840 for p in opt.param_groups[0]["params"])
    assert len(opt.param_groups) == 1
    broken = copy.deepcopy(ck)
    broken["optimizer"]["state"].pop(next(iter(broken["optimizer"]["state"])))
    monkeypatch.setattr(run.sources, "checkpoint", lambda *args: broken)
    opt = run.training.new_optimizer(e, q)
    with pytest.raises(ValueError, match="support|missing|Missing"):
        run.restore_source(meta, run.jobs()[0], e, q, opt)


def test_restore_source_never_accepts_changed_tensor(tmp_path, monkeypatch):
    meta, _, _, ck = experiment(tmp_path, monkeypatch)
    ck["encoder"]["input.bias"][0] += 1
    monkeypatch.setattr(run.sources, "checkpoint", lambda *args: ck)
    e, q = run.load_pair(meta, run.jobs()[0], "cpu")
    with pytest.raises(ValueError, match="signature"):
        run.restore_source(meta, run.jobs()[0], e, q, run.training.new_optimizer(e, q))


@pytest.mark.parametrize("job_index", [0, 1, 2])
def test_committed_continuation_keeps_cumulative_steps_and_no_weight_update(
    tmp_path, monkeypatch, job_index
):
    meta, x, raw, ck = experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    job = run.jobs()[job_index]
    done = run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")
    assert done["updates"] == 4 and done["cumulative_updates"] == 6844
    saved = run.recover_checkpoint(tmp_path / job["name"], run.binding(tmp_path, job))
    assert saved["signature"] == saved["initial_signature"]
    assert [r["global_epoch"] for r in saved["history"]] == [181, 182]
    assert [r["head_passes"] for r in saved["history"]] == [1, 1]
    groups = saved["optimizer"]["param_groups"]
    assert all(
        float(saved["optimizer"]["state"][pid]["step"]) == 6844 for pid in groups[0]["params"]
    )
    if len(groups) == 2:
        assert all(
            float(saved["optimizer"]["state"][pid]["step"]) == 4 for pid in groups[1]["params"]
        )
    assert run.check_job(meta, tmp_path, job) == done
    monkeypatch.setattr(
        run.training, "train_epoch", lambda *a, **kw: pytest.fail("completed reran")
    )
    assert run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu") == done


def test_resume_completed_epoch_does_not_reset_adam(tmp_path, monkeypatch):
    meta, x, raw, ck = experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    original_commit = run.commit_checkpoint

    def interrupt_after_commit(folder, state):
        original_commit(folder, state)
        if state["epoch"] == 1:
            raise RuntimeError("after committed epoch")

    monkeypatch.setattr(run, "commit_checkpoint", interrupt_after_commit)
    job = run.jobs()[0]
    with pytest.raises(RuntimeError, match="after committed"):
        run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")
    monkeypatch.setattr(run, "commit_checkpoint", original_commit)
    seen = []
    original_train = run.training.train_epoch

    def spy(*args, **kwargs):
        seen.append(kwargs["epoch"])
        return original_train(*args, **kwargs)

    monkeypatch.setattr(run.training, "train_epoch", spy)
    done = run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")
    assert seen == [2] and done["cumulative_updates"] == 6844


@pytest.mark.parametrize("observed", [0, 1])
def test_uncertain_updates_never_replay(tmp_path, observed):
    bind = {"seed": 42}
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": bind})
    write(
        tmp_path / "pending_epoch.json", {"epoch": 1, "binding": bind, "observed_steps": observed}
    )
    with pytest.raises(ValueError, match="automatic replay forbidden"):
        run.recover_checkpoint(tmp_path, bind)


def test_commit_receipt_recovery_requires_matching_intent(tmp_path, monkeypatch):
    bind = {"seed": 42}
    run.commit_checkpoint(tmp_path, {"epoch": 0, "binding": bind})
    original = run.atomic_json

    def interrupted(value, path):
        if path.name == "checkpoint.json":
            raise RuntimeError("receipt lost")
        original(value, path)

    monkeypatch.setattr(run, "atomic_json", interrupted)
    with pytest.raises(RuntimeError):
        run.commit_checkpoint(tmp_path, {"epoch": 1, "binding": bind})
    monkeypatch.setattr(run, "atomic_json", original)
    assert run.recover_checkpoint(tmp_path, bind)["epoch"] == 1


def test_replay_checks_retain_failure_evidence_and_exact_shapes():
    result = run.compare_replay({"v": [0.1, 2.0]}, {"v": [0.1, 3.0]})
    assert result["/v/0"]["passed"] and not result["/v/1"]["passed"]
    with pytest.raises(ValueError, match="sequence"):
        run.compare_replay({"v": [1]}, {"v": [1, 2]})


def test_cpu_real_execution_rejected_before_source_or_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run.sources, "verify", lambda *a: pytest.fail("source read"))
    assert run.execute(Path("unused"), tmp_path) == 1
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert "AutoDL CUDA" in failure["error"]


def test_no_active_budget_or_expired_worker_cannot_run(tmp_path):
    write(tmp_path / "budget.json", {"used_seconds": 0, "attempts": []})
    with pytest.raises(ValueError, match="active reserved"):
        run.remaining_seconds(tmp_path)
    write(tmp_path / "budget.json", {"used_seconds": run.SESSION_CAP, "attempts": []})
    with pytest.raises(ValueError, match="exhausted"):
        run.reserve_budget(tmp_path)


def test_source_and_run_cannot_overlap_before_creation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="Separate"):
        run.supervise(source, source / "new")
    assert not (source / "new").exists()


def qualified_synthetic_experiment(tmp_path, monkeypatch):
    """Exercise the real old/new model replay boundary with small random tensors."""
    meta = {
        "encoder": CONFIG,
        "statistics": STATS,
        "counts": {"train": 3},
        "source_identity": {"synthetic": True},
        "model_source": "synthetic",
    }
    write(tmp_path / "manifest.json", meta)
    rng = np.random.default_rng(491)
    x = rng.normal(0, 0.003, (3, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    raw = (100 * np.exp(rng.normal(0, 0.001, (3, 255)).cumsum(1))).astype(np.float64)
    checkpoints = {}
    for seed in (42, 43):
        ck = synthetic_state(seed)
        e, q = run.parent_model.make_models("control", seed, "cpu", CONFIG)
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
        ck["history"][-1]["validation"] = run.parent.validation(
            meta, e, q, x[:1], raw[:1], "control", "cpu", epoch=60
        )
        checkpoints[seed] = ck
    monkeypatch.setattr(
        run.sources, "checkpoint", lambda meta, seed: copy.deepcopy(checkpoints[seed])
    )
    meta["source_identity"]["controls"] = {
        str(seed): {"signature": ck["signature"]} for seed, ck in checkpoints.items()
    }
    write(tmp_path / "manifest.json", meta)
    write(
        tmp_path / "multiscale_raw_train.json",
        {"raw_sha256": "a" * 64, "rows_file_sha256": "b" * 64},
    )
    return meta, x, raw, checkpoints


def test_source_qualification_replays_local60_and_cumulative180_without_updates(
    tmp_path, monkeypatch
):
    meta, x, raw, checkpoints = qualified_synthetic_experiment(tmp_path, monkeypatch)
    validated = []
    original_validator = run.parent.validate_optimizer

    def validate(opt, encoder, query, epoch, steps):
        validated.append((epoch, steps))
        return original_validator(opt, encoder, query, epoch, steps)

    monkeypatch.setattr(run.parent, "validate_optimizer", validate)
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("qualification updated weights")
    )
    result = run.qualify_source(meta, tmp_path, x, x[:1], raw[:1], "cpu")
    assert validated == [(60, 2280), (60, 2280)]
    assert result["passed"] and result["updates"] == 0
    for seed in (42, 43):
        receipt = run.check_initial(meta, tmp_path, seed)
        assert receipt["epoch"] == 180 and receipt["source_run_epoch"] == 60
        assert receipt["updates"] == 0
        assert receipt["signature"] == checkpoints[seed]["signature"]
        assert (
            result["results"][str(seed)]["validation"]
            == checkpoints[seed]["history"][-1]["validation"]
        )
        replay = run.read_json(tmp_path / f"initial_s{seed}" / "source_replay.json")
        for arm in run.training.ARMS:
            assert replay["checks"][arm + "/final_state_exact"]["passed"]
            assert replay["checks"][arm + "/final_prediction_exact"]["passed"]
    monkeypatch.setattr(
        run.sources, "checkpoint", lambda *a: pytest.fail("qualified source replayed")
    )
    assert run.qualify_source(meta, tmp_path, x, x[:1], raw[:1], "cpu") == result


def test_failed_source_validation_keeps_numeric_evidence_and_no_initial_receipt(
    tmp_path, monkeypatch
):
    meta, x, raw, checkpoints = qualified_synthetic_experiment(tmp_path, monkeypatch)
    checkpoints[42]["history"][-1]["validation"]["components"]["objective"] += 1.0
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("qualification updated weights")
    )
    with pytest.raises(ValueError, match="Source validation/parity replay differs"):
        run.qualify_source(meta, tmp_path, x, x[:1], raw[:1], "cpu")
    evidence = run.read_json(tmp_path / "initial_s42/source_replay.json")
    assert not evidence["passed"]
    assert not evidence["checks"]["/components/objective"]["passed"]
    assert not (tmp_path / "initial_s42/checkpoint.json").exists()
    assert not (tmp_path / "source_validation.json").exists()
    with pytest.raises(ValueError, match="Incomplete source qualification"):
        run.qualify_source(meta, tmp_path, x, x[:1], raw[:1], "cpu")


@pytest.fixture
def complete_synthetic_matrix(tmp_path, monkeypatch):
    meta, x, raw, _ = qualified_synthetic_experiment(tmp_path, monkeypatch)
    run.prepare_references(meta, tmp_path, x, raw, "cpu")
    # Qualify against the real historical local60/cumulative180 validator first.
    run.qualify_source(meta, tmp_path, x, x[:1], raw[:1], "cpu")
    monkeypatch.setattr(run, "EPOCHS", 1)
    monkeypatch.setattr(run, "BATCH", 2)
    monkeypatch.setattr(run, "MICRO", 1)
    monkeypatch.setitem(run.data.COUNTS, "train", len(x))
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    for job in run.jobs():
        run.train_job(meta, tmp_path, job, x, raw, x[:1], raw[:1], "cpu")
    return meta, tmp_path


def test_complete_eight_model_lock_binds_equal_architecture_and_references(
    complete_synthetic_matrix,
):
    meta, out = complete_synthetic_matrix
    lock = run.lock_models(meta, out)
    assert set(lock["models"]) == {job["name"] for job in run.evaluation_jobs()}
    assert lock["readout_fits"] == 0
    for seed in (42, 43):
        assert lock["references"][str(seed)] == run.sha256(out / f"training_reference_s{seed}.json")
        assert {
            lock["models"][f"{arm}_s{seed}"]["initial_signature"] for arm in run.training.ARMS
        } == {lock["models"][f"initial_s{seed}"]["signature"]}
        assert {lock["models"][f"{arm}_s{seed}"]["updates"] for arm in run.training.ARMS} == {2}
    assert run.lock_models(meta, out) == lock


@pytest.mark.parametrize("missing", ["control_s42", "fixed_s43", "initial_s43"])
def test_model_lock_requires_every_worker_and_initial_reference(complete_synthetic_matrix, missing):
    meta, out = complete_synthetic_matrix
    member = "checkpoint.json" if missing.startswith("initial") else "completion.json"
    (out / missing / member).unlink()
    error = ValueError if missing.startswith("initial") else FileNotFoundError
    with pytest.raises(error):
        run.lock_models(meta, out)
    assert not (out / "model_selection_lock.json").exists()


def test_reference_is_train_only_frozen_and_changed_lineage_rejected(tmp_path, monkeypatch):
    meta, x, raw, _ = experiment(tmp_path, monkeypatch)
    refs = {seed: run.load_reference(meta, tmp_path, seed) for seed in (42, 43)}
    monkeypatch.setattr(
        run.training, "fit_reference", lambda *a, **k: pytest.fail("reference fit repeated")
    )
    run.prepare_references(meta, tmp_path, x, raw, "cpu")
    assert refs[42]["parents"] == 3 and refs[42]["contexts"] == 15
    write(
        tmp_path / "multiscale_raw_train.json",
        {"raw_sha256": "c" * 64, "rows_file_sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="lineage"):
        run.load_reference(meta, tmp_path, 42)


def test_interrupted_reference_is_not_silently_refit(tmp_path, monkeypatch):
    meta, x, raw, _ = experiment(tmp_path, monkeypatch)
    (tmp_path / "training_reference_s42.json").unlink()
    write(tmp_path / "training_reference_s42.pending.json", {"interrupted": True})
    monkeypatch.setattr(
        run.training,
        "fit_reference",
        lambda *a, **k: pytest.fail("interrupted calibration retried"),
    )
    with pytest.raises(ValueError, match="Interrupted calibration"):
        run.prepare_references(meta, tmp_path, x, raw, "cpu")


def test_history_controller_recurrence_and_reference_ratios_are_verified(tmp_path, monkeypatch):
    meta, x, raw, _ = experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    job = run.jobs()[2]
    run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")
    saved = run.recover_checkpoint(tmp_path / job["name"], run.binding(tmp_path, job))
    ref = run.load_reference(meta, tmp_path, 42)
    history = copy.deepcopy(saved["history"])
    history[0]["multipliers_next"][0] += 0.1
    with pytest.raises(ValueError, match="trajectory"):
        run.history_check(history, job, len(x), run.BATCH, reference=ref)
    history = copy.deepcopy(saved["history"])
    history[0]["group_ratios"][0] += 0.1
    with pytest.raises(ValueError, match="risk ratios"):
        run.history_check(history, job, len(x), run.BATCH, reference=ref)
    assert saved["multipliers"] == saved["history"][-1]["multipliers_next"]
    final = run.read_json(tmp_path / job["name"] / "final_train_risk.json")
    assert final["checkpoint_sha256"] == run.sha256(tmp_path / job["name"] / "last.pt")
    assert final["reference_sha256"] == run.sha256(tmp_path / "training_reference_s42.json")
    assert final["support"] == ref["support"]
    np.testing.assert_allclose(final["risks"], ref["baseline"], rtol=1e-6, atol=1e-8)


def test_bound_checkpoint_controller_corruption_stops_before_next_update(tmp_path, monkeypatch):
    meta, x, raw, _ = experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    original_commit = run.commit_checkpoint

    def interrupt(folder, state):
        original_commit(folder, state)
        if state["epoch"] == 1:
            raise RuntimeError("stop after commit")

    monkeypatch.setattr(run, "commit_checkpoint", interrupt)
    job = run.jobs()[2]
    with pytest.raises(RuntimeError):
        run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")
    monkeypatch.setattr(run, "commit_checkpoint", original_commit)
    folder = tmp_path / job["name"]
    saved = run.recover_checkpoint(folder, run.binding(tmp_path, job))
    saved["multipliers"][0] += 0.1
    original_commit(folder, saved)
    monkeypatch.setattr(
        run.training, "train_epoch", lambda *a, **k: pytest.fail("corrupted controller updated")
    )
    with pytest.raises(ValueError, match="controller/reference|Controller trajectory"):
        run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")


def test_missing_source_stops_before_creating_output(tmp_path, monkeypatch):
    source = tmp_path / "missing"
    output = tmp_path / "new"
    monkeypatch.setattr(
        run.sources, "verify", lambda *a: (_ for _ in ()).throw(ValueError("source absent"))
    )
    with pytest.raises(ValueError, match="source absent"):
        run.supervise(source, output)
    assert not output.exists()


def test_preflight_actual_objective_additivity_and_equal_plain_cost(tmp_path, monkeypatch):
    meta, x, raw, _ = experiment(tmp_path, monkeypatch)
    meta["counts"].update(val=3, test=2, cross_research=2)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("preflight updated"))
    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (64 * 2**30, 64 * 2**30))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic zero-update device")
    monkeypatch.setattr(run, "remaining_seconds", lambda out: 24 * 3600)
    disk = run.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda out: type(disk)(100 * 2**30, 0, 100 * 2**30)
    )
    result = run.preflight(meta, tmp_path, x, raw, "cpu")
    assert result["passed"] and result["zero_updates"]
    assert result["checks"]["matched_plain_initialization"]
    assert result["checks"]["matched_parameter_count"]
    assert all(v["effective_batch_contribution_check"]["passed"] for v in result["arms"].values())


def test_first_epoch_numeric_gate_runs_early_and_does_not_require_bitwise_cuda_hash(tmp_path):
    sample = {
        "group_risks": [1.0] * 21,
        "components": {"query": 1.0, "objective": 1.0},
        "multipliers_used": [1.0] * 21,
        "group_support": [10] * 21,
        "prefix_hash": "p",
        "order_hash": "o",
        "parameter_signature": "a" * 64,
    }
    assert run.first_epoch_parity(tmp_path, 42) is None
    for arm in ("fixed", "feedback"):
        folder = tmp_path / f"{arm}_s42"
        folder.mkdir()
        item = copy.deepcopy(sample)
        if arm == "feedback":
            item["group_risks"][0] += 1e-8
            item["parameter_signature"] = "b" * 64
        (folder / "history.jsonl").write_text(json.dumps(item) + "\n")
    result = run.first_epoch_parity(tmp_path, 42, required=True)
    assert result["passed"] and not result["signature_exact_equal"]
    failed = tmp_path / "failed"
    for arm in ("fixed", "feedback"):
        folder = failed / f"{arm}_s42"
        folder.mkdir(parents=True)
        item = copy.deepcopy(sample)
        if arm == "feedback":
            item["group_risks"][0] += 1
        (folder / "history.jsonl").write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="first epoch differs"):
        run.first_epoch_parity(failed, 42)
    assert not run.read_json(failed / "first_epoch_parity_s42.json")["passed"]


def test_failed_first_epoch_parity_cannot_resume_into_epoch_two(tmp_path, monkeypatch):
    meta, x, raw, _ = experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    fixed, feedback = run.jobs()[1:3]
    run.train_job(meta, tmp_path, fixed, x, raw, x[:2], raw[:2], "cpu")
    original_train = run.training.train_epoch

    def inconsistent_first_report(*args, **kwargs):
        report = original_train(*args, **kwargs)
        if kwargs["epoch"] == 1:
            # Inject an observable matched-first-epoch failure after the synthetic
            # optimizer accounting, without changing the controller recurrence.
            report["components"]["query"] += 1.0
        return report

    monkeypatch.setattr(run.training, "train_epoch", inconsistent_first_report)
    with pytest.raises(ValueError, match="first epoch differs"):
        run.train_job(meta, tmp_path, feedback, x, raw, x[:2], raw[:2], "cpu")
    folder = tmp_path / feedback["name"]
    assert run.read_json(folder / "checkpoint.json")["epoch"] == 1
    assert (folder / "pending_epoch.json").exists()
    failure = run.read_json(tmp_path / "first_epoch_parity_s42.json")
    assert not failure["passed"]
    monkeypatch.setattr(
        run.training, "train_epoch", lambda *a, **k: pytest.fail("failed gate resumed updates")
    )
    with pytest.raises(ValueError, match="first epoch differs"):
        run.train_job(meta, tmp_path, feedback, x, raw, x[:2], raw[:2], "cpu")
    assert run.read_json(folder / "checkpoint.json")["epoch"] == 1
    assert run.read_json(tmp_path / "first_epoch_parity_s42.json") == failure
