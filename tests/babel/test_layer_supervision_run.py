"""Synthetic checkpoint/continuation lifecycle; no real model or optimizer updates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from obson.babel import layer_supervision_run as run

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
    e, q = run.model.make_models("control", seed, "cpu", CONFIG)
    opt = run.training.new_optimizer(e, q)
    for p in opt.param_groups[0]["params"]:
        opt.state[p] = {
            "step": torch.tensor(4560.0),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.full_like(p, 0.1),
        }
    return {
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
    write(tmp_path / "manifest.json", meta)
    rng = np.random.default_rng(123)
    x = rng.normal(0, 0.003, (3, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    raw = (100 * np.exp(rng.normal(0, 0.001, (3, 255)).cumsum(1))).astype(np.float64)
    return meta, x, raw, ck


def step_counter_only(self, closure=None):
    # Account for hypothetical updates without modifying parameters or moments.
    for p in self.param_groups[0]["params"]:
        self.state[p]["step"].add_(1)


def test_protocol_is_bounded_and_separates_source_and_continuation():
    p = run.protocol()
    assert p["epochs"] == 60 and p["source_updates"] == 4560
    assert p["source_epochs"] == 120 and p["session_cap_seconds"] == 24 * 3600
    assert len(p["jobs"]) == 4 and len(p["evaluation_models"]) == 6
    assert p["optimizer"]["restore"] == "all moments/steps, no reset"
    assert "zero readout fits" in p["F11"]
    assert run.training.schedule(3, 42, 1)[1].shape == (3, 5)


def test_restore_source_requires_all_original_moments_and_signature(tmp_path, monkeypatch):
    meta, _, _, ck = experiment(tmp_path, monkeypatch)
    e, q = run.model.make_models("layer2", 42, "cpu", CONFIG)
    opt = run.training.new_optimizer(e, q)
    receipt = run.restore_source(meta, run.jobs()[1], e, q, opt)
    assert receipt["complete_adam_restored"] and not receipt["optimizer_reset"]
    run.validate_optimizer(opt, e, q, 0, 0)
    assert all(float(s["step"]) == 4560 for s in opt.state.values())
    broken = copy.deepcopy(ck)
    broken["optimizer"]["state"].pop(next(iter(broken["optimizer"]["state"])))
    monkeypatch.setattr(run.sources, "checkpoint", lambda *args: broken)
    with pytest.raises(ValueError, match="support"):
        run.restore_source(meta, run.jobs()[0], e, q, opt)


def test_restore_source_never_accepts_changed_tensor(tmp_path, monkeypatch):
    meta, _, _, ck = experiment(tmp_path, monkeypatch)
    ck["encoder"]["input.bias"][0] += 1
    monkeypatch.setattr(run.sources, "checkpoint", lambda *args: ck)
    e, q = run.load_pair(meta, run.jobs()[0], "cpu")
    with pytest.raises(ValueError, match="signature"):
        run.restore_source(meta, run.jobs()[0], e, q, run.training.new_optimizer(e, q))


@pytest.mark.parametrize("job_index", [0, 1])
def test_committed_continuation_keeps_cumulative_steps_and_no_weight_update(
    tmp_path, monkeypatch, job_index
):
    meta, x, raw, ck = experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_counter_only)
    job = run.jobs()[job_index]
    done = run.train_job(meta, tmp_path, job, x, raw, x[:2], raw[:2], "cpu")
    assert done["updates"] == 4 and done["cumulative_updates"] == 4564
    saved = run.recover_checkpoint(tmp_path / job["name"], run.binding(tmp_path, job))
    assert saved["signature"] == ck["signature"]
    assert [r["global_epoch"] for r in saved["history"]] == [121, 122]
    assert [r["alpha"] for r in saved["history"]] == pytest.approx([0.02, 0.04])
    assert all(float(v["step"]) == 4564 for v in saved["optimizer"]["state"].values())
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
    assert seen == [2] and done["cumulative_updates"] == 4564


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
