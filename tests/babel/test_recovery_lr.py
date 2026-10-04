"""V03 synthetic fixtures only; no real weights/data, CUDA or production training."""

import copy
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_recovery_audit import STATS, inputs
from torch import nn

from obson.babel import recovery_lr as run
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json, atomic_save


@pytest.fixture(autouse=True)
def runtime():
    before = run.r0.runtime_state()
    run.r0.apply_runtime(dict(run.r0.RUNTIME_PROFILES["context"], threads=1))
    yield
    run.r0.apply_runtime(before)


def scores(v=1.0):
    return dict.fromkeys(run.FIELDS, v)


def test_gate_requires_all24_and_actual_retention():
    good = scores()
    assert run.retention(good, good)["passed"]
    damaged = dict(good)
    damaged["native/interior/near"] = 1.10001
    assert run.retention(damaged, good)["failed_fields"] == ["native/interior/near"]
    for bad in ({}, dict(good, unknown=1), scores(float("nan")), scores(-1)):
        with pytest.raises(ValueError):
            run.retention(bad, good)


def test_source_output_paths_and_existing_evidence_guard(tmp_path):
    s, q, o = [tmp_path / v for v in ("s", "q", "o")]
    for wrong in (s, s / "child", tmp_path):
        with pytest.raises(ValueError):
            run.paths(s, q, wrong)
    o.mkdir()
    (o / "do_not_touch").write_text("keep")
    with pytest.raises(ValueError, match="unbound"):
        run.supervise(s, q, o)
    assert (o / "do_not_touch").read_text() == "keep"


def toy_models(seed=42):
    torch.manual_seed(seed)
    return nn.Linear(2, 2), nn.Linear(2, 1)


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    atomic_json({"test": "synthetic"}, out / "manifest.json")
    e, q = toy_models()
    folder = source / "prefix_s42"
    folder.mkdir()
    atomic_json(
        {
            "initial_encoder_hash": run.model_signature(e, q)["encoder"],
            "initial_query_hash": run.model_signature(e, q)["query"],
            "initial_validation": scores(),
        },
        folder / "completion.json",
    )
    refs = [
        {
            "epoch": j,
            "validation": scores(1.3),
            "train": {
                "order_hash": str(j),
                "prefix_hash": str(j),
                "steps": 2,
                "examples": 3,
                "supervised_states": 15,
            },
        }
        for j in range(1, 6)
    ]

    def train(meta, job, e, q, x, opt, epoch, device):
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = (q(e(torch.ones(3, 2))) - 2).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(e.parameters()) + list(q.parameters()), 1.0)
            opt.step()
        return dict(refs[epoch - 1]["train"], loss=float(loss.detach()), seconds=0.01)

    def export(meta, e, q, x, device, path):
        np.savez(path, errors=np.zeros((3, 4)))
        return scores()

    monkeypatch.setattr(run.old, "construct", lambda meta, seed, device: toy_models(seed))
    monkeypatch.setattr(run.old, "validation", lambda *a: scores())
    monkeypatch.setattr(run.old, "train_epoch", train)
    monkeypatch.setattr(run, "export_validation", export)
    monkeypatch.setattr(run.r0, "causality", lambda *a: None)
    return SimpleNamespace(
        source=source,
        out=out,
        refs=refs,
        train=np.zeros((3, 255, 28), np.float32),
        meta={},
        epoch=train,
    )


def invoke(t):
    return run.train_seed(t.meta, t.source, t.out, 42, t.train, t.train, t.refs, "cpu")


def test_five_real_toy_epochs_and_idempotent_completion(tiny_run):
    t = tiny_run
    done = invoke(t)
    assert done["epochs"] == 5 and done["optimizer_updates"] == 10 and done["actual_updated_epoch5"]
    ck = torch.load(t.out / "s42/last.pt", weights_only=True)
    assert ck["epoch"] == 5 and len(ck["history"]) == 5
    assert all(h["observed_optimizer_steps"] == 2 for h in ck["history"])
    assert all(h["encoder_lr"] == run.RATES[42][0] for h in ck["history"])
    digest = run.sha256(t.out / "s42/last.pt")
    assert invoke(t) == done
    assert digest == run.sha256(t.out / "s42/last.pt")
    assert run.read_json(t.out / "s42/comparison.json")["high_lr_retention"]["passed"] is False


def test_boundary_commit_resume_matches_uninterrupted_adam(tiny_run, monkeypatch):
    t = tiny_run
    invoke(t)
    expected = torch.load(t.out / "s42/last.pt", weights_only=True)
    # A distinct synthetic output (not real experimental replay) for the interrupted control.
    import shutil

    shutil.rmtree(t.out / "s42")
    original = run.atomic_json
    interrupted = [False]

    def fail_after_checkpoint(value, path):
        if path.name == "checkpoint.json" and value["epoch"] == 2 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("injected after atomic commit")
        return original(value, path)

    monkeypatch.setattr(run, "atomic_json", fail_after_checkpoint)
    with pytest.raises(RuntimeError, match="atomic commit"):
        invoke(t)
    assert run.read_json(t.out / "s42/pending_epoch.json")["epoch"] == 2
    monkeypatch.setattr(run, "atomic_json", original)
    invoke(t)
    actual = torch.load(t.out / "s42/last.pt", weights_only=True)
    for group in ("encoder", "query"):
        for k, v in expected[group].items():
            torch.testing.assert_close(v, actual[group][k], rtol=0, atol=0)
    for key, values in expected["optimizer"]["state"].items():
        for k, v in values.items():
            torch.testing.assert_close(v, actual["optimizer"]["state"][key][k], rtol=0, atol=0)


def test_mid_epoch_interruption_never_silently_replays(tiny_run, monkeypatch):
    t = tiny_run

    def partial(*a, **kw):
        t.epoch(*a, **kw)
        raise RuntimeError("power interruption before commit")

    monkeypatch.setattr(run.old, "train_epoch", partial)
    with pytest.raises(RuntimeError):
        invoke(t)
    monkeypatch.setattr(run.old, "train_epoch", lambda *a: pytest.fail("must not replay"))
    with pytest.raises(ValueError, match="Uncommitted"):
        invoke(t)
    assert torch.load(t.out / "s42/last.pt", weights_only=True)["epoch"] == 0


def test_epoch0_optimizer_noop_cannot_pass(tiny_run, monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="update count"):
        invoke(tiny_run)


def test_corrupt_checkpoint_and_schedule_are_rejected(tiny_run, monkeypatch):
    t = tiny_run
    old = run.atomic_json

    def fail(value, path):
        if path.name == "checkpoint.json" and value["epoch"] == 1:
            raise RuntimeError("after commit")
        return old(value, path)

    monkeypatch.setattr(run, "atomic_json", fail)
    with pytest.raises(RuntimeError):
        invoke(t)
    monkeypatch.setattr(run, "atomic_json", old)
    ck = torch.load(t.out / "s42/last.pt", weights_only=True)
    ck["history"][0]["train"]["order_hash"] = "wrong"
    atomic_save(ck, t.out / "s42/last.pt")
    with pytest.raises(ValueError, match="Sampling"):
        invoke(t)


def test_pending_requires_exact_committed_boundary(tmp_path):
    atomic_json({"epoch": 2}, tmp_path / "pending_epoch.json")
    with pytest.raises(ValueError, match="Uncommitted"):
        run.pending_epoch(tmp_path, 1)
    with pytest.raises(ValueError, match="Stale"):
        run.pending_epoch(tmp_path, 3)
    run.pending_epoch(tmp_path, 2)
    assert not (tmp_path / "pending_epoch.json").exists()


def test_original_epoch_reuse_and_per_window_export(monkeypatch, tmp_path):
    cfg = {"width": 16, "heads": 4, "ff": 32, "layers": 4}
    e, q = uc.make_models("rolling", 42, "cpu", cfg)
    e.eval()
    q.eval()
    a, b = copy.deepcopy(e), copy.deepcopy(q)
    monkeypatch.setattr(run.old.parent, "stats", lambda _: (STATS, {}))
    meta = {"micro": 1, "batch": 2, "validation_prefixes": [32, 64, 96, 128]}
    x = inputs(3)
    opt = run.optimizer(e, q, run.RATES[42])
    original_opt = torch.optim.AdamW(
        [
            {"params": a.parameters(), "lr": run.RATES[42][0], "weight_decay": 0.01},
            {"params": b.parameters(), "lr": run.RATES[42][1], "weight_decay": 0.0001},
        ]
    )
    job = {"mode": "prefix", "seed": 42}
    r = run.old.train_epoch(meta, job, e, q, x, opt, 1, "cpu")
    run.old.train_epoch(meta, job, a, b, x, original_opt, 1, "cpu")
    assert r["steps"] == 2 and r["examples"] == 3
    assert run.model_signature(e, q) == run.model_signature(a, b)
    expected = run.old.validation(meta, e, q, x, "cpu")
    actual = run.export_validation(meta, e, q, x, "cpu", tmp_path / "rows.npz")
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-14, abs=1e-14)
    assert all(c["passed"] for c in run.r0.nested_compare(actual, expected).values())
    with np.load(tmp_path / "rows.npz") as arrays:
        assert all(arrays[k].shape == (3, 4) for k in arrays.files)


def test_supervisor_timeout_budget_survives_resume(tmp_path, monkeypatch):
    s, q, o = [tmp_path / k for k in ("s", "q", "o")]
    monkeypatch.setattr(run, "SESSION_CAP", 2)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))

    def timeout(*a, **kw):
        assert kw["timeout"] == 2
        raise subprocess.TimeoutExpired(a[0], 2)

    monkeypatch.setattr(run.subprocess, "run", timeout)
    assert run.supervise(s, q, o) == 124
    assert run.read_json(o / "budget.json")["used_seconds"] == 2
    assert run.read_json(o / "status.json")["status"] == "timeout"
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(s, q, o)


def test_supervisor_interruption_does_not_reset_budget(tmp_path, monkeypatch):
    s, q, o = [tmp_path / k for k in ("s", "q", "o")]

    def died(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(run.subprocess, "run", died)
    with pytest.raises(KeyboardInterrupt):
        run.supervise(s, q, o)
    assert run.read_json(o / "budget.json")["used_seconds"] == run.SESSION_CAP
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(s, q, o)


def test_wrong_qualification_pin_stops_before_source_read(tmp_path):
    qual = tmp_path / "qual"
    qual.mkdir()
    atomic_json({}, qual / "completion.json")
    with pytest.raises(ValueError, match="reviewed"):
        run.verify_qualification(tmp_path / "source", qual, tmp_path)


def test_failed_export_has_no_model_and_preserves_logs(tmp_path):
    out = tmp_path / "trial"
    out.mkdir()
    atomic_json({"status": "failed"}, out / "status.json")
    (out / "last.pt").write_text("not report")
    np.savez(out / "errors.npz", errors=np.ones(3))
    env = dict(
        os.environ,
        BABEL_R1_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN=str(Path(".venv/bin/python").resolve()),
    )
    p = subprocess.run(
        ["bash", "scripts/babel_recovery_r1_autodl.sh", "export"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    with tarfile.open(tmp_path / "download/trial_reports.tar.gz") as t:
        assert "trial/errors.npz" in t.getnames() and not any(
            n.endswith(".pt") for n in t.getnames()
        )
        assert "run_status=failed" in t.extractfile("trial/export_status.txt").read().decode()


def test_no_cuda_fails_with_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run.execute(tmp_path / "s", tmp_path / "q", tmp_path)
    assert run.read_json(tmp_path / "status.json")["status"] == "failed"


@pytest.mark.parametrize("passed", (True, False))
def test_full_orchestration_exports_gate_without_automatic_rs(tmp_path, monkeypatch, passed):
    source, qual, out = [tmp_path / k for k in ("source", "qual", "out")]
    for p in (source, qual, out):
        p.mkdir()
    for s in (42, 43):
        (source / f"prefix_s{s}").mkdir()
        atomic_json([{"epoch": i} for i in range(1, 6)], source / f"prefix_s{s}/history.json")
    atomic_json({}, qual / "reference_schedule.json")

    def verify(s, q, o):
        atomic_json({"unchanged": True}, o / "identity.json")
        return {}, {"torch": torch.__version__, "gpu": "synthetic"}

    monkeypatch.setattr(run, "verify_qualification", verify)
    monkeypatch.setattr(run, "implementation", lambda: {"synthetic": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic")
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=3 * 2**30))
    monkeypatch.setattr(
        run.old.data,
        "cache",
        lambda source, split: (np.zeros((4789 if split == "train" else 978, 1)), [{}] * 978),
    )
    monkeypatch.setattr(
        run.r0, "schedule_audit", lambda s, m, n, o: atomic_json({}, o / "reference_schedule.json")
    )
    calls = []

    def worker(meta, source, out, seed, *a):
        calls.append(seed)
        return {"epochs": 5, "optimizer_updates": 190, "gate": {"passed": passed}, "files": {}}

    monkeypatch.setattr(run, "train_seed", worker)
    assert run.execute(source, qual, out) == (0 if passed else 3)
    done = run.read_json(out / "completion.json")
    assert (
        done["optimizer_updates"] == 380
        and not done["rs_authorized"]
        and done["r1_passed"] == passed
    )
    assert calls == [42, 43]
    assert run.execute(source, qual, out) == (0 if passed else 3)
    assert calls == [42, 43]  # completed run never retrains
    for n, h in done["files"].items():
        assert run.sha256(out / n) == h


def test_supervisor_charges_all_attempts(tmp_path, monkeypatch):
    s, q, o = [tmp_path / k for k in ("s", "q", "o")]
    remaining = []

    def finished(*a, **kw):
        remaining.append(kw["timeout"])
        return SimpleNamespace(returncode=1)

    clock = iter([10, 14, 20, 23])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(run.subprocess, "run", finished)
    run.supervise(s, q, o)
    run.supervise(s, q, o)
    assert remaining == [run.SESSION_CAP, run.SESSION_CAP - 4]
    assert run.read_json(o / "budget.json")["used_seconds"] == 7
