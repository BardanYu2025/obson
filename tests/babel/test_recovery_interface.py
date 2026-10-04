"""V07 tests use synthetic models only, never historical weights."""

import copy
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_recovery_audit import CONFIG, STATS, inputs

from obson.babel import recovery_interface as run
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json, atomic_save


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    before = run.r0.runtime_state()
    run.r0.apply_runtime(dict(run.r0.RUNTIME_PROFILES["context"], threads=1))

    def forbidden(*a, **k):
        raise AssertionError("No optimizer in V07")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    yield
    run.r0.apply_runtime(before)


def models():
    return uc.make_models("rolling", 42, "cpu", CONFIG)


def test_decomposition_physical_constant_offset_and_unsupported_width():
    # Constant physical 0.2 offset: nonzero level, exactly zero trend.
    y = torch.zeros(2, 4, 127, 7)
    pred = y.clone()
    pred[..., 0] = 0.2 / (torch.arange(1, 128).sqrt() * STATS["delta_scale"][0])
    ps = torch.tensor([32, 64, 96, 128])
    mask = (torch.arange(1, 128)[None, None, :, None] < ps[None, :, None, None]).expand_as(y)
    values, support = run.structure_parts(pred, y, mask, STATS)
    assert (values["structure_level"] > 0).all()
    assert values["structure_trend"].max() < 1e-12
    assert not support["structure_level16"][:, 0].any()
    assert support["structure_level4"][:, 0].all()
    primary = 0.5 * (values["structure_level"] + values["structure_trend"])
    torch.testing.assert_close(primary, run.hs.metrics(pred, y, mask, STATS)["primary"])
    with pytest.raises(ValueError):
        run.structure_parts(pred[..., :-1, :], y, mask, STATS)


def test_load_pair_includes_normalizer_buffers_and_freezes():
    e, q = models()
    before = {"encoder": run.old.cpu_state(e), "query": run.old.cpu_state(q)}
    after = copy.deepcopy(before)
    after["query"]["state_mean"].fill_(0.125)
    states = {0: before, 5: after}
    run.load_pair(e, q, states, (0, 5))
    assert (q.state_mean == 0.125).all()
    assert not e.training and not q.training
    assert all(not p.requires_grad for m in (e, q) for p in m.parameters())
    run.load_pair(e, q, states, (5, 0))
    torch.testing.assert_close(q.state_mean, before["query"]["state_mean"])
    bad = copy.deepcopy(states)
    bad[5]["query"].pop("state_scale")
    with pytest.raises(RuntimeError):
        run.load_pair(e, q, bad, (0, 5))


def test_evaluation_matches_original_24_and_preserves_state(monkeypatch):
    e, q = models()
    meta = {"micro": 2, "validation_prefixes": [32, 64, 96, 128]}
    monkeypatch.setattr(run.old.parent, "stats", lambda m: (STATS, None))
    signature = run.lr.model_signature(e, q)
    arrays, support = run.evaluate(meta, e, q, inputs(3), "cpu")
    expected = run.old.validation(meta, e, q, inputs(3), "cpu")
    run.r0.require(run.r0.nested_compare(run.aggregates(arrays), expected), "original24")
    assert signature == run.lr.model_signature(e, q)
    assert all(p.grad is None for m in (e, q) for p in m.parameters())
    assert not support["native/far"][:, :2].any()
    assert run.describe(arrays, support)["native/far"][0]["mean"] is None
    assert run.describe(arrays, support)["full/far"][0]["n"] == 3
    arrays["full/far"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        run.describe(arrays, support)


def test_crossed_contrasts_known_effects_and_no_support():
    bank = {
        k: {"metric": np.full((3, 4), v)}
        for k, v in [("source", 1.0), ("encoder_swap", 3.0), ("query_swap", 4.0), ("joint", 10.0)]
    }
    support = {"metric": np.ones((3, 4), bool)}
    support["metric"][:, 0] = False
    report = run.contrasts(bank, support)["metric"]
    assert report[0]["source_ratios"]["joint"] is None
    assert report[1]["encoder_delta"]["mean"] == 2
    assert report[1]["query_delta"]["mean"] == 3
    assert report[1]["interaction"]["mean"] == 4
    assert report[1]["joint_delta"]["mean"] == 9
    assert report[1]["source_ratios"]["joint"] == 10


@pytest.fixture
def seed_fixture(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "out"
    (source / "s42").mkdir(parents=True)
    out.mkdir()
    monkeypatch.setattr(run.old, "construct", lambda *a: models())
    monkeypatch.setattr(run.old.parent, "stats", lambda m: (STATS, None))
    meta = {"micro": 2, "validation_prefixes": [32, 64, 96, 128]}
    e, q = models()
    initial = run.lr.model_signature(e, q)
    arrays, _ = run.evaluate(meta, e, q, inputs(3), "cpu")
    with torch.no_grad():
        next(e.parameters()).add_(0.001)
        q.state_mean.add_(0.002)
        q.output.bias.add_(0.003)
    final = run.lr.model_signature(e, q)
    last, _ = run.evaluate(meta, e, q, inputs(3), "cpu")
    binding = {"synthetic": True}
    atomic_save(
        {
            "epoch": 5,
            "binding": binding,
            "current_signature": final,
            "encoder": run.old.cpu_state(e),
            "query": run.old.cpu_state(q),
        },
        source / "s42/last.pt",
    )
    atomic_json(
        {"binding": binding, "initial_signature": initial, "final_signature": final},
        source / "s42/completion.json",
    )
    atomic_json(
        {"initial": run.aggregates(arrays), "low_lr_epoch5": run.aggregates(last)},
        source / "s42/comparison.json",
    )
    return SimpleNamespace(source=source, out=out, meta=meta)


def test_all_four_crosses_run_without_fitting_and_replay(seed_fixture):
    t = seed_fixture
    before = run.sha256(t.source / "s42/last.pt")
    run.run_seed(t.meta, t.source, t.out, 42, inputs(3), "cpu")
    for label in run.COMBINATIONS:
        assert (t.out / f"s42/{label}_errors.npz").exists()
        assert (t.out / f"s42/{label}_causality.json").exists()
    receipt = run.read_json(t.out / "s42/receipt.json")
    assert receipt["optimizer_updates"] == 0 and receipt["selected_model"] is None
    assert run.sha256(t.source / "s42/last.pt") == before
    assert (
        receipt["loaded_modules"]["encoder_swap"]["query"]
        == receipt["loaded_modules"]["source"]["query"]
    )
    assert (
        receipt["loaded_modules"]["query_swap"]["encoder"]
        == receipt["loaded_modules"]["source"]["encoder"]
    )


def test_replay_failure_stops_before_mixed_and_saves_evidence(seed_fixture):
    t = seed_fixture
    p = t.source / "s42/comparison.json"
    d = run.read_json(p)
    d["initial"]["full/endpoint/near"] += 1
    atomic_json(d, p)
    with pytest.raises(ValueError, match="mismatched"):
        run.run_seed(t.meta, t.source, t.out, 42, inputs(3), "cpu")
    assert (t.out / "s42/source_replay.json").exists()
    assert not (t.out / "s42/encoder_swap_errors.npz").exists()


def test_wrong_epoch_rejected(seed_fixture):
    t = seed_fixture
    p = t.source / "s42/last.pt"
    ck = torch.load(p, weights_only=True)
    ck["epoch"] = 0
    atomic_save(ck, p)
    with pytest.raises(ValueError, match="checkpoint identity"):
        run.run_seed(t.meta, t.source, t.out, 42, inputs(3), "cpu")


def test_paths_and_unbound_output_preserved(tmp_path):
    s = tmp_path / "source"
    s.mkdir()
    for out in (s, s / "child", tmp_path):
        with pytest.raises(ValueError):
            run.supervise(s, out)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("yes")
    with pytest.raises(ValueError, match="unbound"):
        run.supervise(s, out)
    assert (out / "keep").read_text() == "yes"


def test_completion_pin_rejects_wrong_source(tmp_path):
    atomic_json({}, tmp_path / "completion.json")
    with pytest.raises(ValueError, match="reviewed R1"):
        run.verify_source(tmp_path, tmp_path / "out")


@pytest.mark.parametrize("timeout", [False, True])
def test_supervisor_bounded_retries_and_timeout(tmp_path, monkeypatch, timeout):
    s, o = tmp_path / "source", tmp_path / "out"
    s.mkdir()
    calls = []

    def child(args, timeout):
        calls.append(timeout)
        if len(calls) == 1 and timed:
            raise subprocess.TimeoutExpired(args, timeout)
        return SimpleNamespace(returncode=0)

    timed = timeout
    monkeypatch.setattr(run.subprocess, "run", child)
    ticks = iter([0.0, 10.0, 20.0, 30.0])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))
    assert run.supervise(s, o) == (124 if timeout else 0)
    assert run.supervise(s, o) == 0
    assert calls == [1800.0, 1790.0]
    if timeout:
        assert run.read_json(o / "status.json")["status"] == "timeout"
    b = run.read_json(o / "budget.json")
    b["used_seconds"] = 1800
    atomic_json(b, o / "budget.json")
    with pytest.raises(ValueError, match="budget exhausted"):
        run.supervise(s, o)


def test_failure_and_export_shell_excludes_weights(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    out.mkdir()
    atomic_json({"status": "failed"}, out / "status.json")
    (out / "do_not_export.pt").write_text("synthetic placeholder")
    env = dict(
        os.environ,
        BABEL_INTERFACE_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN="/bin/false",
    )
    # Failure mode must preserve nonzero exit and still produce a report.
    # Use real Python to parse status, but an intentionally missing source to fail the worker.
    env["PYTHON_BIN"] = str(root / ".venv/bin/python")
    env["BABEL_INTERFACE_SOURCE"] = str(tmp_path / "missing")
    r = subprocess.run(
        ["bash", "scripts/babel_recovery_interface_autodl.sh", "all"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    archive = tmp_path / "download/run_reports.tar.gz"
    with tarfile.open(archive) as f:
        assert not any(n.endswith(".pt") for n in f.getnames())
        assert b"run_status=failed" in f.extractfile("run/export_status.txt").read()
    r = subprocess.run(
        ["bash", "scripts/babel_recovery_interface_autodl.sh", "export"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0


def test_bound_checkpoint_corruption_stops_before_loading(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "last.pt"
    asset.write_text("synthetic")
    atomic_json({"files": {"last.pt": run.sha256(asset)}}, source / "completion.json")
    monkeypatch.setattr(run, "R1_COMPLETION", run.sha256(source / "completion.json"))
    asset.write_text("changed")
    with pytest.raises(ValueError, match="Immutable file mismatch"):
        run.verify_source(source, tmp_path / "out")


def test_execute_failure_records_no_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def bad(*a):
        raise ValueError("source mismatch")

    monkeypatch.setattr(run, "verify_source", bad)
    with pytest.raises(ValueError, match="source mismatch"):
        run.execute(tmp_path / "source", tmp_path)
    status = run.read_json(tmp_path / "status.json")
    assert status["status"] == "failed"
    assert status["optimizer_updates"] == 0 and not status["rs_authorized"]
    assert not (tmp_path / "completion.json").exists()
