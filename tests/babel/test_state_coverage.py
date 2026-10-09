"""Synthetic-only risk tests for full-position targets, optimization and audit lifecycle."""

import copy
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import state_coverage as core
from obson.babel import state_coverage_data as data
from obson.babel import state_coverage_evaluate as ev
from obson.babel import state_coverage_run as run
from obson.babel.ae_extend import atomic_json

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def runtime():
    torch.set_num_threads(1)
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(old)


def inputs(n=3):
    raw = np.random.default_rng(96).normal(0, 0.01, (n, 255, 28)).astype("float32")
    raw[..., 23:28] = 1
    raw[..., 22] = 0.1
    return raw


def models():
    return core.make_models("rolling", 42, "cpu", CONFIG)


def meta(micro=1, batch=2):
    return {"micro": micro, "batch": batch, "statistics": STATS, "encoder": CONFIG}


def job(arm="B"):
    return {"arm": arm, "seed": 42, "name": f"{arm}_s42"}


def test_protocol_json_roundtrip_and_budget():
    assert json.loads(json.dumps(run.protocol())) == run.protocol()
    assert len(run.jobs()) == 8 and run.protocol()["updates"] == 36480
    assert data.COMPLETION == "d794c37e0fd8fe5717b1617a4225ffc9e39f667c4945c05254170eca985c01f2"


def test_sampler_pairing_exclusions_and_full128_coverage():
    plans = {a: core.schedule(4789, 42, 1, a) for a in core.ARMS}
    for a in core.ARMS:
        np.testing.assert_array_equal(plans[a][0], plans["A"][0])
    np.testing.assert_array_equal(plans["A"][1], plans["C"][1])
    np.testing.assert_array_equal(plans["B"][1], plans["D"][1])
    assert np.all(plans["A"][1][:, -1] == 128)
    assert not np.isin(plans["A"][1], [1, 31, 48, 80, 112]).any()
    assert set(plans["B"][1].ravel()) == set(range(1, 129))
    assert np.all(np.diff(plans["B"][1], axis=1) > 0)
    assert not np.array_equal(plans["B"][1], core.schedule(4789, 42, 2, "B")[1])


def test_every_endpoint_analytic_path_boundary_and_mask():
    x = np.zeros((1, 255, 28), dtype="float32")
    x[..., 0] = np.arcsinh(0.2)
    x[..., 1] = np.arcsinh(0.1)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    ps = torch.arange(1, 129)[None]
    y, m = core.targets(torch.tensor(x), ps, STATS)
    expected = -0.3 * np.sqrt(np.arange(1, 128)) / 0.1
    np.testing.assert_allclose(y[0, :, :, 0].numpy(), np.tile(expected, (128, 1)), rtol=1e-6)
    oy, om = core.oracle_targets(x, ps.numpy(), STATS)
    np.testing.assert_allclose(y, oy, atol=2e-6)
    np.testing.assert_array_equal(m, om)
    x[:, 0, 23:28] = 0
    _, m = core.targets(torch.tensor(x), ps, STATS)
    assert not m[0, 0, -1, 3:].any() and m[0, 1, -1, 3:].all()


def test_nontrivial_scaler_zero_sentinel_suspect_and_old_targets():
    raw = inputs(2).astype("float64")
    raw[:, 10:15, 22] = 0
    raw[:, 17, 21] = 2
    stats = copy.deepcopy(STATS)
    stats["x_mean"] = [0.123] * 28
    stats["x_scale"] = [0.81] * 28
    x = ((raw - stats["x_mean"]) / stats["x_scale"]).astype("float32")
    ps = torch.tensor([[1, 2, 31, 32, 127, 128]]).repeat(2, 1)
    y, m = core.targets(torch.tensor(x), ps, stats)
    oy, om = core.oracle_targets(x, ps.numpy(), stats)
    np.testing.assert_allclose(y, oy, atol=1e-5, rtol=2e-5)
    np.testing.assert_array_equal(m, om)
    oldy, oldm = core.original.targets(torch.tensor(x), ps[:, 3:], stats)
    torch.testing.assert_close(y[:, 3:], oldy, atol=0, rtol=0)
    assert torch.equal(m[:, 3:], oldm)


def test_trained_interface_causality_query_independence_and_missing_grad_routes():
    e, q = models()
    x = torch.tensor(inputs(2))
    assert core.audit(e, q, x, STATS)["passed"]
    ps = torch.tensor([[1, 128], [31, 127]])
    a = core.terms(e, q, x, ps, STATS, "A")
    c = core.terms(e, q, x, ps, STATS, "C")
    torch.testing.assert_close(a["objective"], c["objective"] + a["structure"])
    params = list(e.parameters()) + list(q.parameters())
    gq = torch.autograd.grad(c["objective"].sum(), params)
    ga = torch.autograd.grad(a["objective"].sum(), params, retain_graph=True)
    gs = torch.autograd.grad(a["structure"].sum(), params)
    for actual, left, right in zip(ga, gq, gs, strict=True):
        torch.testing.assert_close(actual, left + right, atol=1e-5, rtol=2e-4)
    assert sum(float(g.abs().sum()) for g in gq[: len(list(e.parameters()))]) > 0
    assert sum(float(g.abs().sum()) for g in gq[len(list(e.parameters())) :]) > 0


def test_actual_train_tail_micro_accumulation_equivalence():
    e, q = models()
    other = copy.deepcopy((e, q))
    x = inputs(3)

    # SGD isolates the accumulation/clip algebra. Adam amplifies near-zero rounding;
    # actual Adam state and resume are checked independently in the next test.
    def sgd(pair):
        return torch.optim.SGD([p for m in pair for p in m.parameters()], lr=0.01)

    a = run.train_epoch(meta(1), job(), e, q, x, sgd((e, q)), 1, "cpu")
    b = run.train_epoch(meta(2), job(), *other, x, sgd(other), 1, "cpu")
    for m, n in zip((e, q), other, strict=True):
        for p, v in zip(m.parameters(), n.parameters(), strict=True):
            torch.testing.assert_close(p, v, atol=2e-6, rtol=2e-4)
    assert a["steps"] == b["steps"] == 2 and a["supervised_states"] == 15
    assert sum(a["position_counts"]) == 15 and a["prefix_hash"] == b["prefix_hash"]
    assert all(np.isfinite(v) for v in a["components"].values())


def test_restore_after_atomic_commit_and_refuse_uncommitted_update(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "EPOCHS", 2)
    train = inputs(3)
    val = inputs(1)
    a = tmp_path / "continuous"
    b = tmp_path / "restart"
    a.mkdir()
    b.mkdir()
    for out in (a, b):
        atomic_json({"synthetic": True}, out / "manifest.json")
    torch.manual_seed(4)
    run.train_job(meta(), a, job(), train, val, "cpu")
    original = run.materialize

    def interrupt(ck, folder):
        original(ck, folder)
        if ck["epoch"] == 1:
            raise RuntimeError("synthetic process interruption after committed save")

    torch.manual_seed(4)
    with (
        patch.object(run, "materialize", side_effect=interrupt),
        pytest.raises(RuntimeError, match="synthetic process"),
    ):
        run.train_job(meta(), b, job(), train, val, "cpu")
    run.train_job(meta(), b, job(), train, val, "cpu")
    aa = torch.load(a / job()["name"] / "last.pt", weights_only=True)
    bb = torch.load(b / job()["name"] / "last.pt", weights_only=True)
    for family in ("encoder", "query"):
        for k, v in aa[family].items():
            torch.testing.assert_close(v, bb[family][k], atol=0, rtol=0)
    for k, v in aa["optimizer"]["state"].items():
        for field, tensor in v.items():
            torch.testing.assert_close(tensor, bb["optimizer"]["state"][k][field], atol=0, rtol=0)
    assert torch.equal(aa["rng"]["torch"], bb["rng"]["torch"])
    c = tmp_path / "uncommitted"
    c.mkdir()
    atomic_json({"synthetic": True}, c / "manifest.json")

    def stopped(*args, **kwargs):
        raise RuntimeError("uncommitted")

    with patch.object(run, "train_epoch", side_effect=stopped), pytest.raises(RuntimeError):
        run.train_job(meta(), c, job(), train, val, "cpu")
    with pytest.raises(ValueError, match="Uncommitted"):
        run.train_job(meta(), c, job(), train, val, "cpu")


def test_data_disjointness_and_source_rejection(tmp_path):
    train = [{"key": "a/15/A", "input_start": "2020-01-01", "end": "2020-02-01"}]
    with pytest.raises(ValueError, match="overlap"):
        data.row_audit(train, train)
    assert (
        data.row_audit(
            train, [{"key": "a/15/A", "input_start": "2020-03-01", "end": "2020-04-01"}]
        )["overlapping_intervals"]
        == 0
    )
    atomic_json({}, tmp_path / "completion.json")
    with pytest.raises(ValueError, match="reviewed V06"):
        data.verify(tmp_path, tmp_path / "out")


def rows(n=120):
    return [
        {
            "key": f"a/15/A{i % 3}",
            "symbol": "a",
            "month": f"2020-{i % 6 + 1:02d}",
            "week": str(i % 8),
            "period": 15,
        }
        for i in range(n)
    ]


def test_paired_inference_groups_p95_and_zero_reference():
    rs = rows()
    a = np.arange(120, dtype=float) + 1
    b = 2 * a
    ci = ev.paired(a, b, rs, draws=100)
    assert all(c["high"] < 0 and c["supported"] for c in ci.values())
    tail = ev.paired(a, b, rs, quantile=0.95, draws=100)
    assert tail["month"]["delta"] == pytest.approx(np.quantile(a, 0.95) - np.quantile(b, 0.95))
    assert not ev.gate(np.zeros(120), np.zeros(120), rs, 0.95)["passed"]
    assert not ev.gate(a[:10], b[:10], rs[:10])["passed"]


def test_research_not_read_before_model_readout_locks(tmp_path):
    with patch.object(data.source.old.data, "cache") as cached, pytest.raises(FileNotFoundError):
        data.cache({"context": "/unused"}, tmp_path, "test")
    cached.assert_not_called()
    with pytest.raises(FileNotFoundError):
        ev.check_readouts(tmp_path)


def test_budget_timeout_and_request_collision(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "out"
    with patch.object(run.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 1)):
        assert run.supervise(root, out) == 124
    assert json.loads((out / "status.json").read_text())["status"] == "timeout"
    budget = json.loads((out / "budget.json").read_text())
    assert 0 <= budget["used_seconds"] <= run.SESSION_CAP
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="unbound"):
        run.supervise(other, out)
    budget["used_seconds"] = run.SESSION_CAP
    atomic_json(budget, out / "budget.json")
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(root, out)


def test_export_success_failure_and_invalid_omp(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    out.mkdir()
    download = tmp_path / "download"
    source = tmp_path / "source"
    source.mkdir()
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        BABEL_COVERAGE_RUN=str(out),
        BABEL_COVERAGE_SOURCE=str(source),
        BABEL_DOWNLOAD_DIR=str(download),
        OMP_NUM_THREADS="bad",
    )
    atomic_json({"status": "complete"}, out / "status.json")
    (out / "keep.pt").write_text("not exported")
    (out / "cache").mkdir()
    np.savez(out / "cache/input.npz", x=[1])
    np.savez(out / "predictions.npz", x=[1])
    script = repo / "scripts/babel_state_coverage768_autodl.sh"
    p = subprocess.run(["bash", str(script), "export"], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    with tarfile.open(download / "run_reports.tar.gz") as t:
        assert (
            "run/predictions.npz" in t.getnames()
            and "run/keep.pt" not in t.getnames()
            and "run/cache/input.npz" not in t.getnames()
        )
        assert b"command_exit_code=0" in t.extractfile("run/export_status.txt").read()
    failed = tmp_path / "failed"
    env["BABEL_COVERAGE_RUN"] = str(failed)
    p = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "Invalid value for environment variable" not in p.stderr
    with tarfile.open(download / "failed_reports.tar.gz") as t:
        assert b"run_status=failed" in t.extractfile("failed/export_status.txt").read()


def test_frozen_readout_pipeline_and_research_export(tmp_path, monkeypatch):
    """Actual 845 closed-form selections on synthetic data; no old weights or GPU."""
    (tmp_path / "cache").mkdir()
    atomic_json({"synthetic": True}, tmp_path / "model_selection_lock.json")
    for split in ("train", "val"):
        atomic_json({"synthetic": split}, tmp_path / f"cache/{split}_states_index.json")
    monkeypatch.setattr(ev, "check_selection", lambda out: {})
    sizes = {"train": 24, "val": 12, "test": 120, "cross_research": 120}
    xs = {s: inputs(n) for s, n in sizes.items()}
    monkeypatch.setattr(data, "cache", lambda meta, out, split: (xs[split], rows(sizes[split])))
    banks = {}
    rng = np.random.default_rng(12)
    for split, n in sizes.items():
        bank = {name: rng.normal(size=(n, 16)) for name in ev.variants()}
        for name in ev.variants():
            for metric in (
                "price",
                "near",
                "mid",
                "far",
                "activity",
                "structure",
                "near_activity",
                "mid_activity",
                "far_activity",
            ):
                bank[f"error/{name}/{metric}"] = np.ones((n, 128))
        banks[split] = bank
    monkeypatch.setattr(ev, "extract", lambda meta, out, split, device: banks[split])
    paired = ev.paired
    monkeypatch.setattr(
        ev, "paired", lambda a, b, rs, quantile=None: paired(a, b, rs, quantile, draws=30)
    )
    ev.fit(meta(), tmp_path, "cpu")
    fitted = json.loads((tmp_path / "readout_fit.json").read_text())
    assert fitted["fits"] == 845 and len(fitted["heads"]) == 13
    ev.check_readouts(tmp_path)
    with patch.object(ev.up, "fit_heads", side_effect=AssertionError("must not refit")):
        ev.fit(meta(), tmp_path, "cpu")
    ev.evaluate(meta(), tmp_path, "cpu")
    decision = json.loads((tmp_path / "decision.json").read_text())
    assert not decision["limited_candidate"] and not decision["promoted"]
    with np.load(tmp_path / "test_predictions.npz") as report:
        assert report["targets"].shape == (120, 13)
        assert report["error/D_s42/near"].shape == (120, 128)
    (tmp_path / "readout_weights.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ev.check_readouts(tmp_path)


def test_reconstruction_gain_cannot_replace_utility_or_support(monkeypatch):
    paired = ev.paired
    monkeypatch.setattr(
        ev, "paired", lambda a, b, rs, quantile=None: paired(a, b, rs, quantile, draws=30)
    )
    names = ev.variants() + ["raw3584", "pca768", "current28"]
    scores = {n: {"targets": [{"r2": 0.95}] * 13} for n in names}
    errors = {
        n: {m: np.full(120, 0.01) for m in list(ev.up.NAMES) + list(ev.up.GROUPS)} for n in names
    }
    bank = {
        f"error/{n}/{metric}": np.full((120, 128), 0.2 if n.startswith("D") else 1.0)
        for n in names
        for metric in (
            "near",
            "mid",
            "far",
            "activity",
            "near_activity",
            "mid_activity",
            "far_activity",
            "price",
        )
    }
    result = ev.decisions(bank, scores, errors, np.ones((120, 13), bool), rows())
    assert all(not d["research_progress"] and not d["limited_candidate"] for d in result.values())
    for seed in (42, 43):
        for metric in ev.up.GROUPS:
            errors[f"D_s{seed}"][metric] *= 0.8
    result = ev.decisions(bank, scores, errors, np.ones((120, 13), bool), rows())
    assert all(d["research_progress"] and d["limited_candidate"] for d in result.values())
    one_month = [dict(r, month="2020-01") for r in rows()]
    result = ev.decisions(bank, scores, errors, np.ones((120, 13), bool), one_month)
    assert all(not d["research_progress"] for d in result.values())


def test_profile_zero_updates_and_memory_disk_time_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "EPOCHS", 1)
    with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("profiling updates")):
        report = run.resource_preflight(meta(), tmp_path, inputs(2), inputs(1), 1e9, "cpu")
    assert report["initial_loss_gradient_norms"]["encoder"] > 0
    assert set(report["seconds_per_micro"]) == {"train5", "val9", "research128"}
    with pytest.raises(ValueError, match="exceeds"):
        run.resource_preflight(meta(), tmp_path, inputs(2), inputs(1), 0, "cpu")
    usage = run.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda out: usage._replace(free=1))
    with pytest.raises(ValueError, match="GiB"):
        run.resource_preflight(meta(), tmp_path, inputs(2), inputs(1), 1e9, "cpu")


def test_full128_extraction_resumes_without_losing_examples(tmp_path, monkeypatch):
    atomic_json({"synthetic": True}, tmp_path / "model_selection_lock.json")
    monkeypatch.setattr(ev, "check_selection", lambda out: {})
    monkeypatch.setattr(ev, "check_readouts", lambda out: {})
    monkeypatch.setattr(ev, "variants", lambda: ["A_s42", "B_s42"])
    monkeypatch.setattr(data, "cache", lambda m, out, split: (inputs(2), rows(2)))
    loaded = []

    def load(m, out, name, device):
        loaded.append(name)
        if name == "B_s42" and loaded.count(name) == 1:
            raise RuntimeError("synthetic interrupted extraction")
        return tuple(model.eval().requires_grad_(False) for model in models())

    monkeypatch.setattr(ev, "load", load)
    m = meta(2) | {"source_indexes": {"test": "synthetic"}}
    with pytest.raises(RuntimeError, match="interrupted"):
        ev.extract(m, tmp_path, "test", "cpu")
    bank = ev.extract(m, tmp_path, "test", "cpu")
    assert loaded.count("A_s42") == 1 and loaded.count("B_s42") == 2
    assert bank["error/A_s42/near"].shape == (2, 128)
    with np.load(tmp_path / "test_examples.npz") as examples:
        assert {"A_s42/prediction", "B_s42/prediction", "target", "mask"} == set(examples)
    monkeypatch.setattr(ev, "load", lambda *args: pytest.fail("cached extraction reloaded model"))
    again = ev.extract(m, tmp_path, "test", "cpu")
    np.testing.assert_array_equal(again["A_s42"], bank["A_s42"])
