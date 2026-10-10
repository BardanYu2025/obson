"""Synthetic CPU inference, independent statistics and failure paths; no real weights."""

import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from obson.babel import overlap_content_metrics as metrics
from obson.babel import overlap_content_run as run
from obson.babel.ae_extend import atomic_json
from obson.babel.history_price_truth import RawCloseWindows

STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.3, 0.4],
}


@pytest.fixture(autouse=True)
def runtime():
    with run.prior.r0.replay_runtime("context"):
        torch.set_num_threads(1)
        yield


def inputs(n=9):
    x = np.random.default_rng(7).normal(0, 0.02, (n, 255, 28)).astype("float32")
    x[..., 23:] = 1
    return x


def synthetic(n=9):
    x = inputs(n)
    ps = torch.tensor(metrics.POSITIONS).expand(n, -1)
    y, mask = run.prior.core.targets(torch.from_numpy(x), ps, STATS)
    raw = 100 * np.exp(np.cumsum(np.sinh(x[..., :2].astype(float)).sum(-1) / 100, axis=1))
    closes = np.stack([raw[:, p - 1 : p + 127] for p in metrics.POSITIONS], 1)
    return x, y.numpy(), mask.numpy(), closes


def rows(n=9):
    return [
        {
            "key": "X/15/EX.X1",
            "month": f"2026-{i % 12 + 1:02}",
            "period": 15,
            "end": "2026-08-04 00:00:00",
            "row": 254,
            "input_start": "2026-08-01 08:30:00",
        }
        for i in range(n)
    ]


def test_perfect_physical_prediction_zero_error_but_nonzero_flat_baseline():
    _, y, m, c = synthetic()
    bank = metrics.paired_rows(y, y, m, c, STATS)
    for d in (1, 16, 64):
        for field in ("shape_a_mse", "shape_b_mse", "entered_b_mse", "leaving_a_mse"):
            assert np.max(bank[f"shift{d}/price/{field}"]) < 1e-17
        assert np.all(bank[f"shift{d}/price/flat_shape_baseline_mse"] > 0)
    assert np.array_equal(bank["shift64/body/common_a_mse"], np.zeros(len(y)))


def test_missing_activity_is_neither_zero_error_nor_a_valid_sample():
    _, y, m, c = synthetic()
    m[..., 4] = False
    p = y.copy()
    p[..., 4] += 100
    bank = metrics.paired_rows(p, y, m, c, STATS)
    key = "shift16/oi_change_percent_asinh/entered_b"
    assert np.isnan(bank[key + "_mse"]).all()
    assert not bank[key + "_support"].any()
    report = metrics.summarize(bank, rows())
    v = report["distributions"][key + "_mse"]
    assert v["parents"] == 0 and v["mean"] is None
    for ci in report["paired_mean_intervals"][key + "_minus_mean"].values():
        assert not ci["supported"] and ci["mean"] is None


def test_common_activity_misalignment_is_rejected():
    _, y, m, c = synthetic()
    bad = y.copy()
    # A p127 age1 is B p128 age2; perturb only B.
    bad[0, 3, 1, 2] += 0.1
    with pytest.raises(ValueError, match="Common non-price"):
        metrics.paired_rows(y, bad, m, c, STATS)
    with pytest.raises(ValueError, match="queries"):
        metrics.paired_rows(y[:, :3], y, m, c, STATS)


def test_cluster_bootstrap_matches_independent_whole_parent_resampling():
    rs = rows(120)
    x = np.column_stack([np.arange(120), -np.arange(120)]).astype(float)
    x[::7, 1] = np.nan
    draws = 37
    result = metrics.intervals(x, rs, draws)
    for kind, labels in (
        ("month", [r["month"] for r in rs]),
        ("contract_period_month", [r["key"] + "/" + r["month"] for r in rs]),
    ):
        groups = sorted(set(labels))
        samples = np.random.default_rng(20261010).integers(0, len(groups), (draws, len(groups)))
        values = []
        for sample in samples:
            ix = [i for group in sample for i, label in enumerate(labels) if label == groups[group]]
            values.append(np.nanmean(x[ix], axis=0))
        expected = np.quantile(values, [0.025, 0.975], axis=0)
        for j, v in enumerate(result[kind]):
            np.testing.assert_allclose([v["low"], v["high"]], expected[:, j])
            assert v["supported"]


def test_raw_row_identity_and_price_coordinate_use_original_csv(tmp_path):
    folder = tmp_path / "X"
    folder.mkdir()
    path = folder / "EX.X1_15m.csv"
    ts = [datetime(2026, 8, 1) + timedelta(minutes=15 * i) for i in range(260)]
    changes = np.sin(np.arange(260) / 11) * 0.0001
    prices = 100 * np.exp(np.cumsum(changes))
    path.write_text(
        "datetime,close\n" + "".join(f"{t},{p:.17g}\n" for t, p in zip(ts, prices, strict=True))
    )
    r = {**rows(1)[0], "end": str(ts[254]), "input_start": str(ts[0])}
    loader = RawCloseWindows(tmp_path, {r["key"]: run.sha256(path)})
    x = np.zeros((1, 255, 28), dtype="float32")
    x[0, :, 0] = np.arcsinh(changes[:255] * 100)
    raw, ends, check = run.prepare_raw(loader, [r], x, STATS)
    assert check["passed"]
    np.testing.assert_allclose(raw["raw_close"][0, 3], prices[127:255])
    assert [e["row"] for e in ends[0]] == [190, 238, 253, 254]
    with pytest.raises(ValueError, match="full255"):
        run.prepare_raw(loader, [{**r, "input_start": str(ts[1])}], x, STATS)
    raw2, _, check = run.prepare_raw(loader, [r], x + np.eye(28, dtype="float32")[0], STATS)
    assert not check["passed"] and np.array_equal(raw2["raw_close"], raw["raw_close"])


def setup_extract(tmp_path, monkeypatch, n=9):
    root, out = tmp_path / "source", tmp_path / "output"
    root.mkdir()
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    e, q = run.prior.core.make_models(
        "rolling", 42, "cpu", {"width": 32, "heads": 4, "layers": 4, "ff": 64}
    )
    e.eval().requires_grad_(False)
    q.eval().requires_grad_(False)
    x = inputs(n)
    z, p = run.forward(e, q, x[:, -128:], "cpu")
    np.savez(root / "test_D_s42.npz", embedding=z, query=p)
    monkeypatch.setattr(run.prior.ev, "load", lambda *args: (e, q))

    def resource(out, model, *args):
        atomic_json({"passed": True, "scope": "synthetic_test"}, out / f"resource_{model}.json")

    monkeypatch.setattr(run, "resource_check", resource)
    args = (
        root,
        root,
        out,
        {"statistics": STATS},
        "test",
        "D_s42",
        x,
        "cpu",
        time.monotonic() + 7200,
    )
    return args, e, q


def test_real_extraction_flow_commit_resume_tail_batch_and_counts(tmp_path, monkeypatch):
    args, e, q = setup_extract(tmp_path, monkeypatch)
    bank = run.extract(*args)
    assert bank["query"].shape == (9, 4, 127, 7)
    ledger = run.ForwardLedger(args[2]).value
    assert ledger["main"] == 36 and ledger["single"] == 2
    assert not any(p.grad is not None for model in (e, q) for p in model.parameters())
    monkeypatch.setattr(run, "forward", lambda *args: pytest.fail("Repeated committed forward"))
    np.testing.assert_array_equal(run.extract(*args)["query"], bank["query"])
    # Lost aggregation receipt can be rebuilt using committed chunks, no new inference.
    (args[2] / "test_D_s42.json").unlink()
    np.testing.assert_array_equal(run.extract(*args)["query"], bank["query"])
    path = next((args[2] / "chunks").glob("*.npz"))
    path.write_bytes(b"corrupt")
    (args[2] / "test_D_s42.json").unlink()
    with pytest.raises(ValueError, match="Committed"):
        run.extract(*args)


def test_future_feature_changes_cannot_change_earlier_endpoints(tmp_path, monkeypatch):
    args, e, q = setup_extract(tmp_path, monkeypatch, 2)
    x = args[6]
    first = run.extract(*args)["query"]
    changed = x.copy()
    changed[:, 191:] += 10
    z, p = run.forward(e, q, changed[:, 63:191], "cpu")
    np.testing.assert_allclose(p, first[:, 0], atol=1e-5, rtol=1e-5)


def test_failure_saves_numeric_evidence_and_forbids_uncommitted_repeat(tmp_path, monkeypatch):
    args, e, q = setup_extract(tmp_path, monkeypatch, 2)
    file = args[0] / "test_D_s42.npz"
    with np.load(file) as f:
        bank = dict(f)
    bank["query"] += 1
    np.savez(file, **bank)
    with pytest.raises(ValueError, match="mismatched"):
        run.extract(*args)
    diag = next((args[2] / "chunks").glob("*_qualification.json"))
    assert not json.loads(diag.read_text())["checks"]["source_query"]["passed"]
    monkeypatch.setattr(run, "forward", lambda *args: pytest.fail("Uncommitted retry"))
    with pytest.raises(ValueError, match="Uncommitted inference"):
        run.extract(*args)
    assert not (args[2] / "test_D_s42.json").exists()


def test_forward_mutating_buffer_is_not_committed(tmp_path, monkeypatch):
    args, e, q = setup_extract(tmp_path, monkeypatch, 2)
    original = run.forward

    def mutate(*a):
        z, p = original(*a)
        q.state_mean.add_(0.01)
        return z, p

    monkeypatch.setattr(run, "forward", mutate)
    with pytest.raises(ValueError, match="mismatched"):
        run.extract(*args)
    assert not list((args[2] / "chunks").glob("*.npz"))


def test_resource_failure_prevents_later_resume_forwards(tmp_path, monkeypatch):
    args, e, q = setup_extract(tmp_path, monkeypatch, 2)
    atomic_json({"passed": False}, args[2] / "resource_D_s42.json")
    monkeypatch.setattr(run, "forward", lambda *args: pytest.fail("resource rejected"))
    with pytest.raises(ValueError, match="resource preflight"):
        run.extract(*args)


def test_forward_budget_caps_and_no_real_cpu_entrypoint(tmp_path):
    ledger = run.ForwardLedger(tmp_path)
    ledger.reserve("max", run.MAIN_CAP, run.SINGLE_CAP)
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve("over", 1, 0)
    with (
        patch.object(run.torch.cuda, "is_available", return_value=False),
        pytest.raises(ValueError, match="AutoDL CUDA"),
    ):
        run.execute(tmp_path, tmp_path, tmp_path, 7200)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed"


def test_timeout_supervisor_preserves_stop_receipt_and_source(tmp_path):
    source, raw, out = tmp_path / "source", tmp_path / "raw", tmp_path / "out"
    source.mkdir()
    raw.mkdir()
    child = MagicMock()
    child.wait.side_effect = [subprocess.TimeoutExpired("test", 1), 0]
    with patch.object(run.subprocess, "Popen", return_value=child) as popen:
        assert run.supervise(source, raw, out) == 124
    assert popen.call_args.args[0][2] == run.MODULE
    child.terminate.assert_called_once()
    assert json.loads((out / "status.json").read_text())["status"] == "timeout"
    assert (out / "attempt_001.json").exists()
    with pytest.raises(ValueError, match="separate"):
        run.supervise(source, raw, raw / "child")


def test_export_full_queries_success_failed_stopped_and_bad_threads(tmp_path):
    out = tmp_path / "result"
    out.mkdir()
    (out / "chunks").mkdir()
    np.savez(out / "chunks/pair.npz", query=[1])
    (out / "weight.pt").write_text("no weights exported")
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        BABEL_OVERLAP_CONTENT_RUN=str(out),
        BABEL_OVERLAP_CONTENT_SOURCE=str(tmp_path / "missing"),
        BABEL_RAW_ROOT=str(tmp_path / "raw"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        OMP_NUM_THREADS="invalid",
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_overlap_content768_autodl.sh"
    for status in ("f03_diagnostic_complete_requires_review", "failed", "stopped"):
        atomic_json({"status": status}, out / "status.json")
        p = subprocess.run(["bash", str(script), "export"], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        with tarfile.open(tmp_path / "download/result_reports.tar.gz") as t:
            assert "result/chunks/pair.npz" in t.getnames()
            assert "result/weight.pt" not in t.getnames()
            assert (
                f"run_status={status}".encode() in t.extractfile("result/export_status.txt").read()
            )
    env["BABEL_OVERLAP_CONTENT_RUN"] = str(tmp_path / "failed")
    p = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "Invalid value for environment variable" not in p.stderr
    assert (tmp_path / "download/failed_reports.tar.gz").exists()


def test_protocol_inventory_source_pin_and_fixed_card(tmp_path):
    p = run.protocol()
    assert p["main_endpoints_max"] == len(p["models"]) * sum(p["splits"].values()) * len(
        p["positions"]
    )
    assert p["single_endpoints_max"] == len(p["models"]) * len(p["splits"]) * 2
    assert p["optimizer_updates"] == p["readout_fits"] == 0
    assert len(run.inventory()["files"]) == 349
    atomic_json({}, tmp_path / "completion.json")
    with pytest.raises(ValueError, match="reviewed"):
        run.verify_source(tmp_path, tmp_path / "output")


def test_execute_assembly_failure_then_resume_without_repeating_committed_inference(
    tmp_path, monkeypatch
):
    """Exercise actual orchestration/scoring/completion; mock only source/CUDA/inference."""
    root, out, raw_root = tmp_path / "source", tmp_path / "out", tmp_path / "raw"
    root.mkdir()
    out.mkdir()
    raw_root.mkdir()
    x, y, mask, closes = synthetic(2)
    rs = rows(2)
    for split in ("val", "test", "cross_research"):
        atomic_json(rs, root / f"{split}_rows.json")
    monkeypatch.setattr(run.prior, "SPLITS", {"val": 2, "test": 2, "cross_research": 2})
    monkeypatch.setattr(run, "MAIN_CAP", 96)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run.torch.cuda, "get_device_name", lambda: "synthetic-no-CUDA")
    monkeypatch.setattr(run, "verify_source", lambda *a: (root, {"statistics": STATS}))
    monkeypatch.setattr(run.data, "cache", lambda *a: (x, rs))
    monkeypatch.setattr(
        run,
        "prepare_raw",
        lambda *a: ({"raw_close": closes, "source_target": y, "mask": mask}, [], {"passed": True}),
    )
    calls = []

    def extraction(root, coverage, out, meta, split, model, *args):
        ledger = run.ForwardLedger(out)
        key = split + "/" + model
        if key not in ledger.value["chunks"]:
            ledger.reserve(key, 8, 2)
            calls.append(key)
        return {"query": y.copy()}

    monkeypatch.setattr(run, "extract", extraction)
    original = metrics.summarize
    monkeypatch.setattr(
        metrics,
        "summarize",
        lambda *a: (_ for _ in ()).throw(ValueError("CPU scoring interruption")),
    )
    with pytest.raises(ValueError, match="CPU scoring"):
        run.execute(root, raw_root, out, 7200)
    assert not (out / "completion.json").exists()
    assert len(calls) == 1
    monkeypatch.setattr(metrics, "summarize", original)
    assert run.execute(root, raw_root, out, 7200) == 0
    assert len(calls) == 12 and len(set(calls)) == 12
    done = run.check_complete(out)
    assert done["status"] == "f03_diagnostic_complete_requires_review"
    assert not done["promoted"]
    monkeypatch.setattr(run, "extract", lambda *a: pytest.fail("Completed extraction repeated"))
    assert run.execute(root, raw_root, out, 7200) == 0
