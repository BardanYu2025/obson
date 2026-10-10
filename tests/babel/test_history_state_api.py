"""Only synthetic CPU model tests and source-report arithmetic; no real weights."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from obson.babel import history_state_api as api
from obson.babel import history_state_delivery as run
from obson.babel.ae_extend import atomic_json

STATS = {
    "x_mean": np.linspace(-0.1, 0.1, 28).tolist(),
    "x_scale": np.linspace(0.7, 1.3, 28).tolist(),
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.07, 0.2, 0.4, 0.8],
}
CONFIG = {"width": 32, "heads": 4, "layers": 4, "ff": 64}


@pytest.fixture(autouse=True)
def runtime():
    with api.prior.r0.replay_runtime("context"):
        torch.set_num_threads(1)
        yield


def inputs(n=4):
    raw = np.random.default_rng(8).normal(0, 0.02, (n, 255, 28))
    raw[..., 23:] = 1
    return ((raw - STATS["x_mean"]) / STATS["x_scale"]).astype("float32")


def reader(model="D_s42"):
    e, q = api.prior.core.make_models("rolling", 42, "cpu", CONFIG)
    return api.HistoryStateReader(e, q, STATS, {"model": model, "width": 32}, "cpu")


def rows(n=4):
    return [
        {
            "key": "X/15/ABC",
            "period": 15,
            "end": f"2026-{i % 12 + 1:02}-01 09:00:00",
            "month": f"2026-{i % 12 + 1:02}",
            "row": i,
        }
        for i in range(n)
    ]


def read(r, x, audit=False):
    return r.read_prepared(
        x,
        run.endpoint_metadata(rows(len(x)), "synthetic_full_history"),
        input_signature=r.input_signature,
        audit_observed=audit,
    )


def test_pure_estimate_and_optional_observed_audit_do_not_mix():
    r = reader()
    x = inputs()[:, -128:]
    original = read(r, x)
    audited = read(r, x, True)
    assert "observed_audit" not in original
    for key in ("embedding", "query", "descriptors", "direction", "relative_log_close"):
        assert np.array_equal(original[key], audited[key])
    assert audited["observed_audit"]["source"] == "observed_history_not_model_output"
    assert (
        not original["metadata"]["confidence_calibrated"]
        and not original["metadata"]["quality_assured"]
    )
    restored = r.decode_state(original["embedding"], state_signature=r.state_signature)
    assert np.array_equal(restored["descriptors"], original["descriptors"])
    assert np.all(original["relative_log_close"][:, -1] == 0)


def test_reject_wrong_state_and_preparation_identity_and_raw_dimensions():
    a, b = reader(), reader("macro_s42")
    x = inputs()[:, -128:]
    state = read(a, x)["embedding"]
    with pytest.raises(ValueError, match="different model"):
        b.decode_state(state, state_signature=a.state_signature)
    ends = run.endpoint_metadata(rows(), "synthetic")
    with pytest.raises(ValueError, match="normalization"):
        a.read_prepared(x, ends, input_signature="other")
    for wrong in (x.astype("float64"), inputs(), x[..., :4]):
        with pytest.raises(ValueError, match="float32"):
            a.read_prepared(wrong, ends, input_signature=a.input_signature)
    ends[0]["closed"] = False
    with pytest.raises(ValueError, match="closed"):
        a.read_prepared(x, ends, input_signature=a.input_signature)
    a.encoder.train()
    with pytest.raises(ValueError, match="frozen"):
        read(a, x)


def test_contract_period_and_invalid_values_rejected():
    r = reader()
    x = inputs()[:, -128:]
    for changes in ({"period": 60}, {"feature_history_origin": ""}, {"key": "continuous"}):
        ends = run.endpoint_metadata(rows(), "synthetic")
        ends[0].update(changes)
        with pytest.raises(ValueError):
            r.read_prepared(x, ends, input_signature=r.input_signature)
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        read(r, bad)


def test_independent_readout_units_thresholds_and_no_log_floor_misreporting():
    path = np.stack([np.zeros(128), np.arange(128) * 0.00005, -np.arange(128) * 0.0002])
    v = api.path_outputs(path)
    np.testing.assert_allclose(v["rms_bps"], [[0, 0], [0.5, 0.5], [2, 2]])
    np.testing.assert_array_equal(v["direction"], [[1, 1], [2, 2], [0, 0]])
    assert np.array_equal(v["estimated_low_motion"], [[True, True], [True, True], [False, False]])
    assert v["rms_log_return"][0, 0] == 0 and v["descriptors"][0, 2] == np.log(1e-8)
    np.testing.assert_allclose(
        v["descriptors"], run.independent_descriptors(path), atol=1e-9, rtol=1e-8
    )


def test_observed_flat_failure_is_retained_and_self_low_flag_can_miss_it():
    n = 4
    raw = np.zeros((n, 128, 28))
    x = ((raw - STATS["x_mean"]) / STATS["x_scale"]).astype("float32")
    path = np.tile(np.arange(128) * 0.001, (n, 1))
    audit = api.observed_audit(path, x, STATS)
    assert np.all(audit["stratum"] == 0)
    report = api.boundary_report(path, audit, rows())
    for h in ("16", "64"):
        assert report[h]["strata"]["floor"]["support"] == n
        assert not report[h]["strata"]["floor"]["evidence_sufficient"]
        assert report[h]["estimated_low_motion_confusion"] == [[0, 0], [n, 0]]
        assert not report[h]["quality_certified"]
        assert report[h]["strata"]["floor"]["physical_rms_mae_bps"] > 9.9


def test_actual_extract_resume_and_replay_failure(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    r = reader()
    x = inputs(10)
    rs = rows(10)
    target = read(r, x[:, -128:], True)
    model = "D_s42"
    name = "test_" + model
    np.savez(
        root / (name + ".npz"),
        query=target["query"],
        path=target["relative_log_close"],
        decoded=target["descriptors"],
    )
    np.savez(
        root / "test_predictions.npz",
        targets=target["observed_audit"]["true_descriptors"],
        true_path=target["observed_audit"]["true_path"],
    )
    monkeypatch.setattr(api.prior.ev, "load", lambda *args: (r.encoder, r.query))
    args = (
        root,
        root,
        out,
        {"statistics": STATS, "source_indexes": {"test": "synthetic"}},
        "test",
        model,
        x,
        rs,
        target["embedding"],
        "cpu",
    )
    report = run.extract(*args)
    assert report["interface_numerical_qualification"] and not report["quality_assured"]
    assert report["rows"] == 10
    monkeypatch.setattr(
        api.prior.ev, "load", lambda *args: pytest.fail("repeated committed forward")
    )
    assert run.extract(*args) == report
    (out / (name + ".npz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Committed"):
        run.extract(*args)


def test_failure_saves_tensor_diagnostic_without_committing_result(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    atomic_json({}, out / "manifest.json")
    r = reader()
    x = inputs()
    res = read(r, x[:, -128:], True)
    np.savez(
        root / "test_D_s42.npz",
        query=res["query"] + 1,
        path=res["relative_log_close"],
        decoded=res["descriptors"],
    )
    np.savez(
        root / "test_predictions.npz",
        targets=res["observed_audit"]["true_descriptors"],
        true_path=res["observed_audit"]["true_path"],
    )
    monkeypatch.setattr(api.prior.ev, "load", lambda *args: (r.encoder, r.query))
    with pytest.raises(ValueError, match="mismatched"):
        run.extract(
            root,
            root,
            out,
            {"statistics": STATS, "source_indexes": {"test": "synthetic"}},
            "test",
            "D_s42",
            x,
            rows(),
            res["embedding"],
            "cpu",
        )
    check = json.loads((out / "test_D_s42_qualification.json").read_text())
    assert not check["checks"]["source/query"]["passed"]
    assert not (out / "test_D_s42.json").exists()


def test_source_pinning_and_real_cpu_execution_refused(tmp_path):
    atomic_json({}, tmp_path / "completion.json")
    with pytest.raises(ValueError, match="reviewed"):
        api.verify_source(tmp_path)
    with pytest.raises(ValueError, match="CUDA"):
        api.open_reader(tmp_path, "D_s42", "cpu")
    with pytest.raises(ValueError, match="Explicit"):
        api.open_reader(tmp_path, "best", "cuda")


def test_bounded_supervisor_and_source_output_separation(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "out"
    child = MagicMock()
    child.wait.side_effect = [subprocess.TimeoutExpired("test", 1), 0]
    with patch.object(run.subprocess, "Popen", return_value=child) as popen:
        assert run.supervise(root, out) == 124
    child.terminate.assert_called_once()
    assert popen.call_args.args[0][2] == "obson.babel.history_state_delivery"
    assert json.loads((out / "status.json").read_text())["status"] == "timeout"
    assert (out / "attempt_001.json").exists()
    with pytest.raises(ValueError, match="separate"):
        run.supervise(root, root / "child")
    budget = json.loads((out / "budget.json").read_text())
    budget["used_seconds"] = 7200
    atomic_json(budget, out / "budget.json")
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(root, out)


def test_export_success_failure_and_invalid_threads(tmp_path):
    out = tmp_path / "result"
    out.mkdir()
    download = tmp_path / "download"
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        BABEL_HISTORY_STATE_RUN=str(out),
        BABEL_HISTORY_STATE_SOURCE=str(tmp_path / "missing"),
        BABEL_DOWNLOAD_DIR=str(download),
        OMP_NUM_THREADS="bad",
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_history_state768_autodl.sh"
    atomic_json({"status": "complete"}, out / "status.json")
    np.savez(out / "outputs.npz", a=[1])
    (out / "weights.pt").write_text("not exported")
    p = subprocess.run(["bash", str(script), "export"], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    with tarfile.open(download / "result_reports.tar.gz") as t:
        assert "result/outputs.npz" in t.getnames() and "result/weights.pt" not in t.getnames()
    env["BABEL_HISTORY_STATE_RUN"] = str(tmp_path / "failed")
    p = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "Invalid value for environment variable" not in p.stderr
    with tarfile.open(download / "failed_reports.tar.gz") as t:
        assert b"run_status=failed" in t.extractfile("failed/export_status.txt").read()


def test_locked_budget_and_no_model_selection():
    p = run.protocol()
    assert p["planned_unique_endpoints"] == len(p["models"]) * sum(p["splits"].values()) == 9604
    assert p["single_example_checks"] == len(p["models"]) * len(p["splits"]) * 2 == 24
    assert p["optimizer_updates"] == p["readout_fits"] == 0
    assert json.loads(json.dumps(p)) == p
