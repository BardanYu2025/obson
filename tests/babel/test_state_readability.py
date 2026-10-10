"""Synthetic CPU and report-arithmetic checks only; no pretrained model execution."""

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

from obson.babel import state_readability as study
from obson.babel import state_readability_run as run
from obson.babel.ae_extend import atomic_json

STATS = {
    "x_mean": np.linspace(-0.1, 0.1, 28).tolist(),
    "x_scale": np.linspace(0.7, 1.3, 28).tolist(),
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.07, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def runtime():
    with run.r0.replay_runtime("context"):
        torch.set_num_threads(1)
        yield


def inputs(n):
    raw = np.random.default_rng(90).normal(0, 0.02, (n, 255, 28))
    raw[..., 23:] = 1
    return ((raw - STATS["x_mean"]) / STATS["x_scale"]).astype("float32")


def rows(n=120):
    return [
        {
            "key": f"X{i % 6}/15/C{i % 6}",
            "month": f"2025-{i % 12 + 1:02}",
            "end": f"t{i:06}",
            "row": i,
        }
        for i in range(n)
    ]


def test_age_orientation_scale_units_and_same_physical_targets():
    x = inputs(8)
    y, _ = run.core.targets(torch.tensor(x), torch.full((8, 1), 128), STATS)
    path = study.close_path(y[:, 0].numpy(), STATS)
    raw = run.ev.up.restore_raw(x[:, -128:], STATS)
    truth, _ = run.ev.up.targets(raw)
    np.testing.assert_allclose(path, study.true_path(raw), atol=1e-9, rtol=2e-6)
    np.testing.assert_allclose(study.descriptors(path), truth[:, :6], atol=1e-6, rtol=2e-5)
    np.testing.assert_allclose(study.descriptors(study.true_path(raw)), truth[:, :6], atol=1e-12)
    assert np.array_equal(path[:, -1], np.zeros(8))
    other = y[:, 0].numpy().copy()
    other[:, :, 1:] += 100
    assert np.array_equal(study.close_path(other, STATS), path)


def test_flat_ramp_translation_and_amplitude_effect():
    flat = study.descriptors(np.zeros((1, 128)))[0]
    np.testing.assert_allclose(flat, [0, 0, np.log(1e-8)] * 2)
    path = np.arange(128)[None] * 0.001
    base = study.descriptors(path)
    np.testing.assert_allclose(base[0], [1, 1, np.log(0.001)] * 2)
    np.testing.assert_allclose(study.descriptors(path + 10), base, atol=1e-10)
    scaled = study.descriptors(path * 2)
    np.testing.assert_allclose(scaled[:, [0, 1, 3, 4]], base[:, [0, 1, 3, 4]])
    np.testing.assert_allclose(scaled[:, [2, 5]] - base[:, [2, 5]], np.log(2))
    negative = study.descriptors(-path)
    np.testing.assert_allclose(negative[:, [0, 3]], -base[:, [0, 3]])


def test_horizon_uses_internal_returns_and_includes_current():
    path = np.zeros((2, 128))
    path[0, -1] = 0.01
    path[1, -17] = 0.04  # outside the16-close descriptor
    value = study.descriptors(path)
    assert value[0, 0] == 1
    assert value[0, 2] == pytest.approx(np.log(0.01 / np.sqrt(15)))
    assert value[1, 2] == pytest.approx(np.log(1e-8))
    assert value[1, 5] > np.log(1e-8)


def test_mismatch_never_crosses_contract_or_self_and_masks_singletons():
    rs = rows(120) + [{"key": "Y/60/ONLY", "end": "t0", "row": 0}]
    ids = study.mismatch_indices(rs)
    assert np.array_equal(ids, study.mismatch_indices(rs)) and ids[-1] == -1
    for i, j in enumerate(ids[:-1]):
        assert j != i and rs[j]["key"] == rs[i]["key"]
    assert set(ids[:-1]) == set(range(120))


def test_decoder_route_gates_do_not_promote_or_replace_old_exit():
    rng = np.random.default_rng(20)
    truth = rng.normal(size=(120, 6))
    stats = {"mean": [0.0] * 13, "scale": [1.0] * 13}
    scores, errors, decoded = {}, {}, {}
    for model in run.ev.variants():
        for route, noise in (("decoded", 0.01), ("direct", 0.2)):
            pred = truth + noise
            scores[model + "/" + route], errors[model + "/" + route] = study.measure(
                pred, truth, stats
            )
            if route == "decoded":
                decoded[model] = pred
    _, errors["train_mean"] = study.measure(np.zeros_like(truth), truth, stats)
    result = study.decide(
        scores,
        errors,
        rows(),
        study.mismatch_indices(rows()),
        {"target": stats, "decoded": decoded},
        truth,
    )
    for value in result["seeds"].values():
        assert value["decoder_route_advantage"] and value["recoverability_qualification"]
        assert not value["original_exit_qualification_changed"] and not value["promoted"]
    short = run.ev.gate(np.zeros(10), np.ones(10), rows(10), 0.95)
    assert not short["passed"]


def test_tiny_frozen_model_execution_and_no_raw_decoder_bypass(tmp_path):
    config = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
    e, q = run.core.make_models("rolling", 42, "cpu", config)
    e.eval().requires_grad_(False)
    q.eval().requires_grad_(False)
    x = inputs(4)
    z = run.core.read_states(e, torch.tensor(x), torch.full((4, 1), 128))[:, 0].detach().numpy()
    pred, errors, checks = run.decode(e, q, x, z, STATS, "cpu")
    assert checks["sampled_encoder_state"]["passed"] and set(errors) >= {"price", "activity"}
    mutated = x.copy()
    mutated[2:] *= 0.5
    other, _, _ = run.decode(e, q, mutated, z, STATS, "cpu")
    assert np.array_equal(other, pred)  # raw input is for checking/scoring, never passed to q
    assert all(p.grad is None for m in (e, q) for p in m.parameters())
    q.train()
    with pytest.raises(ValueError, match="frozen"):
        run.decode(e, q, x, z, STATS, "cpu")


def test_committed_model_replay_resume_and_corruption(tmp_path, monkeypatch):
    root, out = tmp_path / "source", tmp_path / "out"
    root.mkdir()
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    e, q = run.core.make_models(
        "rolling", 42, "cpu", {"width": 32, "heads": 4, "ff": 64, "layers": 4}
    )
    e.eval().requires_grad_(False)
    q.eval().requires_grad_(False)
    x = inputs(3)
    z = run.core.read_states(e, torch.tensor(x), torch.full((3, 1), 128))[:, 0].detach().numpy()
    monkeypatch.setattr(run.ev, "load", lambda *args: (e, q))
    direct = np.zeros((3, 13))
    monkeypatch.setattr(run, "direct_prediction", lambda *args: direct)
    args = (
        root,
        out,
        {"statistics": STATS},
        "val",
        "D_s42",
        x,
        z,
        {},
        {},
        {"D_s42": direct},
        "cpu",
    )
    first = run.replay_model(*args)
    monkeypatch.setattr(
        run.ev, "load", lambda *args: pytest.fail("completed frozen replay repeated")
    )
    second = run.replay_model(*args)
    assert np.array_equal(first["decoded"], second["decoded"])
    (out / "val_D_s42.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Committed"):
        run.replay_model(*args)


def test_locked_ridge_prediction_and_failed_replay_keeps_diagnostic(tmp_path, monkeypatch):
    rng = np.random.default_rng(92)
    z = rng.normal(size=(4, 3))
    w = rng.normal(size=(3, 13))
    bias = rng.normal(size=13)
    fit = {
        "heads": {"D_s42": [{"statistics": {"mean": [1.0] * 3, "scale": [2.0] * 3}}] * 13},
        "target_stats": {"mean": [3.0] * 13, "scale": [4.0] * 13},
    }
    got = run.direct_prediction(fit, {"D_s42/weights": w, "D_s42/intercepts": bias}, "D_s42", z)
    np.testing.assert_allclose(got, (((z - 1) / 2) @ w + bias) * 4 + 3)
    atomic_json({}, tmp_path / "manifest.json")
    monkeypatch.setattr(
        run.ev, "load", lambda *args: pytest.fail("bad Ridge must fail before forward")
    )
    with pytest.raises(ValueError, match="mismatched"):
        run.replay_model(
            tmp_path,
            tmp_path,
            {},
            "val",
            "D_s42",
            None,
            z,
            fit,
            {"D_s42/weights": w, "D_s42/intercepts": bias},
            {"D_s42": got + 1},
            "cpu",
        )
    assert (tmp_path / "val_D_s42_direct.json").exists()


def test_source_pin_and_output_containment(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    atomic_json({}, root / "completion.json")
    with pytest.raises(ValueError, match="reviewed"):
        run.verify_source(root, tmp_path / "out")
    with pytest.raises(ValueError, match="separate"):
        run.supervise(root, root / "child")


def test_timeout_cumulative_budget_and_request_binding(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    out = tmp_path / "out"
    child = MagicMock()
    child.wait.side_effect = [subprocess.TimeoutExpired("fake", 1), 0]
    with patch.object(run.subprocess, "Popen", return_value=child):
        assert run.supervise(root, out) == 124
    child.terminate.assert_called_once()
    budget = json.loads((out / "budget.json").read_text())
    assert budget["used_seconds"] >= 0
    assert json.loads((out / "status.json").read_text())["status"] == "timeout"
    with pytest.raises(ValueError, match="unbound"):
        run.supervise(tmp_path / "another", out)
    budget["used_seconds"] = run.SESSION_CAP
    atomic_json(budget, out / "budget.json")
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(root, out)


def test_export_success_failure_and_invalid_threads(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "result"
    out.mkdir()
    download = tmp_path / "download"
    env = dict(
        os.environ,
        PYTHON_BIN=sys.executable,
        BABEL_READABILITY_RUN=str(out),
        BABEL_READABILITY_SOURCE=str(tmp_path / "missing"),
        BABEL_DOWNLOAD_DIR=str(download),
        OMP_NUM_THREADS="bad",
    )
    atomic_json({"status": "complete"}, out / "status.json")
    np.savez(out / "predictions.npz", a=[1])
    (out / "model.pt").write_text("not exported")
    script = repo / "scripts/babel_state_readability768_autodl.sh"
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(download / "result_reports.tar.gz") as tar:
        assert (
            "result/predictions.npz" in tar.getnames() and "result/model.pt" not in tar.getnames()
        )
        assert b"readout_fits=0" in tar.extractfile("result/export_status.txt").read()
    env["BABEL_READABILITY_RUN"] = str(tmp_path / "failed")
    result = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "Invalid value for environment variable" not in result.stderr
    with tarfile.open(download / "failed_reports.tar.gz") as tar:
        assert b"run_status=failed" in tar.extractfile("failed/export_status.txt").read()


def test_protocol_budget_and_no_fit_code():
    p = run.protocol()
    assert p["query_endpoints"] == len(p["models"]) * sum(p["splits"].values()) == 24010
    assert p["optimizer_updates"] == p["readout_fits"] == 0
    assert p["encoder_replays"] == len(p["models"]) * len(p["splits"]) * 2
    assert json.loads(json.dumps(p)) == p


def test_complete_research_scoring_pipeline_synthetic(tmp_path, monkeypatch):
    """Real tiny decoder, original targets, old-report replay, all routes and final gates."""
    root, out = tmp_path / "source", tmp_path / "out"
    root.mkdir()
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    n = 120
    x = inputs(n)
    rs = rows(n)
    raw = run.ev.up.restore_raw(x[:, -128:], STATS)
    truth, mask = run.ev.up.targets(raw)
    e0, q0 = run.core.make_models(
        "rolling", 42, "cpu", {"width": 32, "heads": 4, "ff": 64, "layers": 4}
    )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = e0

        def forward(self, b):
            return torch.nn.functional.pad(self.core(b), (0, 736))

    class Query(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = q0

        def forward(self, z):
            return self.core(z[:, :32])

    e, q = Encoder().eval().requires_grad_(False), Query().eval().requires_grad_(False)
    with torch.no_grad():
        z = run.core.read_states(e, torch.tensor(x), torch.full((n, 1), 128))[:, 0].numpy()
    pred, errors, _ = run.decode(e, q, x, z, STATS, "cpu")
    target = {"mean": truth.mean(0).tolist(), "scale": np.maximum(truth.std(0), 1e-6).tolist()}
    heads = [{"statistics": {"mean": [0.0] * 768, "scale": [1.0] * 768}}] * 13
    fit = {"heads": dict.fromkeys(run.ev.variants(), heads), "target_stats": target}
    atomic_json(fit, root / "readout_fit.json")
    weights = {
        name + "/" + field: np.zeros(shape)
        for name in run.ev.variants()
        for field, shape in (("weights", (768, 13)), ("intercepts", (13,)))
    }
    np.savez(root / "readout_weights.npz", **weights)
    atomic_json(rs, root / "test_rows.json")
    atomic_json(
        {
            name: {k: {"endpoint": float(v.mean())} for k, v in errors.items()}
            for name in run.ev.variants()
        },
        root / "test_reconstruction.json",
    )
    np.savez(
        root / "test_examples.npz", **{name + "/prediction": pred[:2] for name in run.ev.variants()}
    )
    fixed = {
        name + "/predictions": np.broadcast_to(target["mean"], (n, 13)).copy()
        for name in run.ev.variants() + ["raw3584", "pca768", "current28"]
    }
    np.savez(root / "test_predictions.npz", targets=truth, mask=mask, **fixed)
    monkeypatch.setattr(run.data, "cache", lambda *args: (x, rs))
    monkeypatch.setattr(run, "cache_states", lambda *args: dict.fromkeys(run.ev.variants(), z))
    monkeypatch.setattr(run.ev, "load", lambda *args: (e, q))
    monkeypatch.setitem(run.SPLITS, "test", n)
    result = run.evaluate_split(root, out, {"statistics": STATS}, "test", "cpu")
    assert set(result["seeds"]) == {"42", "43"}
    assert all(not v["promoted"] for v in result["seeds"].values())
    with np.load(out / "test_predictions.npz") as bank:
        np.testing.assert_allclose(bank["targets"], truth[:, :6])
        np.testing.assert_allclose(
            bank["D_s42/decoded"], study.descriptors(study.close_path(pred, STATS))
        )
    assert len(json.loads((out / "forward_attempts.json").read_text())) == 10
    # Resume uses committed outputs, no second network execution.
    monkeypatch.setattr(run.ev, "load", lambda *args: pytest.fail("repeated committed forward"))
    assert run.evaluate_split(root, out, {"statistics": STATS}, "test", "cpu") == result
