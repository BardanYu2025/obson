"""Synthetic only. Neural optimizer updates are disabled throughout this module."""

import copy
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from obson.babel import state_rollout as core
from obson.babel import state_rollout_data as data
from obson.babel import state_rollout_evaluate as ev
from obson.babel import state_rollout_run as run
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def no_optimization():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def rows(n):
    return [
        {
            "key": f"c/{(15, 30, 60)[i % 3]}",
            "symbol": f"s{i % 2}",
            "period": (15, 30, 60)[i % 3],
            "row": i + 127,
            "end": f"2024-{i % 12 + 1:02d}-01",
            "month": f"2024-{i % 12 + 1:02d}",
        }
        for i in range(n)
    ]


def test_exact_future_target_and_partition_boundaries():
    c = np.ones(200) * 100
    c[:127] = 10
    c[128:] = 200
    same = np.ones(200, bool)
    y, why = core.labels(c, 127, same)
    assert why is None
    np.testing.assert_allclose(y, 100 * np.log(2))
    for j in (0, 127, 128, 159):
        mask = same.copy()
        mask[j] = False
        assert core.labels(c, 127, mask)[1] == "input_or_target_crosses_partition"
    assert core.labels(c[:159], 127, same[:159])[1] == "insufficient_history_or_future"
    c[130] = np.nan
    with pytest.raises(ValueError):
        core.labels(c, 127, same)


def test_past_and_rolling_windows_do_not_see_future():
    close = np.exp(np.arange(200) / 1000)
    before = core.past(close, 127)
    close[128:] *= 2
    np.testing.assert_array_equal(before, core.past(close, 127))
    features = np.arange(200 * 3).reshape(200, 3)
    stats = {"x_mean": np.zeros(3), "x_scale": np.ones(3)}
    actual = data.rolling_inputs(features, [127, 128], stats)
    np.testing.assert_array_equal(actual[0], features[:128])
    np.testing.assert_array_equal(actual[1], features[1:129])


def test_state_scaling_deduplicates_pairs_and_rejects_collapse():
    z = np.array([[1.0, 2], [2, 5], [4, 7]])
    s = core.state_scaler(z, [[0, 1], [1, 2], [0, 1]])
    assert s["unique_pairs"] == 2
    np.testing.assert_allclose(s["mean"], z.mean(0))
    u = core.normalize(z, s)
    np.testing.assert_allclose(
        s["increment"], np.sqrt(np.mean(np.diff(u, axis=0) ** 2, axis=0)), rtol=1e-6
    )
    with pytest.raises(ValueError):
        core.state_scaler(np.ones((3, 2)), [[0, 1]])


class IdealQuery(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("marker", torch.ones(1))

    def forward(self, z, ages):
        # Synthetic state first coordinate contains cumulative return from known origin.
        out = torch.zeros(len(z), len(ages), 7, device=z.device, dtype=z.dtype)
        out[:, :, 0] = -z[:, 0, None] / ages.to(z.dtype).sqrt()[None]
        return out


def test_price_sign_actual_age_scale_and_no_future_anchor():
    z = torch.zeros(2, 32, 4)
    z[:, :, 0] = torch.arange(1, 33) * 0.1
    p = core.read_prices(IdealQuery(), z, 1.0)
    np.testing.assert_allclose(p.numpy(), z[:, :, 0].numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        100 * np.exp(p.numpy() / 100), 100 * np.exp(z[:, :, 0].numpy() / 100), rtol=1e-6
    )
    with pytest.raises(ValueError):
        core.read_prices(IdealQuery(), z, 1.0, [0] * 32)


def test_full_gradient_through_multistep_and_identity_initialization():
    model = core.Transition("L1", 1).double()
    with torch.no_grad():
        model.delta.weight.fill_(0.2)
        model.delta.bias.zero_()
    x = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    last = model.rollout(x, 8)[:, -1].sum()
    last.backward()
    assert x.grad.item() == pytest.approx(1.2**8)
    assert model.delta.weight.grad.item() == pytest.approx(8 * 1.2**7)
    for arm in core.ARMS[:3]:
        m = core.Transition(arm, 4)
        u = torch.randn(3, 4)
        torch.testing.assert_close(m.rollout(u, 8), u[:, None].expand(-1, 8, -1))


def test_tf_and_free_losses_use_same_targets_but_different_inputs():
    states = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    m = core.Transition("N8", 1)
    loss = core.objective(m, states, torch.ones(1))
    assert loss.item() == pytest.approx(np.mean(np.arange(1, 9) ** 2))
    m.arm = "N1"
    assert core.objective(m, states, torch.ones(1)).item() == 1


def test_micro_accumulation_matches_full_gradients():
    torch.manual_seed(23)
    states = torch.randn(7, 9, 4)
    a = core.Transition("N8", 4)
    b = copy.deepcopy(a)
    core.objective(a, states, torch.ones(4)).backward()
    for chunk in states.split(3):
        (core.objective(b, chunk, torch.ones(4)) * len(chunk) / len(states)).backward()
    for x, y in zip(a.parameters(), b.parameters(), strict=False):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)


def synthetic(tmp_path, n=120, epochs=2):
    meta = {
        "epochs": epochs,
        "batch": 32,
        "seeds": [42, 43],
        "lrs": list(core.LRS),
        "arms": list(core.ARMS),
    }
    atomic_json(meta, tmp_path / "manifest.json")
    atomic_json({"micro": 13, "jobs": 2}, tmp_path / "resources.json")
    rng = np.random.default_rng(8)
    for split in data.SPLITS:
        h = 9 if split == "train" else 33
        bank = {
            "x": rng.normal(size=(n, 4, 3)).astype("float32"),
            "z42": rng.normal(size=(n * h, 4)).astype("float32"),
            "z43": rng.normal(size=(n * h, 4)).astype("float32"),
            "mapping": np.arange(n * h).reshape(n, h),
            "targets": rng.normal(size=(n, 32)) * 0.1,
            "past": np.c_[rng.uniform(0.1, 0.2, n), rng.normal(size=(n, 2)) * 0.01],
            "period": np.array([r["period"] for r in rows(n)]),
            "gaps": np.zeros((n, 32), bool),
            "close": np.ones(n) * 100,
            "history_close": np.ones((n, 128)) * 100,
        }
        data.commit_cache(tmp_path, split, bank, rows(n), {}, [])
    fit = data.transformations(tmp_path)
    tr, _ = data.cache(tmp_path, "train")
    va, _ = data.cache(tmp_path, "val")
    return meta, fit, tr, va


def test_forecast_does_not_read_future_states_targets_or_scale(tmp_path):
    _, fit, _, va = synthetic(tmp_path)
    p = run.prepared(va, fit, 42)
    model = core.Transition("N8", 4)
    a = run.forecast(model, "N8", p, fit, 42, IdealQuery(), 1.0, "cpu", 32, 7)["price"]
    q = copy.deepcopy(p)
    q["u"][q["mapping"][:, 1:].ravel()] *= 100
    q["target"][:] = np.nan
    q["scale"][:] = np.nan
    b = run.forecast(model, "N8", q, fit, 42, IdealQuery(), 1.0, "cpu", 32, 17)["price"]
    np.testing.assert_allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        run.forecast(core.Direct(4), "Z-direct", q, fit, 42, IdealQuery(), 1.0, "cpu", 32)


def test_training_resume_epoch0_and_checkpoint_binding(tmp_path, monkeypatch):
    meta, fit, tr, va = synthetic(tmp_path)
    job = run.jobs(meta)[4]  # N8 seed42
    args = (
        meta,
        tmp_path,
        job,
        run.prepared(tr, fit, 42),
        run.prepared(va, fit, 42),
        fit,
        IdealQuery(),
        1.0,
        "cpu",
    )
    original = run.atomic_save

    def interrupted(obj, path):
        original(obj, path)
        if path.name == "resume.pt" and obj["epoch"] == 1:
            raise RuntimeError("simulated interruption after committed epoch")

    with (
        patch.object(run, "atomic_save", side_effect=interrupted),
        pytest.raises(RuntimeError, match="interruption"),
    ):
        run.fit_job(*args)
    assert not (tmp_path / job["name"] / "completion.json").exists()
    result = run.fit_job(*args)
    assert result["selected_epoch"] == 0  # all optimizer steps mocked, no local learning
    assert result["updates"] == 8
    assert result["decoder_unchanged"]
    assert result == run.fit_job(*args)
    atomic_json({"micro": 14, "jobs": 2}, tmp_path / "resources.json")
    with pytest.raises(ValueError, match="binding"):
        run.fit_job(*args)


def test_fair_initialization_and_validation_only_ties():
    torch.manual_seed(42)
    a = core.Transition("N1", 4)
    torch.manual_seed(42)
    b = core.Transition("N8", 4)
    assert run.parent.ur.bb.state_signature(a) == run.parent.ur.bb.state_signature(b)
    candidates = {}
    for j in run.jobs({"seeds": [42, 43], "arms": list(core.ARMS), "lrs": list(core.LRS)}):
        candidates[j["name"]] = {"job": j, "validation": 1.0, "selected_epoch": 0}
    assert all(v.endswith("lr0") for v in run.choose(candidates).values())


def test_bootstrap_support_simultaneous_band_and_gates():
    r = rows(120)
    b = np.ones((120, 32))
    band = ev.skill_band(0.5 * b, b, r)
    assert band["contiguous_horizon"] == 32
    assert all(v > 0 for v in band["low"])
    unsupported = [dict(v, month="2024-01") for v in r]
    assert ev.skill_band(0.5 * b, b, unsupported)["contiguous_horizon"] == 0
    ref = {k: {"price": b, "incremental": b, "absolute": b} for k in ("flat", *core.ARMS)}
    result = ev.gates(
        {"price": 0.5 * b, "incremental": 0.5 * b, "absolute": 0.5 * b}, ref, r, "flat"
    )
    assert result["future_price_increment_confirmed"]
    assert not ev.gates(
        {"price": 0.5 * b, "incremental": 2 * b, "absolute": 0.5 * b}, ref, r, "flat"
    )["future_price_increment_confirmed"]
    assert not ev.skill_band(b, np.zeros_like(b), r)["supported"]
    with pytest.raises(ValueError):
        core.errors(np.full((120, 8), np.nan), b, b)


def test_price_errors_increment_and_bp_units():
    true = np.array([[1.0, 2.0, 3.0]])
    pred = np.array([[1.0, 3.0, 6.0]])
    err = core.errors(pred, true, np.ones((1, 3)))
    np.testing.assert_array_equal(err["incremental"], [[0, 1, 4]])
    np.testing.assert_array_equal(err["absolute"], [[0, 100, 300]])


def test_realistic_prepare_dedup_and_independent_windows(tmp_path, monkeypatch):
    n = 100
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=300, freq="15min"),
            "close": np.exp(np.arange(300) / 10000) * 100,
        }
    )
    s = SimpleNamespace(
        key="s/15/c", period=15, frame=frame, sessions=np.array(["2024-01-01"] * 300)
    )
    original = [
        {
            "key": s.key,
            "period": 15,
            "symbol": "s",
            "row": 127 + i,
            "end": str(frame.datetime.iloc[127 + i]),
            "month": "2024-01",
        }
        for i in range(n)
    ]
    features = np.tile(np.arange(300, dtype="float32")[:, None] / 100, (1, 28))
    atomic_json(original, tmp_path / "train_inventory.json")
    atomic_json({}, tmp_path / "manifest.json")

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x.mean(-1).cumsum(1)[..., None].expand(-1, -1, 768)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.encoder = Encoder()

    stats = {"x_mean": [0.0] * 28, "x_scale": [1.0] * 28}
    monkeypatch.setattr(data.parent, "tm", lambda m: {})
    monkeypatch.setattr(data.parent, "root", lambda m: tmp_path)
    monkeypatch.setattr(data.parent, "stats", lambda m: (stats, None))
    monkeypatch.setattr(
        data.parent,
        "data",
        lambda m, sp: {"x": data.rolling_inputs(features, np.arange(127, 227), stats)},
    )
    monkeypatch.setattr(data.parent, "construct", lambda *a: (Model(), IdealQuery()))
    monkeypatch.setattr(data.parent.xr.xd, "load_raw", lambda *a, **k: ([s], {}))
    monkeypatch.setattr(data.rp, "time_mask", lambda *a: np.ones(300, bool))
    monkeypatch.setattr(data, "encode", lambda f, p: features[: len(f)])
    monkeypatch.setattr(data.er, "trained_causality", lambda *a: {"status": "passed"})
    data.prepare({"extraction_batch": 13}, tmp_path, ("train",), False, "cpu")
    bank, _ = data.cache(tmp_path, "train")
    assert bank["z42"].shape == (108, 768)
    assert bank["mapping"][0, 1] == bank["mapping"][1, 0]
    expected = features[1:129].mean(-1).sum()
    assert bank["z42"][bank["mapping"][0, 1], 0] == pytest.approx(expected)
    data.prepare({"extraction_batch": 13}, tmp_path, ("train",), False, "cpu")
    with pytest.raises(FileNotFoundError):
        data.prepare({"extraction_batch": 13}, tmp_path, ("test",), False, "cpu")


def test_full_mocked_matrix_selection_and_evaluation(tmp_path, monkeypatch):
    meta, fit, tr, va = synthetic(tmp_path, n=120, epochs=1)

    class Process:
        def __init__(self, args):
            name = args[-1]
            job = next(j for j in run.jobs(meta) if j["name"] == name)
            seed = job["seed"]
            run.fit_job(
                meta,
                tmp_path,
                job,
                run.prepared(tr, fit, seed),
                run.prepared(va, fit, seed),
                fit,
                IdealQuery(),
                1.0,
                "cpu",
            )

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(run.subprocess, "Popen", Process)
    lock = run.fit_all(meta, tmp_path)
    assert len(lock["chosen"]) == 10
    assert all(v["epoch0_fallback"] for v in lock["candidates"].values())
    monkeypatch.setattr(run.parent, "construct", lambda *a: (torch.nn.Identity(), IdealQuery()))
    monkeypatch.setattr(run.parent, "stats", lambda m: ({"delta_scale": [1.0]}, None))
    monkeypatch.setattr(ev, "plot", lambda *a: None)
    ev.pretrain_audit(meta, tmp_path, "cpu")
    ev.pretrain_audit(meta, tmp_path, "cpu")
    ev.evaluate(meta, tmp_path, "cpu")
    report = run.read_json(tmp_path / "rollout_metrics.json")
    assert len(report["selected_validation_replay"]) == 10
    assert not report["primary_passed"]
    assert (tmp_path / "test_s42_predictions.npz").exists()
    history = tmp_path / run.jobs(meta)[0]["name"] / "history.json"
    history.write_text("[]")
    with pytest.raises(ValueError):
        run.check_selection(meta, tmp_path)


def test_failure_export_preserves_nonzero_status(tmp_path):
    import os

    env = dict(
        os.environ,
        BABEL_STATE_ROLLOUT_RUN=str(tmp_path / "run"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN="/usr/bin/false",
    )
    result = subprocess.run(
        ["bash", "scripts/babel_state_rollout768_autodl.sh", "all"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert (tmp_path / "download/run_reports.tar.gz").exists()
    assert "command_exit_code=1, run_status=failed" in result.stdout


def test_worker_failure_stops_remaining_workers(tmp_path, monkeypatch):
    meta, _, _, _ = synthetic(tmp_path, epochs=1)
    launched = []

    class FailedProcess:
        def __init__(self, args):
            self.status = 1 if not launched else None
            self.terminated = False
            launched.append(self)

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated = True
            self.status = -15

        def wait(self, timeout=None):
            return self.status

    monkeypatch.setattr(run.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="failed"):
        run.fit_all(meta, tmp_path)
    assert len(launched) == 2
    assert launched[1].terminated
    assert not (tmp_path / "selection_lock.json").exists()


def test_oom_preflight_halves_micro_without_optimizer_steps(tmp_path, monkeypatch):
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(len(x), 128, 768)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.encoder = Encoder()

    monkeypatch.setattr(run, "disk_budget", lambda *a: {})
    monkeypatch.setattr(run.parent, "construct", lambda *a: (Model(), IdealQuery()))
    monkeypatch.setattr(run.parent, "stats", lambda *a: ({"delta_scale": [1.0]}, None))
    monkeypatch.setattr(run.parent, "data", lambda *a: {"x": np.zeros((2, 128, 28), "float32")})
    for name in ("empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**20)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 2**30, 12 * 2**30))
    original = core.objective
    attempts = []

    def objective(model, states, *args):
        attempts.append(len(states))
        if len(states) > 64:
            raise torch.OutOfMemoryError("synthetic capacity")
        return original(model, states, *args)

    monkeypatch.setattr(core, "objective", objective)
    result = run.preflight({"extraction_batch": 2}, tmp_path, 2, "cpu")
    assert attempts == [128, 64]
    assert result["micro"] == 64 and result["jobs"] == 2
    assert result["optimizer_steps"] == 0


def test_cache_hash_tampering_is_rejected(tmp_path):
    synthetic(tmp_path)
    path = tmp_path / "cache/train_rows.json"
    path.write_text("[]")
    with pytest.raises(ValueError):
        data.cache(tmp_path, "train")
