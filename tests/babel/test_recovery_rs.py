"""Synthetic optimizer/recovery tests for RS; no historical model execution."""

import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_recovery_audit import CONFIG, STATS
from test_recovery_qualification import LOCAL, data
from torch import nn

from obson.babel import recovery_rs as run
from obson.babel import recovery_rs_evaluate as ev
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    old = run.r0.runtime_state()
    run.r0.apply_runtime(dict(run.r0.RUNTIME_PROFILES["context"], threads=1))
    monkeypatch.setattr(run.old.parent, "stats", lambda m: (STATS, LOCAL))
    yield
    run.r0.apply_runtime(old)


def models():
    e, q = uc.make_models("rolling", 42, "cpu", CONFIG)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        local = nn.Linear(32, 112).eval().requires_grad_(False)
    return e.eval(), q.eval(), local


def batch(n=3):
    x, b = data(n)
    return torch.tensor(x), {k: torch.tensor(v) for k, v in b.items()}


def test_loss_and_gradients_match_historical_baseline_and_declared_interventions():
    e, q, local = models()
    x, b = batch()
    ps = torch.tensor([[32, 64, 96, 128]] * len(x))

    def get(arm):
        return run.terms(e, q, local, x, b, ps, STATS, LOCAL, arm)

    z, p = run.old.core.predict(e, q, x, ps, True)
    y, m = uc.targets(x, ps, STATS, True)
    expected = uc.loss_rows(p, y, m, STATS, True)["objective"].mean(1)
    torch.testing.assert_close(get("baseline")["objective"], expected, atol=0, rtol=0)
    params = list(e.parameters()) + list(q.parameters())

    def grad(loss):
        gs = torch.autograd.grad(loss.mean(), params, allow_unused=True)
        return torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for p, g in zip(params, gs, strict=True)
            ]
        )

    g = {arm: grad(get(arm)["objective"]) for arm in run.ARMS}
    local_gradient = 0.25 * grad(get("with_local")["local"])
    remote = grad(get("baseline")["structure"])
    for a, b_, v in [
        ("baseline", "without_remote", remote),
        ("with_local", "baseline", local_gradient),
        ("without_remote_with_local", "without_remote", local_gradient),
        ("with_local", "without_remote_with_local", remote),
    ]:
        torch.testing.assert_close(g[a] - g[b_], v, atol=2e-6, rtol=2e-4)
    assert not get("baseline")["local"].requires_grad
    count = sum(p.numel() for p in e.parameters())
    assert local_gradient[:count].abs().sum() > 0 and not local_gradient[count:].any()
    assert all(p.grad is None and not p.requires_grad for p in local.parameters())


@pytest.mark.parametrize("arm", run.ARMS)
def test_loss_does_not_use_future_after_selected_prefix(arm):
    e, q, local = models()
    x, b = batch()
    ps = torch.full((len(x), 1), 64)
    before = run.terms(e, q, local, x, b, ps, STATS, LOCAL, arm)
    changed = x.clone()
    changed[:, 191:] += 1
    after_b = {k: v.clone() for k, v in b.items()}
    after_b["y"][:, 64:] += 10
    after = run.terms(e, q, local, changed, after_b, ps, STATS, LOCAL, arm)
    for k in before:
        torch.testing.assert_close(before[k], after[k], atol=0, rtol=0)


def test_actual_baseline_step_matches_original_with_micro_tail():
    x, b = data(3)
    meta = {"batch": 128, "micro": 2}
    e, q, local = models()
    e0, q0, _ = models()
    a = run.lr.optimizer(e, q, run.lr.RATES[42])
    bopt = run.lr.optimizer(e0, q0, run.lr.RATES[42])
    observed = []
    t = run.train_epoch(
        meta, {"seed": 42, "arm": "baseline"}, e, q, local, x, b, a, 1, "cpu", observed.append
    )
    old = run.old.train_epoch(meta, {"seed": 42, "mode": "prefix"}, e0, q0, x, bopt, 1, "cpu")
    assert observed == [1] and t["steps"] == old["steps"] == 1
    for k in ("order_hash", "prefix_hash", "examples", "supervised_states"):
        assert t[k] == old[k]
    for m, m0 in ((e, e0), (q, q0)):
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, m0.state_dict()[k], atol=0, rtol=0)
    assert t["preclip_norm"]["mean"] > 0


def test_actual_synthetic_research_extraction_and_cache_replay(tmp_path, monkeypatch):
    x, _ = data(3)
    context, out = tmp_path / "context", tmp_path / "out"
    (context / "cache").mkdir(parents=True)
    out.mkdir()
    atomic_json({}, context / "cache/test_index.json")
    atomic_json({}, out / "model_selection_lock.json")
    rows = [{"key": "X/15/Y", "period": 15, "month": "2020-01"}] * 3
    monkeypatch.setattr(ev, "check_models", lambda _: None)
    monkeypatch.setattr(ev, "check_readouts", lambda _: None)
    monkeypatch.setattr(ev.old.data, "cache", lambda *a: (x, rows))
    monkeypatch.setattr(ev, "variants", lambda: [{"name": "source_s42"}])
    monkeypatch.setattr(ev, "load", lambda *a: models()[:2])
    states, _, _ = ev.extract({"micro": 2}, context, out, "test", "cpu")
    assert states["source_s42"].shape == (3, 32)
    assert states["error/source_s42/full/near"].shape == (3, 4)
    summary = run.read_json(out / "test_reconstruction.json")["source_s42"]
    for p in summary["native/far"]["positions"]:
        if p["prefix"] == 32:
            assert p["supported_windows"] == 0 and p["mean"] is None
    second, _, _ = ev.extract({"micro": 2}, context, out, "test", "cpu")
    for k in states:
        np.testing.assert_array_equal(states[k], second[k])


@pytest.fixture
def worker(tmp_path, monkeypatch):
    meta = {"batch": 128, "micro": 2, "validation_prefixes": [32, 64, 96, 128]}
    x, b = data(3)
    context, r1, out = [tmp_path / k for k in ("context", "r1", "out")]
    (context / "prefix_s42").mkdir(parents=True)
    (r1 / "s42").mkdir(parents=True)
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    e, q, _ = models()
    sig = run.lr.model_signature(e, q)
    initial = run.old.validation(meta, e, q, x, "cpu")
    atomic_json(
        {
            "initial_encoder_hash": sig["encoder"],
            "initial_query_hash": sig["query"],
            "initial_validation": initial,
        },
        context / "prefix_s42/completion.json",
    )
    opt = run.lr.optimizer(e, q, run.lr.RATES[42])
    ref = []
    for epoch in (1, 2):
        t = run.old.train_epoch(meta, {"mode": "prefix", "seed": 42}, e, q, x, opt, epoch, "cpu")
        ref.append({"train": t, "validation": run.old.validation(meta, e, q, x, "cpu")})
    atomic_json(ref, r1 / "s42/history.json")
    monkeypatch.setattr(run, "construct", lambda *a: models())
    monkeypatch.setattr(run, "EPOCHS", 2)
    return SimpleNamespace(
        meta=meta,
        x=x,
        b=b,
        context=context,
        r1=r1,
        out=out,
        job={"seed": 42, "arm": "baseline", "name": "baseline_s42"},
        expected=(e, q),
    )


def invoke(w):
    return run.train_job(w.meta, w.context, w.r1, w.out, w.job, w.x, w.b, w.x, "cpu")


def test_worker_complete_exact_updates_and_idempotent(worker):
    w = worker
    done = invoke(w)
    assert done["epochs"] == 2 and done["optimizer_updates"] == 2
    p = w.out / "baseline_s42/last.pt"
    digest = run.sha256(p)
    ck = torch.load(p, weights_only=True)
    for key, m in zip(("encoder", "query"), w.expected, strict=True):
        for k, v in ck[key].items():
            torch.testing.assert_close(v, m.state_dict()[k], atol=0, rtol=0)
    assert invoke(w) == done and run.sha256(p) == digest
    assert not (p.parent / "pending_epoch.json").exists()


def test_resume_atomic_epoch_before_receipt_write(worker, monkeypatch):
    w = worker
    original = run.atomic_json
    hit = []

    def interrupt(value, path):
        if Path(path).name == "checkpoint.json" and value["epoch"] == 1 and not hit:
            hit.append(True)
            raise RuntimeError("simulated receipt interruption")
        return original(value, path)

    monkeypatch.setattr(run, "atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="receipt interruption"):
        invoke(w)
    monkeypatch.setattr(run, "atomic_json", original)
    done = invoke(w)
    assert done["optimizer_updates"] == 2
    ck = torch.load(w.out / "baseline_s42/last.pt", weights_only=True)
    for key, m in zip(("encoder", "query"), w.expected, strict=True):
        for k, v in ck[key].items():
            torch.testing.assert_close(v, m.state_dict()[k], atol=0, rtol=0)


def test_uncommitted_epoch_blocks_extra_steps(worker, monkeypatch):
    w = worker
    original = run.train_epoch

    def stop(*a, **k):
        original(*a, **k)
        raise RuntimeError("after update")

    monkeypatch.setattr(run, "train_epoch", stop)
    with pytest.raises(RuntimeError, match="after update"):
        invoke(w)
    monkeypatch.setattr(run, "train_epoch", original)
    pending = run.read_json(w.out / "baseline_s42/pending_epoch.json")
    assert pending["observed_steps"] == 1
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        invoke(w)


def test_changed_checkpoint_is_not_silently_resumed(worker):
    w = worker
    invoke(w)
    p = w.out / "baseline_s42/best.pt"
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Immutable"):
        invoke(w)


@pytest.mark.parametrize("corruption", ["optimizer", "selection"])
def test_resume_rejects_recipe_or_selection_changes_even_with_new_receipt(worker, corruption):
    w = worker
    invoke(w)
    folder = w.out / "baseline_s42"
    (folder / "completion.json").unlink()
    p = folder / "last.pt"
    ck = torch.load(p, weights_only=True)
    if corruption == "optimizer":
        ck["optimizer"]["param_groups"][0]["weight_decay"] = 0.9
    else:
        ck["best_score"] = -1.0
    run.atomic_save(ck, p)
    atomic_json({"epoch": ck["epoch"], "sha256": run.sha256(p)}, folder / "checkpoint.json")
    with pytest.raises(ValueError, match="Resume .*changed|Resume selection journal differs"):
        invoke(w)


def test_validation_selection_epoch0_does_not_count_as_progress():
    rows = [{"key": f"X/{i % 3}", "month": str(i % 12)} for i in range(120)]
    states = {}
    for name, value in [("candidate", 0.8), ("baseline", 1.2), ("source", 1.0)]:
        for view in ("full", "native"):
            for metric in run.old.core.METRICS:
                states[f"error/{name}/{view}/{metric}"] = np.full((120, 4), value)
    utility = {
        name: {k: np.full(120, v) for k in ("utility", "direction", "volatility")}
        for name, v in [("candidate", 0.8), ("baseline", 1.2), ("source", 1.0)]
    }
    assert not ev.gate(states, utility, rows, "candidate", "baseline", "source", False)[
        "reconstruction_progress"
    ]
    assert ev.gate(states, utility, rows, "candidate", "baseline", "source", True)[
        "representation_progress"
    ]
    # Better than damaged baseline is insufficient when worse than original source.
    for view in ("full", "native"):
        states[f"error/candidate/{view}/near"].fill(1.01)
    assert not ev.gate(states, utility, rows, "candidate", "baseline", "source", True)[
        "reconstruction_progress"
    ]


def test_paired_blocks_accounting_and_insufficient_evidence():
    rows = [{"key": f"C{i % 5}", "month": str(i % 10)} for i in range(120)]
    a = np.arange(120, dtype=float)
    ci = ev.paired(a, a + 2, rows)
    for c in ci.values():
        assert (
            c["supported"]
            and c["delta"] == -2
            and c["low"] == pytest.approx(-2)
            and c["high"] == pytest.approx(-2)
        )
    assert not ev.check(a[:3], a[:3] + 2, rows[:3], 1)["passed"]
    with pytest.raises(ValueError):
        ev.paired(a, np.ones(3), rows)


def test_factor_interaction_is_not_sum_of_main_effects():
    rows = [{"key": "C", "month": str(i % 10)} for i in range(120)]
    states = {}
    for arm, v in zip(run.ARMS, (1.0, 2.0, 3.0, 8.0), strict=True):
        for view in ("full", "native"):
            for metric in run.old.core.METRICS:
                states[f"error/{arm}_s42_last/{view}/{metric}"] = np.full((120, 4), v)
    effect = ev.effects(states, rows, 42, "last")
    k = "full/endpoint/near"
    assert effect["interaction"][k]["month"]["delta"] == 4
    assert effect["add_L_with_S"][k]["month"]["delta"] == 2
    assert effect["add_L_without_S"][k]["month"]["delta"] == 6


def test_physical_units_and_mask():
    y = torch.zeros(2, 4, 127, 7)
    p = y.clone()
    p[..., 0] = 0.2 / (torch.arange(1, 128).sqrt() * 0.1)
    m = torch.ones_like(p, dtype=torch.bool)
    m[..., 0, 0] = False
    p[..., 0, 0] = 1e8
    physical = ev.physical_rows(p, y, m, STATS)
    torch.testing.assert_close(
        physical["close_mae_logbps"], torch.full((2, 4), 20.0, dtype=torch.float64)
    )
    assert physical["change_rmse_logbps"].max() < 1e-5


def test_research_extraction_requires_readout_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "check_models", lambda out: None)
    with pytest.raises(FileNotFoundError):
        ev.extract({}, tmp_path, tmp_path, "test", "cpu")


def test_model_lock_tamper_and_readout_state_binding(tmp_path, monkeypatch):
    atomic_json({}, tmp_path / "manifest.json")
    atomic_json({"manifest_sha256": "wrong", "files": {}}, tmp_path / "model_selection_lock.json")
    with pytest.raises(ValueError, match="binding"):
        ev.check_models(tmp_path)
    atomic_json(
        {
            "model_lock_sha256": run.sha256(tmp_path / "model_selection_lock.json"),
            "files": {},
            "states_indexes": {"train": "wrong"},
        },
        tmp_path / "readout_lock.json",
    )
    (tmp_path / "cache").mkdir()
    atomic_json({}, tmp_path / "cache/train_index.json")
    with pytest.raises(ValueError, match="input states"):
        ev.check_readouts(tmp_path)


def test_disk_check_never_deletes_source(tmp_path, monkeypatch):
    r1 = tmp_path / "r1"
    out = tmp_path / "out"
    out.mkdir()
    for s in (42, 43):
        (r1 / f"s{s}").mkdir(parents=True)
        (r1 / f"s{s}/last.pt").write_bytes(b"weight")
    monkeypatch.setattr(run.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="no source deleted"):
        run.check_disk(out, r1)
    assert (r1 / "s42/last.pt").read_bytes() == b"weight"


def test_timeout_and_budget_not_reset(tmp_path, monkeypatch):
    s = tmp_path / "source"
    s.mkdir()
    out = tmp_path / "out"
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))

    def stop(*a, **k):
        raise subprocess.TimeoutExpired(a, k["timeout"])

    monkeypatch.setattr(run.subprocess, "run", stop)
    assert run.supervise(s, out) == 124
    budget = run.read_json(out / "budget.json")
    assert budget["used_seconds"] == 10 and budget["attempts"][0]["reserved_seconds"] == 35241
    budget["used_seconds"] = 35241
    atomic_json(budget, out / "budget.json")
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(s, out)


def test_script_failure_export_excludes_weights_and_state_cache(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    (out / "cache").mkdir(parents=True)
    atomic_json({"status": "failed"}, out / "status.json")
    (out / "model.pt").write_text("not exported")
    np.savez(out / "cache/states.npz", x=np.ones(3))
    np.savez(out / "test_predictions.npz", x=np.ones(3))
    env = dict(
        os.environ,
        BABEL_RS_RUN=str(out),
        BABEL_RS_SOURCE=str(tmp_path / "missing"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN=str(root / ".venv/bin/python"),
    )
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_rs_autodl.sh", "all"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as tar:
        assert "run/test_predictions.npz" in tar.getnames()
        assert not any("/cache/" in n or n.endswith(".pt") for n in tar.getnames())
        assert b"run_status=failed" in tar.extractfile("run/export_status.txt").read()


def test_full_evaluation_report_has_both_seeds_splits_kinds_and_fallback(tmp_path, monkeypatch):
    names = ["raw3584", "current28", "pca768"] + [v["name"] for v in ev.variants()]
    stats = {"mean": [0.0] * 13, "scale": [1.0] * 13}
    atomic_json(
        {"target_stats": stats, "raw_stats": {}, "heads": dict.fromkeys(names, [])},
        tmp_path / "readout_fit.json",
    )
    weights = {"pca": np.zeros((1, 1))}
    for n in names:
        weights[n + "/weights"] = np.ones((1, 13))
        weights[n + "/intercepts"] = np.zeros(13)
    np.savez(tmp_path / "readout_weights.npz", **weights)
    workers = {j["name"]: {"best_epoch": 1} for j in run.jobs()}
    workers["with_local_s42"]["best_epoch"] = 0
    atomic_json({"workers": workers}, tmp_path / "model_selection_lock.json")
    monkeypatch.setattr(ev, "check_models", lambda p: None)
    monkeypatch.setattr(ev, "check_readouts", lambda p: None)
    monkeypatch.setattr(ev, "fit", lambda *a: None)
    rows = [{"key": "C" + str(i % 3), "month": str(i % 12), "period": 15} for i in range(120)]
    states = {v["name"]: np.zeros((120, 2)) for v in ev.variants()}
    for v in ev.variants():
        value = 1.0 if v["job"] is None else (1.2 if v["job"]["arm"] == "baseline" else 0.8)
        for view in ("full", "native"):
            for metric in run.old.core.METRICS:
                states[f"error/{v['name']}/{view}/{metric}"] = np.full((120, 4), value)
    seen = []

    def extract(*args):
        seen.append(args[3])
        return states, np.zeros((120, 128, 28)), rows

    monkeypatch.setattr(ev, "extract", extract)
    monkeypatch.setattr(ev, "representation", lambda *a: np.zeros((120, 2)))
    monkeypatch.setattr(ev.up, "targets", lambda x: (np.ones((120, 13)), np.ones((120, 13), bool)))
    monkeypatch.setattr(ev.up, "predict", lambda *a: np.full((120, 13), 2.0))
    ev.run({}, tmp_path, tmp_path, "cpu")
    assert seen == ["test", "cross_research"]
    report = run.read_json(tmp_path / "rs_metrics.json")
    decision = run.read_json(tmp_path / "decision.json")
    for split in ev.SPLITS:
        assert len(report[split]["gates"]) == 12
        assert len(report[split]["factor_contrasts"]) == 4
        assert len(report[split]["utility"]) == 21
    assert decision["arms"]["without_remote"]["reconstruction_progress"]
    assert not decision["arms"]["with_local"]["reconstruction_progress"]
    assert not decision["promoted"]


def test_source_replay_mismatch_stops_before_any_next_epoch(worker):
    w = worker
    p = w.r1 / "s42/history.json"
    ref = run.read_json(p)
    ref[0]["validation"]["full/endpoint/near"] += 1
    atomic_json(ref, p)
    with pytest.raises(ValueError, match="mismatched"):
        invoke(w)
    folder = w.out / "baseline_s42"
    assert (folder / "r1_epoch1_replay.json").exists()
    ck = torch.load(folder / "last.pt", weights_only=True)
    assert ck["epoch"] == 0
    pending = run.read_json(folder / "pending_epoch.json")
    assert pending["observed_steps"] == 1
