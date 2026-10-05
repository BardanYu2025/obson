"""V06 synthetic training paths and delivery; no real weights or local neural runs."""

import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_recovery_audit import CONFIG, STATS
from test_recovery_qualification import data

from obson.babel import recovery_context as core
from obson.babel import recovery_context_evaluate as ev
from obson.babel import recovery_context_run as run
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    before = run.r0.runtime_state()
    run.r0.apply_runtime(dict(run.r0.RUNTIME_PROFILES["context"], threads=1))
    monkeypatch.setattr(run.old.parent, "stats", lambda m: (STATS, {}))
    yield
    run.r0.apply_runtime(before)


def models():
    return tuple(m.eval() for m in uc.make_models("rolling", 42, "cpu", CONFIG))


def test_real_loss_route_audit_and_deliberate_wrong_b_mask(tmp_path, monkeypatch):
    e, q = models()
    x = torch.tensor(data(2)[0])
    before = run.lr.model_signature(e, q)
    checks = core.control_audit(e, q, x, STATS, tmp_path / "ok.json")
    assert all(v["passed"] for v in checks.values())
    assert run.lr.model_signature(e, q) == before
    assert all(p.grad is None for p in list(e.parameters()) + list(q.parameters()))
    monkeypatch.setitem(core.ARMS, "B_full_common", core.Policy(False, False))
    with pytest.raises(ValueError, match="actual training controls"):
        core.control_audit(e, q, x, STATS, tmp_path / "wrong.json")
    assert not run.read_json(tmp_path / "wrong.json")["checks"]["AB_mask"]["passed"]


@pytest.mark.parametrize("arm,native", [("A_prefix_common", True), ("C_full_extended", False)])
def test_actual_training_step_exactly_matches_original_endpoints_and_microtail(arm, native):
    e, q = models()
    old_e, old_q = models()
    x = data(3)[0]
    meta = {"batch": 128, "micro": 2}
    opt = run.lr.optimizer(e, q, run.lr.RATES[42])
    old_opt = run.lr.optimizer(old_e, old_q, run.lr.RATES[42])
    result = run.train_epoch(meta, {"arm": arm, "seed": 42}, e, q, x, opt, 1, "cpu")
    reference = run.old.train_epoch(
        meta,
        {"mode": "prefix" if native else "rolling", "seed": 42},
        old_e,
        old_q,
        x,
        old_opt,
        1,
        "cpu",
    )
    for k in ("order_hash", "prefix_hash", "steps", "examples", "supervised_states"):
        assert result[k] == reference[k]
    for m, original in ((e, old_e), (q, old_q)):
        for key, value in m.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[key], atol=0, rtol=0)


def test_actual_train_ab_targets_reductions_and_bc_inputs_match(monkeypatch):
    captured = {}
    original = core.terms

    def wrapper(e, q, x, p, stats, arm):
        def record(name, xx, pp, pred, y, mask, result):
            captured.setdefault(name, []).append(
                (xx.detach().clone(), pp.clone(), y.clone(), mask.clone())
            )

        return original(e, q, x, p, stats, arm, record)

    monkeypatch.setattr(core, "terms", wrapper)
    x = data(3)[0]
    for arm in run.ARMS:
        e, q = models()
        run.train_epoch(
            {"batch": 128, "micro": 2},
            {"arm": arm, "seed": 42},
            e,
            q,
            x,
            run.lr.optimizer(e, q, run.lr.RATES[42]),
            1,
            "cpu",
        )
    for a, b, c in zip(*(captured[k] for k in run.ARMS), strict=True):
        for i in (1, 2, 3):
            torch.testing.assert_close(a[i], b[i], atol=0, rtol=0)
        torch.testing.assert_close(
            uc.rolling_windows(b[0], b[1]), uc.rolling_windows(c[0], c[1]), atol=0, rtol=0
        )
        # A/B gradients w.r.t. the SAME artificial prediction expose effective weights.
        grads = []
        for sample in (a, b):
            pred = torch.zeros_like(sample[2], requires_grad=True)
            grads.append(
                torch.autograd.grad(
                    core.objective(pred, sample[2], sample[3], STATS)["objective"].sum(), pred
                )[0]
            )
        torch.testing.assert_close(*grads, atol=0, rtol=0)


def test_missing_far_is_unsupported_not_perfect_reconstruction():
    e, q = models()
    values, support = core.evaluate(e, q, data(3)[0], STATS, 2, "cpu")
    k = "B_full_common/far"
    assert not support[k][:, 0].any() and support[k][:, -1].all()
    rows = ev.summaries(values, support)[k]
    assert rows[0]["mean"] is None and rows[0]["supported_windows"] == 0
    assert rows[-1]["mean"] is not None


@pytest.fixture
def worker(tmp_path, monkeypatch):
    meta = {"batch": 128, "micro": 2, "validation_prefixes": [32, 64, 96, 128]}
    x = data(3)[0]
    context, r1, out = [tmp_path / k for k in ("context", "r1", "out")]
    (context / "prefix_s42").mkdir(parents=True)
    (r1 / "s42").mkdir(parents=True)
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    e, q = models()
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
        context=context,
        r1=r1,
        out=out,
        job={"arm": "A_prefix_common", "seed": 42, "name": "A_prefix_common_s42"},
        expected=(e, q),
    )


def invoke(w):
    return run.train_job(w.meta, w.context, w.r1, w.out, w.job, w.x, w.x, "cpu")


def assert_weights(w):
    ck = torch.load(w.out / w.job["name"] / "last.pt", weights_only=True)
    for key, m in zip(("encoder", "query"), w.expected, strict=True):
        for k, v in ck[key].items():
            torch.testing.assert_close(v, m.state_dict()[k], atol=0, rtol=0)


def test_actual_worker_completion_and_idempotence(worker):
    done = invoke(worker)
    assert done["epochs"] == 2 and done["optimizer_updates"] == 2
    assert_weights(worker)
    assert invoke(worker) == done


def test_atomic_epoch_resume_matches_uninterrupted(worker, monkeypatch):
    original = run.atomic_json
    hits = []

    def interrupt(value, path):
        if Path(path).name == "checkpoint.json" and value["epoch"] == 1 and not hits:
            hits.append(True)
            raise RuntimeError("synthetic power loss")
        return original(value, path)

    monkeypatch.setattr(run, "atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="power loss"):
        invoke(worker)
    monkeypatch.setattr(run, "atomic_json", original)
    assert invoke(worker)["optimizer_updates"] == 2
    assert_weights(worker)


def test_uncommitted_steps_refuse_silent_extra_budget(worker, monkeypatch):
    original = run.train_epoch

    def interrupt(*a, **k):
        original(*a, **k)
        raise RuntimeError("after update")

    monkeypatch.setattr(run, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="after update"):
        invoke(worker)
    monkeypatch.setattr(run, "train_epoch", original)
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        invoke(worker)


def test_wrong_source_replay_stops_before_next_epoch(worker):
    p = worker.r1 / "s42/history.json"
    r = run.read_json(p)
    r[0]["validation"]["full/endpoint/near"] += 1
    atomic_json(r, p)
    with pytest.raises(ValueError, match="mismatched"):
        invoke(worker)
    ck = torch.load(worker.out / worker.job["name"] / "last.pt", weights_only=True)
    assert ck["epoch"] == 0


def test_export_failure_includes_controls_excludes_weights(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    out.mkdir()
    atomic_json({"status": "failed"}, out / "status.json")
    (out / "weight.pt").write_bytes(b"keep")
    np.savez(out / "errors.npz", x=np.ones(3))
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_context_autodl.sh", "all"],
        cwd=root,
        env=dict(
            os.environ,
            BABEL_V06_RUN=str(out),
            BABEL_V06_SOURCE=str(tmp_path / "absent"),
            BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
            PYTHON_BIN=str(root / ".venv/bin/python"),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as tar:
        assert "run/errors.npz" in tar.getnames()
        assert not any(p.endswith(".pt") for p in tar.getnames())
        assert b"run_status=failed" in tar.extractfile("run/export_status.txt").read()


def test_evaluation_requires_all_models_locked(tmp_path):
    atomic_json({}, tmp_path / "manifest.json")
    atomic_json(
        {"manifest_sha256": run.sha256(tmp_path / "manifest.json"), "workers": {}, "files": {}},
        tmp_path / "model_selection_lock.json",
    )
    with pytest.raises(ValueError, match="Incomplete"):
        ev.check_lock(tmp_path)


def test_contrasts_masking_and_known_context_effects():
    n = 120
    rows = [{"key": f"C{i % 3}", "month": str(i % 12)} for i in range(n)]
    bank = {}
    support = {}
    for arm in run.ARMS:
        for metric in run.old.core.METRICS:
            support[f"{arm}/{metric}"] = np.ones((n, 4), bool)
    for name, value in [("source_s42", 1.0)] + [
        (f"{a}_s42_last", v) for a, v in zip(run.ARMS, (1.2, 0.8, 0.9), strict=True)
    ]:
        bank[name] = {k: np.full((n, 4), value) for k in support}
    effects = ev.contrasts(bank, support, rows, 42, "last")
    assert effects["deployed_common/interior/price"]["intervals"]["month"][
        "delta"
    ] == pytest.approx(-0.4)
    assert effects["deployed_common/interior/price"]["supported_gain_5pct"]
    assert "relative_change" not in effects["difference_in_differences/interior/price"]
    empty = np.zeros((n, 1), bool)
    assert (
        ev.contrast(np.ones((n, 1)), np.ones((n, 1)), empty, empty, rows)["status"]
        == "no_common_support"
    )
    with pytest.raises(ValueError, match="identical"):
        ev.contrast(np.ones((n, 1)), np.ones((n, 1)), empty, ~empty, rows)


def test_pinned_protocol_and_budget():
    p = run.protocol()
    assert (len(run.jobs()), p["epochs"], p["updates"], p["worker_epochs"]) == (6, 60, 13680, 360)
    assert p["rates"]["42"] == list(run.lr.RATES[42])


def test_actual_extraction_cache_and_tamper_guard(tmp_path, monkeypatch):
    context = tmp_path / "context"
    out = tmp_path / "out"
    (context / "cache").mkdir(parents=True)
    out.mkdir()
    atomic_json({}, context / "cache/test_index.json")
    atomic_json({}, out / "model_selection_lock.json")
    x = data(3)[0]
    rows = [{"key": "C", "month": "2020-01"}] * 3
    monkeypatch.setattr(ev, "check_lock", lambda p: {})
    monkeypatch.setattr(ev.run.old.data, "cache", lambda *a: (x, rows))
    monkeypatch.setattr(ev, "load", lambda *a: models())
    monkeypatch.setattr(
        ev, "variants", lambda: [{"name": "source_s42", "seed": 42, "job": None, "kind": "source"}]
    )
    bank, mask, actual_rows = ev.extract({"micro": 2}, context, out, "test", "cpu")
    assert actual_rows == rows and bank["source_s42"]["B_full_common/near"].shape == (3, 4)
    again, _, _ = ev.extract({"micro": 2}, context, out, "test", "cpu")
    for key in bank["source_s42"]:
        np.testing.assert_array_equal(bank["source_s42"][key], again["source_s42"][key])
    (out / "evaluation/test_source_s42.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Immutable"):
        ev.extract({"micro": 2}, context, out, "test", "cpu")


def test_full_report_matrix_last_primary_and_no_promotion(tmp_path, monkeypatch):
    rows = [{"key": f"C{i % 3}", "month": str(i % 12)} for i in range(120)]
    support = {
        f"{arm}/{m}": np.ones((120, 4), bool) for arm in run.ARMS for m in run.old.core.METRICS
    }
    workers = {j["name"]: {"best_epoch": 0} for j in run.jobs()}
    monkeypatch.setattr(ev, "check_lock", lambda p: {"workers": workers})
    bank = {}
    for v in ev.variants():
        # Bests are source fallbacks; primary last comparisons can still be read independently.
        value = (
            1.0
            if v["job"] is None or v["kind"] == "best"
            else {run.ARMS[0]: 1.2, run.ARMS[1]: 0.8, run.ARMS[2]: 0.9}[v["job"]["arm"]]
        )
        bank[v["name"]] = {k: np.full((120, 4), value) for k in support}
    seen = []

    def extract(meta, context, out, split, device):
        seen.append(split)
        return bank, support, rows

    monkeypatch.setattr(ev, "extract", extract)
    ev.run_evaluation({}, tmp_path, tmp_path, "cpu")
    decision = run.read_json(tmp_path / "decision.json")
    report = run.read_json(tmp_path / "context_metrics.json")
    assert seen == list(ev.SPLITS)
    assert decision["finite_B_benefit"] and not decision["promoted"]
    assert not decision["conclusion_repair_complete"] and set(decision["best_epochs"].values()) == {
        0
    }
    assert len(decision["strata"]) == 4
    for split in ev.SPLITS:
        assert set(report[split]["contrasts"]) == {
            f"s{s}_{k}" for s in (42, 43) for k in ("best", "last")
        }


def test_timeout_is_cumulative_and_exports_status(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    ticks = iter((0.0, 3.0))
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(run.subprocess, "run", timeout)
    assert run.supervise(source, out) == 124
    budget = run.read_json(out / "budget.json")
    assert budget["used_seconds"] == 3
    assert run.read_json(out / "status.json")["status"] == "timeout"
    budget["used_seconds"] = run.SESSION_CAP
    atomic_json(budget, out / "budget.json")
    with pytest.raises(ValueError, match="exhausted"):
        run.supervise(source, out)


def test_source_statistics_and_full_input_overlap_are_checked(tmp_path, monkeypatch):
    context, out = tmp_path / "context", tmp_path / "out"
    (context / "cache").mkdir(parents=True)
    out.mkdir()
    atomic_json(STATS, context / "statistics.json")
    monkeypatch.setattr(run, "STATISTICS_SHA256", run.sha256(context / "statistics.json"))
    atomic_json({"learning_rate_bridge": {}}, out / "identity.json")
    for split in ("train", "val"):
        atomic_json({}, context / f"cache/{split}_audit.json")
        atomic_json({}, context / f"cache/{split}_index.json")
    train = [{"key": "C", "input_start": "2020-01-01", "end": "2020-01-10"}]
    val = [{"key": "C", "input_start": "2020-02-01", "end": "2020-02-10"}]
    meta = {"encoder_lr": 3e-5, "query_lr": 3e-4}
    run.data_evidence(meta, context, out, train, val)
    assert not run.read_json(out / "data_provenance.json")[
        "original_fit_independently_reproduced_this_run"
    ]
    val[0]["input_start"] = "2020-01-10"
    with pytest.raises(ValueError, match="overlap"):
        run.data_evidence(meta, context, out, train, val)


def test_output_cannot_overlap_any_ancestor(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        run.interface.separate(tmp_path / "parent/child", tmp_path / "parent")
    with pytest.raises(ValueError, match="separate"):
        run.interface.separate(tmp_path / "parent", tmp_path / "parent/child")


@pytest.mark.parametrize("corruption", ["optimizer", "selection"])
def test_resume_rejects_changed_optimizer_and_selection(worker, corruption):
    invoke(worker)
    folder = worker.out / worker.job["name"]
    (folder / "completion.json").unlink()
    ck = torch.load(folder / "last.pt", weights_only=True)
    if corruption == "optimizer":
        ck["optimizer"]["param_groups"][0]["weight_decay"] = 0.9
    else:
        ck["best_score"] = -1.0
    run.atomic_save(ck, folder / "last.pt")
    atomic_json(
        {"epoch": ck["epoch"], "sha256": run.sha256(folder / "last.pt")}, folder / "checkpoint.json"
    )
    with pytest.raises(ValueError, match="Resume optimizer|Resume selection"):
        invoke(worker)


def test_zero_error_control_is_not_a_five_percent_gain():
    rows = [{"key": "C", "month": str(i % 12)} for i in range(120)]
    zero = np.zeros((120, 1))
    mask = np.ones_like(zero, dtype=bool)
    assert not ev.contrast(zero, zero, mask, mask, rows)["supported_gain_5pct"]


def test_actual_arm_specific_weight_drift_is_rejected(tmp_path, monkeypatch):
    original = core.objective
    calls = []

    def altered(pred, y, mask, stats):
        values = original(pred, y, mask, stats)
        calls.append(True)
        # Second forward is B. Simulate an unintended B-only objective multiplier.
        if len(calls) == 2:
            values["objective"] = values["objective"] * 0.5
        return values

    monkeypatch.setattr(core, "objective", altered)
    e, q = models()
    with pytest.raises(ValueError, match="actual training controls"):
        core.control_audit(e, q, torch.tensor(data(2)[0]), STATS, tmp_path / "weight.json")
    checks = run.read_json(tmp_path / "weight.json")["checks"]
    assert not checks["actual_loss/B_full_common/objective"]["passed"]
