"""Synthetic-only risks: physical alignment, gradients, matched budgets and recovery."""

import copy
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from obson.babel import content_retention as core
from obson.babel import content_retention_data as data
from obson.babel import content_retention_evaluate as ev
from obson.babel import content_retention_run as run
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
    original = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(original)


def inputs(n=3):
    x = np.random.default_rng(567).normal(0, 0.005, (n, 255, 28)).astype("float32")
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    raw = 100 * np.exp(np.sinh(x[..., :2].astype(float)).sum(-1).cumsum(-1) / 100)
    return x, raw


def models():
    return core.source.make_models("rolling", 42, "cpu", CONFIG)


def meta(micro=1):
    return {"statistics": STATS, "micro": micro, "batch": 2, "coverage_meta": {"encoder": CONFIG}}


def job(arm="content"):
    return {"name": f"{arm}_s42", "arm": arm, "seed": 42}


def tensors(n=3):
    x, raw = inputs(n)
    ps = torch.tensor([[64, 128], [112, 128], [127, 128]][:n])
    y, m = core.source.targets(torch.tensor(x), ps, STATS)
    return torch.tensor(x), torch.tensor(raw), ps, y, m


def test_card_roundtrip_and_exposure_budget():
    p = run.protocol()
    assert json.loads(json.dumps(p)) == p
    assert len(run.jobs()) == 6 and p["optimizer_updates"] == 13680
    assert p["train_state_exposures"] == 3448080 and p["fits"] == 0
    assert core.source.learning_rate(120, 120, 3e-4) == pytest.approx(run.RATE)
    assert data.COMPLETION == "835cce0452c27173f808b530871624b59041dceb71552074bf32e4c722e4139d"


def test_matched_schedule_boundaries_and_not_dense_or_uniform_claim():
    order, ps = core.schedule(4789, 42, 1)
    assert set(order) == set(range(4789))
    assert set(np.diff(ps, axis=1).ravel()) == {1, 16}
    assert set(ps.ravel()) == set(range(1, 129))
    assert np.all((ps >= 1) & (ps <= 128))
    np.testing.assert_array_equal(ps, core.schedule(4789, 42, 1)[1])
    assert not np.array_equal(ps, core.schedule(4789, 42, 2)[1])
    assert not np.array_equal(ps, core.schedule(4789, 43, 1)[1])


def test_raw_geometry_all_shifts_oracle_and_current_exclusion():
    _, raw, ps, y, m = tensors()
    raw_y = core.raw_targets(raw, ps, STATS)
    torch.testing.assert_close(raw_y, y[..., 0].double(), atol=1e-5, rtol=2e-5)
    a, b, ya, yb, valid = core.common_values(y, y, m, raw, ps, STATS)
    assert valid[..., 0].sum(1).tolist() == [63, 111, 126]
    torch.testing.assert_close(ya[valid], yb[valid], atol=1e-10, rtol=1e-10)
    assert core.errors(a, ya, valid).max() < 1e-10
    assert core.errors(b, yb, valid).max() < 1e-10
    changed = raw.clone()
    changed[0, 190:] *= 1.5  # A ends at physical index190; current anchor cancels centering.
    _, _, cy, _, _ = core.common_values(y, y, m, changed, ps, STATS)
    torch.testing.assert_close(cy[0][valid[0]], ya[0][valid[0]], atol=1e-10, rtol=1e-10)


def test_equal_wrong_predictions_not_content_success_and_offset_invariance():
    _, raw, ps, y, m = tensors()
    pred = y.clone()
    pred[..., 0] = 0
    t = core.objectives(pred, y, m, raw, ps, STATS, "aligned")
    assert t["alignment"].max() < 1e-12 and t["content"].min() > 0
    shifted = pred.clone()
    scales = core.source.hq.price_scales(STATS, pred)
    shifted[:, 0, :, 0] += 2 / scales
    a = core.objectives(shifted, y, m, raw, ps, STATS, "aligned")
    # Predictions are float32; adding a physical constant incurs rounding before
    # the double-precision centering. Compare below that input precision.
    torch.testing.assert_close(t["content"], a["content"], atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(t["alignment"], a["alignment"], atol=1e-9, rtol=1e-7)
    assert not torch.allclose(t["query"], a["query"])


def test_mismatch_rejected_and_missing_fields_receive_no_aux_gradient():
    _, raw, ps, y, m = tensors()
    bad = y.clone()
    bad[0, 1, 64, 2] += 1
    with pytest.raises(ValueError, match="nonprice"):
        core.common_values(y, bad, m, raw, ps, STATS)
    m[..., 4:] = False
    pred = y.clone().requires_grad_(True)
    t = core.objectives(pred, y, m, raw, ps, STATS, "content")
    t["content"].sum().backward()
    assert torch.count_nonzero(pred.grad[..., 4:]) == 0


def test_age_coordinate_does_not_discount_same_physical_aux_error():
    _, raw, ps, y, m = tensors()
    pred = y.clone().double()
    errors = torch.linspace(-0.2, 0.2, 63, dtype=torch.float64)
    scale = core.source.hq.price_scales(STATS, pred)
    pred[0, 0, :63, 0] += errors * STATS["delta_scale"][0] / scale[:63]
    pred[0, 1, 64:, 0] += errors * STATS["delta_scale"][0] / scale[64:]
    a, b, ya, yb, valid = core.common_values(pred, y, m, raw, ps, STATS)
    torch.testing.assert_close(
        core.errors(a, ya, valid)[0], core.errors(b, yb, valid)[0], atol=1e-9, rtol=1e-5
    )
    assert core.errors(a, b, valid)[0] < 1e-10


def test_actual_gradient_sum_reaches_encoder_and_existing_query():
    x, raw, ps, _, _ = tensors(2)
    e, q = models()
    params = list(e.parameters()) + list(q.parameters())
    result = core.terms(e, q, x, raw, ps, STATS, "aligned")
    gradients = {
        k: torch.autograd.grad(result[k].sum(), params, retain_graph=True, allow_unused=True)
        for k in ["query", "content", "alignment", "objective"]
    }
    for term in ("content", "alignment"):
        assert (
            sum(
                float(v.abs().sum())
                for v in gradients[term][: len(list(e.parameters()))]
                if v is not None
            )
            > 0
        )
        assert (
            sum(
                float(v.abs().sum())
                for v in gradients[term][len(list(e.parameters())) :]
                if v is not None
            )
            > 0
        )
    for a, b, c, d in zip(
        *(gradients[k] for k in ("query", "content", "alignment", "objective")), strict=True
    ):
        torch.testing.assert_close(d, a + 0.1 * b + 0.01 * c, atol=2e-5, rtol=2e-4)
    before = run.base.signature(e, q)
    audit = run.gradient_audit(e, q, x, raw, ps, STATS)
    assert audit["passed"] and audit["updates"] == 0
    assert before == run.base.signature(e, q)


def test_raw_targets_do_not_enter_encoder_or_query_and_early_state_is_causal():
    x, raw, ps, y, m = tensors(2)
    e, q = models()
    e.eval()
    q.eval()
    z, p = core.source.predict(e, q, x, ps)
    changed = x.clone()
    changed[0, 191:] += 8
    early = core.source.read_states(e, changed, ps[:, :1])
    torch.testing.assert_close(z[0, 0], early[0, 0], atol=1e-5, rtol=1e-5)
    raw2 = raw * 1.1
    first = core.objectives(p, y, m, raw, ps, STATS, "aligned")
    other = core.objectives(p, y, m, raw2, ps, STATS, "aligned")
    for k in first:
        torch.testing.assert_close(first[k], other[k], atol=1e-10, rtol=1e-10)


def test_micro_accumulation_tail_and_sampling_equivalence():
    x, raw = inputs()
    one = models()
    two = copy.deepcopy(one)

    def opt(pair):
        return torch.optim.SGD([p for model in pair for p in model.parameters()], lr=0.001)

    a = run.train_epoch(meta(1), job(), *one, x, raw, opt(one), 1, "cpu")
    b = run.train_epoch(meta(2), job(), *two, x, raw, opt(two), 1, "cpu")
    for k in [
        "steps",
        "parents",
        "states",
        "order_hash",
        "pairs_hash",
        "position_counts",
        "shift_counts",
    ]:
        assert a[k] == b[k]
    assert a["steps"] == 2 and a["states"] == 6
    for x, y in zip(one, two, strict=True):
        for p, q in zip(x.parameters(), y.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-4)


def fake_train_sources(monkeypatch, root, x):
    pair = models()
    monkeypatch.setattr(data, "load_source", lambda *args: copy.deepcopy(pair))
    m = meta()
    m["source"] = str(root)
    root.mkdir(exist_ok=True)
    ps = torch.tensor([[64, 112, 127, 128]]).expand(min(2, len(x)), -1)
    with torch.no_grad():
        _, pred = core.source.predict(*pair, torch.tensor(x[:2]), ps)
    np.savez(root / "val_D_s42.npz", query=pred.numpy())
    return m


def test_real_synthetic_worker_commit_resume_and_adam(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "EPOCHS", 2)
    x, raw = inputs()
    m = fake_train_sources(monkeypatch, tmp_path / "source", x)
    continuous = tmp_path / "continuous"
    restart = tmp_path / "restart"
    for folder in [continuous, restart]:
        folder.mkdir()
        atomic_json(m, folder / "manifest.json")
    run.train_job(m, continuous, job(), x, raw, x, "cpu")
    save = run.atomic_save

    def interrupt_after_commit(value, path):
        save(value, path)
        if value["epoch"] == 1:
            raise RuntimeError("after atomic commit")

    monkeypatch.setattr(run, "atomic_save", interrupt_after_commit)
    with pytest.raises(RuntimeError, match="atomic commit"):
        run.train_job(m, restart, job(), x, raw, x, "cpu")
    monkeypatch.setattr(run, "atomic_save", save)
    run.train_job(m, restart, job(), x, raw, x, "cpu")
    a = torch.load(continuous / job()["name"] / "last.pt", weights_only=True)
    b = torch.load(restart / job()["name"] / "last.pt", weights_only=True)
    assert a["current_signature"] == b["current_signature"]
    assert a["epoch"] == b["epoch"] == 2
    assert not (restart / job()["name"] / "pending_epoch.json").exists()


def test_uncommitted_update_cannot_be_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "EPOCHS", 1)
    x, raw = inputs()
    m = fake_train_sources(monkeypatch, tmp_path / "source", x)
    out = tmp_path / "out"
    out.mkdir()
    atomic_json(m, out / "manifest.json")
    train = run.train_epoch

    def fail(*args, **kwargs):
        train(*args, **kwargs)
        raise RuntimeError("after updates before commit")

    monkeypatch.setattr(run, "train_epoch", fail)
    with pytest.raises(RuntimeError, match="after updates"):
        run.train_job(m, out, job(), x, raw, x, "cpu")
    monkeypatch.setattr(run, "train_epoch", train)
    with pytest.raises(ValueError, match="Uncommitted epoch"):
        run.train_job(m, out, job(), x, raw, x, "cpu")


def test_data_prepare_real_adapter_contract_and_raw_cache_tamper(monkeypatch, tmp_path):
    x, raw = inputs(1)
    frame = pd.DataFrame(
        {"datetime": pd.date_range("2020-01-01", periods=255, freq="15min"), "close": raw[0]}
    )
    from types import SimpleNamespace

    item = SimpleNamespace(key="X/15/C.X1", period=15, frame=frame)
    row = {
        "key": item.key,
        "row": 254,
        "period": 15,
        "input_start": str(frame.datetime.iloc[0]),
        "end": str(frame.datetime.iloc[-1]),
    }
    meta = {
        "statistics": STATS,
        "coverage_meta": {
            "statistics": STATS,
            "source_meta": {},
            "source_indexes": {"train": "locked"},
        },
    }
    atomic_json(meta, tmp_path / "manifest.json")
    monkeypatch.setattr(data, "cache", lambda *args: (x, [row]))
    monkeypatch.setattr(data.source.source.old.parent, "tm", lambda x: x)
    monkeypatch.setattr(
        data.source.source.old.parent.xr.xd, "load_raw", lambda *args, **kwargs: ([item], {})
    )
    monkeypatch.setattr(data.source.original.original, "encode", lambda *args: x[0])
    monkeypatch.setattr(data.source.original.rp, "time_mask", lambda *args: np.ones(255, bool))
    first = data.prepare(meta, tmp_path, ("train",))
    assert first["train"]["feature_max_abs"] == 0
    np.testing.assert_array_equal(data.closes(tmp_path, "train"), raw)
    assert data.prepare(meta, tmp_path, ("train",)) == first
    (tmp_path / "raw/train.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        data.closes(tmp_path, "train")


def test_evaluation_physical_truth_and_guards_are_not_just_agreement():
    x, raw = inputs(120)
    labels = ev.targets(x, raw, STATS)
    bank = ev.errors(labels["target"].copy(), labels, STATS)
    assert bank["shift64/price/shape_b_mse"].max() < 1e-14
    assert set(ev.protection_keys()) <= set(bank)
    rows = [{"month": str(i % 6), "key": str(i % 12)} for i in range(120)]
    # Identical wrong reconstructions have perfect agreement but fail gain.
    bank = {k: np.full(120, 1.0) for k in ev.protection_keys()}
    bank["shift16/price/shape_b_mse"] = np.full(120, 1.0)
    bank["shift64/price/shape_b_mse"] = np.full(120, 1.0)
    result = ev.comparison(bank, bank, rows, ["shift64/price/shape_b_mse"], 0.05)
    assert not result["passed"] and not result["gain_passed"]
    improved = {k: v * 0.8 for k, v in bank.items()}
    assert ev.comparison(improved, bank, rows, ["shift64/price/shape_b_mse"], 0.05)["passed"]
    improved["original/p128/near_activity"] = bank["original/p128/near_activity"] * 1.2
    assert not ev.comparison(improved, bank, rows, ["shift64/price/shape_b_mse"], 0.05)["passed"]
    improved["original/p128/near_activity"][0] = np.nan
    with pytest.raises(ValueError, match="population/support"):
        ev.comparison(improved, bank, rows, [], 0)


def test_alignment_cannot_erase_content_gain():
    rows = [{"month": str(i % 6), "key": str(i % 12)} for i in range(120)]
    baseline = {k: np.ones(120) for k in ev.protection_keys()}
    content = {k: v * 0.9 for k, v in baseline.items()}
    aligned = {k: v.copy() for k, v in content.items()}
    for d in (1, 16, 64):
        aligned[f"shift{d}/price/disagreement_mse"] *= 0.8
    # This fits the 10% protection margin vs content but loses the primary gain
    # against the source/control. Agreement alone must not qualify this arm.
    for d in (16, 64):
        aligned[f"shift{d}/price/shape_b_mse"][:] = 0.98
    banks = {
        f"{arm}_s{s}": values
        for s in core.SEEDS
        for arm, values in (
            ("source", baseline),
            ("control", baseline),
            ("content", content),
            ("aligned", aligned),
        )
    }
    result = ev.decisions(
        dict.fromkeys(("test", "cross_research"), banks),
        dict.fromkeys(("test", "cross_research"), rows),
    )
    assert result["content_supported"] and not result["alignment_increment_supported"]


def test_synthetic_six_workers_lock_extract_and_evaluate(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "EPOCHS", 1)
    monkeypatch.setitem(data.COUNTS, "train", 3)
    x, raw = inputs()
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    m = meta(2)
    m["source"] = str(source)
    m["coverage_meta"]["source_indexes"] = {s: "locked_" + s for s in data.COUNTS}
    pairs = {s: core.source.make_models("rolling", s, "cpu", CONFIG) for s in core.SEEDS}
    monkeypatch.setattr(data, "load_source", lambda m, s, d: copy.deepcopy(pairs[s]))
    rows = [{"month": str(i), "key": str(i)} for i in range(3)]
    monkeypatch.setattr(data, "cache", lambda *args: (x, rows))
    monkeypatch.setattr(data, "closes", lambda *args: raw)
    ps = torch.tensor([[64, 112, 127, 128]]).expand(len(x), -1)
    for seed, pair in pairs.items():
        with torch.no_grad():
            z, p = core.source.predict(*pair, torch.tensor(x), ps)
        for split in ("val", "test", "cross_research"):
            np.savez(
                source / f"{split}_D_s{seed}.npz", query=p.numpy(), endpoint_state=z[:, -1].numpy()
            )
    atomic_json(m, out / "manifest.json")
    for j in run.jobs():
        run.train_job(m, out, j, x, raw, x, "cpu")
    lock = run.lock_models(m, out)
    assert len(lock["records"]) == 6
    result = ev.evaluate(m, out, "cpu")
    assert not result["promoted"] and not result["content_supported"]
    assert len(run.read_json(out / "evaluation_attempts.json")) == 24
    # Completed extraction is reused without loading or forwarding any model.
    monkeypatch.setattr(ev, "load", lambda *args: pytest.fail("must reuse completed extraction"))
    ev.extract(m, out, "content_s42", "test", x, "cpu")
    (out / "predictions/test_content_s42.json").unlink()
    with pytest.raises(ValueError, match="Uncommitted evaluation"):
        ev.extract(m, out, "content_s42", "test", x, "cpu")


def test_session_lease_cannot_launch_unbounded_worker(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.require_session(tmp_path)
    atomic_json({"deadline_unix": 0, "used_seconds": run.SESSION_CAP}, tmp_path / "budget.json")
    with pytest.raises(ValueError, match="active bounded"):
        run.require_session(tmp_path)


def test_completed_run_does_not_consume_new_session(monkeypatch, tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    atomic_json(
        {"schema": run.SCHEMA, "source": str(source), "output": str(out), "jobs": 2},
        out / "request.json",
    )
    atomic_json(
        {"files": {"request.json": run.sha256(out / "request.json")}}, out / "completion.json"
    )
    atomic_json({"used_seconds": run.SESSION_CAP}, out / "budget.json")
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kw: pytest.fail("no rerun"))
    assert run.supervise(source, out, 2) == 0


@pytest.mark.parametrize("status,code", [("complete", 0), ("failed", 1), ("stopped", 130)])
def test_shell_exports_all_outcomes_without_weights(tmp_path, status, code):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs/BABEL_ROADMAP.md").write_text("test")
    script = Path(run.base.lr.REPO) / "scripts/babel_content_retention768_autodl.sh"
    dest = root / "scripts" / script.name
    dest.write_text(script.read_text())
    fake = tmp_path / "python"
    fake.write_text(
        '#!/usr/bin/env bash\nif [[ "$1" == -m ]]; then exit '
        + str(code)
        + "; fi\nexec "
        + sys.executable
        + ' "$@"\n'
    )
    fake.chmod(0o755)
    out = root / "run"
    out.mkdir()
    atomic_json({"status": status}, out / "status.json")
    (out / "last.pt").write_bytes(b"weights")
    np.savez(out / "scores.npz", a=[1])
    download = tmp_path / "download"
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake),
        "BABEL_RETENTION_RUN": str(out),
        "BABEL_DOWNLOAD_DIR": str(download),
    }
    result = subprocess.run(["bash", str(dest), "all"], env=env, capture_output=True, text=True)
    assert result.returncode == code, result.stderr
    with tarfile.open(download / "run_reports.tar.gz") as tar:
        names = tar.getnames()
        assert "run/scores.npz" in names and "run/last.pt" not in names
        exported = tar.extractfile("run/export_status.txt").read().decode()
        assert f"command_exit_code={code}" in exported
        if code:
            assert f"run_status={status}" in exported


def test_stopping_shell_forwards_signal_and_exports(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs/BABEL_ROADMAP.md").write_text("test")
    script = Path(run.base.lr.REPO) / "scripts/babel_content_retention768_autodl.sh"
    dest = root / "scripts" / script.name
    dest.write_text(script.read_text())
    fake = tmp_path / "python"
    ready, stopped = tmp_path / "ready", tmp_path / "stopped"
    fake.write_text(
        f"#!{sys.executable}\nimport os, sys, signal, time\nfrom pathlib import Path\n"
        f"def stop(*args):\n Path({str(stopped)!r}).touch()\n raise SystemExit(130)\n"
        "if sys.argv[1] != '-m': os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"Path({str(ready)!r}).touch()\nwhile True: time.sleep(.1)\n"
    )
    fake.chmod(0o755)
    download = tmp_path / "download"
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake),
        "BABEL_RETENTION_RUN": str(root / "run"),
        "BABEL_DOWNLOAD_DIR": str(download),
    }
    proc = subprocess.Popen(
        ["bash", str(dest), "all"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        limit = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < limit and proc.poll() is None:
            time.sleep(0.01)
        assert ready.exists()
        proc.send_signal(signal.SIGTERM)
        output, error = proc.communicate(timeout=10)
        assert proc.returncode == 130 and stopped.exists(), (output, error)
        with tarfile.open(download / "run_reports.tar.gz") as tar:
            assert "run_status=stopped" in tar.extractfile("run/export_status.txt").read().decode()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
