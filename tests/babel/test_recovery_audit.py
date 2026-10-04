"""R0 safety/failure tests and synthetic end-to-end replay; no real model updates."""

import copy
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from obson.babel import recovery_audit as run
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json, atomic_save

STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
    "fitted_on": "train_only",
}
CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}


@pytest.fixture(autouse=True)
def no_optimizer(monkeypatch):
    torch.set_num_threads(1)
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer may be constructed or stepped in R0")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    yield
    torch.backends.mha.set_fastpath_enabled(old)


def inputs(n=2, length=255):
    x = np.random.default_rng(42).normal(0, 0.01, (n, length, 28)).astype("float32")
    x[..., [11, 23, 24, 25, 26, 27]] = 1
    return x


def test_common_targets_and_band_support():
    checks, support = run.target_contract(inputs(), STATS)
    run.require(checks, "same physical targets")
    np.testing.assert_allclose(support["native_band_weights"][0], [0.5, 0.5, 0])
    np.testing.assert_allclose(support["full_band_weights"][0], [1 / 3] * 3)


def test_nested_comparison_rejects_missing_extra_nan_and_changed_value(tmp_path):
    expected = {"a": {"b": 1.0}}
    for actual in ({}, {"a": {"b": 1.0, "c": 0}}, {"a": {"b": float("nan")}}, {"a": {"b": 1.1}}):
        checks = run.nested_compare(actual, expected)
        with pytest.raises(ValueError):
            run.save_checks(tmp_path, "failed", checks)
        assert json.loads((tmp_path / "failed.json").read_text())["checks"] == checks
    run.require(run.nested_compare(expected, expected), "same")


def test_bound_asset_and_output_safety(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    p = source / "file"
    p.write_text("one")
    digest = run.sha256(p)
    assert run.required_file(source, "file", digest)["bytes"] == 3
    p.write_text("two")
    with pytest.raises(ValueError, match="mismatch"):
        run.required_file(source, "file", digest)
    with pytest.raises(ValueError, match="relative path"):
        run.required_file(source, "../elsewhere", digest)
    for out in (source, source / "child", tmp_path):
        with pytest.raises(ValueError):
            run.safe_output(source, out)
    out = tmp_path / "output"
    out.mkdir()
    (out / "old").write_text("keep")
    with pytest.raises(ValueError):
        run.safe_output(source, out)
    assert (out / "old").read_text() == "keep"


def test_causal_and_gradient_checks_preserve_synthetic_models(tmp_path):
    encoder, query = uc.make_models("rolling", 42, "cpu", CONFIG)
    encoder.eval()
    query.eval()
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    x = torch.tensor(inputs())
    run.causality(encoder, query, x, tmp_path, "toy")
    for native in (True, False):
        report = run.gradient_stats(encoder, query, x, STATS, native)
        assert set(report["tasks"]) == {
            "path",
            "change1",
            "body",
            "activity",
            "structure",
            "price_near",
            "price_mid",
            "price_far",
        }
        assert all(v["encoder_norm"] > 0 and v["query_norm"] > 0 for v in report["tasks"].values())
        assert 0 < report["hypothetical_global_clip_factor"] <= 1
    for k, v in before.items():
        assert torch.equal(v, encoder.state_dict()[k])
    assert all(p.grad is None for p in encoder.parameters())


def test_causality_detects_future_read(tmp_path):
    class Noncausal(nn.Module):
        def forward(self, x):
            h = x[..., :16].repeat(1, 1, 2)
            return h + h.mean(1, keepdim=True)

    _, query = uc.make_models("rolling", 42, "cpu", CONFIG)
    with pytest.raises(ValueError):
        run.causality(Noncausal(), query, torch.tensor(inputs()), tmp_path, "bad")
    assert not json.loads((tmp_path / "bad_causality.json").read_text())["checks"][
        "prefix32/state"
    ]["passed"]


def test_raw_rebuild_checks_every_eligible_row_and_exclusion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "cache").mkdir()
    out = tmp_path / "out"
    out.mkdir()
    raw = inputs(1, 300)[0]
    frame = pd.DataFrame(raw)
    frame["datetime"] = pd.date_range("2020-01-01", periods=300, freq="15min")
    series = SimpleNamespace(key="X/15/C", period=15, frame=frame)
    bounds = {"train_until": "2020-02-01"}
    originals = [
        {"key": series.key, "row": e, "end": str(frame.datetime.iloc[e])} for e in (200, 254, 299)
    ]
    kept = [
        dict(
            originals[i],
            original_index=i,
            input_start=str(frame.datetime.iloc[originals[i]["row"] - 254]),
        )
        for i in (1, 2)
    ]
    x = np.stack([raw[r["row"] - 254 : r["row"] + 1] for r in kept])
    old = {"x": np.stack([raw[r["row"] - 127 : r["row"] + 1] for r in originals])}
    old["y"], old["mask"] = run.ar.ordered_targets(old["x"])
    for split in ("train", "val", "test", "cross_research"):
        atomic_json(originals, source / f"{split}_inventory.json")
        atomic_json(
            {"bounds": bounds, "excluded": [{"index": 0}]}, source / f"cache/{split}_audit.json"
        )
    monkeypatch.setattr(run.previous.parent, "tm", lambda m: m)
    monkeypatch.setattr(run.previous.parent, "root", lambda m: source)
    monkeypatch.setattr(run.previous.parent, "data", lambda m, s: old)
    monkeypatch.setattr(run.previous.parent.xr.xd, "load_raw", lambda *a, **kw: ([series], bounds))
    monkeypatch.setattr(run.rp, "time_mask", lambda *a: np.ones(300, dtype=bool))
    monkeypatch.setattr(run.previous.data, "cache", lambda *a: (x, kept))
    from obson.babel import uniform_context_data as ud

    monkeypatch.setattr(ud.original, "encode", lambda frame, p: frame.iloc[:, :28].to_numpy())
    result = run.raw_replay(source, {}, STATS, out)
    assert all(v["windows"] == 2 and v["excluded"] == 1 for v in result.values())
    x[0, 0, 0] += 0.2
    with pytest.raises(ValueError, match="raw_failure"):
        run.raw_replay(source, {}, STATS, out)
    assert (out / "raw_failure.json").exists()


def test_reference_schedule_detects_changed_sampling(tmp_path):
    meta = {"batch": 128, "encoder_lr": 3e-5, "query_lr": 3e-4}
    for seed in (42, 43):
        folder = tmp_path / f"prefix_s{seed}"
        folder.mkdir()
        rows = []
        for epoch in range(1, 6):
            order, ps = run.previous.core.schedule(3, seed, epoch)
            rows.append(
                {
                    "epoch": epoch,
                    "encoder_lr": 3e-5,
                    "query_lr": 3e-4,
                    "train": {
                        "order_hash": run.previous.parent.ur.bb.cov.ndarray_hash(order),
                        "prefix_hash": run.previous.parent.ur.bb.cov.ndarray_hash(ps),
                        "steps": 1,
                        "examples": 3,
                        "supervised_states": 15,
                    },
                }
            )
        atomic_json(rows, folder / "history.json")
    run.schedule_audit(tmp_path, meta, 3, tmp_path)
    rows[0]["train"]["order_hash"] = "bad"
    atomic_json(rows, tmp_path / "prefix_s43/history.json")
    with pytest.raises(ValueError, match="schedule differs"):
        run.schedule_audit(tmp_path, meta, 3, tmp_path)


def test_six_synthetic_snapshot_replays_including_frozen_restore(tmp_path, monkeypatch):
    source = tmp_path / "context"
    source.mkdir()
    macro = tmp_path / "macro"
    macro.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    out = tmp_path / "audit"
    out.mkdir()
    for path in (source, macro, shared):
        atomic_json({}, path / "manifest.json")
    value = {"score": 1.0}
    chosen = {}
    parent_chosen = {}
    models = {}
    for seed in (42, 43):
        e, q = uc.make_models("rolling", seed, "cpu", CONFIG)
        model = nn.Module()
        model.core = nn.Module()
        model.core.encoder = e
        model.local_head = nn.Linear(32, 16 * 7)
        models[seed] = (model, q)
        selected = f"control_s{seed}.pt"
        atomic_save(
            {
                "epoch": 1,
                "metadata": {"manifest_sha256": run.sha256(shared / "manifest.json")},
                "model": model.state_dict(),
                "query": q.state_dict(),
                "validation": value,
            },
            shared / selected,
        )
        atomic_save(
            {"epoch": 1, "model": model.state_dict(), "query": q.state_dict(), "validation": value},
            macro / selected,
        )
        chosen[str(seed)] = {"path": selected}
        parent_chosen[str(seed)] = {"path": selected, "selected_epoch": 1}
        for kind in ("prefix", "rolling"):
            (source / f"{kind}_s{seed}").mkdir()
        atomic_json(
            {
                "initial_encoder_hash": run.previous.parent.ur.bb.state_signature(e),
                "initial_query_hash": run.previous.parent.ur.bb.state_signature(q),
                "initial_validation": value,
            },
            source / f"prefix_s{seed}/completion.json",
        )
        job = {"name": f"rolling_s{seed}", "mode": "rolling", "seed": seed}
        atomic_save(
            {
                "epoch": 60,
                "metadata": run.previous.binding(source, job),
                "encoder": e.state_dict(),
                "query": q.state_dict(),
                "validation": value,
            },
            source / f"rolling_s{seed}/update_last.pt",
        )
    meta = {"task_source": {"source": str(macro), "chosen": chosen}}
    mm = {"task_source": {"source": str(shared), "chosen": parent_chosen}}

    def construct(m, seed, device):
        model, q = copy.deepcopy(models[seed])
        model.eval().requires_grad_(False)
        return model, q.eval()

    monkeypatch.setattr(run.previous.parent, "construct", construct)
    monkeypatch.setattr(run.previous.parent.base.prior, "construct", construct)
    monkeypatch.setattr(run.previous.parent.base, "validation", lambda *a: dict(value))
    monkeypatch.setattr(run.previous.parent.base.prior, "validation", lambda *a: dict(value))
    monkeypatch.setattr(run.previous, "validation", lambda *a: dict(value))
    monkeypatch.setattr(run, "schedule_audit", lambda *a: None)
    x = inputs()
    y, m = run.ar.ordered_targets(x[:, -128:])
    monkeypatch.setattr(run.previous.data, "cache", lambda *a: (x, []))
    monkeypatch.setattr(
        run.previous.parent, "data", lambda *a: {"x": x[:, -128:], "y": y, "mask": m}
    )
    monkeypatch.setattr(
        run.previous.parent, "stats", lambda *a: (STATS, {"mean": [0.0] * 7, "scale": [1.0] * 7})
    )
    result = run.neural_replay(source, meta, mm, {}, STATS, out, device="cpu")
    assert len(result) == 6
    for seed in (42, 43):
        diagnostic = json.loads((out / f"gradients_s{seed}.json").read_text())
        assert diagnostic["detached_local"]["encoder_all_unused"]
    assert all(not r["precise_resume_claim"] for r in result.values())


def test_no_cuda_failure_is_persisted_without_training(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="AutoDL CUDA-only"):
        run.run(source, out)
    status = json.loads((out / "audit_status.json").read_text())
    assert (
        status["status"] == "failed"
        and status["optimizer_updates"] == 0
        and not status["r1_authorized"]
    )


def test_export_packs_failure_and_excludes_weights(tmp_path):
    out = tmp_path / "audit"
    out.mkdir()
    atomic_json({"status": "failed", "optimizer_updates": 0}, out / "audit_status.json")
    (out / "secret.pt").write_text("not a report")
    env = {
        **os.environ,
        "BABEL_RECOVERY_RUN": str(out),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "PYTHON_BIN": str(Path(".venv/bin/python").resolve()),
    }
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_r0_autodl.sh", "export"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(tmp_path / "download/audit_reports.tar.gz") as archive:
        assert "audit/audit_status.json" in archive.getnames()
        assert not any(n.endswith(".pt") for n in archive.getnames())
        assert "run_status=failed" in archive.extractfile("audit/export_status.txt").read().decode()


def test_price_age_decomposition_preserves_boundary_loss_and_gradient():
    x = torch.tensor(inputs())
    ps = torch.tensor([[32, 64, 96, 128]]).expand(2, -1)
    for native in (True, False):
        y, mask = uc.targets(x, ps, STATS, native)
        pred = (y + 0.01 * torch.randn_like(y)).requires_grad_(True)
        parts = uc.loss_rows(pred, y, mask, STATS, True)
        expected = (
            0.4 * parts["path"].mean() + 0.2 * parts["change1"].mean() + 0.1 * parts["body"].mean()
        )
        actual = sum(run.price_band_losses(pred, y, mask, STATS).values())
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual, pred, retain_graph=True)[0]
        b = torch.autograd.grad(expected, pred, retain_graph=True)[0]
        torch.testing.assert_close(a, b)
        structure = torch.autograd.grad(parts["structure"].mean(), pred)[0]
        assert torch.count_nonzero(structure[..., :16, :]) == 0
        assert torch.count_nonzero(structure[..., 16:, 0]) > 0


def test_supervisor_timeout_records_failure_without_overwriting_existing(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "audit"
    code = (
        Path("scripts/babel_recovery_r0_autodl.sh")
        .read_text()
        .split("<<'PY'\n", 1)[1]
        .split("\nPY\n", 1)[0]
    )
    monkeypatch.setattr(sys, "argv", ["-", str(source), str(out)])

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 3600
        raise subprocess.TimeoutExpired(args[0], 3600)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(SystemExit) as failure:
        exec(code, {})
    assert failure.value.code == 124
    status = json.loads((out / "audit_status.json").read_text())
    assert status["status"] == "timeout" and status["optimizer_updates"] == 0
    original = (out / "audit_status.json").read_bytes()
    with pytest.raises(SystemExit) as failure:
        exec(code, {})
    assert "existing evidence is protected" in failure.value.code
    assert (out / "audit_status.json").read_bytes() == original


def test_completed_audit_requires_review_and_keeps_zero_updates(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic orchestration only")
    monkeypatch.setattr(run.previous, "code_identity", lambda: {})

    def identities(source, out):
        atomic_json({"source": str(source)}, out / "identity.json")
        return {}, {}, {}, {}

    monkeypatch.setattr(run, "identities", identities)
    monkeypatch.setattr(run, "raw_replay", lambda *a: {})
    monkeypatch.setattr(run, "neural_replay", lambda *a: dict.fromkeys(range(6)))
    run.run(source, out)
    status = json.loads((out / "completion.json").read_text())
    assert status["status"] == "audit_complete_requires_review"
    assert not status["r1_authorized"] and status["optimizer_updates"] == 0
    assert status["replayed_checkpoints"] == 6
