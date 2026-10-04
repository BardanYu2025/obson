"""Synthetic-only V02/V14 checks; optimizer construction and updates forbidden."""

import copy
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from test_recovery_audit import CONFIG, STATS, inputs
from torch import nn

from obson.babel import recovery_qualification as q
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json

LOCAL = {"mean": [0.2] * 7, "scale": [0.8] * 7}


@pytest.fixture(autouse=True)
def readonly(monkeypatch):
    old = q.r0.runtime_state()
    q.r0.apply_runtime(dict(q.r0.RUNTIME_PROFILES["context"], threads=1))

    def forbidden(*a, **kw):
        raise AssertionError("Zero updates required")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    yield
    q.r0.apply_runtime(old)


def data(n=3):
    x = inputs(n)
    x[:, 160:170, [23, 24, 25, 27]] = 0
    y, m = q.r0.ar.ordered_targets(x[:, -128:])
    return x, {"x": x[:, -128:], "y": y, "mask": m}


def test_equal_norm_different_vectors_and_nonfinite_fail():
    named = [("encoder.p", nn.Parameter(torch.zeros(2)))]
    check = q.vector_comparison([torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 1.0])], named)
    assert not check["passed"] and check["failed_coordinates"] == 2
    assert check["actual_norm"] == check["expected_norm"]
    assert check["parameters"][0]["witnesses"][0]["residual"] in (-1.0, 1.0)
    for v in (float("nan"), float("inf")):
        check = q.vector_comparison([torch.tensor([v, 0.0])], [torch.zeros(2)], named)
        assert not check["passed"] and not check["finite"]
    with pytest.raises(ValueError, match="shapes"):
        q.vector_comparison([torch.ones(3)], [torch.ones(3)], named)


def test_objective_comparison_detects_unplanned_weight_and_local_changes():
    historical = {"query": torch.ones(2, 4), "structure": torch.full((2, 4), 2.0)}
    terms = {}
    for arm, mode in q.objectives.ARMS.items():
        terms[arm] = {
            "query": torch.ones(2),
            "remote_structure": torch.full((2,), 2.0),
            "local_diagnostic": torch.full((2,), 4.0),
            "optimization_value": torch.full(
                (2,), 1.0 + 2.0 * mode.remote_structure + mode.local_to_encoder
            ),
        }
    assert q.objective_comparison(terms, historical)["passed"]
    terms["with_local"]["optimization_value"][0] += 0.01
    assert not q.objective_comparison(terms, historical)["checks"]["with_local/optimization"][
        "passed"
    ]
    terms["without_remote"]["local_diagnostic"][1] += 0.01
    assert not q.objective_comparison(terms, historical)["checks"]["without_remote/local"]["passed"]


@pytest.mark.parametrize("native", (True, False))
def test_actual_gradient_identity_and_no_mutation(native):
    encoder, query = uc.make_models("rolling", 42, "cpu", CONFIG)
    encoder.eval()
    query.eval()
    before = copy.deepcopy(encoder.state_dict())
    report = q.gradient_diagnostic(encoder, query, torch.tensor(inputs()), STATS, native)
    assert report["passed"], {k: v["residual_norm"] for k, v in report["checks"].items()}
    assert len(report["checks"]) == 4
    assert report["same_forward"] and report["joint_clip_factor_diagnostic"] > 0
    assert all(p.grad is None for p in list(encoder.parameters()) + list(query.parameters()))
    assert all(torch.equal(v, encoder.state_dict()[k]) for k, v in before.items())


def test_wrong_age_decomposition_is_detected(monkeypatch):
    original = q.r0.price_band_losses
    monkeypatch.setattr(
        q.r0, "price_band_losses", lambda *a: {k: 2 * v for k, v in original(*a).items()}
    )
    encoder, query = uc.make_models("rolling", 42, "cpu", CONFIG)
    report = q.gradient_diagnostic(
        encoder.eval(), query.eval(), torch.tensor(inputs()), STATS, True
    )
    assert not report["checks"]["direct_price_vs_sum_age_bands"]["passed"]
    assert report["checks"]["joint_vs_sum_families"]["passed"]


def test_real_target_paths_include_missing_masks_and_detect_shift():
    x, old = data()
    result = q.target_audit(x, old, STATS, LOCAL)
    assert result["passed"], result
    old["y"][:, 40, 0] += 0.1
    assert not q.target_audit(x, old, STATS, LOCAL)["passed"]


def test_train_mean_is_masked_and_has_no_validation_input():
    _, old = data()
    mean, counts = q.train_mean(old, STATS, LOCAL)
    y, m = q.hq.ba.local_targets(
        torch.tensor(old["y"]), torch.tensor(old["mask"]), q.ps_for(3), STATS, LOCAL
    )
    expected = torch.where(m, y.double(), 0).sum(0) / m.sum(0).clamp_min(1)
    torch.testing.assert_close(mean.double(), expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(counts, m.sum(0).double())
    changed = {k: v.copy() for k, v in old.items()}
    changed["y"][~changed["mask"]] = 1e8
    a, b = q.train_mean(changed, STATS, LOCAL)
    torch.testing.assert_close(a, mean, rtol=0, atol=0)
    torch.testing.assert_close(b, counts)


def test_aligned_rows_detect_reordered_identity_and_bad_inputs(tmp_path, monkeypatch):
    x, old = data()
    inventory = [{"key": "X/15/C", "end": str(i)} for i in range(3)]
    rows = [dict(r, original_index=i) for i, r in enumerate(inventory)]
    atomic_json(inventory, tmp_path / "train_inventory.json")
    monkeypatch.setattr(q.previous.parent, "root", lambda m: tmp_path)
    monkeypatch.setattr(q.previous.parent, "data", lambda *a: old)
    monkeypatch.setattr(q.previous.data, "cache", lambda *a: (x, rows))
    assert len(q.aligned_data(tmp_path, {}, "train")[2]) == 3
    rows[1]["original_index"] = 0
    with pytest.raises(ValueError, match="mapping"):
        q.aligned_data(tmp_path, {}, "train")
    rows[1]["original_index"] = 1
    rows[1]["end"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        q.aligned_data(tmp_path, {}, "train")
    rows[1]["end"] = "1"
    old["x"] = old["x"].copy()
    old["x"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="input differs"):
        q.aligned_data(tmp_path, {}, "train")


def test_reviewed_audit_hash_and_source_binding(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    for name in ("manifest.json", "completion.json"):
        atomic_json({}, source / name)
    monkeypatch.setattr(q.previous, "code_identity", lambda: {"bound": "hash"})
    atomic_json(
        {
            "source": str(source),
            "historical_code": {"bound": "hash"},
            "code_sha256": {Path(q.r0.__file__).name: q.sha256(q.r0.__file__)},
        },
        audit / "manifest.json",
    )
    atomic_json(
        {
            "manifest_sha256": q.sha256(source / "manifest.json"),
            "completion_sha256": q.sha256(source / "completion.json"),
        },
        audit / "identity.json",
    )
    atomic_json(
        {
            "status": "audit_complete_requires_review",
            "optimizer_updates": 0,
            "files": {n: q.sha256(audit / n) for n in ("manifest.json", "identity.json")},
        },
        audit / "completion.json",
    )
    monkeypatch.setattr(q, "AUDIT_COMPLETION", q.sha256(audit / "completion.json"))
    assert q.evidence_binding(audit, source)
    atomic_json({"changed": True}, source / "manifest.json")
    with pytest.raises(ValueError, match="identity"):
        q.evidence_binding(audit, source)
    (audit / "completion.json").write_text("{}")
    with pytest.raises(ValueError, match="reviewed"):
        q.evidence_binding(audit, source)


@pytest.mark.parametrize("local_target_failure", (False, True))
def test_synthetic_full_preflight_uses_real_validation_gradients_and_readers(
    tmp_path, monkeypatch, local_target_failure
):
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    x, old = data()
    rows = [{"key": "X/15/C", "end": str(i), "original_index": i} for i in range(3)]
    meta = {"micro": 2, "validation_prefixes": list(q.PREFIXES)}
    models = {}
    sig = q.previous.parent.ur.bb.state_signature
    monkeypatch.setattr(q.previous.parent, "stats", lambda m: (STATS, LOCAL))
    for seed in (42, 43):
        e, reader = uc.make_models("rolling", seed, "cpu", CONFIG)
        model = nn.Module()
        model.core = nn.Module()
        model.core.encoder = e
        model.local_head = nn.Linear(32, 112)
        model.eval()
        reader.eval()
        models[seed] = (model, reader)
        initial = q.previous.validation(meta, e, reader, x, "cpu")
        folder = source / f"prefix_s{seed}"
        folder.mkdir()
        atomic_json(
            {
                "initial_encoder_hash": sig(e),
                "initial_query_hash": sig(reader),
                "initial_validation": initial,
            },
            folder / "completion.json",
        )

    def construct(m, seed, device):
        return copy.deepcopy(models[seed])

    identity = {"fixture": "synthetic"}
    atomic_json(identity, audit / "identity.json")

    def identities(s, o):
        atomic_json(identity, o / "identity.json")
        return meta, {}, {}, STATS

    monkeypatch.setattr(q, "evidence_binding", lambda *a: identity)
    monkeypatch.setattr(q.r0, "identities", identities)
    monkeypatch.setattr(q, "aligned_data", lambda *a: (x, old, rows))
    monkeypatch.setattr(q.r0, "schedule_audit", lambda *a: None)
    monkeypatch.setattr(q.previous.parent, "construct", construct)
    if local_target_failure:
        target_audit = q.target_audit

        def fail_local(*args):
            result = target_audit(*args)
            return dict(result, local_passed=False, passed=False)

        monkeypatch.setattr(q, "target_audit", fail_local)
    result = q.preflight(source, audit, out, "cpu")
    assert result["r1_preflight_passed"] and result["source_unchanged"]
    assert result["rs_target_contract_passed"] != local_target_failure
    if local_target_failure:
        assert not result["rs_reader_qualified"]
    assert len(list(out.glob("gradients_*.json"))) == 4
    for seed in (42, 43):
        assert json.loads((out / f"unchanged_s{seed}.json").read_text())["passed"]
        assert json.loads((out / f"routes_s{seed}.json").read_text())["actual_local_gradient"][
            "passed"
        ]
        assert json.loads((out / f"routes_s{seed}.json").read_text())[
            "historical_objective_parity"
        ]["passed"]
        scores = np.load(out / f"local_s{seed}.npz")
        assert scores["reader"].shape == (3, 4)
        assert (
            bool(np.all(scores["reader"].mean(0)[[3]] < scores["train_mean"].mean(0)[[3]]))
            == result["local_reader"][str(seed)]["values"]["endpoint"]["passed"]
        )


def test_no_cuda_persists_failure(tmp_path, monkeypatch):
    source = tmp_path / "s"
    source.mkdir()
    audit = tmp_path / "a"
    audit.mkdir()
    out = tmp_path / "o"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-only"):
        q.run(source, audit, out)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed" and status["optimizer_updates"] == 0
    assert not status["r1_authorized"] and not status["rs_authorized"]


@pytest.mark.parametrize("accepted", (True, False))
def test_completed_or_blocked_are_not_training_authorization(tmp_path, monkeypatch, accepted):
    s = tmp_path / "s"
    s.mkdir()
    a = tmp_path / "a"
    a.mkdir()
    o = tmp_path / "o"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic-only-no-real-weights")
    monkeypatch.setattr(q.previous, "code_identity", lambda: {})
    monkeypatch.setattr(
        q, "preflight", lambda *a: {"r1_preflight_passed": True, "rs_reader_qualified": accepted}
    )
    assert q.run(s, a, o) == (0 if accepted else 3)
    status = json.loads((o / "completion.json").read_text())
    assert status["status"] == ("qualification_complete_requires_review" if accepted else "blocked")
    assert not status["r1_authorized"] and status["optimizer_updates"] == 0
    with pytest.raises(ValueError, match="Nonempty"):
        q.run(s, a, o)


def test_failure_export_includes_arrays_excludes_weights(tmp_path):
    out = tmp_path / "audit"
    out.mkdir()
    atomic_json({"status": "blocked", "optimizer_updates": 0}, out / "status.json")
    np.savez(out / "local.npz", reader=np.zeros((2, 4)))
    (out / "weight.pt").write_text("excluded")
    env = {
        **os.environ,
        "BABEL_QUAL_RUN": str(out),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "PYTHON_BIN": str(Path(".venv/bin/python").resolve()),
    }
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_qualification_autodl.sh", "export"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(tmp_path / "download/audit_reports.tar.gz") as archive:
        assert "audit/local.npz" in archive.getnames()
        assert not any(n.endswith(".pt") for n in archive.getnames())
        assert (
            "run_status=blocked" in archive.extractfile("audit/export_status.txt").read().decode()
        )


def test_timeout_supervisor_preserves_prior_evidence(tmp_path, monkeypatch):
    import sys

    source = tmp_path / "s"
    source.mkdir()
    audit = tmp_path / "a"
    audit.mkdir()
    out = tmp_path / "o"
    code = (
        Path("scripts/babel_recovery_qualification_autodl.sh")
        .read_text()
        .split("<<'PY'\n", 1)[1]
        .split("\nPY\n", 1)[0]
    )
    monkeypatch.setattr(sys, "argv", ["-", str(source), str(audit), str(out)])

    def timeout(*a, **kw):
        assert kw["timeout"] == 1800
        raise subprocess.TimeoutExpired(a[0], 1800)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(SystemExit) as error:
        exec(compile(code, "<supervisor>", "exec"), {})
    assert error.value.code == 124
    saved = (out / "status.json").read_bytes()
    with pytest.raises(SystemExit):
        exec(compile(code, "<supervisor>", "exec"), {})
    assert (out / "status.json").read_bytes() == saved


def test_reader_gate_requires_each_context_and_training_support():
    class ZeroEncoder(nn.Module):
        def forward(self, x):
            return x.new_zeros((len(x), 128, 8))

    model = nn.Module()
    model.core = nn.Module()
    model.core.encoder = ZeroEncoder()
    model.local_head = nn.Linear(8, 112)
    with torch.no_grad():
        model.local_head.weight.zero_()
        model.local_head.bias.zero_()
    x = np.zeros((3, 128, 28), dtype=np.float32)
    x[..., [23, 24, 25, 27]] = 1
    y, m = q.r0.ar.ordered_targets(x)
    old = {"x": x, "y": y, "mask": m}
    local = {"mean": [0.0] * 7, "scale": [1.0] * 7}
    baseline = torch.ones((4, 16, 7))
    counts = torch.ones_like(baseline)
    result, actual, _ = q.local_reader(model, old, STATS, local, baseline, counts, "cpu")
    assert result["passed"] and not actual.any()
    counts[3, 0, 0] = 0
    result, _, _ = q.local_reader(model, old, STATS, local, baseline, counts, "cpu")
    assert not result["passed"] and not result["train_baseline_covers_validation"]


def test_existing_empty_output_is_not_reused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        q.run(source, audit, out)
    assert not list(out.iterdir())
