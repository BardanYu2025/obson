"""Synthetic-only decoder, gradient, lifecycle and evaluation tests."""

import copy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_history_query import fixture
from test_macro_history import builder_class
from test_overlap_diagnostic import fixture_data
from test_shared_history import lifecycle as old_lifecycle

from obson.babel import linear_history as core
from obson.babel import linear_history_evaluate as ev
from obson.babel import linear_history_run as run
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def lifecycle(tmp_path, stack):
    _, source, model, query, d, stats, local, _ = old_lifecycle(tmp_path, stack)
    del query.shared
    identity = {
        "source": str(source),
        "files": {"manifest.json": run.sha256(source / "manifest.json")},
        "chosen": {str(s): {"sha256": f"seed{s}"} for s in (42, 43)},
    }
    meta = run.make_manifest(source, identity, 2)
    meta.update(epochs=6, calibration_epochs=6, budget=8, batch=8, evaluation_batch=4)
    out = tmp_path / "linear"
    out.mkdir()
    meta["run_directory"] = str(out)
    atomic_json(meta, out / "manifest.json")
    cls = builder_class(stats)
    atomic_json(
        {"rows": cls.plan, "eligible": len(cls.plan)}, run.root(meta) / "train_cross_plan.json"
    )
    stack.enter_context(patch.object(run.xr.xd, "Builder", cls))
    stack.enter_context(patch.object(run, "plan_rows", return_value=cls.plan))

    def construct(meta, seed, device, kind="native"):
        q = (
            copy.deepcopy(query)
            if kind == "native"
            else core.LinearHistory(query.state_mean, query.state_scale)
        )
        return copy.deepcopy(model), q

    stack.enter_context(patch.object(run, "construct", side_effect=construct))
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    return meta, out, model, query, d, stats, local, cls


def calibrate(meta, out):
    for seed in (42, 43):
        for kind in ("native", "linear"):
            run.worker(out, run.calibration_job(seed, kind)["name"], "cpu")
    run.calibration_check(meta, out)


def test_affine_superposition_query_subset_no_batch_information():
    q = core.LinearHistory(torch.arange(8.0) / 7, torch.ones(8) * 2)
    with torch.no_grad():
        q.weight.normal_()
        q.bias.normal_()
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    torch.testing.assert_close(q(0.3 * a + 0.7 * b), 0.3 * q(a) + 0.7 * q(b), atol=2e-6, rtol=2e-5)
    ages = torch.tensor([127, 1, 15, 79, 15])
    torch.testing.assert_close(q(a, ages), q(a)[:, ages - 1], atol=2e-6, rtol=2e-5)
    changed = a.clone()
    changed[1:] += 100
    torch.testing.assert_close(q(changed)[0], q(a)[0])
    for bad in (torch.tensor([0]), torch.tensor([128]), torch.tensor([1.0]), torch.tensor([[1]])):
        with pytest.raises(ValueError):
            q(a, bad)
    with pytest.raises(ValueError):
        core.LinearHistory(torch.zeros(8), torch.zeros(8))


def test_decoder_and_labels_causal_and_exact_resume_state():
    m, native, d, stats, _ = fixture()
    q = core.LinearHistory(native.state_mean, native.state_scale)
    with torch.no_grad():
        q.weight.normal_(std=0.01)
    x = torch.tensor(d["x"][:2])
    ps = torch.full((2, 1), 80)
    with torch.no_grad():
        _, a = core.hq.predict_states(m, q, x, ps)
        x[:, 80:] += 100
        _, b = core.hq.predict_states(m, q, x, ps)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    rebuilt = core.LinearHistory(torch.zeros(8), torch.ones(8))
    rebuilt.load_state_dict(q.state_dict(), strict=True)
    torch.testing.assert_close(rebuilt(torch.ones(2, 8)), q(torch.ones(2, 8)), atol=0, rtol=0)


@pytest.mark.parametrize("mode", core.MODES)
def test_effective_batch_gradients_and_encoder_routes(tmp_path, mode):
    with ExitStack() as stack:
        meta, _, m, native, _, stats, local, cls = lifecycle(tmp_path, stack)
        q = (
            native
            if mode == "native_joint"
            else core.LinearHistory(native.state_mean, native.state_scale)
        )
        if mode != "native_joint":
            with torch.no_grad():
                q.weight.normal_(std=0.01)
        job = {"seed": 42, "mode": mode}
        schedule = run.plan(meta, 42, 1, len(cls.plan))
        dm, dq = copy.deepcopy(m), copy.deepcopy(q)
        groups = core.configure(dm, dq, mode != "linear_frozen")
        b, ps, qs, sh = run.make_batch(cls(), schedule, 0, 8, "cpu")
        parts, _ = core.losses(dm, dq, b, ps, qs, sh, stats, local)
        parts["optimization_value"].mean().backward()
        expected = []
        for _, g, _ in groups:
            torch.nn.utils.clip_grad_norm_(g, 1.0)
            expected.extend([p.grad.clone() if p.grad is not None else None for p in g])
        for micro in (2, 4, 8):
            mm, qq = copy.deepcopy(m), copy.deepcopy(q)
            _, steps = run.train_epoch(
                dict(meta, micro=micro), job, mm, qq, cls(), schedule, None, "cpu"
            )
            actual = [p for _, g, _ in core.configure(mm, qq, mode != "linear_frozen") for p in g]
            for p, e in zip(actual, expected, strict=True):
                if e is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad, e, atol=3e-5, rtol=4e-4)
            assert steps == 1
            if mode == "linear_frozen":
                assert all(p.grad is None for p in mm.core.encoder.parameters())
            else:
                assert any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in mm.core.encoder.parameters()
                )
        assert all(p.grad is None for p in dm.core.decoder.parameters())


def test_real_synthetic_updates_resume_and_all_locks(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        original = run.train_epoch
        calls = 0

        def stop(*args, **kw):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted")
            return original(*args, **kw)

        with (
            patch.object(run, "train_epoch", side_effect=stop),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            run.worker(out, "cal_linear_s42", "cpu")
        state = torch.load(out / "cal_linear_s42/last.pt", weights_only=True)
        assert state["epoch"] == 1 and len(state["optimizers"]) == 2
        calibrate(meta, out)
        calls = 0
        with (
            patch.object(run, "train_epoch", side_effect=stop),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            run.worker(out, "linear_joint_s42", "cpu")
        joint_resume = torch.load(out / "linear_joint_s42/last.pt", weights_only=True)
        assert joint_resume["epoch"] == 1 and len(joint_resume["optimizers"]) == 3
        for job in meta["experiments"]:
            run.worker(out, job["name"], "cpu")
        run.lock_models(meta, out)
        lock = ev.check_models(meta, out)
        assert len(lock["weights"]) == 6
        joint = lock["trials"]["linear_joint_s42"]
        frozen = lock["trials"]["linear_frozen_s42"]
        assert joint["encoder_start_sha256"] != joint["encoder_end_sha256"]
        assert frozen["encoder_start_sha256"] == frozen["encoder_end_sha256"]
        with patch.object(run, "construct", side_effect=AssertionError("No rerun")):
            run.worker(out, "linear_joint_s42", "cpu")
        state = run.read_json(out / "linear_joint_s42/training_state.json")
        state["history"][0]["sampling"]["ids"] = "changed"
        with pytest.raises(ValueError, match="Sampling"):
            run.verify_history(meta, meta["experiments"][2], state, 8)


def test_complete_original_utility_pipeline(tmp_path):
    with ExitStack() as stack:
        meta, out, model, q, d, stats, local, _ = lifecycle(tmp_path, stack)
        calibrate(meta, out)
        for job in meta["experiments"]:
            run.worker(out, job["name"], "cpu")
        run.lock_models(meta, out)
        source = Path(meta["task_source"]["source"])
        raw = run.root(meta)
        y, mask = ev.up.targets(ev.up.restore_raw(d["x"], stats))
        scales = ev.up.target_scales(y, mask)
        atomic_json({"target_stats": scales}, raw / "fit.json")
        (raw / "cache").mkdir(exist_ok=True)
        for split in ("train", "val") + ev.SPLITS:
            np.savez(raw / f"cache/{split}.npz", targets=y, mask=mask)
        ev.fit_readouts(meta, out, "cpu")
        assert run.read_json(out / "fit.json")["candidates"] == 780
        with patch.object(ev.up, "fit_heads", side_effect=AssertionError("No refit")):
            ev.fit_readouts(meta, out, "cpu")
        _, _, _, paired = fixture_data(stats)
        config = {
            "minimum_utility_r2": 0,
            "raw_retention": 0.05,
            "family_retention": 0.1,
            "baseline_gain": 0.05,
            "current_nmse": 0.1,
            "current_r2": 0.8,
            "min_windows": 50,
            "min_weeks": 5,
        }
        cm = {
            "source": str(source),
            "identity": {},
            "packed": {s: {} for s in run.xr.cr.odr.SPLITS},
            "reader": {"manifest": {"decision": config}},
        }
        stack.enter_context(patch.object(run.xr, "parent_meta", return_value=cm))
        stack.enter_context(patch.object(run.xr.cr.odr, "make_manifest", return_value={}))
        stack.enter_context(
            patch.object(
                run.xr.cr.odr, "prepare", return_value=(dict.fromkeys(ev.SPLITS, paired), {})
            )
        )
        model.eval().requires_grad_(False)
        q.eval().requires_grad_(False)
        qscore, _ = ev.de.query_scores(meta, model, q, d, "cpu")
        matched = ev.de.matched_scores(meta, model, q, d, "cpu")
        score, err = ev.up.measure(y, y, mask, scales)
        ref = {
            "query": qscore,
            "matched": matched,
            "query_overlap": ev.de.query_overlap(meta, model, q, paired, "cpu"),
            "utility": {
                "scores": score,
                "errors": {k: v.tolist() for k, v in err.items()},
                "predictions": y.tolist(),
            },
        }
        for split in ev.SPLITS:
            atomic_json(
                [
                    {"week": f"w{i}", "symbol": "A", "period": 15, "month": "2026-01"}
                    for i in range(len(y))
                ],
                raw / f"{split}_inventory.json",
            )
            atomic_json(
                {
                    "predictions": {n: y.tolist() for n in ("raw3584", "pca768", "current28")},
                    "scores": dict.fromkeys(("raw3584", "pca768", "current28"), score),
                },
                source / f"{split}_readouts.json",
            )
            for seed in (42, 43):
                atomic_json(ref, source / f"{split}_control_s{seed}_best.json")
        ev.audit_sources(meta, out, "cpu")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert (
            result["status"] == "matrix_complete_no_automatic_promotion"
            and not result["automatic_promotion"]
        )
        assert set(result["branches"]) == {"linear_joint"}
        assert len(result["checks"]) == 512
        for split in ev.SPLITS:
            assert len(run.read_json(out / f"{split}_readouts.json")["predictions"]) == 17
        lock = run.read_json(out / "model_selection_lock.json")
        lock["weights"]["native_joint_s42"]["best"]["epoch"] = 999
        atomic_json(lock, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="Weight lock"):
            ev.check_models(meta, out)


@pytest.mark.parametrize("code,status", [(3, "blocked"), (7, "failed")])
def test_shell_exports_blocked_and_failed_run_and_preserves_receipt(tmp_path, code, status):
    import os
    import subprocess
    import tarfile

    script = Path(__file__).resolve().parents[2] / "scripts/babel_linear_history768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_LINEAR_HISTORY768.md", "BABEL_RESEARCH_GOAL.md"):
        (root / "docs" / name).write_text("synthetic protocol")
    out = root / "checkpoints/run"
    out.mkdir(parents=True)
    (out / "diagnostic.json").write_text('{"synthetic":true}')
    fake = root / "python"
    fake.write_text(f"#!/bin/sh\nexit {code}\n")
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake),
        BABEL_LINEAR_HISTORY_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(root / "download"),
    )
    for mode, expected in (("all", code), ("export", 0)):
        result = subprocess.run(
            ["bash", str(root / "scripts" / script.name), mode],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected, result.stderr
        with tarfile.open(root / "download/run_reports.tar.gz") as archive:
            receipt = archive.extractfile("run/run_status.txt").read().decode()
            assert f"run_status={status}" in receipt
            assert archive.extractfile("run/diagnostic.json") is not None


def test_decision_requires_utility_against_both_controls_and_information_retention():
    n = 60

    def rec(value):
        def values():
            return [value] * n

        families = {k: values() for k in ("utility", "direction", "volatility", *ev.up.NAMES)}
        errors = {k: values() for k in ("path", "change1", "body", "activity")}
        return {
            "utility": {
                "errors": families,
                "scores": {"targets": [{"r2": 0.99} for _ in range(13)]},
            },
            "matched": {"paired80_128": {"actual_error": errors}},
            "query": {
                "native": {"errors": {"held": errors, "p128": errors}},
                "structure": {"far65": {k: values() for k in ("primary", "level", "trend")}},
            },
            "query_overlap": {str(s): {"combined_gap": values()} for s in (1, 16, 64)},
        }

    meta = {
        "experiments": [
            {"name": f"{mode}_s{s}", "mode": mode, "seed": s}
            for s in (42, 43)
            for mode in core.MODES
        ]
    }
    records = {
        split: {
            name: rec(0.04 if name.startswith("linear_joint") else 0.06)
            for name in [n for n, _, _ in ev.entries(meta)] + [f"source_s{s}" for s in (42, 43)]
        }
        for split in ev.SPLITS
    }
    rows = [
        {"week": f"w{i // 6}", "symbol": "A", "period": 15, "month": "2026-01"} for i in range(n)
    ]
    inventories = dict.fromkeys(ev.SPLITS, rows)
    masks = dict.fromkeys(ev.SPLITS, np.ones((n, 13), bool))
    result = ev.decide(meta, records, inventories, inventories, masks, {})
    assert result["branches"]["linear_joint"]["status"] == "balanced_utility_candidate"
    assert len(result["checks"]) == 512
    for split in ev.SPLITS:
        for seed in (42, 43):
            for kind in ("best", "last"):
                records[split][f"linear_frozen_s{seed}_{kind}"]["utility"]["errors"]["utility"] = [
                    0.04
                ] * n
    result = ev.decide(meta, records, inventories, inventories, masks, {})
    assert not result["branches"]["linear_joint"]["groups"]["utility_gain"]
    assert all(
        not c["passed"]
        for c in result["checks"]
        if c["metric"] == "utility/linear_frozen" and c["group"] == "utility_gain"
    )


def test_main_orchestration_calibration_precedes_joint_and_reentry_is_readonly(tmp_path):
    with ExitStack() as stack:
        meta, _, *_ = lifecycle(tmp_path, stack)
        source = Path(meta["task_source"]["source"])
        identity = meta["task_source"]
        sm = run.read_json(source / "manifest.json")
        sm["task_source"] = {"source": str(source)}
        atomic_json(sm, source / "manifest.json")
        atomic_json(
            {"torch": str(torch.__version__), "numpy": np.__version__}, source / "runtime.json"
        )
        out = tmp_path / "orchestration"
        stack.enter_context(patch.object(run, "source_identity", return_value=identity))
        stack.enter_context(patch.object(run, "make_manifest", return_value=copy.deepcopy(meta)))
        stack.enter_context(patch.object(run, "check_disk"))
        stack.enter_context(patch.object(run.parent.warm, "check_output"))
        stack.enter_context(patch.object(run.xr.xd, "verify_data"))
        stack.enter_context(patch.object(ev, "audit_sources"))
        stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="synthetic"))
        stack.enter_context(patch.object(torch.cuda, "empty_cache"))
        stack.enter_context(patch.object(run, "preflight", return_value=1))
        trace = []
        dispatch = stack.enter_context(
            patch.object(
                run,
                "dispatch",
                side_effect=lambda *a, **k: trace.append("cal" if k.get("calibration") else "main"),
            )
        )
        stack.enter_context(
            patch.object(run, "calibration_check", side_effect=lambda *a: trace.append("verify"))
        )
        stack.enter_context(patch.object(run, "lock_models"))
        fit = stack.enter_context(patch.object(ev, "fit_readouts"))
        stack.enter_context(patch.object(ev, "evaluate"))
        run.run(source, out, jobs=2, micro=2)
        assert trace == ["cal", "verify", "main", "verify"]
        assert dispatch.call_args.args[-1] == 1
        run.run(source, out, jobs=2, micro=2)
        assert dispatch.call_count == 2 and fit.call_count == 1


def test_construct_restores_selected_source_before_switching_reader(tmp_path):
    m, q, _, _, _ = fixture()
    source = tmp_path / "macro"
    source.mkdir()
    atomic_json({"synthetic": True}, source / "manifest.json")
    ck = {
        "epoch": 97,
        "metadata": {"manifest_sha256": run.sha256(source / "manifest.json")},
        "model": run.parent.cpu(m),
        "query": run.parent.cpu(q),
    }
    with torch.no_grad():
        ck["query"]["output.bias"].add_(2)
    torch.save(ck, source / "best.pt")
    identity = {
        "source": str(source),
        "chosen": {
            "42": {"path": "best.pt", "sha256": run.sha256(source / "best.pt"), "epoch": 97}
        },
    }
    with patch.object(
        run.base, "construct", side_effect=lambda *a: (copy.deepcopy(m), copy.deepcopy(q))
    ):
        model, native = run.construct({"task_source": identity}, 42, "cpu")
        assert torch.equal(native.output.bias, ck["query"]["output.bias"])
        _, linear = run.construct({"task_source": identity}, 42, "cpu", "linear")
        assert isinstance(linear, core.LinearHistory) and torch.count_nonzero(linear.weight) == 0
        assert run.ur.bb.state_signature(m) == run.ur.bb.state_signature(model)
        (source / "best.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="changed"):
            run.construct({"task_source": identity}, 42, "cpu")


def test_completed_source_binds_only_selected_weights_and_reports(tmp_path):
    source = tmp_path / "macro"
    source.mkdir()
    original = {"readout_audit": str(tmp_path / "audit"), "other": "immutable"}
    meta = {
        "schema": run.base.SCHEMA,
        "code_sha256": run.base.code_identity(),
        "task_source": original,
    }
    atomic_json(meta, source / "manifest.json")
    lock = {"manifest_sha256": run.sha256(source / "manifest.json"), "weights": {}}
    for seed in (42, 43):
        name = f"control_s{seed}"
        (source / name).mkdir()
        rel = f"{name}/e100_best.pt"
        (source / rel).write_bytes(f"synthetic selected {seed}".encode())
        lock["weights"][name] = {
            "best": {"path": rel, "sha256": run.sha256(source / rel), "epoch": 97}
        }
    atomic_json(lock, source / "model_selection_lock.json")
    files = {str(p.relative_to(source)): run.sha256(p) for p in source.rglob("*") if p.is_file()}
    atomic_json(
        {"status": "complete", "source_unchanged": True, "files": files}, source / "completion.json"
    )
    with patch.object(run.base, "source_identity", return_value=original):
        result = run.source_identity(source)
        assert set(result["chosen"]) == {"42", "43"}
        (source / "control_s42/e100_best.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            run.source_identity(source)
