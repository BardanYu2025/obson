"""Synthetic-only objective routing, selection, resume and complete evaluation."""

import copy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_depth_scaling import lifecycle as depth_lifecycle
from test_history_query import fixture, make_builder
from test_overlap_diagnostic import fixture_data

from obson.babel import activity_weight as core
from obson.babel import activity_weight_evaluate as ev
from obson.babel import activity_weight_run as run
from obson.babel import history_structure as old
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def no_real_training():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def inputs():
    model, q, d, stats, local = fixture()
    b = run.parent.batch_tensors(d, "cpu")
    b = {k: torch.cat([v[:2]] * 3) for k, v in b.items()}
    ps = torch.tensor([[32, 64, 96, 128]] * 6)
    qs = ps.clone()
    sh = torch.tensor([1, 16])
    return model, q, b, ps, qs, sh, stats, local


def test_control_exact_value_and_gradients_and_detached_local():
    model, q, b, ps, qs, sh, stats, local = inputs()
    before = old.losses(model, q, b, ps, qs, sh, stats, local, "structure")["optimization_value"]
    actual = core.losses(model, q, b, ps, qs, sh, stats, local, 0.3)
    torch.testing.assert_close(actual["optimization_value"], before, atol=0, rtol=0)
    params = list(model.core.encoder.parameters()) + list(q.parameters())
    a = torch.autograd.grad(before.sum(), params, allow_unused=True, retain_graph=True)
    c = torch.autograd.grad(
        actual["optimization_value"].sum(), params, allow_unused=True, retain_graph=True
    )
    for x, y in zip(a, c, strict=True):
        if x is not None:
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    local_grads = torch.autograd.grad(
        actual["local"].sum(), list(model.core.encoder.parameters()), allow_unused=True
    )
    assert all(x is None for x in local_grads)


@pytest.mark.parametrize("weight", (0.0, 0.1))
def test_only_activity_target_gradient_changes(weight):
    model, q, b, ps, qs, sh, stats, local = inputs()
    b = {k: v.clone().requires_grad_(k == "y") for k, v in b.items()}
    parts = core.losses(model, q, b, ps, qs, sh, stats, local, weight)
    # Local diagnostic has its own activity target: isolate main query+structure.
    objective = (parts["query_weighted"] + parts["structure"]).sum()
    grad = torch.autograd.grad(objective, b["y"])[0]
    if weight == 0:
        assert torch.count_nonzero(grad[..., 2:]) == 0
    else:
        assert grad[..., 2:].abs().sum() > 0
    assert grad[..., 0].abs().sum() > 0
    shifted = {k: v.detach().clone() for k, v in b.items()}
    shifted["y"][..., 2:] += 1
    other = core.losses(model, q, shifted, ps, qs, sh, stats, local, weight)
    if weight == 0:
        torch.testing.assert_close(parts["query_weighted"], other["query_weighted"], atol=0, rtol=0)
    # Input remains all channels in every arm.
    assert torch.equal(shifted["x"], b["x"])


def test_validation_selection_ignores_activity_and_preserves_price():
    v = {
        "query_price": 1.0,
        "near_price": 1.0,
        "far": 1.0,
        "gap_price": 1.0,
        "activity": 1.0,
        "query_fixed": 1.0,
    }
    good = dict(v, near_price=0.8, far=0.9, activity=100.0, query_fixed=100.0)
    assert run.selected_value(good, v) < run.selected_value(v, v)
    bad = dict(good, near_price=1.06, activity=0.0)
    assert not np.isfinite(run.selected_value(bad, v)).all()


def lifecycle(tmp_path, stack):
    depth_meta, source, model, q, d, stats, local = depth_lifecycle(tmp_path, stack)
    model.eval()
    q.eval()
    identity = {
        "source": str(source),
        "files": {"manifest.json": run.sha256(source / "manifest.json")},
        "chosen": {str(s): {"sha256": f"seed{s}"} for s in (42, 43)},
    }
    meta = run.make_manifest(source, identity, micro=1)
    meta.update(epochs=6, budget=4, batch=2, evaluation_batch=4)
    out = tmp_path / "activity"
    out.mkdir()
    atomic_json(meta, out / "manifest.json")
    stack.enter_context(
        patch.object(
            run, "construct", side_effect=lambda *args: (copy.deepcopy(model), copy.deepcopy(q))
        )
    )
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    return meta, out, model, q, d, stats, local


@pytest.mark.parametrize("weight", (0.0, 0.1, 0.3))
def test_microbatch_gradients_and_fixed_sampling(tmp_path, weight):
    with ExitStack() as stack:
        meta, out, model, q, d, stats, local = lifecycle(tmp_path, stack)
        builder = make_builder(stats)()
        schedule = run.plan(meta, 42, 1, len(builder.plan))
        job = {"seed": 42, "activity_weight": weight}
        results = []
        for micro in (1, 2):
            a, z = copy.deepcopy(model), copy.deepcopy(q)
            opts = [torch.optim.AdamW(g) for g in run.parent.semantics.gradient_groups(a, z)]
            v, steps = run.train_epoch(
                dict(meta, micro=micro), job, a, z, builder, schedule, opts, "cpu"
            )
            results.append(
                (
                    v,
                    [
                        p.grad.clone()
                        for g in run.parent.semantics.gradient_groups(a, z)
                        for p in g
                        if p.grad is not None
                    ],
                )
            )
            assert steps == 2
        for a, b in zip(results[0][1], results[1][1], strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        for k in run.KEYS:
            assert results[0][0][k] == pytest.approx(results[1][0][k], rel=1e-6, abs=1e-6)
        schedules = [
            run.plan(meta, j["seed"], 1, len(builder.plan))[4]
            for j in meta["experiments"]
            if j["seed"] == 42
        ]
        assert all(v == schedules[0] for v in schedules)
        qs = schedule[3]
        assert not np.isin(qs, (48, 80, 112)).any()
        assert (qs[:, -1] == 128).all()


def test_gradient_diagnostic_zero_weight_no_updates(tmp_path):
    with ExitStack() as stack:
        meta, _, model, q, _, stats, _ = lifecycle(tmp_path, stack)
        before = run.parent.cpu(model)
        g = run.gradient_diagnostic(
            meta, {"seed": 42, "activity_weight": 0.0}, model, q, make_builder(stats)(), "cpu"
        )
        assert (
            g["activity_weighted_norm"] == 0
            and g["activity_raw_norm"] > 0
            and g["optimizer_steps"] == 0
        )
        assert -1.00001 <= g["cosine_price_activity"] <= 1.00001
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k])


def test_resume_keeps_optimizer_rng_and_final_checkpoint(tmp_path):
    with ExitStack() as stack:
        meta, out, model, q, d, stats, local = lifecycle(tmp_path, stack)
        name = meta["experiments"][0]["name"]
        original = run.train_epoch
        calls = 0

        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted")
            return original(*args, **kwargs)

        with (
            patch.object(run, "train_epoch", side_effect=interrupt),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            run.worker(out, name, "cpu")
        state = torch.load(out / name / "last.pt", weights_only=True)
        assert state["epoch"] == 2 and len(state["optimizers"]) == 3 and "rng" in state
        run.worker(out, name, "cpu")
        assert not (out / name / "last.pt").exists()
        assert (out / name / "e6_best.pt").exists() and (out / name / "e6_last.pt").exists()
        h = run.read_json(out / name / "history.json")
        assert [x["epoch"] for x in h] == list(range(1, 7))
        with patch.object(run, "construct", side_effect=AssertionError("No retrain")):
            run.worker(out, name, "cpu")


def test_complete_synthetic_pipeline_and_locks(tmp_path):
    with ExitStack() as stack:
        meta, out, model, q, d, stats, local = lifecycle(tmp_path, stack)
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
                atomic_json(ref, source / f"{split}_d4_c12_s{seed}_best.json")
        ev.audit_sources(meta, out, "cpu")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert (
            result["status"] == "matrix_complete_no_automatic_promotion"
            and not result["automatic_promotion"]
        )
        assert set(result["branches"]) == {"a000", "a010"}
        assert len(result["checks"]) == 736
        for split in ev.SPLITS:
            assert len(run.read_json(out / f"{split}_readouts.json")["predictions"]) == 17
        lock = run.read_json(out / "model_selection_lock.json")
        lock["weights"]["a000_s42"]["best"]["epoch"] = 999
        atomic_json(lock, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="Weight lock"):
            ev.check_models(meta, out)


def test_source_selection_and_weight_fingerprints(tmp_path):
    source = tmp_path / "completed_depth"
    source.mkdir()
    meta = {
        "schema": run.parent.SCHEMA,
        "code_sha256": run.parent.code_identity(),
        "budgets": {"d4_c12": 2},
    }
    atomic_json(meta, source / "manifest.json")
    initial = dict.fromkeys(
        ("query", "recent", "gap", "far_primary", "far_level", "far_trend"), 1.0
    )
    lock = {
        "manifest_sha256": run.sha256(source / "manifest.json"),
        "selections": {},
        "weights": {},
    }
    for seed in (42, 43):
        choices = []
        for lr in (1, 3):
            trial = f"d4_lr{lr}_s{seed}"
            (source / trial).mkdir()
            improved = dict.fromkeys(initial, 0.9 if lr == 1 else 0.8)
            atomic_json(
                {
                    "initial_validation": initial,
                    "history": [
                        {"epoch": 1, "validation": initial},
                        {"epoch": 2, "validation": improved},
                    ],
                },
                source / trial / "training_state.json",
            )
            choices.append(
                {
                    "trial": trial,
                    "lr_scale": lr,
                    "selected_epoch": 2,
                    "validation": improved,
                    "score": list(run.parent.selected_value(improved, initial)),
                }
            )
        selected = run.parent.choose_learning_rate(choices)
        name = f"d4_c12_s{seed}"
        lock["selections"][name] = dict(selected, budget=2, candidates=choices)
        rel = f"{selected['trial']}/e2_best.pt"
        (source / rel).write_bytes(b"synthetic hash-only weight")
        lock["weights"][name] = {"best": {"path": rel, "sha256": run.sha256(source / rel)}}
    atomic_json(lock, source / "model_selection_lock.json")
    done = {
        "status": "complete",
        "source_unchanged": True,
        "files": {
            str(p.relative_to(source)): run.sha256(p) for p in source.rglob("*") if p.is_file()
        },
    }
    atomic_json(done, source / "completion.json")
    identity = run.source_identity(source)
    assert identity["chosen"]["42"]["trial"] == "d4_lr3_s42"
    weight = source / identity["chosen"]["42"]["path"]
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run.source_identity(source)
    weight.write_bytes(b"synthetic hash-only weight")
    lock["selections"]["d4_c12_s42"]["selected_epoch"] = 1
    atomic_json(lock, source / "model_selection_lock.json")
    with pytest.raises(ValueError, match="selection"):
        run.source_identity(source)


def test_summary_budget_and_disk_failure_preserve_source(tmp_path):
    from types import SimpleNamespace

    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        source = Path(meta["task_source"]["source"])
        hashes = {p: run.sha256(p) for p in source.rglob("*") if p.is_file()}
        with (
            patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
            pytest.raises(ValueError, match="No source files deleted"),
        ):
            run.check_disk(meta, out, jobs=2)
        assert all(run.sha256(p) == h for p, h in hashes.items())
        job = meta["experiments"][0]
        run.worker(out, job["name"], "cpu")
        state = run.read_json(out / job["name"] / "training_state.json")
        summary = run.read_json(out / job["name"] / "training_summary.json")
        run.verify_summary(meta, state, summary)
        summary["steps"] += 1
        with pytest.raises(ValueError, match="summary"):
            run.verify_summary(meta, state, summary)


def test_shell_failure_and_manual_export_keep_failure_status(tmp_path):
    import os
    import subprocess
    import tarfile

    script = Path(__file__).resolve().parents[2] / "scripts/babel_activity_weight768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_ACTIVITY_WEIGHT768.md", "BABEL_RESEARCH_GOAL.md"):
        (root / "docs" / name).write_text("synthetic experiment note")
    out = root / "checkpoints/run"
    out.mkdir(parents=True)
    (out / "diagnostic.json").write_text('{"reason":"synthetic failure"}')
    fake = root / "fake_python"
    fake.write_text("#!/bin/sh\nexit 7\n")
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake),
        BABEL_ACTIVITY_WEIGHT_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(root / "download"),
    )
    for mode, expected in (("all", 7), ("export", 0)):
        result = subprocess.run(
            ["bash", str(root / "scripts" / script.name), mode],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected, result.stderr
        with tarfile.open(root / "download/run_reports.tar.gz") as archive:
            receipt = archive.extractfile("run/run_status.txt").read().decode()
            assert "run_status=failed" in receipt
            assert archive.extractfile("run/diagnostic.json") is not None
