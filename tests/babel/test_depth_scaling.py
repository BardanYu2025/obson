"""Synthetic gradients, retained warm provenance, matched targets and full lifecycle."""

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_history_query import SMALL, Capture, make_builder
from test_history_query import fixture as base_fixture
from test_overlap_diagnostic import fixture_data

from obson.babel import depth_scaling as core
from obson.babel import depth_scaling_evaluate as ev
from obson.babel import depth_scaling_run as run
from obson.babel import history_query as hq
from obson.babel.ae_extend import atomic_json


def fixture():
    model, query, d, stats, local = base_fixture()
    run.xr.growth.install(model, 4, 16, 42)
    return model, query, d, stats, local


@pytest.fixture(autouse=True)
def no_neural_updates():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def lifecycle(tmp_path, stack):
    parent, query, d, stats, local = fixture()
    builder_type = make_builder(stats)
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    identity = {"manifest": {"identity": {"training_manifest": {}, "training_source": str(source)}}}
    start_files = {}
    for seed in (42, 43):
        for mode in ("plain",):
            p = source / f"{mode}_s{seed}/best.pt"
            p.parent.mkdir()
            torch.save({"model": run.cpu(parent), "query": run.cpu(query)}, p)
            start_files[f"{mode}_s{seed}/best.pt"] = run.sha256(p)
    meta = run.make_manifest(
        source,
        identity,
        {"source": str(source), "files": {f"warm_s{s}/best.pt": f"seed{s}" for s in (42, 43)}},
        {"source": str(source), "files": start_files},
        16,
    )
    meta.update(
        epochs=6,
        budgets={"d4": 6, "d8": 6, "d12": 6, "d4_c8": 12, "d4_c12": 18},
        budget=4,
        batch=2,
        micro=1,
        evaluation_batch=4,
        query_config=SMALL,
    )
    atomic_json(meta, out / "manifest.json")
    atomic_json({"eligible": 8}, source / "train_cross_plan.json")
    stack.enter_context(
        patch.object(
            run,
            "construct",
            side_effect=lambda meta, seed, device, mode="d4": (
                core.install(copy.deepcopy(parent), int(mode[1:]), seed),
                copy.deepcopy(query),
            ),
        )
    )
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    stack.enter_context(patch.object(run.xr.xd, "Builder", builder_type))
    stack.enter_context(patch.object(run.xr, "validation", return_value={"selection": 1.0}))
    return meta, out, parent, query, d, stats, local


def test_complete_synthetic_fit_evaluate_pipeline(tmp_path):
    with ExitStack() as stack:
        meta, out, parent, query, d, stats, local = lifecycle(tmp_path, stack)
        source = run.root(meta)
        parent_source = source / "parent_reference"
        parent_source.mkdir()
        meta["identity"]["manifest"]["source"] = str(parent_source)
        atomic_json(meta, out / "manifest.json")
        for job in meta["experiments"]:
            run.worker(out, job["name"], "main", "cpu")
        run.lock_models(meta, out)
        truth, mask = ev.up.targets(ev.up.restore_raw(d["x"], stats))
        target_stats = ev.up.target_scales(truth, mask)
        atomic_json({"target_stats": target_stats}, source / "fit.json")
        (source / "cache").mkdir()
        for split in ("train", "val") + ev.SPLITS:
            np.savez(source / f"cache/{split}.npz", targets=truth, mask=mask)
        ev.fit_readouts(meta, out, "cpu")
        assert run.read_json(out / "fit.json")["candidates"] == 1300
        with patch.object(ev.up, "fit_heads", side_effect=AssertionError("No repeated fit")):
            ev.fit_readouts(meta, out, "cpu")
        z = ev.states(parent, d, 4, "cpu")
        heads, w, bias = ev.up.fit_heads(z, truth, mask, z, truth, mask, target_stats, "cpu")
        utility = {
            "heads": heads,
            "weights": w.tolist(),
            "intercepts": bias.tolist(),
            "target_stats": target_stats,
        }
        pred = ev.up.predict(heads, w, bias, z, target_stats)
        u, ue = ev.up.measure(pred, truth, mask, target_stats)
        score, error, _, _ = run.ur.pf.score(parent, d, stats, local, 4, "cpu")
        _, _, _, paired = fixture_data(stats)
        overlap = run.xr.ce.overlap(parent, paired, stats, 4, "cpu")
        previous = {
            "reconstruction": {
                "scores": score,
                "errors": {k: {n: v.tolist() for n, v in e.items()} for k, e in error.items()},
            },
            "utility": {
                "scores": u,
                "errors": {k: v.tolist() for k, v in ue.items()},
                "predictions": pred.tolist(),
            },
            "overlap": overlap,
        }
        baseline, _ = ev.up.measure(truth, truth, mask, target_stats)
        for split in ev.SPLITS:
            atomic_json(
                [
                    {"week": f"w{i}", "symbol": "A", "period": 15, "month": "2026-01"}
                    for i in range(len(z))
                ],
                source / f"{split}_inventory.json",
            )
            atomic_json(
                {"predictions": {n: truth.tolist() for n in ("raw3584", "pca768", "current28")}},
                parent_source / f"{split}_predictions.json",
            )
            for seed in (42, 43):
                atomic_json(previous, parent_source / f"{split}_control_s{seed}_best.json")
        atomic_json(
            {
                "datasets": {
                    s: {"scores": dict.fromkeys(("raw3584", "pca768", "current28"), baseline)}
                    for s in ev.SPLITS
                }
            },
            parent_source / "readout_metrics.json",
        )
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
        stack.enter_context(
            patch.object(
                run.rs,
                "load_bundle",
                side_effect=lambda *a: SimpleNamespace(
                    aligned=copy.deepcopy(parent), utility=utility
                ),
            )
        )
        parent.eval().requires_grad_(False)
        query.eval().requires_grad_(False)
        qs, _ = ev.query_scores(meta, parent, query, d, "cpu")
        reference = {
            "original": previous["reconstruction"],
            "utility": previous["utility"],
            "overlap": previous["overlap"],
            "query": qs,
            "matched": ev.matched_scores(meta, parent, query, d, "cpu"),
            "query_overlap": ev.query_overlap(meta, parent, query, paired, "cpu"),
            "structure_overlap": ev.structure_overlap(meta, parent, query, paired, "cpu"),
        }
        for split in ev.SPLITS:
            for seed in (42, 43):
                for mode in ("plain",):
                    atomic_json(reference, source / f"{split}_{mode}_s{seed}_best.json")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert result["status"] == "matrix_complete_no_automatic_promotion"
        assert all(b["status"] == "no_balanced_gain" for b in result["branches"].values())
        assert len(result["checks"]) == 1800
        assert not result["automatic_promotion"]
        for split in ev.SPLITS:
            for mode in meta["budgets"]:
                for seed in (42, 43):
                    rec = run.read_json(out / f"{split}_{mode}_s{seed}_best.json")
                    assert set(rec["matched"]["lengths"]) == {"32", "64", "80", "128"}
                    assert len(rec["query"]["oldest_endpoint"]["primary"]) == len(d["x"])


@pytest.mark.parametrize("mode", ("d4", "d8", "d12"))
def test_microbatch_accumulation(mode):
    parent, q, _, stats, local = fixture()
    builder = make_builder(stats)()
    schedule = (
        np.array([0, 1]),
        np.array([1, 16]),
        np.array([[32, 64, 96, 128]] * 2),
        hq.sample_prefixes(2, 42, 1),
        {},
    )
    runs = []
    for micro in (1, 2):
        model, query = core.install(copy.deepcopy(parent), int(mode[1:]), 42), copy.deepcopy(q)
        model.requires_grad_(False)
        model.core.encoder.requires_grad_(True)
        model.local_head.requires_grad_(True)
        opts = [Capture(v.parameters()) for v in (model.core.encoder, model.local_head, query)]
        _, steps = run.train_epoch(
            model, query, builder, schedule, stats, local, 2, micro, "cpu", opts, mode
        )
        assert steps == 1
        runs.append(opts)
    for a, b in zip(*runs, strict=True):
        for x, y in zip(a.gradients, b.gradients, strict=True):
            if x is not None:
                torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-4)


def test_resume_budgets_shared_fork_epoch0_and_tamper(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        original = run.train_epoch
        calls = []

        def interrupt(*a, **kw):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("synthetic interruption")
            return original(*a, **kw)

        with patch.object(run, "train_epoch", side_effect=interrupt), pytest.raises(RuntimeError):
            run.worker(out, "d4_lr1_s42", "main", "cpu")
        assert torch.load(out / "d4_lr1_s42/last.pt", weights_only=True)["epoch"] == 2
        for job in meta["experiments"]:
            run.worker(out, job["name"], "main", "cpu")
        lock = run.lock_models(meta, out)
        assert len(lock["selections"]) == 10
        for name, selected in lock["selections"].items():
            assert selected["selected_epoch"] == 0 and selected["lr_scale"] == 1
            assert selected["budget"] == meta["budgets"][name.rsplit("_s", 1)[0]]
        for seed in (42, 43):
            h = [
                run.read_json(out / f"d{d}_lr{lr}_s{seed}/history.json")[:6]
                for d in core.DEPTHS
                for lr in core.LR_SCALES
            ]
            assert all([v["sampling"] for v in x] == [v["sampling"] for v in h[0]] for x in h)
        with patch.object(run, "train_epoch", side_effect=AssertionError("No retraining")):
            run.worker(out, "d4_lr1_s42", "main", "cpu")
        (out / lock["weights"]["d8_s43"]["best"]["path"]).write_bytes(b"tamper")
        with pytest.raises(ValueError):
            ev.check_models(meta, out)


def test_matched_endpoint_targets_and_oldest_band():
    model, query, d, stats, local = fixture()
    with patch.object(run, "stats", return_value=(stats, local)):
        report = ev.matched_scores({"evaluation_batch": 4}, model, query, d, "cpu")
        full, _ = ev.query_scores({"evaluation_batch": 4}, model, query, d, "cpu")
    assert set(report["lengths"]) == {"32", "64", "80", "128"}
    np.testing.assert_allclose(
        report["lengths"]["128"]["errors"]["primary"],
        full["recent"]["errors"]["p128"]["primary"],
        atol=1e-6,
        rtol=2e-5,
    )
    assert len(full["oldest_endpoint"]["primary"]) == len(d["x"])
    b = run.batch_tensors(d, "cpu")
    ps = torch.full((len(d["x"]), 1), 128)
    y, m = hq.targets(b, ps, stats)
    m[:, :, :64] = False
    altered = y.clone()
    altered[:, :, :64] += 100
    assert torch.all(hq.metrics(altered, y, m, stats)["primary"] == 0)


def test_export_preserves_failure_and_omits_weights(tmp_path):
    import os
    import subprocess
    import tarfile
    from pathlib import Path

    out = tmp_path / "out"
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    (out / "best.pt").write_bytes(b"not a report")
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake),
        BABEL_DEPTH_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_depth768_autodl.sh"
    result = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert result.returncode == 7 and "run_status=failed" in result.stdout
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and "run_status=failed" in result.stdout
    with tarfile.open(tmp_path / "download/out_reports.tar.gz") as archive:
        assert not any(n.endswith(".pt") for n in archive.getnames())


def test_source_identity_retained_best_without_last_and_strict_restore(tmp_path):
    parent, query, _, _, _ = fixture()
    source = tmp_path / "old"
    source.mkdir()
    identity = {"fixture": True}
    meta = run.source_run.make_manifest(source, identity, {}, {"source": str(source)}, 16)
    atomic_json(meta, source / "manifest.json")
    trials = {}
    for seed in (42, 43):
        for mode in ("plain",):
            name = f"{mode}_s{seed}"
            path = source / name
            path.mkdir()
            validation = {"query": 1.0}
            torch.save(
                {
                    "metadata": {
                        "manifest_sha256": run.sha256(source / "manifest.json"),
                        "job": {"name": name, "mode": mode, "seed": seed},
                        "phase": "main",
                    },
                    "epoch": 100,
                    "validation": validation,
                    "model": run.cpu(parent),
                    "query": run.cpu(query),
                },
                path / "best.pt",
            )
            trials[name] = {"epochs": 100, "selected_epoch": 100, "validation": validation}
    atomic_json({"trials": trials}, source / "model_selection_lock.json")
    files = {str(p.relative_to(source)): run.sha256(p) for p in source.rglob("*") if p.is_file()}
    atomic_json(
        {"status": "complete", "source_unchanged": True, "files": files}, source / "completion.json"
    )
    with patch.object(run.source_run, "start_identity", return_value=meta["start"]):
        start = run.start_identity(source, identity)
    assert not list(source.rglob("last.pt"))
    new = run.make_manifest(source, identity, {}, start)
    with patch.object(
        run.previous,
        "construct",
        side_effect=lambda *args: (copy.deepcopy(parent), copy.deepcopy(query)),
    ):
        restored, q = run.construct(new, 42, "cpu")
    assert run.ur.bb.state_signature(restored) == run.ur.bb.state_signature(parent)
    assert run.ur.bb.state_signature(q) == run.ur.bb.state_signature(query)
    (source / "plain_s42/best.pt").write_bytes(b"changed")
    with (
        patch.object(run.source_run, "start_identity", return_value=meta["start"]),
        pytest.raises(ValueError),
    ):
        run.start_identity(source, identity)
    with (
        patch.object(run.previous, "construct", side_effect=lambda *args: (parent, query)),
        pytest.raises(ValueError),
    ):
        run.construct(new, 42, "cpu")


def test_source_replay_and_zero_update_backward_preflight(tmp_path):
    with ExitStack() as stack:
        meta, out, parent, query, d, stats, local = lifecycle(tmp_path, stack)
        source = run.root(meta)
        atomic_json(
            {
                "trials": {
                    f"plain_s{s}": {"validation": {"original": {"selection": 1.0}}}
                    for s in (42, 43)
                }
            },
            source / "model_selection_lock.json",
        )
        with patch.object(
            torch.optim.AdamW, "__init__", side_effect=AssertionError("No optimizer in preflight")
        ):
            run.preflight(meta, out, "cpu")
        report = run.read_json(out / "gradient_preflight.json")
        assert report["optimizer_steps"] == 0 and len(report["gradients"]) == 3
        parent.eval().requires_grad_(False)
        query.eval().requires_grad_(False)
        q, _ = ev.query_scores(meta, parent, query, d, "cpu")
        reference = {"query": q, "matched": ev.matched_scores(meta, parent, query, d, "cpu")}
        for split in ev.SPLITS:
            for seed in (42, 43):
                for mode in ("plain",):
                    atomic_json(reference, source / f"{split}_{mode}_s{seed}_best.json")
        checks = ev.audit_sources(meta, out, "cpu")
        assert len(checks) == 4 and all(c["passed"] for c in checks)
        reference["matched"]["paired80_128"]["gap"]["primary"][0] += 1
        atomic_json(reference, source / "test_plain_s42_best.json")
        with pytest.raises(ValueError, match="before training"):
            ev.audit_sources(meta, out, "cpu")
        assert not run.read_json(out / "source_replay_preflight.json")[-1]["passed"]


def test_selection_minimax_guards_and_val_does_not_query_held80(tmp_path):
    initial = dict.fromkeys(
        ("query", "recent", "gap", "far_primary", "far_level", "far_trend"), 4.0
    )
    assert run.selected_value(initial, initial) == (1.0, 1.0)
    assert run.selected_value(dict(initial, recent=3.0, far_primary=2.0), initial) == (0.75, 0.625)
    assert np.isinf(run.selected_value(dict(initial, recent=4.3), initial)[0])
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            run.selected_value(initial, dict(initial, gap=bad))
    with ExitStack() as stack:
        meta, _, parent, query, _, _, _ = lifecycle(tmp_path, stack)
        original = ev.matched_scores
        called = []

        def capture(*args, **kwargs):
            called.append(kwargs)
            return original(*args, **kwargs)

        with patch.object(ev, "matched_scores", side_effect=capture):
            values = run.validation(meta, parent, query, "cpu")
        assert called == [{"lengths": (64, 128), "pair": (64, 128)}]
        assert values["gap"] > 0 and values["recent"] > 0


@pytest.mark.parametrize("depth", core.DEPTHS)
def test_identity_growth_and_live_causality(depth):
    parent, query, d, stats, local = fixture()
    before = run.cpu(parent)
    rng = torch.random.get_rng_state().clone()
    model = core.install(copy.deepcopy(parent), depth, 42)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert len(model.core.encoder.backbone.layers) == depth
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())
    x = torch.tensor(d["x"])
    with torch.no_grad():
        torch.testing.assert_close(model.core.encoder(x), parent.core.encoder(x), atol=0, rtol=0)
        for layer in list(model.core.encoder.backbone.layers)[4:]:
            layer.self_attn.out_proj.weight.fill_(0.01)
            layer.linear2.weight.fill_(0.02)
    model.eval().requires_grad_(False)
    original = model.core.encoder(x)
    changed = x.clone()
    changed[:, 80:] += 5
    torch.testing.assert_close(
        model.core.encoder(changed)[:, :80], original[:, :80], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        model.core.encoder(x[:, :80]), original[:, :80], atol=2e-5, rtol=2e-5
    )
    appended = torch.cat([x, torch.ones_like(x[:, :16]) * 10], 1)
    torch.testing.assert_close(
        model.core.encoder(appended)[:, :128], original, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(model.core.encoder(x[:1]), original[:1], atol=2e-5, rtol=2e-5)


def test_new_layers_learn_and_common_layers_initialize_identically():
    parent, query, d, stats, local = fixture()
    a = core.install(copy.deepcopy(parent), 8, 42)
    b = core.install(copy.deepcopy(parent), 12, 42)
    for i in range(8):
        assert all(
            torch.equal(v, b.core.encoder.backbone.layers[i].state_dict()[k])
            for k, v in a.core.encoder.backbone.layers[i].state_dict().items()
        )
    x = torch.tensor(d["x"])
    a.core.encoder(x).square().mean().backward()
    for layer in list(a.core.encoder.backbone.layers)[4:]:
        assert layer.self_attn.out_proj.weight.grad.abs().sum() > 0
        assert layer.linear2.weight.grad.abs().sum() > 0
        assert layer.self_attn.in_proj_weight.grad.abs().sum() == 0
    a.zero_grad(set_to_none=True)
    with torch.no_grad():
        for layer in list(a.core.encoder.backbone.layers)[4:]:
            layer.self_attn.out_proj.weight.fill_(0.01)
            layer.linear2.weight.fill_(0.01)
    a.core.encoder(x).square().mean().backward()
    assert all(
        layer.self_attn.in_proj_weight.grad.abs().sum() > 0
        for layer in list(a.core.encoder.backbone.layers)[4:]
    )


def test_compute_budgets_shared_schedule_and_lr_restarts():
    b = core.budgets()
    assert b == {"d4": 100, "d8": 100, "d12": 100, "d4_c8": 191, "d4_c12": 281}
    for depth in (8, 12):
        target = 100 * core.matmul_proxy(depth)
        actual = b[f"d4_c{depth}"] * core.matmul_proxy(4)
        assert target <= actual < target + core.matmul_proxy(4)
    meta = {"budgets": b, "encoder_lr": 1e-5, "head_lr": 3e-5, "query_lr": 3e-4}
    for epoch in (1, 5, 50, 100):
        rates = [run.learning_rates(meta, {"depth": d, "lr_scale": 1}, epoch) for d in core.DEPTHS]
        assert rates[0] == rates[1] == rates[2]
    for epoch in (101, 192):
        assert run.learning_rates(meta, {"depth": 4, "lr_scale": 1}, epoch)[
            "encoder_lr"
        ] == pytest.approx(2e-6)
    for depth in core.DEPTHS:
        a = run.learning_rates(meta, {"depth": depth, "lr_scale": 1}, 50)
        c = run.learning_rates(meta, {"depth": depth, "lr_scale": 3}, 50)
        assert c["encoder_lr"] == pytest.approx(3 * a["encoder_lr"])
        assert c["query_lr"] == a["query_lr"] and c["head_lr"] == a["head_lr"]
    with pytest.raises(ValueError):
        run.learning_rates(meta, {"depth": 8, "lr_scale": 1}, 101)


def test_budget_snapshot_is_immutable_and_can_finish_partial_write(tmp_path):
    state = {
        "metadata": {"source": "same"},
        "best_epoch": 3,
        "epoch": 6,
        "best_validation": {"query": 1},
        "history": [{"validation": {"query": 2}}],
        "best_model": {"p": torch.ones(2)},
        "model": {"p": torch.ones(2) * 2},
        "best_query": {"p": torch.ones(2)},
        "query": {"p": torch.ones(2) * 3},
    }
    original = run.atomic_save

    def fail_last(value, path):
        if path.name.endswith("_last.pt"):
            raise OSError("interrupted disk write")
        original(value, path)

    with patch.object(run, "atomic_save", side_effect=fail_last), pytest.raises(OSError):
        run.snapshot(state, tmp_path, 6)
    first = run.sha256(tmp_path / "e6_best.pt")
    run.snapshot(state, tmp_path, 6)
    assert run.sha256(tmp_path / "e6_best.pt") == first
    assert torch.load(tmp_path / "e6_last.pt", weights_only=True)["epoch"] == 6
    bad = copy.deepcopy(state)
    bad["best_model"]["p"] += 1
    with pytest.raises(ValueError, match="weights changed"):
        run.snapshot(bad, tmp_path, 6)


def test_disk_budget_includes_shallow_milestones_and_no_auto_delete(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        estimate = run.storage_budget(meta, 2)
        for x, j in zip(estimate["checkpoints"], meta["experiments"], strict=True):
            assert x["milestone_bytes"] == 2 * len(core.stages(meta, j["depth"])) * x["state_bytes"]
        assert estimate["atomic_temporary_bytes"] == 2 * max(
            2 * x["last_bytes"] + x["best_bytes"] for x in estimate["checkpoints"]
        )
        before = set(out.iterdir())
        with (
            patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
            pytest.raises(ValueError),
        ):
            run.check_disk(out, 2, meta)
        assert not run.read_json(out / "disk_preflight.json")["passed"]
        assert before.issubset(set(out.iterdir()))


def test_depth_requires_exposure_and_compute_not_one_lucky_seed():
    rows = [
        {"week": f"w{i // 10}", "month": "2026-01", "symbol": "A", "period": 15} for i in range(60)
    ]
    errors = {k: [1.0] * 60 for k in ("primary", "path", "body", "activity", "change1")}
    utility = {k: [1.0] * 60 for k in ("utility", "direction", "volatility")}
    utility.update({k: [0.01] * 60 for k in ev.up.NAMES[6:]})
    sample = {
        "query": {
            "native": {"errors": {k: copy.deepcopy(errors) for k in ("held", "p128")}},
            "oldest_endpoint": copy.deepcopy(errors),
            "structure": {"far65": {k: [1.0] * 60 for k in ("primary", "level", "trend")}},
        },
        "matched": {"paired80_128": {k: copy.deepcopy(errors) for k in ("gap", "actual_error")}},
        "query_overlap": {str(n): {"combined_gap": [1.0] * 60} for n in (1, 16, 64)},
        "utility": {"errors": utility, "scores": {"targets": [{"r2": 0.99} for _ in ev.up.NAMES]}},
    }
    meta = {"budgets": core.budgets()}
    names = {f"{m}_s{s}" for m in ("parent", "prior_plain") for s in (42, 43)} | {
        n for n, _, _ in ev.entries(meta)
    }
    records = {s: {n: copy.deepcopy(sample) for n in names} for s in ev.SPLITS}
    for bank in records.values():
        for name, r in bank.items():
            if name.startswith(("d8_", "d12_")):
                value = 0.6 if name.startswith("d12_") else 0.8
                r["query"]["structure"]["far65"]["primary"] = [value] * 60
                r["utility"]["errors"]["utility"] = [value] * 60
    inv = dict.fromkeys(ev.SPLITS, rows)
    masks = {s: np.ones((60, 13), bool) for s in ev.SPLITS}

    def decide():
        return ev.decision(meta, records, inv, inv, masks, {})

    result = decide()
    assert len(result["checks"]) == 1800
    assert all(x["both_budget_axes_supported"] for x in result["depth_conclusions"].values())
    # A much better longer shallow control removes the compute advantage alone.
    records["test"]["d4_c8_s42_last"]["query"]["structure"]["far65"]["primary"] = [0.1] * 60
    result = decide()
    assert result["branches"]["d8_vs_d4"]["status"] == "representation_candidate"
    assert not result["depth_conclusions"]["d8"]["both_budget_axes_supported"]
    assert result["depth_conclusions"]["d12"]["both_budget_axes_supported"]
    del records["test"]["d12_s43_last"]
    with pytest.raises(ValueError):
        decide()


def test_lr_and_budget_selection_cannot_use_later_validation(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        active = {}

        def train(*a, **kw):
            active["epoch"] += 1
            return {"query": 1.0}, 2

        def val(*a, **kw):
            e = active["epoch"]
            lr = active["job"]["lr_scale"]
            v = 1.0 if e == 0 else (0.7 if lr == 3 else (0.8 if e <= 6 else 0.5))
            return {
                "original": {"selection": v},
                "query": v,
                "recent": v,
                "gap": v,
                "far_primary": v,
                "far_level": v,
                "far_trend": v,
            }

        stack.enter_context(patch.object(run, "train_epoch", side_effect=train))
        stack.enter_context(patch.object(run, "validation", side_effect=val))
        for job in meta["experiments"]:
            active.update(job=job, epoch=0)
            run.worker(out, job["name"], "main", "cpu")
            assert not (out / job["name"] / "last.pt").exists()
            assert (out / job["name"] / "training_state.json").exists()
        locked = run.lock_models(meta, out)
        for seed in (42, 43):
            assert locked["selections"][f"d4_s{seed}"]["lr_scale"] == 3
            assert locked["selections"][f"d4_c8_s{seed}"]["lr_scale"] == 1
            assert locked["selections"][f"d4_c12_s{seed}"]["lr_scale"] == 1
            assert locked["weights"][f"d4_s{seed}"]["last"]["path"] == f"d4_lr3_s{seed}/e6_last.pt"
        ev.check_models(meta, out)
        broken = copy.deepcopy(locked)
        broken["selections"]["d4_s42"]["score"] = [0.0, 0.0]
        atomic_json(broken, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="LR selection"):
            ev.check_models(meta, out)


def test_completion_cleanup_never_removes_an_unverified_checkpoint(tmp_path):
    (tmp_path / "last.pt").write_bytes(b"resumable Adam")
    (tmp_path / "best.pt").write_bytes(b"duplicate best")
    (tmp_path / "e100_last.pt").write_bytes(b"permanent weights")
    atomic_json(
        {"status": "complete", "files": {"e100_last.pt": "wrong"}}, tmp_path / "completion.json"
    )
    with pytest.raises(ValueError):
        run.cleanup_completed(tmp_path)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    atomic_json(
        {"status": "complete", "files": {"e100_last.pt": run.sha256(tmp_path / "e100_last.pt")}},
        tmp_path / "completion.json",
    )
    run.cleanup_completed(tmp_path)
    assert not (tmp_path / "last.pt").exists() and (tmp_path / "e100_last.pt").exists()
    run.cleanup_completed(tmp_path)
