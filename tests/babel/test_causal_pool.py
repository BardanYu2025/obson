"""Synthetic gradients, retained warm provenance, matched targets and full lifecycle."""

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_history_query import SMALL, Capture, fixture, make_builder
from test_overlap_diagnostic import fixture_data

from obson.babel import causal_pool as core
from obson.babel import causal_pool_evaluate as ev
from obson.babel import causal_pool_run as run
from obson.babel import history_query as hq
from obson.babel.ae_extend import atomic_json


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
        for mode in ("control", "structure", "consistent"):
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
    meta.update(epochs=6, budget=4, batch=2, micro=1, evaluation_batch=4, query_config=SMALL)
    atomic_json(meta, out / "manifest.json")
    atomic_json({"eligible": 8}, source / "train_cross_plan.json")
    stack.enter_context(
        patch.object(
            run,
            "construct",
            side_effect=lambda meta, seed, device, mode="plain": (
                core.install(copy.deepcopy(parent), mode, seed),
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
        assert run.read_json(out / "fit.json")["candidates"] == 780
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
                for mode in ("control", "structure", "consistent"):
                    atomic_json(reference, source / f"{split}_{mode}_s{seed}_best.json")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert result["status"] == "matrix_complete_no_automatic_promotion"
        assert all(b["status"] == "no_balanced_gain" for b in result["branches"].values())
        assert len(result["checks"]) == 1224
        assert not result["automatic_promotion"]
        for split in ev.SPLITS:
            for mode in core.MODES:
                for seed in (42, 43):
                    rec = run.read_json(out / f"{split}_{mode}_s{seed}_best.json")
                    assert set(rec["matched"]["lengths"]) == {"32", "64", "80", "128"}
                    assert len(rec["query"]["oldest_endpoint"]["primary"]) == len(d["x"])


@pytest.mark.parametrize("mode", core.MODES)
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
        model, query = core.install(copy.deepcopy(parent), mode, 42), copy.deepcopy(q)
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
            if len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            return original(*a, **kw)

        with patch.object(run, "train_epoch", side_effect=interrupt), pytest.raises(RuntimeError):
            run.worker(out, "scale_pool_s42", "main", "cpu")
        assert torch.load(out / "scale_pool_s42/last.pt", weights_only=True)["epoch"] == 1
        for job in meta["experiments"]:
            run.worker(out, job["name"], "main", "cpu")
        lock = run.lock_models(meta, out)
        for value in lock["trials"].values():
            assert value["epochs"] == 6 and value["selected_epoch"] == 0
            assert value["encoder_steps"] == value["local_steps"] == value["query_steps"] == 12
        for seed in (42, 43):
            histories = [run.read_json(out / f"{mode}_s{seed}/history.json") for mode in core.MODES]
            assert [[v["sampling"] for v in h] for h in histories] == [
                [v["sampling"] for v in histories[0]]
            ] * 3
        with patch.object(run, "train_epoch", side_effect=AssertionError("No retraining")):
            run.worker(out, "scale_pool_s42", "main", "cpu")
        (out / "full_pool_s43/best.pt").write_bytes(b"tamper")
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
        BABEL_CAUSAL_POOL_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_causal_pool768_autodl.sh"
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
        for mode in ("control", "structure", "consistent"):
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
    (source / "control_s42/best.pt").write_bytes(b"changed")
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
                    f"control_s{s}": {"validation": {"original": {"selection": 1.0}}}
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
                for mode in ("control", "structure", "consistent"):
                    atomic_json(reference, source / f"{split}_{mode}_s{seed}_best.json")
        checks = ev.audit_sources(meta, out, "cpu")
        assert len(checks) == 12 and all(c["passed"] for c in checks)
        reference["matched"]["paired80_128"]["gap"]["primary"][0] += 1
        atomic_json(reference, source / "test_control_s42_best.json")
        with pytest.raises(ValueError, match="before training"):
            ev.audit_sources(meta, out, "cpu")
        assert not run.read_json(out / "source_replay_preflight.json")[-1]["passed"]


@pytest.mark.parametrize("mode", core.MODES[1:])
def test_live_pool_causality_short_contexts_batch_and_prefix(mode):
    parent, q, d, _, _ = fixture()
    model = core.install(parent, mode, 42)
    pool = model.core.encoder.pool
    with torch.no_grad():
        pool.output.weight.copy_(torch.eye(8) * 0.2)
    x = torch.tensor(d["x"][:2])
    full = model.core.encoder(x)
    for prefix in (1, 15, 16, 17, 31, 32, 33, 64, 65, 80, 112, 128):
        sliced = model.core.encoder(x[:, :prefix])
        torch.testing.assert_close(sliced, full[:, :prefix], atol=2e-6, rtol=2e-5)
        changed = x.clone()
        changed[:, prefix:] += 30
        torch.testing.assert_close(
            model.core.encoder(changed)[:, :prefix], sliced, atol=2e-6, rtol=2e-5
        )
    torch.testing.assert_close(model.core.encoder(x[:1]), full[:1], atol=2e-6, rtol=2e-5)
    q(full[:, -1]).square().sum().backward()
    for module in (model.core.encoder.backbone, pool.q, pool.k, pool.v, pool.output):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
    assert pool.age_bias.grad.abs().sum() > 0
    appended = torch.cat([x, torch.randn(2, 16, 28)], dim=1)
    torch.testing.assert_close(model.core.encoder(appended)[:, :128], full, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError):
        pool(torch.randn(1, 0, 8))
    with pytest.raises(ValueError):
        q(full)  # No sequence bypass around the single-vector decoder.


def test_pool_lane_boundaries_and_no_shared192_bottleneck():
    pool = core.HistoryPool(8, "scale_pool", 42)
    q = torch.zeros(1, 128, 8)
    k, v = q.clone(), q.clone()
    base = pool.pool(q, k, v)
    for lane, limit in enumerate(core.RANGES):
        # Independent K/V values: boundary inside contributes, one older cannot.
        inside = v.clone()
        inside[:, 128 - limit, lane * 2 : (lane + 1) * 2] = 1
        got = pool.pool(q, k, inside)
        assert torch.all(got[:, -1, lane * 2 : (lane + 1) * 2] > 0)
        if limit < 128:
            outside = v.clone()
            outside[:, 127 - limit, lane * 2 : (lane + 1) * 2] = 1e6
            torch.testing.assert_close(pool.pool(q, k, outside)[:, -1], base[:, -1], atol=0, rtol=0)
    # At a one-bar context all four full-width value slices survive independently.
    identity = torch.eye(8).reshape(8, 1, 8)
    result = pool.pool(torch.zeros_like(identity), torch.zeros_like(identity), identity)
    assert torch.equal(result, identity)
    wide = core.HistoryPool(768, "scale_pool", 42)
    assert sum(p.numel() for p in wide.parameters()) == 2364416


def test_common_initial_function_parameters_rng_and_objective():
    parent, query, d, stats, local = fixture()
    x = torch.tensor(d["x"])
    before = torch.get_rng_state().clone()
    variants = {m: core.install(copy.deepcopy(parent), m, 42) for m in core.MODES}
    assert torch.equal(before, torch.get_rng_state())
    for m in variants.values():
        assert torch.equal(m.core.encoder(x), parent.core.encoder(x))
    a, b = [variants[m].core.encoder.pool.state_dict() for m in core.MODES[1:]]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    builder = make_builder(stats)()
    rows = builder(np.array([0, 1]), np.array([1, 16]))[:3]
    batch = run.batch_tensors({k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, "cpu")
    ps = torch.tensor([[32, 64, 96, 128]] * 6)
    qp = torch.tensor(np.tile(hq.sample_prefixes(2, 42, 1), (3, 1)))
    outputs, grads = [], []
    for mode, model in variants.items():
        q = copy.deepcopy(query)
        parts = core.losses(model, q, batch, ps, qp, torch.tensor([1, 16]), stats, local, mode)
        parts["optimization_value"].sum().backward()
        outputs.append(parts["optimization_value"].detach())
        base = {
            n: p.grad for n, p in model.core.encoder.named_parameters() if not n.startswith("pool.")
        }
        grads.append(base)
        if mode != "plain":
            pool = model.core.encoder.pool
            assert pool.output.weight.grad.abs().sum() > 0
            assert pool.q.weight.grad.abs().sum() == 0
    for value in outputs[1:]:
        torch.testing.assert_close(value, outputs[0], atol=0, rtol=0)
    for other in grads[1:]:
        for key, value in grads[0].items():
            torch.testing.assert_close(value, other[key], atol=0, rtol=0)


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


def test_storage_guard_derived_from_actual_states(tmp_path):
    parent, query, _, _, _ = fixture()
    meta = {"experiments": [{"name": f"{m}_s42", "mode": m, "seed": 42} for m in core.MODES]}
    with patch.object(
        run,
        "construct",
        side_effect=lambda meta, seed, device, mode: (
            core.install(copy.deepcopy(parent), mode, seed),
            copy.deepcopy(query),
        ),
    ):
        report = run.storage_budget(meta, 2)
        counts = report["checkpoints"]
        assert counts[1]["state_bytes"] == counts[2]["state_bytes"] > counts[0]["state_bytes"]
        assert report["atomic_temporary_bytes"] == 2 * max(c["last_bytes"] for c in counts)
        with (
            patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
            pytest.raises(ValueError),
        ):
            run.check_disk(tmp_path, 2, meta)
        assert not run.read_json(tmp_path / "disk_preflight.json")["passed"]


def test_decision_requires_balanced_gains_and_isolates_scale_comparison():
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
        "structure_overlap": {str(n): [1.0] * 60 for n in (1, 16, 64)},
        "utility": {"errors": utility, "scores": {"targets": [{"r2": 0.99} for _ in ev.up.NAMES]}},
    }
    meta = {
        "experiments": [
            {"name": f"{mode}_s{s}", "mode": mode, "seed": s}
            for s in (42, 43)
            for mode in core.MODES
        ]
    }
    names = {
        f"{m}_s{s}"
        for m in ("parent", "prior_control", "prior_structure", "prior_consistent")
        for s in (42, 43)
    } | {n for n, _, _ in ev.entries(meta)}
    records = {s: {n: copy.deepcopy(sample) for n in names} for s in ev.SPLITS}
    for bank in records.values():
        for name, r in bank.items():
            if name.startswith(("full_pool_", "scale_pool_")):
                r["query"]["structure"]["far65"]["primary"] = [
                    0.6 if name.startswith("scale_pool_") else 0.8
                ] * 60
                r["utility"]["errors"]["utility"] = [
                    0.6 if name.startswith("scale_pool_") else 0.8
                ] * 60
    inv = dict.fromkeys(ev.SPLITS, rows)
    masks = {s: np.ones((60, 13), bool) for s in ev.SPLITS}

    def decide():
        return ev.decision(meta, records, inv, inv, masks, {})

    result = decide()
    assert len(result["checks"]) == 1224
    assert all(b["status"] == "representation_candidate" for b in result["branches"].values())
    target = records["test"]["scale_pool_s42_last"]
    target["utility"]["errors"]["utility"] = [0.8] * 60
    assert decide()["branches"]["scale_pool_vs_full_pool"]["status"] == "balanced_state_gain"
    target["query"]["oldest_endpoint"]["change1"] = [2.0] * 60
    result = decide()
    assert result["branches"]["full_pool_vs_plain"]["status"] == "representation_candidate"
    assert result["branches"]["scale_pool_vs_plain"]["status"] == "gain_with_tradeoffs"
    assert result["branches"]["scale_pool_vs_full_pool"]["status"] == "gain_with_tradeoffs"
    del records["cross_research"]["prior_consistent_s43"]
    with pytest.raises(ValueError):
        decide()
