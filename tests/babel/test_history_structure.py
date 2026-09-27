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

from obson.babel import history_query as hq
from obson.babel import history_structure as core
from obson.babel import history_structure_evaluate as ev
from obson.babel import history_structure_run as run
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
        for mode in ("uniform", "additive"):
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
            run, "construct", side_effect=lambda *a: (copy.deepcopy(parent), copy.deepcopy(query))
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
        meta["identity"]["manifest"]["source"] = str(source)
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
                source / f"{split}_predictions.json",
            )
            for seed in (42, 43):
                atomic_json(previous, source / f"{split}_control_s{seed}_best.json")
        atomic_json(
            {
                "datasets": {
                    s: {"scores": dict.fromkeys(("raw3584", "pca768", "current28"), baseline)}
                    for s in ev.SPLITS
                }
            },
            source / "readout_metrics.json",
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
        qs.pop("structure")
        reference = {
            "original": previous["reconstruction"],
            "utility": previous["utility"],
            "overlap": previous["overlap"],
            "query": qs,
            "matched": ev.matched_scores(meta, parent, query, d, "cpu"),
            "query_overlap": ev.query_overlap(meta, parent, query, paired, "cpu"),
        }
        for split in ev.SPLITS:
            for seed in (42, 43):
                for mode in ("uniform", "additive"):
                    atomic_json(reference, source / f"{split}_{mode}_s{seed}_best.json")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert result["status"] == "matrix_complete_no_automatic_promotion"
        assert all(b["status"] == "no_structure_gain" for b in result["branches"].values())
        assert len(result["checks"]) == 784
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
        model, query = copy.deepcopy(parent), copy.deepcopy(q)
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
            run.worker(out, "consistent_s42", "main", "cpu")
        assert torch.load(out / "consistent_s42/last.pt", weights_only=True)["epoch"] == 1
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
            run.worker(out, "consistent_s42", "main", "cpu")
        (out / "structure_s43/best.pt").write_bytes(b"tamper")
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


def test_disk_guard_selection_and_research_lock(tmp_path):
    with (
        patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1024)),
        pytest.raises(ValueError, match="Need"),
    ):
        run.check_disk(tmp_path, 1)
    assert not run.read_json(tmp_path / "disk_preflight.json")["passed"]
    with patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)):
        run.check_disk(tmp_path, 2)
    initial = dict.fromkeys(("query", "recent", "far_primary", "far_level", "far_trend"), 4.0)
    good = dict.fromkeys(initial, 2.0)
    assert run.selected_value(good, initial) == 0.5
    assert np.isinf(run.selected_value(dict(good, recent=5.0), initial))
    assert np.isinf(run.selected_value(dict(good, far_primary=5.0), initial))
    with pytest.raises(FileNotFoundError):
        ev.fit_readouts({"experiments": []}, tmp_path, "cpu")
    with pytest.raises(FileNotFoundError):
        ev.evaluate({"experiments": []}, tmp_path, "cpu")


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
        BABEL_HISTORY_STRUCTURE_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_history_structure768_autodl.sh"
    result = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert result.returncode == 7 and "run_status=failed" in result.stdout
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and "run_status=failed" in result.stdout
    with tarfile.open(tmp_path / "download/out_reports.tar.gz") as archive:
        assert not any(n.endswith(".pt") for n in archive.getnames())


def test_structure_targets_masked_bins_and_physical_scaling():
    _, _, d, stats, _ = fixture()
    b = run.batch_tensors(d, "cpu")
    ps = torch.tensor([[32, 64, 128]] * len(d["x"]))
    y, m = hq.targets(b, ps, stats)
    exact = core.metrics(y, y, m, stats)
    assert all(torch.equal(v, torch.zeros_like(v)) for v in exact.values())
    changed = y.clone()
    changed[~m] = 1e6
    assert all(
        torch.equal(v, torch.zeros_like(v)) for v in core.metrics(changed, y, m, stats).values()
    )
    # Oldest diagnostics cannot be improved by changing the recent part.
    changed = y.clone()
    changed[..., :64, :] += 10
    assert core.metrics(changed, y, m, stats, lo=65)["primary"].max() == 0
    pred = y.clone()
    scale = hq.price_scales(stats, pred)
    pred[..., 0] += 2 / scale
    values = core.metrics(pred, y, m, stats)
    assert values["level"].max() > 0
    assert values["trend"].max() < 1e-10
    # Compare pooling to direct arithmetic, including the final partial bin.
    v = torch.arange(127, dtype=torch.float64)[None]
    mask = torch.ones_like(v, dtype=torch.bool)
    mask[:, 24] = False
    for width in core.WIDTHS:
        pooled, valid, centers = core.pools(v, mask, width)
        for i, start in enumerate(range(16, 127, width)):
            end = min(start + width, 127)
            assert bool(valid[0, i]) == bool(mask[0, start:end].all())
            if valid[0, i]:
                torch.testing.assert_close(pooled[0, i], v[0, start:end].mean())
            assert centers[i] == (start + end + 1) / 2


def test_physical_overlap_mapping_and_no_truth_shortcut():
    _, _, _, stats, _ = fixture()
    shifts = torch.tensor([1, 16, 64])
    ages = torch.arange(1, 128, dtype=torch.float64)
    # Shared physical t=-age in A equals t=shift-age in B, non-linear signal.
    a = (-ages).square().expand(3, -1) / 500
    b = (shifts[:, None] - ages).square() / 500
    pa = torch.zeros(3, 127, 7, dtype=torch.float64)
    pb = pa.clone()
    pa[..., 0] = a / hq.price_scales(stats, pa)
    pb[..., 0] = (b + 23) / hq.price_scales(stats, pb)
    mask = torch.ones_like(pa, dtype=torch.bool)
    value = core.overlap(pa, pb, mask, mask, shifts, stats)
    assert value.max() < 1e-20
    wrong = core.overlap(pa, pb, mask, mask, torch.ones(3, dtype=torch.long), stats)
    assert wrong[1:].min() > 0
    pb.requires_grad_(True)
    bad = pb.clone()
    bad[..., 0] = bad[..., 0] + torch.sin(ages)[None]
    core.overlap(pa, bad, mask, mask, shifts, stats).sum().backward()
    assert pb.grad is not None and pb.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        core.overlap(pa, pb, mask, mask, torch.tensor([0, 1, 2]), stats)


def test_added_objectives_gradients_and_control_exact_previous_uniform():
    from obson.babel import bar_semantics as previous

    parent, query, _, stats, local = fixture()
    a, b, c, _, _ = make_builder(stats)()(np.array([0, 1]), np.array([1, 16]))
    batch = run.batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, "cpu")
    ps = torch.tensor([[32, 64, 96, 128]] * 6)
    qp = torch.tensor(np.tile(hq.sample_prefixes(2, 42, 1), (3, 1)))
    shifts = torch.tensor([1, 16])
    grads = {}
    for mode in (*core.MODES, "old_uniform"):
        model, q = copy.deepcopy(parent), copy.deepcopy(query)
        model.requires_grad_(False)
        model.core.encoder.requires_grad_(True)
        model.local_head.requires_grad_(True)
        parts = (
            previous.losses(model, q, batch, ps, qp, shifts, stats, local, "uniform")
            if mode == "old_uniform"
            else core.losses(model, q, batch, ps, qp, shifts, stats, local, mode)
        )
        parts["optimization_value"].sum().backward()
        grads[mode] = [
            torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in mod.parameters()
                ]
            )
            for mod in (model.core.encoder, model.local_head, q)
        ]
        assert all(p.grad is None for p in model.core.decoder.parameters())
    for x, y in zip(grads["control"], grads["old_uniform"], strict=True):
        torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
    for mode in core.MODES:
        torch.testing.assert_close(grads[mode][1], grads["control"][1], atol=0, rtol=0)
        assert all(torch.isfinite(v).all() for v in grads[mode])
    for index in (0, 2):
        assert (grads["structure"][index] - grads["control"][index]).abs().sum() > 0
        assert (grads["consistent"][index] - grads["structure"][index]).abs().sum() > 0


def test_source_identity_retained_best_without_last_and_strict_restore(tmp_path):
    parent, query, _, _, _ = fixture()
    source = tmp_path / "old"
    source.mkdir()
    identity = {"fixture": True}
    meta = run.previous.make_manifest(source, identity, {}, 16)
    atomic_json(meta, source / "manifest.json")
    trials = {}
    for seed in (42, 43):
        for mode in ("uniform", "additive"):
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
    (source / "uniform_s42/best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
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
                    f"uniform_s{s}": {"validation": {"original": {"selection": 1.0}}}
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
        q.pop("structure")
        reference = {"query": q, "matched": ev.matched_scores(meta, parent, query, d, "cpu")}
        for split in ev.SPLITS:
            for seed in (42, 43):
                for mode in ("uniform", "additive"):
                    atomic_json(reference, source / f"{split}_{mode}_s{seed}_best.json")
        checks = ev.audit_sources(meta, out, "cpu")
        assert len(checks) == 8 and all(c["passed"] for c in checks)
        reference["matched"]["paired80_128"]["gap"]["primary"][0] += 1
        atomic_json(reference, source / "test_uniform_s42_best.json")
        with pytest.raises(ValueError, match="before training"):
            ev.audit_sources(meta, out, "cpu")
        assert not run.read_json(out / "source_replay_preflight.json")[-1]["passed"]


def test_decision_reports_candidates_separately_and_keeps_old_recovery():
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
        f"{m}_s{s}" for m in ("parent", "prior_uniform", "prior_additive") for s in (42, 43)
    } | {n for n, _, _ in ev.entries(meta)}
    records = {s: {n: copy.deepcopy(sample) for n in names} for s in ev.SPLITS}
    for bank in records.values():
        for name, r in bank.items():
            if name.startswith(("structure_", "consistent_")):
                r["query"]["structure"]["far65"]["primary"] = [0.8] * 60
                r["utility"]["errors"]["utility"] = [0.8] * 60
    inv = dict.fromkeys(ev.SPLITS, rows)
    masks = {s: np.ones((60, 13), bool) for s in ev.SPLITS}

    def decide():
        return ev.decision(meta, records, inv, inv, masks, {})

    result = decide()
    assert len(result["checks"]) == 784
    assert all(b["status"] == "representation_candidate" for b in result["branches"].values())
    target = records["test"]["consistent_s42_last"]
    target["utility"]["errors"]["utility"] = [1.0] * 60
    assert decide()["branches"]["consistent"]["status"] == "structure_gain_only"
    target["query"]["oldest_endpoint"]["change1"] = [2.0] * 60
    result = decide()
    assert result["branches"]["structure"]["status"] == "representation_candidate"
    assert result["branches"]["consistent"]["status"] == "structure_gain_with_tradeoffs"
    assert not result["branches"]["consistent"]["groups"]["recovery"]
    del records["cross_research"]["prior_additive_s43"]
    with pytest.raises(ValueError):
        decide()
