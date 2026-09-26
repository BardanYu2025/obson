"""Synthetic-only historical queries, gradient routing and resumable fixed budgets."""

import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_capacity_growth import small
from test_overlap_diagnostic import fixture_data

from obson.babel import history_query as hq
from obson.babel import history_query_evaluate as ev
from obson.babel import history_query_run as run
from obson.babel.ae_extend import atomic_json

SMALL = {"slots": 2, "width": 4, "heads": 2, "layers": 1, "ff": 8}


@pytest.fixture(autouse=True)
def no_neural_updates():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def fixture():
    model, d, stats, local = small()
    run.rs.pf.install(model, stats, True)
    with torch.no_grad():
        model.core.encoder.backbone.input.projection.weight.fill_(0.02)
    query = hq.HistoryQuery(8, {"mean": [0.0] * 8, "scale": [1.0] * 8}, 42, SMALL)
    return model.eval(), query.eval(), d, stats, local


def test_targets_exclude_current_and_future_reconstruct_correct_coordinates():
    _, _, d, stats, local = fixture()
    b = run.batch_tensors(d, "cpu")
    ps = torch.tensor([[32, 48, 80, 112, 128]] * len(b["x"]))
    y, m = hq.targets(b, ps, stats)
    values = hq.physical(y, stats)
    physical = b["y"].double() * torch.tensor(stats["y_scale"], dtype=torch.float64) + torch.tensor(
        stats["y_mean"], dtype=torch.float64
    )
    anchors = hq.anchors(b, ps, stats)
    for j, p in enumerate(ps[0]):
        for age in (1, 16, int(p) - 1):
            actual = values[:, j, age - 1, 0]
            expected = physical[:, p - 1 - age, 0] - anchors[:, j]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-6)
        assert not m[:, j, int(p) - 1 :].any()
        assert m[:, j, : int(p) - 1, 0].all()
    # Masked current y is deliberately garbage: its zero sentinel isn't a close.
    changed = {k: v.clone() for k, v in b.items()}
    changed["y"][:, -1, 0] = 123456
    torch.testing.assert_close(hq.targets(changed, ps, stats)[0], y, atol=0, rtol=0)
    # Future OHLCV/targets cannot change a prefix's labels.
    changed = {k: v.clone() for k, v in b.items()}
    changed["x"][:, 80:] += 10
    changed["y"][:, 80:] += 30
    p = torch.full((len(b["x"]), 1), 80)
    torch.testing.assert_close(
        hq.targets(changed, p, stats)[0], hq.targets(b, p, stats)[0], atol=0, rtol=0
    )
    # Own predicted age17 reproduces the original local16 target exactly in scale.
    original, mask = hq.ba.local_targets(b["y"], b["mask"], ps, stats, local)
    actual = hq.recent_prediction(y, stats, local)
    torch.testing.assert_close(actual[mask], original[mask], atol=2e-5, rtol=2e-5)


def test_queries_are_subset_batch_and_causal_prefix_invariant():
    model, query, d, _, _ = fixture()
    x = torch.tensor(d["x"][:2])
    ps = torch.full((2, 1), 64)
    with torch.inference_mode():
        z, p = hq.predict_states(model, query, x, ps)
        changed = x.clone()
        changed[:, 64:] += 12
        _, p2 = hq.predict_states(model, query, changed, ps)
        torch.testing.assert_close(p, p2, atol=2e-6, rtol=2e-5)
        a = query(z[:, 63])
        subset = query(z[:, 63], torch.tensor([1, 16, 63, 127]))
        torch.testing.assert_close(subset, a[:, [0, 15, 62, 126]], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(query(z[:1, 63]), a[:1], atol=2e-6, rtol=2e-5)
    for age in (0, -1, 128):
        with pytest.raises(ValueError):
            query(z[:, 63], torch.tensor([age]))
    assert not any("self_attn" in n for n, _ in query.named_modules())


def test_full768_query_has_no_external_information_and_live_gradients():
    query = hq.HistoryQuery(768, {"mean": [0.0] * 768, "scale": [1.0] * 768}, 43)
    z = torch.randn(2, 768, requires_grad=True)
    q = query(z)
    assert q.shape == (2, 127, 7)
    q.square().mean().backward()
    assert z.grad.abs().sum() > 0
    assert query.memory.weight.grad.abs().sum() > 0
    assert query.blocks[0].cross.in_proj_weight.grad.abs().sum() > 0
    assert query.output.weight.grad.abs().sum() > 0


def test_loss_masks_unavailable_history_and_compares_bands_not_lengths():
    _, _, d, stats, _ = fixture()
    b = run.batch_tensors(d, "cpu")
    ps = torch.tensor([[32, 64, 128]] * len(b["x"]))
    y, m = hq.targets(b, ps, stats)
    p = y.clone().requires_grad_()
    zero = hq.metrics(p, y, m, stats)
    assert all(torch.all(v == 0) for v in zero.values())
    bad = y.clone()
    bad[~m] = 999
    assert torch.all(hq.metrics(bad, y, m, stats)["primary"] == 0)
    # Only oldest band's constant path error: same per-band weight at128.
    prediction = y.clone()
    prediction[:, :, 64:, 0] += 1
    score = hq.metrics(prediction, y, m, stats)
    torch.testing.assert_close(
        score["path"][:, -1], torch.full((len(y),), 1 / 3), rtol=1e-6, atol=1e-6
    )


def test_schedule_is_paired_and_never_uses_held_prefixes():
    a = hq.sample_prefixes(4789, 42, 601)
    b = hq.sample_prefixes(4789, 42, 601)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4789, 5) and (a[:, -1] == 128).all()
    assert not np.isin(a, hq.ba.HELD_PREFIXES).any()
    assert (np.diff(a, axis=1) > 0).all()


class Capture:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.gradients = None

    def zero_grad(self, set_to_none=True):
        for p in self.parameters:
            p.grad = None

    def step(self):
        self.gradients = [
            None if p.grad is None else p.grad.detach().clone() for p in self.parameters
        ]


def make_builder(stats):
    _, _, _, views = fixture_data(stats)

    class Builder:
        plan = list(range(8))

        def __init__(self, *args):
            pass

        def __call__(self, ids, shifts):
            a = {
                k: np.stack(
                    [
                        views["views"][int(s)][k][int(i) % 3]
                        for i, s in zip(ids, shifts, strict=True)
                    ]
                )
                for k in ("x", "y", "mask")
            }
            b = {k: views["views"][0][k][np.asarray(ids) % 3] for k in ("x", "y", "mask")}
            return a, b, copy.deepcopy(b), None, None

    return Builder


def test_gradient_routing_changes_encoder_only_query_updates_are_matched():
    model, query, _, stats, local = fixture()
    builder_type = make_builder(stats)
    ids = np.array([0, 1])
    shifts = np.array([1, 16])
    ps = np.array([[32, 64, 96, 128]] * 2)
    qp = hq.sample_prefixes(2, 42, 1)
    schedule = (ids, shifts, ps, qp, {})
    gradients = []
    query_grads = []
    head_grads = []
    for mode in ("control", "joint"):
        m, q = copy.deepcopy(model), copy.deepcopy(query)
        m.requires_grad_(False)
        m.core.encoder.requires_grad_(True)
        m.local_head.requires_grad_(True)
        enc, head, dec = (
            Capture(m.core.encoder.parameters()),
            Capture(m.local_head.parameters()),
            Capture(q.parameters()),
        )
        run.train_epoch(
            m, q, builder_type(), schedule, stats, local, 2, 2, "cpu", (enc, head, dec), mode, 0.25
        )
        gradients.append(enc.gradients)
        query_grads.append(dec.gradients)
        head_grads.append(head.gradients)
    assert any(not torch.equal(a, b) for a, b in zip(*gradients, strict=True) if a is not None)
    for a, b in zip(*query_grads, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(*head_grads, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Frozen leaves ALL original parameters gradient-free, yet decoder learns.
    m, q = copy.deepcopy(model).requires_grad_(False), copy.deepcopy(query)
    dec = Capture(q.parameters())
    run.train_epoch(
        m, q, builder_type(), schedule, stats, local, 2, 1, "cpu", (None, None, dec), "frozen", 0.25
    )
    assert all(p.grad is None for p in m.parameters())
    assert any(g is not None and g.abs().sum() > 0 for g in dec.gradients)


def lifecycle(tmp_path, stack):
    parent, query, d, stats, local = fixture()
    builder_type = make_builder(stats)
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    identity = {"manifest": {"identity": {"training_manifest": {}, "training_source": str(source)}}}
    meta = run.make_manifest(source, identity, 16)
    meta.update(
        epochs=6, warm_epochs=6, budget=4, batch=2, micro=1, evaluation_batch=4, query_config=SMALL
    )
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


def test_six_arm_full_budget_lock_resume_epoch0_and_tamper(tmp_path):
    with ExitStack() as stack:
        meta, out, _, _, _, _, _ = lifecycle(tmp_path, stack)
        original = run.train_epoch
        calls = []

        def interrupt(*a, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            return original(*a, **kw)

        with (
            patch.object(run, "train_epoch", side_effect=interrupt),
            pytest.raises(RuntimeError, match="synthetic interruption"),
        ):
            run.worker(out, "frozen_s42", "warm", "cpu")
        assert torch.load(out / "warm_s42/last.pt", weights_only=True)["epoch"] == 1
        for seed in (42, 43):
            run.worker(out, f"frozen_s{seed}", "warm", "cpu")
        for job in meta["experiments"]:
            run.worker(out, job["name"], "main", "cpu")
        lock = run.lock_models(meta, out)
        assert len(lock["trials"]) == 6
        for name, s in lock["trials"].items():
            assert s["epochs"] == 6 and s["query_steps"] == 12 and s["selected_epoch"] == 0
            assert s["encoder_steps"] == (0 if name.startswith("frozen") else 12)
            assert s["views"] == 72
        for seed in (42, 43):
            histories = [
                run.read_json(out / f"{mode}_s{seed}/history.json")
                for mode in ("frozen", "control", "joint")
            ]
            assert [[v["sampling"] for v in h] for h in histories] == [
                [v["sampling"] for v in histories[0]]
            ] * 3
        with patch.object(run, "train_epoch", side_effect=AssertionError("No retraining")):
            run.worker(out, "joint_s43", "main", "cpu")
        (out / "joint_s43/best.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            ev.check_models(meta, out)


def test_reader_fitting_and_research_cannot_start_before_model_lock(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.fit_readouts({"experiments": []}, tmp_path, "cpu")
    with pytest.raises(FileNotFoundError):
        ev.evaluate({"experiments": []}, tmp_path, "cpu")


def test_query_scores_rebases_with_own_anchor_and_reports_all_positions(tmp_path):
    model, query, d, stats, local = fixture()
    with patch.object(run, "stats", return_value=(stats, local)):
        report, z = ev.query_scores({"evaluation_batch": 4}, model, query, d, "cpu")
    assert z.shape == (len(d["x"]), 8)
    assert set(report["native"]["errors"]) == {f"p{p}" for p in ev.PREFIXES} | {"held", "trained"}
    assert len(report["examples"]) == 4
    for family in ("native", "recent"):
        held = np.mean(
            [report[family]["errors"][f"p{p}"]["primary"] for p in hq.ba.HELD_PREFIXES], axis=0
        )
        np.testing.assert_array_equal(held, report[family]["errors"]["held"]["primary"])


def test_microbatch_accumulation_matches_old_objective_and_full_batch():
    model, query, _, stats, local = fixture()
    builder_type = make_builder(stats)
    ids, shifts = np.array([0, 1]), np.array([1, 16])
    ps = np.array([[32, 64, 96, 128]] * 2)
    schedule = (ids, shifts, ps, hq.sample_prefixes(2, 42, 1), {})
    values = []
    for micro in (1, 2):
        m, q = copy.deepcopy(model), copy.deepcopy(query)
        m.requires_grad_(False)
        m.core.encoder.requires_grad_(True)
        m.local_head.requires_grad_(True)
        opts = [Capture(v.parameters()) for v in (m.core.encoder, m.local_head, q)]
        result, steps = run.train_epoch(
            m, q, builder_type(), schedule, stats, local, 2, micro, "cpu", opts, "control", 0.25
        )
        assert steps == 1
        values.append((opts, result))
    for a, b in zip(values[0][0], values[1][0], strict=True):
        for x, y in zip(a.gradients, b.gradients, strict=True):
            if x is not None:
                torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-4)
    # Independent original objective: no query loss can affect control encoder.
    m = copy.deepcopy(model)
    m.requires_grad_(False)
    m.core.encoder.requires_grad_(True)
    m.local_head.requires_grad_(True)
    a, b, c, _, _ = builder_type()(ids, shifts)
    joined = run.batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, "cpu")
    g, base = run.xr.xp.oc.supervised(m, joined, torch.tensor(np.tile(ps, (3, 1))), stats, local)
    within = run.xr.xp.oc.consistency_rows(
        g[:2], g[2:4], joined["mask"][:2], joined["mask"][2:4], torch.tensor(shifts), stats
    )
    (base.reshape(3, 2).mean(0) + 0.1 * within).mean().backward()
    torch.nn.utils.clip_grad_norm_(m.core.encoder.parameters(), 1.0)
    for x, p in zip(values[1][0][0].gradients, m.core.encoder.parameters(), strict=True):
        if x is not None:
            torch.testing.assert_close(x, p.grad, atol=2e-6, rtol=2e-5)


def test_complete_synthetic_fit_evaluate_pipeline(tmp_path):
    from types import SimpleNamespace

    with ExitStack() as stack:
        meta, out, parent, _, d, stats, local = lifecycle(tmp_path, stack)
        source = run.root(meta)
        meta["identity"]["manifest"]["source"] = str(source)
        atomic_json(meta, out / "manifest.json")
        for seed in (42, 43):
            run.worker(out, f"frozen_s{seed}", "warm", "cpu")
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
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert result["status"] == "no_joint_query_gain" and not result["automatic_promotion"]
        assert len(result["checks"]) > 500
        assert all(
            run.read_json(out / f"{s}_parent_s{seed}_replay.json")["passed"]
            for s in ev.SPLITS
            for seed in (42, 43)
        )
        # Reuse actual report schema to exercise positive gates and single-cell failures.
        names = [f"parent_s{s}" for s in (42, 43)] + [n for n, _, _ in ev.entries(meta)]
        records = {s: {n: run.read_json(out / f"{s}_{n}.json") for n in names} for s in ev.SPLITS}
        rows = {s: run.read_json(source / f"{s}_inventory.json") for s in ev.SPLITS}
        for bank in records.values():
            for name, r in bank.items():
                if name.startswith("joint"):
                    for family in ("native", "recent"):
                        for group in r["query"][family]["errors"].values():
                            for k, vals in group.items():
                                group[k] = (np.array(vals) * 0.8).tolist()
                    for i, target in enumerate(ev.up.NAMES[6:], 6):
                        r["utility"]["errors"][target] = [0.0] * len(z)
                        r["utility"]["scores"]["targets"][i]["r2"] = 0.9

        def interval(a, b, cohort):
            delta = float(np.mean(np.asarray(a) - b))
            return {"supported": True, "high": delta, "low": delta}

        with patch.object(ev.ur, "interval", side_effect=interval):
            result = ev.decision(
                meta,
                records,
                rows,
                dict.fromkeys(ev.SPLITS, paired["rows"]),
                dict.fromkeys(ev.SPLITS, mask),
                {},
            )
            assert result["status"] == "history_query_research_candidate"
            records["test"]["joint_s43_last"]["query"]["native"]["errors"]["held"]["primary"] = (
                records["test"]["control_s43_last"]["query"]["native"]["errors"]["held"]["primary"]
            )
            result = ev.decision(
                meta,
                records,
                rows,
                dict.fromkeys(ev.SPLITS, paired["rows"]),
                dict.fromkeys(ev.SPLITS, mask),
                {},
            )
            assert result["status"] == "no_joint_query_gain"


def test_script_exports_failure_status_and_excludes_weights(tmp_path):
    import os
    import subprocess
    import tarfile
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    out = tmp_path / "result"
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    (out / "last.pt").write_bytes(b"not a report")
    log = tmp_path / "run.log"
    log.write_text("synthetic traceback retained")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake_python.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake_python),
        BABEL_HISTORY_QUERY_RUN=str(out),
        BABEL_HISTORY_QUERY_LOG=str(log),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    script = project / "scripts/babel_history_query768_autodl.sh"
    failed = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert failed.returncode == 7 and "run_status=failed" in failed.stdout
    for mode in ("initial", "manual"):
        if mode == "manual":
            result = subprocess.run(
                ["bash", str(script), "export"], env=env, capture_output=True, text=True
            )
            assert result.returncode == 0 and "run_status=failed" in result.stdout
        with tarfile.open(tmp_path / "download/result_reports.tar.gz") as archive:
            names = archive.getnames()
            assert "result/manifest.json" in names and "result/experiment_notes.md" in names
            assert not any(n.endswith(".pt") for n in names)
            assert (
                "run_status=failed" in archive.extractfile("result/run_status.txt").read().decode()
            )
            assert archive.extractfile("result/run.log").read().decode() == log.read_text()


def test_empty_or_partial_research_cannot_pass():
    with pytest.raises(ValueError, match="Both fixed research"):
        ev.decision({"experiments": []}, {}, {}, {}, {}, {})


def test_nested_inventory_checks_bytes_and_rejects_path_escapes(tmp_path):
    root = tmp_path / "delivery"
    (root / "bundle").mkdir(parents=True)
    weight = root / "bundle/control_s42_best.pt"
    weight.write_bytes(b"synthetic checkpoint")
    files = {"bundle/control_s42_best.pt": run.sha256(weight)}
    run.verify_files(root, files)
    weight.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run.verify_files(root, files)
    weight.unlink()
    with pytest.raises(ValueError, match="file missing"):
        run.verify_files(root, files)
    external = tmp_path / "external.pt"
    external.write_bytes(b"external")
    for name in ("../external.pt", str(external), "bundle/../../external.pt"):
        with pytest.raises(ValueError, match="Invalid inventory path"):
            run.verify_files(root, {name: run.sha256(external)})
    weight.symlink_to(external)
    with pytest.raises(ValueError, match="escapes root"):
        run.verify_files(root, {"bundle/control_s42_best.pt": run.sha256(external)})
    with pytest.raises(ValueError, match="Empty file inventory"):
        run.verify_files(root, {})


def test_delivery_source_identity_accepts_nested_bundle_and_still_rejects_tampering(tmp_path):
    source = tmp_path / "delivery"
    (source / "bundle").mkdir(parents=True)
    zero = dict.fromkeys(("encoder_updates", "head_updates", "reader_fits", "statistics_fits"), 0)
    identity = {"synthetic": True}
    meta = {
        "schema": run.delivery.SCHEMA,
        "code_sha256": run.delivery.code_identity(),
        "source": str(tmp_path / "upstream"),
        "identity": identity,
        **zero,
    }
    atomic_json(meta, source / "manifest.json")
    weight = source / "bundle/control_s42_best.pt"
    weight.write_bytes(b"synthetic checkpoint bytes; never loaded")
    atomic_json(
        {"source_manifest": run.sha256(source / "manifest.json")}, source / "bundle/index.json"
    )
    atomic_json(
        {"status": "passed", "index_sha256": run.sha256(source / "bundle/index.json")},
        source / "bundle/validation.json",
    )
    files = {str(p.relative_to(source)): run.sha256(p) for p in source.rglob("*") if p.is_file()}
    atomic_json(
        {"status": "complete", "source_unchanged": True, "files": files, **zero},
        source / "completion.json",
    )
    with patch.object(run.warm, "source_identity", return_value=identity):
        result = run.source_identity(source)
        assert result["manifest"] == meta
        assert result["files"]["bundle/control_s42_best.pt"] == files["bundle/control_s42_best.pt"]
        weight.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            run.source_identity(source)


def test_completed_run_checks_nested_outputs_without_training(tmp_path):
    source, out = tmp_path / "source", tmp_path / "experiment"
    source.mkdir()
    out.mkdir()
    identity = {"manifest": {"source": str(source), "identity": {}}}
    atomic_json({"torch": str(torch.__version__), "numpy": np.__version__}, source / "runtime.json")
    meta = run.make_manifest(source, identity)
    atomic_json(meta, out / "manifest.json")
    (out / "joint_s42").mkdir()
    checkpoint = out / "joint_s42/best.pt"
    checkpoint.write_bytes(b"synthetic")
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "files": {"joint_s42/best.pt": run.sha256(checkpoint)},
        },
        out / "completion.json",
    )
    with (
        patch.object(run, "source_identity", return_value=identity),
        patch.object(run.warm, "check_output"),
        patch.object(run, "dispatch", side_effect=AssertionError("Must not train")),
    ):
        run.run(source, out, device="cpu")
        checkpoint.write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            run.run(source, out, device="cpu")
