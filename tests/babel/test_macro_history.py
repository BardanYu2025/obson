"""Synthetic-only macro geometry, frozen-reader routing and complete lifecycle."""

import copy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_grounded_history import builder_class as old_builder
from test_history_query import fixture
from test_overlap_diagnostic import fixture_data
from test_shared_history import lifecycle as old_lifecycle

from obson.babel import macro_history as core
from obson.babel import macro_history_data as gd
from obson.babel import macro_history_evaluate as ev
from obson.babel import macro_history_run as run
from obson.babel.ae_extend import atomic_json

REAL_DATA_VERIFY = gd.verify


@pytest.fixture(autouse=True)
def no_real_training():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def rows(n=8, pair="15->30"):
    return [
        {
            "key": f"X/15/C{i // 2}",
            "pair": pair,
            "end": f"t{i}",
            "week": f"w{i // 2}",
            "links": [[2 * c - 127, 2 * (c + 1) - 127, c, c + 1] for c in range(70, 120)],
        }
        for i in range(n)
    ]


def builder_class(stats):
    cls = old_builder(stats)
    cls.plan = rows()
    return cls


def lifecycle(tmp_path, stack):
    _, source, m, q, d, stats, local, _ = old_lifecycle(tmp_path, stack)
    del q.shared
    identity = {
        "source": str(source),
        "files": {"manifest.json": run.sha256(source / "manifest.json")},
        "chosen": {str(s): {"sha256": f"seed{s}"} for s in (42, 43)},
        "grounded_source": str(source),
    }
    meta = run.make_manifest(source, identity, 2)
    meta.update(epochs=6, budget=8, batch=8, evaluation_batch=4)
    out = tmp_path / "macro"
    out.mkdir()
    meta["run_directory"] = str(out)
    atomic_json(meta, out / "manifest.json")
    cls = builder_class(stats)
    atomic_json(
        {"rows": cls.plan, "eligible": len(cls.plan)}, run.root(meta) / "train_cross_plan.json"
    )
    stack.enter_context(patch.object(run.xr.xd, "Builder", cls))
    stack.enter_context(patch.object(run, "plan_rows", return_value=cls.plan))
    stack.enter_context(
        patch.object(run, "construct", side_effect=lambda *a: (copy.deepcopy(m), copy.deepcopy(q)))
    )
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    stack.enter_context(patch.object(gd, "verify", return_value=None))
    atomic_json({"status": "eligible", "qualified": True}, out / "audit_status.json")
    for seed in (42, 43):
        rng = np.random.default_rng(seed)
        truth = np.tile(rng.normal(size=(1, 60, 2, 7)), (3, 1, 1, 1))
        payload = {
            "rows": rows(60),
            "predictions": truth.tolist(),
            "targets": truth.tolist(),
            "duration": [[30, 30]] * 60,
        }
        name = f"s{seed}_macro_baseline.json"
        atomic_json(payload, out / name)
        atomic_json(
            dict(
                ev.gate_metrics(payload),
                manifest_sha256=run.sha256(out / "manifest.json"),
                files={name: run.sha256(out / name)},
            ),
            out / f"s{seed}_macro_gate.json",
        )
    return meta, out, m, q, d, stats, local, cls


def test_physical_targets_future_and_anchor_independence():
    _, _, _, stats, _ = fixture()
    cls = builder_class(stats)
    schedule = (
        np.arange(4),
        np.array([1, 1, 16, 16]),
        np.full((4, 1), 128),
        np.full((4, 1), 128),
        {},
    )
    b, _, _, _, g, active = run.make_batch(cls(), schedule, 0, 4, "cpu")
    assert active.all() and (g["ages"][:, g["mask"]] > 0).all()
    y = core.targets(b, g, stats)
    torch.testing.assert_close(y[0], y[2], atol=2e-6, rtol=2e-5)
    changed = {k: v.clone() for k, v in b.items()}
    changed["y"][:, -1, 0] = 1
    torch.testing.assert_close(core.targets(changed, g, stats), y, atol=2e-6, rtol=2e-5)
    changed = {k: v.clone() for k, v in b.items()}
    changed["y"][8:, 80:110, 0] += torch.arange(30)
    with pytest.raises(ValueError, match="differ"):
        core.targets(changed, g, stats)
    bad = copy.deepcopy(g)
    bad["ages"][0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="historical"):
        core.targets(b, bad, stats)


def test_geometry_rejects_ambiguous_future_incomplete_and_preserves_bins():
    r = rows(1)[0]
    g, _ = gd.tensors([r], [16], "cpu")
    assert g["mask"].sum() == 32 and torch.all(g["bins"][0, 0].bincount() == 4)
    assert torch.equal(g["ages"][1] - g["ages"][0], torch.full_like(g["ages"][0], 16))
    bad = copy.deepcopy(r)
    bad["links"].append([1, 2, 80, 81])
    with pytest.raises(ValueError, match="Ambiguous"):
        gd.geometry(bad)
    bad = copy.deepcopy(r)
    bad["links"] = bad["links"][-8:]
    assert gd.geometry(bad) is None
    with pytest.raises(ValueError):
        gd.geometry(r, shift=0)
    a = gd.schedule(rows(), 42, 9, 128)
    b = gd.schedule(rows(), 42, 9, 128)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("mode", core.MODES)
def test_frozen_reader_encoder_routes_and_microbatch_equivalence(tmp_path, mode):
    with ExitStack() as stack:
        meta, _, m, q, _, stats, local, cls = lifecycle(tmp_path, stack)
        schedule = run.plan(meta, 42, 1, len(cls.plan))
        dm, dq = copy.deepcopy(m), copy.deepcopy(q)
        reader = copy.deepcopy(q).requires_grad_(False)
        signature = run.ur.bb.state_signature(reader)
        b, ps, qs, sh, g, active = run.make_batch(cls(), schedule, 0, 8, "cpu")
        y = core.targets(b, g, stats)
        parts, z = core.losses(dm, dq, b, ps, qs, sh, stats, local)
        aux, _, _ = core.auxiliary(reader, z, g, y, active, mode, stats)
        (parts["optimization_value"] + aux).mean().backward()
        expected = []
        for group in core.groups(dm, dq):
            torch.nn.utils.clip_grad_norm_(group, 1)
            expected.extend([p.grad.clone() if p.grad is not None else None for p in group])
        for micro in (2, 4, 8):
            mm, qq = copy.deepcopy(m), copy.deepcopy(q)
            result, steps = run.train_epoch(
                dict(meta, micro=micro),
                {"seed": 42, "mode": mode},
                mm,
                qq,
                reader,
                cls(),
                schedule,
                None,
                "cpu",
            )
            for p, e in zip([p for gr in core.groups(mm, qq) for p in gr], expected, strict=True):
                if e is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad, e, atol=3e-5, rtol=3e-4)
            assert steps == 1 and np.isfinite(result["optimization_value"])
        assert all(p.grad is None for p in reader.parameters())
        assert run.ur.bb.state_signature(reader) == signature
        z = torch.randn(24, 8, requires_grad=True)
        aux, _, _ = core.auxiliary(reader, z, g, y, active, mode, stats)
        aux.mean().backward()
        assert bool(z.grad.abs().sum() > 0) == (mode != "control")


def test_gate_prevents_worker_and_tampering(tmp_path):
    with ExitStack() as stack:
        _, out, *_ = lifecycle(tmp_path, stack)
        atomic_json({"status": "blocked", "qualified": False}, out / "audit_status.json")
        with (
            patch.object(run, "construct", side_effect=AssertionError("No model")),
            pytest.raises(ValueError, match="gate"),
        ):
            run.worker(out, "control_s42", "cpu")


def test_macro_counterfactual_and_collapse_rejection():
    rng = np.random.default_rng(33)
    y = np.tile(rng.normal(size=(1, 120, 2, 7)), (3, 1, 1, 1))
    p = {
        "rows": [{"key": f"X/15/C{i // 20}"} for i in range(120)],
        "duration": [[30, 30]] * 120,
        "targets": y.tolist(),
        "predictions": y.tolist(),
    }
    scores = ev.scores(p)
    assert np.all(scores["correct"] == 0) and scores["wrong_query"].mean() > 0
    p["predictions"] = np.zeros_like(y).tolist()
    scores = ev.scores(p)
    assert np.array_equal(scores["correct"], scores["zero"]) and np.all(scores["cross"] == 0)
    p["predictions"] = np.tile(y.mean(2, keepdims=True), (1, 1, 2, 1)).tolist()
    scores = ev.scores(p)
    assert np.array_equal(scores["correct"], scores["wrong_query"])


def test_worker_resume_and_equal_source_locks(tmp_path):
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
            run.worker(out, "control_s42", "cpu")
        state = torch.load(out / "control_s42/last.pt", weights_only=True)
        assert state["epoch"] == 1 and len(state["optimizers"]) == 3
        for job in meta["experiments"]:
            run.worker(out, job["name"], "cpu")
        run.lock_models(meta, out)
        lock = ev.check_models(meta, out)
        assert len(lock["weights"]) == 6
        with patch.object(run, "construct", side_effect=AssertionError("no rerun")):
            run.worker(out, "control_s42", "cpu")


def test_complete_original_utility_and_macro_pipeline(tmp_path):
    with ExitStack() as stack:
        meta, out, model, q, d, stats, local, _ = lifecycle(tmp_path, stack)
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
        # Synthetic physically matched price fixture, real grounded evaluator and readers.
        shared_rows = rows(8) + rows(8, "30->60")

        def evaluation_batch(meta, rs, split, shift, device, bands=core.TRAIN_BANDS):
            cls = builder_class(stats)
            count = len(rs)
            a, b, c, _, _ = cls()(np.arange(count), np.full(count, shift))
            batch = run.parent.batch_tensors(
                {k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device
            )
            geometry, active = gd.tensors(rs, np.full(count, shift), device, bands)
            return batch, geometry, active, core.targets(batch, geometry, stats)

        stack.enter_context(patch.object(gd, "evaluation_batch", side_effect=evaluation_batch))
        for split in ev.SPLITS:
            atomic_json(
                {"rows": shared_rows, "bands": [list(b) for b in core.HELD_BANDS], "shift": 8},
                out / f"{split}_macro_plan.json",
            )
        ev.audit_sources(meta, out, "cpu")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert (
            result["status"] == "matrix_complete_no_automatic_promotion"
            and not result["automatic_promotion"]
        )
        assert set(result["branches"]) == {"content", "aligned"}
        assert len(result["checks"]) == 744
        assert len(result["macro_checks"]) == 464
        assert all(
            b["macro_status"] == "no_balanced_macro_upgrade" for b in result["branches"].values()
        )
        for split in ev.SPLITS:
            assert len(run.read_json(out / f"{split}_readouts.json")["predictions"]) == 17
        lock = run.read_json(out / "model_selection_lock.json")
        lock["weights"]["control_s42"]["best"]["epoch"] = 999
        atomic_json(lock, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="Weight lock"):
            ev.check_models(meta, out)


@pytest.mark.parametrize("code,status", [(3, "blocked"), (7, "failed")])
def test_shell_exports_blocked_and_failed_run_and_preserves_receipt(tmp_path, code, status):
    import os
    import subprocess
    import tarfile

    script = Path(__file__).resolve().parents[2] / "scripts/babel_macro_history768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_MACRO_HISTORY768.md", "BABEL_RESEARCH_GOAL.md"):
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
        BABEL_MACRO_HISTORY_RUN=str(out),
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


def test_completed_readout_source_must_replay_original_identity(tmp_path):
    from test_grounded_readout_audit import synthetic_source

    with ExitStack() as stack:
        ground, meta = synthetic_source(tmp_path, stack)
        audit_out = tmp_path / "audit"
        run.audit.run(ground, audit_out, "cpu")
        identity = run.source_identity(audit_out)
        assert identity["grounded_source"] == str(ground)
        assert identity["source"] == meta["task_source"]["source"]
        (audit_out / "metrics.json").write_text("{}")
        with pytest.raises(ValueError, match="SHA256"):
            run.source_identity(audit_out)


def test_gate_rejects_forged_pass_and_payload_change(tmp_path):
    with ExitStack() as stack:
        _, out, *_ = lifecycle(tmp_path, stack)
        run.require_gate(out)
        p = out / "s42_macro_gate.json"
        r = run.read_json(p)
        r["checks"][0]["passed"] = False
        atomic_json(r, p)
        with pytest.raises(ValueError, match="arithmetic"):
            run.require_gate(out)


def test_data_preparation_binds_all_source_plans_and_support(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        source = Path(meta["task_source"]["source"])
        raw = run.root(meta)
        rs = rows(60) + rows(60, "30->60")
        for split in ("train", "test", "cross_research"):
            atomic_json({"rows": rs}, raw / f"{split}_cross_plan.json")
        atomic_json({"rows": rs}, source / "val_grounded_plan.json")
        gd.prepare(meta, out)
        REAL_DATA_VERIFY(meta, out)
        assert len(run.read_json(out / "val_macro_plan.json")["rows"]) == 60
        plan = raw / "test_cross_plan.json"
        r = run.read_json(plan)
        r["rows"][0]["end"] = "changed"
        atomic_json(r, plan)
        with pytest.raises(ValueError, match="source plan"):
            REAL_DATA_VERIFY(meta, out)


def test_raw_physical_coarse_path_is_independent_of_anchor_and_unused_nodes():
    stats = {"delta_scale": [0.2], "y_mean": [0] * 7, "y_scale": [1] * 7}
    geom, _ = gd.tensors(rows(2), [1, 16], "cpu")
    ages = torch.arange(1, 128, dtype=torch.float64)
    pred = torch.randn(6, 127, 7, dtype=torch.float64)
    first = core.describe(pred, geom, stats)
    changed = pred.clone()
    changed[:, :, 0] += 17 / (ages.sqrt() * 0.2)
    torch.testing.assert_close(core.describe(changed, geom, stats), first, atol=1e-13, rtol=1e-13)
    # One scalar vector per state; other rows cannot influence this row's readout.
    changed = pred.clone()
    changed[1::2] += 100
    torch.testing.assert_close(
        core.describe(changed, geom, stats)[:, 0], first[:, 0], atol=0, rtol=0
    )


@pytest.mark.parametrize("passed", [True, False])
def test_one_command_orchestration_gate_and_auto_worker_budget(tmp_path, passed):
    with ExitStack() as stack:
        meta, oldout, *_ = lifecycle(tmp_path, stack)
        source = tmp_path / "audit_entry"
        source.mkdir()
        out = tmp_path / "new_run"
        identity = meta["task_source"]
        atomic_json(
            {"torch": str(torch.__version__), "numpy": np.__version__},
            Path(identity["source"]) / "runtime.json",
        )
        stack.enter_context(patch.object(run, "source_identity", return_value=identity))
        stack.enter_context(patch.object(run, "make_manifest", return_value=copy.deepcopy(meta)))
        stack.enter_context(patch.object(run, "check_disk"))
        stack.enter_context(patch.object(run.parent.warm, "check_output"))
        stack.enter_context(patch.object(run.xr.xd, "verify_data"))
        stack.enter_context(patch.object(gd, "prepare"))
        stack.enter_context(patch.object(ev, "audit_sources"))
        stack.enter_context(patch.object(ev, "baseline_gate", return_value={"passed": passed}))
        stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="synthetic"))
        stack.enter_context(patch.object(torch.cuda, "empty_cache"))
        preflight = stack.enter_context(patch.object(run, "preflight", return_value=1))
        stack.enter_context(patch.object(run, "require_gate"))
        dispatch = stack.enter_context(patch.object(run, "dispatch"))
        lock = stack.enter_context(patch.object(run, "lock_models"))
        fit = stack.enter_context(patch.object(ev, "fit_readouts"))
        evaluate = stack.enter_context(patch.object(ev, "evaluate"))
        if passed:
            run.run(source, out, jobs=2, micro=2)
            assert dispatch.call_args.args[-1] == 1
            assert lock.call_count == fit.call_count == evaluate.call_count == 1
            assert run.read_json(out / "completion.json")["source_unchanged"]
            # Complete reentry only verifies immutable outputs; no repeated fitting.
            run.run(source, out, jobs=2, micro=2)
            assert dispatch.call_count == fit.call_count == 1
        else:
            with pytest.raises(SystemExit) as e:
                run.run(source, out, jobs=2, micro=2)
            assert e.value.code == 3
            assert run.read_json(out / "audit_status.json")["status"] == "blocked"
            assert (
                preflight.call_count
                == dispatch.call_count
                == lock.call_count
                == fit.call_count
                == evaluate.call_count
                == 0
            )
            assert not (out / "completion.json").exists()


def test_frozen_reader_cannot_be_substituted_by_current_head():
    _, q, _, stats, _ = fixture()
    geom, active = gd.tensors(rows(4), [1] * 4, "cpu")
    z = torch.randn(12, 8, requires_grad=True)
    target = torch.zeros(3, 4, 2, 7)
    with pytest.raises(ValueError, match="frozen"):
        core.auxiliary(q, z, geom, target, active, "content", stats)


def test_adapted_coarse_node_groups_keep_gaps_without_interpolation():
    r = rows(1)[0]
    r["links"] = [x for x in r["links"] if x[2] % 7 in (0, 2)]
    g, _ = gd.tensors([r], [8], "cpu")
    nodes = {(a, c) for a, b, c, d in r["links"]} | {(b, d) for a, b, c, d in r["links"]}
    for q in range(2):
        valid = g["mask"][0, q]
        actual = list(
            zip(
                (127 - g["ages"][1, 0, q, valid]).tolist(),
                (127 - g["ages"][2, 0, q, valid]).tolist(),
                strict=True,
            )
        )
        assert set(actual).issubset(nodes)
        counts = g["bins"][0, q, valid].bincount(minlength=4)
        assert counts.min() >= 2 and counts.max() - counts.min() <= 1


def test_strict_original_restore_removes_only_unused_auxiliary_reader(tmp_path):
    with ExitStack() as stack:
        meta, source, m, q, d, _, _, _ = old_lifecycle(tmp_path, stack)
        for job in meta["experiments"]:
            run.prior.worker(source, job["name"], "cpu")
        run.prior.lock_models(meta, source)
        atomic_json(
            {
                "status": "complete",
                "source_unchanged": True,
                "files": {
                    str(p.relative_to(source)): run.sha256(p)
                    for p in source.rglob("*")
                    if p.is_file()
                },
            },
            source / "completion.json",
        )
        identity = run.grounded.source_identity(source)
        new = run.make_manifest(source, identity, 2)
        model, reader = run.construct(new, 42, "cpu")
        assert not hasattr(reader, "shared")
        with torch.no_grad():
            x = torch.tensor(d["x"])
            expected = q(m.core.encoder(x)[:, -1])
            torch.testing.assert_close(
                reader(model.core.encoder(x)[:, -1]), expected, atol=0, rtol=0
            )
        path = source / identity["chosen"]["42"]["path"]
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="checkpoint changed"):
            run.construct(new, 42, "cpu")
