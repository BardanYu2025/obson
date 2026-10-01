"""Synthetic gradients and mocked optimizer lifecycle; never fit real data locally."""

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

from obson.babel import position_supervision as core
from obson.babel import position_supervision_evaluate as ev
from obson.babel import position_supervision_run as run
from obson.babel import shared_history as old
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def synthetic_only():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def lifecycle(tmp_path, stack):
    _, source, model, query, d, stats, local, _ = old_lifecycle(tmp_path, stack)
    del query.shared
    identity = {
        "source": str(source),
        "files": {"manifest.json": run.sha256(source / "manifest.json")},
        "chosen": {str(s): {"sha256": f"seed{s}"} for s in (42, 43)},
    }
    meta = run.make_manifest(source, identity, 2)
    meta.update(epochs=6, budget=4, batch=4, evaluation_batch=2)
    out = tmp_path / "positions"
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
        patch.object(
            run, "construct", side_effect=lambda *a: (copy.deepcopy(model), copy.deepcopy(query))
        )
    )
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    return meta, out, model, query, d, stats, local, cls


def test_dense_weights_are_original_estimator_expectation():
    ps = torch.tensor(core.hq.sample_prefixes(3, 42, 1))
    dense, weights = core.position_weights(ps, "dense_matched")
    assert not torch.isin(dense, torch.tensor([48, 80, 112])).any()
    torch.testing.assert_close(weights.sum(1), torch.ones(3))
    assert weights[0, -1].item() == pytest.approx(0.2)
    assert weights[0, 0].item() == pytest.approx(4 / len(core.INTERIOR) / 5)
    # Exact linear expectation of the without-replacement sample sum, no MC assertion.
    arbitrary_losses = torch.arange(len(core.INTERIOR) + 1).square().float()
    expected = 0.8 * arbitrary_losses[:-1].mean() + 0.2 * arbitrary_losses[-1]
    torch.testing.assert_close((weights[0] * arbitrary_losses).sum(), expected)
    all_ps, all_w = core.position_weights(ps, "dense_all")
    assert all_ps[0].tolist() == list(range(2, 129))
    torch.testing.assert_close(all_w.sum(1), torch.ones(3))
    assert core.exposure("dense_all", 12)["query_contexts"] == 12 * 127
    for bad in (ps.float(), ps[:, :4], torch.full_like(ps, 128)):
        with pytest.raises(ValueError):
            core.position_weights(bad, "sampled")


def test_target_extension_equivalent_and_causal_in_short_prefixes():
    model, query, d, stats, _ = fixture()
    b = run.parent.batch_tensors(d, "cpu")
    ps = torch.tensor([[32, 48, 80, 128]] * len(b["x"]))
    a, am = core.targets(b, ps, stats)
    y, ym = core.hq.targets(b, ps, stats)
    torch.testing.assert_close(a, y, atol=0, rtol=0)
    assert torch.equal(am, ym)
    for p in (2, 3, 16, 31):
        ps = torch.full((len(b["x"]), 1), p)
        a, mask = core.targets(b, ps, stats)
        assert not mask[:, :, p - 1 :].any()
        assert mask[:, :, : p - 1, 0].all()
        altered = {k: v.clone() for k, v in b.items()}
        altered["y"][:, p:] += 10
        altered["x"][:, p:] -= 10
        torch.testing.assert_close(core.targets(altered, ps, stats)[0], a, atol=0, rtol=0)
        with torch.no_grad():
            z = model.core.encoder(b["x"])
            zz = model.core.encoder(altered["x"])
            torch.testing.assert_close(
                query(z[:, p - 1]), query(zz[:, p - 1]), atol=3e-6, rtol=3e-5
            )
    with pytest.raises(ValueError, match="no past"):
        core.targets(b, torch.ones((len(b["x"]), 1), dtype=torch.long), stats)


def dense_reference(model, query, b, ps, qs, shifts, stats, local, mode):
    z = model.core.encoder(b["x"])
    positions, w = core.position_weights(qs, mode)
    terms = core.query_terms(query, z, b, positions, stats)
    value = ((terms["query_fixed"] + terms["structure"]) * w).sum(1).mean()
    rows = torch.arange(len(z))[:, None]
    p = model.local_head(z.detach()[rows, ps - 1]).reshape(len(z), -1, 16, 7)
    y, mask = core.hq.ba.local_targets(b["y"], b["mask"], ps, stats, local)
    return value + 0.25 * core.hq.ba.local_rows(p, y, mask, True)["primary"].mean()


@pytest.mark.parametrize("mode", core.MODES)
def test_chunked_gradient_equals_full_objective_and_microbatch(tmp_path, mode):
    with ExitStack() as stack:
        meta, _, model, query, _, stats, local, cls = lifecycle(tmp_path, stack)
        schedule = run.plan(meta, 42, 1, len(cls.plan))
        b, ps, qs, sh = run.make_batch(cls(), schedule, 0, 4, "cpu")
        dm, dq = copy.deepcopy(model), copy.deepcopy(query)
        core.configure(dm, dq, True)
        loss = dense_reference(dm, dq, b, ps, qs, sh, stats, local, mode)
        if mode == "sampled":
            parts, _ = old.losses(dm, dq, b, ps, qs, sh, stats, local)
            torch.testing.assert_close(
                loss, parts["optimization_value"].mean(), atol=2e-6, rtol=2e-5
            )
        loss.backward()
        expected = []
        for _, g, _ in core.configure(dm, dq, True):
            torch.nn.utils.clip_grad_norm_(g, 1)
            expected.extend([p.grad.clone() if p.grad is not None else None for p in g])
        for micro, chunk in ((1, 1), (2, 7), (4, 32)):
            m, q = copy.deepcopy(model), copy.deepcopy(query)
            result, steps = run.train_epoch(
                dict(meta, micro=micro, position_chunk=chunk),
                {"mode": mode, "seed": 42},
                m,
                q,
                cls(),
                schedule,
                None,
                "cpu",
            )
            assert steps == 1
            assert result["optimization_value"] == pytest.approx(float(loss.detach()), rel=3e-6)
            actual = [p for _, g, _ in core.configure(m, q, True) for p in g]
            for p, e in zip(actual, expected, strict=True):
                if e is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad, e, atol=5e-5, rtol=7e-4)
            assert all(p.grad is None for p in m.core.decoder.parameters())
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.encoder.parameters()
            )


def test_resume_lock_replay_exposure_and_no_refit(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        original, calls = run.train_epoch, 0

        def interrupt(*args, **kw):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted")
            return original(*args, **kw)

        with (
            patch.object(run, "train_epoch", side_effect=interrupt),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            run.worker(out, "sampled_s42", "cpu")
        state = torch.load(out / "sampled_s42/last.pt", weights_only=True)
        assert state["epoch"] == 1 and len(state["optimizers"]) == 3
        for job in meta["experiments"]:
            run.worker(out, job["name"], "cpu")
        run.lock_models(meta, out)
        lock = ev.check_models(meta, out)
        assert len(lock["weights"]) == 6
        assert all(r["selected_epoch"] == 0 for r in lock["trials"].values())
        with patch.object(run, "construct", side_effect=AssertionError("No rerun")):
            run.worker(out, "sampled_s42", "cpu")
        state = run.read_json(out / "sampled_s42/training_state.json")
        state["history"][0]["exposure"]["query_contexts"] += 1
        with pytest.raises(ValueError, match="Sampling"):
            run.verify_history(meta, meta["experiments"][0], state, 8)
        # Completed weights must actually be checked, not just receipt existence.
        (out / "sampled_s42/e6_best.pt").write_bytes(b"tampered")
        with pytest.raises(ValueError):
            run.worker(out, "sampled_s42", "cpu")


def test_all_prefix_profile_and_same_manifest_budget(tmp_path):
    with ExitStack() as stack:
        meta, _, m, q, d, *_ = lifecycle(tmp_path, stack)
        profile = ev.position_profile(meta, m, q, d, "cpu")
        assert profile["prefixes"] == list(range(2, 129))
        assert np.asarray(profile["per_window"]["path"]).shape == (len(d["x"]), 127)
        assert profile["means"]["path"][-1] >= 0
        assert meta["schema"] == "babel-position-supervision768-v1"
        assert len(meta["experiments"]) == 6
        assert "calibration_epochs" not in meta
        assert len({tuple(run.rates(meta, i).values()) for i in range(1, 3)}) == 2
        assert Path(run.__file__).name in meta["code_sha256"]


def test_complete_original_utility_pipeline(tmp_path):
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
        ev.audit_sources(meta, out, "cpu")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert (
            result["status"] == "matrix_complete_no_automatic_promotion"
            and not result["automatic_promotion"]
        )
        assert set(result["branches"]) == {"dense_matched", "dense_all"}
        assert len(result["checks"]) > 800
        for split in ev.SPLITS:
            assert len(run.read_json(out / f"{split}_readouts.json")["predictions"]) == 17
        lock = run.read_json(out / "model_selection_lock.json")
        lock["weights"]["sampled_s42"]["best"]["epoch"] = 999
        atomic_json(lock, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="Weight lock"):
            ev.check_models(meta, out)


@pytest.mark.parametrize("code,status", [(3, "blocked"), (7, "failed")])
def test_shell_exports_blocked_and_failed_run_and_preserves_receipt(tmp_path, code, status):
    import os
    import subprocess
    import tarfile

    script = Path(__file__).resolve().parents[2] / "scripts/babel_position_supervision768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_POSITION_SUPERVISION768.md", "BABEL_RESEARCH_GOAL.md"):
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
        BABEL_POSITION_SUPERVISION_RUN=str(out),
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
            "position_profile": {
                "per_window": {
                    k: [[value] * 127 for _ in range(n)] for k in ("path", "change1", "body")
                }
            },
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
            name: rec(0.04 if name.startswith("dense_all") else 0.06)
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
    assert result["branches"]["dense_all"]["status"] == "balanced_utility_candidate"
    assert len(result["checks"]) > 800
    for split in ev.SPLITS:
        for seed in (42, 43):
            for kind in ("best", "last"):
                records[split][f"dense_matched_s{seed}_{kind}"]["utility"]["errors"]["utility"] = [
                    0.04
                ] * n
    result = ev.decide(meta, records, inventories, inventories, masks, {})
    assert not result["branches"]["dense_all"]["groups"]["utility_gain"]
    assert all(
        not c["passed"]
        for c in result["checks"]
        if c["metric"] == "utility/dense_matched" and c["group"] == "utility_gain"
    )


def test_main_orchestration_train_lock_fit_evaluate_and_readonly_reentry(tmp_path):
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
        stack.enter_context(patch.object(run, "lock_models"))
        fit = stack.enter_context(patch.object(ev, "fit_readouts"))
        stack.enter_context(patch.object(ev, "evaluate"))
        run.run(source, out, jobs=2, micro=2)
        assert trace == ["main"]
        assert dispatch.call_args.args[-1] == 1
        run.run(source, out, jobs=2, micro=2)
        assert dispatch.call_count == 1 and fit.call_count == 1


def test_construct_restores_wrapped_input_and_selected_query(tmp_path):
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
        assert run.ur.bb.state_signature(m) == run.ur.bb.state_signature(model)
        (source / "best.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="changed"):
            run.construct({"task_source": identity}, 42, "cpu")
