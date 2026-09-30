"""Synthetic-only qualification, gradients, recovery and scientific controls."""

import copy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_history_query import fixture, make_builder
from test_overlap_diagnostic import fixture_data
from test_shared_history import lifecycle as old_lifecycle
from test_shared_history import rows

from obson.babel import grounded_history as core
from obson.babel import grounded_history_data as gd
from obson.babel import grounded_history_evaluate as ev
from obson.babel import grounded_history_run as run
from obson.babel import grounded_history_warmup as qw
from obson.babel.ae_extend import atomic_json, atomic_save


@pytest.fixture(autouse=True)
def no_real_training():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def builder_class(stats):
    base = make_builder(stats)

    class Builder(base):
        plan = rows()

        def __call__(self, ids, shifts):
            a, b, c, _, _ = super().__call__(ids, np.where(shifts == 8, 1, shifts))
            for v, kind in ((a, "a"), (b, "b"), (c, "c")):
                for i, (idx, shift) in enumerate(zip(ids, shifts, strict=True)):
                    t = np.arange(128, dtype=float)
                    if kind == "a":
                        t -= shift
                    if kind == "c":
                        t = 2 * t - 127
                    v["y"][i, :, 0] = np.sin((t + int(idx)) / 8) * 0.02
                    v["mask"][i, :, 0] = True
            return a, b, c, None, None

    return Builder


def lifecycle(tmp_path, stack):
    _, source, m, q, d, stats, local, _ = old_lifecycle(tmp_path, stack)
    identity = {
        "source": str(source),
        "files": {"manifest.json": run.sha256(source / "manifest.json")},
        "chosen": {str(s): {"sha256": f"seed{s}"} for s in (42, 43)},
    }
    meta = run.make_manifest(source, identity, 2)
    meta.update(epochs=6, qualification_epochs=2, budget=8, batch=8, evaluation_batch=4)
    meta["shared"]["width"] = 4
    out = tmp_path / "grounded"
    out.mkdir()
    meta["run_directory"] = str(out)
    atomic_json(meta, out / "manifest.json")
    core.install(q, 42, 4)
    cls = builder_class(stats)
    stack.enter_context(patch.object(run.xr.xd, "Builder", cls))
    stack.enter_context(patch.object(run, "plan_rows", return_value=cls.plan))
    stack.enter_context(
        patch.object(run, "construct", side_effect=lambda *a: (copy.deepcopy(m), copy.deepcopy(q)))
    )
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    for seed in (42, 43):
        p = out / f"qualification_s{seed}"
        p.mkdir()
        atomic_json({"passed": True}, p / "qualification.json")
        atomic_json({"mean": 0.0, "scale": 0.01}, p / "target_stats.json")
        atomic_save(
            {
                "head": q.shared.state_dict(),
                "passed": True,
                "target_stats": {"mean": 0.0, "scale": 0.01},
            },
            p / "selected.pt",
        )
        atomic_json(
            {
                "identity": {"manifest": run.sha256(out / "manifest.json")},
                "files": {
                    n: run.sha256(p / n)
                    for n in ("qualification.json", "selected.pt", "target_stats.json")
                },
            },
            p / "completion.json",
        )
    return meta, out, m, q, d, stats, local, cls


def test_target_geometry_and_future_exclusion():
    _, _, _, stats, _ = fixture()
    cls = builder_class(stats)
    schedule = (
        np.arange(4),
        np.array([1, 1, 16, 16]),
        np.full((4, 1), 128),
        np.full((4, 1), 128),
        {},
    )
    b, _, _, _, a, m, _ = run.make_batch(cls(), schedule, 0, 4, "cpu")
    q = core.queries(a, m)
    assert q.shape == (3, 4, 2, 2, 2)
    y = core.targets(b, q, stats)
    torch.testing.assert_close(y[0], y[2], atol=2e-7, rtol=2e-5)
    changed = {k: v.clone() for k, v in b.items()}
    changed["y"][:, -1, 0] = 999
    torch.testing.assert_close(core.targets(changed, q, stats), y, atol=0, rtol=0)
    changed["y"][8:, 103:116, 0] += torch.arange(13) * 100
    with pytest.raises(ValueError, match="disagree"):
        core.targets(changed, q, stats)
    m[:, 2:] = False
    with pytest.raises(ValueError, match="four"):
        core.queries(a, m)


def test_decoder_has_no_query_or_state_bypass_and_original_head_unchanged():
    model, q, d, _, _ = fixture()
    with torch.no_grad():
        z = model.core.encoder(torch.tensor(d["x"]))[:, -1]
        before = q(z)
        core.install(q, 42, 4)
        torch.testing.assert_close(q(z), before, atol=0, rtol=0)
        coords = torch.tensor([[[24, 23], [23, 22]]] * len(z))
        r, pred = q.shared(z, coords)
        torch.testing.assert_close(q.shared.decoder(r), pred, atol=0, rtol=0)
        _, base = q.shared(z, coords, "state_only")
        _, wrong = q.shared(z, coords - 8, "state_only")
        torch.testing.assert_close(base, wrong, atol=0, rtol=0)


def test_qualification_rejects_query_blind_and_query_only_shortcuts():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(3, 120, 2, 2))
    y[:] = y[1]
    cohort = [{"key": f"X/15/C{i // 20}"} for i in range(120)]
    zero = np.zeros_like(y)
    blind = np.repeat(y.mean(2, keepdims=True), 2, axis=2)
    report = qw.qualify({"none": blind, "state_only": blind, "query_only": zero}, y, cohort)
    assert not report["passed"]
    assert all(not c["passed"] for c in report["checks"] if c["reference"] == "wrong_query")
    assert qw.qualify({"none": y, "state_only": zero, "query_only": zero}, y, cohort)["passed"]
    assert not qw.qualify({"none": y, "state_only": zero, "query_only": y}, y, cohort)["passed"]
    same = np.repeat(y[:, :, :1], 2, axis=2)
    assert not qw.qualify(dict.fromkeys(qw.ABLATIONS, same), same, cohort)["passed"]


@pytest.mark.parametrize("mode", core.MODES)
def test_exact_microbatch_gradients_and_declared_encoder_routes(tmp_path, mode):
    with ExitStack() as stack:
        meta, out, model, q, _, stats, local, cls = lifecycle(tmp_path, stack)
        schedule = run.plan(meta, 42, 1, 8)
        dm, dq = copy.deepcopy(model), copy.deepcopy(q)
        b, ps, qs, sh, a, m, active = run.make_batch(cls(), schedule, 0, 8, "cpu")
        coords = core.queries(a, m)
        target = core.targets(b, coords, stats) / 0.01
        parts, z = core.losses(dm, dq, b, ps, qs, sh, stats, local)
        aux, _ = core.auxiliary(dq, z, coords, target, active, mode)
        (parts["optimization_value"] + aux).mean().backward()
        expected = []
        for g in core.groups(dm, dq):
            torch.nn.utils.clip_grad_norm_(g, 1)
            expected.extend([p.grad.clone() if p.grad is not None else None for p in g])
        for micro in (2, 4, 8):
            mm, qq = copy.deepcopy(model), copy.deepcopy(q)
            result, steps = run.train_epoch(
                dict(meta, micro=micro),
                {"seed": 42, "mode": mode},
                mm,
                qq,
                cls(),
                schedule,
                None,
                "cpu",
            )
            for p, e in zip([p for g in core.groups(mm, qq) for p in g], expected, strict=True):
                if e is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad, e, atol=3e-5, rtol=3e-4)
            assert steps == 1 and np.isfinite(result["optimization_value"])
        z = torch.randn(24, 8, requires_grad=True)
        aux, _ = core.auxiliary(q, z, coords, target, active, mode)
        aux.mean().backward()
        if mode == "control":
            assert torch.count_nonzero(z.grad) == 0
        else:
            assert z.grad.abs().sum() > 0


def test_gate_blocks_workers_before_model_construction(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        p = out / "qualification_s42"
        atomic_json({"passed": False}, p / "qualification.json")
        done = run.read_json(p / "completion.json")
        done["files"]["qualification.json"] = run.sha256(p / "qualification.json")
        atomic_json(done, p / "completion.json")
        with (
            patch.object(run, "construct", side_effect=AssertionError("must not construct")),
            pytest.raises(ValueError, match="prohibited"),
        ):
            run.worker(out, "control_s42", "cpu")


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
        assert state["epoch"] == 1 and len(state["optimizers"]) == 4
        for job in meta["experiments"]:
            run.worker(out, job["name"], "cpu")
        run.lock_models(meta, out)
        lock = ev.check_models(meta, out)
        assert len(lock["weights"]) == 6
        with patch.object(run, "construct", side_effect=AssertionError("no rerun")):
            run.worker(out, "control_s42", "cpu")


def test_complete_original_utility_and_grounded_pipeline(tmp_path):
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

        def evaluation_batch(meta, rs, split, shift, device, band=(103, 115)):
            cls = builder_class(stats)
            count = len(rs)
            a, b, c, _, _ = cls()(np.arange(count), np.full(count, shift))
            batch = run.parent.batch_tensors(
                {k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device
            )
            ages, mask, active = run.shared_data.tensors(
                rs, np.full(count, shift), device, band, shift
            )
            coords = core.queries(ages, mask)
            return batch, coords, active, core.targets(batch, coords, stats)

        stack.enter_context(patch.object(gd, "evaluation_batch", side_effect=evaluation_batch))
        for split in ev.SPLITS:
            atomic_json(
                {"rows": shared_rows, "band": [99, 111], "shift": 8},
                out / f"{split}_grounded_plan.json",
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
        assert len(result["shared_checks"]) == 48
        assert all(
            b["status"] == "no_shared_representation_upgrade" for b in result["branches"].values()
        )
        for split in ev.SPLITS:
            assert len(run.read_json(out / f"{split}_readouts.json")["predictions"]) == 17
        lock = run.read_json(out / "model_selection_lock.json")
        lock["weights"]["control_s42"]["best"]["epoch"] = 999
        atomic_json(lock, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="Weight lock"):
            ev.check_models(meta, out)


def test_frozen_warmup_resume_fail_gate_and_immutable_binding(tmp_path):
    with ExitStack() as stack:
        meta, out, _, _, _, stats, _, cls = lifecycle(tmp_path, stack)
        p = out / "qualification_s42"
        (p / "completion.json").unlink()
        atomic_json({"synthetic": True}, p / "cache_lock.json")
        schedule = run.plan(meta, 42, 0, 8)
        b, _, _, _, ages, mask, _ = run.make_batch(cls(), schedule, 0, 8, "cpu")
        coords = core.queries(ages, mask)
        y = core.targets(b, coords, stats) / 0.01
        dataset = {"z": torch.randn(3, 8, 8), "q": coords, "y": y, "rows": cls.plan}
        cache = {"train": dataset, "val": dataset, "target_stats": {"mean": 0.0, "scale": 0.01}}
        stack.enter_context(patch.object(qw, "cache", return_value=cache))
        calls = 0

        def interrupt(*args, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("warmup interrupted")

        with (
            patch.object(torch.optim.AdamW, "step", side_effect=interrupt),
            pytest.raises(RuntimeError, match="warmup interrupted"),
        ):
            qw.fit(meta, out, 42, "cpu")
        state = torch.load(p / "none_resume.pt", weights_only=True)
        assert state["epoch"] == 1
        report = qw.fit(meta, out, 42, "cpu")
        assert not report["passed"]
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No refit")):
            assert qw.fit(meta, out, 42, "cpu") == report
        with pytest.raises(ValueError, match="prohibited"):
            qw.require_passed(out)
        (p / "selected.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="fingerprint"):
            qw.fit(meta, out, 42, "cpu")


def test_completed_shared_source_binding_and_reader_replacement(tmp_path):
    with ExitStack() as stack:
        meta, source, _, _, _, _, _, _ = old_lifecycle(tmp_path, stack)
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
        identity = run.source_identity(source)
        assert set(identity["chosen"]) == {"42", "43"}
        new = run.make_manifest(source, identity, 2)
        m, q = run.construct(new, 42, "cpu")
        assert isinstance(q.shared, core.GroundedReader)
        p = source / identity["chosen"]["42"]["path"]
        p.write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA256"):
            run.source_identity(source)


@pytest.mark.parametrize("code,status", [(3, "blocked"), (7, "failed")])
def test_shell_exports_blocked_and_failed_run_and_preserves_receipt(tmp_path, code, status):
    import os
    import subprocess
    import tarfile

    script = Path(__file__).resolve().parents[2] / "scripts/babel_grounded_history768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_GROUNDED_HISTORY768.md", "BABEL_RESEARCH_GOAL.md"):
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
        BABEL_GROUNDED_HISTORY_RUN=str(out),
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


def test_training_cache_normalization_never_uses_validation_targets(tmp_path):
    with ExitStack() as stack:
        meta, out, _, _, _, stats, _, cls = lifecycle(tmp_path, stack)
        p = out / "qualification_s42"
        atomic_json({"rows": cls.plan}, out / "val_grounded_plan.json")
        atomic_json({"synthetic": True}, out / "grounded_data_lock.json")

        def val_batch(*args, **kw):
            rs = args[1]
            schedule = (
                np.arange(len(rs)),
                np.ones(len(rs), int),
                np.full((len(rs), 1), 128),
                np.full((len(rs), 1), 128),
                {},
            )
            b, _, _, _, a, m, active = run.make_batch(cls(), schedule, 0, len(rs), "cpu")
            coords = core.queries(a, m)
            target = core.targets(b, coords, stats) + 100
            return b, coords, active, target

        stack.enter_context(patch.object(gd, "evaluation_batch", side_effect=val_batch))
        result = qw.cache(meta, out, 42, "cpu")
        train = result["train"]["y"].double()
        assert abs(float(train.mean())) < 1e-6 and float(
            train.std(unbiased=False)
        ) == pytest.approx(1, rel=1e-5)
        assert result["val"]["y"].mean() > 100
        assert abs(result["target_stats"]["mean"]) < 1
        with patch.object(run, "construct", side_effect=AssertionError("No re-encoding")):
            qw.cache(meta, out, 42, "cpu")
        (p / "cache.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="fingerprint"):
            qw.cache(meta, out, 42, "cpu")


def test_failed_qualification_stops_top_level_before_dispatch(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        source = Path(meta["task_source"]["source"])
        atomic_json(
            {"torch": str(torch.__version__), "numpy": np.__version__}, source / "runtime.json"
        )
        stack.enter_context(patch.object(run, "source_identity", return_value=meta["task_source"]))
        stack.enter_context(patch.object(run, "make_manifest", return_value=copy.deepcopy(meta)))
        stack.enter_context(patch.object(run, "check_disk"))
        stack.enter_context(patch.object(run.xr.xd, "verify_data"))
        stack.enter_context(patch.object(gd, "prepare"))
        stack.enter_context(patch.object(ev, "audit_sources"))
        stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="synthetic"))
        stack.enter_context(patch.object(qw, "fit", return_value={"passed": False}))
        stack.enter_context(
            patch.object(run, "preflight", side_effect=AssertionError("No joint preflight"))
        )
        stack.enter_context(
            patch.object(run, "dispatch", side_effect=AssertionError("No encoder training"))
        )
        with pytest.raises(SystemExit) as stopped:
            run.run(source, out, 1, 2)
        assert stopped.value.code == 3
        report = run.read_json(out / "audit_status.json")
        assert report["status"] == "blocked" and not report["encoder_training_started"]
