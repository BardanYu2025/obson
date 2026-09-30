"""Synthetic-only shared-history gradients, coordinates, batch semantics and lifecycle."""

import copy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_activity_weight import lifecycle as prior_lifecycle
from test_architecture_benchmark import features
from test_history_query import fixture, make_builder
from test_overlap_diagnostic import fixture_data

from obson.babel import shared_history as core
from obson.babel import shared_history_data as sd
from obson.babel import shared_history_evaluate as ev
from obson.babel import shared_history_run as run
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def no_real_training():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def rows(n=8, pair="15->30"):
    return [
        {
            "key": "X/15/X01",
            "pair": pair,
            "end": f"t{i}",
            "week": f"w{i // 2}",
            "links": [[2 * c - 127, 2 * (c + 1) - 127, c, c + 1] for c in range(98, 116)],
        }
        for i in range(n)
    ]


def lifecycle(tmp_path, stack):
    _, source, model, query, d, stats, local = prior_lifecycle(tmp_path, stack)
    plan = rows()
    atomic_json(
        {"rows": plan, "eligible": len(plan)},
        run.root(run.read_json(source / "manifest.json")) / "train_cross_plan.json",
    )
    identity = {
        "source": str(source),
        "files": {"manifest.json": run.sha256(source / "manifest.json")},
        "chosen": {str(s): {"sha256": f"seed{s}"} for s in (42, 43)},
    }
    meta = run.make_manifest(source, identity, micro=2)
    meta.update(epochs=6, budget=8, batch=8, evaluation_batch=4)
    meta["shared"]["width"] = 4
    out = tmp_path / "shared"
    out.mkdir()
    atomic_json(meta, out / "manifest.json")
    core.install(query, 42, latent=4)
    cls = make_builder(stats)
    cls.plan = plan
    stack.enter_context(patch.object(run.xr.xd, "Builder", cls))
    stack.enter_context(patch.object(run, "plan_rows", return_value=plan))
    stack.enter_context(
        patch.object(
            run, "construct", side_effect=lambda *a: (copy.deepcopy(model), copy.deepcopy(query))
        )
    )
    stack.enter_context(patch.object(run, "stats", return_value=(stats, local)))
    stack.enter_context(patch.object(run, "data", return_value=d))
    return meta, out, model, query, d, stats, local, cls


def test_shared_coordinates_no_future_and_identity_checks():
    r = rows(4)
    ages, mask, active = sd.tensors(r, np.array([1, 1, 16, 16]), "cpu")
    assert active.all() and (ages[:, mask] > 0).all()
    # A is earlier than B: same historical interval is closer to A's own endpoint.
    assert torch.equal(
        ages[1] - ages[0], torch.tensor([1, 1, 16, 16])[:, None, None].expand_as(ages[0])
    )
    bad = copy.deepcopy(r)
    bad[1]["key"] = "other"
    with pytest.raises(ValueError, match="identity"):
        sd.tensors(bad, np.ones(4, int), "cpu")
    reader = core.IntervalReader(8, 42, 4)
    wrong = ages[0].clone()
    wrong[0, 0, 1] = 0
    with pytest.raises(ValueError, match="Strictly past"):
        reader(torch.randn(4, 8), wrong, mask)


def test_reader_cannot_use_other_states_or_padded_geometry():
    _, q, _, _, _ = fixture()
    core.install(q, 42, 4)
    a, m, _ = sd.tensors(rows(4), np.ones(4, int), "cpu")
    m[:, -1] = False
    z = torch.randn(4, 8)
    before = q.shared(z, a[0], m)
    altered = z.clone()
    altered[1:] += 100
    torch.testing.assert_close(q.shared(altered, a[0], m)[0], before[0])
    coords = a[0].clone()
    coords[:, -1] = -999
    torch.testing.assert_close(q.shared(z, coords, m), before, atol=0, rtol=0)


@pytest.mark.parametrize("mode", core.MODES)
def test_encoder_routes_and_shared_head_train_in_all_arms(mode):
    _, q, _, _, _ = fixture()
    core.install(q, 42, 4)
    ages, mask, active = sd.tensors(rows(4), np.ones(4, int), "cpu")
    z = torch.randn(12, 8, requires_grad=True)
    f = core.reads(q, z, ages, mask, mode)
    loss, _ = core.auxiliary(f, active)
    loss.backward()
    if mode == "control":
        assert torch.count_nonzero(z.grad) == 0
    if mode == "within":
        assert z.grad[:8].abs().sum() > 0 and torch.count_nonzero(z.grad[8:]) == 0
    if mode == "cross":
        assert z.grad[8:].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in q.shared.parameters())


def test_nuisance_only_vectors_fail_variance_and_shuffle_diagnostic():
    # Different contracts/geometry may give different constant vectors. Within-pair
    # centering still identifies this shortcut as having no temporal content.
    z = torch.arange(16.0).reshape(4, 4).repeat_interleave(2, 0)
    var, cov = core.centered_statistics(z)
    assert var > 0.98 and cov == 0
    d = ev.diagnostics(z.numpy(), z.numpy())
    assert not d["noncollapsed"] and d["ratio"] is None and d["effective_rank"] == 0
    a = torch.randn(16, 4)
    v, c = core.centered_statistics(a)
    assert torch.isfinite(v + c)


@pytest.mark.parametrize("mode", core.MODES)
def test_cached_effective_batch_gradient_matches_direct_and_micro(tmp_path, mode):
    with ExitStack() as stack:
        meta, out, model, q, d, stats, local, cls = lifecycle(tmp_path, stack)
        schedule = run.plan(meta, 42, 1, 8)
        job = {"seed": 42, "mode": mode}
        direct_model, direct_q = copy.deepcopy(model), copy.deepcopy(q)
        b, ps, qs, sh, ages, mask, active = run.make_batch(cls(), schedule, 0, 8, "cpu")
        parts, z = core.losses(direct_model, direct_q, b, ps, qs, sh, stats, local)
        aux, _ = core.auxiliary(core.reads(direct_q, z, ages, mask, mode), active)
        (parts["optimization_value"].mean() + aux).backward()
        expected = []
        for group in core.groups(direct_model, direct_q):
            torch.nn.utils.clip_grad_norm_(group, 1.0)
            expected.extend([p.grad.clone() if p.grad is not None else None for p in group])
        for micro in (2, 4, 8):
            m, zq = copy.deepcopy(model), copy.deepcopy(q)
            result, steps = run.train_epoch(
                dict(meta, micro=micro), job, m, zq, cls(), schedule, None, "cpu"
            )
            actual = [p.grad for group in core.groups(m, zq) for p in group]
            for a, e in zip(actual, expected, strict=True):
                if e is None:
                    assert a is None
                else:
                    torch.testing.assert_close(a, e, atol=3e-5, rtol=3e-4)
            assert steps == 1 and np.isfinite(result["optimization_value"])
        # Control exactly retains the old reconstruction arithmetic and encoder gradient.
        legacy = core.reconstruction.losses(model, q, b, ps, qs, sh, stats, local, 0.3)
        torch.testing.assert_close(
            parts["optimization_value"], legacy["optimization_value"], atol=0, rtol=0
        )


def test_schedule_shared_across_modes_and_distinct_time_pairs(tmp_path):
    with ExitStack() as stack:
        meta, _, _, _, _, _, _, _ = lifecycle(tmp_path, stack)
        plans = [run.plan(meta, j["seed"], 1, 8) for j in meta["experiments"] if j["seed"] == 42]
        assert all(p[4] == plans[0][4] for p in plans)
        ids, shifts, _, qs, _ = plans[0]
        assert np.all(ids[::2] != ids[1::2]) and np.array_equal(shifts[::2], shifts[1::2])
        assert not np.isin(qs, (48, 80, 112)).any()


def test_resume_and_six_worker_selection_locks(tmp_path):
    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        job = meta["experiments"][0]
        original = run.train_epoch
        calls = 0

        def stop(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted")
            return original(*args, **kwargs)

        with (
            patch.object(run, "train_epoch", side_effect=stop),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            run.worker(out, job["name"], "cpu")
        state = torch.load(out / job["name"] / "last.pt", weights_only=True)
        assert state["epoch"] == 1 and len(state["optimizers"]) == 4 and "rng" in state
        for j in meta["experiments"]:
            run.worker(out, j["name"], "cpu")
        run.lock_models(meta, out)
        lock = ev.check_models(meta, out)
        assert len(lock["weights"]) == 6 and all(
            set(v) == {"best", "last"} for v in lock["weights"].values()
        )
        with patch.object(run, "construct", side_effect=AssertionError("No retrain")):
            run.worker(out, job["name"], "cpu")
        lock["weights"][job["name"]]["best"]["epoch"] = 99
        atomic_json(lock, out / "model_selection_lock.json")
        with pytest.raises(ValueError, match="Weight lock"):
            ev.check_models(meta, out)


def test_ratio_bootstrap_scale_invariance_and_complete_denominator():
    r = [{"key": f"X/15/C{i // 20}", "week": f"w{i // 2}"} for i in range(120)]
    a = {"distance": np.full(120, 0.2), "shuffled": np.ones(120)}
    b = {"distance": np.full(120, 0.8), "shuffled": np.ones(120)}
    ci = ev.ratio_interval(a, b, r, 0.9)
    scaled = {k: v * 100 for k, v in a.items()}
    other = ev.ratio_interval(scaled, b, r, 0.9)
    assert ci["supported"] and ci["high"] < 0
    assert ci["delta"] == pytest.approx(other["delta"])
    zero = {"distance": np.zeros(120), "shuffled": np.zeros(120)}
    assert not ev.ratio_interval(zero, b, r, 0.9)["supported"]


def test_complete_synthetic_pipeline_and_locks(tmp_path):
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
                atomic_json(ref, source / f"{split}_a030_s{seed}_best.json")
        # Real shared scoring over a synthetic packed raw bank; no evaluator mock.
        shared_rows = rows(8) + rows(8, "30->60")
        raw_values = features(20).reshape(-1, 28)
        bank_path = out / "synthetic_bank.npy"
        np.save(bank_path, raw_values)
        for i, row in enumerate(shared_rows):
            row.update(fine_bank_end=255 + i * 128, coarse_bank_end=383 + i * 128)
        stack.enter_context(patch.object(run.xr.xd, "bank_info", return_value=(bank_path, None)))
        for split in ev.SPLITS:
            for band_name, band, shift in (
                ("trained", core.TRAIN_BAND, 16),
                ("held", core.HELD_BAND, 8),
            ):
                atomic_json(
                    {
                        "rows": shared_rows,
                        "ids": list(range(16)),
                        "band": list(band),
                        "shift": shift,
                    },
                    out / f"{split}_{band_name}_shared_plan.json",
                )
        ev.audit_sources(meta, out, "cpu")
        ev.evaluate(meta, out, "cpu")
        result = run.read_json(out / "decision.json")
        assert (
            result["status"] == "matrix_complete_no_automatic_promotion"
            and not result["automatic_promotion"]
        )
        assert set(result["branches"]) == {"within", "cross"}
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


def test_shell_failure_and_manual_export_keep_failure_status(tmp_path):
    import os
    import subprocess
    import tarfile

    script = Path(__file__).resolve().parents[2] / "scripts/babel_shared_history768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_SHARED_HISTORY768.md", "BABEL_RESEARCH_GOAL.md"):
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
        BABEL_SHARED_HISTORY_RUN=str(out),
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


def test_geometry_support_lock_and_mutation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    atomic_json({"schema": "synthetic"}, out / "manifest.json")
    plan = rows(60) + rows(60, "30->60")
    for i, r in enumerate(plan):
        r["key"] = f"X/{r['pair'].split('->')[0]}/C{(i % 60) // 12}"
    for split in ("train", "test", "cross_research"):
        atomic_json({"rows": plan}, raw / f"{split}_cross_plan.json")
    sd.prepare({}, out, raw)
    lock = run.read_json(out / "shared_data_lock.json")
    sd.prepare({}, out, raw)
    assert run.read_json(out / "shared_data_lock.json") == lock
    (out / "test_held_shared_plan.json").write_text("{}")
    with pytest.raises(ValueError, match="fingerprint"):
        sd.prepare({}, out, raw)
    # Irregular actual aggregation links survive verbatim; no hardcoded ratio.
    row = rows(1)[0]
    row["links"][7][0] += 1
    g = sd.geometry(row)
    ages, mask, _ = sd.tensors([row, dict(row, end="different")], np.ones(2, int), "cpu")
    np.testing.assert_array_equal(127 - ages[1, 0, mask[0]].numpy(), np.array(g)[:, :2])


def test_install_keeps_original_decoder_function():
    model, q, d, stats, local = fixture()
    model.eval()
    q.eval()
    with torch.no_grad():
        state = model.core.encoder(torch.tensor(d["x"]))[:, -1]
        before = q(state)
        core.install(q, 42, 4)
        torch.testing.assert_close(q(state), before, atol=0, rtol=0)


def test_completed_source_checkpoint_binding(tmp_path):
    with ExitStack() as stack:
        meta, source, _, _, _, _, _ = prior_lifecycle(tmp_path, stack)
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
        p = source / identity["chosen"]["42"]["path"]
        p.write_bytes(b"changed")
        with pytest.raises(ValueError, match="SHA256"):
            run.source_identity(source)


def test_disk_failure_preserves_existing_sources(tmp_path):
    from types import SimpleNamespace

    with ExitStack() as stack:
        meta, out, *_ = lifecycle(tmp_path, stack)
        source = Path(meta["task_source"]["source"])
        hashes = {p: run.sha256(p) for p in source.rglob("*") if p.is_file()}
        with (
            patch.object(run.shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
            pytest.raises(ValueError, match="No source files deleted"),
        ):
            run.check_disk(meta, out, 2)
        assert all(run.sha256(p) == h for p, h in hashes.items())


def test_auxiliary_gain_alone_never_upgrades_state(tmp_path):
    cohort = [{"key": f"X/15/C{i // 20}", "week": f"w{i // 2}"} for i in range(120)]

    def scores(distance):
        return {
            "distance": [distance] * 120,
            "shuffled": [1.0] * 120,
            "ratio": distance,
            "noncollapsed": True,
        }

    bank = {}
    for mode, distance in (("control", 0.5), ("within", 0.2), ("cross", 0.05)):
        for seed in (42, 43):
            for kind in ("best", "last"):
                bank[f"{mode}_s{seed}_{kind}"] = {
                    "held": {
                        p: {"rows": cohort, "within": scores(distance), "cross": scores(distance)}
                        for p in ("15->30", "30->60")
                    }
                }
    result = {
        "branches": {
            n: {"groups": {"retention": True, "activity_retention": True, "utility_gain": True}}
            for n in ("within", "cross")
        }
    }
    scored = ev.shared_decision(copy.deepcopy(result), dict.fromkeys(ev.SPLITS, bank), tmp_path)
    assert len(scored["shared_checks"]) == 48
    assert all(
        b["status"] == "shared_representation_candidate" for b in scored["branches"].values()
    )
    result["branches"]["cross"]["groups"]["utility_gain"] = False
    scored = ev.shared_decision(result, dict.fromkeys(ev.SPLITS, bank), tmp_path)
    assert scored["branches"]["cross"]["status"] == "no_shared_representation_upgrade"
    assert not scored["automatic_promotion"]
