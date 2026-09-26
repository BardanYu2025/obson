"""Synthetic warm-start pipeline; neural optimizer steps are always no-ops."""

import copy
import os
import subprocess
import tarfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from test_capacity_growth import small

from obson.babel import local_warmstart_run as r
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256


@pytest.fixture(autouse=True)
def forbid_local_training():
    torch.set_num_threads(1)
    with patch.object(
        torch.optim.AdamW, "step", side_effect=AssertionError("No local neural training")
    ):
        yield


def pipeline(root, stack):
    parent, data, stats, local = small()
    source, training, out = (root / name for name in ("source", "training", "out"))
    for p in (source, training, out):
        p.mkdir()
    jobs = [{"name": f"control_s{s}", "seed": s} for s in (42, 43)]
    tm = {"experiments": jobs, "state_width": 8, "evaluation_batch": 4}
    meta = {
        "schema": r.SCHEMA,
        "code_sha256": r.code_identity(),
        "source": str(source),
        "identity": {"training_source": str(training), "training_manifest": tm},
        "experiments": jobs,
        "epochs": 6,
        "batch": 4,
        "extract_batch": 4,
        "evaluation_batch": 4,
        "width": 8,
        "hidden": 8,
        "lr": 3e-5,
        "train_prefixes": list(r.ba.TRAIN_PREFIXES),
        "val_prefixes": list(r.ba.VAL_PREFIXES),
        "held_prefixes": list(r.ba.HELD_PREFIXES),
    }
    atomic_json(meta, out / "manifest.json")
    atomic_json(
        {"status": "no_cross_period_upgrade", "original_utility_protocol": {}},
        source / "decision.json",
    )
    models = {}
    endpoints = {}
    validation = {}
    for job in jobs:
        m = copy.deepcopy(parent)
        with torch.no_grad():
            m.local_head[2].bias.add_((job["seed"] - 42) * 0.01)
        m.eval().requires_grad_(False)
        models[job["name"]] = m
        x = torch.tensor(data["x"])
        ps = torch.tensor([r.ba.VAL_PREFIXES] * len(x))
        with torch.no_grad():
            z = m.core.encoder(x)
            endpoints[job["name"] + "_best"] = z[:, -1].numpy()
            y, mask = r.ba.local_targets(
                torch.tensor(data["y"]), torch.tensor(data["mask"]), ps, stats, local
            )
            prediction = r.lr.predict(
                m.local_head, z[:, np.array(r.ba.VAL_PREFIXES) - 1], r.identity_scales(8)
            )
            validation[job["name"]] = {
                "local_primary": float(r.ba.local_rows(prediction, y, mask)["primary"].mean())
            }
    stack.enter_context(patch.object(r.xr, "parent_meta", return_value={}))
    stack.enter_context(patch.object(r.xr.cr, "alignment", return_value={}))
    stack.enter_context(patch.object(r, "statistics", return_value=(stats, local)))
    stack.enter_context(patch.object(r.ur.bb, "load_data", return_value=data))
    stack.enter_context(patch.object(r.xr.rr.old, "verify_cache"))
    stack.enter_context(patch.object(r.xr.rr.old, "arrays", return_value=endpoints))
    stack.enter_context(
        patch.object(r.xr, "validation", side_effect=lambda tm, j, m, d: validation[j["name"]])
    )
    stack.enter_context(
        patch.object(
            r,
            "original_model",
            side_effect=lambda meta, j, d: (
                copy.deepcopy(models[j["name"]]),
                validation[j["name"]],
            ),
        )
    )
    for split in ("train", "val", "test", "cross_research"):
        rows = [
            {"index": i, "symbol": "x", "period": 15, "month": "m1", "week": f"w{i // 2}"}
            for i in range(len(data["x"]))
        ]
        atomic_json(rows, training / f"{split}_inventory.json")
        atomic_json(rows, out / f"{split}_inventory.json")
        if split in ("test", "cross_research"):
            for job in jobs:
                score, *_ = r.ur.pf.score(models[job["name"]], data, stats, local, 4, "cpu")
                atomic_json(
                    {"reconstruction": {"scores": score}},
                    source / f"{split}_{job['name']}_best.json",
                )
    return meta, out, models, data


def test_cache_causality_original_weights_and_identity_normalization(tmp_path):
    with ExitStack() as stack:
        meta, out, models, data = pipeline(tmp_path, stack)
        with pytest.raises(FileNotFoundError):
            r.prepare(meta, out, "test", "cpu")
        for split in ("train", "val"):
            r.prepare(meta, out, split, "cpu")
        assert r.verify_cache(meta, out, "train")["positions"] == list(r.ba.TRAIN_PREFIXES)
        assert not set(r.ba.HELD_PREFIXES) & set(r.verify_cache(meta, out, "train")["positions"])
        assert r.verify_cache(meta, out, "val")["positions"] == list(r.ba.VAL_PREFIXES)
        for job in meta["experiments"]:
            head = r.original_head(meta, out, job, "cpu")
            original = models[job["name"]]
            assert r.ur.bb.state_signature(head) == r.ur.bb.state_signature(original.local_head)
            x = torch.tensor(np.asarray(r.arrays(out, "train", job["name"])["x"]))
            with torch.no_grad():
                torch.testing.assert_close(
                    r.lr.predict(head, x, r.identity_scales(8)),
                    original.local_head(x).reshape(12, 94, 16, 7),
                    rtol=0,
                    atol=0,
                )
                z = original.core.encoder(torch.tensor(data["x"][:, :37]))[:, -1]
            np.testing.assert_allclose(
                x[:, list(r.ba.TRAIN_PREFIXES).index(37)].numpy(), z.numpy(), atol=5e-5, rtol=2e-4
            )
        with patch.object(
            r, "original_model", side_effect=AssertionError("No repeated cache extraction")
        ):
            r.prepare(meta, out, "train", "cpu")
        audit = out / "control_s42_frozen_audit.json"
        old = r.read_json(audit)
        old["validation"]["local_primary"] += 1
        atomic_json(old, audit)
        with pytest.raises(ValueError):
            r.original_head(meta, out, meta["experiments"][0], "cpu")


def test_full_budget_resume_epoch0_and_six_head_evaluations(tmp_path):
    with ExitStack() as stack:
        meta, out, models, _ = pipeline(tmp_path, stack)
        for split in ("train", "val"):
            r.prepare(meta, out, split, "cpu")
        before = {n: r.ur.bb.state_signature(m) for n, m in models.items()}
        original_epoch = r.lr.epoch
        calls = []

        def interrupted(*a, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return original_epoch(*a, **kw)

        stack.enter_context(patch.object(torch.optim.AdamW, "step", return_value=None))
        stack.enter_context(
            patch.object(r.lr, "scales", side_effect=AssertionError("No state normalization fit"))
        )
        with (
            patch.object(r.lr, "epoch", side_effect=interrupted),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            r.worker(out, "control_s42", "cpu")
        state = torch.load(out / "control_s42/last.pt", weights_only=True)
        assert state["epoch"] == 1
        for ident, value in zip(
            state["optimizer"]["param_groups"][0]["params"], state["model"].values(), strict=True
        ):
            state["optimizer"]["state"][ident] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.full_like(value, 0.001),
                "exp_avg_sq": torch.full_like(value, 0.002),
            }
        r.publish(state, out / "control_s42")
        for job in meta["experiments"]:
            r.worker(out, job["name"], "cpu")
        restored = torch.load(out / "control_s42/last.pt", weights_only=True)
        for ident, value in state["optimizer"]["state"].items():
            for k, tensor in value.items():
                torch.testing.assert_close(restored["optimizer"]["state"][ident][k], tensor)
        r.lock_selection(meta, out)
        r.check_selection(meta, out)
        for job in meta["experiments"]:
            summary = r.read_json(out / job["name"] / "training_summary.json")
            assert summary["selected_epoch"] == 0 and summary["steps"] == 18
            assert summary["prefix_exposures"] == 288 and summary["encoder_updates"] == 0
            with patch.object(r.lr, "epoch", side_effect=AssertionError("No repeat training")):
                r.worker(out, job["name"], "cpu")
        for split in ("test", "cross_research"):
            r.prepare(meta, out, split, "cpu")
        r.evaluate(meta, out, "cpu")
        result = r.read_json(out / "local_warmstart_metrics.json")
        assert len(result["summary"]["test"]) == 6
        assert len(result["decision"]["checks"]) == 104
        assert (
            result["decision"]["status"] == "no_warm_local_readout_gain"
        )  # unchanged epoch0 is no improvement
        assert before == {n: r.ur.bb.state_signature(m) for n, m in models.items()}
        assert all(
            v["original_head_replayed"] for v in r.read_json(out / "frozen_replay.json").values()
        )
        (out / "control_s43/best.pt").write_bytes(b"bad")
        with pytest.raises(ValueError):
            r.check_selection(meta, out)


def decision_fixture():
    rows = {
        s: [{"week": f"w{i // 10}", "symbol": "x", "period": 15, "month": "m1"} for i in range(60)]
        for s in ("test", "cross_research")
    }
    records = {}
    for split in rows:
        records[split] = {}
        for seed in (42, 43):
            for kind in ("original", "best", "last"):
                records[split][f"control_s{seed}_{kind}"] = {
                    "errors": {
                        task: {
                            metric: [1.0 if kind == "original" else 0.9] * 60
                            for metric in ("primary", "path", "body", "activity", "change1")
                        }
                        for task in ["held", "trained"]
                        + [f"p{p}" for p in r.ba.VAL_PREFIXES + r.ba.HELD_PREFIXES]
                    }
                }
    return records, rows


@pytest.mark.parametrize(
    "failure", [None, "no_gain", "last", "position", "activity", "seed", "set", "support"]
)
def test_decision_requires_gain_without_sacrificing_positions(tmp_path, failure):
    records, rows = decision_fixture()
    item = records["test"]["control_s42_best"]["errors"]
    if failure == "no_gain":
        item["held"]["primary"] = [1.0] * 60
    if failure == "last":
        records["test"]["control_s42_last"]["errors"]["held"]["primary"] = [1.0] * 60
    if failure == "position":
        item["p48"]["primary"] = [1.2] * 60
    if failure == "activity":
        item["held"]["activity"] = [1.2] * 60
    if failure == "seed":
        records["test"]["control_s43_best"]["errors"]["held"]["primary"] = [1.0] * 60
    if failure == "set":
        records["cross_research"]["control_s42_best"]["errors"]["held"]["primary"] = [1.0] * 60
    if failure == "support":
        rows["test"] = [{"week": "one", "symbol": "x", "period": 15, "month": "m1"}] * 60
    result = r.decide(records, rows)
    assert result["status"] == (
        "no_warm_local_readout_gain" if failure else "warm_local_readout_gain"
    )
    assert not result["automatic_promotion"]


def test_controller_order_and_worker_failure_forwarding(tmp_path):
    with ExitStack() as stack:
        meta, out, _, _ = pipeline(tmp_path, stack)
        source = Path(meta["source"])
        events = []
        atomic_json(
            {"torch": str(torch.__version__), "numpy": np.__version__}, source / "runtime.json"
        )
        stack.enter_context(patch.object(r, "source_identity", return_value=meta["identity"]))
        stack.enter_context(patch.object(r, "check_output"))
        stack.enter_context(patch.object(r, "make_manifest", return_value=meta))
        stack.enter_context(patch.object(r.ur.bb, "state_signature", return_value="unchanged"))
        stack.enter_context(
            patch.object(r.gr.rr.up.probe, "inventory_audit", return_value={"passed": True})
        )
        stack.enter_context(
            patch.object(r, "prepare", side_effect=lambda meta, out, s, device: events.append(s))
        )
        stack.enter_context(
            patch.object(r, "run_jobs", side_effect=lambda *a: events.append("training"))
        )
        stack.enter_context(
            patch.object(r, "release_cache_cuda", side_effect=lambda *a: events.append("release"))
        )
        stack.enter_context(
            patch.object(r, "lock_selection", side_effect=lambda *a: events.append("lock"))
        )
        stack.enter_context(
            patch.object(r, "evaluate", side_effect=lambda *a: events.append("evaluate"))
        )
        stack.enter_context(
            patch.object(r, "check_selection", side_effect=lambda *a: events.append("verify_lock"))
        )
        r.run(source, out, 1, "cpu")
        assert events == [
            "train",
            "val",
            "release",
            "training",
            "lock",
            "test",
            "cross_research",
            "evaluate",
        ]
        r.run(source, out, 1, "cpu")
        assert events[-1] == "verify_lock" and events.count("training") == 1
    failed = MagicMock()
    failed.poll.return_value = 1
    failed.returncode = 1
    running = MagicMock()
    running.poll.return_value = None
    (out / "control_s42").mkdir()
    (out / "control_s42/run.log").write_text("CUDA out of memory\n")
    with (
        patch.object(r.subprocess, "Popen", side_effect=[failed, running]),
        pytest.raises(RuntimeError, match="CUDA out of memory"),
    ):
        r.run_jobs(out, 2)
    running.terminate.assert_called_once()
    running.wait.assert_called_once()


def test_source_identity_binds_recheck_and_locked_control600(tmp_path):
    source = tmp_path / "report"
    source.mkdir()
    tm = {
        "experiments": [{"name": f"control_s{s}", "weight": 0} for s in (42, 43)],
        "state_width": 768,
        "evaluation_batch": 64,
    }
    files = {"manifest.json": "original"}
    report = {
        "schema": "babel-cross-period-evaluation-recheck-v1",
        "evaluator_sha256": sha256(r.recheck.__file__),
        "source": str(tmp_path / "training"),
        "original_manifest": tm,
        "source_files": files,
    }
    atomic_json(report, source / "manifest.json")
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "files": {"manifest.json": sha256(source / "manifest.json")},
        },
        source / "completion.json",
    )
    lock = {"trials": {f"control_s{s}": {"selected_epoch": 100} for s in (42, 43)}}
    with (
        patch.object(r.recheck, "verify_inputs", return_value=(tm, files)),
        patch.object(r.xe, "check_models", return_value=lock),
    ):
        identity = r.source_identity(source)
        meta = r.make_manifest(source, identity)
        assert meta["lr"] == 3e-5 and meta["evaluation_batch"] == 64
        assert len(meta["experiments"]) == 2
        assert [v["seed"] for v in meta["experiments"]] == [42, 43]
        lock["trials"]["control_s43"]["selected_epoch"] = 99
        with pytest.raises(ValueError, match="control600"):
            r.source_identity(source)
        lock["trials"]["control_s43"]["selected_epoch"] = 100
        with (
            patch.object(
                r.recheck, "verify_inputs", return_value=(tm, {"manifest.json": "changed"})
            ),
            pytest.raises(ValueError, match="lineage"),
        ):
            r.source_identity(source)
    with pytest.raises(ValueError):
        r.check_output(source, source / "nested")


def test_changed_cache_or_initial_weights_cannot_resume(tmp_path):
    with ExitStack() as stack:
        meta, out, _, _ = pipeline(tmp_path, stack)
        for split in ("train", "val"):
            r.prepare(meta, out, split, "cpu")
        (out / "cache/control_s42_original_head.pt").write_bytes(b"bad")
        with pytest.raises(ValueError):
            r.worker(out, "control_s42", "cpu")


def test_failure_export_preserves_history_and_defaults_are_one_worker(tmp_path):
    run = tmp_path / "run"
    (run / "control_s42").mkdir(parents=True)
    (run / "control_s42/best.pt").write_bytes(b"omit")
    atomic_json([{"epoch": 100}], run / "control_s42/history.json")
    env = os.environ | {
        "BABEL_LOCAL_WARMSTART_RUN": str(run),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "BABEL_LOCAL_WARMSTART_LOG": str(tmp_path / "missing"),
        "PYTHON_BIN": "/usr/bin/false",
    }
    result = subprocess.run(
        ["bash", "scripts/babel_local_warmstart768_autodl.sh", "all"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as archive:
        assert "run/control_s42/history.json" in archive.getnames()
        assert "run/control_s42/best.pt" not in archive.getnames()
        assert "run_status=failed" in archive.extractfile("run/run_status.txt").read().decode()
    import inspect

    assert inspect.signature(r.run).parameters["jobs"].default == 1
