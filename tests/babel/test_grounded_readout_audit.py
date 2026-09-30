"""Synthetic-only, zero-optimizer audit of source identities and matched readouts."""

import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_history_query import fixture
from test_shared_history import rows

from obson.babel import grounded_readout_audit as audit
from obson.babel.ae_extend import atomic_json, atomic_save


@pytest.fixture(autouse=True)
def prohibit_updates():
    torch.set_num_threads(1)
    with patch.object(
        torch.optim.AdamW, "step", side_effect=AssertionError("Zero updates required")
    ):
        yield


def test_standardized_forward_exact_original_function_and_batch_independence():
    _, q, _, _, _ = fixture()
    with torch.no_grad():
        q.state_mean.copy_(torch.linspace(-2, 2, 8))
        q.state_scale.copy_(torch.linspace(0.1, 3, 8))
        z = torch.randn(6, 8)
        normalized = (z - q.state_mean) / q.state_scale
        torch.testing.assert_close(audit.from_standardized(q, normalized), q(z), atol=0, rtol=0)
        torch.testing.assert_close(
            audit.from_standardized(q, normalized[:2]), q(z)[:2], atol=1e-6, rtol=2e-5
        )


def test_physical_interval_conversion_has_correct_sign_age_scale_and_no_anchor():
    stats = {"delta_scale": [0.2], "y_mean": [0] * 7, "y_scale": [1] * 7}
    ages = torch.arange(1, 128, dtype=torch.float64)
    prices = -ages * 0.3 + torch.sin(ages / 7)
    pred = torch.zeros(2, 127, 7, dtype=torch.float64)
    pred[:, :, 0] = prices / (ages.sqrt() * 0.2)
    coords = torch.tensor([[[[50, 48], [48, 46]], [[30, 28], [28, 26]]]] * 2)
    value = audit.interval_changes(pred, coords, stats)
    expected = prices[coords[..., 1] - 1] - prices[coords[..., 0] - 1]
    torch.testing.assert_close(value, expected, atol=1e-14, rtol=1e-14)
    pred[:, :, 0] += 10 / (ages.sqrt() * 0.2)
    torch.testing.assert_close(
        audit.interval_changes(pred, coords, stats), value, atol=1e-14, rtol=1e-14
    )
    coords[0, 0, 0, 1] = 0
    with pytest.raises(ValueError, match="Current/future"):
        audit.interval_changes(pred, coords, stats)


def synthetic_source(tmp_path, stack):
    _, query, _, stats, _ = fixture()
    source = tmp_path / "source"
    source.mkdir()
    origin = {"source": str(tmp_path / "origin"), "chosen": {}}
    meta = {
        "schema": audit.gr.SCHEMA,
        "code_sha256": audit.gr.code_identity(),
        "task_source": origin,
        "budget": 60,
        "batch": 128,
        "qualification_epochs": 1,
        "shared": {"width": 4},
    }
    atomic_json(meta, source / "manifest.json")
    atomic_json({"torch": str(torch.__version__), "numpy": np.__version__}, source / "runtime.json")
    atomic_json(
        {"status": "blocked", "qualified": False, "encoder_training_started": False},
        source / "audit_status.json",
    )
    cohort = rows(60)
    for i, r in enumerate(cohort):
        r["key"] = f"X/15/C{i // 12}"
    atomic_json({"rows": cohort}, source / "val_grounded_plan.json")
    atomic_json(
        {
            "manifest_sha256": audit.sha256(source / "manifest.json"),
            "files": {"val_grounded_plan.json": audit.sha256(source / "val_grounded_plan.json")},
        },
        source / "grounded_data_lock.json",
    )
    ages, mask, _ = audit.sd.tensors(cohort, np.full(60, 16), "cpu")
    coords = audit.core.queries(ages, mask)
    for seed in (42, 43):
        torch.manual_seed(seed)
        p = source / f"qualification_s{seed}"
        p.mkdir()
        raw = torch.randn(1, 60, 2, 2).expand(3, -1, -1, -1).clone()
        mean, scale = float(raw.mean()), float(raw.std(unbiased=False))
        stats_target = {"mean": mean, "scale": scale}
        d = {"rows": cohort, "z": torch.randn(3, 60, 8), "q": coords, "y": (raw - mean) / scale}
        cache = {"train": copy.deepcopy(d), "val": d, "target_stats": stats_target}
        atomic_save(cache, p / "cache.pt")
        atomic_json(stats_target, p / "target_stats.json")
        atomic_json(
            {
                "identity": {
                    "manifest": audit.sha256(source / "manifest.json"),
                    "seed": seed,
                    "data_lock": audit.sha256(source / "grounded_data_lock.json"),
                },
                "files": {n: audit.sha256(p / n) for n in ("cache.pt", "target_stats.json")},
            },
            p / "cache_lock.json",
        )
        identity = {
            "manifest": audit.sha256(source / "manifest.json"),
            "cache": audit.sha256(p / "cache_lock.json"),
        }
        predictions = {}
        for mode in audit.qw.ABLATIONS:
            head = audit.core.GroundedReader(8, seed, 4)
            with torch.no_grad():
                pred = audit.qw.predict(head, d, "cpu", mode)
            mse = float((pred - d["y"]).square().mean())
            h = {
                "epoch": 1,
                "history": [{"epoch": 1, "val_mse": mse, "steps": 1}],
                "best_loss": mse,
                "best_epoch": 1,
            }
            atomic_json(h, p / f"{mode}_history.json")
            predictions[mode] = pred.numpy()
            if mode == "none":
                atomic_save(
                    {
                        "identity": identity,
                        "head": head.state_dict(),
                        "target_stats": stats_target,
                        "passed": False,
                    },
                    p / "selected.pt",
                )
            else:
                atomic_save(
                    dict(h, identity=identity, best_head=head.state_dict()), p / f"{mode}_resume.pt"
                )
        atomic_json(audit.qw.qualify(predictions, d["y"].numpy(), cohort), p / "qualification.json")
        names = ["selected.pt", "qualification.json", "target_stats.json", "cache_lock.json"] + [
            f"{a}_history.json" for a in audit.qw.ABLATIONS
        ]
        atomic_json(
            {"identity": identity, "files": {n: audit.sha256(p / n) for n in names}},
            p / "completion.json",
        )
    stack.enter_context(patch.object(audit.gr, "source_identity", return_value=origin))

    # Constructing returns an unusable encoder: audit must never execute it.
    class NoEncoder:
        def __call__(self, *args):
            raise AssertionError("No encoder forward allowed")

    stack.enter_context(
        patch.object(
            audit.gr, "construct", side_effect=lambda *args: (NoEncoder(), copy.deepcopy(query))
        )
    )
    stack.enter_context(patch.object(audit.gr, "stats", return_value=(stats, {})))
    return source, meta


def test_complete_pipeline_replays_all_gates_and_exports_reproducible_arrays(tmp_path):
    with ExitStack() as stack:
        source, meta = synthetic_source(tmp_path, stack)
        out = tmp_path / "audit"
        before = {p: audit.sha256(p) for p in source.rglob("*") if p.is_file()}
        audit.run(source, out, "cpu")
        assert all(audit.sha256(p) == h for p, h in before.items())
        summaries = audit.read_json(out / "metrics.json")
        for seed in (42, 43):
            payload = audit.read_json(out / f"s{seed}_predictions.json")
            # Float32 is the stored source cache type; restores the exact filter arithmetic.
            y = np.asarray(payload["targets"], dtype=np.float32)
            ids = np.flatnonzero(np.square(y[1, :, 0] - y[1, :, 1]).mean(-1) >= 0.05).tolist()
            assert ids == payload["informative_ids"]
            original = {
                k: np.asarray(payload["predictions"][k], dtype=np.float32)
                for k in audit.qw.ABLATIONS
            }
            replay = audit.qw.qualify(original, y, payload["rows"])
            assert replay == audit.read_json(source / f"qualification_s{seed}/qualification.json")
            full = {k: np.asarray(v) for k, v in payload["predictions"].items()}
            assert (
                audit.compare(full, y, payload["rows"], payload["target_stats"]["scale"])
                == summaries[str(seed)]
            )
        with patch.object(audit.gr, "construct", side_effect=AssertionError("No rerun")):
            audit.run(source, out, "cpu")
        (out / "metrics.json").write_text("{}")
        with pytest.raises(ValueError, match="SHA256"):
            audit.run(source, out, "cpu")


def test_cache_coordinates_and_source_mutation_are_rejected(tmp_path):
    with ExitStack() as stack:
        source, meta = synthetic_source(tmp_path, stack)
        cache = torch.load(source / "qualification_s42/cache.pt", weights_only=True)
        bad = copy.deepcopy(cache)
        bad["val"]["q"][0, 0, 0, 0, 0] -= 1
        with pytest.raises(ValueError, match="coordinates"):
            audit.validate_cache(meta, source, 42, bad)
        bad = copy.deepcopy(cache)
        bad["train"]["y"] += 0.1
        with pytest.raises(ValueError, match="standardized"):
            audit.validate_cache(meta, source, 42, bad)
        (source / "qualification_s42/selected.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match="fingerprint"):
            audit.source_identity(source)


def test_resume_weights_must_replay_reported_metrics(tmp_path):
    with ExitStack() as stack:
        source, meta = synthetic_source(tmp_path, stack)
        p = source / "qualification_s42/state_only_resume.pt"
        ck = torch.load(p, weights_only=True)
        ck["best_head"]["decoder.2.bias"] += 10
        atomic_save(ck, p)
        # Unbound legacy resume weights are pinned now, and must numerically replay old results.
        cache = torch.load(source / "qualification_s42/cache.pt", weights_only=True)
        with pytest.raises(ValueError, match="MSE replay"):
            audit.replay_auxiliary(meta, source, 42, cache, "cpu")


def test_paired_comparison_reports_perfect_original_without_promoting_model():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(3, 60, 2, 2)).astype(np.float32)
    rows = [{"key": f"X/15/C{i // 12}"} for i in range(60)]
    p = dict.fromkeys(("none", "state_only", "query_only", "train_mean"), np.zeros_like(y)) | {
        "original": y
    }
    result = audit.compare(p, y, rows, 1.0)
    for view in audit.VIEWS:
        v = result["all"][view]
        assert v["scores"]["original"]["mse"] == 0
        assert v["paired_mse_intervals"]["original/none"]["high"] < 0


@pytest.mark.parametrize("exit_code", [7, 3])
def test_failure_export_and_manual_export_preserve_status(tmp_path, exit_code):
    import os
    import subprocess
    import tarfile
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/babel_grounded_readout768_autodl.sh"
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "scripts" / script.name).write_text(script.read_text())
    for name in ("BABEL_GROUNDED_READOUT768.md", "BABEL_RESEARCH_GOAL.md"):
        (root / "docs" / name).write_text("synthetic audit protocol")
    out = root / "checkpoints/run"
    out.mkdir(parents=True)
    (out / "diagnostic.json").write_text('{"synthetic":true}')
    fake = root / "fake_python"
    fake.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake),
        BABEL_GROUNDED_READOUT_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(root / "download"),
    )
    for mode, code in (("all", exit_code), ("export", 0)):
        result = subprocess.run(
            ["bash", str(root / "scripts" / script.name), mode],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == code, result.stderr
        with tarfile.open(root / "download/run_reports.tar.gz") as archive:
            status = "blocked" if exit_code == 3 else "failed"
            assert (
                f"run_status={status}" in archive.extractfile("run/run_status.txt").read().decode()
            )
            assert archive.extractfile("run/diagnostic.json") is not None


def test_best_selection_tampering_is_rejected(tmp_path):
    with ExitStack() as stack:
        source, meta = synthetic_source(tmp_path, stack)
        path = source / "qualification_s42"
        h = audit.read_json(path / "none_history.json")
        h["best_epoch"] = 0
        atomic_json(h, path / "none_history.json")
        with pytest.raises(ValueError, match="selection"):
            audit.history_check(meta, path, "none")
