"""Synthetic evaluation-only recovery and strict numerical failure handling."""

import copy
import os
import subprocess
import tarfile
from unittest.mock import patch

import numpy as np
import pytest

from obson.babel import cross_period_recheck as r
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256


def overlap(offset=0.0):
    return {
        str(d): {
            "summary": {
                "all": {
                    "path_signed_offset_a_bps": {
                        "p50": offset,
                        "count": 20,
                    }
                }
            }
        }
        for d in (1, 16, 64)
    }


def test_near_zero_summary_failures_preserve_values_and_counts():
    differences = r.mismatches(overlap(2e-6), overlap())
    assert len(differences) == 3
    assert differences[0]["actual"] == 2e-6
    assert differences[0]["expected"] == 0.0
    assert differences[0]["allowed"] == 1e-6
    changed = overlap()
    changed["1"]["summary"]["all"]["path_signed_offset_a_bps"]["count"] = 21
    assert r.mismatches(changed, overlap())[0]["reason"] == "value differs"
    assert r.array_replay(np.array([np.nan]), np.array([0.0]))["passed"] is False


@pytest.mark.parametrize("failure", [None, "reference", "state", "prediction", "support"])
def test_reference_batch_replay_requires_both_checks(tmp_path, failure):
    target = overlap()
    current = overlap(2e-6)
    reference = copy.deepcopy(target)
    if failure == "reference":
        reference = overlap(3e-6)
    if failure == "support":
        reference["16"]["summary"]["all"]["path_signed_offset_a_bps"]["count"] = 19
    paired = {"views": {d: {"x": np.zeros((20, 2))} for d in (0, 1, 16, 64)}}
    stats = {"y_mean": np.zeros(7), "y_scale": np.ones(7)}
    calls = []

    def predict(model, x, scale, batch, device):
        calls.append(batch)
        z = np.zeros((len(x), 8))
        y = np.zeros((len(x), 128, 7))
        if batch == 64:
            z += 0.01 if failure == "state" else 1e-6
            y += 0.01 if failure == "prediction" else 1e-6
        return z, y

    path = tmp_path / "replay.json"
    before = copy.deepcopy(current)
    with (
        patch.object(r.ce, "overlap", return_value=reference) as recompute,
        patch.object(r.cr.odr, "predict", side_effect=predict),
    ):
        if failure:
            with pytest.raises(ValueError, match="remains unverified"):
                r.parent_overlap_replay(None, paired, stats, current, target, 64, 128, "cpu", path)
        else:
            r.parent_overlap_replay(None, paired, stats, current, target, 64, 128, "cpu", path)
        assert recompute.call_args.args[3] == 128
    assert calls == [64, 128] * 4
    assert current == before  # Do not replace scoring results with reference numbers.
    report = r.read_json(path)
    assert report["status"] == (
        "failed" if failure else "source_batch_replay_and_batch_controls_passed"
    )
    assert report["summary_atol"] == 1e-6
    assert len(report["current_summary_differences"]) == 3
    assert set(report["batch_controls"]) == {"0", "1", "16", "64"}


def test_same_batch_failure_is_not_bypassed(tmp_path):
    with (
        patch.object(r.ce, "overlap", side_effect=AssertionError("No alternate batch")),
        pytest.raises(ValueError, match="source batch"),
    ):
        r.parent_overlap_replay(
            None, {}, {}, overlap(2e-6), overlap(), 128, 128, "cpu", tmp_path / "replay.json"
        )
    assert r.read_json(tmp_path / "replay.json")["status"] == "failed_same_batch_reference"


def test_original_strict_pass_needs_no_extra_inference(tmp_path):
    with patch.object(r.ce, "overlap", side_effect=AssertionError("No extra inference")):
        report = r.parent_overlap_replay(
            None, {}, {}, overlap(), overlap(), 64, 128, "cpu", tmp_path / "replay.json"
        )
    assert report["status"] == "original_replay_passed"


@pytest.mark.parametrize("failure", [None, "replay", "source_changed"])
def test_recheck_never_trains_fits_or_changes_original_manifest(tmp_path, failure):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    meta = {
        "source": str(tmp_path / "parent"),
        "identity": {"manifest": {"evaluation_batch": 128}},
        "evaluation_batch": 64,
        "raw_root": str(tmp_path / "raw"),
        "raw_audit": str(tmp_path / "audit"),
    }
    atomic_json(meta, source / "manifest.json")
    digest = sha256(source / "manifest.json")
    files = {"manifest.json": digest}
    verified = [
        (meta, files),
        (meta, files if failure != "source_changed" else {"manifest.json": "changed"}),
    ]

    def evaluate(meta, inputs, reports, device):
        assert inputs == source and reports == out
        if failure == "replay":
            atomic_json({"actual": 2e-6, "expected": 0.0}, reports / "numeric_replay.json")
            raise ValueError("numeric failure")
        atomic_json({"status": "no_cross_period_upgrade"}, reports / "decision.json")

    with (
        patch.object(r, "verify_inputs", side_effect=verified),
        patch.object(r.run, "check_output"),
        patch.object(r, "evaluate", side_effect=evaluate),
        patch.object(r.run, "worker", side_effect=AssertionError("No training")),
        patch.object(r.ev, "fit", side_effect=AssertionError("No fitting")),
    ):
        if failure:
            with pytest.raises(ValueError):
                r.recheck(source, out, "cpu")
            assert not (out / "completion.json").exists()
            assert r.read_json(out / "failure.json")["training_updates"] == 0
        else:
            r.recheck(source, out, "cpu")
            assert r.read_json(out / "completion.json")["source_unchanged"]
    assert sha256(source / "manifest.json") == digest
    if failure == "replay":
        assert (out / "numeric_replay.json").exists()


def test_failure_archive_includes_training_and_diagnostics_without_weights(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    atomic_json({"epochs": 100}, source / "training_summary.json")
    (source / "best.pt").write_bytes(b"excluded")
    (source / "cache.npz").write_bytes(b"excluded")
    atomic_json({"actual": 2e-6}, out / "numeric_replay.json")
    env = os.environ | {
        "BABEL_CROSS_PERIOD_RECHECK_SOURCE": str(source),
        "BABEL_CROSS_PERIOD_RECHECK_RUN": str(out),
        "BABEL_CROSS_PERIOD_RECHECK_LOG": str(tmp_path / "absent"),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "PYTHON_BIN": "/usr/bin/false",
    }
    result = subprocess.run(
        ["bash", "scripts/babel_cross_period_recheck_autodl.sh", "all"],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 1
    with tarfile.open(tmp_path / "download/out_reports.tar.gz") as archive:
        names = archive.getnames()
        assert "out/numeric_replay.json" in names
        assert "out/training_source/training_summary.json" in names
        assert not any(n.endswith((".pt", ".npz")) for n in names)
        assert b"run_status=failed" in archive.extractfile("out/run_status.txt").read()
