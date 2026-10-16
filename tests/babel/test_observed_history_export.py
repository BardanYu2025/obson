"""Failure-safe report packaging and shell wrappers on synthetic artifacts only."""

import fcntl
import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

from obson.babel import observed_history_export as export


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixed_arrays(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=np.zeros((2, 8, 127, 7), np.float32),
        targets=np.zeros((2, 8, 127, 7), np.float64),
        masks=np.ones((2, 8, 127, 7), bool),
        raw_closes=np.full((2, 255), 100.0, np.float64),
        positions=np.array(export.POSITIONS),
    )


def read_archive(path, name):
    with tarfile.open(path) as tar:
        assert all(not m.name.startswith("/") and ".." not in Path(m.name).parts for m in tar)
        return json.load(tar.extractfile(f"{name}/export_manifest.json")), tar.getnames()


def test_include_fixed_evidence_and_list_omitted_weights_raw_full128(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    put_json(out / "status.json", {"status": "complete"})
    (out / "last.pt").write_bytes(b"synthetic weights")
    (out / "raw").mkdir()
    np.savez(out / "raw/train.npz", close=np.ones((2, 255)))
    (out / "predictions/control_s42/test").mkdir(parents=True)
    np.save(
        out / "predictions/control_s42/test/parents_00000_00002.npy",
        np.zeros((2, 128, 127, 7), np.float32),
    )
    fixed_arrays(out / "results/test_control_s42_predictions.npz")
    evidence = out / "evaluation/control_s42/test"
    evidence.mkdir(parents=True)
    np.savez(evidence / "parent_rows.npz", **{"errors::full/price/near": np.ones(2)})
    np.savez(evidence / "position_rows.npz", **{"errors::price/near": np.ones((2, 128))})
    (out / "epoch.jsonl").write_text('{"epoch":120}\n')
    result = export.export_reports(out, download)
    manifest, names = read_archive(result["archive"], "run")
    assert "run/results/test_control_s42_predictions.npz" in names
    assert "run/evaluation/control_s42/test/position_rows.npz" in names
    assert not any(
        name.endswith((".pt", ".npy")) or name.endswith("raw/train.npz") for name in names
    )
    omitted = {row["path"]: row for row in manifest["omitted"]}
    assert omitted["last.pt"]["reason"] == "weights_or_optimizer_state"
    shard = "predictions/control_s42/test/parents_00000_00002.npy"
    assert omitted[shard]["reason"] == "full128_prediction_shard"
    assert omitted[shard]["bytes"] == (out / shard).stat().st_size
    assert omitted[shard]["sha256"] == hashlib.sha256((out / shard).read_bytes()).hexdigest()
    with tarfile.open(result["archive"]) as tar:
        for item in manifest["included"]:
            assert (
                hashlib.sha256(tar.extractfile("run/" + item["path"]).read()).hexdigest()
                == item["sha256"]
            )
    assert manifest["snapshot_consistency"] == "producer_locks_held"
    assert not list(download.glob(".observed_history_export_*"))


def test_new_archive_preserves_previous_failure_or_success_bytes(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    put_json(out / "status.json", {"status": "failed"})
    first = export.export_reports(out, download, command_exit_code=1, mode="all")
    old = Path(first["archive"]).read_bytes()
    put_json(out / "status.json", {"status": "complete"})
    second = export.export_reports(out, download, mode="evaluate")
    assert Path(second["previous_archive"]).read_bytes() == old
    manifest, _ = read_archive(second["archive"], "run")
    assert manifest["run_status"] == "complete"


@pytest.mark.parametrize(
    "lockname",
    ("session.lock", "worker.lock", "jobs/control_s42/worker.lock", "evaluation/evaluation.lock"),
)
def test_active_producer_refuses_inconsistent_snapshot(tmp_path, lockname):
    out = tmp_path / "run"
    path = out / lockname
    path.parent.mkdir(parents=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active"):
            export.export_reports(out, tmp_path / "download")
    assert not list((tmp_path / "download").glob("*.tar.gz"))


def test_traversal_symlink_special_files_and_nested_destinations_rejected(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    out.mkdir()
    with pytest.raises(ValueError, match="traversal"):
        export.export_reports(out / ".." / "run", download)
    with pytest.raises(ValueError, match="Separate"):
        export.export_reports(out, out / "download")
    secret = tmp_path / "external.json"
    secret.write_text("sensitive outside artifact")
    (out / "leak.json").symlink_to(secret)
    with pytest.raises(ValueError, match="Symlink"):
        export.export_reports(out, download)
    (out / "leak.json").unlink()
    os.mkfifo(out / "pipe.txt")
    with pytest.raises(ValueError, match="regular"):
        export.export_reports(out, download)
    assert secret.read_text() == "sensitive outside artifact"


def test_invalid_fixed_prediction_is_recorded_not_silently_exported(tmp_path):
    out = tmp_path / "run"
    put_json(out / "status.json", {"status": "failed"})
    path = out / "results/test_control_s42_predictions.npz"
    path.parent.mkdir()
    np.savez(path, positions=np.arange(1, 129), predictions=np.zeros(2))
    result = export.export_reports(out, tmp_path / "download", command_exit_code=1)
    report, names = read_archive(result["archive"], "run")
    assert "run/results/test_control_s42_predictions.npz" not in names
    entry = next(row for row in report["omitted"] if row["path"].startswith("results/"))
    assert entry["reason"].startswith("invalid_fixed_prediction_evidence")


@pytest.mark.parametrize(
    "code,status",
    [(0, "complete"), (1, "failed"), (124, "timeout"), (130, "stopped"), (143, "stopped")],
)
def test_exit_status_metrics_and_export_zero_work_are_preserved(tmp_path, code, status):
    out = tmp_path / "run"
    put_json(out / "status.json", {"status": "complete"})
    put_json(out / "observed_history_metrics.json", {"promoted": False})
    log = tmp_path / "run.log"
    log.write_text("synthetic stopped or completed worker\n")
    result = export.export_reports(
        out, tmp_path / "download", command_exit_code=code, mode="all", log=log
    )
    report, names = read_archive(result["archive"], "run")
    assert report["command_exit_code"] == code and report["run_status"] == status
    assert report["schema"] == "babel-observed-history-export-v1"
    assert report["task_ids"] == ["F01", "F05"]
    assert (
        report["export_training_updates"]
        == report["export_readout_fits"]
        == report["export_neural_forwards"]
        == 0
    )
    assert "run/observed_history_metrics.json" in names and "run/wrapper.log" in names


def test_no_output_export_remains_available_and_does_not_invent_execution(tmp_path):
    result = export.export_reports(tmp_path / "missing", tmp_path / "download", command_exit_code=1)
    report, _ = read_archive(result["archive"], "missing")
    assert report["run_status"] == "failed"
    assert report["snapshot_consistency"] == "no_output_directory"
    assert not (tmp_path / "missing").exists()


def test_no_unlisted_npz_or_disguised_embedding_arrays_exported(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    np.savez(out / "embeddings.npz", z=np.ones((2, 768)))
    directory = out / "evaluation/full_history_s42/test"
    directory.mkdir(parents=True)
    np.savez(directory / "parent_rows.npz", embeddings=np.ones((2, 768)))
    fixed_arrays(out / "results/test_full_history_s42_predictions.npz")
    fixed_arrays(out / "results/test_multiscale_s42_predictions.npz")
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_archive(result["archive"], "run")
    assert "run/results/test_full_history_s42_predictions.npz" in names
    assert not any(
        "embeddings" in name or "multiscale" in name or "parent_rows.npz" in name for name in names
    )
    reasons = {item["path"]: item["reason"] for item in report["omitted"]}
    assert reasons["embeddings.npz"] == "undeclared_binary_report"
    assert reasons["evaluation/full_history_s42/test/parent_rows.npz"].startswith(
        "invalid_score_evidence"
    )
