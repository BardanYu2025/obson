"""CPU-only regressions for the signed diagnostic and immutable export repair."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from obson.babel import observed_history_export as historical
from obson.babel import observed_history_export_v2 as export

OLD_HASH = "7729aa07c990d20ad0d3bbe97f2402e9ff6bed27cba149f8ebb8008e2a32d485"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def score_arrays(per_position=False, count=2):
    metrics = (
        export.POSITION_METRICS if per_position else export.REGULAR_METRICS | export.OVERLAP_METRICS
    )
    shape = (count, 128) if per_position else (count,)
    bank = {}
    for metric in metrics:
        if metric in export.OVERLAP_METRICS:
            _, shift, field = metric.split("/")
            shift = int(shift[5:])
            width = (
                shift
                if field.startswith(("entered", "leaving", "flat_entered", "flat_leaving"))
                else 127 - shift
            )
            contexts = 1
        else:
            width = 127 if metric.endswith("/composite") else export.BANDS[metric.rsplit("/", 1)[1]]
            contexts = (
                1
                if per_position
                else {"full": 128, "interior": 127, "endpoint": 1}[metric.split("/", 1)[0]]
            )
        bank["errors::" + metric] = np.full(shape, 1e-6, np.float64)
        bank["support::" + metric] = np.full(shape, width * contexts, np.int64)
        if not per_position and metric in export.REGULAR_METRICS:
            bank["contexts::" + metric] = np.full(shape, contexts, np.int64)
    if not per_position:
        # A valid cancellation with a deliberately negative rounding residual.
        # The older exporter rejects this diagnostic even though all MSEs >= 0.
        for shift in (1, 16, 64):
            prefix = f"errors::overlap/shift{shift}/"
            mean = bank[prefix + "mean_shape_mse"]
            disagreement = np.full(count, 4e-7)
            consensus = np.nextafter(mean - disagreement / 4, np.inf)
            bank[prefix + "disagreement_mse"] = disagreement
            bank[prefix + "consensus_shape_mse"] = consensus
            bank[prefix + "identity_residual"] = mean - consensus - disagreement / 4
            assert (bank[prefix + "identity_residual"] < 0).all()
    return bank


def save_scores(path, per_position=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **score_arrays(per_position))


def fixed_predictions(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=np.zeros((2, 8, 127, 7), np.float32),
        targets=np.zeros((2, 8, 127, 7), np.float64),
        masks=np.ones((2, 8, 127, 7), bool),
        raw_closes=np.full((2, 255), 100.0, np.float64),
        positions=np.array([1, 32, 64, 80, 96, 112, 127, 128]),
    )


def synthetic_complete(out):
    """Small internally bound complete payload, never a neural qualification."""
    for relative in export._required_payload():
        path = out / relative
        if path.name in ("parent_rows.npz", "position_rows.npz"):
            save_scores(path, path.name == "position_rows.npz")
        elif path.suffix == ".npz":
            fixed_predictions(path)
        elif path.name == "rows.json":
            json_file(path, [{"key": "synthetic/A"}, {"key": "synthetic/B"}])
        else:
            json_file(path, {})
    json_file(out / "status.json", {"status": "complete_requires_review"})
    for run in export.RUNS:
        files = {p.name: sha(p) for p in (out / run).iterdir() if p.name != "completion.json"}
        json_file(out / run / "completion.json", {"files": files, "epoch": 120})
        for cohort in export.COHORTS:
            prefix = f"evaluation/{run}/{cohort}"
            paths = [p for p in (out / prefix).iterdir() if p.name != "complete.json"]
            paths.append(out / f"results/{cohort}_{run}_predictions.npz")
            json_file(
                out / prefix / "complete.json",
                {
                    "files": {p.relative_to(out).as_posix(): sha(p) for p in paths},
                    "binding": {"parents": 2},
                    "status": "complete",
                },
            )
    bound = export._required_payload() - {"status.json", "budget.json", "completion.json"}
    json_file(out / "completion.json", {"files": {name: sha(out / name) for name in bound}})
    return out


def read_report(path):
    with tarfile.open(path) as tar:
        names = tar.getnames()
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
        manifest = next(n for n in names if n.endswith("/export_manifest.json"))
        return json.load(tar.extractfile(manifest)), names


def test_exact_schema_counts_and_old_bound_source_remains_unchanged():
    assert len(export.POSITION_METRICS) == 22
    assert len(export.REGULAR_METRICS) == 66
    assert len(export.OVERLAP_METRICS) == 45
    assert len(export.SIGNED_METRICS) == 3
    assert sha(Path(historical.__file__)) == OLD_HASH
    script = Path(__file__).resolve().parents[2] / "scripts/babel_observed_history768_autodl.sh"
    assert sha(script) == "9ad3a2db262a50359d268202c6a0d01826bc69b2978a4d48326537d429efa109"


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_negative_identity_roundoff_allowed_at_actual_error_scale(tmp_path, scale):
    bank = score_arrays()
    for key in bank:
        if key.startswith("errors::"):
            bank[key] *= scale
    path = tmp_path / "parent_rows.npz"
    np.savez(path, **bank)
    with pytest.raises(ValueError, match="nonnegative"):
        historical._score_array(path, False)
    assert export._score_array(path, False) == 2


@pytest.mark.parametrize("per_position", [False, True])
def test_negative_mse_still_rejected(tmp_path, per_position):
    bank = score_arrays(per_position)
    metric = "price/near" if per_position else "full/price/near"
    bank["errors::" + metric].flat[0] = -np.nextafter(0.0, 1.0)
    path = tmp_path / "score.npz"
    np.savez(path, **bank)
    with pytest.raises(ValueError, match="MSE.*nonnegative"):
        export._score_array(path, per_position)


@pytest.mark.parametrize(
    "field", ["identity_residual", "mean_shape_mse", "consensus_shape_mse", "disagreement_mse"]
)
def test_non_roundoff_or_inconsistent_identity_rejected(tmp_path, field):
    bank = score_arrays()
    # 1e-15 is tiny in absolute terms but much too large for 1e-6 error scale.
    bank[f"errors::overlap/shift16/{field}"][0] += 1e-15
    path = tmp_path / "parent_rows.npz"
    np.savez(path, **bank)
    with pytest.raises(ValueError, match="identity_residual"):
        export._score_array(path, False)


def test_signed_whitelist_cannot_be_extended_by_naming(tmp_path):
    bank = score_arrays()
    bank["errors::overlap/shift2/identity_residual"] = np.zeros(2)
    path = tmp_path / "parent_rows.npz"
    np.savez(path, **bank)
    with pytest.raises(ValueError, match="Exact locked"):
        export._score_array(path, False)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metric",
        "disguised_embedding",
        "positive_support_nan",
        "negative_support",
        "extra_context",
        "bad_overlap_width",
        "bad_context_count",
    ],
)
def test_schema_and_support_failures_not_hidden(tmp_path, mutation):
    bank = score_arrays()
    key = "errors::full/price/near"
    if mutation == "missing_metric":
        del bank[key]
    elif mutation == "disguised_embedding":
        bank[key] = np.ones((2, 768))
    elif mutation == "positive_support_nan":
        bank[key][0] = np.nan
    elif mutation == "negative_support":
        bank["support::full/price/near"][0] = -1
    elif mutation == "extra_context":
        bank["contexts::overlap/shift1/shape_a_mse"] = np.ones(2, np.int64)
    elif mutation == "bad_overlap_width":
        bank["support::overlap/shift1/shape_a_mse"][0] = 3
    else:
        bank["contexts::endpoint/price/near"][0] = 2
    path = tmp_path / "parent_rows.npz"
    np.savez(path, **bank)
    with pytest.raises(ValueError):
        export._score_array(path, False)


@pytest.mark.parametrize("per_position", [False, True])
def test_missing_nonprice_support_remains_explicit_nan(tmp_path, per_position):
    bank = score_arrays(per_position)
    metric = "nonprice/turnover_asinh/near"
    if not per_position:
        metric = "full/" + metric
        bank["contexts::" + metric][0] = 0
    bank["errors::" + metric][0] = np.nan
    bank["support::" + metric][0] = 0
    path = tmp_path / "scores.npz"
    np.savez(path, **bank)
    assert export._score_array(path, per_position) == 2


def test_complete_payload_includes_negative_residual_and_preserves_source_and_old_archive(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    (out / "session.lock").touch()
    (out / "worker.lock").touch()
    download = tmp_path / "download"
    download.mkdir()
    old_archive = download / "run_reports.tar.gz"
    old_archive.write_bytes(b"original immutable historical archive")
    before = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}
    result = export.export_reports(out, download)
    report, names = read_report(result["archive"])
    assert result["archive"].endswith("run_reports_v2.tar.gz")
    assert result["report_integrity"] == report["report_integrity"]["status"] == "complete"
    assert report["run_status"] == "complete_requires_review"
    assert len([n for n in names if n.endswith("parent_rows.npz")]) == 8
    assert report["source_files_created_or_rewritten"] == 0
    assert report["held_producer_locks"] == ["session.lock", "worker.lock"]
    assert before == {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}
    assert old_archive.read_bytes() == b"original immutable historical archive"
    assert report["historical_exporter_sha256"] == OLD_HASH
    with tarfile.open(result["archive"]) as tar:
        for item in report["included"]:
            assert (
                hashlib.sha256(tar.extractfile("run/" + item["path"]).read()).hexdigest()
                == item["sha256"]
            )
    first = Path(result["archive"]).read_bytes()
    second = export.export_reports(out, download)
    assert Path(second["previous_archive"]).read_bytes() == first


@pytest.mark.parametrize(
    "broken",
    [
        "missing_parent",
        "invalid_parent",
        "changed_byte_binding",
        "empty_completion_map",
        "different_parent_count",
    ],
)
def test_incomplete_payload_cannot_inherit_training_complete_status(tmp_path, broken):
    out = synthetic_complete(tmp_path / "run")
    parent = out / "evaluation/control_s42/test/parent_rows.npz"
    if broken == "missing_parent":
        parent.unlink()
    elif broken == "invalid_parent":
        bank = score_arrays()
        bank["errors::overlap/shift1/shape_a_mse"][0] = -1e-24
        np.savez(parent, **bank)
    elif broken == "changed_byte_binding":
        (out / "manifest.json").write_text('{"changed":true}')
    elif broken == "empty_completion_map":
        json_file(out / "completion.json", {"files": {}})
    else:
        path = out / "evaluation/control_s42/test/rows.json"
        json_file(path, [{}])
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_report(result["archive"])
    assert report["run_status"] == "complete_requires_review"
    assert result["report_integrity"] == report["report_integrity"]["status"] == "incomplete"
    assert report["report_integrity"]["binding_issues"]
    if broken in ("missing_parent", "invalid_parent"):
        assert (
            parent.relative_to(out).as_posix() in report["report_integrity"]["missing_or_rejected"]
        )
        assert "run/evaluation/control_s42/test/parent_rows.npz" not in names
    if broken == "different_parent_count":
        assert any(
            x["reason"] == "parent_count_mismatch"
            for x in report["report_integrity"]["binding_issues"]
        )


@pytest.mark.parametrize(
    "code,status", [(1, "failed"), (124, "timeout"), (130, "stopped"), (143, "stopped")]
)
def test_failed_and_stopped_reports_are_exported_without_source_mutation(tmp_path, code, status):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": "complete_requires_review"})
    log = tmp_path / "run.log"
    log.write_text("failed before models were available\n")
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    result = export.export_reports(
        out, tmp_path / "download", command_exit_code=code, mode="all", log=log
    )
    report, names = read_report(result["archive"])
    assert report["run_status"] == status
    assert report["report_integrity"]["status"] == "incomplete"
    assert report["command_exit_code"] == code
    assert (
        report["export_training_updates"]
        == report["export_readout_fits"]
        == report["export_neural_forwards"]
        == 0
    )
    assert "run/wrapper.log" in names
    assert before == sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))


@pytest.mark.parametrize(
    "lock", ["session.lock", "worker.lock", "nested/worker.lock", "evaluation/evaluation.lock"]
)
def test_active_producer_still_prevents_export(tmp_path, lock):
    out = tmp_path / "run"
    path = out / lock
    path.parent.mkdir(parents=True)
    with path.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active"):
            export.export_reports(out, tmp_path / "download")
    assert not list((tmp_path / "download").glob("*.tar.gz"))


def test_source_snapshot_change_rejected_without_replacing_prior_archive(tmp_path, monkeypatch):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": "failed"})
    download = tmp_path / "download"
    download.mkdir()
    existing = download / "run_reports_v2.tar.gz"
    existing.write_bytes(b"previous export")
    copyfile = export.shutil.copyfile

    def mutate(source, target):
        result = copyfile(source, target)
        if Path(source) == out / "status.json":
            Path(source).write_text('{"status":"changed during copy"}')
        return result

    monkeypatch.setattr(export.shutil, "copyfile", mutate)
    with pytest.raises(ValueError, match="changed during export"):
        export.export_reports(out, download)
    assert existing.read_bytes() == b"previous export"


def test_path_symlink_and_special_file_rejections(tmp_path):
    out, download = tmp_path / "run", tmp_path / "download"
    out.mkdir()
    with pytest.raises(ValueError, match="traversal"):
        export.export_reports(out / ".." / "run", download)
    with pytest.raises(ValueError, match="Separate"):
        export.export_reports(out, out / "download")
    outside = tmp_path / "secret.json"
    outside.write_text("outside")
    (out / "symlink.json").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        export.export_reports(out, download)
    (out / "symlink.json").unlink()
    os.mkfifo(out / "pipe.txt")
    with pytest.raises(ValueError, match="regular"):
        export.export_reports(out, download)


def test_weights_full_predictions_and_undeclared_arrays_remain_omitted(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "last.pt").write_bytes(b"synthetic weights")
    (out / "predictions").mkdir()
    np.save(out / "predictions/all.npy", np.ones((2, 128)))
    np.savez(out / "latent.npz", z=np.ones((2, 768)))
    result = export.export_reports(out, tmp_path / "download", command_exit_code=1)
    report, names = read_report(result["archive"])
    assert not any(n.endswith((".pt", ".npy", ".npz")) for n in names)
    reasons = {item["path"]: item["reason"] for item in report["omitted"]}
    assert reasons["last.pt"] == "weights_or_optimizer_state"
    assert reasons["predictions/all.npy"] == "full128_prediction_shard"
    assert reasons["latent.npz"] == "undeclared_binary_report"


def test_cli_failed_export_without_output_does_not_invent_training_or_directory(tmp_path):
    out, download = tmp_path / "absent", tmp_path / "download"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "obson.babel.observed_history_export_v2",
            "--out",
            str(out),
            "--download",
            str(download),
            "--command-exit-code",
            "1",
            "--mode",
            "all",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "run_status=failed, report_integrity=incomplete" in result.stdout
    assert not out.exists()
    report, _ = read_report(download / "absent_reports_v2.tar.gz")
    assert report["snapshot_consistency"] == "no_output_directory"
    assert report["held_producer_locks"] == []
