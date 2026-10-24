"""CPU-only report snapshot and wrapper regressions; no model execution."""

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

from obson.babel import content_guard_export as export
from obson.babel import content_guard_training as training


def identities(n):
    return [
        {"key": f"TEST/15/c{i % 3}", "month": f"2025-{i % 6 + 1:02d}", "parent": i}
        for i in range(n)
    ]


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


def synthetic_complete(out):
    for relative in export._required_payload():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name in ("parent_rows.npz", "position_rows.npz"):
            np.savez_compressed(path, **score_arrays(path.name == "position_rows.npz"))
        elif path.suffix == ".npz":
            np.savez_compressed(
                path,
                predictions=np.zeros((2, 8, 127, 7), np.float32),
                targets=np.zeros((2, 8, 127, 7), np.float64),
                masks=np.ones((2, 8, 127, 7), bool),
                raw_closes=np.full((2, 255), 100.0, np.float64),
                positions=np.array(export.POSITIONS),
            )
        elif path.name == "rows.json":
            json_file(path, identities(2))
        else:
            json_file(path, {})
    json_file(out / "status.json", {"status": "complete_requires_review"})
    references = {}
    models = {run: {"signature": hashlib.sha256(run.encode()).hexdigest()} for run in export.RUNS}
    for seed in (42, 43):
        order, ps = training.calibration_schedule(2, seed)
        reference = {
            "schema": training.REFERENCE_SCHEMA,
            "seed": seed,
            "parents": 2,
            "contexts": 10,
            "calibration_epoch": 180,
            "group_names": list(training.GROUP_NAMES),
            "baseline": [1.0] * 21,
            "q0": 1.0,
            "support": [10] * 21,
            "order_hash": training.array_hash(order),
            "prefix_hash": training.array_hash(ps),
            "source_signature": models[f"initial_s{seed}"]["signature"],
        }
        json_file(out / f"training_reference_s{seed}.json", reference)
        references[str(seed)] = reference
    for run in export.TRAINING_RUNS:
        checkpoint = out / run / "last.pt"
        checkpoint.write_bytes(b"synthetic omitted weight identity")
        ref = references[run[-2:]]
        binding = {"name": run}
        json_file(
            out / run / "final_train_risk.json",
            {
                **{
                    k: ref[k]
                    for k in ("parents", "contexts", "order_hash", "prefix_hash", "support")
                },
                "risks": [0.8] * 21,
                "group_ratios": [0.8] * 21,
                "query": 0.8,
                "within_source_mean": [True] * 21,
                "all_within_source_mean": True,
                "model_signature": models[run]["signature"],
                "checkpoint_sha256": sha(checkpoint),
                "reference_sha256": sha(out / f"training_reference_s{run[-2:]}.json"),
                "binding": binding,
            },
        )
        files = {p.name: sha(p) for p in (out / run).iterdir() if p.name != "completion.json"}
        models[run].update(files=files, epoch=60, binding=binding)
        json_file(out / run / "completion.json", models[run])
    json_file(
        out / "training_reference_lock.json",
        {
            "source_identity": "synthetic",
            "files": {
                f"training_reference_s{seed}.json": sha(out / f"training_reference_s{seed}.json")
                for seed in (42, 43)
            },
        },
    )
    json_file(
        out / "model_selection_lock.json",
        {
            "schema": "babel-content-guard768-v1",
            "selection": "fixed_last60",
            "source": "synthetic",
            "models": models,
            "reference_lock": sha(out / "training_reference_lock.json"),
            "references": {
                str(seed): sha(out / f"training_reference_s{seed}.json") for seed in (42, 43)
            },
        },
    )
    for run in export.RUNS:
        for cohort in export.COHORTS:
            prefix = f"evaluation/{run}/{cohort}"
            paths = [p for p in (out / prefix).iterdir() if p.name != "complete.json"]
            paths.append(out / f"results/{cohort}_{run}_predictions.npz")
            json_file(
                out / prefix / "complete.json",
                {
                    "files": {p.relative_to(out).as_posix(): sha(p) for p in paths},
                    "binding": {
                        "parents": 2,
                        "name": run,
                        "split": cohort,
                        "objective_arm": run.rsplit("_s", 1)[0],
                        "model_signature": models[run]["signature"],
                        "lock": {
                            "model_lock": sha(out / "model_selection_lock.json"),
                            "source": "synthetic",
                        },
                        "model_contract": {
                            "arm": "plain",
                            "attention_mode": "causal",
                            "public_output": "organized_final_state_only",
                            "private_memory_exposed": False,
                            "memory_write_sources": export.WRITE_SOURCES["plain"],
                        },
                    },
                    "schema": "babel-content-guard-evaluation-v1",
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
        return json.load(
            tar.extractfile(next(n for n in names if n.endswith("/export_manifest.json")))
        ), names


def test_complete_download_retains_all16_parent_position_prediction_evidence(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    (out / "session.lock").touch()
    before = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_report(result["archive"])
    assert result["archive"].endswith("run_reports.tar.gz")
    assert result["report_integrity"] == "complete"
    for suffix in ("parent_rows.npz", "position_rows.npz", "_predictions.npz"):
        assert sum(name.endswith(suffix) for name in names) == 16
    assert report["metrics_file"] == "content_guard_metrics.json"
    assert report["source_files_created_or_rewritten"] == 0
    assert before == {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    with tarfile.open(result["archive"]) as tar:
        for item in report["included"]:
            assert (
                hashlib.sha256(tar.extractfile("run/" + item["path"]).read()).hexdigest()
                == item["sha256"]
            )
    first = Path(result["archive"]).read_bytes()
    second = export.export_reports(out, tmp_path / "download")
    assert Path(second["previous_archive"]).read_bytes() == first


@pytest.mark.parametrize(
    "broken",
    [
        "missing_parent",
        "negative_mse",
        "initial_missing",
        "source_changed",
        "parent_count",
        "lock_missing",
    ],
)
def test_complete_training_does_not_hide_incomplete_download(tmp_path, broken):
    out = synthetic_complete(tmp_path / "run")
    parent = out / "evaluation/feedback_s43/test/parent_rows.npz"
    if broken == "missing_parent":
        parent.unlink()
    elif broken == "negative_mse":
        bank = score_arrays()
        bank["errors::full/price/near"][0] = -1e-24
        np.savez(parent, **bank)
    elif broken == "initial_missing":
        (out / "initial_s42/checkpoint.json").unlink()
    elif broken == "source_changed":
        json_file(out / "source_validation.json", {"changed": True})
    elif broken == "parent_count":
        json_file(out / "evaluation/feedback_s43/test/rows.json", [{}])
    else:
        json_file(out / "model_selection_lock.json", {"models": {"control_s42": {}}})
    result = export.export_reports(out, tmp_path / "download")
    report, _ = read_report(result["archive"])
    assert report["run_status"] == "complete_requires_review"
    assert result["report_integrity"] == "incomplete"


@pytest.mark.parametrize(
    "code,status", [(1, "failed"), (124, "timeout"), (130, "stopped"), (143, "stopped")]
)
def test_failed_and_stopped_export_and_omissions(tmp_path, code, status):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": "running"})
    (out / "model.pt").write_bytes(b"synthetic")
    (out / "raw").mkdir()
    np.savez(out / "raw/source.npz", x=np.zeros((1, 255, 28)))
    (out / "predictions").mkdir()
    np.save(out / "predictions/all.npy", np.zeros((1, 128)))
    result = export.export_reports(out, tmp_path / "download", command_exit_code=code)
    report, names = read_report(result["archive"])
    assert report["run_status"] == status and result["report_integrity"] == "incomplete"
    assert not any(name.endswith((".npy", ".npz", ".pt")) for name in names)
    assert (
        report["export_neural_forwards"]
        == report["export_readout_fits"]
        == report["export_training_updates"]
        == 0
    )


def test_export_refuses_active_producer_and_symlink(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with (out / "session.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active"):
            export.export_reports(out, tmp_path / "download")
    (out / "bad.json").symlink_to(tmp_path / "secret.json")
    with pytest.raises(ValueError, match="Symlink"):
        export.export_reports(out, tmp_path / "download")


def test_script_exports_early_failure_and_normalizes_thread_env(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/babel_content_guard768_autodl.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    out = tmp_path / "never_created"
    env = os.environ | {
        "PYTHON_BIN": sys.executable,
        "BABEL_CONTENT_GUARD_RUN": str(out),
        "BABEL_CONTENT_GUARD_LOG": str(tmp_path / "none.log"),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "OMP_NUM_THREADS": "bad",
    }
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "report_integrity=incomplete" in result.stdout
    assert "libgomp" not in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    "producer_status,command_code,expected_code,expected_status",
    [
        ("complete_requires_review", 0, 2, "complete_requires_review"),
        ("failed", 0, 0, "failed"),
        ("complete_requires_review", 1, 0, "failed"),
        ("stopped", 130, 0, "stopped"),
        ("timeout", 124, 0, "timeout"),
    ],
)
def test_export_cli_signals_completed_but_incomplete_after_saving_archive(
    tmp_path, producer_status, command_code, expected_code, expected_status
):
    out, download = tmp_path / "run", tmp_path / "download"
    json_file(out / "status.json", {"status": producer_status})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "obson.babel.content_guard_export",
            "--out",
            str(out),
            "--download",
            str(download),
            "--command-exit-code",
            str(command_code),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_code, result.stderr
    assert "Download archive:" in result.stdout and "report_integrity=incomplete" in result.stdout
    report, _ = read_report(download / "run_reports.tar.gz")
    assert report["run_status"] == expected_status
    assert report["report_integrity"]["status"] == "incomplete"
    if expected_code:
        assert "Producer completed, but the saved report is incomplete" in result.stderr


def test_wrapper_propagates_integrity_failure_but_preserves_inspectable_archive(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/babel_content_guard768_autodl.sh"
    out, download = tmp_path / "run", tmp_path / "download"
    json_file(out / "status.json", {"status": "complete_requires_review"})
    env = os.environ | {
        "PYTHON_BIN": sys.executable,
        "BABEL_CONTENT_GUARD_RUN": str(out),
        "BABEL_CONTENT_GUARD_LOG": str(tmp_path / "none.log"),
        "BABEL_DOWNLOAD_DIR": str(download),
    }
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "original command exit=0" in result.stderr
    assert "report_integrity=incomplete" in result.stdout
    report, _ = read_report(download / "run_reports.tar.gz")
    assert report["run_status"] == "complete_requires_review"
    assert report["command_exit_code"] == 0  # Original producer result is not rewritten.


def test_complete_export_cli_still_succeeds(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "obson.babel.content_guard_export",
            "--out",
            str(out),
            "--download",
            str(tmp_path / "download"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "report_integrity=complete" in result.stdout


def test_wrapper_keeps_producer_failure_and_exports_diagnostics(tmp_path):
    """A fake supervisor fails; the real exporter must keep its original status."""
    script = Path(__file__).resolve().parents[2] / "scripts/babel_content_guard768_autodl.sh"
    bridge = tmp_path / "python_bridge.py"
    observed = tmp_path / "observed.json"
    bridge.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "if sys.argv[2] == 'obson.babel.content_guard_run':\n"
        f" Path({str(observed)!r}).write_text(json.dumps({{'args':sys.argv[1:],'threads':os.environ['OMP_NUM_THREADS']}}))\n"
        " sys.exit(7)\n"
        f"os.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])\n"
    )
    bridge.chmod(0o755)
    out, download = tmp_path / "never_created", tmp_path / "download"
    env = os.environ | {
        "PYTHON_BIN": str(bridge),
        "BABEL_CONTENT_GUARD_RUN": str(out),
        "BABEL_CONTENT_GUARD_SOURCE": str(tmp_path / "source_control180"),
        "BABEL_CONTENT_GUARD_LOG": str(tmp_path / "none.log"),
        "BABEL_DOWNLOAD_DIR": str(download),
        "OMP_NUM_THREADS": "invalid",
    }
    result = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    report, _ = read_report(download / "never_created_reports.tar.gz")
    assert report["command_exit_code"] == 7 and report["run_status"] == "failed"
    request = json.loads(observed.read_text())
    assert request["threads"] == "4"
    assert request["args"] == [
        "-m",
        "obson.babel.content_guard_run",
        "--source",
        str(tmp_path / "source_control180"),
        "--out",
        str(out),
    ]
    assert not out.exists()
