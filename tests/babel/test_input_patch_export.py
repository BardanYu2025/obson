"""CPU-only report snapshot and wrapper regressions; no model execution."""

import fcntl
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from obson.babel import input_patch as model
from obson.babel import input_patch_evaluate as evaluation
from obson.babel import input_patch_export as export


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


TINY = {"width": 16, "heads": 4, "ff": 32, "layers": 4}


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
    json_file(
        out / "source_validation.json",
        {
            "source_identity": "synthetic-data-only",
            "source_neural_weights_loaded": False,
            "source_optimizer_loaded": False,
            "optimizer_updates": 0,
        },
    )
    initialized = {}
    for seed in (42, 43):
        initialized[seed] = {
            "seed": seed,
            "passed": True,
            "checks": {
                "common_backbone_coordinates_query_equal": True,
                "local4_patch4_full_tensors_equal": True,
                "query_scalers_zero_one": True,
                "parameter_difference_only_input": True,
            },
            "arms": {
                arm: {
                    "signature": hashlib.sha256(
                        f"{seed}-{'bar' if arm == 'bar' else 'shared4'}".encode()
                    ).hexdigest(),
                    "common_signature": hashlib.sha256(f"{seed}-common".encode()).hexdigest(),
                    "model_contract": model.contract(arm, TINY),
                }
                for arm in evaluation.ARMS
            },
        }
        json_file(out / f"initialization_s{seed}.json", initialized[seed])
    models = {}
    for run in export.RUNS:
        arm, seed = run.rsplit("_s", 1)
        seed = int(seed)
        (out / run / "last.pt").write_bytes(b"synthetic omitted weight identity")
        binding = {
            "manifest": sha(out / "manifest.json"),
            "model_contract": model.contract(arm, TINY),
            "job": {"name": run, "arm": arm, "seed": seed},
            "initialization_sha256": sha(out / f"initialization_s{seed}.json"),
        }
        models[run] = {
            "binding": binding,
            "epoch": 120,
            "updates": 4560,
            "signature": hashlib.sha256((run + "-final").encode()).hexdigest(),
            "initial_signature": initialized[seed]["arms"][arm]["signature"],
            "files": {p.name: sha(p) for p in (out / run).iterdir() if p.name != "completion.json"},
        }
        json_file(out / run / "completion.json", models[run])
    json_file(
        out / "model_selection_lock.json",
        {
            "schema": "babel-input-patch768-v1",
            "selection": "fixed_last120",
            "manifest": sha(out / "manifest.json"),
            "readout_fits": 0,
            "source": "synthetic-data-only",
            "models": models,
            "initialization": {
                str(seed): sha(out / f"initialization_s{seed}.json") for seed in (42, 43)
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
                        "model_signature": models[run]["signature"],
                        "model_contract": models[run]["binding"]["model_contract"],
                        "rows_digest": evaluation._digest(identities(2)),
                        "positions": list(range(1, 129)),
                        "sample_positions": list(export.POSITIONS),
                        "lock": {
                            "model_lock": sha(out / "model_selection_lock.json"),
                            "source": "synthetic-data-only",
                        },
                    },
                    "schema": evaluation.SCHEMA,
                    "status": "complete",
                    "states": 256,
                    "updates": 0,
                    "fits": 0,
                    "embeddings_saved": False,
                    "rows_digest": evaluation._digest(identities(2)),
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


def test_complete_download_retains_all12_banks_and_init_receipts_unchanged(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    (out / "session.lock").touch()
    before = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_report(result["archive"])
    assert result["report_integrity"] == "complete"
    assert report["metrics_file"] == "input_patch_metrics.json"
    assert report["source_files_created_or_rewritten"] == 0
    assert (
        report["export_neural_forwards"]
        == report["export_training_updates"]
        == report["export_readout_fits"]
        == 0
    )
    for suffix in ("parent_rows.npz", "position_rows.npz", "_predictions.npz"):
        assert sum(name.endswith(suffix) for name in names) == 12
    assert sum(name.endswith("/metrics.json") for name in names) == 12
    assert all("run/initialization_s" + str(seed) + ".json" in names for seed in (42, 43))
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
        "summary",
        "parent",
        "negative_mse",
        "initialization",
        "lock",
        "checkpoint",
        "contract",
        "signature",
        "positions",
        "worker_completion",
    ],
)
def test_completed_producer_does_not_hide_missing_or_relabelled_evidence(tmp_path, broken):
    out = synthetic_complete(tmp_path / "run")
    if broken == "summary":
        (out / "evaluation/local4_s42/test/metrics.json").unlink()
    elif broken == "parent":
        (out / "evaluation/patch4_s43/cross_research/parent_rows.npz").unlink()
    elif broken == "negative_mse":
        bank = score_arrays()
        bank["errors::full/price/near"][0] = -1e-24
        np.savez(out / "evaluation/patch4_s43/test/parent_rows.npz", **bank)
    elif broken == "initialization":
        json_file(out / "initialization_s42.json", {"passed": True})
    elif broken == "lock":
        json_file(out / "model_selection_lock.json", {"models": {"bar_s42": {}}})
    elif broken == "checkpoint":
        (out / "patch4_s43/last.pt").write_bytes(b"different")
    elif broken == "worker_completion":
        json_file(out / "patch4_s42/completion.json", {"files": {"last.pt": "0" * 64}})
    else:
        path = out / "evaluation/patch4_s42/test/complete.json"
        data = json.loads(path.read_text())
        if broken == "contract":
            data["binding"]["model_contract"]["token_stride"] = 1
        elif broken == "signature":
            data["binding"]["model_signature"] = "0" * 64
        else:
            data["binding"]["positions"] = list(range(1, 33))
        json_file(path, data)
    result = export.export_reports(out, tmp_path / "download")
    report, _ = read_report(result["archive"])
    assert report["run_status"] == "complete_requires_review"
    assert result["report_integrity"] == "incomplete"


@pytest.mark.parametrize(
    "code,status", [(1, "failed"), (124, "timeout"), (130, "stopped"), (143, "stopped")]
)
def test_failed_stopped_reports_keep_diagnostics_and_omit_weights(tmp_path, code, status):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": "running"})
    (out / "last.pt").write_bytes(b"synthetic")
    (out / "raw").mkdir()
    np.savez(out / "raw/source.npz", x=np.zeros((1, 255, 28)))
    result = export.export_reports(out, tmp_path / "download", command_exit_code=code)
    report, names = read_report(result["archive"])
    assert result["run_status"] == status and result["report_integrity"] == "incomplete"
    assert not any(name.endswith((".pt", ".npz")) for name in names)
    assert {x["reason"] for x in report["omitted"]} >= {
        "weights_or_optimizer_state",
        "raw_or_feature_cache_or_full_predictions",
    }


def test_no_active_export_no_symlink_and_no_source_creation(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with (out / "session.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active"):
            export.export_reports(out, tmp_path / "download")
    (out / "bad.json").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="Symlink"):
        export.export_reports(out, tmp_path / "download")
    missing = tmp_path / "never_created"
    result = export.export_reports(missing, tmp_path / "download", command_exit_code=1)
    assert not missing.exists() and result["report_integrity"] == "incomplete"


@pytest.mark.parametrize(
    "producer,code,want",
    [
        ("complete_requires_review", 0, 2),
        ("failed", 0, 0),
        ("complete_requires_review", 1, 0),
        ("stopped", 130, 0),
    ],
)
def test_cli_saves_report_and_distinguishes_producer_from_integrity(tmp_path, producer, code, want):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": producer})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "obson.babel.input_patch_export",
            "--out",
            str(out),
            "--download",
            str(tmp_path / "download"),
            "--command-exit-code",
            str(code),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == want, result.stderr
    assert "Download archive:" in result.stdout and "report_integrity=incomplete" in result.stdout
    assert (tmp_path / "download/run_reports.tar.gz").is_file()


def indexed_files(out):
    paths = [p for p in out.rglob("*") if p.is_file()]
    inventory = {p.relative_to(out).as_posix(): sha(p) for p in paths}
    present = {p.relative_to(out).as_posix(): sha(p) for p in paths if p.suffix != ".pt"}
    return present, inventory


@pytest.mark.parametrize(
    "change",
    [
        "common_signature",
        "initial_contract",
        "selected_contract",
        "initial_missing",
        "warmstart",
        "budget",
    ],
)
def test_semantic_lock_errors_rejected_even_after_outer_hashes_are_rebound(tmp_path, change):
    out = synthetic_complete(tmp_path / "run")
    lock = json.loads((out / "model_selection_lock.json").read_text())
    init = json.loads((out / "initialization_s42.json").read_text())
    if change == "common_signature":
        init["arms"]["patch4"]["common_signature"] = "f" * 64
    elif change == "initial_contract":
        init["arms"]["patch4"]["model_contract"] = model.contract("local4", TINY)
    elif change == "initial_missing":
        del init["arms"]["local4"]
    elif change == "selected_contract":
        lock["models"]["patch4_s42"]["binding"]["model_contract"] = model.contract("local4", TINY)
    elif change == "budget":
        lock["models"]["patch4_s42"]["epoch"] = 119
    else:
        source = json.loads((out / "source_validation.json").read_text())
        source["source_neural_weights_loaded"] = True
        json_file(out / "source_validation.json", source)
    json_file(out / "initialization_s42.json", init)
    lock["initialization"]["42"] = sha(out / "initialization_s42.json")
    for arm in evaluation.ARMS:
        selected = lock["models"][f"{arm}_s42"]
        selected["binding"]["initialization_sha256"] = lock["initialization"]["42"]
        json_file(out / f"{arm}_s42/completion.json", selected)
    json_file(out / "model_selection_lock.json", lock)
    present, inventory = indexed_files(out)
    with pytest.raises(ValueError):
        export._validate_model_lock(out, present, inventory)


def test_export_model_signature_check_is_distinct_from_file_hash_checks(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    path = out / "evaluation/patch4_s42/test/complete.json"
    receipt = json.loads(path.read_text())
    receipt["binding"]["model_signature"] = "a" * 64
    json_file(path, receipt)
    # Rebind the root completion after changing receipt contents. Presence and
    # SHA alone now pass; the final signature must still match the six-model lock.
    complete = json.loads((out / "completion.json").read_text())
    complete["files"][path.relative_to(out).as_posix()] = sha(path)
    json_file(out / "completion.json", complete)
    result = export.export_reports(out, tmp_path / "download")
    report, _ = read_report(result["archive"])
    issues = report["report_integrity"]["binding_issues"]
    assert result["report_integrity"] == "incomplete"
    assert any("Endpoint evidence" in issue.get("detail", "") for issue in issues)
    assert not any(
        issue["reason"] == "source_file_missing_or_fingerprint_changed" for issue in issues
    )
