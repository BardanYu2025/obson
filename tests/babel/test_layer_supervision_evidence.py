"""Synthetic endpoint, matrix and immutable export tests; no real weights."""

import copy
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
import torch

from obson.babel import layer_supervision as model
from obson.babel import layer_supervision_evaluate as evaluation
from obson.babel import layer_supervision_export as export
from obson.babel import multiscale_evaluate as inherited
from obson.babel import observed_history_export_v2 as corrected

STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}
TINY = {"width": 16, "heads": 4, "ff": 32, "layers": 4}


@pytest.fixture(autouse=True)
def cpu_runtime():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def identities(n):
    return [
        {"key": f"TEST/15/c{i % 3}", "month": f"2025-{i % 6 + 1:02d}", "parent": i}
        for i in range(n)
    ]


def bank(n=120):
    rows = identities(n)
    keys = ["full/price/composite", *evaluation.protection_keys()]
    support_keys = [f"price/{b}" for b in ("near", "mid", "far", "composite")]
    support_keys += [f"nonprice/{f}/{b}" for f in evaluation.FIELDS for b in ("near", "mid", "far")]
    return {
        "errors": {k: np.ones(n) for k in keys},
        "support": {k: np.ones(n, np.int64) for k in keys},
        "contexts": {k: np.ones(n, np.int64) for k in keys if not k.startswith("overlap/")},
        "position_support": {k: np.ones((n, 128), np.int64) for k in support_keys},
        "mask_digest": "same-synthetic-masks",
        "rows_digest": evaluation._digest(rows),
        "data_digest": "same-synthetic-data",
    }, rows


def named_banks(factors):
    base, rows = bank()
    banks = {}
    for split in ("test", "cross_research"):
        banks[split] = {}
        for name in evaluation.RUNS:
            arm = name.rsplit("_s", 1)[0]
            item = copy.deepcopy(base)
            item["errors"] = {k: v * factors[arm] for k, v in item["errors"].items()}
            item.update(
                model_name=name,
                split=split,
                model_lock_digest="synthetic-complete-six",
                model_contract={"arm": evaluation.MODEL_ARMS[arm], "attention_mode": "causal"},
            )
            banks[split][name] = item
    return banks, dict.fromkeys(banks, rows)


def test_physical_scoring_and_gate_functions_are_unchanged():
    for name in (
        "score_rows",
        "overlap_rows",
        "aggregate_rows",
        "comparison",
        "paired_intervals",
        "protection_keys",
    ):
        assert getattr(evaluation, name) is getattr(inherited, name)
    assert export._score_array is corrected._score_array


@pytest.mark.parametrize(
    "factors,primary,budget",
    [
        ({"initial": 1.0, "control": 0.8, "layer2": 0.8}, False, True),
        ({"initial": 1.0, "control": 1.0, "layer2": 0.8}, True, False),
        ({"initial": 1.0, "control": 0.8, "layer2": 0.6}, True, True),
    ],
)
def test_budget_and_mechanism_conclusions_do_not_substitute(factors, primary, budget):
    banks, rows = named_banks(factors)
    result = evaluation.decisions(banks, rows)
    assert result["limited_hypothesis_supported"] is primary
    assert result["budget_hypothesis_supported"] is budget
    assert result["primary_comparison"] == "layer2_vs_control"
    assert all(len(v["comparisons"]) == 4 for v in result["comparison_groups"].values())
    assert result["promoted"] is False and result["fit_count"] == 0
    assert result["f11"]["general_reuse"] == "unknown_not_tested"


@pytest.mark.parametrize(
    "mutation",
    ["missing_initial", "mislabeled_initial", "other_lock", "noncausal", "missing_cohort"],
)
def test_full_six_model_two_cohort_identity_required(mutation):
    banks, rows = named_banks({"initial": 1.0, "control": 0.8, "layer2": 0.6})
    if mutation == "missing_initial":
        del banks["test"]["initial_s42"]
    elif mutation == "mislabeled_initial":
        banks["test"]["initial_s42"]["model_contract"]["arm"] = "initial"
    elif mutation == "other_lock":
        banks["test"]["initial_s43"]["model_lock_digest"] = "different"
    elif mutation == "noncausal":
        banks["test"]["layer2_s42"]["model_contract"]["attention_mode"] = "full_observed"
    else:
        del banks["cross_research"]
    with pytest.raises(ValueError):
        evaluation.decisions(banks, rows)


def test_primary_gain_cannot_hide_activity_damage():
    banks, rows = named_banks({"initial": 1.0, "control": 0.8, "layer2": 0.6})
    banks["cross_research"]["layer2_s43"]["errors"]["full/nonprice/oi_per_volume_asinh/far"][:] = (
        2.0
    )
    result = evaluation.decisions(banks, rows)
    assert not result["limited_hypothesis_supported"] and result["budget_hypothesis_supported"]


def source(n=1):
    x = np.zeros((n, 255, 28), np.float32)
    x[..., 23:28] = 1
    x[..., 18:23] = 0.01
    raw = 100 * np.exp(np.random.default_rng(7).normal(0, 0.001, (n, 255)).cumsum(1))
    return x, raw


def extract(out, name="layer2_s42", *, pair=None, verifier=None):
    arm = evaluation.MODEL_ARMS[name.rsplit("_s", 1)[0]]
    if pair is None:
        pair = model.make_models(arm, 42, "cpu", TINY)
    if verifier is None:

        def verifier(*_):
            return {
                "model_lock": "synthetic-six-models",
                "source": "synthetic",
                "raw_receipt": "synthetic",
            }

    x, raw = source()
    bank = evaluation.extract(
        {"statistics": STATS, "epochs": 60, "batch": 128, "micro": 8},
        out,
        name,
        *pair,
        x,
        raw,
        identities(1),
        "test",
        "cpu",
        verify_lock=verifier,
    )
    return bank, pair


def test_final_only_extraction_all128_fixed8_and_resume(tmp_path, monkeypatch):
    pair = model.make_models("layer2", 42, "cpu", TINY)

    def forbid(*_):
        raise AssertionError("Training-only intermediate path used in final evaluation")

    monkeypatch.setattr(pair[0], "forward_levels", forbid)
    calls = []
    pair[0].register_forward_pre_hook(lambda *args: calls.append(1))
    bank, _ = extract(tmp_path, pair=pair)
    assert len(calls) == 26
    receipt = json.loads((tmp_path / bank["receipt"]).read_text())
    assert receipt["states"] == 128 and receipt["updates"] == receipt["fits"] == 0
    with np.load(tmp_path / "results/test_layer2_s42_predictions.npz") as saved:
        assert saved["predictions"].shape == (1, 8, 127, 7)
        np.testing.assert_array_equal(saved["positions"], evaluation.SAMPLE_POSITIONS)
    with np.load(tmp_path / "evaluation/layer2_s42/test/position_rows.npz") as saved:
        assert saved["errors::price/near"].shape == (1, 128)
    extract(tmp_path, pair=pair)
    assert len(calls) == 26


def test_initial_maps_to_control_and_partial_research_is_not_replayed(tmp_path):
    bank, pair = extract(tmp_path, "initial_s42")
    assert bank["model_contract"]["arm"] == "control"
    path = tmp_path / "predictions/initial_s42/test/parents_00000_00001.npy"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        extract(tmp_path, "initial_s42", pair=pair)
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    pair = model.make_models("layer2", 42, "cpu", TINY)

    def fail(*_):
        raise RuntimeError("synthetic failure")

    hook = pair[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        extract(interrupted, pair=pair)
    hook.remove()
    with pytest.raises(ValueError, match="export-only"):
        extract(interrupted, pair=pair)


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
    json_file(
        out / "model_selection_lock.json",
        {
            "schema": "babel-layer-supervision768-v1",
            "selection": "fixed_last60",
            "source": "synthetic",
            "models": {run: {} for run in export.RUNS},
        },
    )
    for run in export.TRAINING_RUNS:
        files = {p.name: sha(p) for p in (out / run).iterdir() if p.name != "completion.json"}
        json_file(out / run / "completion.json", {"files": files, "epoch": 60})
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
                        "lock": {
                            "model_lock": sha(out / "model_selection_lock.json"),
                            "source": "synthetic",
                        },
                        "model_contract": {
                            "arm": "layer2" if run.startswith("layer2_") else "control",
                            "attention_mode": "causal",
                            "public_output": "final_block_only",
                        },
                    },
                    "schema": "babel-layer-supervision-evaluation-v1",
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


def test_complete_download_retains_all12_parent_position_prediction_evidence(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    (out / "session.lock").touch()
    before = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_report(result["archive"])
    assert result["archive"].endswith("run_reports.tar.gz")
    assert result["report_integrity"] == "complete"
    for suffix in ("parent_rows.npz", "position_rows.npz", "_predictions.npz"):
        assert sum(name.endswith(suffix) for name in names) == 12
    assert report["metrics_file"] == "layer_supervision_metrics.json"
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
    parent = out / "evaluation/layer2_s43/test/parent_rows.npz"
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
        json_file(out / "evaluation/layer2_s43/test/rows.json", [{}])
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
    script = Path(__file__).resolve().parents[2] / "scripts/babel_layer_supervision768_autodl.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    out = tmp_path / "never_created"
    env = os.environ | {
        "PYTHON_BIN": sys.executable,
        "BABEL_LAYER_SUPERVISION_RUN": str(out),
        "BABEL_LAYER_SUPERVISION_LOG": str(tmp_path / "none.log"),
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


def test_six_model_two_cohort_tiny_pipeline_exports_every_score_bank(tmp_path):
    """Actual tiny neural forwards, no real-data acceptance or training simulation."""
    out = tmp_path / "tiny"
    out.mkdir()
    models = {
        name: model.make_models(
            evaluation.MODEL_ARMS[name.rsplit("_s", 1)[0]], int(name[-2:]), "cpu", TINY
        )
        for name in evaluation.RUNS
    }
    banks, row_bank = {}, {}
    lock_calls = []

    def verifier(meta, path, name, split):
        assert path == out and name in models and len(models) == 6
        lock_calls.append((name, split))
        return {
            "model_lock": "synthetic-six-models",
            "source": "synthetic",
            "raw_receipt": "synthetic",
        }

    for split in export.COHORTS:
        x, raw = source()
        rows = identities(1)
        row_bank[split] = rows
        banks[split] = {
            name: evaluation.extract(
                {"statistics": STATS, "epochs": 60, "micro": 8, "batch": 128},
                out,
                name,
                *pair,
                x,
                raw,
                rows,
                split,
                "cpu",
                verify_lock=verifier,
            )
            for name, pair in models.items()
        }
    assert len(lock_calls) == 24
    decision = evaluation.decisions(banks, row_bank)
    assert not decision["limited_hypothesis_supported"]
    assert not decision["budget_hypothesis_supported"]
    assert all(
        v["checks"]["gain"]["status"] == "insufficient"
        for group in decision["comparison_groups"].values()
        for v in group["comparisons"].values()
    )
    json_file(out / "layer_supervision_metrics.json", decision)
    json_file(out / "status.json", {"status": "synthetic_component_complete"})
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_report(result["archive"])
    assert result["report_integrity"] == "incomplete"  # No training/source receipts simulated.
    for suffix in ("parent_rows.npz", "position_rows.npz", "_predictions.npz"):
        assert sum(name.endswith(suffix) for name in names) == 12
    assert sum(item["reason"] == "full128_prediction_shard" for item in report["omitted"]) == 12
    assert not any(name.endswith((".pt", ".npy")) for name in names)


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
            "obson.babel.layer_supervision_export",
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
    script = Path(__file__).resolve().parents[2] / "scripts/babel_layer_supervision768_autodl.sh"
    out, download = tmp_path / "run", tmp_path / "download"
    json_file(out / "status.json", {"status": "complete_requires_review"})
    env = os.environ | {
        "PYTHON_BIN": sys.executable,
        "BABEL_LAYER_SUPERVISION_RUN": str(out),
        "BABEL_LAYER_SUPERVISION_LOG": str(tmp_path / "none.log"),
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
            "obson.babel.layer_supervision_export",
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
