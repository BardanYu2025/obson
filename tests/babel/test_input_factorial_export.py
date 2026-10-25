"""Synthetic report integrity; no real checkpoint or neural execution."""

import copy
import fcntl
import hashlib
import json
import subprocess
import sys
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from obson.babel import input_factorial as model
from obson.babel import input_factorial_evaluate as evaluation
from obson.babel import input_factorial_export as export
from obson.babel import input_factorial_run as runner


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


@lru_cache(maxsize=2)
def initialized(seed):
    return model.initialization_receipt(seed, "cpu", TINY)


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
    json_file(out / "manifest.json", {"encoder": TINY, "protocol": runner.protocol()})
    inits = {seed: copy.deepcopy(initialized(seed)) for seed in (42, 43)}
    for seed, init in inits.items():
        json_file(out / f"initialization_s{seed}.json", init)
    models = {}
    for run in export.TRAIN_RUNS:
        arm, seed = run.rsplit("_s", 1)
        seed = int(seed)
        (out / run / "last.pt").write_bytes(b"synthetic omitted checkpoint")
        job = next(j for j in runner.jobs() if j["name"] == run)
        binding = runner.binding(out, job)
        final_signature = hashlib.sha256((run + "final").encode()).hexdigest()
        initial_signature = inits[seed]["arms"][arm]["signature"]
        signatures = [hashlib.sha256(f"{run}/{i}".encode()).hexdigest() for i in range(1, 121)]
        signatures[-1] = final_signature
        json_file(
            out / run / "training_summary.json",
            {"history": [{"parameter_signature": v} for v in signatures]},
        )
        for epoch in (0, 10, 120):
            json_file(
                out / run / f"gradient_diagnostic_epoch{epoch}.json",
                {
                    "binding": {
                        "binding": binding,
                        "epoch": epoch,
                        "signature": initial_signature if epoch == 0 else signatures[epoch - 1],
                        "x_sha256": "1" * 64,
                        "raw_sha256": "2" * 64,
                    },
                    "complete": True,
                    "updates": 0,
                    "fits": 0,
                    "report": {
                        "schema": "babel-input-factorial-gradient-diagnostic-v1",
                        "arm": arm,
                        "seed": seed,
                        "updates": 0,
                        "panel": "first8_locked_train_parents_epoch1_uniform5_sampling_slots",
                        "parents": 8,
                        "supervised_states": 40,
                        "weights": {"path": 0.4, "change1": 0.2, "body": 0.1, "activity": 0.3},
                        "all_component_sums_match_objective": True,
                        "parameters_and_buffers_unchanged": True,
                        "global_clip": {"applied": False},
                    },
                },
            )
        models[run] = {
            "binding": binding,
            "epoch": 120,
            "updates": 4560,
            "signature": final_signature,
            "initial_signature": initial_signature,
            "files": {p.name: sha(p) for p in (out / run).iterdir() if p.name != "completion.json"},
        }
        models[run].update(
            arm=arm,
            initialization=binding["initialization"],
            file=f"{run}/last.pt",
            sha256=models[run]["files"]["last.pt"],
            model_contract=binding["model_contract"],
        )
        json_file(out / run / "completion.json", models[run])
    refs, controls = {}, {}
    for seed in (42, 43):
        controls[str(seed)] = {
            "job": {"name": f"bar_s{seed}", "arm": "bar", "seed": seed},
            "checkpoint": f"bar_s{seed}/last.pt",
            "checkpoint_sha256": hashlib.sha256(f"old{seed}".encode()).hexdigest(),
            "signature": hashlib.sha256(f"old{seed}signature".encode()).hexdigest(),
            "binding": {"source": "old-source", "seed": seed},
            "epoch": 120,
            "updates": 4560,
        }
    source = {
        "schema": "babel-input-factorial-source-v1",
        "source": "/historical",
        "controls": controls,
    }
    refs = runner.reference_records(
        {"model_source": source["source"], "source_identity": source, "encoder": TINY}
    )
    json_file(
        out / "source_validation.json",
        {
            "source_identity": source,
            "training_source_weights_loaded": False,
            "source_optimizer_loaded": False,
            "optimizer_updates": 0,
            "reference_weights_loaded": True,
            "reference_use": "historical_frozen_anchor_not_concurrent_factorial_control",
            "references": refs,
        },
    )
    json_file(
        out / "model_selection_lock.json",
        {
            "schema": "babel-input-factorial768-v1",
            "selection": "fixed_last120",
            "manifest": sha(out / "manifest.json"),
            "readout_fits": 0,
            "source": source,
            "models": models,
            "references": refs,
            "initialization": {
                str(seed): sha(out / f"initialization_s{seed}.json") for seed in (42, 43)
            },
        },
    )
    for run in export.RUNS:
        selected = refs[run] if run in refs else models[run]
        contract = selected.get("model_contract", selected.get("binding", {}).get("model_contract"))
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
                        "model_signature": selected["signature"],
                        "model_contract": contract,
                        "rows_digest": evaluation._digest(identities(2)),
                        "positions": list(range(1, 129)),
                        "sample_positions": list(export.POSITIONS),
                        "lock": {
                            "model_lock": sha(out / "model_selection_lock.json"),
                            "source": source,
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


def test_complete_snapshot_retains_twenty_banks_eight_workers_and_all_diagnostics(tmp_path):
    out = synthetic_complete(tmp_path / "run")
    (out / "session.lock").touch()
    before = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    result = export.export_reports(out, tmp_path / "download")
    report, names = read_report(result["archive"])
    assert result["report_integrity"] == "complete", report["report_integrity"]
    assert report["metrics_file"] == "input_factorial_metrics.json"
    assert (
        report["source_files_created_or_rewritten"]
        == report["export_neural_forwards"]
        == report["export_training_updates"]
        == 0
    )
    for suffix in ("parent_rows.npz", "position_rows.npz", "_predictions.npz"):
        assert sum(name.endswith(suffix) for name in names) == 20
    assert sum("gradient_diagnostic_epoch" in n for n in names) == 24
    assert not any(n.endswith(".pt") for n in names)
    assert before == {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
    first = Path(result["archive"]).read_bytes()
    second = export.export_reports(out, tmp_path / "download")
    assert Path(second["previous_archive"]).read_bytes() == first


@pytest.mark.parametrize(
    "damage",
    [
        "diagnostic",
        "reference_bank",
        "checkpoint",
        "negative_mse",
        "contract",
        "signature",
        "initializer",
    ],
)
def test_complete_producer_cannot_hide_missing_or_corrupt_evidence(tmp_path, damage):
    out = synthetic_complete(tmp_path / "run")
    if damage == "diagnostic":
        (out / "dense_random_s42/gradient_diagnostic_epoch10.json").unlink()
    elif damage == "reference_bank":
        (out / "evaluation/reference_bar_s43/cross_research/parent_rows.npz").unlink()
    elif damage == "checkpoint":
        (out / "hybrid_endpoint_s43/last.pt").write_bytes(b"changed")
    elif damage == "negative_mse":
        bank = score_arrays()
        bank["errors::full/price/near"][0] = -1e-24
        np.savez(out / "evaluation/hybrid_endpoint_s43/test/parent_rows.npz", **bank)
    elif damage == "initializer":
        json_file(out / "initialization_s42.json", {"passed": True})
    else:
        path = out / "evaluation/reference_bar_s42/test/complete.json"
        data = json.loads(path.read_text())
        if damage == "contract":
            data["binding"]["model_contract"] = model.contract("dense_endpoint", TINY)
        else:
            data["binding"]["model_signature"] = "a" * 64
        json_file(path, data)
        complete = json.loads((out / "completion.json").read_text())
        complete["files"][path.relative_to(out).as_posix()] = sha(path)
        json_file(out / "completion.json", complete)
    result = export.export_reports(out, tmp_path / "download")
    report, _ = read_report(result["archive"])
    assert result["report_integrity"] == "incomplete"
    assert report["run_status"] == "complete_requires_review"


@pytest.mark.parametrize(
    "damage",
    ["warmstart", "reference_updates", "copied_init", "common", "missing_arm", "budget", "parity"],
)
def test_semantic_damage_remains_rejected_after_outer_hash_rebinding(tmp_path, damage):
    out = synthetic_complete(tmp_path / "run")
    lock = json.loads((out / "model_selection_lock.json").read_text())
    init = json.loads((out / "initialization_s42.json").read_text())
    source = json.loads((out / "source_validation.json").read_text())
    if damage == "warmstart":
        source["training_source_weights_loaded"] = True
    elif damage == "reference_updates":
        lock["references"]["reference_bar_s42"]["updates"] = 1
        source["references"] = lock["references"]
    elif damage == "copied_init":
        init["arms"]["hybrid_endpoint"]["signature"] = "f" * 64
    elif damage == "common":
        init["arms"]["hybrid_endpoint"]["common_signature"] = "f" * 64
    elif damage == "missing_arm":
        del lock["models"]["dense_random_s43"]
    elif damage == "budget":
        lock["models"]["hybrid_endpoint_s42"]["epoch"] = 119
    else:
        init["hybrid_endpoint_original_bar_equivalent"] = True
    json_file(out / "initialization_s42.json", init)
    json_file(out / "source_validation.json", source)
    lock["initialization"]["42"] = sha(out / "initialization_s42.json")
    for arm in evaluation.ARMS:
        selected = lock["models"][f"{arm}_s42"]
        selected["binding"]["initialization"] = lock["initialization"]["42"]
        selected["initialization"] = lock["initialization"]["42"]
        json_file(out / f"{arm}_s42/completion.json", selected)
    json_file(out / "model_selection_lock.json", lock)
    paths = [p for p in out.rglob("*") if p.is_file()]
    inventory = {p.relative_to(out).as_posix(): sha(p) for p in paths}
    present = {p.relative_to(out).as_posix(): sha(p) for p in paths if p.suffix != ".pt"}
    with pytest.raises(ValueError):
        export._validate_model_lock(out, present, inventory)


@pytest.mark.parametrize(
    "code,status", [(1, "failed"), (124, "timeout"), (130, "stopped"), (143, "stopped")]
)
def test_failed_stopped_exports_are_readonly_and_diagnostic_only(tmp_path, code, status):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": "running"})
    (out / "last.pt").write_bytes(b"synthetic")
    result = export.export_reports(out, tmp_path / "download", command_exit_code=code)
    report, names = read_report(result["archive"])
    assert result["run_status"] == status and result["report_integrity"] == "incomplete"
    assert not any(n.endswith(".pt") for n in names)
    assert report["source_files_created_or_rewritten"] == 0


def test_active_lock_symlink_and_missing_source_are_handled(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with (out / "session.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Active"):
            export.export_reports(out, tmp_path / "download")
    (out / "bad.json").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="Symlink"):
        export.export_reports(out, tmp_path / "download")
    missing = tmp_path / "missing"
    result = export.export_reports(missing, tmp_path / "download", command_exit_code=1)
    assert not missing.exists() and result["report_integrity"] == "incomplete"


def test_cli_completed_producer_but_incomplete_report_exits_nonzero(tmp_path):
    out = tmp_path / "run"
    json_file(out / "status.json", {"status": "complete_requires_review"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "obson.babel.input_factorial_export",
            "--out",
            str(out),
            "--download",
            str(tmp_path / "download"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert "report_integrity=incomplete" in result.stdout


@pytest.mark.parametrize(
    "damage", ["gradient_update", "gradient_epoch", "gradient_panel", "reference_source"]
)
def test_gradient_and_reference_semantics_rejected_after_file_hash_rebinding(tmp_path, damage):
    out = synthetic_complete(tmp_path / "run")
    lock = json.loads((out / "model_selection_lock.json").read_text())
    if damage == "reference_source":
        lock["references"]["reference_bar_s42"]["checkpoint_sha256"] = "f" * 64
        source = json.loads((out / "source_validation.json").read_text())
        source["references"] = lock["references"]
        json_file(out / "source_validation.json", source)
    else:
        run = "dense_random_s42"
        path = out / run / "gradient_diagnostic_epoch10.json"
        diagnostic = json.loads(path.read_text())
        if damage == "gradient_update":
            diagnostic["report"]["updates"] = 1
        elif damage == "gradient_epoch":
            diagnostic["binding"]["epoch"] = 11
        else:
            diagnostic["binding"]["x_sha256"] = "f" * 64
        json_file(path, diagnostic)
        selected = lock["models"][run]
        selected["files"][path.name] = sha(path)
        json_file(out / run / "completion.json", selected)
    json_file(out / "model_selection_lock.json", lock)
    paths = [p for p in out.rglob("*") if p.is_file()]
    inventory = {p.relative_to(out).as_posix(): sha(p) for p in paths}
    present = {p.relative_to(out).as_posix(): sha(p) for p in paths if p.suffix != ".pt"}
    with pytest.raises(ValueError, match="gradient|Historical"):
        export._validate_model_lock(out, present, inventory)
