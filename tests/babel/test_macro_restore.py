"""Recovery gates, historical-kernel equivalence, interrupted training and actual exports.

Neural inputs/weights in this file are tiny synthetic fixtures only.
"""

import copy
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from test_bar_alignment import small_model

from obson.babel import macro_restore_backup as backup
from obson.babel import macro_restore_data as data
from obson.babel import macro_restore_run as run
from obson.babel import macro_restore_stage as stage
from obson.babel import model_release
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny():
    model, arrays, stats, local = small_model()
    card = stage.recipe(data.source_contract())
    card.update(epochs=3, batch=2, micro=1, evaluation_micro=1, budget=4, pool_size=4)
    return model, arrays, stats, local, card


def compare(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            compare(x, y)
    else:
        assert a == b


def test_actual_historical_training_kernel_and_selection_preserved(tmp_path):
    model, arrays, stats, local, card = tiny()
    reference = copy.deepcopy(model)
    initial_decoder = copy.deepcopy(model.core.decoder.state_dict())
    binding = {"synthetic": True}
    result = stage.train(
        model, arrays, arrays, stats, local, card, binding, 42, tmp_path / "new", "cpu"
    )
    enc, head = stage.optimizers(reference, card)
    choices = [(0, stage.validate_model(reference, arrays, stats, local, card, "cpu"))]
    for epoch in range(1, 4):
        for opt, key in ((enc, "encoder_lr"), (head, "head_lr")):
            for g in opt.param_groups:
                g["lr"] = stage.ba.ar.learning_rate(epoch, card["epochs"], card[key])
        ids = stage.ws.sample_ids(4, 4, 42, epoch)
        ps = stage.ba.position_plan(4, 42, epoch)
        stage.ba.run_epoch(
            reference,
            stage.ws.subset(arrays, ids),
            stats,
            local,
            2,
            1,
            "cpu",
            True,
            ps,
            enc,
            head,
            42 + epoch * 1009,
        )
        choices.append((epoch, stage.validate_model(reference, arrays, stats, local, card, "cpu")))
    compare(result["model"], reference.state_dict())
    compare(result["encoder_optimizer"], enc.state_dict())
    compare(result["head_optimizer"], head.state_dict())
    assert (result["best_epoch"], result["best_validation"]) == min(
        choices, key=lambda r: r[1]["selection"]
    )
    compare(initial_decoder, model.core.decoder.state_dict())
    assert not all(
        torch.equal(v, result["model"][k]) for k, v in small_model()[0].state_dict().items()
    )


def test_interrupted_epoch_resume_exact_and_rebuilds_missing_best(tmp_path):
    model, arrays, stats, local, card = tiny()
    initial = copy.deepcopy(model)
    binding = {"synthetic": "same source"}
    whole = stage.train(
        copy.deepcopy(initial),
        arrays,
        arrays,
        stats,
        local,
        card,
        binding,
        42,
        tmp_path / "whole",
        "cpu",
    )
    path = tmp_path / "resume"
    first = stage.train(
        copy.deepcopy(initial),
        arrays,
        arrays,
        stats,
        local,
        card,
        binding,
        42,
        path,
        "cpu",
        stop_after=1,
    )
    assert first["epoch"] == 1
    (path / "best.pt").unlink()  # Power loss after authoritative last publication.
    resumed = stage.train(
        copy.deepcopy(initial), arrays, arrays, stats, local, card, binding, 42, path, "cpu"
    )
    for key in ("model", "best_model", "encoder_optimizer", "head_optimizer", "rng"):
        compare(whole[key], resumed[key])
    assert whole["best_epoch"] == resumed["best_epoch"]
    assert (path / "best.pt").is_file()
    assert data.read(path / "resume_check.json")["epoch"] == 1
    with pytest.raises(ValueError, match="identity"):
        stage.train(
            copy.deepcopy(initial),
            arrays,
            arrays,
            stats,
            local,
            card,
            {"synthetic": "wrong source"},
            42,
            path,
            "cpu",
        )


def test_tampered_sampling_or_best_is_not_resumed(tmp_path):
    model, arrays, stats, local, card = tiny()
    result = stage.train(
        model, arrays, arrays, stats, local, card, {}, 42, tmp_path, "cpu", stop_after=1
    )
    result["history"][0]["ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sampling"):
        stage.verify_state(result, {}, 42, card)


def test_real_worker_rejects_cpu_before_constructing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(stage, "build", lambda *a: pytest.fail("Real constructor invoked locally"))
    with pytest.raises(ValueError, match="CUDA"):
        stage.worker(tmp_path, 42)


def test_input_hash_failure_cannot_be_promoted_by_target_success(tmp_path):
    (tmp_path / "cache").mkdir()
    arrays = {
        "x": np.ones((2, 128, 28), np.float32),
        "y": np.ones((2, 128, 7), np.float32),
        "mask": np.ones((2, 128, 7), bool),
    }
    expected = {
        f"train_{k}.npy": data.write_array(tmp_path / f"{k}.npy", v) for k, v in arrays.items()
    }
    arrays["x"][0, 0, 0] += 1e-6
    report = data.cache_check(expected, tmp_path, "train", arrays)
    assert report["y"]["exact"] and report["mask"]["exact"] and not report["x"]["exact"]
    atomic_json({"training_allowed": False, "files": {}}, tmp_path / "data_receipt.json")
    with pytest.raises(ValueError, match="zero neural"):
        data.verify_prepared(tmp_path)
    assert not data.verify_prepared(tmp_path, require_inputs=False)["training_allowed"]


def test_frozen_data_receipt_rejects_modified_file(tmp_path):
    (tmp_path / "array.npy").write_bytes(b"original")
    atomic_json(
        {"training_allowed": True, "files": {"array.npy": data.digest(tmp_path / "array.npy")}},
        tmp_path / "data_receipt.json",
    )
    (tmp_path / "array.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        data.verify_prepared(tmp_path)


def test_original_contract_preserves_453_not_later439():
    contract = data.source_contract()
    assert contract["counts"] == {"train": 4789, "val": 978, "test": 984, "cross_research": 453}
    assert contract["dense_count"] == 38549
    assert contract["config"]["attention_layers"] == 2
    assert contract["config"]["attention_ff"] == 512
    card = stage.recipe(contract)
    assert card["seeds"] == [42, 43] and card["epochs"] == 200
    assert card["per_seed"]["updates"] == 7600 and not card["final_macro_restored"]


def synthetic_snapshot_run(tmp, monkeypatch, complete=True):
    repo = tmp / "repo"
    repo.mkdir()
    (repo / "code.py").write_text("# synthetic source")
    monkeypatch.setattr(data, "REPO", repo)
    out = tmp / "run"
    out.mkdir()
    meta = {
        "stage": {"stage": stage.STAGE, "seeds": [42, 43]},
        "binding": {"git_commit": "a" * 40, "files": {"code.py": data.digest(repo / "code.py")}},
    }
    # Give the allowed source its real role in the explicit binding.
    (repo / "src").mkdir()
    (repo / "code.py").rename(repo / "src/code.py")
    meta["binding"]["files"] = {"src/code.py": data.digest(repo / "src/code.py")}
    atomic_json(meta, out / "manifest.json")
    atomic_json({"synthetic": True}, out / "runtime.json")
    atomic_json({"used_seconds": 91.0, "active_since": None}, out / "usage.json")
    root = out / "data"
    root.mkdir()
    (root / "scalers.json").write_text('{"synthetic":true}')
    atomic_json(
        {"training_allowed": True, "files": {"scalers.json": data.digest(root / "scalers.json")}},
        root / "data_receipt.json",
    )
    sr = out / "stages" / stage.STAGE
    sr.mkdir(parents=True)
    (sr / "stage.lock").touch()
    for seed in [42, 43] if complete else [42]:
        p = sr / f"s{seed}"
        p.mkdir()
        for name in ("best.pt", "last.pt", "example_inputs.npz", "examples.npz", "continuation.pt"):
            (p / name).write_bytes(b"synthetic byte asset, not real model weights")
        atomic_json([], p / "history.json")
        if complete:
            atomic_json(
                {
                    "status": "passed",
                    "synthetic_fixture_only": True,
                    "binding": {"run_manifest": data.digest(out / "manifest.json")},
                    "files": {
                        n: data.digest(p / n)
                        for n in (
                            "best.pt",
                            "last.pt",
                            "history.json",
                            "examples.npz",
                            "continuation.pt",
                        )
                    },
                },
                p / "completion.json",
            )
    return out


def test_stage_backup_self_contained_byte_roundtrip_and_idempotent(tmp_path, monkeypatch):
    out = synthetic_snapshot_run(tmp_path, monkeypatch)
    saved = backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "complete")
    again = backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "complete")
    assert saved["archive_sha256"] == again["archive_sha256"]
    target = tmp_path / "restored"
    model_release.scan(Path(saved["archive"]), saved["archive_sha256"], target)
    assert (target / f"run/stages/{stage.STAGE}/s43/last.pt").read_bytes().startswith(b"synthetic")
    assert (target / "source_code/src/code.py").is_file()
    assert (target / "run/data/scalers.json").is_file()
    assert data.read(target / "run/usage.json")["used_seconds"] == 91.0
    assert (
        data.read(out / "backup_receipt.json")["off_instance"]
        == "pending_download_and_local_verification"
    )


def test_partial_seed_snapshot_keeps_partial_status_and_cannot_claim_complete(
    tmp_path, monkeypatch
):
    out = synthetic_snapshot_run(tmp_path, monkeypatch, complete=False)
    saved = backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "failed")
    assert saved["run_status"] == "failed"
    with pytest.raises(ValueError, match="CUDA restore"):
        backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "complete")


def test_budget_cannot_reset_on_resume(tmp_path, monkeypatch):
    atomic_json({"used_seconds": 11.0, "active_since": 100.0}, tmp_path / "usage.json")
    monkeypatch.setattr(run.time, "time", lambda: 120.0)
    with pytest.raises(ValueError, match="budget exhausted"):
        run.usage_start(tmp_path, 30)


def test_report_exports_even_without_started_model(tmp_path):
    out, download = tmp_path / "not_started", tmp_path / "download"
    backup.export(out, download, 17)
    with tarfile.open(download / "not_started_reports.tar.gz") as tar:
        meta = json.load(tar.extractfile("export_manifest.json"))
    assert meta["command_exit_code"] == 17
    assert meta["status"]["status"] == "failed_before_initialization"
    assert not meta["final_macro_restored"]


def test_stopped_export_preserves_stopped_model_status(tmp_path, monkeypatch):
    out = synthetic_snapshot_run(tmp_path, monkeypatch, complete=False)
    atomic_json({"status": "stopped"}, out / "status.json")
    backup.export(out, tmp_path / "download", 130)
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as tar:
        meta = json.load(tar.extractfile("export_manifest.json"))
    assert meta["command_exit_code"] == 130
    assert meta["status"]["status"] == meta["backup"]["run_status"] == "stopped"


def test_actual_shell_install_failure_exports_and_preserves_exit(tmp_path):
    wrapper = tmp_path / "python_wrapper"
    wrapper.write_text(
        '#!/usr/bin/env bash\nif [[ "$2" == obson.babel.raw_snapshot ]]; then exit 17; fi\nexec '
        + shlex.quote(sys.executable)
        + ' "$@"\n'
    )
    wrapper.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(wrapper),
        BABEL_RESTORE_RUN=str(tmp_path / "run"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    result = subprocess.run(
        ["bash", "scripts/babel_macro_restore_autodl.sh", "all"],
        cwd=data.REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17, result.stderr
    assert (tmp_path / "download/run_reports.tar.gz").is_file(), result.stderr


def test_interrupted_snapshot_publication_is_preserved_and_retryable(tmp_path, monkeypatch):
    out = synthetic_snapshot_run(tmp_path, monkeypatch)
    saved = backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "complete")
    location = Path(data.read(out / "backup_receipt.json")["location"])
    (location / "release_receipt.json").unlink()
    again = backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "complete")
    assert saved["archive_sha256"] == again["archive_sha256"]
    assert len(list(location.parent.glob(location.name + ".incomplete-*"))) == 1


def test_saved_run_can_reopen_without_git_and_data_failure_never_starts_model(
    tmp_path, monkeypatch
):
    raw, out, download = (tmp_path / n for n in ("raw", "run", "download"))
    raw.mkdir()
    out.mkdir()
    source = data.source_contract()
    meta = {
        "schema": "babel-macro-recovery-bootstrap-v1",
        "binding": {"git_commit": "1" * 40, "files": {}},
        "stage": stage.recipe(source),
        "final_macro_restored": False,
        "remaining_stages": 12,
        "four_arm_experiment": "retained_not_started",
    }
    atomic_json(meta, out / "manifest.json")
    monkeypatch.setattr(run, "binding", lambda: pytest.fail("Offline restore must not need Git"))
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 100 * 1024**3})
    )
    monkeypatch.setattr(
        run, "run_child", lambda *a, **kw: pytest.fail("Unqualified input cannot train")
    )
    monkeypatch.setattr(data, "prepare", lambda *a: {"training_allowed": False})
    with pytest.raises(ValueError, match="Original input hashes differ"):
        run.run(raw, out, download)
    assert data.read(out / "usage.json")["active_since"] is None
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert data.read(out / "status.json")["status"] == "failed"


def test_active_budget_is_frozen_into_snapshot_without_editing_live_journal(tmp_path, monkeypatch):
    out = synthetic_snapshot_run(tmp_path, monkeypatch)
    atomic_json({"used_seconds": 91.0, "active_since": 100.0}, out / "usage.json")
    monkeypatch.setattr(backup.time, "time", lambda: 125.0)
    saved = backup.stage_snapshot(out, stage.STAGE, tmp_path / "download", "complete")
    target = tmp_path / "restored"
    model_release.scan(Path(saved["archive"]), saved["archive_sha256"], target)
    assert data.read(target / "run/usage.json") == {"used_seconds": 116.0, "active_since": None}
    assert data.read(out / "usage.json")["active_since"] == 100.0


def test_discarded_continuation_replays_all_weights_and_adam_without_changing_checkpoint(tmp_path):
    model, arrays, stats, local, card = tiny()
    state = stage.train(model, arrays, arrays, stats, local, card, {}, 42, tmp_path, "cpu")
    before = copy.deepcopy(state)
    before_rng = stage.rng_state()
    original_digest = data.digest(tmp_path / "last.pt")
    expected = stage.continuation_probe(model, arrays, stats, local, card, state, 42, "cpu")
    compare(state, before)
    compare(stage.rng_state(), before_rng)
    compare(model.state_dict(), state["model"])
    loaded = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    actual = stage.continuation_probe(
        small_model()[0], arrays, stats, local, card, loaded, 42, "cpu"
    )
    stage.compare_continuation(actual, expected)
    assert data.digest(tmp_path / "last.pt") == original_digest
    assert any(not torch.equal(v, expected["model"][k]) for k, v in loaded["model"].items())
    for value in expected["encoder_optimizer"]["state"].values():
        assert value["step"].item() == 7  # Three epochs x two real batches + one diagnostic.
    damaged = copy.deepcopy(expected)
    next(iter(damaged["model"].values())).add_(1)
    with pytest.raises(AssertionError):
        stage.compare_continuation(actual, damaged)
