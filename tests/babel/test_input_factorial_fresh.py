"""Portable recovery and actual tiny five-arm lifecycle; synthetic CPU only."""

import fcntl
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_input_factorial_evaluate import banks
from test_input_factorial_run import CONFIG, STATS, synthetic, write

from obson.babel import input_factorial_fresh_data as data
from obson.babel import input_factorial_fresh_export as export
from obson.babel import input_factorial_fresh_run as run
from obson.babel import raw_snapshot as snapshot


@pytest.fixture(autouse=True)
def cpu():
    threads = torch.get_num_threads()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fast)


def package(tmp):
    raw, dataset, bundle = tmp / "raw", tmp / "dataset", tmp / "bundle"
    (raw / "rb").mkdir(parents=True)
    (raw / "rb" / "rb2401_15m.csv").write_text("date,close\n2024-01-01,1\n")
    (raw / "aggregate.csv").write_text("excluded")
    dataset.mkdir()
    (dataset / "rows.json").write_text("[]")
    snapshot.pack(raw, dataset, bundle)
    return raw, dataset, bundle


def test_portable_roundtrip_deterministic_and_idempotent(tmp_path):
    raw, dataset, bundle = package(tmp_path)
    second = tmp_path / "second"
    assert snapshot.pack(raw, dataset, second) == snapshot.manifest(bundle)
    target = tmp_path / "restored"
    got = snapshot.install(bundle, target)
    assert got["raw_files"] == 1 and len(got["files"]) == 2
    assert snapshot.install(bundle, target) == got
    assert not (target / "contracts/aggregate.csv").exists()
    (target / "contracts/rb/extra_15m.csv").write_text("new")
    with pytest.raises(ValueError, match="Additional"):
        snapshot.verify(bundle, target)


@pytest.mark.parametrize("damage", ["missing", "hash", "order", "overwrite", "symlink"])
def test_data_loss_or_collision_rejected_before_replacement(tmp_path, damage):
    _, _, bundle = package(tmp_path)
    target = tmp_path / "restored"
    part = bundle / "payload.part000"
    if damage == "missing":
        part.unlink()
    elif damage == "hash":
        part.write_bytes(b"bad")
    elif damage == "order":
        m = json.loads((bundle / "manifest.json").read_text())
        m["parts"][0]["name"] = "payload.part001"
        write(bundle / "manifest.json", m)
    elif damage == "overwrite":
        (target / "contracts/rb").mkdir(parents=True)
        (target / "contracts/rb/rb2401_15m.csv").write_text("must retain")
    else:
        target.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        snapshot.install(bundle, target)
    if damage == "overwrite":
        assert (target / "contracts/rb/rb2401_15m.csv").read_text() == "must retain"


@pytest.mark.parametrize("kind", ["traversal", "linked", "duplicate", "missing_member"])
def test_valid_hash_cannot_hide_unsafe_tar(tmp_path, kind):
    _, _, bundle = package(tmp_path)
    m = json.loads((bundle / "manifest.json").read_text())
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        name = next(iter(m["files"]))
        info = tarfile.TarInfo("../outside" if kind == "traversal" else name)
        if kind == "linked":
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside"
        else:
            info.size = m["files"][name]["bytes"]
        content = b"date,close\n2024-01-01,1\n"
        tar.addfile(info, io.BytesIO(content))
        if kind == "duplicate":
            tar.addfile(info, io.BytesIO(content))
    payload = stream.getvalue()
    (bundle / "payload.part000").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    m["parts"] = [{"name": "payload.part000", "bytes": len(payload), "sha256": sha}]
    m["payload_sha256"] = sha
    write(bundle / "manifest.json", m)
    with pytest.raises(ValueError):
        snapshot.install(bundle, tmp_path / "target")
    assert not (tmp_path / "outside").exists()


def test_selection_full_span_partition_and_training_scalers(monkeypatch):
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=600, freq="15min"),
            "open": np.arange(600) + 100.0,
            "close": np.arange(600) + 101.0,
        }
    )
    series = SimpleNamespace(
        key="TEST/15/a", code="TEST", period=15, frame=frame, main=np.ones(600, bool)
    )
    series.main[270] = False
    same = np.arange(600) < 500
    monkeypatch.setattr(data.partitions, "time_mask", lambda *args: same)
    rows, pool = data.choose_rows([series], "train", 5)
    assert rows == data.choose_rows([series], "train", 5)[0]
    assert pool > 5 and all(254 <= r["row"] < 500 and r["row"] != 270 for r in rows)
    x = np.zeros((600, 28), np.float32)
    x[:, 23:28] = 1
    x[:, 1] = np.arange(600) / 1000
    stats = data.fit_statistics({series.key: series}, {series.key: x}, rows, split="train")
    x[500:] = 1e10
    assert data.fit_statistics({series.key: series}, {series.key: x}, rows, split="train") == stats
    with pytest.raises(ValueError, match="Only locked training"):
        data.fit_statistics({series.key: series}, {series.key: x}, rows, split="val")
    with pytest.raises(ValueError, match="Insufficient"):
        data.choose_rows([series], "train", 1000)


def test_masks_preserve_missing_and_suspect_activity():
    x = np.zeros((3, 28), np.float32)
    x[:, 23:28] = 1
    x[0, 20] = 2  # impossible nonzero OI change with zero denominator
    x[1, 27] = 0
    x[2, 21] = 2  # OI/volume outside original support
    mask = data.activity_mask(x)
    assert not mask[0, 4:6].any() and not mask[1, 3] and not mask[2, 4:6].any()


def test_five_arm_actual_tiny_training_resume_and_common_initialization(tmp_path, monkeypatch):
    for module in (run.bar, run.factorial):
        monkeypatch.setattr(module, "EPOCHS", 2)
        monkeypatch.setattr(module, "BATCH", 4)
        monkeypatch.setattr(module, "MICRO", 2)
    monkeypatch.setattr(run.factorial, "DIAGNOSTIC_EPOCHS", (0, 2))
    meta = {
        "encoder": CONFIG,
        "statistics": STATS,
        "counts": {"train": 8},
        "source_identity": {"synthetic": True},
    }
    write(tmp_path / "manifest.json", meta)
    run.prepare_initializations(meta, tmp_path, "cpu")
    x, close = synthetic()
    for job in run.jobs()[:5]:
        module, mm, root, actual = run.worker_location(meta, tmp_path, job)
        done = module.train_job(mm, root, actual, x, close, x[:2], close[:2], "cpu")
        assert done["epoch"] == 2 and done["updates"] == 4
        assert module.train_job(mm, root, actual, x, close, x[:2], close[:2], "cpu") == done
    assert run.remaining_epochs(meta, tmp_path)["reference_bar_s42"] == 118


def test_new_decisions_keep_all_gates_but_do_not_claim_historical_replay():
    values, rows = banks()
    result = run.decisions(values, rows)
    assert result["historical_replay"] is False
    assert "historical_bar_anchor" not in result["comparison_groups"]
    assert "concurrent_bar_anchor" in result["comparison_groups"]
    assert (
        sum(
            len(case["checks"])
            for g in result["comparison_groups"].values()
            for case in g["comparisons"].values()
        )
        == 1580
    )
    assert not result["promoted"]


def test_preflight_failure_cannot_be_bypassed_on_restart(tmp_path, monkeypatch):
    bundle, target, out = (tmp_path / n for n in ("bundle", "data", "run"))
    out.mkdir()
    write(out / "preflight.json", {"passed": False})
    monkeypatch.setattr(run, "make_manifest", lambda *a: {"synthetic": True})
    calls = []

    class Process:
        pid = 123
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command[-1])

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(run.subprocess, "Popen", Process)
    with pytest.raises(ValueError, match="preflight failed"):
        run.supervise(bundle, target, out)
    assert calls == ["initialize"]
    assert run.read_json(out / "budget.json")["active"] is None


def test_unsettled_budget_rejected_and_no_worker_started(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    write(out / "budget.json", {"elapsed_seconds": 0, "active": {"stage": "lost"}})
    monkeypatch.setattr(run, "make_manifest", lambda *a: {})
    monkeypatch.setattr(run.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="without settling"):
        run.supervise(tmp_path / "bundle", tmp_path / "data", out)


def test_complete_receipt_cannot_omit_bank_evidence(tmp_path):
    write(
        tmp_path / "completion.json",
        {"schema": run.SCHEMA, "status": "complete_requires_review", "files": {}},
    )
    with pytest.raises(ValueError, match="inventory"):
        run.check_completion({}, tmp_path)


def test_full_backup_keeps_checkpoints_data_and_predictions_report_does_not(tmp_path):
    out, download, dataset = tmp_path / "run", tmp_path / "download", tmp_path / "dataset"
    write(dataset / "rows.json", [{"synthetic": True}])
    write(out / "manifest.json", {"source": str(dataset), "implementation": {}})
    write(out / "status.json", {"status": "failed"})
    for name in (
        "bar_control/bar_s42/last.pt",
        "predictions/a/test/parents_00000_00008.npy",
        "pending_epoch.json",
    ):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic")
    export.export(out, download, code=1)
    with tarfile.open(download / "run_reports.tar.gz") as tar:
        assert not any(n.endswith(".pt") for n in tar.getnames())
        manifest = json.load(tar.extractfile("export_manifest.json"))
        assert not manifest["execution_complete"]
    with tarfile.open(download / "run_backup.tar.gz") as tar:
        names = tar.getnames()
        assert "obson/checkpoints/run/bar_control/bar_s42/last.pt" in names
        assert "obson/checkpoints/run/predictions/a/test/parents_00000_00008.npy" in names
        assert "data/factorial_fresh/rows.json" in names
        for name, value in json.load(tar.extractfile("backup_manifest.json"))["files"].items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == value["sha256"]
    with (out / "run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            export.export(out, download)


def test_failed_atomic_export_retains_previous_archive(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"before")
    dest = tmp_path / "reports.tar.gz"
    dest.write_bytes(b"previous")
    original = export.digest
    count = 0

    def changed(path):
        nonlocal count
        count += 1
        return original(path) if count == 1 else "changed"

    monkeypatch.setattr(export, "digest", changed)
    with pytest.raises(ValueError, match="changed while"):
        export.archive({"data": source}, dest, {"kind": "reports"})
    assert dest.read_bytes() == b"previous"


@pytest.mark.parametrize("failed_module", ["none", "raw_snapshot", "input_factorial_fresh_run"])
def test_real_shell_stops_on_error_and_always_exports(tmp_path, failed_module):
    import os
    import subprocess
    import sys

    fake = tmp_path / "python"
    log = tmp_path / "calls.jsonl"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,json\n"
        "with open(os.environ['CALLS'],'a') as f: f.write(json.dumps({'args':sys.argv[1:], 'omp':os.environ['OMP_NUM_THREADS']})+'\\n')\n"
        "sys.exit(9 if sys.argv[2].rsplit('.',1)[-1] == os.environ['FAIL_MODULE'] else 0)\n"
    )
    fake.chmod(0o755)
    env = os.environ | {
        "PYTHON_BIN": str(fake),
        "CALLS": str(log),
        "FAIL_MODULE": failed_module,
        "BABEL_FRESH_RUN": str(tmp_path / "run"),
        "BABEL_DATA_ROOT": str(tmp_path / "data"),
        "BABEL_DOWNLOAD_DIR": str(tmp_path / "download"),
        "OMP_NUM_THREADS": "invalid",
    }
    completed = subprocess.run(
        ["bash", str(run.REPO / "scripts/babel_input_factorial_fresh_autodl.sh"), "all"],
        env=env,
        capture_output=True,
    )
    assert completed.returncode == (0 if failed_module == "none" else 9)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(c["omp"] == "4" for c in calls)
    assert calls[-1]["args"][1] == "obson.babel.input_factorial_fresh_export"
    assert calls[-1]["args"][-1] == str(completed.returncode)
    assert len(calls) == (2 if failed_module == "raw_snapshot" else 3)


def test_worker_refuses_real_cpu_market_execution(tmp_path, monkeypatch):
    meta = {"bundle": "bundle", "data_root": "data"}
    write(tmp_path / "manifest.json", meta)
    monkeypatch.setattr(run, "make_manifest", lambda *a: meta)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires AutoDL CUDA"):
        run.worker(tmp_path, "initialize")


def test_release_cannot_substitute_different_but_well_formed_snapshot(tmp_path):
    _, _, bundle = package(tmp_path)
    target = tmp_path / "restored"
    snapshot.install(bundle, target)
    with pytest.raises(ValueError, match="predeclared cohort"):
        run.source_identity(bundle, target)


def test_second_launcher_cannot_overwrite_running_status(tmp_path, monkeypatch):
    import sys

    out = tmp_path / "run"
    out.mkdir()
    write(out / "status.json", {"status": "running", "stage": "owned"})
    monkeypatch.setattr(sys, "argv", ["fresh", "--out", str(out)])
    with (out / "run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run.main()
    assert json.loads((out / "status.json").read_text()) == {"status": "running", "stage": "owned"}
