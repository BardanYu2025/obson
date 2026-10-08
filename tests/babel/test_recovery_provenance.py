"""V14 synthetic arithmetic, actual worker commit path, negative controls/export."""

import copy
import os
import random
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
import test_recovery_context as fixture_support
import torch

from obson.babel import prefix_readout as original
from obson.babel import recovery_provenance as core
from obson.babel import recovery_provenance_run as run
from obson.babel.ae_extend import atomic_json

runtime = fixture_support.runtime
worker = fixture_support.worker


def targets():
    y = np.zeros((3, 128, 7), np.float32)
    y[..., 0] = np.arange(128)
    for c in range(1, 7):
        y[..., c] = c
    return {"y": y, "mask": np.ones_like(y, dtype=bool)}


def test_independent_local_oracle_anchor_exclusion_mask_and_floor():
    data = targets()
    stats = {"y_mean": [0.0] * 7, "y_scale": [1.0] * 7}
    # None of these queried current positions may enter its own preceding16.
    data["y"][:, [31, 63, 95, 127], :] = 1e6
    data["mask"][0, 20, 5] = False
    data["y"][0, 20, 5] = np.nan  # masked values are not observations.
    result = core.local_moments(data, stats)
    assert result["mean"][0] == 8.5
    assert result["scale"][0] == pytest.approx(np.arange(1, 17).std())
    assert result["delta_scale"] == 1.0
    assert result["mean"][1:] == list(range(1, 7))
    assert result["scale"][1:] == [1e-5] * 6
    assert result["count"][5] == 3 * 4 * 16 - 1
    data["mask"][0, 14, 0] = False
    with pytest.raises(ValueError, match="anchor"):
        core.local_moments(data, stats)


def test_independent_local_rounding_matches_bound_original_algorithm():
    rng = np.random.default_rng(412)
    data = {
        "y": rng.normal(size=(33, 128, 7)).astype(np.float32),
        "mask": rng.random((33, 128, 7)) > 0.2,
    }
    data["mask"][..., 0] = True
    stats = {"y_mean": rng.normal(size=7).tolist(), "y_scale": np.exp(rng.normal(size=7)).tolist()}
    reference = original.fit_target_scales(
        [original.targets(data, stats, p) for p in original.PREFIXES]
    )
    refit = core.local_moments(data, stats)
    assert all(v["passed"] for v in run.scalar_checks(refit, reference, 1e-10, 1e-10).values())
    refit["mean"][0] += 0.00001
    assert not run.scalar_checks(refit, reference, 1e-10, 1e-10)["mean"]["passed"]


def test_state_population_moments_and_nonfinite_rejection():
    result = core.state_moments([[1, 5], [3, 5], [5, 5]])
    assert result["mean"] == [3.0, 5.0]
    assert result["scale"] == [pytest.approx(np.sqrt(8 / 3)), 1e-6]
    with pytest.raises(ValueError, match="Finite"):
        core.state_moments([[float("nan")], [1]])


def test_exact_tree_catches_optimizer_rng_dtype_and_shape():
    ref = {
        "optimizer": {"state": {0: {"exp_avg": torch.ones(3), "step": torch.tensor(2.0)}}},
        "rng": torch.arange(10, dtype=torch.uint8),
    }
    assert not core.compare_tree(copy.deepcopy(ref), ref)
    for field in ("moment", "step", "rng", "dtype", "shape"):
        bad = copy.deepcopy(ref)
        if field == "moment":
            bad["optimizer"]["state"][0]["exp_avg"][1] += 0.1
        elif field == "step":
            bad["optimizer"]["state"][0]["step"] += 1
        elif field == "rng":
            bad["rng"][0] += 1
        elif field == "dtype":
            bad["rng"] = bad["rng"].float()
        else:
            bad["rng"] = bad["rng"][None]
        assert core.compare_tree(bad, ref)


def synthetic_probe():
    return {
        "python": random.random(),
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4).tolist(),
        "torch_cuda": [],
    }


def invoke_audit(fixture, root, mode, monkeypatch):
    production = run.production
    root.mkdir(exist_ok=True)
    atomic_json({"synthetic": True}, root / "manifest.json")
    atomic_json([0, 1, 2], root / "audit_train_rows.json")
    job = {"arm": "B_full_common", "seed": 42, "name": "resume_s42"}
    folder = root / job["name"]
    folder.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    monkeypatch.setattr(run, "rng_probe", synthetic_probe)
    hooks = run.install_hooks(folder, mode)
    try:
        return production.train_job(
            fixture.meta, fixture.context, fixture.r1, root, job, fixture.x, fixture.x, "cpu"
        )
    finally:
        production.train_epoch, run.lr.pending_epoch = hooks


def test_real_worker_commit_hook_and_restore_equivalence_synthetic(worker, monkeypatch, tmp_path):
    # The real CUDA entrypoint additionally enforces distinct subprocess PIDs.
    uninterrupted = tmp_path / "continuous_s42"
    resumed = tmp_path / "restarted_s42"
    assert invoke_audit(worker, uninterrupted, "continuous", monkeypatch)["optimizer_updates"] == 2
    with pytest.raises(SystemExit) as e:
        invoke_audit(worker, resumed, "interrupt", monkeypatch)
    assert e.value.code == 73
    proof = run.read_json(resumed / "resume_s42/intentional_exit.json")
    assert proof["updates"] == 1
    assert proof["checkpoint_sha256"] == run.sha256(resumed / "resume_s42/last.pt")
    assert invoke_audit(worker, resumed, "resume", monkeypatch)["optimizer_updates"] == 2
    assert all(v["passed"] for v in run.compare_paths(tmp_path, 42).values())


def test_omitted_rng_restore_fails_even_if_eval_training_is_deterministic(
    worker, monkeypatch, tmp_path
):
    invoke_audit(worker, tmp_path / "continuous_s42", "continuous", monkeypatch)
    with pytest.raises(SystemExit):
        invoke_audit(worker, tmp_path / "restarted_s42", "interrupt", monkeypatch)
    monkeypatch.setattr(run.production, "restore_rng", lambda state: None)
    invoke_audit(worker, tmp_path / "restarted_s42", "resume", monkeypatch)
    with pytest.raises(ValueError, match="mismatched"):
        run.compare_paths(tmp_path, 42)
    saved = run.read_json(tmp_path / "resume_comparison_s42.json")
    assert not saved["checks"]["rng_before_epoch2.json"]["passed"]


def test_omitted_optimizer_restore_stops_before_second_update(worker, monkeypatch, tmp_path):
    folder = tmp_path / "restarted_s42"
    with pytest.raises(SystemExit):
        invoke_audit(worker, folder, "interrupt", monkeypatch)
    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="Optimizer state update count"):
        invoke_audit(worker, folder, "resume", monkeypatch)
    assert not (folder / "resume_s42/rng_before_epoch2.json").exists()


def test_protocol_budget_and_source_binding(tmp_path):
    p = run.protocol()
    assert p["budget"]["total_updates"] == 8
    assert (
        p["budget"]["updates_per_path"] * p["budget"]["paths_per_seed"] * len(p["budget"]["seeds"])
        == 8
    )
    f = tmp_path / "source"
    f.write_text("old")
    receipt = {}
    run.bound(f, run.sha256(f), receipt)
    f.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        run.bound(f, receipt[str(f.resolve())], {})


def test_export_success_and_failed_receipts(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    out.mkdir()
    atomic_json({"status": "v14_checks_complete_requires_review"}, out / "status.json")
    np.savez(out / "state_train_s42.npz", states=np.ones((3, 2)))
    (out / "last.pt").write_bytes(b"not exported")
    env = dict(
        os.environ,
        BABEL_V14_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN=str(repo / ".venv/bin/python"),
    )
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_provenance_autodl.sh", "export"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as tar:
        assert "run/state_train_s42.npz" in tar.getnames()
        assert not any(n.endswith(".pt") for n in tar.getnames())
        assert (
            b"v14_checks_complete_requires_review"
            in tar.extractfile("run/export_status.txt").read()
        )
    # A failed initial invocation must still provide a downloadable report.
    env["BABEL_V14_RUN"] = str(tmp_path / "failed")
    env["BABEL_V14_SOURCE"] = str(tmp_path / "absent")
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_provenance_autodl.sh", "all"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    with tarfile.open(tmp_path / "download/failed_reports.tar.gz") as tar:
        assert b"run_status=failed" in tar.extractfile("failed/export_status.txt").read()


def test_actual_subprocess_exit_and_committed_resume(worker, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "synthetic_child.py"
    script.write_text("""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_recovery_context import models, STATS
from test_recovery_qualification import data
from test_recovery_provenance import invoke_audit
from obson.babel import recovery_provenance_run as run
p = Path(sys.argv[1])
mp = pytest.MonkeyPatch()
run.r0.apply_runtime(dict(run.r0.RUNTIME_PROFILES['context'], threads=1))
mp.setattr(run.production, 'construct', lambda *a: models())
mp.setattr(run.production, 'EPOCHS', 2)
mp.setattr(run.production.old.parent, 'stats', lambda m: (STATS, {}))
f = SimpleNamespace(meta={'batch':128, 'micro':2, 'validation_prefixes':[32,64,96,128]},
                    context=p/'context', r1=p/'r1', x=data(3)[0])
invoke_audit(f, p/sys.argv[2], sys.argv[3], mp)
""")
    env = dict(os.environ, PYTHONPATH=f"{repo / 'src'}:{repo / 'tests/babel'}")
    for folder, mode, code in (
        ("continuous_s42", "continuous", 0),
        ("restarted_s42", "interrupt", 73),
        ("restarted_s42", "resume", 0),
    ):
        result = subprocess.run(
            [str(repo / ".venv/bin/python"), str(script), str(tmp_path), folder, mode],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == code, result.stdout + result.stderr
    assert all(v["passed"] for v in run.compare_paths(tmp_path, 42).values())


def test_timeout_exports_partial_report_and_stops_receipt(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "run"
    out.mkdir()
    atomic_json({"status": "running"}, out / "status.json")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    timer = fakebin / "timeout"
    timer.write_text("#!/bin/sh\nexit 124\n")
    timer.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
        BABEL_V14_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN=str(repo / ".venv/bin/python"),
    )
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_provenance_autodl.sh", "all"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 124
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as tar:
        assert b"run_status=timeout" in tar.extractfile("run/export_status.txt").read()


def test_shell_invokes_only_bounded_v14_worker_and_passes_bundle_source(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    timer = fakebin / "timeout"
    args = tmp_path / "args.txt"
    timer.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$V14_TEST_ARGS"\nexit 124\n')
    timer.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
        BABEL_V14_RUN=str(tmp_path / "audit"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        BABEL_V14_BUNDLE_SOURCE=str(tmp_path / "original_bundle"),
        V14_TEST_ARGS=str(args),
    )
    result = subprocess.run(
        ["bash", "scripts/babel_recovery_provenance_autodl.sh", "all"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 124
    lines = args.read_text().splitlines()
    assert lines[:3] == ["--signal=TERM", "--kill-after=30s", "7200s"]
    assert lines[4:6] == ["-m", "obson.babel.recovery_provenance_startup"]
    assert lines[lines.index("--bundle-source") + 1] == str(tmp_path / "original_bundle")
    assert "obson.babel.recovery_context_run" not in lines


def test_preflight_uses_bound_original_train_and_retained_bundle_only(tmp_path, monkeypatch):
    source, bundle, training, original_cache, alignment, out = [
        tmp_path / n
        for n in ("source", "bundle", "training", "architecture/cache", "alignment/cache", "out")
    ]
    for p in (source, bundle / "bundle", training, original_cache, alignment, out):
        p.mkdir(parents=True)
    stats = {"x_mean": [0.0] * 28, "x_scale": [1.0] * 28, "y_mean": [0.0] * 7, "y_scale": [1.0] * 7}
    local = {"mean": [0.0] * 7, "scale": [1.0] * 7}
    atomic_json(stats, original_cache / "statistics.json")
    atomic_json(local, alignment / "local_scales.json")
    for k, shape, dtype in [
        ("x", (4789, 128, 28), "float32"),
        ("y", (4789, 128, 7), "float32"),
        ("mask", (4789, 128, 7), "bool"),
    ]:
        arr = np.lib.format.open_memmap(
            original_cache / f"train_{k}.npy", mode="w+", dtype=dtype, shape=shape
        )
        arr.flush()
        del arr
    idx = {"files": {p.name: run.sha256(p) for p in original_cache.iterdir()}}
    atomic_json(idx, original_cache / "index.json")
    atomic_json({}, training / "fit.json")
    lineage = {
        "schema": "babel-causal-path768-v1",
        "source": str(alignment.parent),
        "identity": {
            "manifest": {
                "schema": "babel-bar-alignment-v1",
                "original_source": str(original_cache.parent),
            }
        },
    }
    delivery = {
        "schema": "babel-control600-delivery-v1",
        "identity": {
            "training_source": str(training),
            "training_files": {"fit.json": run.sha256(training / "fit.json")},
            "training_manifest": lineage,
        },
    }
    atomic_json(delivery, bundle / "manifest.json")
    atomic_json({}, bundle / "bundle/index.json")
    for seed in (42, 43):
        (bundle / f"bundle/control_s{seed}_best.pt").write_bytes(b"synthetic identity only")
    meta = {
        "identity": {
            "manifest": delivery,
            "files": {
                str(p.relative_to(bundle)): run.sha256(p) for p in bundle.rglob("*") if p.is_file()
            },
        }
    }
    monkeypatch.setattr(
        run.production,
        "verify_source",
        lambda *a: (meta, source, source, {"torch": torch.__version__, "gpu": "synthetic"}),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic")
    monkeypatch.setattr(run, "ARCH_INDEX", run.sha256(original_cache / "index.json"))
    monkeypatch.setattr(run, "LOCAL_SHA", run.sha256(alignment / "local_scales.json"))
    monkeypatch.setattr(
        run.production, "STATISTICS_SHA256", run.sha256(original_cache / "statistics.json")
    )
    monkeypatch.setattr(run.production.old.parent, "stats", lambda m: (stats, local))
    found = run.preflight(source, bundle, out)
    assert found[3]["x"].shape == (4789, 128, 28)
    assert found[4] == stats and found[5] == local
    assert len(found[-1]) == 11
    (original_cache / "train_y.npy").write_bytes(b"bad")
    with pytest.raises(ValueError, match="train_y.npy"):
        run.preflight(source, bundle, out)
