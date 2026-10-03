"""Synthetic CPU checks only; all neural optimizer updates disabled."""

import copy
import os
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import context_transfer as core
from obson.babel import context_transfer_data as data
from obson.babel import context_transfer_evaluate as ev
from obson.babel import context_transfer_run as run
from obson.babel import uniform_context as uc
from obson.babel.ae_extend import atomic_json, atomic_save
from obson.babel.dual_state import sha256

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def no_optimization():
    torch.set_num_threads(1)
    old = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield
    torch.backends.mha.set_fastpath_enabled(old)


def inputs(n=3):
    raw = np.random.default_rng(42).normal(size=(n, 255, 28)).astype("float32") * 0.01
    raw[..., [11, 23, 24, 25, 26, 27]] = 1
    return torch.tensor(raw)


def model(seed=42):
    return uc.make_models("rolling", seed, "cpu", CONFIG)


def scores(value=1.0):
    return {
        f"{view}/{context}/{metric}": value
        for view in ("full", "native")
        for context in ("endpoint", "interior")
        for metric in core.METRICS
    }


def cache(out, split, n=3):
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    np.savez(folder / f"{split}.npz", x=inputs(n).numpy())
    atomic_json(
        [{"month": f"2024-{i % 12 + 1:02d}", "period": 15, "symbol": "X"} for i in range(n)],
        folder / f"{split}_rows.json",
    )
    atomic_json({}, folder / f"{split}_audit.json")
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {
                name: sha256(folder / name)
                for name in (f"{split}.npz", f"{split}_rows.json", f"{split}_audit.json")
            },
        },
        folder / f"{split}_index.json",
    )


def test_original_forward_checkpoint_and_native_full_context():
    e, q = model()
    x = inputs(2)
    ps = torch.tensor([[32, 128], [64, 96]])
    direct = e(uc.rolling_windows(x, ps).flatten(0, 1))[:, -1].reshape(2, 2, -1)
    actual = core.states(e, x, ps)
    torch.testing.assert_close(actual, direct)
    native = core.states(e, x, ps, True)
    torch.testing.assert_close(native, e(x[:, -128:])[torch.arange(2)[:, None], ps - 1])
    a, b = copy.deepcopy(e), copy.deepcopy(e)
    core.states(a, x, ps, checkpointing=True).square().sum().backward()
    core.states(b, x, ps, checkpointing=False).square().sum().backward()
    for p, v in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, v.grad)


def test_targets_have_full_history_and_ignore_future():
    x = inputs(2)
    ps = torch.full((2, 1), 32)
    full, fm = uc.targets(x, ps, STATS)
    native, nm = uc.targets(x, ps, STATS, True)
    assert fm[..., 0].sum() == 2 * 127 and nm[..., 0].sum() == 2 * 31
    torch.testing.assert_close(full[:, :, :31], native[:, :, :31])
    y = x.clone()
    y[:, 159:] += 10
    a, _ = uc.targets(y, ps, STATS)
    torch.testing.assert_close(a, full)
    e, _ = model()
    torch.testing.assert_close(core.states(e, x, ps), core.states(e, y, ps))


def test_source_protection_epoch_zero_and_minimax():
    initial = scores()
    assert core.score(initial, initial) == {"eligible": True, "score": 1.0, "worst_retention": 1.0}
    better = scores(0.8)
    assert core.score(better, initial)["eligible"]
    better["native/interior/far"] = 1.11
    assert not core.score(better, initial)["eligible"]
    better = scores(0.5)
    better["full/endpoint/price"] = 0.9
    assert core.score(better, initial)["score"] == 0.9
    better["full/endpoint/activity"] = np.nan
    with pytest.raises(ValueError):
        core.score(better, initial)
    with pytest.raises(ValueError):
        core.score({}, initial)


def test_time_stop_is_independent_of_scores_and_finite():
    assert core.stop("rolling", 60, 1, None)
    assert not core.stop("prefix", 59, 1000, 100)
    assert core.stop("prefix", 60, 100, 100)
    assert not core.stop("prefix", 120, 99, 100)
    assert core.stop("prefix", 300, 90, 100)
    assert not core.time_match(90, 100)["matched"]
    assert core.time_match(102, 100)["matched"]
    assert not core.time_match(106, 100)["matched"]
    with pytest.raises(ValueError):
        core.stop("prefix", 1, 1, None)


def test_micro_gradient_equivalence(monkeypatch):
    monkeypatch.setattr(run.parent, "stats", lambda _: (STATS, {}))
    a, q = model()
    b, r = copy.deepcopy(a), copy.deepcopy(q)
    x = inputs().numpy()

    def opt(e, q):
        return torch.optim.AdamW(list(e.parameters()) + list(q.parameters()))

    meta = {"batch": 3, "micro": 3}
    job = {"seed": 42, "mode": "rolling"}
    run.train_epoch(meta, job, a, q, x, opt(a, q), 1, "cpu")
    meta["micro"] = 2
    run.train_epoch(meta, job, b, r, x, opt(b, r), 1, "cpu")
    for p, v in zip(
        list(a.parameters()) + list(q.parameters()),
        list(b.parameters()) + list(r.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(p.grad, v.grad, atol=2e-6, rtol=2e-4)


def test_bounded_worker_update_snapshot_resume_and_time_accounting(tmp_path, monkeypatch):
    meta = {
        "code_sha256": {},
        "seeds": [42],
        "base_epochs": 2,
        "prefix_cap": 6,
        "batch": 2,
        "micro": 1,
        "encoder_lr": 3e-5,
        "query_lr": 3e-4,
        "validation_prefixes": list(core.PREFIXES),
    }
    atomic_json(meta, tmp_path / "manifest.json")
    cache(tmp_path, "train", 2)
    cache(tmp_path, "val", 2)
    monkeypatch.setattr(run, "code_identity", lambda: {})
    monkeypatch.setattr(run.parent, "stats", lambda _: (STATS, {}))
    monkeypatch.setattr(run, "construct", lambda meta, seed, device: model(seed))
    monkeypatch.setattr(run, "validation", lambda *args: scores())
    train = run.train_epoch

    def training(meta, job, e, q, x, opt, epoch, device):
        result = train(meta, job, e, q, x, opt, epoch, "cpu")
        result["seconds"] = 1.0 if job["mode"] == "rolling" else 0.4
        return result

    monkeypatch.setattr(run, "train_epoch", training)
    jobs = run.jobs(meta)
    with pytest.raises(FileNotFoundError):
        run.target_time(meta, tmp_path, jobs[1])
    for job in jobs:
        run.worker(tmp_path, job)
    lock = run.lock_selection(meta, tmp_path)
    assert lock["time_controls"]["42"]["matched"]
    folder = tmp_path / "prefix_s42"
    last = torch.load(folder / "last.pt", weights_only=True)
    assert last["epoch"] == 5 and last["update_last"]["epoch"] == 2 and last["best_epoch"] == 0
    assert torch.load(folder / "update_last.pt", weights_only=True)["epoch"] == 2
    # Simulate interruption immediately after epoch2 journal commit but before snapshot writes.
    last["epoch"] = 2
    last["history"] = last["history"][:2]
    last["training_seconds"] = 0.8
    atomic_save(last, folder / "last.pt")
    (folder / "completion.json").unlink()
    (folder / "update_last.pt").unlink()
    (folder / "history.json").unlink()
    (tmp_path / "model_selection_lock.json").unlink()
    run.worker(tmp_path, jobs[1])
    run.lock_selection(meta, tmp_path)
    assert run.read_json(folder / "completion.json")["epochs"] == 5
    h = run.read_json(folder / "history.json")
    h[-1]["train"]["seconds"] = 100
    atomic_json(h, folder / "history.json")
    with pytest.raises(ValueError, match="SHA256"):
        run.check_selection(tmp_path)


def test_cache_rebinding_and_research_guard(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    atomic_json({}, source / "manifest.json")
    for split in data.SPLITS:
        cache(source, split)
    out = tmp_path / "run"
    out.mkdir()
    atomic_json({"new": True}, out / "manifest.json")
    identity = {
        "source": str(source),
        "files": {str(p.relative_to(source)): sha256(p) for p in (source / "cache").iterdir()},
    }
    meta = {"cache_source": identity}
    data.prepare(meta, out, ("train", "val"))
    x, _ = data.cache(out, "train")
    assert x.shape == (3, 255, 28)
    assert sha256(source / "cache/train.npz") == sha256(out / "cache/train.npz")
    with pytest.raises(FileNotFoundError):
        data.prepare(meta, out, ("test",))
    monkeypatch.setattr(ev, "check_readouts", lambda _: {})
    data.prepare(meta, out, ("test",))
    assert (out / "cache/test.npz").exists()
    (source / "cache/cross_research.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        data.prepare(meta, out, ("cross_research",))


def test_time_mismatch_or_native_damage_blocks_promotion():
    rows = [{"month": str(i % 12)} for i in range(120)]
    meta = {"seeds": [42]}
    states = {}
    utility = {}
    names = [
        "rolling_s42_update_best",
        "prefix_s42_update_best",
        "prefix_s42_time_best",
        "macro_s42",
    ]
    for name in names:
        value = 0.7 if name.startswith("rolling") else 1.0
        utility[name] = {k: np.ones(120) * value for k in ("utility", "direction", "volatility")}
        for view in ("full", "native"):
            for metric in core.METRICS:
                states[f"error/{name}/{view}/{metric}"] = np.full((120, 4), value)
    match = {"42": {"matched": True, "ratio": 1.0}}
    assert ev.gates(meta, states, utility, rows, match)["42"]["representation_upgrade"]
    match["42"]["matched"] = False
    assert not ev.gates(meta, states, utility, rows, match)["42"]["reconstruction_progress"]
    match["42"]["matched"] = True
    states["error/rolling_s42_update_best/native/far"][:] = 1.2
    assert not ev.gates(meta, states, utility, rows, match)["42"]["reconstruction_progress"]


def test_bound_source_code_untouched():
    current = run.code_identity()
    old = data.previous.code_identity()
    assert all(current[k] == v for k, v in old.items()) and len(current) == len(old) + 4


def test_export_failure_preserves_receipt(tmp_path):
    subprocess.run(["bash", "-n", "scripts/babel_context_transfer768_autodl.sh"], check=True)
    env = dict(
        os.environ,
        PYTHON_BIN="/usr/bin/false",
        BABEL_CONTEXT_TRANSFER_RUN=str(tmp_path / "absent"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    proc = subprocess.run(
        ["bash", "scripts/babel_context_transfer768_autodl.sh", "all"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and "run_status=failed" in proc.stdout
    with tarfile.open(tmp_path / "download/absent_reports.tar.gz") as tar:
        assert "command_exit_code=1" in tar.extractfile("absent/run_status.txt").read().decode()
    script = Path("scripts/babel_context_transfer768_autodl.sh").read_text()
    assert "-name 'readout_weights.npz'" in script


def test_completed_cache_source_requires_matching_macro_and_hashes(tmp_path, monkeypatch):
    macro = {"source": "macro", "files": {}}
    atomic_json(
        {"schema": data.previous.SCHEMA, "code_sha256": {}, "task_source": macro},
        tmp_path / "manifest.json",
    )
    monkeypatch.setattr(data.previous, "code_identity", lambda: {})
    for split in data.SPLITS:
        cache(tmp_path, split)
    for name in ("statistics.json", "model_selection_lock.json", "readout_lock.json"):
        atomic_json({}, tmp_path / name)
    files = {str(p.relative_to(tmp_path)): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    atomic_json(
        {"status": "complete", "source_unchanged": True, "files": files},
        tmp_path / "completion.json",
    )
    identity = data.source_identity(tmp_path, macro)
    assert not any(n.endswith(".pt") for n in identity["files"])
    with pytest.raises(ValueError, match="ancestry"):
        data.source_identity(tmp_path, {"source": "wrong"})
    (tmp_path / "cache/train.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        data.source_identity(tmp_path, macro)


def test_warm_loader_preserves_original_module_state_and_strict_snapshot(tmp_path, monkeypatch):
    meta = {}
    job = {"name": "prefix_s42", "mode": "prefix", "seed": 42}
    atomic_json(meta, tmp_path / "manifest.json")
    folder = tmp_path / job["name"]
    folder.mkdir()
    e, q = model()
    before = run.cpu_state(e)

    def construct(meta, seed, device):
        return model(seed)

    monkeypatch.setattr(run, "construct", construct)
    ck = {
        "encoder": before,
        "query": run.cpu_state(q),
        "epoch": 2,
        "metadata": run.binding(tmp_path, job),
    }
    atomic_save(ck, folder / "update_best.pt")
    loaded, reader = ev.load_model(meta, tmp_path, "prefix_s42_update_best", "cpu")
    for k, v in loaded.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert not any(v.requires_grad for v in loaded.parameters())
    ck["encoder"].pop(next(iter(ck["encoder"])))
    atomic_save(ck, folder / "update_best.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        ev.load_model(meta, tmp_path, "prefix_s42_update_best", "cpu")


def test_synthetic_full_native_evaluation_export_and_resume(tmp_path, monkeypatch):
    from obson.babel import utility_probe as up

    meta = {"seeds": [42, 43], "micro": 4}
    atomic_json(meta, tmp_path / "manifest.json")
    atomic_json(
        {
            "time_controls": {
                "42": {"matched": True, "ratio": 1.0},
                "43": {"matched": False, "ratio": 0.8},
            }
        },
        tmp_path / "model_selection_lock.json",
    )
    for split in ("test", "cross_research"):
        cache(tmp_path, split, 12)
    monkeypatch.setattr(run, "check_selection", lambda _: {})
    monkeypatch.setattr(ev, "check_readouts", lambda _: {})
    monkeypatch.setattr(ev.parent, "stats", lambda _: (STATS, {}))
    monkeypatch.setattr(ev, "load_model", lambda meta, out, name, device: model())
    weights = {"pca": np.zeros((3584, 768))}
    heads = {}
    for name in ["raw3584", "current28", "pca768"] + ev.variants(meta):
        width = {"raw3584": 3584, "current28": 28, "pca768": 768}.get(name, 32)
        heads[name] = [
            {"statistics": {"mean": [0.0] * width, "scale": [1.0] * width}} for _ in up.NAMES
        ]
        weights[name + "/weights"] = np.zeros((width, 13))
        weights[name + "/intercepts"] = np.zeros(13)
    atomic_json(
        {
            "heads": heads,
            "raw_stats": {"mean": [0.0] * 3584, "scale": [1.0] * 3584},
            "target_stats": {"mean": [0.0] * 13, "scale": [1.0] * 13},
        },
        tmp_path / "readout_fit.json",
    )
    np.savez(tmp_path / "readout_weights.npz", **weights)
    ev.evaluate(meta, tmp_path, "cpu")
    report = run.read_json(tmp_path / "context_transfer_metrics.json")
    assert len(report["test"]["reconstruction"]) == 14
    assert not report["test"]["gates"]["43"]["reconstruction_progress"]
    assert not run.read_json(tmp_path / "decision.json")["representation_upgrade"]
    with np.load(tmp_path / "test_predictions.npz") as values:
        assert values["error/rolling_s42_update_best/full/price"].shape == (12, 4)
        assert values["error/rolling_s42_update_best/native/price"].shape == (12, 4)
        assert values["targets"].shape == (12, 13)
    assert (tmp_path / "context_transfer.png").stat().st_size > 1000
    before = sha256(tmp_path / "cache/test_states.npz")
    ev.extract(meta, tmp_path, "test", "cpu")
    assert sha256(tmp_path / "cache/test_states.npz") == before


def test_resume_disk_reserve_credits_only_known_files(tmp_path):
    meta = {"seeds": [42, 43]}
    assert run.disk_reserve(meta, tmp_path) == 10 * 2**30
    folder = tmp_path / "prefix_s42"
    folder.mkdir()
    with (folder / "last.pt").open("wb") as f:
        f.truncate(3 * 2**30)
    with (tmp_path / "unrelated.pt").open("wb") as f:
        f.truncate(9 * 2**30)
    assert run.disk_reserve(meta, tmp_path) == 7 * 2**30
    (folder / "time_last.pt").symlink_to(tmp_path / "unrelated.pt")
    assert run.disk_reserve(meta, tmp_path) == 7 * 2**30
    assert run.disk_reserve(meta, tmp_path, trained=True) == 2 * 2**30


def test_fixed_update_snapshot_survives_later_best(tmp_path):
    job = {"mode": "prefix"}
    ck = {
        "metadata": {"job": job},
        "encoder": {"weight": torch.tensor([2.0])},
        "query": {"weight": torch.tensor([2.0])},
        "best_epoch": 2,
        "epoch": 2,
        "best_encoder": {"weight": torch.tensor([2.0])},
        "best_query": {"weight": torch.tensor([2.0])},
        "best_validation": {"price": 0.8},
        "history": [{"validation": {"price": 0.8}}],
    }
    ck["update_best"] = run.snapshot(ck, True)
    ck["update_last"] = run.snapshot(ck)
    run.materialize(ck, tmp_path)
    old_hash = sha256(tmp_path / "update_best.pt")
    ck.update(
        best_epoch=5,
        epoch=5,
        best_encoder={"weight": torch.tensor([5.0])},
        best_query={"weight": torch.tensor([5.0])},
        best_validation={"price": 0.7},
    )
    run.materialize(ck, tmp_path)
    assert sha256(tmp_path / "update_best.pt") == old_hash
    run.materialize(ck, tmp_path, refresh=True)
    fixed = torch.load(tmp_path / "update_best.pt", weights_only=True)
    later = torch.load(tmp_path / "time_best.pt", weights_only=True)
    assert fixed["epoch"] == 2 and later["epoch"] == 5
    assert fixed["encoder"]["weight"].item() == 2 and later["encoder"]["weight"].item() == 5
