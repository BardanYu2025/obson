"""Synthetic forward/backward and lifecycle tests; no neural optimizer updates."""

import copy
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import architecture as ar
from obson.babel import history_query as hq
from obson.babel import uniform_context as core
from obson.babel import uniform_context_data as data
from obson.babel import uniform_context_evaluate as ev
from obson.babel import uniform_context_run as run
from obson.babel.ae_extend import atomic_json
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
    original = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield
    torch.backends.mha.set_fastpath_enabled(original)


def inputs(n=3):
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(n, 255, 28)).astype("float32") * 0.01
    raw[..., [11, 23, 24, 25, 26, 27]] = 1
    return torch.tensor(raw)


def test_paired_initialization_no_extra_trainable_parameters():
    models = [core.make_models(m, 42, "cpu", CONFIG) for m in core.MODES]
    for e, q in models[1:]:
        for a, b in zip(models[0][0].state_dict().values(), e.state_dict().values(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(models[0][1].state_dict().values(), q.state_dict().values(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert 1 + sum(w - 1 for w in core.WINDOWS) == 128
    with pytest.raises(ValueError):
        models[0][0](inputs())


def test_relative_chunk_equivalence_and_exact_receptive_field():
    e, _ = core.make_models("relative", 42, "cpu", CONFIG)
    e.checkpointing = False
    e.eval()
    x = inputs(1).requires_grad_(True)
    z = e(x)
    ps = torch.tensor([[32, 64, 96, 128]])
    rolling = e(core.rolling_windows(x, ps).flatten(0, 1))[:, -1]
    torch.testing.assert_close(z[0, ps[0] + 126], rolling, atol=2e-6, rtol=2e-5)
    # Exactly128 features, including current: earliest reachable index127 at endpoint254.
    z[0, -1].square().sum().backward()
    assert torch.count_nonzero(x.grad[0, :127]) == 0
    assert x.grad[0, 127].abs().sum() > 0
    assert ev.chunk_audit(e, x.detach())["passed"]


def test_relative_prefix_future_changes_and_checkpoint_backward():
    a, _ = core.make_models("relative", 9, "cpu", CONFIG)
    b = copy.deepcopy(a)
    b.checkpointing = False
    x = inputs(2)
    change = x.clone()
    change[:, 159:] += 10
    za = a(x)
    zb = a(change)
    torch.testing.assert_close(za[:, :159], zb[:, :159], atol=0, rtol=0)
    za[:, -1].square().sum().backward()
    b(x)[:, -1].square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


def test_full_context_same_physical_windows_and_native_prefix():
    x = inputs(2)
    ps = torch.tensor([[32, 128], [64, 96]])
    windows = core.rolling_windows(x, ps)
    for i in range(2):
        for j, p in enumerate(ps[i]):
            torch.testing.assert_close(windows[i, j], x[i, p - 1 : p + 127])
    e, _ = core.make_models("prefix", 4, "cpu", CONFIG)
    torch.testing.assert_close(
        core.read_states(e, x, ps, True), e(x[:, -128:])[torch.arange(2)[:, None], ps - 1]
    )
    torch.testing.assert_close(
        core.read_states(e, x, ps), e(windows.flatten(0, 1))[:, -1].reshape(2, 2, -1)
    )


def test_targets_match_original_decoder_and_masks_without_current_or_future():
    x = inputs(2)
    ps = torch.tensor([[32, 128], [64, 96]])
    for native in (True, False):
        y, mask = core.targets(x, ps, STATS, native)
        for i in range(2):
            for j, p in enumerate(ps[i]):
                raw = (
                    x[i : i + 1, -128:].numpy() if native else x[i : i + 1, p - 1 : p + 127].numpy()
                )
                yy, mm = ar.ordered_targets(raw)
                _, yy, mm = ar.normalize(raw, yy, mm, STATS)
                b = {"x": torch.tensor(raw), "y": torch.tensor(yy), "mask": torch.tensor(mm)}
                py, pm = hq.targets(b, torch.tensor([[p if native else 128]]), STATS)
                torch.testing.assert_close(y[i, j], py[0, 0], atol=2e-6, rtol=2e-5)
                assert torch.equal(mask[i, j], pm[0, 0])
    changed = x.clone()
    changed[:, 159:] = 10
    a, m = core.targets(x, torch.full((2, 1), 32), STATS)
    b, n = core.targets(changed, torch.full((2, 1), 32), STATS)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(m, n)
    assert a.shape == (2, 1, 127, 7)


def test_zero_sentinels_and_activity_missingness():
    stats = copy.deepcopy(STATS)
    stats["x_mean"][20] = 0.71
    stats["x_scale"][20] = 3.7
    stats["x_mean"][22] = 0.23
    stats["x_scale"][22] = 0.33
    raw = inputs(1).numpy()
    raw[:, :, 20] = 0
    raw[:, :, 22] = 0
    x = torch.tensor(((raw - stats["x_mean"]) / stats["x_scale"]).astype("float32"))
    _, mask = core.targets(x, torch.tensor([[128]]), stats)
    assert mask[..., 4:6].all()  # zero sentinels must not become suspect nonzero OI.
    x[:, 200, 27] = 0
    x[:, 201, 24] = 0
    _, mask = core.targets(x, torch.tensor([[128]]), stats)
    assert not mask[0, 0, 53, 3]
    assert not mask[0, 0, 52, 5]


def test_age_bands_dont_hide_far_harm():
    x = inputs()
    ps = torch.full((3, 1), 128)
    y, mask = core.targets(x, ps, STATS)
    pred = y.clone()
    pred[:, :, 64:, 0] += 1
    scores = core.band_rows(pred, y, mask, STATS)
    assert (scores["near"] == 0).all() and (scores["mid"] == 0).all()
    assert (scores["far"] > 0).all()


def test_strict_partition_burnin_and_same_schedule():
    same = np.ones(400, bool)
    assert data.eligible(254, same) and not data.eligible(253, same)
    same[0] = False
    assert not data.eligible(254, same) and data.eligible(255, same)
    a, p = run.schedule(21, 42, 1)
    b, q = run.schedule(21, 42, 1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(p, q)
    assert (p[:, -1] == 128).all() and not np.isin(p[:, :4], [48, 80, 112]).any()
    assert all(len(set(v)) == 5 for v in p)


def test_micro_accumulation_matches_full_gradient(monkeypatch):
    monkeypatch.setattr(run.parent, "stats", lambda _: (STATS, {}))
    x = inputs(3).numpy()
    job = {"mode": "rolling", "seed": 42}
    a, q = core.make_models("rolling", 42, "cpu", CONFIG)
    b, r = copy.deepcopy(a), copy.deepcopy(q)
    meta = {"batch": 3, "micro": 3}

    def opt(e, d):
        return torch.optim.AdamW(list(e.parameters()) + list(d.parameters()))

    run.train_epoch(meta, job, a, q, x, opt(a, q), 1, "cpu")
    meta["micro"] = 2
    run.train_epoch(meta, job, b, r, x, opt(b, r), 1, "cpu")
    for p, v in zip(
        list(a.parameters()) + list(q.parameters()),
        list(b.parameters()) + list(r.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(p.grad, v.grad, atol=2e-6, rtol=2e-4)


def synthetic_cache(out, split, x):
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    np.savez(folder / f"{split}.npz", x=x)
    atomic_json(
        [{"month": f"2024-{i % 12 + 1:02d}", "period": 15, "symbol": "X"} for i in range(len(x))],
        folder / f"{split}_rows.json",
    )
    files = {n: sha256(folder / n) for n in (f"{split}.npz", f"{split}_rows.json")}
    atomic_json(
        {"manifest_sha256": sha256(out / "manifest.json"), "files": files},
        folder / f"{split}_index.json",
    )


def test_worker_journal_epoch_zero_fallback_resume_and_lock(tmp_path, monkeypatch):
    meta = {
        "code_sha256": {},
        "encoder": CONFIG,
        "batch": 2,
        "micro": 1,
        "epochs": 2,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "validation_prefixes": [32, 64, 96, 128],
        "seeds": [42],
    }
    atomic_json(meta, tmp_path / "manifest.json")
    synthetic_cache(tmp_path, "train", inputs(2).numpy())
    synthetic_cache(tmp_path, "val", inputs(2).numpy())
    monkeypatch.setattr(run, "code_identity", lambda: {})
    monkeypatch.setattr(run.parent, "stats", lambda _: (STATS, {}))
    make = core.make_models
    train = run.train_epoch
    monkeypatch.setattr(
        core, "make_models", lambda mode, seed, device, config: make(mode, seed, "cpu", config)
    )
    monkeypatch.setattr(run, "validation", lambda *args: {"query": 1.0, "structure": 1.0})
    monkeypatch.setattr(
        run,
        "train_epoch",
        lambda meta, job, e, q, x, opt, epoch, device: train(meta, job, e, q, x, opt, epoch, "cpu"),
    )
    for job in run.jobs(meta):
        run.worker(tmp_path, job)
    lock = run.lock_selection(meta, tmp_path)
    assert len(lock["files"]) == 9
    for job in run.jobs(meta):
        folder = tmp_path / job["name"]
        ck = torch.load(folder / "last.pt", weights_only=True)
        assert ck["epoch"] == 2 and ck["best_epoch"] == 0 and len(ck["history"]) == 2
        # Interrupted after journal commit, before derived best/history materialization.
        (folder / "completion.json").unlink()
        (folder / "best.pt").unlink()
        run.worker(tmp_path, job)
        assert (folder / "best.pt").exists()
    # Re-materializing identical torch archives may change file hashes; lock detects it.
    (tmp_path / "model_selection_lock.json").unlink()
    run.lock_selection(meta, tmp_path)
    file = tmp_path / "prefix_s42/history.json"
    file.write_text("[]")
    with pytest.raises(ValueError, match="SHA256"):
        run.check_selection(tmp_path)


def test_readout_lock_and_research_access_are_guarded(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.prepare({}, tmp_path, ("test",))
    with pytest.raises(ValueError):
        data.prepare({}, tmp_path, ("train",), True)
    atomic_json({}, tmp_path / "manifest.json")
    synthetic_cache(tmp_path, "train", inputs(2).numpy())
    file = tmp_path / "cache/train.npz"
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        data.cache(tmp_path, "train")


def test_paired_months_and_gate_requirements():
    rows = [{"month": str(i % 12)} for i in range(120)]
    ci = ev.paired(np.ones((120, 4)), np.ones((120, 4)) * 2, rows)
    assert ci["high"] == -1 and ci["supported"]
    assert not ev.paired(np.ones(3), np.ones(3), rows[:3])["supported"]
    states = {}
    utility = {}
    for seed in (42, 43):
        for mode, value in [("relative", 0.8), ("rolling", 1.0)]:
            name = f"{mode}_s{seed}_best"
            utility[name] = {
                k: np.ones(120) * value for k in ("utility", "direction", "volatility")
            }
            for metric in ("price", "near", "mid", "far", "activity"):
                states[f"error/{name}/{metric}"] = np.ones((120, 4)) * value
    gates = ev.gates({"seeds": [42, 43]}, states, utility, rows)
    assert all(g["passed"] for g in gates.values())
    states["error/relative_s43_best/far"][:] = 1.5
    assert not ev.gates({"seeds": [42, 43]}, states, utility, rows)["43"]["passed"]


def test_old_bound_modules_unchanged_and_shell_export(tmp_path):
    subprocess.run(["bash", "-n", "scripts/babel_uniform_context768_autodl.sh"], check=True)
    text = Path("scripts/babel_uniform_context768_autodl.sh").read_text()
    assert "-name 'best.pt'" not in text and "-name 'last.pt'" not in text
    assert "BABEL_UNIFORM_CONTEXT_JOBS:-1" in text and "trap finish EXIT" in text
    from obson.babel import state_rollout_run as previous

    assert set(previous.parent.code_identity()).issubset(run.code_identity())


def test_cosine_phase_checkpoint_is_weights_only_loadable(tmp_path):
    value = core.learning_rate(11, 120, 3e-4)
    assert type(value) is float
    assert core.learning_rate(120, 120, 3e-4) == pytest.approx(3e-4 * 0.05)
    path = tmp_path / "checkpoint.pt"
    torch.save({"optimizer": {"lr": value}}, path)
    assert torch.load(path, weights_only=True)["optimizer"]["lr"] == value


def test_failed_run_shell_still_exports_receipt(tmp_path):
    import os
    import tarfile

    env = dict(
        os.environ,
        BABEL_UNIFORM_CONTEXT_RUN=str(tmp_path / "absent"),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
        PYTHON_BIN="/usr/bin/false",
    )
    p = subprocess.run(
        ["bash", "scripts/babel_uniform_context768_autodl.sh", "all"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 1 and "run_status=failed" in p.stdout
    with tarfile.open(tmp_path / "download/absent_reports.tar.gz") as archive:
        receipt = archive.extractfile("absent/run_status.txt").read().decode()
        assert "command_exit_code=1" in receipt and "run_status=failed" in receipt
        assert "absent/experiment_notes.md" in archive.getnames()


def test_synthetic_research_extraction_and_reporting(tmp_path, monkeypatch):
    from obson.babel import utility_probe as up

    meta = {"seeds": [42, 43], "micro": 4, "encoder": CONFIG}
    atomic_json(meta, tmp_path / "manifest.json")
    atomic_json({}, tmp_path / "model_selection_lock.json")
    for split in ("test", "cross_research"):
        synthetic_cache(tmp_path, split, inputs(12).numpy())
    monkeypatch.setattr(run, "check_selection", lambda _: {})
    monkeypatch.setattr(ev, "check_readouts", lambda _: {})
    monkeypatch.setattr(ev.parent, "stats", lambda _: (STATS, {}))

    def model(meta, out, name, device):
        mode = "relative" if name.startswith("relative") else "rolling"
        e, q = core.make_models(mode, 42, "cpu", CONFIG)
        return e.eval().requires_grad_(False), q.eval().requires_grad_(False)

    monkeypatch.setattr(ev, "load_model", model)
    names = ["raw3584", "current28", "pca768"] + ev.variants(meta)
    heads = {}
    weights = {"pca": np.zeros((3584, 768))}
    for name in names:
        width = {"raw3584": 3584, "current28": 28, "pca768": 768}.get(name, 32)
        heads[name] = [
            {"statistics": {"mean": [0.0] * width, "scale": [1.0] * width}} for _ in up.NAMES
        ]
        weights[name + "/weights"] = np.zeros((width, len(up.NAMES)))
        weights[name + "/intercepts"] = np.zeros(len(up.NAMES))
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
    report = run.read_json(tmp_path / "uniform_context_metrics.json")
    assert len(report["test"]["reconstruction"]) == 14 and len(report["test"]["utility"]) == 17
    assert not run.read_json(tmp_path / "decision.json")["research_direction_supported"]
    assert (tmp_path / "uniform_context.png").stat().st_size > 1000
    with np.load(tmp_path / "test_predictions.npz") as values:
        assert values["targets"].shape == (12, 13)
        assert values["error/relative_s42_best/price"].shape == (12, 4)
    # A resumed extraction verifies and reuses immutable outputs.
    before = sha256(tmp_path / "cache/test_states.npz")
    ev.extract(meta, tmp_path, "test", "cpu")
    assert sha256(tmp_path / "cache/test_states.npz") == before
