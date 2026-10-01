"""Synthetic protocol/lifecycle checks; local neural optimizer steps are disabled."""

import copy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from obson.babel import volatility_transfer as core
from obson.babel import volatility_transfer_run as run
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def no_training():
    torch.set_num_threads(1)
    with patch.object(torch.optim.AdamW, "step", return_value=None):
        yield


def rows(n=120):
    return [
        {
            "symbol": f"s{i % 2}",
            "period": (15, 30, 60)[i % 3],
            "key": f"s{i % 2}/{(15, 30, 60)[i % 3]}/c",
            "row": i + 127,
            "month": f"2024-{i % 12 + 1:02d}",
        }
        for i in range(n)
    ]


def test_future_exact_h_new_returns_and_no_historical_return():
    c = np.ones(230) * 100
    c[:127] = 10  # Huge historical jump must not enter the target.
    c[128] = 200
    c[129:] = 200
    y, why = core.future_target(c, 127, np.ones(len(c), bool))
    assert why is None
    np.testing.assert_allclose(y, np.log(np.log(2) / np.sqrt([16, 64])))
    assert (
        core.future_target(np.ones(230), 127, np.ones(230, bool))[0].tolist() == [np.log(1e-8)] * 2
    )


@pytest.mark.parametrize("bad", [0, 127, 128, 190, 191])
def test_partition_purges_every_input_and_target_bar(bad):
    same = np.ones(230, bool)
    same[bad] = False
    assert core.future_target(np.ones(230), 127, same)[1] == "input_or_target_crosses_partition"
    assert (
        core.future_target(np.ones(191), 127, np.ones(191, bool))[1]
        == "insufficient_history_or_future"
    )


def test_causal_baseline_changes_only_when_observed_price_changes():
    c = np.exp(np.linspace(4, 4.2, 230))
    p = core.past_features(c, 127)
    c[128:] *= np.linspace(1, 3, len(c) - 128)
    np.testing.assert_array_equal(p, core.past_features(c, 127))
    c[127] *= 1.1
    assert not np.array_equal(p, core.past_features(c, 127))


def test_raw_feature_prefix_has_no_future_dependency():
    t = np.arange(220)
    close = 100 + np.sin(t / 13)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=len(t), freq="15min"),
            "open": close,
            "close": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "volume": t + 1.0,
            "oi": t + 1000.0,
            "oi_available": True,
        }
    )
    a = run.encode(df, 15)
    np.testing.assert_array_equal(a[:128], run.encode(df.iloc[:128], 15))
    changed = df.copy()
    changed.loc[128:, ["open", "high", "low", "close"]] *= 2
    np.testing.assert_array_equal(a[:128], run.encode(changed, 15)[:128])


def test_few_label_ids_nested_shared_deterministic_stratified():
    r = rows()
    small = core.subset(r, 0.1, 42)
    full = core.subset(r, 1.0, 42)
    assert set(small) <= set(full) == set(range(len(r)))
    assert len(small) == 12
    assert {(r[i]["symbol"], r[i]["period"]) for i in small} == {
        (v["symbol"], v["period"]) for v in r
    }
    np.testing.assert_array_equal(small, core.subset(copy.deepcopy(r), 0.1, 42))
    assert not np.array_equal(small, core.subset(r, 0.1, 43))
    with pytest.raises(ValueError):
        core.subset(r, 0.2, 42)


def test_baselines_train_only_per_period_correct_residuals():
    y = np.arange(120 * 2).reshape(120, 2) / 100
    past = np.ones((120, 4)) * 2
    periods = [v["period"] for v in rows()]
    fit = core.baseline_fit(y, past, periods)
    a = core.baseline_predict(fit, past, periods, "constant")
    b = core.baseline_predict(fit, past, periods, "persistence")
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(core.baseline_predict(fit, past + 1, periods, "persistence"), b + 1)
    with pytest.raises(ValueError):
        core.baseline_fit(y[:1], past[:1], periods[:1])


@pytest.mark.parametrize("family", core.READERS)
def test_head_gradient_stops_at_frozen_cache_and_quantile_loss(family):
    encoder = torch.nn.Linear(3, 5).requires_grad_(False)
    z = encoder(torch.ones(8, 3)).detach()
    reader = core.Reader(5, family)
    loss = core.pinball(reader(z), torch.ones(8, 2)).mean()
    loss.backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.parameters())
    pred = run.prediction(reader, z, {"mean": [0, 1], "scale": [1, 2]})
    assert (np.diff(pred, axis=-1) >= 0).all()


def test_scores_uncertainty_and_primary_can_be_recomputed():
    y = np.zeros((120, 2))
    p = np.tile([-1.0, 0.0, 1.0], (120, 2, 1))
    m, e = core.score(p, y, [1, 2])
    np.testing.assert_allclose(e[:, 0], 0.2 / 3)
    assert m["horizons"]["16"]["coverage80"] == 1
    assert m["horizons"]["16"]["width80_log_rms"] == 2
    assert m["primary"] == float(e.mean())
    with pytest.raises(ValueError):
        core.score(p[:, :, ::-1], y, [1, 2])


def test_intervals_insufficient_groups_cannot_pass_and_all_gates_required():
    errors = {k: np.ones((120, 2)) for k in ("raw", "pca", "simple")}
    errors["embedding"] = np.ones((120, 2)) * 0.8
    result = core.comparisons(errors, rows(), "simple")
    assert result["passed"]
    bad = [dict(r, month="2024-01") for r in rows()]
    assert not core.comparisons(errors, bad, "simple")["passed"]
    errors["embedding"][:, 1] = 1.5
    assert not core.comparisons(errors, rows(), "simple")["passed"]


def lifecycle(tmp_path):
    meta = {
        "reader_epochs": 3,
        "reader_batch": 16,
        "reader_seeds": [42, 43],
        "fractions": [0.1, 1.0],
        "reader_lrs": [0.001, 0.0003],
        "pca_rank": 4,
        "transfer_protocol": {"scope": "synthetic only"},
    }
    atomic_json(meta, tmp_path / "manifest.json")
    (tmp_path / "cache").mkdir()
    rng = np.random.default_rng(12)
    for s in run.SPLITS:
        bank = {
            "x": rng.normal(size=(120, 4, 3)).astype("float32"),
            "simple": rng.normal(size=(120, 4)),
            "targets": rng.normal(size=(120, 2)),
            "period": core.period_context(rows()),
            "embedding_s42": rng.normal(size=(120, 4)),
            "embedding_s43": rng.normal(size=(120, 4)),
        }
        np.savez(tmp_path / f"cache/{s}.npz", **bank)
        atomic_json(rows(), tmp_path / f"cache/{s}_rows.json")
        atomic_json(
            {
                "manifest_sha256": run.sha256(tmp_path / "manifest.json"),
                "files": {
                    n: run.sha256(tmp_path / "cache" / n) for n in (f"{s}.npz", f"{s}_rows.json")
                },
            },
            tmp_path / f"cache/{s}_index.json",
        )
    return meta


def test_all64_fits_selection_lock_evaluation_and_tamper(tmp_path):
    meta = lifecycle(tmp_path)
    lock = run.fit_all(meta, tmp_path, "cpu")
    assert len(lock["candidates"]) == 64 and len(lock["chosen"]) == 8
    for summary in lock["candidates"].values():
        assert summary["epochs"] == 3
        assert summary["updates"] == 3 * int(np.ceil(summary["labeled_windows"] / 16))
    assert lock == run.fit_all(meta, tmp_path, "cpu")
    run.evaluate(meta, tmp_path, "cpu")
    report = run.read_json(tmp_path / "transfer_metrics.json")
    assert report["encoder_updates"] == 0 and set(report["results"]) == {"test", "cross_research"}
    for group, c in lock["chosen"].items():
        for rep, name in c.items():
            candidates = [
                v
                for v in lock["candidates"].values()
                if v["job"]["name"].startswith(rep + "_" + group + "_")
            ]
            assert lock["candidates"][name]["validation_pinball"] == min(
                v["validation_pinball"] for v in candidates
            )
    first = next(iter(lock["files"]))
    (tmp_path / first).write_bytes(b"changed")
    with pytest.raises(ValueError):
        run.check_selection(meta, tmp_path)


def test_reader_interruption_resumes_budget_and_epoch0_without_local_updates(tmp_path):
    meta = lifecycle(tmp_path)
    train, r = run.cache(tmp_path, "train")
    val, _ = run.cache(tmp_path, "val")
    fit, components = run.transformations(meta, tmp_path, train, "cpu")
    job = run.jobs(meta)[0]
    ids = core.subset(r, 0.1, 42)
    stats = core.scaler(train["targets"][ids])
    x = run.inputs(train, fit, components, "simple", 42)
    v = run.inputs(val, fit, components, "simple", 42)
    original = run.atomic_save

    def interrupt(value, path):
        original(value, path)
        if path.name == "resume.pt" and value["epoch"] == 1:
            raise RuntimeError("synthetic interruption")

    with (
        patch.object(run, "atomic_save", side_effect=interrupt),
        pytest.raises(RuntimeError, match="interruption"),
    ):
        run.fit_reader(
            meta, tmp_path, job, x, v, train["targets"], val["targets"], stats, ids, "cpu"
        )
    summary = run.fit_reader(
        meta, tmp_path, job, x, v, train["targets"], val["targets"], stats, ids, "cpu"
    )
    assert summary["epochs"] == 3 and summary["updates"] == 3 and summary["selected_epoch"] == 0
    hist = run.read_json(tmp_path / job["name"] / "history.json")
    assert [r["epoch"] for r in hist] == [1, 2, 3]
    changed = copy.deepcopy(job)
    changed["lr"] *= 2
    with pytest.raises(ValueError, match="source changed"):
        run.fit_reader(
            meta, tmp_path, changed, x, v, train["targets"], val["targets"], stats, ids, "cpu"
        )


def test_missing_selection_lock_blocks_research_before_loading_raw(tmp_path):
    with patch.object(run.parent.xr.xd, "load_raw") as load:
        with pytest.raises(FileNotFoundError):
            run.prepare({}, tmp_path, ("test",), False, "cpu")
        load.assert_not_called()


def test_input_representation_only_uses_causal_bank_not_labels(tmp_path):
    meta = lifecycle(tmp_path)
    train, _ = run.cache(tmp_path, "train")
    fit, components = run.transformations(meta, tmp_path, train, "cpu")
    for rep in core.REPRESENTATIONS:
        before = run.inputs(train, fit, components, rep, 42)
        train["targets"] += 10000
        np.testing.assert_array_equal(before, run.inputs(train, fit, components, rep, 42))
    assert run.inputs(train, fit, components, "pca", 42).shape[1] == 7


def test_prepare_raw_lineage_purge_cache_frozen_model_and_corruption(tmp_path):
    from types import SimpleNamespace

    n = 800
    close = 100 + np.sin(np.arange(n) / 13)
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": close,
            "close": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "volume": np.arange(n) + 1.0,
            "oi": np.arange(n) + 1000.0,
            "oi_available": True,
        }
    )
    key = "s/15/c"
    ends = np.arange(511, 800)
    inventory = [
        {
            "key": key,
            "symbol": "s",
            "period": 15,
            "row": int(e),
            "end": str(frame.datetime.iloc[e]),
            "month": str(frame.datetime.iloc[e].to_period("M")),
        }
        for e in ends
    ]
    s = SimpleNamespace(
        key=key, period=15, frame=frame, sessions=frame.datetime.to_numpy().astype("datetime64[D]")
    )
    bounds = {"train_until": "2024-12-31", "val_until": "2025-12-31", "test_until": "2026-12-31"}
    features = run.encode(frame, 15)
    d = {"x": np.stack([features[e - 127 : e + 1] for e in ends])}
    statistics = {"x_mean": [0.0] * 28, "x_scale": [1.0] * 28}
    atomic_json(inventory, tmp_path / "train_inventory.json")
    atomic_json({"synthetic": True}, tmp_path / "manifest.json")
    meta = {"extraction_batch": 16}
    model = torch.nn.Module()
    model.core = torch.nn.Module()
    model.core.encoder = torch.nn.Linear(28, 768)
    before = copy.deepcopy(model.state_dict())
    with (
        patch.object(run.parent, "tm", return_value={}),
        patch.object(run.parent.xr.xd, "load_raw", return_value=([s], bounds)),
        patch.object(run.parent, "root", return_value=tmp_path),
        patch.object(run.parent, "data", return_value=d),
        patch.object(run.parent, "stats", return_value=(statistics, {})),
        patch.object(
            run.parent, "construct", side_effect=lambda *args: (copy.deepcopy(model), None)
        ),
        patch.object(run.er, "trained_causality", return_value={"status": "passed"}),
    ):
        run.prepare(meta, tmp_path, ("train",), False, "cpu")
        bank, kept = run.cache(tmp_path, "train")
        assert len(kept) == 225 and bank["embedding_s42"].shape == (225, 768)
        assert kept[-1]["row"] == 735
        audit = run.read_json(tmp_path / "cache/train_audit.json")
        assert audit["counts"] == {"insufficient_history_or_future": 64}
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name])
        # Existing valid caches skip all encoder forwards; damaged caches fail closed.
        with patch.object(
            run.parent, "construct", side_effect=AssertionError("must not re-extract")
        ):
            run.prepare(meta, tmp_path, ("train",), False, "cpu")
        (tmp_path / "cache/train.npz").write_bytes(b"corrupt")
        with pytest.raises(ValueError):
            run.prepare(meta, tmp_path, ("train",), False, "cpu")


def test_selection_cannot_change_lr_epoch_or_skip_comparison(tmp_path):
    meta = lifecycle(tmp_path)
    run.fit_all(meta, tmp_path, "cpu")
    original = run.read_json(tmp_path / "selection_lock.json")
    mutated = copy.deepcopy(original)
    group = next(iter(mutated["chosen"]))
    mutated["chosen"][group].pop("raw")
    atomic_json(mutated, tmp_path / "selection_lock.json")
    with pytest.raises(ValueError, match="representation"):
        run.check_selection(meta, tmp_path)
    mutated = copy.deepcopy(original)
    rep = "embedding"
    winner = mutated["chosen"][group][rep]
    mutated["chosen"][group][rep] = winner[:-1] + ("1" if winner[-1] == "0" else "0")
    atomic_json(mutated, tmp_path / "selection_lock.json")
    with pytest.raises(ValueError, match="Learning rate"):
        run.check_selection(meta, tmp_path)


def test_raw_unit_validation_replay_comparable_between_label_budgets(tmp_path):
    meta = lifecycle(tmp_path)
    lock = run.fit_all(meta, tmp_path, "cpu")
    audit = run.read_json(tmp_path / "validation_predictions.json")
    truth = np.array(audit["targets"])
    for name, pred in audit["predictions"].items():
        actual = core.score(np.array(pred), truth, (1.0, 1.0))[0]["primary"]
        assert actual == pytest.approx(lock["candidates"][name]["validation_pinball"], abs=1e-12)
    labels = run.read_json(tmp_path / "label_fit_audit.json")
    for seed in (42, 43):
        for fraction in (0.1, 1.0):
            ids = labels["label_ids"][f"s{seed}_f{int(100 * fraction)}"]
            expected = core.scaler(np.array(labels["targets"])[ids])
            assert (
                expected == lock["baselines"][f"mlp_s{seed}_f{int(100 * fraction)}"]["target_stats"]
            )


def test_shell_export_success_and_failure_receipts(tmp_path):
    import os
    import subprocess
    import tarfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    folder = tmp_path / "output"
    folder.mkdir()
    download = tmp_path / "download"
    atomic_json({"status": "complete"}, folder / "completion.json")
    env = os.environ | {
        "BABEL_VOLATILITY_TRANSFER_RUN": str(folder),
        "BABEL_DOWNLOAD_DIR": str(download),
    }
    result = subprocess.run(
        ["bash", "scripts/babel_volatility_transfer768_autodl.sh", "export"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "run_status=complete" in result.stdout
    with tarfile.open(download / "output_reports.tar.gz") as archive:
        assert b"run_status=complete" in archive.extractfile("output/run_status.txt").read()
        assert archive.getmember("output/experiment_notes.md")
    (folder / "completion.json").unlink()
    shim = tmp_path / "fail"
    shim.write_text("#!/bin/sh\nexit 7\n")
    shim.chmod(0o700)
    result = subprocess.run(
        ["bash", "scripts/babel_volatility_transfer768_autodl.sh", "all"],
        cwd=repo,
        env=env | {"PYTHON_BIN": str(shim)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7 and "run_status=failed" in result.stdout
    with tarfile.open(download / "output_reports.tar.gz") as archive:
        assert b"command_exit_code=7" in archive.extractfile("output/run_status.txt").read()
