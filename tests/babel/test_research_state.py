"""Research packaging tests: synthetic weights only; no optimizer/fitting."""

import copy
import json
import os
import subprocess
import tarfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from test_babel import frame
from test_capacity_growth import small

from obson.babel import research_state as rs
from obson.babel import research_state_cli as cli
from obson.babel import research_state_delivery as delivery
from obson.babel import research_state_view as view
from obson.babel import window_state as ws
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256


@pytest.fixture(autouse=True)
def no_training():
    torch.set_num_threads(1)
    with (
        patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("No optimizer")),
        patch.object(rs.up, "fit_heads", side_effect=AssertionError("No fitting")),
    ):
        yield


def utility(width):
    return {
        "heads": [
            {"statistics": {"mean": [0.0] * width, "scale": [1.0] * width}} for _ in rs.up.NAMES
        ],
        "weights": np.random.default_rng(42).normal(size=(width, 13)).tolist(),
        "intercepts": [0.0] * 13,
        "target_stats": {"mean": [0.0] * 13, "scale": [1.0] * 13},
    }


def engine():
    model, _, stats, local = small()
    return rs.ResearchState(
        model,
        stats,
        local,
        utility(8),
        {"supported_periods": [15], "seed": 42},
        "A/15/contract",
        15,
    )


def prepare(root, stack, adapter=None):
    model, data, stats, local = small()
    if adapter is not None:
        rs.pf.install(model, stats, adapter)
        delivery.xr.growth.install(model, 4, 32, 42)
        with torch.no_grad():
            model.core.encoder.backbone.input.projection.weight.fill_(0.125)
    model.eval().requires_grad_(False)
    source, training, out = (root / p for p in ("source", "training", "out"))
    for p in (source, training, out):
        p.mkdir()
    u = utility(8)
    jobs = [{"name": f"control_s{s}", "seed": s} for s in (42, 43)]
    meta = {
        "source": str(source),
        "identity": {"training_source": str(training), "training_manifest": {"experiments": jobs}},
    }
    atomic_json(meta, out / "manifest.json")
    fit = {"target_stats": u["target_stats"], "heads": {}}
    heads = {}
    for job in jobs:
        n = job["name"] + "_best"
        fit["heads"][n] = {"targets": u["heads"]}
        heads[n + "_weights"] = u["weights"]
        heads[n + "_intercepts"] = u["intercepts"]
        (training / job["name"]).mkdir()
        (training / job["name"] / "best.pt").write_text("source identity sentinel")
    atomic_json(fit, training / "fit.json")
    np.savez(training / "heads.npz", **heads)
    rows = [
        {
            "key": "A/15/contract",
            "period": 15,
            "row": 512 + i,
            "end": f"2025-01-{i + 1:02} 00:00:00",
        }
        for i in range(len(data["x"]))
    ]
    for split in ("train", "val", "test", "cross_research"):
        atomic_json(rows, training / f"{split}_inventory.json")
    stack.enter_context(patch.object(delivery.xr, "statistics", return_value=(stats, local)))
    stack.enter_context(
        patch.object(delivery.xe, "load", side_effect=lambda *a: (copy.deepcopy(model), {}))
    )
    stack.enter_context(patch.object(delivery, "data_for", return_value=data))
    with torch.inference_mode():
        score, errors, _, pred = delivery.warm.ur.pf.score(model, data, stats, local, 64, "cpu")
        z = model.core.encoder(torch.tensor(data["x"]))[:, -1].numpy()
    up = rs.up.predict(
        u["heads"], np.asarray(u["weights"]), np.asarray(u["intercepts"]), z, u["target_stats"]
    )
    for split in ("test", "cross_research"):
        for job in jobs:
            atomic_json(
                {
                    "reconstruction": {
                        "scores": score,
                        "errors": {
                            k: {n: v.tolist() for n, v in e.items()} for k, e in errors.items()
                        },
                    },
                    "utility": {"predictions": up.tolist()},
                    "examples": [
                        {
                            "index": 0,
                            "prediction": pred[0].tolist(),
                            "target": data["y"][0].tolist(),
                            "mask": data["mask"][0].tolist(),
                        }
                    ],
                },
                source / f"{split}_{job['name']}_best.json",
            )
    return meta, out, model, data


@pytest.mark.parametrize("adapter", [None, False, True])
def test_bundle_keeps_original_functions_and_does_not_need_old_checkpoints(tmp_path, adapter):
    with ExitStack() as stack:
        meta, out, model, data = prepare(tmp_path, stack, adapter)
        bundle = delivery.build_bundle(meta, out)
        with pytest.raises(FileNotFoundError):
            rs.load_bundle(bundle, 42, "A/15/contract", 15)
        rs.certify(bundle)
        loaded = rs.load_bundle(bundle, 42, "A/15/contract", 15)
        assert delivery.warm.ur.bb.state_signature(model) == delivery.warm.ur.bb.state_signature(
            loaded.aligned
        )
        with torch.inference_mode():
            x = torch.tensor(data["x"])
            z = model.core.encoder(x)
            torch.testing.assert_close(loaded.model.encoder(x), z, atol=0, rtol=0)
        # Runtime loader is independent of every source-checkpoint construction helper.
        with patch.object(
            delivery.xe, "load", side_effect=AssertionError("No upstream dependency")
        ):
            assert rs.load_bundle(bundle, 43, "A/15/contract", 15).identity["seed"] == 43
        scores = delivery.cached_replay(meta, bundle, out, "cpu")
        assert len(scores) == 4
        assert all(v["reconstruction_replay"] and v["utility_replay"] for v in scores.values())
        model_file = bundle / "control_s42_best.pt"
        model_file.write_bytes(model_file.read_bytes() + b"tamper")
        with pytest.raises(ValueError):
            rs.load_bundle(bundle, 42, "A/15/contract", 15)


def test_new_outputs_use_same_vector_original_head_and_frozen_readouts():
    e = engine()
    bars = list(ws.frame_bars(ws.data.validate_frame(frame(514)), e.key, e.period))
    for b in bars[:511]:
        e.warm(b, as_of=ws.streaming.timestamp(b.datetime) + pd.Timedelta(minutes=15))
    assert e.current()["readouts"] is None
    out = e.push(
        bars[511], as_of=ws.streaming.timestamp(bars[511].datetime) + pd.Timedelta(minutes=15)
    )
    assert len(out["history"]["values"]) == 127 and len(out["original_local"]["values"]) == 16
    assert out["metadata"]["schema"] == rs.SCHEMA and not out["metadata"]["forecast"]
    z = torch.tensor([out["embedding"]])
    with torch.inference_mode():
        expected = (
            e.aligned.local_head(z).reshape(16, 7).numpy() * e.local["scale"] + e.local["mean"]
        )
    np.testing.assert_array_equal(out["original_local"]["values"], expected)
    actual = rs.up.predict(
        e.utility["heads"],
        np.asarray(e.utility["weights"]),
        np.asarray(e.utility["intercepts"]),
        z.numpy(),
        e.utility["target_stats"],
    )[0]
    np.testing.assert_array_equal(list(out["readouts"]["values"].values()), actual)
    other = engine()
    other.restore(json.loads(json.dumps(e.snapshot())))
    for b in bars[512:]:
        asof = ws.streaming.timestamp(b.datetime) + pd.Timedelta(minutes=15)
        assert e.push(b, as_of=asof) == other.push(b, as_of=asof)


def test_extra_head_failure_is_transactional():
    e = engine()
    bars = list(ws.frame_bars(ws.data.validate_frame(frame(512)), e.key, e.period))
    for b in bars[:511]:
        e.warm(b, as_of=ws.streaming.timestamp(b.datetime) + pd.Timedelta(minutes=15))
    before = e.snapshot()
    with (
        patch.object(e.aligned.local_head, "forward", side_effect=RuntimeError("bad head")),
        pytest.raises(RuntimeError),
    ):
        e.push(bars[-1], as_of=ws.streaming.timestamp(bars[-1].datetime) + pd.Timedelta(minutes=15))
    assert e.snapshot() == before


def test_cli_filters_future_before_values_and_keeps_snapshot(tmp_path):
    e = engine()
    df = frame(514)
    df.loc[513, "close"] = np.nan
    csv = tmp_path / "raw.csv"
    df.to_csv(csv, index=False)
    with patch.object(rs, "load_bundle", return_value=e):
        cli.run(
            tmp_path / "bundle",
            42,
            e.key,
            15,
            csv,
            str(df.datetime.iloc[513]),
            tmp_path / "result.jsonl",
            last_only=True,
            snapshot_out=tmp_path / "snapshot.json",
        )
    result = json.loads((tmp_path / "result.jsonl").read_text())
    assert result["metadata"]["observed_bars"] == 513 and result["readouts"] is not None


def test_replay_failure_has_diagnostics(tmp_path):
    with pytest.raises(ValueError, match="Pinned source replay"):
        delivery.require_replay({"v": [1.0]}, {"v": [2.0]}, "failure", tmp_path)
    assert not json.loads((tmp_path / "failure_replay.json").read_text())["passed"]


def test_offline_view_excludes_unscored_current_and_escapes_script(tmp_path):
    with ExitStack() as stack:
        meta, out, _, _ = prepare(tmp_path, stack)
        source = Path(meta["source"])
        training = Path(meta["identity"]["training_source"])
        atomic_json({"source": str(training)}, source / "manifest.json")
        stats, _ = delivery.xr.statistics({})
        atomic_json({"reconstruction": stats}, source / "evaluation_scales.json")
        atomic_json(
            {
                "status": "complete",
                "source_unchanged": True,
                "files": {p.name: sha256(p) for p in source.iterdir()},
            },
            source / "completion.json",
        )
        data = view.report_data(source)
        assert len(data["cases"]) == 4 and len(data["cases"][0]["pred"]) == 127
        data["cases"][0]["row"]["key"] = "</script><script>bad()</script>"
        with patch.object(view, "report_data", return_value=data):
            view.render_source(source, out / "view.html")
        html = (out / "view.html").read_text()
        assert "__REPORT_DATA__" not in html and "</script><script>bad()" not in html
        assert "fetch(" not in html and "前127根完整历史" in html


def test_export_reports_exclude_weights_and_auto_copy_view(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "bundle").mkdir()
    (run / "model_review.html").write_text("offline")
    (run / "bundle/weights.pt").write_bytes(b"weights")
    atomic_json({"status": "complete"}, run / "completion.json")
    atomic_json({"status": "passed"}, run / "bundle/validation.json")
    env = dict(
        os.environ, BABEL_CONTROL600_RUN=str(run), BABEL_DOWNLOAD_DIR=str(tmp_path / "download")
    )
    subprocess.run(
        ["bash", "scripts/babel_control600_autodl.sh", "export"],
        env=env,
        check=True,
        capture_output=True,
    )
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as archive:
        assert not any(n.endswith(".pt") for n in archive.getnames())
    assert (tmp_path / "download/run_review.html").read_text() == "offline"
    assert (tmp_path / "download/run_bundle.tar.gz").exists()


def test_raw_rolling_snapshot_and_original_cache_replay(tmp_path):
    e = engine()
    df = ws.data.validate_frame(frame(530))
    csv = tmp_path / "contract.csv"
    df.to_csv(csv, index=False)
    end = 524
    features = np.column_stack(
        (ws.ae_context.encode_context(df, 15, "ema8_32")["x"], ws.aa.activity_features(df, 15)[0])
    )
    cached = (
        (features[end - 127 : end + 1] - e.statistics["x_mean"]) / e.statistics["x_scale"]
    ).astype(np.float32)
    plan = [
        {
            "split": "test",
            "index": 0,
            "row": {"key": e.key, "period": 15, "row": end, "end": str(df.datetime.iloc[end])},
            "path": str(csv),
            "sha256": sha256(csv),
        }
    ]
    out = tmp_path / "out"
    out.mkdir()
    with (
        patch.object(rs, "load_bundle", side_effect=lambda *a, **kw: engine()),
        patch.object(delivery, "data_for", return_value={"x": cached[None]}),
    ):
        delivery.raw_replay({}, tmp_path / "bundle", out, plan, "cpu")
    report = json.loads((out / "raw_replay.json").read_text())
    assert len(report) == 2 and all(r["steps"] == 9 and r["snapshot_exact"] for r in report)
    assert all(c["passed"] for r in report for c in r["checks"])
    assert (out / "example_state_0_s42.json").exists()


def test_failed_delivery_revokes_certificate_and_completion(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "bundle").mkdir()
    (out / "bundle/validation.json").write_text("{}")
    (out / "completion.json").write_text("{}")
    identity = {"training_manifest": {}, "training_source": str(tmp_path / "training")}
    atomic_json({"torch": str(torch.__version__), "numpy": np.__version__}, source / "runtime.json")
    meta = {
        "schema": delivery.SCHEMA,
        "source": str(source.resolve()),
        "identity": identity,
        "code_sha256": delivery.code_identity(),
        "encoder_updates": 0,
        "head_updates": 0,
        "reader_fits": 0,
        "statistics_fits": 0,
    }
    atomic_json(meta, out / "manifest.json")
    with (
        patch.object(delivery.warm, "source_identity", return_value=identity),
        patch.object(delivery.warm, "check_output"),
        patch.object(delivery, "build_bundle", return_value=out / "bundle"),
        patch.object(delivery, "raw_plan", return_value=[]),
        patch.object(
            delivery, "cached_replay", side_effect=ValueError("deliberate replay mismatch")
        ),
        pytest.raises(ValueError, match="deliberate replay"),
    ):
        delivery.run(source, out, "cpu")
    assert not (out / "bundle/validation.json").exists() and not (out / "completion.json").exists()


def test_full_768_four_layer_wrapped_architecture_roundtrip():
    """Match the real topology with synthetic weights, including float64 buffers."""
    config = dict(rs.ba.pt.CONFIG, latent=768, attention_layers=4, attention_ff=3072, heads=8)
    pca = {
        "components": np.zeros((768, 128 * 7), np.float32),
        "mean": np.zeros(128 * 7, np.float32),
    }
    coordinates = {"mean": [0.0] * 768, "scale": [1.0] * 768}
    stats = {
        "x_mean": [0.123456789012345] * 28,
        "x_scale": [1.23456789012345] * 28,
        "y_scale": [0.987654321098765] * 7,
    }
    model = rs.ba.AlignedStudent(config, 42, pca, coordinates).eval().requires_grad_(False)
    rs.pf.install(model, stats, True)
    with torch.no_grad():
        model.core.encoder.backbone.input.projection.weight.fill_(0.03)
        model.core.decoder.basis[:, :127].fill_(0.002)
    ck = {
        "config": config,
        "seed": 42,
        "statistics": stats,
        "input_adapter": "causal_path_v1",
        "model": model.state_dict(),
    }
    restored = rs.restore_model(ck)
    assert isinstance(restored.core.encoder.backbone.input, rs.pf.PathInput)
    assert restored.core.encoder.backbone.input.mean.dtype == torch.float64
    assert restored.core.encoder.backbone.input.mean.tolist() == stats["x_mean"][:2]
    assert delivery.warm.ur.bb.state_signature(model) == delivery.warm.ur.bb.state_signature(
        restored
    )
    x = torch.randn(1, 128, 28, generator=torch.Generator().manual_seed(41)) * 0.02
    prefixes = torch.tensor([[32, 64, 96, 128]])
    with torch.inference_mode():
        torch.testing.assert_close(model.core.encoder(x), restored.core.encoder(x), atol=0, rtol=0)
        expected, expected_local = model(x, prefixes, True)
        actual, actual_local = restored(x, prefixes, True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_local, expected_local, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Unknown pinned"):
        rs.restore_model(dict(ck, input_adapter="unrecognized"))
    invalid = dict(ck, model=dict(ck["model"]))
    invalid["model"]["core.encoder.backbone.input.enabled"] = torch.tensor(2.0)
    with pytest.raises(ValueError, match="Invalid pinned"):
        rs.restore_model(invalid)
    incomplete = dict(ck, model=dict(ck["model"]))
    del incomplete["model"]["core.encoder.backbone.input.projection.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        rs.restore_model(incomplete)
