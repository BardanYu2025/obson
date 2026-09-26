"""Synthetic frozen checks: same state/target, causal masks and complete source replay."""

import copy
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_capacity_growth import small

from obson.babel import decoder_parity as audit
from obson.babel.ae_extend import atomic_json


@pytest.fixture(autouse=True)
def no_optimizer_or_reader_fit():
    torch.set_num_threads(1)
    with (
        patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("No optimization")),
        patch.object(
            audit.upstream.ur.up, "fit_heads", side_effect=AssertionError("No reader fitting")
        ),
    ):
        yield


def fixture():
    model, d, stats, local = small()
    audit.upstream.rs.pf.install(model, stats, True)
    with torch.no_grad():
        model.core.encoder.backbone.input.projection.weight.fill_(0.025)
    model.eval().requires_grad_(False)
    return SimpleNamespace(aligned=model, statistics=stats, local=local), d


def test_shared_decoder_crop_uses_only_own_anchor_and_same_target():
    engine, d = fixture()
    b = {k: torch.tensor(v) for k, v in d.items()}
    for p in audit.PREFIXES:
        # A perfect full-path prediction must yield the exact original local target.
        actual = audit.shared_crop(b["y"], p, engine.statistics, engine.local)
        target, mask = audit.observed_targets(b, p, engine.statistics, engine.local)
        torch.testing.assert_close(actual[mask], target[mask], atol=0, rtol=0)
        changed = b["y"].clone()
        changed[:, p - 1 :] += 1e5
        changed[:, : p - 18] -= 1e5
        torch.testing.assert_close(
            audit.shared_crop(changed, p, engine.statistics, engine.local), actual, atol=0, rtol=0
        )
        shifted = b["y"].clone()
        shifted[..., 0] += 10
        torch.testing.assert_close(
            audit.shared_crop(shifted, p, engine.statistics, engine.local),
            actual,
            atol=5e-4,
            rtol=5e-4,
        )
    with pytest.raises(ValueError):
        audit.shared_crop(b["y"], 17, engine.statistics, engine.local)


def test_native_scores_exclude_current_and_future_and_decoder_is_causal():
    engine, d = fixture()
    b = {k: torch.tensor(v) for k, v in d.items()}
    model = engine.aligned
    with torch.inference_mode():
        full = model.core.encoder(b["x"])
        prediction = model.core.decoder(full[:, 79])
        x = b["x"].clone()
        x[:, 80:] += 10
        changed = model.core.decoder(model.core.encoder(x)[:, 79])
        torch.testing.assert_close(prediction, changed, atol=2e-6, rtol=2e-5)
        short = model.core.decoder(model.core.encoder(b["x"][:, :80])[:, -1])
        torch.testing.assert_close(prediction, short, atol=2e-6, rtol=2e-5)
    score = audit.native_rows(prediction, b, 80, engine.statistics)
    altered = {k: v.clone() for k, v in b.items()}
    altered["y"][:, 79:] = 1e5
    again = audit.native_rows(prediction, altered, 80, engine.statistics)
    for k in score:
        torch.testing.assert_close(score[k], again[k], atol=0, rtol=0)


def test_same_endpoint_short_context_has_identical_truth():
    engine, d = fixture()
    record = audit.evaluate_model(engine, d, 4, "cpu")
    assert len(record) == 11
    for p in audit.LENGTHS:
        assert record[f"context{p}"]["target_sha256"] == record["prefix128"]["target_sha256"]
        assert record[f"context{p}"]["mask_sha256"] == record["prefix128"]["mask_sha256"]
    assert record["prefix80"]["target_sha256"] != record["prefix128"]["target_sha256"]
    # Independently construct the perfect short-context path from cached physical prices.
    b = {k: torch.tensor(v) for k, v in d.items()}
    truth, mask = audit.observed_targets(b, 128, engine.statistics, engine.local)
    for length in audit.LENGTHS:
        short = torch.zeros_like(b["y"])
        short[:, :length] = b["y"][:, -length:]
        # The context's absolute price origin can shift; own-anchor rebasing removes it.
        short[:, :length, 0] -= 3
        p = audit.shared_crop(short, length, engine.statistics, engine.local)
        torch.testing.assert_close(p[mask], truth[mask], atol=5e-4, rtol=5e-4)


def test_complete_frozen_pipeline_replay_resume_and_tamper(tmp_path):
    engine, d = fixture()
    source, reference, training, out = [
        tmp_path / n for n in ("source", "reference", "training", "out")
    ]
    for p in (source, reference, training):
        p.mkdir()
    identity = {
        "manifest": {
            "source": str(reference),
            "identity": {"training_source": str(training), "training_manifest": {}},
        }
    }
    atomic_json(
        {"torch": str(torch.__version__), "numpy": np.__version__}, reference / "runtime.json"
    )
    rows = [
        {
            "key": "A/15/C",
            "symbol": "A",
            "period": 15,
            "row": 512 + i,
            "end": f"2026-01-{i + 1:02d} 10:00:00",
            "week": f"w{i}",
            "month": "2026-01",
        }
        for i in range(len(d["x"]))
    ]
    with torch.inference_mode():
        scores, errors, _, _ = audit.ur.pf.score(
            engine.aligned, d, engine.statistics, engine.local, 64, "cpu"
        )
    expected = {
        "reconstruction": {
            "scores": scores,
            "errors": {k: {n: v.tolist() for n, v in e.items()} for k, e in errors.items()},
        }
    }
    for split in audit.SPLITS:
        atomic_json(rows, training / f"{split}_inventory.json")
        for seed in (42, 43):
            atomic_json(expected, reference / f"{split}_control_s{seed}_best.json")
    with (
        patch.object(audit.upstream, "source_identity", return_value=identity),
        patch.object(audit.upstream.warm, "check_output"),
        patch.object(audit, "data", return_value=d),
        patch.object(audit, "load_engine", side_effect=lambda *a: copy.deepcopy(engine)),
    ):
        audit.run(source, out, "cpu")
        done = audit.read_json(out / "completion.json")
        assert (
            done["status"] == "complete"
            and done["encoder_updates"] == 0
            and done["reader_fits"] == 0
        )
        audit.upstream.verify_files(out, done["files"])
        for split in audit.SPLITS:
            for seed in (42, 43):
                assert audit.read_json(out / f"{split}_s{seed}_replay.json")["passed"]
        assert len(audit.read_json(out / "decision.json")["checks"]) == 140
        with patch.object(
            audit, "evaluate_model", side_effect=AssertionError("No repeated inference")
        ):
            audit.run(source, out, "cpu")
        (out / "test_s42.json").write_text("tampered")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            audit.run(source, out, "cpu")


def test_missing_cohort_and_failed_replay_cannot_pass(tmp_path):
    with pytest.raises(ValueError):
        audit.decide({}, {})
    engine, d = fixture()
    records = audit.evaluate_model(engine, d, 4, "cpu")
    expected = {
        "reconstruction": {
            "errors": {f"p{p}": records[f"prefix{p}"]["errors"]["local"] for p in audit.PREFIXES}
        }
    }
    expected["reconstruction"]["errors"]["recent"] = records["prefix128"]["errors"]["shared"]
    expected["reconstruction"]["errors"]["global"] = records["prefix128"]["native"]["errors"]
    expected = copy.deepcopy(expected)
    expected["reconstruction"]["errors"]["p80"]["primary"][0] += 1
    with pytest.raises(ValueError, match="source replay differs"):
        audit.replay(records, expected, tmp_path, "synthetic")
    assert not audit.read_json(tmp_path / "synthetic_replay.json")["passed"]


def test_decision_requires_every_position_seed_cohort_and_supported_pairs():
    rows = [
        {"week": f"w{i // 10}", "month": "2026-01", "symbol": "A", "period": 15} for i in range(60)
    ]
    entry = {
        "errors": {
            head: {k: [1.0] * 60 for k in ("primary", "path", "body", "activity", "change1")}
            for head in ("shared", "local")
        }
    }
    record = {
        f"{kind}{p}": copy.deepcopy(entry)
        for kind, positions in (("prefix", audit.PREFIXES), ("context", audit.LENGTHS))
        for p in positions
    }
    records = {f"{s}/s{seed}": copy.deepcopy(record) for s in audit.SPLITS for seed in (42, 43)}
    inventories = dict.fromkeys(audit.SPLITS, rows)
    good = audit.decide(records, inventories)
    assert good["status"] == "frozen_shared_readout_within_scope"
    assert len(good["checks"]) == 140 and len(good["context_comparisons"]) == 24
    records["cross_research/s43"]["prefix80"]["errors"]["shared"]["primary"] = [1.2] * 60
    bad = audit.decide(records, inventories)
    assert bad["status"] == "frozen_shared_readout_not_established"
    failed = [c for c in bad["checks"] if not c["passed"]]
    assert len(failed) == 1 and failed[0]["prefix"] == 80
    unsupported = {s: [dict(r, week="same_week") for r in rows] for s in audit.SPLITS}
    assert not any(c["passed"] for c in audit.decide(records, unsupported)["checks"])


def test_export_keeps_failed_status_and_omits_weights(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    atomic_json({"synthetic": True}, out / "manifest.json")
    (out / "secret.pt").write_bytes(b"synthetic weight omitted")
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(fake),
        BABEL_DECODER_PARITY_RUN=str(out),
        BABEL_DOWNLOAD_DIR=str(tmp_path / "download"),
    )
    script = Path(__file__).resolve().parents[2] / "scripts/babel_decoder_parity768_autodl.sh"
    result = subprocess.run(["bash", str(script), "all"], env=env, capture_output=True, text=True)
    assert result.returncode == 7 and "run_status=failed" in result.stdout
    result = subprocess.run(
        ["bash", str(script), "export"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0 and "run_status=failed" in result.stdout
    with tarfile.open(tmp_path / "download/run_reports.tar.gz") as archive:
        assert not any(n.endswith(".pt") for n in archive.getnames())
        assert b"run_status=failed" in archive.extractfile("run/run_status.txt").read()
