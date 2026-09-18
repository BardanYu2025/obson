"""Synthetic forward/backward and report export checks; no optimizer steps."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_babel import series
from obson.babel.ar_codec import encode_frame
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import (
    HistoryAE, HistoryWindows, evaluate, reconstruction_loss, to_ohlc, write_review,
)
from obson.babel.representation import time_mask


class HistoryAETests(unittest.TestCase):
    def test_single_bottleneck_and_causal_prefix(self):
        model = HistoryAE(hidden=16, layers=1, latent=8, window=16, dropout=0).eval()
        x = torch.randn(2, 16, 14)
        changed = x.clone()
        changed[:, 10:] += 10
        a, b = model(x), model(changed)
        torch.testing.assert_close(a["h"][:, :10], b["h"][:, :10])
        torch.testing.assert_close(a["h"][:, :10], model.encode(x[:, :10]))
        self.assertEqual(a["z"].shape, (2, 8))
        self.assertEqual(a["reconstruction"].shape, (2, 16, 7))
        torch.testing.assert_close(model.decode(a["z"]), a["reconstruction"])
        with self.assertRaises(ValueError):
            model.decode(a["h"])
        y = torch.randn(2, 16, 7)
        y[..., 2:] = y[..., 2:].abs()
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[..., 5] = False
        before = reconstruction_loss(a["reconstruction"], y, mask)
        changed_y = y.clone()
        changed_y[..., 5] += 100
        torch.testing.assert_close(before, reconstruction_loss(a["reconstruction"], changed_y, mask))
        before.mean().backward()
        self.assertGreater(model.input.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.expand.weight.grad.abs().sum().item(), 0)

    def test_reconstruction_targets_roundtrip_and_full_split(self):
        data = series(400)
        encoded = [encode_frame(s.frame, s.period) for s in data]
        bounds = split_boundaries(data)
        for split in ("train", "val", "test"):
            ds = HistoryWindows(data, encoded, bounds, split, window=16)
            for i, row in ds.items:
                self.assertTrue(time_mask(data[i], bounds, split)[row - 15:row + 1].all())
            sample = ds[0]
            i, end = ds.items[0]
            truth = data[i].frame[["open", "high", "low", "close"]].iloc[end - 15:end + 1].to_numpy()
            np.testing.assert_allclose(to_ohlc(sample["y"], sample["anchor"]), truth, rtol=1e-6)
            original_x, original_y = sample["x"].copy(), sample["y"].copy()
            # Re-encoding altered FUTURE prices cannot change the historical sample.
            f = data[i].frame.copy()
            f.loc[end + 1:, ["open", "high", "low", "close"]] *= 3
            future_encoded = encoded.copy()
            future_encoded[i] = encode_frame(f, data[i].period)
            np.testing.assert_array_equal(original_x, future_encoded[i]["x"][end - 15:end + 1])
            np.testing.assert_array_equal(original_y, ds[0]["y"])

    def test_reconstruction_baselines_and_review(self):
        data = series(400)
        encoded = [encode_frame(s.frame, s.period) for s in data]
        bounds = split_boundaries(data)
        datasets = [HistoryWindows(data, encoded, bounds, split, window=16) for split in ("train", "val", "test")]
        model = HistoryAE(hidden=16, layers=1, latent=8, window=16, dropout=0).eval()
        report, examples = evaluate(model, datasets, "cpu", 8, 42)
        self.assertEqual(report["pca_rank"], 8)
        self.assertTrue(np.isfinite(report["reconstruction"]["autoencoder"]["loss"]))
        self.assertEqual(len(examples), min(6, len(datasets[-1])))
        json.dumps(report, allow_nan=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.html"
            write_review(path, examples)
            content = path.read_text()
            self.assertIn("<svg", content)
            self.assertNotIn("<script", content)

    def test_export_to_download_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "checkpoints" / "fixture"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text('{"synthetic":true}')
            (run / "ae_metrics.json").write_text('{"test":true}')
            (run / "best.pt").write_text('must not export weights')
            log = root / "training.log"
            log.write_text("synthetic log\n")
            env = dict(os.environ, BABEL_AE_RUN=str(run), BABEL_DOWNLOAD_DIR=str(root / "download"), BABEL_AE_LOG=str(log))
            subprocess.run(["bash", "scripts/babel_ae_autodl.sh", "export"], env=env, check=True, capture_output=True)
            destination = root / "download" / "fixture"
            self.assertEqual((destination / "training.log").read_text(), log.read_text())
            self.assertEqual((destination / "ae_metrics.json").read_text(), '{"test":true}')
            self.assertFalse((destination / "best.pt").exists())
            self.assertTrue((destination / "experiment_notes.md").exists())
