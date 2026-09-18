"""Stage-2 causality and controlled-comparison checks; no optimizer steps."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_babel import series
from obson.babel.ae_context import encode_context
from obson.babel.ar_codec import encode_frame
from obson.babel.data import manifest, split_boundaries
from obson.babel.history_autoencoder import (
    SCHEMA, HistoryAE, HistoryWindows, evaluate, frozen_features, reconstruction_loss, train,
)
from obson.babel.ae_diagnostics import diagnose


class AEContextTests(unittest.TestCase):
    def test_causal_price_scale_invariant_context(self):
        s = series(400)[0]
        base = encode_frame(s.frame, s.period)
        plain = encode_context(s.frame, s.period)
        for key in base:
            np.testing.assert_array_equal(base[key], plain[key])
        full = encode_context(s.frame, s.period, "ema8_32")
        np.testing.assert_array_equal(full["x"][:, :14], base["x"])
        prefix = encode_context(s.frame.iloc[:150], s.period, "ema8_32")
        np.testing.assert_array_equal(full["x"][:150], prefix["x"])
        changed = s.frame.copy()
        prices = ["open", "high", "low", "close"]
        changed.loc[150:, prices] *= 3
        altered = encode_context(changed, s.period, "ema8_32")
        np.testing.assert_array_equal(full["x"][:150], altered["x"][:150])
        changed = s.frame.copy()
        changed[prices] *= 7
        scaled = encode_context(changed, s.period, "ema8_32")
        np.testing.assert_allclose(full["x"][:, 14:], scaled["x"][:, 14:], atol=1e-6)

    def test_identical_initialization_adapter_gradient_and_legacy_config(self):
        kwargs = dict(hidden=16, layers=1, latent=8, window=16, dropout=0)
        torch.manual_seed(42)
        base = HistoryAE(**kwargs).eval()
        base_rng = torch.get_rng_state()
        torch.manual_seed(42)
        context = HistoryAE(**kwargs, context="ema8_32").eval()
        torch.testing.assert_close(torch.get_rng_state(), base_rng)
        for key, weight in base.state_dict().items():
            torch.testing.assert_close(weight, context.state_dict()[key], rtol=0, atol=0)
        legacy = HistoryAE(**kwargs)
        legacy.load_state_dict(base.state_dict(), strict=True)
        x = torch.randn(2, 16, 18)
        out = context(x)
        torch.testing.assert_close(out["reconstruction"], base(x[..., :14])["reconstruction"])
        # Nonzero adapter exercises actual causal attention, not just its zero initialization.
        with torch.no_grad():
            context.context_input.weight.fill_(.02)
        changed = x.clone()
        changed[:, 10:] += 4
        torch.testing.assert_close(context.encode(x)[:, :10], context.encode(changed)[:, :10])
        torch.testing.assert_close(context.encode(x)[:, :10], context.encode(x[:, :10]))
        y = torch.randn_like(out["reconstruction"])
        y[..., 2:] = y[..., 2:].abs()
        loss = reconstruction_loss(context(x)["reconstruction"], y, torch.ones_like(y, dtype=torch.bool)).mean()
        loss.backward()
        grad = context.context_input.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_identical_windows_and_targets(self):
        data = series(400)
        bounds = split_boundaries(data)
        base = [encode_context(s.frame, s.period) for s in data]
        extra = [encode_context(s.frame, s.period, "ema8_32") for s in data]
        for split in ("train", "val", "test"):
            a = HistoryWindows(data, base, bounds, split, window=16)
            b = HistoryWindows(data, extra, bounds, split, window=16)
            self.assertEqual(a.items, b.items)
            for i in range(len(a)):
                for key in ("y", "mask", "anchor", "series", "row"):
                    np.testing.assert_array_equal(a[i][key], b[i][key])

    def test_missing_baseline_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "baseline-run"):
                train([], [], {}, Path(tmp), 30, 32, 42, "cpu", context="ema8_32")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_baseline_budget_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = series(400)
            bounds = split_boundaries(data)
            baseline = dict(schema=SCHEMA, config=HistoryAE().config,
                            manifest=manifest(data, bounds), epochs=30, batch_size=32, seed=42, stride=16)
            (root / "manifest.json").write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "must match baseline"):
                train(data, [], bounds, root / "new", 31, 32, 42, "cpu", "ema8_32", root)
            self.assertFalse((root / "new").exists())

    def test_context_evaluation_diagnosis_and_raw_comparator(self):
        data = series(400)
        bounds = split_boundaries(data)
        encoded = [encode_context(s.frame, s.period, "ema8_32") for s in data]
        datasets = [HistoryWindows(data, encoded, bounds, name, window=32) for name in ("train", "val", "test")]
        model = HistoryAE(hidden=16, layers=1, latent=8, window=32, dropout=0, context="ema8_32").eval()
        report, _ = evaluate(model, datasets, "cpu", 8, 42)
        json.dumps(report, allow_nan=False)
        _, raw, _ = frozen_features(model, datasets[-1], "cpu", 8)
        self.assertEqual(raw.shape[1], 16 * 14)
        np.testing.assert_array_equal(raw[0], datasets[-1][0]["x"][-16:, :14].flatten())
        with tempfile.TemporaryDirectory() as tmp:
            result = diagnose(model, datasets, "cpu", 8, 42, Path(tmp))
            self.assertEqual(result["test_windows"], len(datasets[-1]))

    def test_stage2_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "ema_run"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({"synthetic": True}))
            env = dict(os.environ, BABEL_AE_RUN=str(run), BABEL_DOWNLOAD_DIR=str(Path(tmp) / "download"))
            subprocess.run(["bash", "scripts/babel_ae_stage2_autodl.sh", "export"], env=env, check=True, capture_output=True)
            target = Path(tmp) / "download" / "ema_run"
            self.assertTrue((target / "stage2_notes.md").exists())
            self.assertIn("exit_code=0", (target / "run_status.txt").read_text())
