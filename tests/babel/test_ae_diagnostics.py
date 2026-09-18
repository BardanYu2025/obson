import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_babel import series
from obson.babel.ae_diagnostics import diagnose, gap_mask, group_metrics, metrics, summarize
from obson.babel.ar_codec import encode_frame
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import HistoryAE, HistoryWindows


class DiagnosticTests(unittest.TestCase):
    def test_identity_and_constant_support(self):
        y = np.zeros((128, 7))
        y[:, :2] = np.sin(np.arange(128)[:, None] / 3)
        y[:, 2:4] = .2
        result = metrics(y, y)
        self.assertEqual(result["close_mae_bps"], 0)
        self.assertAlmostEqual(result["change_1_std_ratio"], 1)
        self.assertAlmostEqual(result["change_4_correlation"], 1)
        constant = metrics(np.zeros_like(y), np.zeros_like(y))
        self.assertIsNone(constant["change_1_std_ratio"])
        summary = summarize({"all": [result, constant]})
        self.assertEqual(summary["all"]["metrics"]["change_1_std_ratio"]["supported_windows"], 1)

    def test_gap_and_position_assignment(self):
        y = np.zeros((128, 7))
        y[:, 2:4] = .1
        y[80:, :2] = 2
        self.assertEqual(np.flatnonzero(gap_mask(y)).tolist(), [80])
        pred = y.copy()
        pred[:32, 1] += .1
        groups = group_metrics(y.tolist(), pred.tolist(), "range")
        self.assertAlmostEqual(groups["position/Q1"]["close_mae_bps"], 10)
        self.assertEqual(groups["position/Q4"]["close_mae_bps"], 0)
        self.assertIn("window/with_large_gap", groups)

    def test_diagnosis_preserves_weights_and_artifacts(self):
        data = series(400)
        encoded = [encode_frame(s.frame, s.period) for s in data]
        bounds = split_boundaries(data)
        datasets = [HistoryWindows(data, encoded, bounds, name, window=32) for name in ("train", "val", "test")]
        model = HistoryAE(hidden=16, layers=1, latent=8, window=32, dropout=0).eval()
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "best.pt").write_bytes(b"synthetic fingerprint fixture")
            (path / "ae_metrics.json").write_text('{"preserve": true}')
            report = diagnose(model, datasets, "cpu", 8, 42, path)
            self.assertEqual(report["test_windows"], len(datasets[-1]))
            self.assertEqual((path / "ae_metrics.json").read_text(), '{"preserve": true}')
            self.assertTrue((path / "ae_diagnostics.md").exists())
        for key, value in model.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), weights[key].numpy())
