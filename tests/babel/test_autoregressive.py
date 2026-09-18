"""Synthetic checks, no optimizer steps or actual training."""

import copy
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from test_babel import frame, series
from obson.babel.ar_codec import State, consume, decode, encode_frame, geometry
from obson.babel.ar_model import ARModel, nll, sample
from obson.babel.autoregressive import Windows, evaluate_one, extract, fit_baselines, probe, rollout
from obson.babel.data import split_boundaries
from obson.babel.representation import time_mask


class ARTests(unittest.TestCase):
    def test_codec_prefix_and_scale_invariance(self):
        f = frame(120)
        a = encode_frame(f, 60)
        b = encode_frame(f.iloc[:80], 60)
        for key in a:
            np.testing.assert_allclose(a[key][:80], b[key])
        changed = f.copy()
        changed.loc[80:, ["open", "high", "low", "close"]] *= 3
        c = encode_frame(changed, 60)
        np.testing.assert_array_equal(a["x"][:80], c["x"][:80])
        scaled = f.copy()
        scaled[["open", "high", "low", "close"]] *= 100
        d = encode_frame(scaled, 60)
        np.testing.assert_allclose(a["x"], d["x"], atol=2e-6)
        state = State(float(f.open.iloc[0]))
        elapsed = f.datetime.diff().dt.total_seconds().fillna(3600).to_numpy() / 3600
        for j, row in enumerate(f.itertuples()):
            x, y, _ = consume(state, [row.open, row.high, row.low, row.close], row.volume, row.oi, row.oi_available, 60, elapsed[j])
            np.testing.assert_allclose(x, a["x"][j], atol=1e-6)
            np.testing.assert_allclose(y, a["y"][j], atol=1e-6)
            self.assertAlmostEqual(state.variance, a["variance"][j], places=12)

    def test_encode_decode_roundtrip_including_gap(self):
        state = State(100.)
        bar = [115., 120., 109., 112.]
        before = copy.copy(state)
        _, y, _ = consume(state, bar, 1234., 4567., True, 15, 25.)
        decoded, volume, oi, elapsed = decode(before, y)
        np.testing.assert_allclose(decoded, bar, rtol=1e-6)
        np.testing.assert_allclose([volume, oi, elapsed], [1234, 4567, 25], rtol=1e-6)
        new = copy.copy(before)
        consume(new, decoded, volume, oi, True, 15, elapsed)
        self.assertAlmostEqual(new.sigma, state.sigma, places=7)
        with self.assertRaises(ValueError):
            geometry([0, 2, 0, 1], 1)
        with self.assertRaises(ValueError):
            decode(before, np.array([1000, 0, 1, 1, 1, 1, 0]))

    def test_causal_embeddings_likelihood_and_sampling(self):
        torch.manual_seed(42)
        model = ARModel(hidden=16, layers=1, dropout=0).eval()
        x = torch.randn(2, 32, 14)
        changed = x.clone()
        changed[:, 20:] += 10
        out, other = model(x), model(changed)
        torch.testing.assert_close(out["h"][:, :20], other["h"][:, :20])
        torch.testing.assert_close(out["mean"][:, :20], other["mean"][:, :20])
        torch.testing.assert_close(out["seq"], out["h"][:, -1])
        y = torch.randn(2, 32, 7)
        y[..., 2:] = y[..., 2:].abs()
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[..., 5] = False
        loss = nll(out, y, mask)
        y[..., 5] += 20
        torch.testing.assert_close(loss, nll(out, y, mask))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(model.input.weight.grad).all())
        self.assertGreater(model.input.weight.grad.abs().sum().item(), 0)
        draws = sample(out, torch.Generator().manual_seed(3))
        self.assertTrue((draws[..., 2:] >= 0).all())

    def test_split_masks_evaluation_rollout_no_future_feedback(self):
        data = series(400)
        encoded = [encode_frame(s.frame, s.period) for s in data]
        bounds = split_boundaries(data)
        for split in ("train", "val", "test"):
            ds = Windows(data, encoded, bounds, split, horizon=16, window=32)
            for i, row in ds.items:
                self.assertTrue(time_mask(data[i], bounds, split)[row:row + 17].all())
                b = ds[0]
                si, sr = ds.items[0]
                np.testing.assert_array_equal(b["y"][-1], encoded[si]["y"][sr + 1])
        tr = Windows(data, encoded, bounds, "train", window=32)
        te = Windows(data, encoded, bounds, "test", horizon=16, window=32)
        model = ARModel(hidden=16, layers=1, dropout=0).eval()
        # Stable synthetic predictions for testing the rollout mechanics.
        with torch.no_grad():
            model.distribution.weight.zero_()
            model.distribution.bias.zero_()
            bias = model.distribution.bias.view(5, 15)
            bias[:, 8:] = -3
        report = evaluate_one(model, te, "cpu", 8, fit_baselines(tr))
        self.assertTrue(np.isfinite(report["mean_nll_per_observed_coordinate"]["model"]))
        extracted, labels = extract(model, te, "cpu", 8)
        self.assertEqual(extracted["embedding"].shape, (len(te), 16))
        self.assertEqual(extracted["raw"].shape, (len(te), 16 * 14))
        self.assertEqual(len(labels["current_structure"]), len(te))
        # One chosen anchor; perturb only its future raw prices and future encodings.
        te.items = te.items[:1]
        first = rollout(model, te, "cpu", count=1, paths=2)
        i, end = te.items[0]
        data[i].frame.loc[end + 1:, ["open", "high", "low", "close"]] *= 2
        encoded[i]["x"][end + 1:] += 20
        second = rollout(model, te, "cpu", count=1, paths=2)
        self.assertEqual(first["failed_paths"], 0)
        self.assertEqual(second["failed_paths"], 0)
        self.assertEqual(first["cases"][0]["trace"]["generated_ohlc"],
                         second["cases"][0]["trace"]["generated_ohlc"])
        for h in ("1", "4", "16"):
            self.assertEqual(first["cases"][0]["horizons"][h]["sample_std_log_return"],
                             second["cases"][0]["horizons"][h]["sample_std_log_return"])
        # Classification regularization selected using validation labels only.
        x = np.eye(3).repeat(10, 0)
        y = np.arange(3).repeat(10)
        p = probe(x, y, x, y, x, y, 3)
        self.assertEqual(p["test"]["ba"], 1.)
