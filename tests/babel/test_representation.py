"""Synthetic forward/loss checks only; no optimizer steps or real training."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from test_babel import series
from obson.babel.data import split_boundaries
from obson.babel.model import Config
from obson.babel.representation import (
    BarEncoder, BarWindows, HORIZONS, constant_quantiles, forecast_eval,
    losses, ridge_probe, time_mask,
)


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.series = series(400)
        self.bounds = split_boundaries(self.series)
        self.cfg = Config(hidden=16, layers=1, window=32, warmup=8, dropout=0)

    def test_future_targets_never_cross_split_or_contract(self):
        for split in ("train", "val", "test"):
            ds = BarWindows(self.series, self.bounds, split, self.cfg, stride=1)
            for i, row in ds.items:
                s = self.series[i]
                self.assertLess(row + max(HORIZONS), len(s.x))
                self.assertTrue(time_mask(s, self.bounds, split)[row + np.array(HORIZONS)].all())
            batch = ds[len(ds) - 1]
            i, end = ds.items[-1]
            np.testing.assert_array_equal(batch["future"][-1], self.series[i].x[end + np.array(HORIZONS), :7])

    def test_causal_embeddings_and_ordered_quantiles(self):
        model = BarEncoder(self.cfg).eval()
        x = torch.randn(2, 32, 12)
        y = x.clone()
        y[:, 20:] += 50
        with torch.no_grad():
            a, b = model(x), model(y)
        torch.testing.assert_close(a["h"][:, :20], b["h"][:, :20])
        torch.testing.assert_close(a["quantiles"][:, :20], b["quantiles"][:, :20])
        self.assertTrue((a["quantiles"][..., 0] <= a["quantiles"][..., 1]).all())
        self.assertTrue((a["quantiles"][..., 1] <= a["quantiles"][..., 2]).all())

    def test_loss_and_absent_oi_mask(self):
        for s in self.series:
            s.x[:, 7] = 0
        ds = BarWindows(self.series, self.bounds, "train", self.cfg)
        batch = next(iter(DataLoader(ds, batch_size=2)))
        model = BarEncoder(self.cfg).eval()
        output = model(batch["x"])
        before = losses(output, batch, "cpu")
        batch["future"][..., 5:7] += 1000
        after = losses(output, batch, "cpu")
        torch.testing.assert_close(before[0], after[0])
        self.assertTrue(all(torch.isfinite(v) for v in before))
        (before[0] + .2 * before[1] + .1 * before[2]).backward()
        self.assertGreater(model.backbone.input.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.forecast.weight.grad.abs().sum().item(), 0)
        baseline = constant_quantiles(ds)
        report = forecast_eval(model, ds, "cpu", 8, baseline)
        self.assertTrue(np.isfinite(report["pinball"]))
        self.assertEqual(np.array(report["support"])[:, 5:7].sum(), 0)

    def test_fixed_probe(self):
        x = np.eye(3).repeat(20, axis=0)
        y = np.arange(3).repeat(20)
        result = ridge_probe(x, y, x, y)
        self.assertAlmostEqual(result["ba"], 1.)
