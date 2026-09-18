"""Recovery and validation checks, without optimizer steps or local training."""
import json
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_babel import series
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import atomic_save, check_source, continuation_state, extend, publish, restore_rng, rng_state, validation_report
from obson.babel.history_autoencoder import HistoryAE, HistoryWindows, SCHEMA, WEIGHTS, validate
from obson.babel.data import manifest, split_boundaries


class AEExtendTests(unittest.TestCase):
    def test_continuation_preserves_full_state_and_source(self):
        state = dict(resume_schema="babel-ae-resume-v1", epoch=100,
                     metadata={"epochs":100, "source_checkpoint_epoch":29},
                     model={"w":torch.tensor([1.])}, optimizer={"step":1000},
                     rng=rng_state(), history=[{"epoch":100}], best_epoch=98,
                     best_model={"w":torch.tensor([2.])}, best_loss=.05)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root / "last.pt"
            atomic_save(state, source)
            before=source.read_bytes()
            result=continuation_state(source, root / "new", 200)
            self.assertEqual(result["metadata"]["epochs"],200)
            self.assertFalse(result["metadata"]["continuations"][0]["optimizer_reset"])
            for k in ("epoch", "optimizer", "history", "best_epoch", "best_loss"):
                self.assertEqual(result[k],state[k])
            for k in ("model", "best_model"):
                torch.testing.assert_close(result[k]["w"],state[k]["w"])
            torch.testing.assert_close(result["rng"]["torch"],state["rng"]["torch"])
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,"empty"):
                continuation_state(source,root,200)
            with self.assertRaisesRegex(ValueError,"larger"):
                continuation_state(source,root / "new",100)
            state["epoch"]=99
            atomic_save(state,source)
            with self.assertRaisesRegex(ValueError,"completed"):
                continuation_state(source,root / "new",200)
            atomic_save({"model":{}},source)
            with self.assertRaisesRegex(ValueError,"full last.pt"):
                continuation_state(source,root / "new",200)

    def test_checkpoint_roundtrip_rng_and_optimizer_state(self):
        model = HistoryAE(hidden=16, layers=1, latent=8, window=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        # Synthetic optimizer buffers: verify serialization/restoration without training.
        parameter = next(model.parameters())
        optimizer.state[parameter] = {"step": torch.tensor(7.), "exp_avg": torch.ones_like(parameter),
                                      "exp_avg_sq": torch.full_like(parameter, 2.)}
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()}
        expected = (random.random(), np.random.random(), torch.rand(5))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            atomic_save(state, path)
            loaded = torch.load(path, weights_only=True, map_location="cpu")
            clone = HistoryAE(**model.config)
            opt = torch.optim.AdamW(clone.parameters(), lr=.1)
            clone.load_state_dict(loaded["model"])
            opt.load_state_dict(loaded["optimizer"])
            self.assertEqual(opt.param_groups[0]["lr"], 3e-4)
            self.assertEqual(opt.state[next(clone.parameters())]["step"].item(), 7)
            torch.testing.assert_close(opt.state[next(clone.parameters())]["exp_avg_sq"], torch.full_like(parameter, 2.))
            restore_rng(loaded["rng"])
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            torch.testing.assert_close(torch.rand(5), expected[2])
            self.assertFalse(path.with_name("last.pt.tmp").exists())

    def test_validation_matches_existing_loss_and_reports_multiscale(self):
        data = series(400)
        encoded = [encode_context(s.frame, s.period, "ema8_32") for s in data]
        dataset = HistoryWindows(data, encoded, split_boundaries(data), "val", window=32)
        model = HistoryAE(hidden=16, layers=1, latent=8, window=32, context="ema8_32").eval()
        report = validation_report(model, dataset, "cpu", 8)
        self.assertAlmostEqual(report["loss"], validate(model, dataset, "cpu", 8))
        for h in (1, 4, 16):
            self.assertIn(f"change_{h}_correlation", report["metrics"])
        self.assertEqual(report["windows"], len(dataset))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        json.dumps(report, allow_nan=False)

    def test_publish_repairs_partial_reports_and_preserves_best(self):
        state = {"metadata": {"epochs": 100}, "initial_validation": {"loss": .1},
                 "best_epoch": 29, "best_model": {"weight": torch.tensor([2.])},
                 "history": [{"epoch": 31, "validation_loss": .2}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "history.jsonl").write_text('partial data')
            publish(state, root)
            rows = [json.loads(l) for l in (root / "history.jsonl").read_text().splitlines()]
            self.assertEqual(rows, state["history"])
            self.assertEqual(torch.load(root / "best.pt", weights_only=True)["epoch"], 29)
            self.assertEqual(json.loads((root / "manifest.json").read_text())["epochs"], 100)

    def test_configuration_mismatch_rejected(self):
        from obson.babel.ae_context import feature_names
        ck = dict(schema=SCHEMA, manifest={}, config=HistoryAE(context="ema8_32").config,
                  features=feature_names("ema8_32"), weights=WEIGHTS, delta_loss_weight=.5,
                  batch_size=32, seed=42, stride=16)
        check_source(ck, {}, "ema8_32", 32, 42)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            check_source(ck, {}, "ema8_32", 64, 42)

    def test_extension_export_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "warm_start_validation.json").write_text('{"loss":0.1}')
            env = dict(os.environ, BABEL_AE_RUN=str(run), BABEL_DOWNLOAD_DIR=str(root / "download"))
            subprocess.run(["bash", "scripts/babel_ae_extend_autodl.sh", "export"], env=env, check=True, capture_output=True)
            self.assertTrue((root / "download/run/warm_start_validation.json").exists())
            self.assertTrue((root / "download/run/extension_notes.md").exists())

    def test_completed_resume_restores_reports_without_optimizer_steps(self):
        from obson.babel.ae_context import feature_names
        data = series(1600)
        bounds = split_boundaries(data)
        encoded = [encode_context(s.frame, s.period, "ema8_32") for s in data]
        model = HistoryAE(context="ema8_32")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        metadata = dict(schema=SCHEMA, manifest=manifest(data, bounds), config=model.config,
                        features=feature_names("ema8_32"), weights=WEIGHTS, delta_loss_weight=.5,
                        batch_size=32, seed=42, stride=16, epochs=100)
        state = dict(resume_schema="babel-ae-resume-v1", metadata=metadata, epoch=100,
                     model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                     best_model=model.state_dict(), best_epoch=90, best_loss=.05,
                     history=[{"epoch":100,"validation_loss":.06}], initial_validation={"loss":.1})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_save(state, root / "last.pt")
            extend(data, encoded, bounds, root, 100, 32, 42, "cpu", "ema8_32", resume=True)
            self.assertEqual(torch.load(root / "best.pt", weights_only=True)["epoch"], 90)
            self.assertEqual(json.loads((root / "history.jsonl").read_text())["epoch"], 100)
            with self.assertRaisesRegex(ValueError, "budget"):
                extend(data, encoded, bounds, root, 101, 32, 42, "cpu", "ema8_32", resume=True)
