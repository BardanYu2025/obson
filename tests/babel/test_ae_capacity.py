"""Capacity profiles and synthetic forward/backward tests, no optimizer updates."""
import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_babel import series
from obson.babel.ae_capacity import build_model, check_metadata, config, train_capacity
from obson.babel.ae_context import encode_context, feature_names
from obson.babel.ae_extend import atomic_save, rng_state
from obson.babel.data import manifest, split_boundaries
from obson.babel.history_autoencoder import SCHEMA, WEIGHTS, HistoryAE, reconstruction_loss


class CapacityTests(unittest.TestCase):
    def test_fresh_initialization_without_training_and_baseline_guard(self):
        data = series(1600)
        bounds = split_boundaries(data)
        encoded = [encode_context(s.frame,s.period,"ema8_32") for s in data]
        baseline = dict(schema=SCHEMA,config=HistoryAE(context="ema8_32").config,
                        manifest=manifest(data,bounds),epochs=200,batch_size=32,seed=42,
                        stride=16,weights=WEIGHTS,delta_loss_weight=.5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root / "baseline"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps(baseline))
            (source / "history.jsonl").write_text('{"epoch":200}\n')
            (source / "ae_metrics.json").write_text('{}')
            (source / "best.pt").write_bytes(b'fingerprint-only synthetic fixture')
            with patch("obson.babel.ae_capacity.run_epochs") as loop:
                train_capacity(data,encoded,bounds,root / "new",200,32,42,"cpu",128,source)
                loop.assert_called_once()
            saved=torch.load(root / "new/last.pt",weights_only=True)
            self.assertEqual(saved["epoch"],0)
            self.assertEqual(saved["history"],[])
            self.assertEqual(saved["optimizer"]["state"],{})
            self.assertEqual(saved["metadata"]["training_mode"],"from_scratch")
            with self.assertRaisesRegex(ValueError,"mismatch"):
                train_capacity(data,encoded,bounds,root / "bad",199,32,42,"cpu",128,source)
            self.assertFalse((root / "bad").exists())

    def test_matched_backbone_rng_causality_and_bottleneck(self):
        small = build_model(128, 42).eval()
        state = torch.get_rng_state()
        wide = build_model(256, 42).eval()
        torch.testing.assert_close(state, torch.get_rng_state())
        for name in ("input", "encoder", "decoder", "output", "context_input"):
            for key, value in getattr(small, name).state_dict().items():
                torch.testing.assert_close(value, getattr(wide, name).state_dict()[key], rtol=0, atol=0)
        x = torch.randn(1, 128, 18)
        for model, latent in ((small,128), (wide,256)):
            result = model(x)
            self.assertEqual(result["z"].shape, (1,latent))
            self.assertEqual(result["reconstruction"].shape, (1,128,7))
            torch.testing.assert_close(result["reconstruction"], model.decode(result["z"]))
            torch.testing.assert_close(result["h"][:,:32], model.encode(x[:,:32]), rtol=1e-4, atol=1e-5)
            y = torch.randn_like(result["reconstruction"])
            y[...,2:] = y[...,2:].abs()
            reconstruction_loss(result["reconstruction"], y, torch.ones_like(y,dtype=torch.bool)).mean().backward()
            self.assertGreater(model.compress[1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(model.input.weight.grad).all())

    def test_completed_capacity_resume_and_profile_guard(self):
        data = series(1600)
        bounds = split_boundaries(data)
        encoded = [encode_context(s.frame,s.period,"ema8_32") for s in data]
        model = build_model(256,42)
        opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        metadata = dict(schema=SCHEMA, training_family="capacity", config=config(256),
                        manifest=manifest(data,bounds), epochs=200, batch_size=32, seed=42,
                        stride=16, weights=WEIGHTS, delta_loss_weight=.5, features=feature_names("ema8_32"),
                        parameters=sum(p.numel() for p in model.parameters()))
        state = dict(resume_schema="babel-ae-resume-v1", metadata=metadata, epoch=200,
                     model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                     initial_validation={"loss":1.},history=[{"epoch":200}],
                     best_epoch=190,best_loss=.1,best_model=model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_save(state,root / "last.pt")
            train_capacity(data,encoded,bounds,root,200,32,42,"cpu",256,resume=True)
            self.assertTrue((root / "initial_validation.json").exists())
            saved=torch.load(root / "best.pt",weights_only=True)
            self.assertEqual(saved["epoch"],190)
            restored=HistoryAE(**saved["config"])
            restored.load_state_dict(saved["model"],strict=True)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                check_metadata(metadata,manifest(data,bounds),128,200,32,42)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                check_metadata(metadata,manifest(data,bounds),256,201,32,42)

    def test_export_ignores_stale_extension_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run=root / "babel_ae_w256_l6_z128_s42"
            run.mkdir()
            (run / "initial_validation.json").write_text('{"loss":1}')
            env=dict(os.environ,BABEL_CAPACITY_ROOT=str(root), BABEL_DOWNLOAD_DIR=str(root / "download"),
                     BABEL_AE_RUN="wrong",BABEL_AE_CONTINUE_FROM="wrong",BABEL_AE_RESUME="1")
            subprocess.run(["bash","scripts/babel_ae_capacity_autodl.sh","128","export"],env=env,check=True,capture_output=True)
            self.assertTrue((root / "download" / run.name / "capacity_notes.md").exists())
            self.assertTrue((root / "download" / run.name / "initial_validation.json").exists())
