import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from obson.babel.ae_extend import atomic_save, rng_state
from obson.babel.ar_reconstruction import (ARDecoder, build, objective, rollout_weight, derangement,
    validate, evaluate, worker, run_jobs, paired_error_interval, read_arrays, preflight)

CONFIG = dict(latent=16, ar_width=16, parallel_width=16)


def cache(source, n=6):
    root = source/'target_cache'; root.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for split in ('train', 'val', 'test'):
        z = rng.normal(size=(n, 16)).astype(np.float32)
        y = rng.normal(size=(n, 64, 7)).astype(np.float32)*.05; y[..., 2:] = np.abs(y[..., 2:])
        mask = np.ones(y.shape, bool); mask[0, :, 5] = False; y[0, :, 5] = 0
        for name, value in (('short', z), ('recent_y', y), ('recent_mask', mask)):
            np.save(root/f'{split}_{name}.npy', value, allow_pickle=False)


class ARReconstructionTests(unittest.TestCase):
    def test_strict_teacher_shift_and_rollout_agree_without_true_seed(self):
        torch.manual_seed(5)
        model = ARDecoder(16, 16).eval(); z = torch.randn(2, 16); y = torch.randn(2, 64, 7)
        with torch.no_grad():
            a = model.teacher(z, y)
            changed = y.clone(); changed[:, 20:] += 10
            b = model.teacher(z, changed)
            torch.testing.assert_close(a[:, :21], b[:, :21], rtol=0, atol=0)
            self.assertGreater(float((a[:, 21:]-b[:, 21:]).abs().max()), .001)
            generated = model.rollout(z)
            torch.testing.assert_close(model.teacher(z, generated), generated, rtol=2e-5, atol=2e-5)
            self.assertTrue((generated[..., 2:] >= 0).all())
            with self.assertRaises(TypeError): model.rollout(z, y)

    def test_unrolled_gradient_and_noz_control(self):
        model = ARDecoder(16, 16).train(); z = torch.randn(2, 16)
        y = torch.randn(2, 64, 7)*.1; y[..., 2:] = y[..., 2:].abs()
        b = dict(z=z, y=y, mask=torch.ones_like(y, dtype=torch.bool))
        objective(model, 'ar_mix', b, 15).mean().backward()
        for p in (model.condition.weight, model.previous.weight, model.rnn.weight_hh_l0):
            self.assertIsNotNone(p.grad); self.assertGreater(float(p.grad.abs().sum()), 0.)
        no_z = ARDecoder(16, 16, no_z=True).eval()
        with torch.no_grad():
            torch.testing.assert_close(no_z.rollout(z), no_z.rollout(z+100), rtol=0, atol=0)
        self.assertEqual(rollout_weight('ar_mix', 5), 0)
        self.assertEqual(rollout_weight('ar_mix', 6), .1)
        self.assertEqual(rollout_weight('ar_mix', 15), 1)
        self.assertEqual(rollout_weight('ar_tf', 60), 0)

    def test_derangement_and_week_paired_interval(self):
        a = derangement(15)
        np.testing.assert_array_equal(np.sort(a), np.arange(15))
        self.assertTrue((a != np.arange(15)).all())
        np.testing.assert_array_equal(a, derangement(15))
        v = paired_error_interval(np.arange(20)+2, np.arange(20), np.repeat(np.arange(5), 4), repeats=20)
        self.assertEqual((v['delta_close_mae_bps'], v['low'], v['high']), (2., 2., 2.))

    def test_disposable_preflight_no_optimizer(self):
        with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('no training')):
            result = preflight(CONFIG, 2, 'cpu')
        self.assertEqual(set(result['stages']), {'parallel', 'ar_mix'})
        self.assertTrue(all(np.isfinite(v['gradient_norm']) for v in result['stages'].values()))

    def test_completed_resume_does_not_update_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; cache(source)
            out = root/'out'; out.mkdir(); path = out/'ar_tf_s42'; path.mkdir()
            job = dict(name='ar_tf_s42', mode='ar_tf', seed=42)
            meta = dict(config=CONFIG, experiments=[job], source=str(source), batch=3, epochs=2, lr=3e-4)
            (out/'manifest.json').write_text(json.dumps(meta))
            model = build('ar_tf', CONFIG)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            data = {k: torch.from_numpy(v) for k, v in read_arrays(source, 'val').items()}
            score = validate(model, data, 3)
            state = dict(metadata=dict(manifest=meta, job=job), epoch=2, best_epoch=1,
                         best_validation=score, history=[], model=model.state_dict(), best_model=model.state_dict(),
                         optimizer=opt.state_dict(), rng=rng_state())
            atomic_save(state, path/'last.pt')
            with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('no training')):
                worker(out, job['name'], 'cpu')
            best = torch.load(path/'best.pt', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, best['model'][key], rtol=0, atol=0)

    def test_end_to_end_reports_and_selected_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; cache(source)
            out = root/'out'; out.mkdir(); (source/'short').mkdir()
            jobs = [dict(name=f'{mode}_s42', mode=mode, seed=42) for mode in ('parallel', 'ar_tf', 'ar_mix', 'ar_noz_mix')]
            meta = dict(config=CONFIG, source=str(source), experiments=jobs, seeds=[42])
            (out/'manifest.json').write_text(json.dumps(meta))
            inventory = [dict(key='rb/60/rb2505', symbol='rb', period=60, row=511+i*128,
                              end=f'example {i}', anchor=100., week=str(i)) for i in range(6)]
            (out/'test_inventory.json').write_text(json.dumps(inventory))
            va = {k: torch.from_numpy(v) for k, v in read_arrays(source, 'val').items()}
            for job in jobs:
                model = build(job['mode'], CONFIG); path = out/job['name']; path.mkdir()
                score = validate(model, va, 3)
                atomic_save(dict(metadata=dict(manifest=meta, job=job), model=model.state_dict(), epoch=0,
                                 validation=score), path/'best.pt')
                if job['mode'] == 'parallel':
                    atomic_save(dict(model=model.state_dict(), epoch=33), source/'short/best.pt')
            evaluate(out, 3, 'cpu')
            report = json.loads((out/'ar_reconstruction_metrics.json').read_text())
            self.assertEqual(len(report['variants']), 5)
            self.assertEqual(len(report['paired']), 5)
            self.assertEqual(report['variants']['ar_noz_mix_s42']['z_control_max_output_change'], 0.)
            self.assertEqual(len(report['variants']['ar_mix_s42']['close_mae_by_step_bps']), 64)
            self.assertTrue((out/'ar_mix_s42/examples.html').exists())
            ck = torch.load(out/'ar_mix_s42/best.pt', weights_only=True)
            ck['validation']['free_loss'] += 10; atomic_save(ck, out/'ar_mix_s42/best.pt')
            with self.assertRaisesRegex(ValueError, 'did not reproduce'):
                evaluate(out, 3, 'cpu')

    def test_worker_failure_stops_other_active_worker(self):
        class Proc:
            def __init__(self, fail): self.pid = int(fail)+1; self.code = 1 if fail else None; self.terminated = False
            def poll(self): return self.code
            def terminate(self): self.terminated = True; self.code = -15
            def wait(self, timeout=None): return self.code
        bad, other = Proc(True), Proc(False)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'manifest.json').write_text(json.dumps(dict(experiments=[dict(name='a'), dict(name='b')])))
            with patch('obson.babel.ar_reconstruction.subprocess.Popen', side_effect=[bad, other]):
                with self.assertRaisesRegex(RuntimeError, 'exited 1'): run_jobs(out, 2)
            self.assertTrue(other.terminated)

    def test_export_failure_status_and_excluded_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            (run/'metrics.json').write_text('{}'); (run/'best.pt').write_bytes(b'weights')
            (run/'cache.npy').write_bytes(b'cache')
            env = dict(os.environ, BABEL_ARDEC_RUN=str(run), BABEL_DOWNLOAD_DIR=str(root/'download'),
                       BABEL_ARDEC_LOG=str(root/'none.log'), PYTHON_BIN='/usr/bin/false')
            for mode, code in (('export', 0), ('all', 1)):
                result = subprocess.run(['bash', 'scripts/babel_ardec_autodl.sh', mode], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as tar:
                    self.assertFalse(any(n.endswith(('.pt', '.npy')) for n in tar.getnames()))
                    status = tar.extractfile('run/run_status.txt').read().decode()
                    self.assertIn(f'command_exit_code={code}', status)
                    self.assertIn('run_status=failed' if code else 'run_status=partial', status)


if __name__ == '__main__': unittest.main()
