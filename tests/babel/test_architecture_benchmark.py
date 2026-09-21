"""Synthetic causality, target, selection and recovery checks; no optimizer steps."""
import copy
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

from obson.babel import architecture as ar, architecture_benchmark as ab
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256

SMALL = dict(ar.CONFIG, latent=8, gru_width=8, attention_width=8, conv_width=8,
             attention_layers=1, conv_layers=3, heads=2, decoder_width=8, decoder_layers=1)


def features(n=12, seed=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .1, (n, 128, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = .2
    return x


def fixture(root, epochs=2):
    out = root/'run'; cache = out/'cache'; cache.mkdir(parents=True)
    meta = ab.make_manifest(root/'transfer', root/'cross', dict(encoder_run=str(root/'encoder')), {}, epochs, 4, 2)
    meta['config'] = SMALL
    meta = json.loads(json.dumps(meta))
    atomic_json(meta, out/'manifest.json')
    files = {}
    for split, n in [('train', 12), ('val', 6), ('test', 6), ('cross_research', 6)]:
        x = features(n, n+len(split)); y, m = ar.ordered_targets(x)
        if split=='train':
            stats = ar.fit_scales(x, y, m); atomic_json(stats, cache/'statistics.json'); files['statistics.json'] = sha256(cache/'statistics.json')
        x, y, m = ar.normalize(x, y, m, stats)
        for name, value in [('x', x), ('y', y), ('mask', m)]:
            p = cache/f'{split}_{name}.npy'; np.save(p, value); files[p.name] = sha256(p)
        if split in ('test', 'cross_research'):
            p = cache/f'{split}_inventory.json'
            atomic_json([dict(key=f'A/15/C{i}', symbol='A', period=15, week=f'w{i}', month='2026-01', end=f'2026-01-{i+1:02d}') for i in range(n)], p); files[p.name] = sha256(p)
    atomic_json(dict(manifest=meta, files=files), cache/'index.json')
    return meta, out


class OrderedArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causal_prefix_gradients_and_common_decoder(self):
        decoders = []
        for kind in ar.ENCODERS:
            model = ar.OrderedModel(kind, SMALL, 42).eval()
            decoders.append(model.decoder.state_dict())
            x = torch.randn(2, 128, 28, requires_grad=True)
            y = model.encoder(x)
            changed = x.detach().clone(); changed[:, 50:] += torch.randn_like(changed[:, 50:])*10
            torch.testing.assert_close(y[:, :50], model.encoder(changed)[:, :50], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(y[:, :50], model.encoder(x[:, :50]), atol=1e-6, rtol=1e-6)
            y[:, 49, 0].sum().backward()
            self.assertEqual(float(x.grad[:, 50:].abs().max()), 0.)
            self.assertGreater(float(x.grad[:, :50].abs().sum()), 0.)
            with self.assertRaises(ValueError): model.decoder(y)
        for d in decoders[1:]:
            for key in d: self.assertTrue(torch.equal(d[key], decoders[0][key]))

    def test_observed_price_geometry_masks_and_current_exclusion(self):
        x = np.zeros((2, 128, 28), np.float32)
        x[..., 0] = np.arcsinh(.1); x[..., 1] = np.arcsinh(.2)
        x[..., 18:23] = [.3, .4, .5, .6, .7]; x[..., 23:28] = 1
        y, mask = ar.ordered_targets(x)
        np.testing.assert_allclose(y[0, :127, 0], np.arange(1, 128)*.3, rtol=1e-6)
        np.testing.assert_allclose(y[:, :127, 1], .2, rtol=1e-6)
        self.assertFalse(mask[:, -1].any())
        x[:, 3, 24] = 0; x[:, 4, 23] = 0; x[:, 5, 27] = 0; x[:, 6, 21] = np.arcsinh(2.)
        x[:, 7, 22] = 0
        y2, m2 = ar.ordered_targets(x)
        self.assertFalse(m2[:, 3, 5].any()); self.assertFalse(m2[:, 4, 4].any())
        self.assertFalse(m2[:, 5, 3].any()); self.assertFalse(m2[:, 6:8, 4:6].any())
        self.assertTrue(m2[:, :127, :2].all())
        np.testing.assert_array_equal(y2[~m2], 0)
        # Appending or changing later input cannot alter earlier observed target geometry.
        x[:, 80:] *= 2
        y3, m3 = ar.ordered_targets(x)
        np.testing.assert_array_equal(y3[:, :80], y2[:, :80])

    def test_extraction_never_crosses_contract_partition(self):
        bank = np.vstack([np.ones((256, 28)), np.ones((256, 28))*2])
        specs = [dict(offset=0, length=256, endpoints=[[127, 1]]), dict(offset=256, length=256, endpoints=[[255, 0]])]
        w = ab.extract_windows(bank, specs, 2)
        self.assertTrue((w[0]==2).all()); self.assertTrue((w[1]==1).all())
        for endpoint in (126, 256):
            bad = copy.deepcopy(specs); bad[0]['endpoints'][0][0] = endpoint
            with self.assertRaises(ValueError): ab.extract_windows(bank, bad, 2)
        bad = copy.deepcopy(specs); bad[1]['endpoints'][0][1] = 1
        with self.assertRaises(ValueError): ab.extract_windows(bank, bad, 2)

    def test_masked_objective_perfect_reconstruction_and_delta_scale(self):
        x = features(3); y, m = ar.ordered_targets(x); stats = ar.fit_scales(x, y, m)
        _, yn, _ = ar.normalize(x, y, m, stats)
        t = torch.tensor(yn); mask = torch.tensor(m)
        self.assertEqual(float(ar.error_rows(t, t, mask, stats)['primary'].max()), 0.)
        corrupt = t.clone(); corrupt[~mask] = 10000
        self.assertEqual(float(ar.error_rows(corrupt, t, mask, stats)['primary'].max()), 0.)
        shift = t.clone(); shift[..., 0] += 2
        rows = ar.error_rows(shift, t, mask, stats)
        torch.testing.assert_close(rows['path'], torch.ones(3)*4)
        torch.testing.assert_close(rows['changes'], torch.zeros(3), atol=1e-10, rtol=0)
        self.assertTrue(np.all(np.array(stats['y_scale'])>0))

    def test_microbatch_gradient_matches_effective_batch_without_step(self):
        x = features(7); y, m = ar.ordered_targets(x); stats = ar.fit_scales(x, y, m)
        x, y, m = ar.normalize(x, y, m, stats); data = dict(x=x, y=y, mask=m)
        initial = ar.OrderedModel('gru', SMALL, 42)
        states = []
        with patch.object(torch.optim.AdamW, 'step', return_value=None):
            for micro in (7, 3):
                model = copy.deepcopy(initial)
                opt = torch.optim.AdamW(model.parameters(), lr=.001)
                with patch.object(torch.nn.utils, 'clip_grad_norm_', return_value=0.):
                    _, updates = ab.run_epoch(model, data, stats, 7, micro, 'cpu', opt)
                self.assertEqual(updates, 1)
                states.append([p.grad.clone() for p in model.parameters()])
                for a, b in zip(model.parameters(), initial.parameters()): self.assertTrue(torch.equal(a, b))
        for a, b in zip(*states): torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)

    def test_transparent_baselines_budget_and_no_per_sample_mask_decode(self):
        x = features(12); y, m = ar.ordered_targets(x); stats = ar.fit_scales(x, y, m)
        _, y, m = ar.normalize(x, y, m, stats); data = dict(y=y, mask=m)
        pca = ab.fit_pca(data, 8)
        self.assertEqual(pca['components'].shape, (8, 896))
        self.assertEqual(pca['components'].dtype, np.float32)
        pred = ab.pca_predict(pca, data)
        self.assertTrue(np.isfinite(pred).all())
        altered = dict(data, mask=~m)
        np.testing.assert_array_equal(pred, ab.pca_predict(pca, altered))
        base = ab.baseline_predictions(data)
        self.assertEqual(base['coarse112'].shape, y.shape)
        permuted = ar.permute_old_blocks(x)
        np.testing.assert_array_equal(permuted[:, -16:], x[:, -16:])
        np.testing.assert_array_equal(np.sort(permuted, axis=1), np.sort(x, axis=1))

    def test_complete_matrix_selection_lock_and_no_training_on_research(self):
        with tempfile.TemporaryDirectory() as td:
            meta, out = fixture(Path(td)); original_run = ab.run_epoch; original_load = ab.load_arrays
            seen = []
            def no_update(model, data, stats, batch, micro, device, opt=None, order_seed=0):
                # Exercise actual numerical validation and publication, never update a neural parameter.
                return original_run(model, data, stats, batch, micro, device, None, order_seed)
            def train_load(path, split):
                self.assertIn(split, ('train', 'val')); seen.append(split)
                return original_load(path, split)
            with patch.object(ab, 'run_epoch', side_effect=no_update), patch.object(ab, 'load_arrays', side_effect=train_load), patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No optimizer updates locally')):
                for job in meta['experiments']: ab.worker(out, job['name'], 'cpu')
            self.assertEqual(len(seen), 24)
            def eval_load(path, split):
                if split in ('test', 'cross_research'):
                    self.assertTrue((out/'selection_lock.json').exists()); self.assertTrue((out/'baseline_lock.json').exists())
                return original_load(path, split)
            with patch.object(ab, 'load_arrays', side_effect=eval_load): result = ab.evaluate(meta, out, 'cpu')
            self.assertEqual(len(result['trials']), 12)
            for split in ('test', 'cross_research'):
                self.assertEqual(len(result['datasets'][split]['variants']), 10)
                self.assertEqual(len(result['datasets'][split]['all_trials']), 12)
                self.assertEqual(len(result['datasets'][split]['paired']), 18)
            self.assertFalse(result['goal']['automatic_promotion'])
            self.assertTrue(all(not row['trained_selection'] for row in result['trials'].values()))
            self.assertNotIn('NaN', (out/'architecture_metrics.json').read_text())
            self.assertTrue((out/'examples.html').exists())
            chosen = read(out/'selection_lock.json'); again = ab.lock_selection(meta, out)
            self.assertEqual(chosen, again)
            # Completed runs skip; tampered checkpoint must be rejected.
            with patch.object(ab, 'run_epoch', side_effect=AssertionError('Completed worker reran')):
                ab.worker(out, meta['experiments'][0]['name'], 'cpu')
            (out/meta['experiments'][0]['name']/'best.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError): ab.lock_selection(meta, out)

    def test_epoch_resume_replays_validation_and_restores_lr_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            meta, out = fixture(Path(td), epochs=3); job = meta['experiments'][0]; original_run = ab.run_epoch
            calls = []
            def interrupted(model, data, stats, batch, micro, device, opt=None, order_seed=0):
                if opt is not None:
                    calls.append(opt.param_groups[0]['lr'])
                    if len(calls)==2: raise RuntimeError('synthetic interruption')
                return original_run(model, data, stats, batch, micro, device, None)
            with patch.object(ab, 'run_epoch', side_effect=interrupted), patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No training')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic'): ab.worker(out, job['name'], 'cpu')
            path = out/job['name']; first = read(path/'history.json')[0]
            calls.clear()
            def resume(model, data, stats, batch, micro, device, opt=None, order_seed=0):
                if opt is not None: calls.append(opt.param_groups[0]['lr'])
                return original_run(model, data, stats, batch, micro, device, None)
            with patch.object(ab, 'run_epoch', side_effect=resume): ab.worker(out, job['name'], 'cpu')
            self.assertEqual(read(path/'history.json')[0], first)
            self.assertEqual(read(path/'resume_validation.json')['epoch'], 1)
            self.assertEqual(calls, [ar.learning_rate(i, 3, job['lr']) for i in (2, 3)])

    def test_prepare_uses_identical_endpoints_and_train_only_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); out = root/'out'; out.mkdir()
            transfer, cross, bank = root/'transfer', root/'cross', root/'encoder/cache'
            for directory in (transfer/'cache', cross/'cache', bank): directory.mkdir(parents=True)
            raw = {}
            for split in ab.SPLITS:
                n = 3; x = features(n, 100+len(split))
                if split!='train': x[..., :2] += .1
                raw[split] = x
                directory = cross/'cache' if split=='cross_research' else bank
                prefix = 'test' if split=='cross_research' else split
                np.save(directory/f'{prefix}_x.npy', x.reshape(-1, 28))
                atomic_json([dict(offset=i*128, length=128, endpoints=[[127, i]]) for i in range(n)], directory/f'{prefix}_sequences.json')
                source = cross/'cache' if split=='cross_research' else transfer/'cache'
                np.save(source/f'{prefix}_y.npy', np.zeros((n, 12), np.float32))
                if split in ('test', 'cross_research'):
                    atomic_json([dict(key=f'A/15/{i}', symbol='A', period=15, week=f'w{i}', end='2026-01-01') for i in range(n)], source/'test_inventory.json')
            meta = dict(bank=str(bank), transfer=str(transfer), cross_run=str(cross))
            ab.prepare(meta, out)
            stats = read(out/'cache/statistics.json')
            expected = ar.fit_scales(raw['train'], *ar.ordered_targets(raw['train']))
            self.assertEqual(stats, expected)
            for split in ab.SPLITS:
                data = ab.load_arrays(out, split)
                x, y, mask = ar.normalize(raw[split], *ar.ordered_targets(raw[split]), expected)
                np.testing.assert_array_equal(data['x'], x); np.testing.assert_array_equal(data['mask'], mask)
                np.testing.assert_array_equal(data['y'], y)
            with patch.object(ab, 'extract_windows', side_effect=AssertionError('Cache should be reused')):
                ab.prepare(meta, out)
            (out/'cache/test_x.npy').write_bytes(b'bad')
            with self.assertRaises(ValueError): ab.prepare(meta, out)

    def test_lr_selection_uses_validation_and_locks_all_weights(self):
        with tempfile.TemporaryDirectory() as td:
            meta, out = fixture(Path(td))
            for i, job in enumerate(meta['experiments']):
                directory = out/job['name']; directory.mkdir()
                summary = dict(validation=dict(primary=2. if job['lr']==1e-4 else 1.), selected_epoch=5, trained_selection=True)
                atomic_json(summary, directory/'training_summary.json')
                for kind in ('best', 'last'): (directory/f'{kind}.pt').write_bytes(str(i).encode())
                files = {f.name: sha256(f) for f in directory.iterdir()}
                atomic_json(dict(status='complete', metadata=dict(manifest=meta, job=job), files=files), directory/'completion.json')
            with patch.object(ab, 'load_arrays', side_effect=AssertionError('Selection must not open research arrays')):
                lock = ab.lock_selection(meta, out)
            self.assertEqual(len(lock['chosen']), 6)
            self.assertTrue(all(name.endswith('lr0.0003') for name in lock['chosen'].values()))
            self.assertEqual(len(lock['weights']), 12)

    def test_shell_exports_failure_and_omits_binary_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = root/'run'; run.mkdir(); download = root/'download'
            atomic_json(dict(status='partial'), run/'metrics.json')
            (run/'last.pt').write_bytes(b'weights must stay on server')
            (run/'cache.npy').write_bytes(b'cache must stay on server')
            fake = root/'python-fails'; fake.write_text('#!/bin/sh\nexit 7\n'); fake.chmod(0o755)
            env = dict(os.environ, BABEL_ARCH_RUN=str(run), BABEL_ARCH_LOG=str(root/'absent.log'),
                       BABEL_DOWNLOAD_DIR=str(download), PYTHON_BIN=str(fake))
            repo = Path(__file__).resolve().parents[2]
            command = ['bash', str(repo/'scripts/babel_architecture_autodl.sh')]
            result = subprocess.run(command+['all'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            archive = download/'run_reports.tar.gz'
            with tarfile.open(archive) as stream:
                names = stream.getnames()
                self.assertIn('run/metrics.json', names)
                self.assertFalse(any(name.endswith(('.pt', '.npy')) for name in names))
                status = stream.extractfile('run/run_status.txt').read().decode()
                self.assertIn('run_status=failed', status); self.assertIn('command_exit_code=7', status)
            result = subprocess.run(command+['export'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with tarfile.open(archive) as stream:
                self.assertIn('run_status=failed', stream.extractfile('run/run_status.txt').read().decode())

    def test_train_only_scales_and_identity_guards(self):
        x = features(3); y, mask = ar.ordered_targets(x)
        stats = ar.fit_scales(x, y, mask)
        val = features(2)*100
        ar.normalize(val, *ar.ordered_targets(val), stats)
        self.assertEqual(stats, ar.fit_scales(x, y, mask))
        flags = [11, 23, 24, 25, 26, 27]
        self.assertEqual([stats['x_scale'][i] for i in flags], [1.]*6)
        self.assertEqual([stats['x_mean'][i] for i in flags], [0.]*6)
        with self.assertRaises(ValueError): ab.verify_code(dict(code_sha256={}))
        with self.assertRaises(ValueError): ab.make_manifest(Path('t'), Path('c'), dict(encoder_run='e'), {}, micro=65)


def read(path): return json.loads(path.read_text())


if __name__ == '__main__': unittest.main()
