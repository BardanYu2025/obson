"""Synthetic capacity/causality/integrity checks; neural optimizer updates prohibited."""
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
from obson.babel import capacity as ca, capacity_benchmark as cb
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import fixture

SMALL = dict(ca.CONFIG, latent=8, attention_layers=1, attention_ff=8,
             conv_inner=6, heads=2, decoder_width=4, decoder_layers=1)


def read(path): return json.loads(path.read_text())


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causality_prefix_and_backward_all_encoders(self):
        for name in ca.ENCODERS:
            model = ca.CapacityModel(name, SMALL, 42, 8).eval()
            x = torch.randn(2, 128, 28, requires_grad=True)
            h = model.encoder(x)
            changed = x.detach().clone(); changed[:, 50:] += 7
            torch.testing.assert_close(h[:, :50], model.encoder(changed)[:, :50], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(h[:, :50], model.encoder(x[:, :50]), atol=1e-6, rtol=1e-6)
            h[:, 49].square().sum().backward()
            self.assertEqual(float(x.grad[:, 50:].abs().max()), 0.)
            self.assertGreater(float(x.grad[:, :50].abs().sum()), 0.)
            pred = model(x.detach()); pred.square().mean().backward()
            self.assertEqual(tuple(pred.shape), (2, 128, 7))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_convolution_reaches_oldest_bar_without_future(self):
        model = ca.MultiScaleEncoder(SMALL).double()
        self.assertEqual(model.receptive_field, 128)
        x = torch.randn(1, 128, 28, dtype=torch.double, requires_grad=True)
        model(x)[0, -1].square().sum().backward()
        self.assertGreater(float(x.grad[0, 0].abs().sum()), 0.)

    def test_rank_nullspace_and_no_nominal_only_expansion(self):
        for rank in (4, 8):
            d = ca.MemoryDecoder(SMALL, rank).double().eval()
            z = torch.randn(1, 8, dtype=torch.double, requires_grad=True)
            jac = torch.autograd.functional.jacobian(lambda v: d.conditioning(v).flatten(), z).reshape(8, 8)
            self.assertEqual(int(torch.linalg.matrix_rank(jac, atol=1e-5)), rank)
            self.assertEqual(d.conditioning_audit()['numerical_rank'], rank)
            if rank == 4:
                p = d.projector / (2**.5)
                null = torch.randn_like(z) @ (torch.eye(8, dtype=z.dtype)-p)
                torch.testing.assert_close(d.conditioning(z), d.conditioning(z+null), atol=2e-7, rtol=1e-6)
                torch.testing.assert_close(d(z), d(z+null), atol=2e-7, rtol=1e-6)
                full = ca.MemoryDecoder(SMALL, 8).double()
                self.assertGreater(float((full.conditioning(z)-full.conditioning(z+null)).abs().max().detach()), .01)
            with self.assertRaises(ValueError): d(torch.zeros(1, 128, 8))

    def test_common_initialization_parameters_and_projector_rng(self):
        ds = []
        for mode in ca.ENCODERS:
            low = ca.CapacityModel(mode, SMALL, 42, 4)
            full = ca.CapacityModel(mode, SMALL, 42, 8)
            for (an, a), (bn, b) in zip(low.named_parameters(), full.named_parameters()):
                self.assertEqual(an, bn); self.assertTrue(torch.equal(a, b), an)
            ds.append(dict(full.decoder.named_parameters()))
        for d in ds[1:]:
            for k, v in d.items(): self.assertTrue(torch.equal(v, ds[0][k]), k)
        state = torch.get_rng_state().clone(); ca.orthogonal_basis(8)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for rank in (0, 3, 9):
            with self.assertRaises(ValueError): ca.MemoryDecoder(SMALL, rank)


class CapacityPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        meta, source = fixture(cls.root, epochs=1)
        original = ab.run_epoch
        def no_update(model, data, stats, batch, micro, device, opt=None, order_seed=0):
            return original(model, data, stats, batch, micro, device, None, order_seed)
        cls.no_update = staticmethod(no_update)
        with patch.object(ab, 'run_epoch', side_effect=no_update), patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No neural updates locally')):
            for job in meta['experiments']: ab.worker(source, job['name'], 'cpu')
        ab.evaluate(meta, source, 'cpu')
        # The legacy report uses the literal pca512 label even in tiny fixtures.
        for split in ('test', 'cross_research'):
            path = source/f'{split}_per_window_errors.json'; rows = read(path)
            rows['pca8'] = rows.pop('pca512'); atomic_json(rows, path)
        path = source/'architecture_metrics.json'; rows = read(path)
        for split in ('test', 'cross_research'):
            variants = rows['datasets'][split]['variants']; variants['pca8'] = variants.pop('pca512')
        atomic_json(rows, path)
        files = {f.name: sha256(f) for f in source.iterdir() if f.is_file() and f.suffix in ('.json', '.html', '.md')}
        files['cache/index.json'] = sha256(source/'cache/index.json')
        atomic_json(dict(status='complete', source_unchanged=True, files=files), source/'completion.json')
        cls.source = source; cls.identity = cb.source_identity(source)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def make_run(self, root, epochs=2):
        out = Path(root)/'capacity'; out.mkdir()
        with patch.object(ca, 'CONFIG', SMALL):
            meta = cb.make_manifest(self.source, self.identity, epochs, 4, 2)
        atomic_json(meta, out/'manifest.json')
        return meta, out

    def test_read_only_source_and_pca_prefix_replay(self):
        before = {str(f): (sha256(f), f.stat().st_mtime_ns) for f in self.source.rglob('*') if f.is_file()}
        with tempfile.TemporaryDirectory() as td:
            meta, out = self.make_run(td)
            report = cb.audit(meta, out)
            self.assertIn('pca4', report['datasets']['test']['variants'])
            pca = cb.source_pca(meta); half = cb.prefix_pca(pca, 4)
            np.testing.assert_array_equal(half['components'], pca['components'][:4])
            self.assertEqual(cb.source_identity(self.source), self.identity)
            # A replay mismatch must fail, not silently loosen tolerances or rerun training.
            path = self.source/'test_per_window_errors.json'; original = path.read_bytes()
            bad = read(path); bad['pca8']['primary'][0] += .1; atomic_json(bad, path)
            try:
                with self.assertRaisesRegex(ValueError, 'PCA per-window replay'): cb.audit(meta, out)
                with self.assertRaises(ValueError): cb.source_identity(self.source)
            finally:
                path.write_bytes(original)
                os.utime(path, ns=(path.stat().st_atime_ns, before[str(path)][1]))
        after = {str(f): (sha256(f), f.stat().st_mtime_ns) for f in self.source.rglob('*') if f.is_file()}
        self.assertEqual(before, after)

    def test_full_mocked_matrix_selection_and_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            meta, out = self.make_run(td, 1); loader = ab.load_arrays
            cb.audit(meta, out); cb.preflight(meta, out, 'cpu')
            def train_load(path, split):
                self.assertEqual(path.resolve(), self.source.resolve()); self.assertIn(split, ('train', 'val'))
                return loader(path, split)
            with patch.object(ab, 'run_epoch', side_effect=self.no_update), patch.object(ab, 'load_arrays', side_effect=train_load), patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No neural training')):
                for job in meta['experiments']: cb.worker(out, job['name'], 'cpu')
            self.assertEqual(len(meta['experiments']), 12)
            def eval_load(path, split):
                if split in ('test', 'cross_research'): self.assertTrue((out/'selection_lock.json').exists())
                return loader(path, split)
            with patch.object(ab, 'load_arrays', side_effect=eval_load): result = cb.evaluate(meta, out, 'cpu')
            for split in ('test', 'cross_research'):
                self.assertEqual(len(result['datasets'][split]['variants']), 17)
                self.assertEqual(len(result['datasets'][split]['fixed_last']), 12)
                self.assertEqual(len(result['datasets'][split]['paired']), 30)
            self.assertTrue((out/'examples.html').exists())
            self.assertFalse(result['goal']['automatic_promotion'])
            self.assertNotIn('NaN', (out/'capacity_metrics.json').read_text())
            self.assertTrue(all(not r['trained_selection'] for r in result['trials'].values()))
            with patch.object(ab, 'run_epoch', side_effect=AssertionError('Rerun completed worker')):
                cb.worker(out, meta['experiments'][0]['name'], 'cpu')
            (out/meta['experiments'][0]['name']/'best.pt').write_bytes(b'bad')
            with self.assertRaises(ValueError): cb.lock_selection(meta, out)

    def test_resume_replay_and_incomplete_lock(self):
        with tempfile.TemporaryDirectory() as td:
            meta, out = self.make_run(td, 3); job = meta['experiments'][0]; calls = []
            with self.assertRaises(FileNotFoundError): cb.lock_selection(meta, out)
            def interrupt(model, data, stats, batch, micro, device, opt=None, order_seed=0):
                if opt is not None:
                    calls.append(opt.param_groups[0]['lr'])
                    if len(calls)==2: raise RuntimeError('interrupt')
                return self.no_update(model, data, stats, batch, micro, device)
            with patch.object(ab, 'run_epoch', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupt'): cb.worker(out, job['name'], 'cpu')
            first = read(out/job['name']/'history.json')[0]
            with patch.object(ab, 'run_epoch', side_effect=self.no_update): cb.worker(out, job['name'], 'cpu')
            self.assertEqual(read(out/job['name']/'history.json')[0], first)
            self.assertEqual(read(out/job['name']/'resume_validation.json')['epoch'], 1)
            self.assertEqual(len(read(out/job['name']/'history.json')), 3)
            bad = copy.deepcopy(meta); bad['code_sha256'] = {}
            with self.assertRaises(ValueError): cb.verify_code(bad)

    def test_export_failure_and_binary_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = root/'run'; run.mkdir(); download = root/'download'
            atomic_json(dict(status='partial'), run/'metrics.json'); (run/'last.pt').write_bytes(b'binary')
            fake = root/'fail'; fake.write_text('#!/bin/sh\nexit 7\n'); fake.chmod(0o755)
            env = dict(os.environ, BABEL_CAPACITY_RUN=str(run), BABEL_CAPACITY_LOG=str(root/'missing'), BABEL_DOWNLOAD_DIR=str(download), PYTHON_BIN=str(fake))
            repo = Path(__file__).resolve().parents[2]
            cmd = ['bash', str(repo/'scripts/babel_capacity_autodl.sh')]
            r = subprocess.run(cmd+['all'], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 7, r.stderr)
            with tarfile.open(download/'run_reports.tar.gz') as stream:
                self.assertFalse(any(n.endswith('.pt') for n in stream.getnames()))
                self.assertIn('run_status=failed', stream.extractfile('run/run_status.txt').read().decode())
            r = subprocess.run(cmd+['export'], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with tarfile.open(download/'run_reports.tar.gz') as stream:
                self.assertIn('run_status=failed', stream.extractfile('run/run_status.txt').read().decode())


if __name__ == '__main__': unittest.main()
