import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_babel import frame, series
from test_large_history import SMALL
from obson.babel.ae_context import encode_context
from obson.babel.large_history import LargeHistory
from obson.babel.short_state import ShortState
from obson.babel.structure import annotate
from obson.babel.state_readout import (AGES, price_features, sample_rows, rolling_long,
    fit_ridge, predict, paired_interval, evaluate, score, prepare_split)


class StateReadoutTests(unittest.TestCase):
    def test_price_and_rule_targets_are_prefix_invariant(self):
        df = frame(400)
        changed = df.copy()
        changed.loc[240:, ['open', 'high', 'low', 'close']] *= 1.3
        x = encode_context(df, 60, 'ema8_32')['x']
        altered = encode_context(changed, 60, 'ema8_32')['x']
        a = price_features(df, x); b = price_features(changed, altered)
        self.assertEqual(a.shape, (400, 25))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a[:240], b[:240])
        np.testing.assert_array_equal(annotate(df)['state'][:240], annotate(changed)['state'][:240])

    def test_sampling_respects_partition_current_eligibility_and_grid(self):
        ss = series(900)
        bounds = dict(train_until=str(ss[0].sessions[699]), val_until='2029-01-01', test_until='2030-01-01')
        ss[0].main[527] = False
        old = dict(keys=np.array([[0, 511], [0, 639]]), blocks=np.array([4, 5]))
        rows, ids = sample_rows(ss, bounds, 'train', old)
        self.assertNotIn(527, rows[:, 1])
        self.assertIn(511, rows[:, 1])  # Do not condition on a future sample being eligible.
        self.assertTrue((rows[:, 1] <= 699).all())
        self.assertTrue((rows[:, 1] - rows[:, 2] == rows[:, 3]).all())
        np.testing.assert_array_equal(rows[:, 2], old['keys'][ids, 1])
        old['keys'][0, 1] = 512
        with self.assertRaisesRegex(ValueError, 'Invalid source'):
            sample_rows(ss, bounds, 'train', old)

    def test_rolling_refresh_matches_direct_causal_encoder(self):
        torch.manual_seed(42)
        model = LargeHistory(SMALL, blocks=4, aggregate_layers=1).eval()
        ss = series(850)
        encoded = [encode_context(s.frame, s.period, 'ema8_32') for s in ss]
        rows = np.array([[0, 639, 639, 0, 4], [0, 655, 639, 16, 4], [0, 687, 639, 48, 4]])
        held = np.zeros((3, SMALL['latent']), np.float32)
        actual = rolling_long(model, ss, encoded, rows, held, 3, 'cpu')
        np.testing.assert_array_equal(actual[0], held[0])
        with torch.no_grad():
            for j in (1, 2):
                end = int(rows[j, 1]); begin = end + 1 - 512
                x = torch.tensor(encoded[0]['x'][begin:end+1].reshape(4, 128, 18))
                local = model.local.encode(x)[:, -1][None]
                close = ss[0].frame.close.to_numpy()
                anchors = close[np.arange(begin-1, end-128+1, 128)]
                offsets = torch.tensor((np.log(anchors) - np.log(close[end-128])) * 100, dtype=torch.float32)[None]
                expected = model.summarize(local, offsets, torch.ones(1, 4, dtype=torch.bool))
                np.testing.assert_allclose(actual[j], expected[0].numpy(), atol=2e-6, rtol=2e-6)
        changed = [dict(e) for e in encoded]
        changed[0]['x'] = encoded[0]['x'].copy(); changed[0]['x'][688:] += 100
        np.testing.assert_array_equal(actual, rolling_long(model, ss, changed, rows, held, 2, 'cpu'))

    def test_ridge_train_only_normalization_and_paired_uncertainty(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(80, 5)); y = np.arange(80) % 4
        val = rng.normal(size=(32, 5)) + 7; labels = np.arange(32) % 4
        fit = fit_ridge(x, y, val, labels)
        np.testing.assert_array_equal(fit['mean'], x.mean(0))
        np.testing.assert_array_equal(fit['scale'], x.std(0).clip(.01))
        self.assertEqual(fit['validation_ba'], max(c['validation_ba'] for c in fit['candidates']))
        pred = predict(fit, val)
        weeks = np.repeat(np.arange(8), 4)
        interval = paired_interval(pred, pred, labels, weeks, repeats=30)
        self.assertEqual((interval['delta_ba'], interval['low'], interval['high']), (0., 0., 0.))
        self.assertIsNone(paired_interval(pred, pred, labels, np.zeros(32), repeats=30)['low'])
        self.assertEqual(score(np.array([0, 1]), np.array([0, 1]))['present_classes'], 2)

    def test_shared_cache_alignment_resume_and_hash_guard(self):
        torch.manual_seed(12)
        ss = series(800)[:1]
        encoded = [encode_context(ss[0].frame, 60, 'ema8_32')]
        base = SimpleNamespace(series=ss, encoded=encoded)
        engine = SimpleNamespace(long_model=LargeHistory(SMALL, blocks=4, aggregate_layers=1).eval(),
                                 short_model=ShortState(SMALL['latent'], 2).eval())
        bounds = dict(train_until='2029-01-01', val_until='2030-01-01', test_until='2031-01-01')
        old = dict(keys=np.array([[0, 511]]), blocks=np.array([4]), long=np.ones((1, SMALL['latent']), np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            values = prepare_split(engine, base, bounds, 'train', old, out, {'version': 1}, 4, 2, 'cpu')
            with torch.no_grad():
                h, _ = engine.short_model(torch.tensor(encoded[0]['x'][:639])[None])
            np.testing.assert_allclose(values['short'], h[0, values['rows'][:, 1]].numpy(), rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(values['labels'], ss[0].labels['state'][values['rows'][:, 1]])
            with patch('obson.babel.state_readout.extract_short', side_effect=AssertionError('must reuse')):
                loaded = prepare_split(engine, base, bounds, 'train', old, out, {'version': 1}, 4, 2, 'cpu')
            np.testing.assert_array_equal(loaded['short'], values['short'])
            path = out/'cache/train.npz'
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                prepare_split(engine, base, bounds, 'train', old, out, {'version': 1}, 4, 2, 'cpu')

    def test_full_report_same_head_refresh_and_predictions(self):
        rng = np.random.default_rng(9); data = {}
        for split in ('train', 'val', 'test'):
            n = 60; ages = np.resize(AGES, n)
            rows = np.column_stack((np.zeros(n, int), np.arange(n)*128+511+ages,
                                    np.arange(n)*128+511, ages, np.full(n, 4)))
            z = rng.normal(size=(n, 4))
            data[split] = dict(rows=rows, labels=np.column_stack([(np.arange(n)+k)%4 for k in range(3)]),
                previous_labels=np.column_stack([(np.arange(n)+k)%4 for k in range(3)]),
                price_ema=rng.normal(size=(n, 3)), short=rng.normal(size=(n, 4)), long_held=z, long_fresh=z.copy(),
                symbol=np.repeat(['rb', 'sr'], 30), period=np.full(n, 60), week=np.repeat(np.arange(6), 10))
        def interval(*args, **kwargs):
            return paired_interval(*args, repeats=20)
        with tempfile.TemporaryDirectory() as tmp, patch('obson.babel.state_readout.paired_interval', side_effect=interval):
            out = Path(tmp); report = evaluate(data, out)
            self.assertEqual(len(report['variants']), 13)
            for name in ('long_held', 'dual_held', 'price_dual_held'):
                for scale, result in report['variants'][name].items():
                    self.assertEqual(result['test'], report['variants'][name+'_refresh_same_head'][scale]['test'])
            self.assertEqual(len((out/'test_predictions.jsonl').read_text().splitlines()), 60)
            self.assertTrue((out/'summary.md').exists())
            self.assertEqual(len(list((out/'readouts').glob('*.npz'))), 24)

    def test_export_excludes_arrays_and_preserves_failure_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            (run/'state_metrics.json').write_text('{}')
            (run/'inference_bundle.pt').write_bytes(b'private weights')
            (run/'train.npz').write_bytes(b'large arrays')
            env = dict(os.environ, BABEL_READOUT_RUN=str(run), BABEL_DOWNLOAD_DIR=str(root/'download'),
                       BABEL_READOUT_LOG=str(root/'missing.log'), PYTHON_BIN='/usr/bin/false')
            for mode, code in (('export', 0), ('all', 1)):
                result = subprocess.run(['bash', 'scripts/babel_readout_autodl.sh', mode], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as tar:
                    names = tar.getnames()
                    self.assertFalse(any(n.endswith(('.pt', '.npz')) for n in names))
                    status = tar.extractfile('run/run_status.txt').read().decode()
                    self.assertIn(f'command_exit_code={code}', status)
                    self.assertIn('run_status=failed' if code else 'run_status=partial', status)


if __name__ == '__main__':
    unittest.main()
