import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from test_babel import frame, series
from test_detail_alignment import fixture
from obson.babel import detail_alignment as da
from obson.babel import ema_ablation as ea
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import atomic_save, rng_state


def ema_fixture(root):
    old,meta,out,data,scales,reference = fixture(root)
    rng = np.random.default_rng(47)
    for split in da.SPLITS:
        p = out/f'cache/{split}_x.npy'; x = np.load(p)
        np.save(p,np.column_stack((x,rng.normal(size=(len(x),5)).astype(np.float32))))
    meta['experiments'] = [dict(name=f'{mode}_s42',context=mode,mode='joint_detail',seed=42) for mode in ea.CONTEXT_MODES]
    (out/'manifest.json').write_text(json.dumps(meta))
    (out/'warm_start_validation.json').write_text('{}')
    return old,meta,out,data,scales,reference


class EmaAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_features_recursive_identity_prefix_and_window_continuity(self):
        f = frame(320); e = encode_context(f,60,'ema8_32'); bank = ea.feature_bank(f,e)
        np.testing.assert_array_equal(bank[:,:18],e['x'])
        p = np.log(f.close.to_numpy()); states = np.repeat(p[0],5); raw = []
        for value in p:
            states = states+2/(np.array([4,8,16,32,64])+1)*(value-states)
            raw.append(np.r_[value-states[0],states[:-1]-states[1:]])
        sigma = np.sqrt(np.maximum(np.r_[.01**2,e['variance'][:-1]],1e-8))
        np.testing.assert_allclose(bank[:,18:],np.arcsinh(np.array(raw)/sigma[:,None]),atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(np.array(raw).sum(1),p-pd.Series(p).ewm(span=64,adjust=False).mean().to_numpy(),atol=2e-14)
        changed = f.copy(); changed.loc[200:,'close'] += 10; changed.loc[200:,'high'] += 10
        other = ea.feature_bank(changed,encode_context(changed,60,'ema8_32'))
        np.testing.assert_array_equal(bank[:200],other[:200])
        prefix = ea.feature_bank(f.iloc[:200],encode_context(f.iloc[:200],60,'ema8_32'))
        np.testing.assert_array_equal(bank[:200],prefix)
        # A wrongly window-reset EMA would be zero at this first position.
        self.assertGreater(float(np.abs(bank[128,18:]).sum()),0.)
        self.assertFalse(np.any(ea.select_features(bank,'none')[:,14:]))
        with self.assertRaises(ValueError): ea.select_features(bank,'bad')

    def test_neutral_initial_outputs_equal_retained_reproduces_old_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,meta,out,data,scales,ref = ema_fixture(Path(tmp))
            zs = []; counts = []
            for mode in ea.CONTEXT_MODES:
                job = dict(context=mode); m = ea.initial_model(meta,'cpu',job).eval()
                counts.append(sum(p.numel() for p in m.parameters()))
                score,z,_ = da.run_epoch(m,data,ea.ContextStreams(out/'cache','val',job),True,True,scales,2,collect=True)
                if mode != 'retained_ema8_32': zs.append(z)
                else: np.testing.assert_allclose(z,data['z'].numpy(),atol=2e-5,rtol=2e-5)
            self.assertEqual(len(set(counts)),1)
            for z in zs[1:]: np.testing.assert_array_equal(z,zs[0])
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')):
                result = ea.preflight(meta,out,'cpu')
            self.assertEqual(result['variants']['none']['context_gradient_norm'],0.)
            self.assertGreater(result['variants']['multiscale']['context_gradient_norm'],0.)

    def test_prepare_original_replay_neutral_audit_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,meta,out,data,scales,ref = ema_fixture(Path(tmp))
            ss = series(1152); encoded = [encode_context(x.frame,x.period,'ema8_32') for x in ss]
            bounds = dict(train_until=str(ss[0].sessions[383]),val_until=str(ss[0].sessions[767]),test_until=str(ss[0].sessions[-1]))
            meta.update(root='synthetic',long_run='synthetic')
            for split,lo in zip(da.SPLITS,(0,384,768)):
                keys = np.array([[i,lo+j] for i in range(2) for j in (127,255,383)])
                np.save(Path(meta['source'])/f'target_cache/{split}_keys.npy',keys)
                with torch.no_grad():
                    z = np.concatenate([old.encoder(torch.tensor(e['x'][lo:lo+384])[None])[0][0,[127,255,383]].numpy() for e in encoded])
                np.save(Path(meta['source'])/f'target_cache/{split}_short.npy',z)
            with patch.object(ea,'load_data',return_value=(dict(boundaries=bounds),ss,encoded,None)):
                ea.prepare(meta,out,'cpu')
            audit = json.loads((out/'warm_start_validation.json').read_text())
            self.assertTrue(audit['variants']['retained_ema8_32']['cached_comparison']['passed'])
            self.assertTrue(all(audit['variants'][m]['identical_neutral_output'] for m in ea.CONTEXT_MODES[:3]))
            with patch.object(ea,'load_data',side_effect=AssertionError('cache should be reused')): ea.prepare(meta,out,'cpu')
            (out/'cache/val_sequences.json').write_text('[]')
            with self.assertRaises(ValueError): ea.prepare(meta,out,'cpu')

    def test_first_eligible_candidate_replaces_rejected_initial(self):
        ref = dict(base=1.,detail=1.,close_bps=10.,change16_mse=1.)
        rejected = dict(ref,detail=.5,close_bps=20.)
        acceptable = dict(ref,detail=.9)
        self.assertTrue(da.improves(acceptable,rejected,ref))
        self.assertFalse(da.improves(rejected,acceptable,ref))

    def test_completed_context_resume_preserves_weights_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,meta,out,data,scales,ref = ema_fixture(Path(tmp)); job = meta['experiments'][2]
            model = ea.initial_model(meta,'cpu',job); model.configure(True)
            opt = torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr']),
                dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=meta['encoder_lr'])],weight_decay=.01)
            score = da.run_epoch(model,data,ea.ContextStreams(out/'cache','val',job),True,True,scales,2)[0]
            path = out/job['name']; path.mkdir(); state = dict(metadata=dict(manifest=meta,job=job),epoch=1,best_epoch=0,history=[],
                best_validation=score,model=model.state_dict(),best_model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
            atomic_save(state,path/'last.pt')
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')):
                da.worker(out,job['name'],'cpu',model_factory=ea.initial_model,streams_factory=ea.ContextStreams,allow_initial_regression=True)
            ck = torch.load(path/'best.pt',weights_only=True)
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,ck['model'][k],rtol=0,atol=0)
            self.assertEqual(ck['metadata']['job']['context'],'multiscale')

    def test_full_context_report_and_rejected_candidate_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,meta,out,data,scales,ref = ema_fixture(Path(tmp))
            # Force every candidate to fail retention; evaluation must remain explicit.
            aux = json.loads((out/'cache/statistics.json').read_text()); aux['reference']['close_bps'] = 1e-9
            (out/'cache/statistics.json').write_text(json.dumps(aux))
            for job in meta['experiments']:
                model = ea.initial_model(meta,'cpu',job).eval(); path = out/job['name']; path.mkdir()
                score = da.run_epoch(model,data,ea.ContextStreams(out/'cache','val',job),True,True,scales,2)[0]
                atomic_save(dict(metadata=dict(manifest=meta,job=job),epoch=0,model=model.state_dict(),validation=score),path/'best.pt')
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')): ea.evaluate(out,'cpu')
            report = json.loads((out/'ema_metrics.json').read_text())
            self.assertEqual(len(report['variants']),5); self.assertEqual(len(report['paired']),5)
            self.assertEqual(report['accepted_candidates'],[])
            self.assertEqual(report['paired']['multiscale_s42_minus_none_s42']['normalized_detail']['delta_mean'],0.)
            self.assertTrue((out/'multiscale_s42/examples.html').exists())
            self.assertIn('False',(out/'summary.md').read_text())

    def test_export_completion_and_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            (run/'metrics.json').write_text('{}'); (run/'best.pt').write_bytes(b'weights'); (run/'bank.npy').write_bytes(b'cache')
            env = dict(os.environ,BABEL_EMA_RUN=str(run),BABEL_EMA_LOG=str(root/'none.log'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                proc = subprocess.run(['bash','scripts/babel_ema_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(proc.returncode,code,proc.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertFalse(any(n.endswith(('.pt','.npy')) for n in t.getnames()))
                    status = t.extractfile('run/run_status.txt').read().decode()
                    self.assertIn(f'command_exit_code={code}',status)
                    self.assertIn('run_status=failed' if code else 'run_status=partial',status)
