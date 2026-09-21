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
from obson.babel import activity_ablation as aa
from obson.babel import detail_alignment as da
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import atomic_save, rng_state


def activity_fixture(root):
    old,meta,out,data,scales,reference = fixture(root); rng = np.random.default_rng(47)
    for split in da.SPLITS:
        p = out/f'cache/{split}_x.npy'; x = np.load(p); extra = rng.normal(size=(len(x),10)).astype(np.float32)
        extra[:,5:] = 1
        np.save(p,np.column_stack((x,extra)))
        target = rng.normal(size=(6,10)).astype(np.float32)
        np.save(out/f'cache/{split}_activity.npy',target); np.save(out/f'cache/{split}_activity_mask.npy',np.ones_like(target,dtype=bool))
    meta['experiments'] = [dict(name=f'{mode}_s42',activity=mode,mode='joint_detail',seed=42) for mode in aa.MODES]
    (out/'manifest.json').write_text(json.dumps(meta)); (out/'warm_start_validation.json').write_text('{}')
    return old,meta,out,data,scales,reference


class ActivityAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_activity_open_priority_fallback_gaps_missing_and_zero_volume(self):
        f = frame(8); f['datetime'] = pd.date_range('2025-01-01',periods=8,freq='h')
        f['oi'] = np.arange(8,dtype=float)+110; f['oi_available'] = True; f['volume'] = 10.
        f['open_oi'] = [100.,np.nan,0.,100.,100.,100.,100.,100.]
        f.loc[2:,'datetime'] += pd.Timedelta(hours=1); f.loc[3,'volume'] = 0.; f.loc[4,'oi'] = 0.
        x,a = aa.activity_features(f,60)
        self.assertAlmostEqual(float(x[0,2]),float(np.arcsinh(10.)),places=6)
        self.assertEqual(x[0,8],1.); self.assertEqual(x[1,8],0.); self.assertEqual(x[1,5],1.)
        self.assertEqual(x[2,5],0.); self.assertEqual(x[3,5],1.); self.assertEqual(x[3,6],0.); self.assertEqual(x[3,3],0.)
        self.assertEqual(x[4,5],0.); self.assertEqual(a['previous_close_fallback'],1); self.assertTrue(np.isfinite(x).all())
        f['oi_available'] = False; y,_ = aa.activity_features(f,60); self.assertFalse(np.any(y[:,2:9]))

    def test_prefix_causality_and_probe_masks(self):
        f = frame(80); x,_ = aa.activity_features(f,60)
        changed = f.copy(); changed.loc[40:,'volume'] *= 100; changed.loc[40:,'oi'] *= 2
        y,_ = aa.activity_features(changed,60); np.testing.assert_array_equal(x[:40],y[:40])
        z,_ = aa.activity_features(f.iloc[:40],60); np.testing.assert_array_equal(x[:40],z)
        target,mask = aa.probe_targets([x],np.array([[0,31],[0,63]]))
        np.testing.assert_array_equal(target[0,:5],x[31,:5]); self.assertFalse(mask[0,7]) # overnight fallback gaps
        self.assertTrue(mask[0,5]); self.assertEqual(target.shape,(2,10))

    def test_matched_initial_output_rng_parameter_budget_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,meta,out,data,scales,_ = activity_fixture(Path(tmp)); counts=[]; randoms=[]
            for mode in aa.MODES:
                torch.manual_seed(123); job=dict(activity=mode); m=aa.initial_model(meta,'cpu',job).eval(); randoms.append(torch.rand(8))
                _,z,_=da.run_epoch(m,data,aa.ActivityStreams(out/'cache','val',job),True,True,scales,2,collect=True)
                np.testing.assert_allclose(z,data['z'].numpy(),atol=2e-5,rtol=2e-5); counts.append(sum(p.numel() for p in m.parameters()))
            self.assertEqual(counts[0],counts[1]); self.assertEqual(counts[2]-counts[1],2)
            for x in randoms[1:]: torch.testing.assert_close(x,randoms[0],atol=0,rtol=0)
            self.assertEqual(sorted(aa.PRICE_IDS+aa.ACTIVITY_IDS),list(range(28)))
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')):
                self.assertEqual(len(aa.preflight(meta,out,'cpu')['variants']),3)

    def test_ridge_selection_never_uses_test_and_missing_support(self):
        rng=np.random.default_rng(3); zs=[rng.normal(size=(n,6)) for n in (80,30,30)]; w=rng.normal(size=(6,10))
        ys=[(x@w).astype(np.float32) for x in zs]; masks=[np.ones(y.shape,bool) for y in ys]
        for m in masks:m[:,9]=False
        report,errors=aa.fit_activity_probe(zs,ys,masks)
        self.assertGreater(report['targets'][0]['test_r2'],.95); self.assertIn('skipped',report['targets'][9])
        altered=[ys[0],ys[1],ys[2]*100+70]; other,_=aa.fit_activity_probe(zs,altered,masks)
        for a,b in zip(report['targets'][:9],other['targets'][:9]):
            self.assertEqual(a['alpha'],b['alpha']); self.assertEqual(a['validation_normalized_mse'],b['validation_normalized_mse'])
        self.assertTrue(np.isnan(errors[:,9]).all())

    def test_prepare_data_identity_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,meta,out,data,scales,_=activity_fixture(Path(tmp)); ss=series(1152)
            for s in ss:s.frame['open_oi']=s.frame.oi.shift(1).fillna(s.frame.oi.iloc[0])
            encoded=[encode_context(s.frame,s.period,'ema8_32') for s in ss]
            bounds=dict(train_until=str(ss[0].sessions[383]),val_until=str(ss[0].sessions[767]),test_until=str(ss[0].sessions[-1]))
            meta.update(root='synthetic',long_run='synthetic')
            for split,lo in zip(da.SPLITS,(0,384,768)):
                keys=np.array([[i,lo+j] for i in range(2) for j in (127,255,383)])
                np.save(Path(meta['source'])/f'target_cache/{split}_keys.npy',keys)
                with torch.no_grad():z=np.concatenate([old.encoder(torch.tensor(e['x'][lo:lo+384])[None])[0][0,[127,255,383]].numpy() for e in encoded])
                np.save(Path(meta['source'])/f'target_cache/{split}_short.npy',z)
            with patch.object(aa,'load_data',return_value=(dict(boundaries=bounds),ss,encoded,None)):aa.prepare(meta,out,'cpu')
            check=json.loads((out/'warm_start_validation.json').read_text()); self.assertTrue(all(x['comparison']['passed'] for x in check['variants'].values()))
            with patch.object(aa,'load_data',side_effect=AssertionError('reuse cache')):aa.prepare(meta,out,'cpu')
            (out/'cache/activity_audit.json').write_text('{}')
            with self.assertRaises(ValueError):aa.prepare(meta,out,'cpu')

    def test_completed_resume_diagnostic_and_full_report_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,meta,out,data,scales,_=activity_fixture(Path(tmp))
            for job in meta['experiments']:
                model=aa.initial_model(meta,'cpu',job); model.configure(True)
                score=da.run_epoch(model,data,aa.ActivityStreams(out/'cache','val',job),True,True,scales,2)[0]
                opt=torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr']),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=meta['encoder_lr'])],weight_decay=.01)
                path=out/job['name']; path.mkdir()
                state=dict(metadata=dict(manifest=meta,job=job),epoch=1,best_epoch=0,history=[],best_validation=score,best_model=model.state_dict(),
                           model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
                atomic_save(state,path/'last.pt')
                with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')):
                    da.worker(out,job['name'],'cpu',model_factory=aa.initial_model,streams_factory=aa.ActivityStreams,track_diagnostic=True)
                ck=torch.load(path/'diagnostic_best.pt',weights_only=True)
                for k,v in model.state_dict().items():torch.testing.assert_close(v,ck['model'][k],atol=0,rtol=0)
            aa.evaluate(out,'cpu'); report=json.loads((out/'activity_metrics.json').read_text())
            self.assertEqual(len(report['variants']),4); self.assertEqual(len(report['paired_activity']),3)
            self.assertEqual(report['variants']['branches_s42']['branch_gates'],[0.,0.])
            self.assertIn('body_mae_bps',report['variants']['features_s42']['candle_activity_reconstruction'])
            diag=aa.reconstruction_diagnostics(data['y'].numpy(),data)
            self.assertEqual(diag['body_mae_bps']['mae'],0.)
            self.assertEqual(diag['encoded_volume_change1']['mae'],0.)
            self.assertEqual(report['variants']['features_s42']['diagnostic_candidate']['epoch'],0)
            self.assertTrue((out/'branches_s42/examples.html').exists())

    def test_export_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(run/'metrics.json').write_text('{}');(run/'weights.pt').write_bytes(b'weights');(run/'cache.npy').write_bytes(b'cache')
            env=dict(os.environ,BABEL_ACTIVITY_RUN=str(run),BABEL_ACTIVITY_LOG=str(root/'none.log'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                proc=subprocess.run(['bash','scripts/babel_activity_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(proc.returncode,code,proc.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertFalse(any(n.endswith(('.pt','.npy')) for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())
