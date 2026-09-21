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

from test_babel import frame
from test_detail_alignment import fixture
from obson.babel import detail_alignment as da
from obson.babel import wick_ablation as wa
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import atomic_save, rng_state
from obson.babel.reconstruction_fusion import recent_loss


def wick_fixture(root):
    model,meta,out,data,scales,ref = fixture(root)
    for split in da.SPLITS: np.save(out/f'cache/{split}_sigma.npy',np.full((6,64),.1,np.float32))
    train = wa.arrays_for(meta,out,'train','cpu'); data = wa.arrays_for(meta,out,'val','cpu')
    stats = wa.fit_statistics(train['y'],train['mask'],train['sigma'])
    _,reference,_,_ = wa.measure(model,data,da.PackedStreams(out/'cache','val'),scales,stats,2)
    stats['reference'] = reference; (out/'wick_statistics.json').write_text(json.dumps(stats))
    meta['experiments'] = [dict(name=f'{mode}_s42',objective=mode,mode='joint_detail',seed=42) for mode in wa.MODES]
    (out/'manifest.json').write_text(json.dumps(meta))
    return model,meta,out,data,scales,stats


class WickAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_original_loss_and_gradient_equivalence_and_fixed_denominator(self):
        y = torch.randn(3,64,7)*.1; y[...,2:] = y[...,2:].abs()
        mask = torch.ones_like(y,dtype=torch.bool); mask[0,:,5] = False
        sigma = torch.ones(3,64)*.1; b = dict(y=y,mask=mask,sigma=sigma)
        stats = wa.fit_statistics(y,mask,sigma); scales = da.detail_scales(y.numpy())
        p = (y+torch.randn_like(y)*.01); p[...,2:] = p[...,2:].abs(); p.requires_grad_()
        old = recent_loss(p,y,mask)+.25*da.detail_values(p,y,scales)
        new = wa.objective(p,b,scales,stats,'original_loss')
        torch.testing.assert_close(old,new,atol=1e-7,rtol=1e-6)
        g_old = torch.autograd.grad(old.sum(),p,retain_graph=True)[0]
        g_new = torch.autograd.grad(new.sum(),p,retain_graph=True)[0]
        torch.testing.assert_close(g_old,g_new,atol=1e-7,rtol=1e-6)
        low = wa.objective(p,b,scales,stats,'wick_low'); g_low = torch.autograd.grad(low.sum(),p)[0]
        torch.testing.assert_close(g_low[...,[0,1,4,5,6]],g_old[...,[0,1,4,5,6]],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(g_low[...,2:4],g_old[...,2:4]*.2,atol=1e-7,rtol=1e-6)

    def test_coarse_interval_zero_inside_penalty_outside_and_no_extra_head(self):
        y = torch.zeros(2,64,7); y[...,1] = .1; y[...,2:4] = .05
        b = dict(y=y,mask=torch.ones_like(y,dtype=torch.bool),sigma=torch.ones(2,64)*.1)
        stats = dict(bins=[[.2,.8],[.2,.8],[.1,.5],[.1,.5]],scales=[1.,1.,1.,1.],body_deadband=.01)
        p = y.clone(); p[...,2:4] = .06
        self.assertEqual(float(wa.coarse_values(p,b,stats).sum()),0.)
        p[...,2:4] = .4; p.requires_grad_()
        loss = wa.coarse_values(p,b,stats).sum(); self.assertGreater(float(loss.detach()),0.)
        loss.backward(); self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()),0.)
        flat = torch.zeros_like(y); empty_bins = wa.fit_statistics(flat,b['mask'],b['sigma'])
        self.assertTrue(all(x==[] for x in empty_bins['bins']))
        self.assertTrue(torch.isfinite(wa.coarse_values(flat,dict(b,y=flat),empty_bins)).all())

    def test_train_only_bins_do_not_mutate_and_geometry_metrics(self):
        y = torch.rand(3,64,7); sigma = torch.ones(3,64); mask = torch.ones_like(y,dtype=torch.bool)
        stats = wa.fit_statistics(y,mask,sigma); before = json.dumps(stats,sort_keys=True)
        b = dict(y=y,mask=mask,sigma=sigma)
        sh = wa.shape_summary(y,b,[1.,1.,1.],stats)
        self.assertEqual(sh['body_mae_bps'],0.); self.assertEqual(sh['body_direction_accuracy'],1.)
        self.assertTrue(all(v['balanced_accuracy']==1. for v in sh['descriptor_classes'].values()))
        changed = dict(b,y=y*100)
        wa.shape_summary(y,changed,[1.,1.,1.],stats)
        self.assertEqual(before,json.dumps(stats,sort_keys=True))
        mask[...,5] = False
        self.assertIsNone(wa.shape_summary(y,b,[1.,1.,1.],stats)['log_oi_mae'])

    def test_sigma_alignment_and_future_prefix(self):
        f = frame(300); e = encode_context(f,60,'ema8_32')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); hashes = wa.save_sigmas(path,'train',np.array([[0,127],[0,255]]),[e])
            self.assertIn('train_sigma.npy',hashes); sigma = np.load(path/'train_sigma.npy')
            np.testing.assert_allclose(sigma[0],np.exp(e['x'][64:128,8])*100)
            np.testing.assert_allclose(sigma[1],np.exp(e['x'][192:256,8])*100)
            changed = f.copy(); changed.loc[256:,['open','high','low','close']] += 20
            ee = encode_context(changed,60,'ema8_32'); wa.save_sigmas(path,'train',np.array([[0,127],[0,255]]),[ee])
            np.testing.assert_array_equal(sigma,np.load(path/'train_sigma.npy'))

    def test_common_retention_does_not_gate_exact_wick_loss(self):
        ref = dict(close_bps=10.,body_mae_bps=5.,change16_mse=1.,coarse=.1,detail=.2,base=.3)
        self.assertTrue(wa.retention(dict(ref,base=99.),ref))
        for k,v in [('close_bps',10.3),('body_mae_bps',5.3),('change16_mse',1.06),('coarse',.111)]:
            self.assertFalse(wa.retention(dict(ref,**{k:v}),ref))

    def test_preflight_gradient_audit_report_and_completed_resume_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,meta,out,data,scales,stats = wick_fixture(Path(tmp))
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')):
                scores,_,_ = da.run_epoch(model,data,da.PackedStreams(out/'cache','val'),True,True,scales,2,
                    objective_fn=lambda p,b,parts: wa.objective(p,b,scales,stats,'wick_low'))
                self.assertIn('optimized_objective',scores)
                self.assertTrue(np.isfinite(scores['optimized_objective']))
                self.assertEqual(len(wa.preflight(meta,out,'cpu')['variants']),3)
                wa.gradient_audit(meta,out,'cpu')
                for job in meta['experiments']:
                    model.configure(True)
                    opt = torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr']),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=meta['encoder_lr'])],weight_decay=.01)
                    score,sel,_,_ = wa.measure(model,data,da.PackedStreams(out/'cache','val'),scales,stats,2)
                    state = dict(metadata=dict(manifest=meta,job=job),epoch=1,best_epoch=0,history=[],best_validation=score,best_selection=sel,
                        best_model=model.state_dict(),diagnostic_epoch=1,diagnostic_validation=score,diagnostic_selection=sel,diagnostic_model=model.state_dict(),
                        model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
                    path = out/job['name']; path.mkdir(); wa.publish(state,path)
                    wa.worker(out,job['name'],'cpu')
                    ck = torch.load(path/'best.pt',weights_only=True)
                    for k,v in model.state_dict().items(): torch.testing.assert_close(v,ck['model'][k],atol=0,rtol=0)
                wa.evaluate(out,'cpu')
            report = json.loads((out/'wick_metrics.json').read_text())
            self.assertEqual(len(report['variants']),4); self.assertEqual(len(report['paired_shape']),3)
            self.assertEqual(report['variants']['wick_coarse_s42']['diagnostic_candidate']['epoch'],1)
            self.assertTrue((out/'wick_low_s42/examples.html').exists())
            self.assertEqual(report['paired_shape']['wick_low_s42_minus_original_loss_s42']['body_mae_bps']['delta_mean'],0.)
            self.assertEqual(report['improved_candidates'],[])

    def test_export_excludes_weights_and_preserves_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            (run/'metrics.json').write_text('{}'); (run/'diagnostic_best.pt').write_bytes(b'weights'); (run/'sigma.npy').write_bytes(b'cache')
            env = dict(os.environ,BABEL_WICK_RUN=str(run),BABEL_WICK_LOG=str(root/'none.log'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                proc = subprocess.run(['bash','scripts/babel_wick_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(proc.returncode,code,proc.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertFalse(any(n.endswith(('.pt','.npy')) for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())
