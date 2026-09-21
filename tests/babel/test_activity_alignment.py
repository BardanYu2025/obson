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

from test_activity_ablation import activity_fixture
from obson.babel import activity_alignment as al
from obson.babel import activity_ablation as aa
from obson.babel import detail_alignment as da
from obson.babel.ae_extend import atomic_save, rng_state
from obson.babel.dual_state import sha256


def fixture(root):
    old, meta, upstream, data, scales, ref = activity_fixture(root)
    for split in da.SPLITS:
        path=upstream/f'cache/{split}_x.npy'; x=np.load(path); x[:,21]=np.tanh(x[:,21])*.5; x[:,22]=np.abs(x[:,22])+.1; np.save(path,x)
    (upstream/'cache/activity_audit.json').write_text('{}'); (upstream/'coverage.json').write_text('{}')
    files = {p.name:sha256(p) for p in (upstream/'cache').iterdir() if p.name!='index.json'}
    (upstream/'cache/index.json').write_text(json.dumps(dict(manifest=meta,files=files)))
    out = root/'align'; out.mkdir()
    meta.update(activity_run=str(upstream),probe_epochs=1,
        experiments=[dict(name=f'{mode}_s42',activity='features',mode='joint_detail',seed=42,aux_weight=w) for mode,w in al.MODES.items()])
    (out/'manifest.json').write_text(json.dumps(meta)); al.prepare(meta,out)
    return old,meta,out,data,scales,ref


class ActivityAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_past_only_and_causal_masks_do_not_cross_contracts(self):
        x=np.zeros((80,28),np.float32); a=x[:,18:];a[:,:5]=np.arange(80)[:,None]/100;a[:,5:]=1
        specs=[dict(offset=20,length=40,endpoints=[[20,0]])]
        y,m,c,now=al.targets_from_bank(x,specs,1)
        np.testing.assert_allclose(y[0,5:10],x[24:40,18:23].mean(0))
        np.testing.assert_array_equal(y[0,10:15],x[36,18:23]);np.testing.assert_array_equal(y[0,15:],x[24,18:23])
        changed=x.copy();changed[41:,:]=999;yy,mm,_,_=al.targets_from_bank(changed,specs,1)
        np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(m,mm)
        changed=x.copy();changed[40,18]=999;yy,_,_,_=al.targets_from_bank(changed,specs,1)
        np.testing.assert_array_equal(y[:,5:],yy[:,5:])
        x[30,21]=2.; _,m,c,_=al.targets_from_bank(x,specs,1)
        self.assertFalse(c[0]);self.assertFalse(m[0,7]);self.assertFalse(m[0,8]);self.assertTrue(m[0,2])
        with self.assertRaisesRegex(ValueError,'Invalid'):al.targets_from_bank(x,[dict(offset=20,length=40,endpoints=[[15,0]])],1)
        with self.assertRaisesRegex(ValueError,'Incomplete'):al.targets_from_bank(x,specs,2)

    def test_zero_volume_anomaly_and_masked_loss(self):
        x=np.zeros((40,28),np.float32);x[:,23:]=1;x[:,22]=1;x[20,22]=0;x[20,20]=.2
        y,m,c,_=al.targets_from_bank(x,[dict(offset=0,length=40,endpoints=[[20,0]])],1)
        self.assertFalse(m[0,2]);self.assertFalse(m[0,3]);self.assertFalse(c[0])
        pred=torch.zeros(1,20,requires_grad=True);target=torch.ones(1,20);mask=torch.ones(1,20,dtype=torch.bool);mask[:,5:10]=False
        loss=al.activity_loss(pred,target,mask);self.assertAlmostEqual(loss.item(),.375)
        loss.sum().backward();self.assertEqual(pred.grad[:,5:10].abs().sum().item(),0)

    def test_stats_ignore_masked_outliers_and_error_support(self):
        y=np.arange(80,dtype=float).reshape(4,20);mask=np.ones_like(y,bool);mask[0,0]=False
        s=al.fit_target_stats(y,mask);y[0,0]=1e12;self.assertEqual(s,al.fit_target_stats(y,mask))
        normalized=(y-np.array(s['mean']))/np.array(s['scale'])
        report,err=al.score_probe(normalized,y,mask,s)
        self.assertEqual(report['targets'][0]['test_normalized_mse'],0);self.assertTrue(np.isnan(err[0,0]))
        self.assertEqual(al.error_summary(err,np.array([False,True,True,True]))['clean']['support'][0],3)

    def test_cache_reuse_hash_and_equal_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,meta,out,data,scales,_=fixture(Path(tmp))
            self.assertTrue((out/'cache/train_x.npy').is_symlink());al.prepare(meta,out)
            outputs=[];rngs=[]
            for job in meta['experiments']:
                torch.manual_seed(123);model=al.initial_model(meta,'cpu',job).eval();rngs.append(torch.rand(3))
                outputs.append(da.run_epoch(model,data,aa.ActivityStreams(out/'cache','val',job),True,True,scales,2,collect=True)[1])
            for z in outputs:np.testing.assert_allclose(z,data['z'].numpy(),atol=2e-5,rtol=2e-5)
            for r in rngs[1:]:torch.testing.assert_close(r,rngs[0],atol=0,rtol=0)
            (out/'cache/train_activity.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):al.prepare(meta,out)

    def test_aux_gradient_reaches_encoder_and_zero_weight_leaves_price_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,meta,out,_,_,_=fixture(Path(tmp));job=meta['experiments'][1];model=al.initial_model(meta,'cpu',job);model.configure(True)
            x=torch.randn(2,32,28);z=model.encoder(x)[0][:,-1]
            b=dict(activity=torch.randn(2,20),activity_mask=torch.ones(2,20,dtype=torch.bool))
            parts={};al.latent_objective(model,.05)(z,b,parts).sum().backward()
            self.assertGreater(model.encoder.input[0].weight.grad.abs().sum().item(),0)
            model.zero_grad(set_to_none=True);z=model.encoder(x)[0][:,-1]
            al.latent_objective(model,0)(z,b,{}).sum().backward()
            self.assertEqual(model.encoder.input[0].weight.grad.abs().sum().item(),0)
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                self.assertTrue(np.isfinite(al.preflight(meta,out,'cpu')['gradient_norm']))

    def test_completed_resume_and_full_report_without_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,meta,out,_,scales,_=fixture(Path(tmp))
            for job in meta['experiments']:
                model=al.initial_model(meta,'cpu',job);model.configure(True)
                score=da.run_epoch(model,al.arrays_for(meta,out,'val','cpu'),aa.ActivityStreams(out/'cache','val',job),True,True,scales,2,latent_objective_fn=al.latent_objective(model,job['aux_weight']))[0]
                opt=torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr']),dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=meta['encoder_lr']),dict(params=model.activity_head.parameters(),lr=meta['decoder_lr'])],weight_decay=.01)
                path=out/job['name'];path.mkdir()
                state=dict(metadata=dict(manifest=meta,job=job),epoch=1,best_epoch=0,history=[],best_validation=score,best_model=model.state_dict(),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
                atomic_save(state,path/'last.pt')
                with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):al.worker(out,job['name'],'cpu')
            with patch.object(al,'nonlinear_probe',side_effect=lambda z,y,m,*args:al.ridge_probe(z,y,m)),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                al.evaluate(out,'cpu')
            r=json.loads((out/'alignment_metrics.json').read_text());self.assertEqual(len(r['variants']),4)
            self.assertEqual(len(r['variants']['aux005_s42']['linear_activity_readout']['names']),20)
            self.assertFalse(r['stage_decisions']['aux005']['evidence_pass']);self.assertIn('current_input_control',r)
            self.assertNotIn('trained_auxiliary_head',r['variants']['control_s42'])

    def test_gpu_only_nonlinear_probe_and_paired_support(self):
        with self.assertRaisesRegex(ValueError,'CUDA'):al.nonlinear_probe([],[],[],'cpu')
        a=np.ones((6,20));b=a*2;a[0,7]=np.nan
        r=al.paired_errors(a,b,np.arange(6),np.ones(6,bool))
        self.assertEqual(r['all']['past_only_primary']['support'],5)
        self.assertLess(r['all']['past_only_primary']['high'],0)

    def test_upstream_identity_rejects_changed_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'activity';(run/'cache').mkdir(parents=True)
            meta=dict(source='source',short_run='short',long_run='long',sources={'digest':'original'})
            (run/'manifest.json').write_text(json.dumps(meta));(run/'completion.json').write_text('{"status":"complete"}')
            (run/'cache/one.json').write_text('{}')
            (run/'cache/index.json').write_text(json.dumps(dict(manifest=meta,files={'one.json':sha256(run/'cache/one.json')})))
            with patch.object(al,'identity_for',return_value={'digest':'original'}):
                actual,identity=al.source_identity(run,root/'fusion');self.assertEqual(actual,meta)
                self.assertEqual(identity['source_weights'],meta['sources'])
            with patch.object(al,'identity_for',return_value={'digest':'changed'}):
                with self.assertRaisesRegex(ValueError,'artifacts changed'):al.source_identity(run,root/'fusion')

    def test_export_success_and_failure_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(run/'metrics.json').write_text('{}');(run/'weights.pt').write_bytes(b'x');(run/'cache.npy').write_bytes(b'x')
            env=dict(os.environ,BABEL_ALIGNMENT_RUN=str(run),BABEL_ALIGNMENT_LOG=str(root/'none.log'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                p=subprocess.run(['bash','scripts/babel_activity_alignment_autodl.sh',mode],env=env,capture_output=True,text=True);self.assertEqual(p.returncode,code,p.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertFalse(any(n.endswith(('.pt','.npy')) for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())
