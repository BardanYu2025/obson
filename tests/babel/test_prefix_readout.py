"""Past-only labels, masked ridge, frozen provenance and probe recovery without neural updates."""
import copy
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from obson.babel import prefix_readout as pr, prefix_readout_benchmark as pb
from obson.babel import sampling_benchmark as sb, architecture as ar, architecture_benchmark as ab
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
import test_window_sampling as sampling_tests
from test_pca_anneal import no_update, read
from test_architecture_benchmark import features


def no_probe_update(model,data,batch,device,opt=None,seed=0):
    result,_=original_probe_epoch(model,data,batch,device,None,seed)
    return result,(len(data['x'])+batch-1)//batch if opt is not None else 0


original_probe_epoch=pr.run_epoch


class ReadoutMathTests(unittest.TestCase):
    def test_targets_reanchor_and_exclude_current_and_future(self):
        x=features(3);y,mask=ar.ordered_targets(x);stats=ar.fit_scales(x,y,mask)
        xx,yy,mm=ar.normalize(x,y,mask,stats);data=dict(x=xx,y=yy,mask=mm)
        for p in pr.PREFIXES:
            result,valid=pr.targets(data,stats,p)
            expected,ok=ar.ordered_targets(x[:,p-17:p])
            np.testing.assert_allclose(result,expected[:,:-1],atol=2e-6,rtol=2e-5)
            np.testing.assert_array_equal(valid,ok[:,:-1])
            changed={k:v.copy() for k,v in data.items()};changed['y'][:,p-1:]+=999;changed['x'][:,p-1:]+=99
            np.testing.assert_array_equal(pr.targets(changed,stats,p)[0],result)
            self.assertEqual(result.shape,(3,16,7))
        with self.assertRaises(ValueError):pr.targets(data,stats,16)
        bad=dict(data,mask=data['mask'].copy());bad['mask'][:,14,0]=False
        with self.assertRaisesRegex(ValueError,'anchors'):pr.targets(bad,stats,32)

    def test_masked_ridge_omits_invalid_labels_and_matches_direct_solution(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(80,4)).astype(np.float32)
        y=(x@rng.normal(size=(4,112))+rng.normal(size=112)).reshape(80,16,7).astype(np.float32)
        mask=np.ones_like(y,dtype=bool);mask[:20,:,4]=False
        fit=pr.ridge_candidates(x,y,mask,[.01],'cpu')[.01]
        corrupt=y.copy();corrupt[~mask]=1e9
        other=pr.ridge_candidates(x,corrupt,mask,[.01],'cpu')[.01]
        np.testing.assert_array_equal(fit['coef'],other['coef']);np.testing.assert_array_equal(fit['bias'],other['bias'])
        a=np.column_stack((x.astype(float),np.ones(80)));penalty=np.eye(5)*.01;penalty[-1,-1]=0
        for column in (0,4,111):
            ok=mask.reshape(80,-1)[:,column];z=a[ok];b=y.reshape(80,-1)[ok,column]
            expected=np.linalg.solve(z.T@z/len(z)+penalty,z.T@b/len(z))
            np.testing.assert_allclose(np.r_[fit['coef'][:,column],fit['bias'][column]],expected,atol=1e-10,rtol=1e-10)

    def test_probe_backward_and_paired_initialization(self):
        torch.set_num_threads(1);a=pr.Probe(512,42);b=pr.Probe(512,42)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(3,512);pred=a(x);y=torch.randn_like(pred);mask=torch.ones_like(y,dtype=torch.bool)
        pr.loss_rows(pred,y,mask)['primary'].mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))
        self.assertEqual(tuple(pred.shape),(3,16,7))

    def test_metric_empty_channel_and_amplitude(self):
        y=np.arange(16,dtype=np.float32)[None,:,None]*np.ones((2,16,7),np.float32);mask=np.ones_like(y,dtype=bool);mask[...,6]=False
        # Nonconstant differences make the amplitude/correlation diagnostic identifiable.
        y[...,0]=np.arange(16,dtype=np.float32)**2
        result,_=pr.measure(y*.5,y,mask,dict(scale=[1.]*7,delta_scale=1.))
        self.assertAlmostEqual(result['change1']['std_ratio'],.5)
        self.assertAlmostEqual(result['change1']['correlation'],1.)
        self.assertEqual(result['channels'][6],dict(support=0,mse=None,r2=None))


class ReadoutPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sampling_tests.SamplingPipelineTests.setUpClass();cls.temp=tempfile.TemporaryDirectory()
        helper=sampling_tests.SamplingPipelineTests();sm,cls.source=helper.make_run(cls.temp.name)
        with patch.object(sb,'run_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
            for j in sm['experiments']:sb.worker(cls.source,j['name'],'cpu')
        sb.evaluate(sm,cls.source,'cpu')
        files={p.name:sha256(p) for p in cls.source.iterdir() if p.is_file() and p.suffix=='.json'}
        for name in ('cache/index.json','candidates/index.json'):files[name]=sha256(cls.source/name)
        atomic_json(dict(status='complete',source_unchanged=True,files=files),cls.source/'completion.json')
        cls.identity=pb.source_identity(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup();sampling_tests.SamplingPipelineTests.tearDownClass()

    def make_run(self,td):
        out=Path(td)/'prefix';out.mkdir();meta=pb.make_manifest(self.source,self.identity,epochs=2,batch=4,hidden=8)
        atomic_json(meta,out/'manifest.json')
        for s in ('train','val'):pb.prepare_split(meta,out,s,'cpu')
        return meta,out

    def test_complete_pipeline_locks_before_research_and_keeps_encoders_frozen(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            loader=ab.load_arrays;seen=[];real_encoder=pb.encoder
            def check_load(root,split):self.assertIn(split,('train','val'));return loader(root,split)
            def spy(meta,name,device):
                model=real_encoder(meta,name,device)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                model.encoder.register_forward_pre_hook(lambda m,args:seen.append(args[0].shape[1]))
                return model
            with patch.object(ab,'load_arrays',side_effect=check_load),patch.object(pb,'encoder',side_effect=spy):meta,out=self.make_run(td)
            self.assertEqual(set(seen),{32,64,96,128})
            scales_hash=sha256(out/'cache/target_scales.json')
            with patch.object(pr,'run_epoch',side_effect=no_probe_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
                for job in meta['experiments']:pb.worker(out,job['name'],'cpu')
            def gated(root,split):
                if split in ('test','cross_research'):self.assertTrue((out/'selection_lock.json').exists())
                return loader(root,split)
            with patch.object(ab,'load_arrays',side_effect=gated),patch.object(pr,'fit_target_scales',side_effect=AssertionError('No test fitting')),patch.object(pr,'feature_scales',side_effect=AssertionError('No transfer/test refit')):
                report=pb.evaluate(meta,out,'cpu')
            self.assertEqual(sha256(out/'cache/target_scales.json'),scales_hash)
            self.assertEqual(len(report['selections']),20)
            self.assertNotIn('NaN',(out/'prefix_readout_metrics.json').read_text())
            terminal=next(j for j in meta['experiments'] if j['name']=='sampled_s42_p128')
            _,transfer=pb.predict_probe(meta,out,terminal,'test','ridge',32,'cpu')
            expected=pr.design(np.load(out/'cache/test_p32_sampled_s42.npy'),read(out/'sampled_s42_p128/feature_scales.json'))
            np.testing.assert_array_equal(transfer['x'],expected)
            for ds in report['datasets'].values():
                self.assertEqual(len(ds['scores']),94)
                self.assertIn('sampled_s42_p32/transfer128_ridge',ds['scores'])
                self.assertEqual(len(ds['paired']),96)
            for job in meta['experiments']:
                row=read(out/job['name']/'training_summary.json')
                self.assertEqual(row['encoder_updates'],0);self.assertEqual(row['optimizer_steps'],6);self.assertEqual(row['selected_epoch'],0)
            with patch.object(pr,'run_epoch',side_effect=AssertionError('No rerun')):pb.worker(out,meta['experiments'][0]['name'],'cpu')
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_research_cache_requires_lock(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td)
            with patch.object(ab,'load_arrays',side_effect=AssertionError('No research read')):
                with self.assertRaisesRegex(ValueError,'locked'):pb.prepare_split(meta,out,'test','cpu')

    def test_resume_retains_optimizer_and_data_order(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0];calls=[]
            def interrupted(model,data,batch,device,opt=None,seed=0):
                if opt is not None:
                    calls.append(seed)
                    if len(calls)==2:raise RuntimeError('interrupted')
                    p=opt.param_groups[0]['params'][0];opt.state[p]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,.12),exp_avg_sq=torch.full_like(p,.34))
                return no_probe_update(model,data,batch,device,opt,seed)
            with patch.object(pr,'run_epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):pb.worker(out,job['name'],'cpu')
            first=read(out/job['name']/'history.json')[0];restored=[];order=[];load=torch.optim.AdamW.load_state_dict
            def capture(self,state):restored.append(copy.deepcopy(state));return load(self,state)
            def resumed(model,data,batch,device,opt=None,seed=0):
                if opt is not None:order.append(seed)
                return no_probe_update(model,data,batch,device,opt,seed)
            with patch.object(pr,'run_epoch',side_effect=resumed),patch.object(torch.optim.AdamW,'load_state_dict',new=capture):pb.worker(out,job['name'],'cpu')
            self.assertEqual(order,[calls[-1]]);self.assertEqual(read(out/job['name']/'history.json')[0],first)
            moment=next(iter(restored[0]['state'].values()));self.assertTrue(torch.all(moment['exp_avg']==.12))
            self.assertEqual(read(out/job['name']/'resume_validation.json')['epoch'],1)

    def test_cache_and_budget_tamper_fail(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0]
            with patch.object(pr,'run_epoch',side_effect=no_probe_update):pb.worker(out,job['name'],'cpu')
            h=read(out/job['name']/'history.json')
            for k,v in [('lr',.1),('windows',13),('optimizer_steps',9)]:
                bad=copy.deepcopy(h);bad[0][k]=v
                with self.assertRaises(ValueError):pb.verify_history(bad,job,12,4)
            (out/'cache/train_p32_y.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):pb.worker(out,job['name'],'cpu')

    def test_export_on_failure_and_reexport(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download'
            atomic_json(dict(status='partial'),run/'metrics.json')
            for suffix in ('pt','npy','npz'):(run/f'data.{suffix}').write_bytes(b'weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGUMENT_LOG"\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,ARGUMENT_LOG=str(root/'arguments'),BABEL_PREFIX_RUN=str(run),BABEL_PREFIX_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            cmd=['bash',str(Path(__file__).resolve().parents[2]/'scripts/babel_prefix_readout_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                self.assertIn('obson.babel.prefix_readout_benchmark',(root/'arguments').read_text())
                with tarfile.open(download/'run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
