"""Causality, teacher isolation, lineage, resume and export; no neural optimizer updates."""
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
from obson.babel import pca_teacher as pt, pca_teacher_benchmark as tb
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features
import test_capacity_benchmark as capacity_tests

SMALL=dict(pt.CONFIG,latent=8,attention_layers=1,attention_ff=8,heads=2,decoder_width=4,residual_width=6)


def read(path):return json.loads(path.read_text())


def synthetic(n=12):
    x=features(n);y,mask=ar.ordered_targets(x);stats=ar.fit_scales(x,y,mask)
    x,y,mask=ar.normalize(x,y,mask,stats);data=dict(x=x,y=y,mask=mask)
    pca=ab.fit_pca(data,8);c=pt.coefficients(pca,data);scales=pt.fit_coordinate_scales(c)
    return dict(data,q=pt.normalize_coordinates(c,scales)),stats,pca,scales


class TeacherModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fixed_inverse_and_zero_residual(self):
        data,stats,pca,scales=synthetic();expected=ab.pca_predict(pca,data)
        for residual in (False,True):
            model=pt.Student(SMALL,42,pca,scales,residual)
            actual=model.decoder(torch.tensor(data['q'])).detach().numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=2e-5)
            self.assertTrue(all(not b.requires_grad for b in model.decoder.buffers()))
            with self.assertRaises(ValueError):model.decoder(torch.zeros(2,128,8))
        original=copy.deepcopy(scales);pt.normalize_coordinates(data['q']*100,scales)
        self.assertEqual(scales,original)

    def test_causal_states_and_gradient_through_fixed_basis(self):
        data,stats,pca,scales=synthetic();model=pt.Student(SMALL,42,pca,scales)
        x=torch.randn(2,128,28,requires_grad=True);h=model.encoder(x)
        changed=x.detach().clone();changed[:,50:]+=9
        torch.testing.assert_close(h[:,:50],model.encoder(changed)[:,:50],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(h[:,:50],model.encoder(x[:,:50]),atol=1e-6,rtol=1e-6)
        h[:,49].square().sum().backward();self.assertEqual(float(x.grad[:,50:].abs().max()),0.)
        model.zero_grad();z=model.encoder(torch.tensor(data['x']))[:,-1];pred=model.decoder(z)
        ar.error_rows(pred,torch.tensor(data['y']),torch.tensor(data['mask']),stats,True)['primary'].mean().backward()
        self.assertGreater(float(model.encoder.coordinates.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))

    def test_initialization_and_auxiliary_gradient(self):
        data,stats,pca,scales=synthetic();a=pt.Student(SMALL,42,pca,scales);b=pt.Student(SMALL,42,pca,scales,True)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        x=torch.tensor(data['x']);torch.testing.assert_close(a(x),b(x),atol=0,rtol=0)
        z=a.encoder(x)[:,-1];q=torch.tensor(data['q']);loss=pt.alignment_rows(z,q)['coordinate_smooth'].mean()
        loss.backward();self.assertGreater(float(a.encoder.coordinates.weight.grad.abs().sum()),0.)
        self.assertEqual(float(pt.alignment_rows(q,q)['coordinate_mse'].max()),0.)

    def test_microbatch_gradient_matches_full_batch_without_update(self):
        data,stats,pca,scales=synthetic();data={k:v[:7] for k,v in data.items()};initial=pt.Student(SMALL,42,pca,scales)
        gradients=[]
        with patch.object(torch.optim.AdamW,'step',return_value=None),patch.object(torch.nn.utils,'clip_grad_norm_',return_value=0.):
            for micro in (7,3):
                model=copy.deepcopy(initial);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
                _,updates=tb.run_epoch(model,data,stats,7,micro,'cpu',.25,opt)
                self.assertEqual(updates,1);gradients.append([p.grad.clone() for p in model.parameters() if p.requires_grad])
                for a,b in zip(model.parameters(),initial.parameters()):self.assertTrue(torch.equal(a,b))
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4)

    def test_teacher_schedule(self):
        job=dict(schedule='release',teacher_weight=.25)
        self.assertEqual([pt.teacher_weight(job,e,50) for e in (0,1,50,100)],[.25,.25,0.,0.])
        self.assertAlmostEqual(pt.teacher_weight(job,25,50),.25*(1-24/49))
        self.assertEqual(pt.teacher_weight(dict(schedule='constant',teacher_weight=0.),5,50),0.)


class TeacherPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capacity_tests.CapacityPipelineTests.setUpClass()
        cls.source=capacity_tests.CapacityPipelineTests.source;cls.identity=capacity_tests.CapacityPipelineTests.identity
        original=tb.run_epoch
        def no_update(model,data,stats,batch,micro,device,weight,opt=None,order_seed=0):
            return original(model,data,stats,batch,micro,device,weight,None,order_seed)
        cls.no_update=staticmethod(no_update)

    @classmethod
    def tearDownClass(cls):capacity_tests.CapacityPipelineTests.tearDownClass()

    def make_run(self,root,parent_epochs=1,branch_epochs=2):
        out=Path(root)/'teacher';out.mkdir()
        with patch.object(pt,'CONFIG',SMALL):meta=tb.make_manifest(self.source,self.identity,parent_epochs,branch_epochs,4,2)
        atomic_json(meta,out/'manifest.json');tb.prepare(meta,out);return meta,out

    def run_parents(self,meta,out):
        with patch.object(tb,'run_epoch',side_effect=self.no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
            for job in meta['experiments']:
                if job['phase']=='base':tb.worker(out,job['name'],'cpu')
        tb.lock_selection(meta,out,True)

    def test_complete_matrix_no_teacher_input_or_research_training(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            loader=ab.load_arrays
            def safe_load(path,split):self.assertIn(split,('train','val'));return loader(path,split)
            with patch.object(ab,'load_arrays',side_effect=safe_load):
                meta,out=self.make_run(td);tb.preflight(meta,out,'cpu');self.run_parents(meta,out)
                with patch.object(tb,'run_epoch',side_effect=self.no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No training')):
                    for job in meta['experiments']:
                        if job['phase']=='branch':tb.worker(out,job['name'],'cpu')
            self.assertEqual(len(meta['experiments']),14)
            for seed in (42,43):
                names=[f'teacher_{kind}_s{seed}' for kind in ('fixed','residual','release')]
                rows=[read(out/n/'branch_initialization.json') for n in names]
                self.assertEqual(rows[0],rows[1]);self.assertEqual(rows[1],rows[2])
                self.assertTrue(rows[0]['matched'])
            def eval_load(path,split):
                if split in ('test','cross_research'):self.assertTrue((out/'selection_lock.json').exists())
                return loader(path,split)
            with patch.object(ab,'load_arrays',side_effect=eval_load):report=tb.evaluate(meta,out,'cpu')
            for split in ('test','cross_research'):
                self.assertEqual(len(report['datasets'][split]['variants']),19)
                self.assertEqual(len(report['datasets'][split]['fixed_last']),14)
                self.assertEqual(len(report['datasets'][split]['paired']),32)
                self.assertIn('true_pca_coordinate_diagnostic',report['datasets'][split]['variants']['teacher_release_s42'])
            self.assertFalse(report['goal']['automatic_promotion']);self.assertTrue((out/'examples.html').exists())
            self.assertNotIn('NaN',(out/'teacher_metrics.json').read_text())
            with patch.object(tb,'run_epoch',side_effect=AssertionError('Completed job reran')):tb.worker(out,'teacher_release_s42','cpu')
            (out/'teacher_s42/best.pt').write_bytes(b'bad parent')
            with self.assertRaises(ValueError):tb.parent_lineage(meta,out,next(j for j in meta['experiments'] if j['name']=='teacher_release_s42'))
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_release_resume_and_missing_parent_guard(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td,1,3);job=next(j for j in meta['experiments'] if j['name']=='teacher_release_s42')
            with self.assertRaises(FileNotFoundError):tb.worker(out,job['name'],'cpu')
            self.run_parents(meta,out);calls=[]
            def interrupted(model,data,stats,batch,micro,device,weight,opt=None,order_seed=0):
                if opt is not None:
                    calls.append(weight)
                    if len(calls)==2:raise RuntimeError('synthetic interruption')
                return self.no_update(model,data,stats,batch,micro,device,weight)
            with patch.object(tb,'run_epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):tb.worker(out,job['name'],'cpu')
            first=read(out/job['name']/'history.json')[0];calls=[]
            def resumed(model,data,stats,batch,micro,device,weight,opt=None,order_seed=0):
                if opt is not None:calls.append(weight)
                return self.no_update(model,data,stats,batch,micro,device,weight)
            with patch.object(tb,'run_epoch',side_effect=resumed):tb.worker(out,job['name'],'cpu')
            self.assertEqual(calls,[.125,0.]);self.assertEqual(read(out/job['name']/'history.json')[0],first)
            self.assertEqual(read(out/job['name']/'resume_validation.json')['epoch'],1)
            with self.assertRaises(FileNotFoundError):tb.lock_selection(meta,out)

    def test_train_only_coordinate_scales_and_cache_guards(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);pca=tb.cb.source_pca(meta);data=ab.load_arrays(self.source,'train')
            self.assertEqual(read(out/'cache/coordinate_scales.json'),pt.fit_coordinate_scales(pt.coefficients(pca,data)))
            with patch.object(pt,'fit_coordinate_scales',side_effect=AssertionError('Refit cache')):tb.prepare(meta,out)
            (out/'cache/val_q.npy').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):tb.verify_cache(meta,out)
            with self.assertRaises(ValueError):tb.verify_code(dict(code_sha256={}))

    def test_export_failure_and_binary_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download'
            atomic_json(dict(status='partial'),run/'metrics.json');(run/'last.pt').write_bytes(b'binary')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_TEACHER_RUN=str(run),BABEL_TEACHER_LOG=str(root/'missing'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            repo=Path(__file__).resolve().parents[2];cmd=['bash',str(repo/'scripts/babel_pca_teacher_autodl.sh')]
            r=subprocess.run(cmd+['all'],env=env,capture_output=True,text=True);self.assertEqual(r.returncode,7,r.stderr)
            with tarfile.open(download/'run_reports.tar.gz') as f:
                self.assertFalse(any(n.endswith('.pt') for n in f.getnames()));self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())
            r=subprocess.run(cmd+['export'],env=env,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            with tarfile.open(download/'run_reports.tar.gz') as f:self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
