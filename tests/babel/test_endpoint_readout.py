"""Frozen routes, own predicted anchors, trained causal checks and immutable resume."""
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from obson.babel import endpoint_readout as er, endpoint_readout_audit as ea
from obson.babel import bar_alignment as ba, bar_alignment_benchmark as bb
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_pca_anneal import read
import test_bar_alignment as fixture


class EndpointMathTests(unittest.TestCase):
    def test_crop_uses_predicted_anchor_matches_target_and_excludes_current(self):
        _,data,stats,local=fixture.small_model();y=torch.tensor(data['y']);mask=torch.tensor(data['mask'])
        ps=torch.full((4,1),128);expected,valid=ba.local_targets(y,mask,ps,stats,local)
        got=er.crop_prediction(y,stats,local)
        torch.testing.assert_close(got[valid[:,0]],expected[:,0][valid[:,0]],atol=0,rtol=0)
        # Changing only the model-predicted anchor must shift every local close.
        pred=y.clone();pred[:,110,0]+=3.
        changed=er.crop_prediction(pred,stats,local)
        shift=3*stats['y_scale'][0]/local['scale'][0]
        torch.testing.assert_close(changed[...,0],got[...,0]-shift,atol=3e-6,rtol=1e-6)
        torch.testing.assert_close(changed[...,1:],got[...,1:],atol=0,rtol=0)
        # A whole-path offset cancels, while row127/current never enters the crop.
        pred=y.clone();pred[...,0]+=3.;pred[:,127]+=10000
        torch.testing.assert_close(er.crop_prediction(pred,stats,local),got,atol=3e-6,rtol=1e-6)
        with self.assertRaises(ValueError):er.crop_prediction(y[:,:127],stats,local)

    def test_causality_future_append_and_reset(self):
        model,data,_,_=fixture.small_model();x=torch.tensor(data['x']);before=bb.state_signature(model)
        result=er.trained_causality(model,x)
        self.assertEqual(result['status'],'passed');self.assertIn('p128/appended_future_global',result['fp32']['checks'])
        self.assertFalse(result['fp32']['reset_context']['equality_required'])
        self.assertEqual(before,bb.state_signature(model))

    def test_noncausal_encoder_is_rejected_even_with_double_replay(self):
        model,data,_,_=fixture.small_model()
        class Leaky(torch.nn.Module):
            def __init__(self,encoder):super().__init__();self.encoder=encoder
            def forward(self,x):return self.encoder(x)+x.mean((1,2))[:,None,None]
        model.core.encoder=Leaky(model.core.encoder)
        result=er.trained_causality(model,torch.tensor(data['x']))
        self.assertEqual(result['status'],'failed');self.assertFalse(result['fp64']['passed'])
        self.assertFalse(result['fp32']['checks']['p128/appended_future_state']['passed'])

    def test_fp32_drift_is_reported_without_relaxing_tolerance(self):
        model,data,_,_=fixture.small_model();before=bb.state_signature(model)
        with patch.object(er,'causal_checks',side_effect=[dict(passed=False),dict(passed=True)]) as check:
            result=er.trained_causality(model,torch.tensor(data['x']))
        self.assertEqual(result['status'],'passed_with_fp32_drift')
        self.assertEqual(check.call_args_list[1].args[1].dtype,torch.float64)
        self.assertEqual(check.call_args_list[1].args[2],er.TOLERANCES['fp64'])
        self.assertEqual(before,bb.state_signature(model))

    def test_production_width_causal_checks_without_training(self):
        _,data,_,_=fixture.small_model()
        for width in (512,768):
            pca=dict(components=np.eye(896,dtype=np.float32)[:width],mean=np.zeros(896,dtype=np.float32))
            scales=dict(mean=[0.]*width,scale=[1.]*width)
            model=ba.AlignedStudent(dict(bb.pt.CONFIG,latent=width),42,pca,scales).eval()
            result=er.trained_causality(model,torch.tensor(data['x'][:2]))
            self.assertEqual(result['status'],'passed')


class EndpointPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AlignmentPipelineTests.setUpClass();cls.temp=tempfile.TemporaryDirectory()
        helper=fixture.AlignmentPipelineTests();meta,cls.source=helper.make_run(cls.temp.name)
        with patch.object(ba,'run_epoch',side_effect=fixture.no_alignment_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local neural training')):
            for job in meta['experiments']:bb.worker(cls.source,job['name'],'cpu')
        bb.evaluate(meta,cls.source,'cpu')
        atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__),cls.source/'runtime.json')
        files={p.name:sha256(p) for p in cls.source.iterdir() if p.is_file() and p.suffix in ('.json','.md')}
        files['cache/index.json']=sha256(cls.source/'cache/index.json')
        atomic_json(dict(status='complete',cells=8,source_unchanged=True,encoder_steps=48,head_steps=48,files=files),cls.source/'completion.json')
        cls.identity=ea.source_identity(cls.source)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup();fixture.AlignmentPipelineTests.tearDownClass()

    def test_complete_audit_without_fitting_source_mutation_or_resume_inference(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'audit'
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer')),patch.object(bb.ab,'fit_pca',side_effect=AssertionError('No fitting')):
                ea.run(self.source,out,'cpu',batch=2)
            done=read(out/'completion.json');self.assertEqual(done['encoder_updates'],0);self.assertEqual(done['head_updates'],0)
            report=read(out/'endpoint_metrics.json');self.assertEqual(len(report['trials']),16)
            self.assertEqual(report['source_decision'],read(self.source/'decision.json'));self.assertFalse(report['automatic_promotion'])
            for trial in report['trials'].values():
                self.assertTrue(trial['weights_unchanged'])
                for row in trial['datasets'].values():
                    self.assertTrue(row['global_and_local_source_replay'])
                    self.assertEqual(set(row['scores']),{'local_head','global_crop'})
            self.assertIn('predicted anchor',(out/'examples.html').read_text())
            with patch.object(er,'predict',side_effect=AssertionError('Resume must not infer')):ea.run(self.source,out,'cpu',batch=2)
            target=out/'trials'/next(iter(report['trials']))/'metrics.json';target.write_text('{}')
            with self.assertRaises(ValueError):ea.run(self.source,out,'cpu',batch=2)
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_source_and_checkpoint_tamper_rejected(self):
        runtime=self.source/'runtime.json';old=runtime.read_bytes()
        try:
            runtime.write_text('{}')
            with self.assertRaises(ValueError):ea.source_identity(self.source)
        finally:runtime.write_bytes(old)
        meta=ea.make_manifest(self.source,self.identity);job=self.identity['manifest']['experiments'][0]
        ck=torch.load(self.source/job['name']/'best.pt',map_location='cpu',weights_only=True);ck['epoch']=999
        with patch.object(torch,'load',return_value=ck):
            with self.assertRaisesRegex(ValueError,'Best checkpoint'):ea.load_model(meta,job,'best','cpu')
        with self.assertRaisesRegex(ValueError,'separate'):ea.run(self.source,self.source/'bad','cpu')
        self.assertFalse((self.source/'bad').exists())

    def test_causal_failure_persists_diagnostic_before_research(self):
        meta=ea.make_manifest(self.source,self.identity);job=self.identity['manifest']['experiments'][0]
        failure=dict(status='failed',fp32=dict(passed=False),fp64=dict(passed=False))
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with patch.object(er,'trained_causality',return_value=failure),patch.object(er,'predict',side_effect=AssertionError('No research after failed audit')):
                with self.assertRaisesRegex(ValueError,'causal/reset'):ea.trial(meta,out,job,'best','cpu')
            path=out/'trials'/(job['name']+'_best')
            self.assertEqual(read(path/'causality.json')['status'],'failed')
            self.assertFalse((path/'completion.json').exists())

    def test_failure_export_and_reexport(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download';atomic_json(dict(status='partial'),run/'metrics.json')
            for suffix in ('pt','npy','npz'):(run/f'data.{suffix}').write_bytes(b'weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_ENDPOINT_RUN=str(run),BABEL_ENDPOINT_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            cmd=['bash',str(Path(__file__).resolve().parents[2]/'scripts/babel_endpoint_readout_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                with tarfile.open(download/'run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
