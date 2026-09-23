"""Past-only labels, exact train-only fits, locked evaluation and frozen provenance."""
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

from obson.babel import window_state_probe as p, window_state_probe_run as run
from obson.babel import window_state as ws, window_state_delivery as wd
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
import test_window_state as delivery_fixture


def inventories(counts):
    result={}
    for j,(split,n) in enumerate(counts.items()):
        symbol='B' if split=='cross_research' else 'A';year=2020+min(j,2)
        result[split]=[dict(key=f'{symbol}/15/C',symbol=symbol,period=15,row=511+10000*j+i*128,
            end=f'{year}-01-{i+1:02d} 09:00:00',month=f'{year}-01',week=f'{year}-w{i//2}') for i in range(n)]
    return result


class ProbeMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_targets_match_direct_close_formulas_and_exclude_outside_history(self):
        rng=np.random.default_rng(1);r=rng.normal(0,.001,(4,128));raw=np.zeros((4,128,28));raw[...,1]=np.arcsinh(r*100)
        got=p.descriptors(raw)
        for i,h in enumerate(p.HORIZONS):
            closes=np.cumsum(r,axis=1)[:,-h:];changes=np.diff(closes,axis=1)
            expected=changes.sum(1)/np.abs(changes).sum(1)
            np.testing.assert_allclose(got[:,i*3],expected,atol=1e-12)
            for n,path in enumerate(closes):
                coefficients=np.polyfit(np.arange(h),path,1);prediction=np.polyval(coefficients,np.arange(h))
                self.assertAlmostEqual(got[n,i*3+1],1-np.sum((prediction-path)**2)/np.sum((path-path.mean())**2))
            np.testing.assert_allclose(got[:,i*3+2],np.log(np.sqrt(np.square(changes).mean(1))),atol=1e-12)
        changed=raw.copy();changed[:,:65,:]=4.;np.testing.assert_array_equal(got,p.descriptors(changed))
        changed=raw.copy();changed[:,-1,1]+=.2;self.assertFalse(np.array_equal(got,p.descriptors(changed)))
        # Same past endpoint slice remains unchanged after arbitrary future edits.
        bank=np.concatenate([raw,raw],axis=1);before=p.descriptors(bank[:,:128]);bank[:,128:]=100
        np.testing.assert_array_equal(before,p.descriptors(bank[:,:128]))

    def test_flat_and_monotone_state_semantics(self):
        raw=np.zeros((2,128,28));raw[1,:,1]=np.arcsinh(.1)
        target=p.descriptors(raw)
        np.testing.assert_allclose(target[0],[0,0,np.log(1e-8)]*2)
        np.testing.assert_allclose(target[1],[1,1,np.log(.001)]*2,atol=1e-12)
        c=p.classification(np.array([-.8,0,.7]),np.array([-.4,.1,.9]));self.assertEqual(c['balanced_accuracy_present'],1.)
        self.assertEqual(p.direction(np.array([-.2,.2])).tolist(),[1,1])
        self.assertFalse(p.classification(np.zeros(3),np.zeros(3))['all_classes_present'])

    def test_ridge_matches_direct_normal_equations_and_train_scaling(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(30,7));x[:,-1]=1
        y=rng.normal(size=(30,6));v=rng.normal(size=(13,7));vy=rng.normal(size=(13,6))
        head,grid=p.ridge_select(x,y,v,vy,'cpu');xn=p.normalize(x,head['statistics']);ym=y.mean(0)
        expected=np.linalg.solve(xn.T@xn/len(x)+head['alpha']*np.eye(7),xn.T@(y-ym)/len(x))
        np.testing.assert_allclose(head['weights'],expected,atol=1e-12)
        np.testing.assert_allclose(p.predict(head,v),p.normalize(v,head['statistics'])@expected+ym,atol=1e-12)
        self.assertEqual(head['alpha'],min(grid,key=lambda z:z['validation_nmse'])['alpha'])
        alternate,_=p.ridge_select(x,y,v+100,vy,'cpu');self.assertEqual(head['statistics'],alternate['statistics'])
        self.assertEqual(len(grid),5)

    def test_thin_eigensystem_and_constant_target_scoring(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(6,10));x-=x.mean(0)
        val,vec=p.eigensystem(x,'cpu');np.testing.assert_allclose(vec.numpy()@np.diag(val.numpy())@vec.numpy().T,x.T@x/6,atol=1e-12)
        y=np.ones((5,6));report,errors=p.measure(y,y,p.scales(y))
        self.assertEqual(report['primary'],0.);self.assertTrue(all(v['r2'] is None for v in report['targets']))
        self.assertEqual(set(errors),{'primary',*p.FAMILIES})

    def test_partition_overlap_and_symbol_leakage_rejected(self):
        rows=inventories(dict(train=3,val=2,test=2,cross_research=2));self.assertEqual(p.inventory_audit(rows)['train']['windows'],3)
        for change in ('row','time','symbol','duplicate'):
            bad=copy.deepcopy(rows)
            if change=='row':bad['val'][0]['row']=rows['train'][-1]['row']+127
            if change=='time':bad['val'][0]['end']='2019-01-01 00:00:00'
            if change=='symbol':bad['cross_research'][0].update(symbol='A',key='A/15/C')
            if change=='duplicate':bad['train'].append(bad['train'][0])
            with self.assertRaises(ValueError):p.inventory_audit(bad)


class ProbePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        delivery_fixture.DeliveryTests.setUpClass();cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.source=cls.root/'state';raw=cls.root/'raw';raw.mkdir()
        with patch.object(wd,'raw_plan',return_value=[]),patch.object(wd,'raw_replay',return_value=[]):
            wd.run(delivery_fixture.DeliveryTests.source,cls.source,raw,'cpu')
        sm=ws.read_json(cls.source/'manifest.json')['identity']['manifest']['identity']['manifest']
        sampling=ws.read_json(Path(sm['sampling_source'])/'manifest.json');cls.coverage=sampling['coverage_identity']
        cls.rows=inventories({s:len(ws.ea.bb.load_data(sm,s)['x']) for s in run.SPLITS})

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup();delivery_fixture.DeliveryTests.tearDownClass()

    def test_end_to_end_frozen_selection_resume_and_tamper(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()};out=self.root/'probe'
        # Ancestor fixtures have synthetic coverage, not raw contracts; only its
        # external coverage audit/inventories are substituted. Extraction, PCA,
        # head fitting, lock, scoring, hashes and frozen models run unmocked.
        original=run.prepare;events=[]
        def ordered(meta,out,split,rows,device):
            if split in run.SPLITS[2:]:self.assertTrue((out/'selection_lock.json').exists())
            events.append(split);return original(meta,out,split,rows,device)
        with patch.object(ws.ea.bb.ws,'coverage_identity',return_value=self.coverage),patch.object(run,'inventory',return_value=self.rows),patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No neural optimizer')):
            with patch.object(run,'prepare',side_effect=ordered):run.run(self.source,out,batch=4,device='cpu')
            self.assertEqual(events,list(run.SPLITS));done=ws.read_json(out/'completion.json');self.assertEqual(done['encoder_updates'],0)
            fitted=ws.read_json(out/'fit.json');self.assertEqual(len(fitted['readouts']),5)
            meta=ws.read_json(out/'manifest.json');stats=fitted['target_stats']
            for split in run.SPLITS[2:]:
                values=ws.read_json(out/f'{split}_predictions.json');reported=ws.read_json(out/'state_probe_metrics.json')['datasets'][split]
                for name,values_pred in values['predictions'].items():
                    actual,errors=p.measure(values_pred,values['targets'],stats)
                    self.assertEqual(actual,reported['scores'][name])
                    for key in errors:np.testing.assert_array_equal(errors[key],values['per_window_errors'][name][key])
            with patch.object(run,'prepare',side_effect=AssertionError('Completed resume must not infer')),patch.object(run,'fit',side_effect=AssertionError('Completed resume must not fit')):
                run.run(self.source,out,batch=4,device='cpu')
            with self.assertRaisesRegex(ValueError,'configuration'):run.run(self.source,out,batch=8,device='cpu')
            path=out/'fit.json';original_bytes=path.read_bytes()
            try:
                path.write_text('{}')
                with self.assertRaises(ValueError):run.run(self.source,out,batch=4,device='cpu')
            finally:path.write_bytes(original_bytes)
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_research_requires_lock_and_output_cannot_overlap_source(self):
        with self.assertRaises(ValueError):run.check_output(self.source,self.source/'bad')
        self.assertFalse((self.source/'bad').exists())
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):run.prepare({},Path(td),'test',[],'cpu')

    def test_failure_export_reports_exclude_model_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';out.mkdir();atomic_json(dict(status='partial'),out/'diagnostic.json')
            for suffix in ('pt','npy','npz'):(out/f'weights.{suffix}').write_bytes(b'weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_STATE_PROBE_RUN=str(out),BABEL_STATE_PROBE_LOG=str(root/'absent'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN=str(fake))
            script=Path(__file__).resolve().parents[2]/'scripts/babel_window_state_probe_autodl.sh'
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(['bash',str(script),mode],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,code,result.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as archive:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in archive.getnames()))
                    self.assertIn('run_status=failed',archive.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
