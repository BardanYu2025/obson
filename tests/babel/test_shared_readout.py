"""Shared-position fitting boundaries, recovery, source immutability and PCA controls."""
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

from obson.babel import shared_readout as sr, shared_readout_benchmark as sh
from obson.babel import prefix_readout as pr, prefix_readout_benchmark as pb
from obson.babel import architecture as ar, architecture_benchmark as ab
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features
from test_pca_anneal import read
import test_prefix_readout as prefix_tests
from test_prefix_readout import no_probe_update


class SharedMathTests(unittest.TestCase):
    def test_held_targets_are_causal_and_anchor_targets_exact(self):
        x = features(3); y, mask = ar.ordered_targets(x); stats = ar.fit_scales(x, y, mask)
        xx, yy, mm = ar.normalize(x, y, mask, stats); data = dict(x=xx, y=yy, mask=mm)
        for p in sr.PREFIXES:
            value, valid = sr.targets(data, stats, p)
            expected, ok = ar.ordered_targets(x[:, p-17:p])
            np.testing.assert_allclose(value, expected[:, :-1], atol=2e-6, rtol=2e-5)
            np.testing.assert_array_equal(valid, ok[:, :-1])
            changed = {k: v.copy() for k, v in data.items()}; changed['y'][:, p-1:] += 888
            np.testing.assert_array_equal(value, sr.targets(changed, stats, p)[0])
            if p in sr.TRAIN_PREFIXES:
                np.testing.assert_array_equal(value, pr.targets(data, stats, p)[0])
        with self.assertRaises(ValueError): sr.targets(data, stats,17)

    def test_pool_rejects_held_positions_and_preserves_window_order(self):
        row = lambda p: dict(x=np.full((3,2),p), y=np.full((3,16,7),p), mask=np.ones((3,16,7),bool))
        data = {p: row(p) for p in sr.TRAIN_PREFIXES}; result = sr.pool(data)
        np.testing.assert_array_equal(result['x'][:,0],np.repeat(sr.TRAIN_PREFIXES,3))
        with self.assertRaises(ValueError): sr.pool(data | {48: row(48)})
        with self.assertRaises(ValueError): sr.pool({p:row(p) for p in sr.HELD_PREFIXES})

    def test_pca_curve_nested_and_full_rank_roundtrip(self):
        rng=np.random.default_rng(4);y=rng.normal(size=(100,3,4)).astype(np.float32)
        data=dict(y=y);pca=ab.fit_pca(data,12);fractions=sr.pca_spectrum(pca,(2,4,8,12))
        self.assertEqual(list(fractions),['2','4','8','12']);self.assertAlmostEqual(fractions['12'],1)
        errors=[]
        for rank in (2,4,8,12):
            model=dict(pca,components=pca['components'][:rank]);pred=ab.pca_predict(model,data)
            errors.append(np.mean((pred-y)**2))
        self.assertTrue(all(a>=b for a,b in zip(errors,errors[1:])))
        np.testing.assert_allclose(pred,y,atol=1e-6)


class SharedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix_tests.ReadoutPipelineTests.setUpClass();cls.temp=tempfile.TemporaryDirectory()
        # Production CLI canonicalizes paths before creating immutable identities.
        prefix_tests.ReadoutPipelineTests.source=prefix_tests.ReadoutPipelineTests.source.resolve()
        prefix_tests.ReadoutPipelineTests.identity=pb.source_identity(prefix_tests.ReadoutPipelineTests.source)
        helper=prefix_tests.ReadoutPipelineTests();meta,cls.source=helper.make_run(cls.temp.name)
        with patch.object(pr,'run_epoch',side_effect=no_probe_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local neural updates')):
            for j in meta['experiments']:pb.worker(cls.source,j['name'],'cpu')
        pb.evaluate(meta,cls.source,'cpu')
        files={p.name:sha256(p) for p in cls.source.iterdir() if p.is_file() and p.suffix=='.json'}
        files.update({f'cache/{s}_index.json':sha256(cls.source/f'cache/{s}_index.json') for s in ('train','val','test','cross_research')})
        atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,files=files),cls.source/'completion.json')
        cls.identity=sh.source_identity(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup();prefix_tests.ReadoutPipelineTests.tearDownClass()

    def make_run(self,td):
        out=Path(td)/'shared';out.mkdir()
        meta=sh.make_manifest(self.source,self.identity,epochs=2,batch=4,hidden=8,ranks=(4,8,12))
        atomic_json(meta,out/'manifest.json')
        for split in ('train','val'):sh.prepare_split(meta,out,split,'cpu')
        return meta,out

    def test_complete_pipeline_never_fits_shared_on_held_and_preserves_sources(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            loader=ab.load_arrays;encoder=pb.encoder;seen=[]
            def guarded(root,split):self.assertIn(split,('train','val'));return loader(root,split)
            def spy(meta,name,device):
                model=encoder(meta,name,device);self.assertFalse(any(p.requires_grad for p in model.parameters()))
                model.encoder.register_forward_pre_hook(lambda m,args:seen.append(args[0].shape[1]));return model
            with patch.object(ab,'load_arrays',side_effect=guarded),patch.object(pb,'encoder',side_effect=spy):
                meta,out=self.make_run(td);sh.prepare_references(meta,out,'cpu')
            self.assertEqual(set(seen),set(sr.HELD_PREFIXES))
            target_hash=sha256(out/'cache/target_scales.json');calls=[];old_pool=sr.pool
            def pool_spy(rows):calls.append(set(rows));return old_pool(rows)
            with patch.object(sr,'pool',side_effect=pool_spy),patch.object(pr,'run_epoch',side_effect=no_probe_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
                for job in meta['experiments']:sh.worker(out,job['name'],'cpu')
            self.assertTrue(calls);self.assertTrue(all(v==set(sr.TRAIN_PREFIXES) for v in calls))
            def gate(root,split):
                if split in ('test','cross_research'):self.assertTrue((out/'selection_lock.json').is_file())
                return loader(root,split)
            with patch.object(ab,'load_arrays',side_effect=gate),patch.object(pr,'feature_scales',side_effect=AssertionError('No evaluation fitting')),patch.object(ab,'fit_pca',side_effect=AssertionError('No evaluation PCA fitting')):
                report=sh.evaluate(meta,out,'cpu')
            self.assertEqual(sha256(out/'cache/target_scales.json'),target_hash)
            for ds in report['datasets'].values():
                self.assertEqual(len(ds['scores']),187)
                self.assertIn('sampled_s42_p48/independent_ridge',ds['scores'])
                self.assertIn('sampled_s42/held/ridge',ds['scores'])
                self.assertEqual(len(ds['paired']),216)
            err=read(out/'test_errors.json')
            np.testing.assert_allclose(err['sampled_s42/held/ridge']['primary'],np.mean([err[f'sampled_s42_p{p}/ridge']['primary'] for p in sr.HELD_PREFIXES],axis=0))
            for job in meta['experiments']:
                summary=read(out/job['name']/'training_summary.json');self.assertEqual(summary['optimizer_steps'],24);self.assertEqual(summary['encoder_updates'],0)
                self.assertEqual(summary['selected_epoch'],0);self.assertTrue((out/job['name']/'initial_validation.json').exists())
            curve=read(out/'capacity_metrics.json')
            self.assertTrue(all(ds['original_control_replay']['matched'] for ds in curve['datasets'].values()))
            self.assertNotIn('NaN',(out/'shared_readout_metrics.json').read_text())
            with patch.object(pr,'run_epoch',side_effect=AssertionError('No duplicate training')):sh.worker(out,meta['experiments'][0]['name'],'cpu')
            # Even internally rehashed metadata must not bypass epoch0 selection.
            path=out/meta['experiments'][0]['name'];summary=read(path/'training_summary.json')
            summary['selected_epoch']=1;atomic_json(summary,path/'training_summary.json')
            done=read(path/'completion.json');done['files']['training_summary.json']=sha256(path/'training_summary.json')
            atomic_json(done,path/'completion.json')
            with self.assertRaisesRegex(ValueError,'selection'):sh.lock_selection(meta,out)
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_research_requires_lock_and_held_changes_do_not_affect_pool(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0]
            before=sh.pooled_data(out,'train',job['name'])
            f=out/f'cache/train_p48_{job["name"]}.npy';a=np.load(f);np.save(f,a+1000)
            after=sh.pooled_data(out,'train',job['name'])
            np.testing.assert_array_equal(before['x'],after['x'])
            with self.assertRaises(ValueError):sh.worker(out,job['name'],'cpu')
            with patch.object(ab,'load_arrays',side_effect=AssertionError('No research read')):
                with self.assertRaisesRegex(ValueError,'locked'):sh.prepare_split(meta,out,'test','cpu')
                with self.assertRaisesRegex(ValueError,'lock'):sh.evaluate_capacity(meta,out)

    def test_resume_reuses_optimizer_initial_validation_and_absolute_order(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0];calls=[]
            def interrupted(model,data,batch,device,opt=None,seed=0):
                if opt is not None:
                    calls.append(seed)
                    if len(calls)==2:raise RuntimeError('interrupted')
                    p=opt.param_groups[0]['params'][0];opt.state[p]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,.12),exp_avg_sq=torch.full_like(p,.34))
                return no_probe_update(model,data,batch,device,opt,seed)
            with patch.object(pr,'run_epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):sh.worker(out,job['name'],'cpu')
            initial=sha256(out/job['name']/'initial_validation.json');first=read(out/job['name']/'history.json')[0]
            restored=[];order=[];load=torch.optim.AdamW.load_state_dict
            def capture(self,state):restored.append(copy.deepcopy(state));return load(self,state)
            def resumed(model,data,batch,device,opt=None,seed=0):
                if opt is not None:order.append(seed)
                return no_probe_update(model,data,batch,device,opt,seed)
            with patch.object(pr,'run_epoch',side_effect=resumed),patch.object(torch.optim.AdamW,'load_state_dict',new=capture):sh.worker(out,job['name'],'cpu')
            self.assertEqual(order,[calls[-1]]);self.assertEqual(first,read(out/job['name']/'history.json')[0]);self.assertEqual(initial,sha256(out/job['name']/'initial_validation.json'))
            self.assertTrue(torch.all(next(iter(restored[0]['state'].values()))['exp_avg']==.12))

    def test_export_failure_and_no_binary_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download'
            atomic_json(dict(status='partial'),run/'metrics.json')
            for suffix in ('pt','npy','npz'):(run/f'data.{suffix}').write_bytes(b'weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_SHARED_RUN=str(run),BABEL_SHARED_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            cmd=['bash',str(Path(__file__).resolve().parents[2]/'scripts/babel_shared_readout_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                with tarfile.open(download/'run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
