import json
import os
import tempfile
import subprocess
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from obson.babel import state_transfer as st, price_readapt as pr
from obson.babel.ae_extend import atomic_json, atomic_save
from obson.babel.dual_state import sha256
from test_activity_extend import assert_tree


def bank(n,seed):
    rng=np.random.default_rng(seed);x=rng.normal(0,.04,(n,28)).astype(np.float32)
    x[:,9]=rng.normal(1.,.1,n);x[:,20]=rng.normal(0,.2,n);x[:,21]=.2;x[:,22]=.1;x[:,23:]=1
    return x


def fixture(root):
    parent=root/'readapt';encoder=root/'encoder';(parent/'cache').mkdir(parents=True);(encoder/'cache').mkdir(parents=True)
    atomic_json(dict(long_run=str(root/'unavailable')),encoder/'manifest.json')
    old=dict(schema=pr.SCHEMA,parent_run=str(encoder),parent_identity={'synthetic':True},seeds=[42],
        experiments=[dict(name=f'{m}_s42') for m in ('control','aux020')])
    atomic_json(old,parent/'manifest.json');atomic_json(dict(status='complete'),parent/'completion.json')
    atomic_json(dict(stage_screen=[]),parent/'readapt_metrics.json');files={}
    for split,n in [('train',12),('val',8),('test',8)]:
        x=bank(n*128,11+n);np.save(encoder/f'cache/{split}_x.npy',x)
        specs=[dict(offset=0,length=len(x),endpoints=[[127+i*128,i] for i in range(n)])]
        atomic_json(specs,encoder/f'cache/{split}_sequences.json')
        for j,job in enumerate(old['experiments']):
            path=parent/f'cache/{job["name"]}_{split}_z.npy';np.save(path,np.random.default_rng(n+j).normal(size=(n,8)).astype(np.float32));files[path.name]=sha256(path)
    inventory=[dict(end=f'2026-01-{i+1:02d}',week=f'week{i}') for i in range(8)]
    atomic_json(inventory,encoder/'cache/test_inventory.json')
    atomic_json(dict(manifest=old,files=files),parent/'cache/index.json')
    for job in old['experiments']:
        (parent/job['name']).mkdir()
        for f in ('best.pt','last.pt','selection.json'):(parent/job['name']/f).write_text('synthetic identity only')
    with patch.object(pr,'source_identity',return_value=({},old['parent_identity'])):_,identity=st.source_identity(parent)
    meta=st.make_manifest(parent,old,identity,2,8);meta['hidden']=8
    out=root/'transfer';out.mkdir();atomic_json(meta,out/'manifest.json');st.prepare(meta,out)
    return meta,out,parent


def no_update(model,opt,x,y,mask,batch):
    with torch.no_grad():return float(st.masked_mean((model(x)-y)**2,mask))


class StateTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_targets_causal_partition_and_internal_return(self):
        x=bank(400,7);spec=[dict(offset=30,length=300,endpoints=[[127,0],[255,1]])]
        y,m,_,stats=st.targets_from_bank(x,spec,2)
        changed=x.copy();changed[286:]=100
        yy,mm,_,_=st.targets_from_bank(changed,spec,2)
        np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(m,mm)
        changed=x.copy();changed[:30]=100;changed[30,:2]=100 # first in-window return must not borrow outside partition
        yy,_,_,_=st.targets_from_bank(changed,spec,2);np.testing.assert_array_equal(y,yy)
        self.assertEqual(stats.shape,(2,61))
        with self.assertRaisesRegex(ValueError,'Invalid'):st.targets_from_bank(x,[dict(offset=0,length=200,endpoints=[[126,0]])],1)
        w=bank(128,8);w[:,0]=np.arcsinh(.01);w[:,1]=0
        yy,_,_=st.window_descriptors(w);np.testing.assert_allclose(yy[:3],1,atol=1e-6)

    def test_missing_oi_and_constant_volume_are_masked_not_targets(self):
        x=bank(128,9);x[:,23]=0;x[:,9]=1
        y,m,stats=st.window_descriptors(x)
        self.assertFalse(m[6:].any());self.assertTrue(m[:6].all());self.assertTrue(np.isfinite(stats).all())
        x[:,23]=1;x[:,21]=2.
        _,m,_=st.window_descriptors(x);self.assertFalse(m[9:].any())
        x=bank(128,4);x[:,:2]=0;yy,mm,_=st.window_descriptors(x)
        np.testing.assert_array_equal(yy[:6],0);self.assertFalse(mm[6:].any())

    def test_train_normalizers_ignore_missing_values_and_mask_gradients(self):
        y=np.arange(120,dtype=float).reshape(10,12);mask=np.ones_like(y,bool);mask[0,0]=False
        before=st.target_stats(y,mask);y[0,0]=1e10;after=st.target_stats(y,mask)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        pred=torch.zeros((2,12),requires_grad=True);valid=torch.ones_like(pred,dtype=torch.bool);valid[:,3]=False
        loss=st.masked_mean((pred-1)**2,valid);loss.backward()
        self.assertEqual(float(pred.grad[:,3].abs().sum()),0);self.assertGreater(float(pred.grad[:,0].abs().sum()),0)
        with self.assertRaisesRegex(ValueError,'No valid'):st.masked_mean(pred,torch.zeros_like(valid))

    def test_cache_and_source_tampering_holdout_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out,parent=fixture(Path(tmp))
            self.assertFalse(json.loads((out/'coverage_audit.json').read_text())['independent_holdout'])
            with patch.object(st,'targets_from_bank',side_effect=AssertionError('reuse')):st.prepare(meta,out)
            (out/'cache/train_statistics.npy').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'fingerprint'):st.prepare(meta,out)
            (parent/'cache/control_s42_train_z.npy').write_bytes(b'bad')
            with patch.object(pr,'source_identity',return_value=({}, {'synthetic':True})):
                with self.assertRaisesRegex(ValueError,'fingerprint'):st.source_identity(parent)

    def test_mlp_selection_never_reads_test_and_exact_resume_without_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out,_=fixture(Path(tmp));xs,ys,masks=st.load_arrays(out,'control_s42');path=out/'trial'
            args=(xs[:2],ys[:2],masks[:2],path,{'test':'identity'},1701,.01,2,8,8,.001,'cpu')
            with patch.object(st,'train_probe_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                value=st.mlp_trial(*args)
            self.assertFalse(value['trained_selection'])
            ck=torch.load(path/'last.pt',weights_only=True);saved=ck['rng'];ck['epoch']=1;ck['history']=ck['history'][:1]
            for idx,param in enumerate(ck['model'].values()):
                ck['optimizer']['state'][idx]=dict(step=torch.tensor(5.),exp_avg=torch.full_like(param,.03),exp_avg_sq=torch.full_like(param,.01))
            moments=ck['optimizer']['state'];atomic_save(ck,path/'last.pt')
            with patch.object(st,'train_probe_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):st.mlp_trial(*args)
            final=torch.load(path/'last.pt',weights_only=True);assert_tree(self,final['rng'],saved);assert_tree(self,final['optimizer']['state'],moments)
            before=sha256(path/'last.pt')
            with patch.object(st,'train_probe_epoch',side_effect=AssertionError('completed retrained')):st.mlp_trial(*args)
            self.assertEqual(sha256(path/'last.pt'),before)
            ys[2]=ys[2]*100+200
            with patch.object(st,'train_probe_epoch',side_effect=no_update):same=st.mlp_trial(xs,ys,masks,out/'other',{'test':'identity'},1701,.01,2,8,8,.001,'cpu')
            self.assertEqual(same,value)
            with self.assertRaisesRegex(ValueError,'AutoDL'):st.train_probe_epoch(None,None,torch.zeros(2),None,None,8)

    def test_complete_matrix_reports_and_read_only_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out,parent=fixture(Path(tmp));hashes={p:sha256(p) for p in parent.rglob('*') if p.is_file()}
            with patch.object(st,'train_probe_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                for job in meta['experiments']:st.worker(out,job['name'],'cpu')
            report=st.evaluate(out)
            self.assertEqual(len(report['variants']),6);self.assertFalse(report['evidence_screen']['fresh_holdout_confirmed'])
            self.assertFalse(report['evidence_screen']['automatic_promotion'])
            self.assertIn('aux020_s42_plus_statistics_minus_statistics',report['paired'])
            for p,digest in hashes.items():self.assertEqual(sha256(p),digest)
            with patch.object(st,'mlp_trial',side_effect=AssertionError('completed worker retrained')):st.worker(out,'current','cpu')
            (out/'current/linear_errors.npy').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'fingerprint'):st.worker(out,'current','cpu')

    def test_export_reports_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(run/'transfer_metrics.json').write_text('{}');(run/'best.pt').write_bytes(b'x')
            env=dict(os.environ,BABEL_TRANSFER_RUN=str(run),BABEL_TRANSFER_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                p=subprocess.run(['bash','scripts/babel_state_transfer_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(p.returncode,code,p.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertIn('run/transfer_metrics.json',t.getnames());self.assertFalse(any(n.endswith('.pt') for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())
