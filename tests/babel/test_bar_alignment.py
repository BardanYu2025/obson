"""Causal local supervision, isolated control gradients, fixed budgets and route convergence."""
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

from obson.babel import bar_alignment as ba, bar_alignment_benchmark as bb
from obson.babel import shared_readout_benchmark as sh, prefix_readout as pr
from obson.babel import architecture as ar, architecture_benchmark as ab, pca_teacher as pt
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features
from test_pca_anneal import read
from test_prefix_readout import no_probe_update
import test_shared_readout as shared_tests

original_epoch = ba.run_epoch


def no_alignment_update(model,data,original,local,batch,micro,device,joint,prefixes,encoder_opt=None,head_opt=None,order_seed=0):
    result,_=original_epoch(model,data,original,local,batch,micro,device,joint,prefixes,order_seed=order_seed)
    return result,(len(data['x'])+batch-1)//batch if encoder_opt is not None else 0


def small_model():
    torch.set_num_threads(1);x=features(4);y,m=ar.ordered_targets(x);stats=ar.fit_scales(x,y,m)
    xx,yy,mm=ar.normalize(x,y,m,stats);data=dict(x=xx,y=yy,mask=mm)
    pca=ab.fit_pca(data,4);scales=pt.fit_coordinate_scales(pt.coefficients(pca,data))
    config=dict(pt.CONFIG,latent=4,heads=2,attention_layers=2,attention_ff=8,residual_width=8)
    model=ba.AlignedStudent(config,42,pca,scales,hidden=8)
    local=pr.fit_target_scales([pr.targets(data,stats,p) for p in ba.VAL_PREFIXES])
    return model,data,stats,local


class AlignmentMathTests(unittest.TestCase):
    def test_targets_match_old_cache_and_exclude_current_future(self):
        model,data,stats,local=small_model();y=torch.tensor(data['y']);mask=torch.tensor(data['mask'])
        ps=torch.tensor(ba.VAL_PREFIXES)[None].expand(4,-1)
        target,valid=ba.local_targets(y,mask,ps,stats,local)
        for i,p in enumerate(ba.VAL_PREFIXES):
            yy,mm=pr.targets(data,stats,p);expected=pr.normalize(yy,mm,local)
            np.testing.assert_array_equal(target[:,i].numpy(),expected)
            np.testing.assert_array_equal(valid[:,i].numpy(),mm)
        ps=torch.full((4,1),48);a=ba.local_targets(y,mask,ps,stats,local)[0]
        changed=y.clone();changed[:,47:]+=10000
        torch.testing.assert_close(a,ba.local_targets(changed,mask,ps,stats,local)[0],rtol=0,atol=0)
        with self.assertRaises(ValueError):ba.local_targets(y,mask,torch.full((4,1),17),stats,local)

    def test_gradients_detach_control_but_joint_reaches_encoder(self):
        model,data,stats,local=small_model();x=torch.tensor(data['x']);y=torch.tensor(data['y']);mask=torch.tensor(data['mask'])
        ps=torch.tensor(ba.VAL_PREFIXES)[None].expand(4,-1);t,m=ba.local_targets(y,mask,ps,stats,local)
        _,p=model(x,ps,False);ba.local_rows(p,t,m,True)['primary'].mean().backward()
        self.assertTrue(all(v.grad is None for v in model.core.parameters()))
        self.assertTrue(all(v.grad is not None for v in model.local_head.parameters()))
        model.zero_grad(set_to_none=True)
        old=ar.error_rows(model.core(x),y,mask,stats,True)['primary'].mean();old.backward()
        original={k:v.grad.clone() for k,v in model.core.named_parameters() if v.grad is not None}
        model.zero_grad(set_to_none=True);g,p=model(x,ps,False)
        (ar.error_rows(g,y,mask,stats,True)['primary'].mean()+ba.LOCAL_WEIGHT*ba.local_rows(p,t,m,True)['primary'].mean()).backward()
        for k,v in model.core.named_parameters():
            if k in original:torch.testing.assert_close(v.grad,original[k],rtol=0,atol=0)
        model.zero_grad(set_to_none=True);_,p=model(x,ps,True);ba.local_rows(p,t,m,True)['primary'].mean().backward()
        self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in model.core.encoder.parameters()))
        # All current state computations are causal even when extracting in one full pass.
        with torch.no_grad():
            full=model.core.encoder(x)
            for prefix in bb.EVAL_PREFIXES:torch.testing.assert_close(full[:,prefix-1],model.core.encoder(x[:,:prefix])[:,-1],atol=1e-6,rtol=1e-5)

    def test_position_plan_reproducible_distinct_and_held_excluded(self):
        a=ba.position_plan(4789,42,1);b=ba.position_plan(4789,42,1)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(4789,4));self.assertTrue((np.diff(a,axis=1)>0).all())
        self.assertFalse(np.isin(a,ba.HELD_PREFIXES).any());self.assertEqual(set(a.flatten()),set(ba.TRAIN_PREFIXES))
        self.assertFalse(np.array_equal(a,ba.position_plan(4789,42,2)))
        self.assertFalse(np.array_equal(a,ba.position_plan(4789,43,1)))

    def test_production_widths_causal_backward_without_optimizer_updates(self):
        _,data,stats,local=small_model()
        x=torch.tensor(data['x'][:2]);y=torch.tensor(data['y'][:2]);mask=torch.tensor(data['mask'][:2])
        ps=torch.tensor(ba.VAL_PREFIXES)[None].expand(2,-1)
        target,valid=ba.local_targets(y,mask,ps,stats,local)
        for width in (512,768):
            # Synthetic orthonormal decoder fixture; no market-data PCA fitting or training.
            pca=dict(components=np.eye(896,dtype=np.float32)[:width],mean=np.zeros(896,dtype=np.float32))
            scales=dict(mean=[0.]*width,scale=[1.]*width)
            model=ba.AlignedStudent(dict(pt.CONFIG,latent=width),42,pca,scales).eval()
            with torch.no_grad():
                full=model.core.encoder(x)
                for prefix in bb.EVAL_PREFIXES:
                    torch.testing.assert_close(full[:,prefix-1],model.core.encoder(x[:,:prefix])[:,-1],atol=1e-5,rtol=5e-5)
            pred,detail=model(x,ps,True)
            loss=ar.error_rows(pred,y,mask,stats,True)['primary'].mean()+ba.LOCAL_WEIGHT*ba.local_rows(detail,target,valid,True)['primary'].mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_screen_prefers_small_model_and_can_stop(self):
        meta=dict(widths=[8,12],decision=dict(local_improvement=.1,global_retention=.05,family_retention=.1,upgrade_gain=.05))
        inventory=[dict(symbol='A',period=15,month='2026-01',week=f'w{i//10}') for i in range(60)]
        def errors(local,global_):
            row={}
            for name in local:
                for seed in [42,43]:
                    row[f'{name}_s{seed}/held']={'primary':np.full(60,local[name])}
                    row[f'{name}_s{seed}/global']={k:np.full(60,global_[name]) for k in bb.GLOBAL_METRICS}
            return {s:{k:copy.deepcopy(row) for k in ['best','last']} for s in ['test','cross_research']}
        local={'w8_endpoint':1.,'w8_joint':.7,'w12_endpoint':.68,'w12_joint':.67};glob={k:1. for k in local}
        inv={s:inventory for s in ['test','cross_research']}
        self.assertEqual(bb.route_decision(meta,{},errors(local,glob),inv)['recommended_route'],'w8_joint')
        local['w12_joint']=.5
        self.assertEqual(bb.route_decision(meta,{},errors(local,glob),inv)['recommended_route'],'w12_joint')
        glob={k:1.2 if k!='w8_endpoint' else 1. for k in local}
        self.assertEqual(bb.route_decision(meta,{},errors(local,glob),inv)['status'],'retain_baseline_and_stop')
        # A single bad seed/checkpoint is not hidden by averaging.
        e=errors(local,{k:1. for k in local});e['test']['last']['w12_joint_s43/held']['primary'][:]=1.1
        self.assertFalse(bb.route_decision(meta,{},e,inv)['screens']['w12_joint']['passed'])


class AlignmentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared_tests.SharedPipelineTests.setUpClass();cls.temp=tempfile.TemporaryDirectory()
        helper=shared_tests.SharedPipelineTests();meta,cls.source=helper.make_run(cls.temp.name)
        sh.prepare_references(meta,cls.source,'cpu')
        with patch.object(pr,'run_epoch',side_effect=no_probe_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local neural training')):
            for job in meta['experiments']:sh.worker(cls.source,job['name'],'cpu')
        sh.evaluate(meta,cls.source,'cpu')
        files={p.name:sha256(p) for p in cls.source.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
        files.update({f'cache/{s}_index.json':sha256(cls.source/f'cache/{s}_index.json') for s in ('train','val','test','cross_research')})
        atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,files=files),cls.source/'completion.json')
        cls.identity=bb.source_identity(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup();shared_tests.SharedPipelineTests.tearDownClass()

    def make_run(self,td):
        out=Path(td)/'alignment';out.mkdir()
        meta=bb.make_manifest(self.source,self.identity,widths=(8,12),epochs=2,batch=4,micro=2,hidden=8)
        atomic_json(meta,out/'manifest.json');bb.prepare(meta,out)
        return meta,out

    def test_complete_matrix_budgets_selection_and_sources(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            loader=bb.load_data
            def guard(meta,split):self.assertIn(split,('train','val','pool'));return loader(meta,split)
            with patch.object(bb,'load_data',side_effect=guard):
                meta,out=self.make_run(td);bb.preflight(meta,out,'cpu')
                with patch.object(ba,'run_epoch',side_effect=no_alignment_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
                    for job in meta['experiments']:bb.worker(out,job['name'],'cpu')
            def gate(meta,split):
                if split in ('test','cross_research'):self.assertTrue((out/'selection_lock.json').exists())
                return loader(meta,split)
            with patch.object(bb,'load_data',side_effect=gate):report=bb.evaluate(meta,out,'cpu')
            self.assertEqual(len(read(out/'validation_replay.json')),16)
            for job in meta['experiments']:
                s=read(out/job['name']/'training_summary.json')
                self.assertEqual(s['encoder_steps'],6);self.assertEqual(s['head_steps'],6);self.assertEqual(s['local_exposures'],96);self.assertEqual(s['selected_epoch'],0)
            for ds in report['datasets'].values():
                for result in ds.values():self.assertEqual(len(result['scores']),80);self.assertEqual(len(result['paired']),10)
            self.assertEqual(read(out/'decision.json')['status'],'retain_baseline_and_stop')
            self.assertNotIn('NaN',(out/'alignment_metrics.json').read_text())
            with patch.object(ba,'run_epoch',side_effect=AssertionError('No rerun')):bb.worker(out,meta['experiments'][0]['name'],'cpu')
            # Even rehashed metadata cannot change selected epochs independently of history.
            path=out/meta['experiments'][0]['name'];s=read(path/'training_summary.json');s['selected_epoch']=1;atomic_json(s,path/'training_summary.json')
            done=read(path/'completion.json');done['files']['training_summary.json']=sha256(path/'training_summary.json');atomic_json(done,path/'completion.json')
            with self.assertRaisesRegex(ValueError,'selection'):bb.lock_selection(meta,out)
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})

    def test_resume_restores_both_optimizers_and_position_plan(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0];calls=[]
            def interrupted(model,data,original,local,batch,micro,device,joint,prefixes,encoder_opt=None,head_opt=None,order_seed=0):
                if encoder_opt is not None:
                    calls.append((order_seed,prefixes.copy()))
                    if len(calls)==2:raise RuntimeError('interrupted')
                    for opt,value in [(encoder_opt,.12),(head_opt,.34)]:
                        p=opt.param_groups[0]['params'][0];opt.state[p]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,value),exp_avg_sq=torch.full_like(p,.5))
                return no_alignment_update(model,data,original,local,batch,micro,device,joint,prefixes,encoder_opt,head_opt,order_seed)
            with patch.object(ba,'run_epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):bb.worker(out,job['name'],'cpu')
            first=read(out/job['name']/'history.json')[0];restored=[];orders=[];load=torch.optim.AdamW.load_state_dict
            def capture(self,state):restored.append(copy.deepcopy(state));return load(self,state)
            def resumed(model,data,original,local,batch,micro,device,joint,prefixes,encoder_opt=None,head_opt=None,order_seed=0):
                if encoder_opt is not None:orders.append((order_seed,prefixes.copy()))
                return no_alignment_update(model,data,original,local,batch,micro,device,joint,prefixes,encoder_opt,head_opt,order_seed)
            with patch.object(ba,'run_epoch',side_effect=resumed),patch.object(torch.optim.AdamW,'load_state_dict',new=capture):bb.worker(out,job['name'],'cpu')
            self.assertEqual(orders[0][0],calls[-1][0]);np.testing.assert_array_equal(orders[0][1],calls[-1][1]);self.assertEqual(first,read(out/job['name']/'history.json')[0])
            self.assertEqual(len(restored),2)
            for state,value in zip(restored,[.12,.34]):self.assertTrue(torch.all(next(iter(state['state'].values()))['exp_avg']==value))

    def test_budget_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0]
            with patch.object(ba,'run_epoch',side_effect=no_alignment_update):bb.worker(out,job['name'],'cpu')
            history=read(out/job['name']/'history.json')
            for key,value in [('encoder_steps',99),('head_steps',99),('local_exposures',99),('lr',1.),('head_lr',1.)]:
                bad=copy.deepcopy(history);bad[0][key]=value
                with self.assertRaises(ValueError):bb.verify_history(meta,out,job,bad)
            bad=copy.deepcopy(history);bad[0]['sampling']['prefixes_sha256']='changed'
            with self.assertRaises(ValueError):bb.verify_history(meta,out,job,bad)

    def test_export_on_failure_and_reexport(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download'
            atomic_json(dict(status='partial'),run/'metrics.json')
            for suffix in ('pt','npy','npz'):(run/f'data.{suffix}').write_bytes(b'weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_ALIGN_RUN=str(run),BABEL_ALIGN_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            cmd=['bash',str(Path(__file__).resolve().parents[2]/'scripts/babel_bar_alignment_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                with tarfile.open(download/'run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
