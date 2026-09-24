"""Causal deterministic features, identical parent start and paired continuation."""
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

from obson.babel import path_feature as pf, path_feature_benchmark as run
from obson.babel.ae_extend import atomic_json, atomic_save, rng_state
from obson.babel.dual_state import sha256
from test_bar_alignment import small_model, no_alignment_update


class PathFeatureTests(unittest.TestCase):
    def test_path_matches_independent_log_prices_and_ignores_future(self):
        returns=torch.tensor([[.2,-.1,.4,-.3,.05]],dtype=torch.float64)
        prices=100*torch.exp(returns.cumsum(1)/100)
        raw=torch.zeros(1,5,28,dtype=torch.float64);raw[...,0]=torch.asinh(returns*.3);raw[...,1]=torch.asinh(returns*.7)
        mean=torch.tensor([.03,-.02],dtype=torch.float64);scale=torch.tensor([.8,.4],dtype=torch.float64)
        x=raw.clone();x[...,:2]=(raw[...,:2]-mean)/scale
        got=pf.causal_path(x,mean,scale,torch.tensor(3.))
        torch.testing.assert_close(got[...,0],torch.log(prices/100)*100/3,atol=1e-13,rtol=1e-12)
        changed=x.clone();changed[:,3:]=4
        torch.testing.assert_close(pf.causal_path(changed,mean,scale,torch.tensor(3.))[:,:3],got[:,:3],atol=0,rtol=0)
        torch.testing.assert_close(pf.causal_path(x[:,:3],mean,scale,torch.tensor(3.)),got[:,:3],atol=0,rtol=0)

    def test_zero_projection_is_exact_and_active_gradient_and_causality(self):
        original,data,stats,_=small_model();original.eval();x=torch.tensor(data['x'])
        control=copy.deepcopy(original);candidate=copy.deepcopy(original)
        a=pf.install(control,stats,False);b=pf.install(candidate,stats,True)
        with torch.no_grad():
            expected=original.core(x)
            self.assertTrue(torch.equal(expected,control.core(x)));self.assertTrue(torch.equal(expected,candidate.core(x)))
        control.core(x).square().mean().backward();candidate.core(x).square().mean().backward()
        self.assertEqual(float(a.grad.abs().max()),0.);self.assertGreater(float(b.grad.abs().max()),0.)
        # No optimizer update; activate a deterministic projection for prefix checks.
        with torch.no_grad():b.fill_(.01)
        result=run.er.trained_causality(candidate,x)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(pf.learning_rate(1,100,3e-5),3e-5)
        self.assertAlmostEqual(pf.learning_rate(100,100,3e-5),3e-6)

    def test_path_error_decomposition_distinguishes_offset_and_shape(self):
        pred=np.zeros((2,128,7));y=np.zeros_like(pred);mask=np.ones_like(pred,bool);mask[:,-1]=False
        pred[0,:127,0]=2
        pred[1,:127,0]=3+.1*(np.arange(127)-63)
        pred[:,-1,0]=1e8 # Current bar is not a historical scored target.
        d=pf.decompose_path(pred,y,mask,4.)
        np.testing.assert_allclose(d['offset'],[4.,9.]);np.testing.assert_allclose(d['remainder'],0,atol=1e-25)
        self.assertAlmostEqual(d['linear_drift'][1],.01*np.var(np.arange(127)))
        np.testing.assert_allclose(d['path'],d['offset']+d['linear_drift']+d['remainder'])
        np.testing.assert_allclose(d['signed_offset_bps'],[800,1200])


class ContinuationTests(unittest.TestCase):
    def fixture(self,root):
        model,data,stats,local=small_model();source=root/'source';(source/'cache').mkdir(parents=True)
        atomic_json({},source/'cache/index.json');atomic_json(stats,source/'cache/statistics.json');atomic_json(local,source/'cache/local_scales.json')
        parents={str(s):dict(name=f'w768_joint_s{s}',width=768,joint=True,seed=s,epochs=200,lr=3e-4,head_lr=1e-3) for s in (42,43)}
        sm=dict(batch=2,micro=1,windows_per_epoch=4,experiments=list(parents.values())+
            [dict(name=f'w512_endpoint_s{s}',width=512,joint=False,seed=s,epochs=200,lr=3e-4,head_lr=1e-3) for s in (42,43)],
            decision=dict(local_improvement=.1,global_retention=.05,family_retention=.1))
        weights={};lock={}
        for seed,parent in parents.items():
            path=source/parent['name'];path.mkdir()
            enc=torch.optim.AdamW([p for p in model.core.parameters() if p.requires_grad],lr=3e-5,weight_decay=.01)
            head=torch.optim.AdamW(model.local_head.parameters(),lr=1e-4,weight_decay=1e-4)
            # Synthetic optimizer moments, constructed without taking a training step.
            for opt in (enc,head):
                for p in opt.param_groups[0]['params']:
                    opt.state[p]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(p,.01),exp_avg_sq=torch.full_like(p,.02))
            expected=run.bb.validation(model,data,stats,local,sm,parent,'cpu')
            ck=dict(metadata=dict(manifest=sm,job=parent,cache_sha256=sha256(source/'cache/index.json')),
                epoch=200,history=[dict(epoch=200,validation=expected)],model=run.bb.ab.cpu_state(model),
                encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state())
            atomic_save(ck,path/'last.pt');weights[seed]=sha256(path/'last.pt');lock[parent['name']]=dict(last=weights[seed])
            atomic_json([dict(epoch=200,lr=3e-5,head_lr=1e-4,validation=expected)],path/'history.json')
        atomic_json(dict(weights=lock),source/'selection_lock.json')
        meta=run.make_manifest(source,dict(manifest=sm),epochs=2,micro=2)
        out=root/'run';out.mkdir();atomic_json(meta,out/'manifest.json')
        return model,data,stats,local,meta,out

    def test_parent_moments_and_parameter_order_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            model,data,stats,local,meta,out=self.fixture(Path(td));job=meta['experiments'][1]
            with patch.object(run.bb,'model_for',side_effect=lambda *a:copy.deepcopy(model)):
                restored,enc,head,ck=run.construct(meta,job,'cpu',True)
            self.assertEqual(len(enc.param_groups),2)
            self.assertEqual(enc.param_groups[0]['lr'],3e-5);self.assertEqual(head.param_groups[0]['lr'],1e-4)
            for p in enc.param_groups[0]['params']:
                self.assertTrue(torch.equal(enc.state[p]['exp_avg'],torch.full_like(p,.01)))
                self.assertEqual(float(enc.state[p]['step']),7.)
            new=restored.core.encoder.backbone.input.projection.weight
            self.assertIs(enc.param_groups[1]['params'][0],new);self.assertNotIn(new,enc.state)
            x=torch.tensor(data['x']);self.assertTrue(torch.equal(restored.core(x),model.core(x)))

    def test_four_arm_budget_selection_resume_and_no_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as td:
            model,data,stats,local,meta,out=self.fixture(Path(td))
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch.object(run.bb,'model_for',side_effect=lambda *a:copy.deepcopy(model)))
                stack.enter_context(patch.object(run.bb,'load_data',return_value=data))
                stack.enter_context(patch.object(run.bb.ba,'run_epoch',side_effect=no_alignment_update))
                stack.enter_context(patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')))
                run.preflight(meta,out,'cpu')
                self.assertEqual(meta['micro'],2);self.assertEqual(meta['evaluation_batch'],1)
                # Interrupt a worker after one saved epoch, then restore and complete.
                calls=0
                def interrupt(*args,**kwargs):
                    nonlocal calls
                    if len(args)>9 and args[9] is not None:
                        calls+=1
                        if calls==2:raise RuntimeError('synthetic interruption')
                    return no_alignment_update(*args,**kwargs)
                first=meta['experiments'][0]
                with patch.object(run.bb.ba,'run_epoch',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):run.worker(out,first['name'],'cpu')
                for job in meta['experiments']:run.worker(out,job['name'],'cpu')
                lock=run.lock_selection(meta,out)
                self.assertEqual(len(lock['trials']),4)
                self.assertTrue(all(r['encoder_steps']==4 and r['windows']==8 for r in lock['trials'].values()))
                self.assertTrue((out/first['name']/'resume_validation.json').exists())
                for seed in (42,43):
                    a=run.read_json(out/f'control_s{seed}/history.json');b=run.read_json(out/f'path_s{seed}/history.json')
                    self.assertEqual([r['sampling'] for r in a],[r['sampling'] for r in b])
                    self.assertEqual([r['sampling']['absolute_epoch'] for r in a],[201,202])
                before=sha256(out/first['name']/'last.pt');run.worker(out,first['name'],'cpu')
                self.assertEqual(before,sha256(out/first['name']/'last.pt'))
                # Recreate derived artifacts from authoritative last after an interrupted publish.
                for file in ('completion.json','best.pt','history.json'):(out/first['name']/file).unlink()
                run.worker(out,first['name'],'cpu');run.lock_selection(meta,out)

                # Complete real scoring code on synthetic arrays; research only after lock.
                source=Path(meta['source']);inventory=[dict(key=f'X/15/C{i}',symbol='X',period=15,end=f'2024-01-0{i+1}',week='w1',month='2024-01') for i in range(len(data['x']))]
                _,error,_,_=run.score(model,data,stats,local,2,'cpu')
                for split in run.SPLITS:
                    atomic_json(inventory,source/f'{split}_inventory.json')
                    for kind in ('best','last'):
                        old={j['name']+'/'+task:{k:v.tolist() for k,v in e.items()} for j in meta['identity']['manifest']['experiments'] for task,e in error.items()}
                        atomic_json(old,source/f'{split}_{kind}_errors.json')
                def load_data(sm,split):
                    if split in run.SPLITS:self.assertTrue((out/'selection_lock.json').exists())
                    return data
                with patch.object(run.bb,'load_data',side_effect=load_data),patch.object(run.ea,'load_model',side_effect=lambda *a:(copy.deepcopy(model).requires_grad_(False),{})):
                    run.evaluate(meta,out,'cpu')
                report=run.read_json(out/'path_feature_metrics.json')
                self.assertEqual(len(report['datasets']['test']['best']['scores']),8)
                self.assertEqual(run.read_json(out/'decision.json')['status'],'no_full_upgrade')
                self.assertEqual(len(run.read_json(out/'validation_replay.json')),8)

                # Even internally rehashed summaries cannot claim a different exposure.
                path=out/first['name'];summary=run.read_json(path/'training_summary.json')
                summary['windows']+=1;atomic_json(summary,path/'training_summary.json')
                done=run.read_json(path/'completion.json');done['files']['training_summary.json']=sha256(path/'training_summary.json')
                atomic_json(done,path/'completion.json')
                with self.assertRaisesRegex(ValueError,'Summary exposure budget mismatch'):run.lock_selection(meta,out)

    def test_old_retention_and_matched_budget_checks_cannot_be_averaged_away(self):
        inv=[dict(symbol='X',period=15,month='2026-01',week=f'w{i//10}') for i in range(60)]
        errors={};inventories={s:inv for s in run.SPLITS};metrics=run.bb.GLOBAL_METRICS
        for split in run.SPLITS:
            errors[split]={}
            for kind in ('best','last'):
                rows={};errors[split][kind]=rows
                for seed in (42,43):
                    for mode,value in (('w512_endpoint',1.),('control',.6),('path',.5)):
                        for task in ('global','held','recent'):
                            rows[f'{mode}_s{seed}/{task}']={k:np.full(60,value) for k in metrics}
        meta=dict(decision=dict(path_gain=.05,primary_retention=.05,family_retention=.1,local_retention=.05),
            identity=dict(manifest=dict(decision=dict(local_improvement=.1,global_retention=.05,family_retention=.1))))
        self.assertEqual(run.decide(meta,errors,inventories)['status'],'candidate_for_review')
        errors['test']['best']['path_s43/global']['path'][:]=1.2
        self.assertEqual(run.decide(meta,errors,inventories)['status'],'no_full_upgrade')


class ExportTests(unittest.TestCase):
    def test_partial_complete_and_failure_archives_preserve_exit_code_and_exclude_weights(self):
        repo=Path(__file__).resolve().parents[2]
        script=repo/'scripts/babel_path768_autodl.sh'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run with spaces';(out/'path_s42').mkdir(parents=True)
            atomic_json(dict(epoch=1),out/'path_s42/history.json')
            (out/'path_s42/last.pt').write_bytes(b'not a report')
            (out/'cache.npy').write_bytes(b'not a report')
            (out/'path_s42/run.log').write_text('worker log')
            log=root/'main.log';log.write_text('main log')
            env=dict(os.environ,BABEL_PATH_RUN=str(out),BABEL_PATH_LOG=str(log),BABEL_DOWNLOAD_DIR=str(root/'downloads'))
            def invoke(mode,expected_code,status):
                result=subprocess.run(['bash',str(script),mode],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,expected_code,result.stdout+result.stderr)
                with tarfile.open(root/'downloads/run with spaces_reports.tar.gz') as archive:
                    names=archive.getnames()
                    self.assertIn('run with spaces/path_s42/history.json',names)
                    self.assertIn('run with spaces/path_s42/run.log',names)
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in names))
                    state=archive.extractfile('run with spaces/run_status.txt').read().decode()
                    self.assertIn(f'run_status={status}',state)
                    self.assertIn(f'command_exit_code={expected_code}',state)
            invoke('export',0,'partial')
            atomic_json(dict(status='complete'),out/'completion.json');invoke('export',0,'complete')
            (out/'completion.json').unlink()
            atomic_json(dict(status='failed'),out/'failure.json');invoke('export',0,'failed')
            fake=root/'failure-python';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env['PYTHON_BIN']=str(fake);invoke('all',7,'failed')


if __name__=='__main__':unittest.main()
