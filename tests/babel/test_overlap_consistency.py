"""No local neural optimizer updates: independent loss/gradient and lifecycle tests."""
import copy
import os
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from obson.babel import overlap_consistency as oc,overlap_consistency_run as run,overlap_consistency_evaluate as ev
from obson.babel.ae_extend import atomic_json,atomic_save,rng_state
from obson.babel.dual_state import sha256
from test_bar_alignment import small_model
from test_overlap_diagnostic import fixture_data
import test_path_feature as pf_test


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_independent_loss_gradient_shared_masks_and_scale(self):
        torch.manual_seed(8);a=torch.randn(2,128,7,requires_grad=True);b=torch.randn(2,128,7,requires_grad=True);mask=torch.ones_like(a,dtype=torch.bool);mask[:,-1]=False
        ds=torch.tensor([1,16]);stats=dict(y_scale=[2.]*7,delta_scale=[.5]);values=oc.consistency_rows(a,b,mask,mask,ds,stats,False)
        for i,d in enumerate(ds.tolist()):
            x=a[i,d:127].detach().numpy();y=b[i,:127-d].detach().numpy()
            expected=(((np.diff(x[:,0])-np.diff(y[:,0]))*4)**2).mean()+((x[:,1]-y[:,1])**2).mean()+((x[:,2:]-y[:,2:])**2).mean()
            self.assertAlmostEqual(values[i].item(),expected/3,places=4)
        values.sum().backward();self.assertGreater(a.grad.abs().sum(),0);self.assertGreater(b.grad.abs().sum(),0)
        self.assertEqual(a.grad[0,:1].abs().sum(),0);self.assertEqual(a.grad[:,-1].abs().sum(),0);self.assertEqual(b.grad[1,111:].abs().sum(),0)
        aa=a.detach().clone();bb=b.detach().clone();aa[:,:,:1]+=123;bb[:,:,:1]-=78
        torch.testing.assert_close(oc.consistency_rows(aa,bb,mask,mask,ds,stats,False),values,atol=1e-3,rtol=1e-5)
        missing=mask.clone();missing[:,:,2:]=False
        self.assertTrue(torch.isfinite(oc.consistency_rows(a,b,missing,missing,ds,stats)).all())
        wrong=mask.clone();wrong[0,1,3]=False
        with self.assertRaises(ValueError):oc.consistency_rows(a,b,mask,wrong,ds,stats)

    def test_micro_accumulation_equal_exposures_without_weight_updates(self):
        model,_,stats,local=small_model();_,_,_,data=fixture_data(stats);a=data['views'][16];b=data['views'][0];indices=np.arange(3);ds=np.full(3,16);ps=np.tile([32,64,96,128],(3,1))
        class NoUpdate:
            def __init__(self,parameters):self.parameters=list(parameters);self.steps=0
            def zero_grad(self,set_to_none=True):
                for p in self.parameters:p.grad=None
            def step(self):self.steps+=1
        def build(ids,shifts):return ({k:v[ids] for k,v in a.items()},{k:v[ids] for k,v in b.items()})
        outputs=[]
        for micro in (1,3):
            m=copy.deepcopy(model);before=run.ur.bb.state_signature(m);enc=NoUpdate(m.core.parameters());head=NoUpdate(m.local_head.parameters())
            result,steps=oc.run_epoch(m,build,indices,ds,ps,stats,local,3,micro,'cpu',.1,enc,head)
            self.assertEqual(steps,1);self.assertEqual(enc.steps,1);self.assertEqual(before,run.ur.bb.state_signature(m));outputs.append((result,[p.grad.clone() for p in m.parameters() if p.requires_grad]))
        for key in outputs[0][0]:self.assertAlmostEqual(outputs[0][0][key],outputs[1][0][key],places=6)
        for a,b in zip(outputs[0][1],outputs[1][1]):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)

    def test_dense_boundary_and_deterministic_balanced_schedule(self):
        specs=[dict(series=0,lo=0,offset=0,length=800,endpoints=[[639,0]])];original=[dict(key='A',row=639)]
        candidates=[dict(key='A',row=x) for x in (511,575,639,799,800)]+[dict(key='B',row=639)]
        rows,excluded=oc.eligible(specs,original,candidates,800);self.assertEqual([r['index'] for r in rows],[1,2,3]);self.assertEqual(sum(excluded.values()),3)
        a=oc.schedule(100,31,42,1);b=oc.schedule(100,31,42,1)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(len(set(a[0])),31);self.assertLessEqual(abs(sum(a[1]==1)-sum(a[1]==16)),1)
        self.assertFalse(np.array_equal(a[0],oc.schedule(100,31,42,2)[0]))

    def fixture(self,root):
        model,data,stats,local,path_meta,path=pf_test.ContinuationTests().fixture(root)
        path_meta['epochs']=100;path_meta['micro']=1;atomic_json(path_meta,path/'manifest.json');weights={};trials={}
        with patch.object(run.ur.bb,'model_for',side_effect=lambda *a:copy.deepcopy(model)):
            for seed in (42,43):
                job=next(j for j in path_meta['experiments'] if j['name']==f'control_s{seed}');m,enc,head,parent=run.pf.construct(path_meta,job,'cpu',True)
                for g in enc.param_groups:g['lr']=3e-6
                for g in head.param_groups:g['lr']=1e-5
                for opt in (enc,head):
                    for p,state in opt.state.items():state['step']=torch.tensor(17.);state['exp_avg'].fill_(.03)
                expected=run.pf.validation(path_meta,job,m,data,stats,local,'cpu');history=[dict(epoch=e,lr=3e-6,head_lr=1e-5,validation=expected) for e in range(1,101)]
                ck=dict(metadata=dict(manifest=path_meta,job=job),epoch=100,history=history,model=run.ur.bb.ab.cpu_state(m),encoder_optimizer=enc.state_dict(),head_optimizer=head.state_dict(),rng=rng_state())
                cell=path/job['name'];cell.mkdir();atomic_save(ck,cell/'last.pt');atomic_save(dict(metadata=ck['metadata'],epoch=100,validation=expected,model=ck['model']),cell/'best.pt');atomic_json(history,cell/'history.json')
                weights[job['name']]={k:sha256(cell/f'{k}.pt') for k in ('best','last')};trials[job['name']]=dict(selected_epoch=100,validation=expected)
        atomic_json(dict(weights=weights,trials=trials),path/'selection_lock.json')
        identity=dict(manifest=path_meta,coverage='unused',files={});meta=run.make_manifest(path,identity,{s:{} for s in ('train','val','test','cross_research')},dict(root=str(root/'reader'),manifest={}),epochs=2,micro=1);meta['batch']=2;meta['budget']=4
        out=root/'new';out.mkdir();atomic_json(meta,out/'manifest.json');atomic_json(dict(eligible=8,rows=[{}]*8),out/'train_plan.json')
        return model,data,stats,local,meta,out

    def test_restore_last300_moments_and_all_workers_lock_resume(self):
        with tempfile.TemporaryDirectory() as td:
            model,data,stats,local,meta,out=self.fixture(Path(td))
            class Builder:
                plan=[{}]*8
                def __init__(self,*a):pass
            def fake_epoch(model,builder,ids,ds,ps,stats,local,batch,micro,device,weight,enc,head):
                return dict(supervised=1.,consistency=.2,objective=1.+weight*.2),(len(ids)+batch-1)//batch
            with ExitStack() as stack:
                stack.enter_context(patch.object(run.ur.bb,'model_for',side_effect=lambda *a:copy.deepcopy(model)))
                stack.enter_context(patch.object(run.ur.bb,'load_data',return_value=data))
                stack.enter_context(patch.object(run,'prepare'))
                stack.enter_context(patch.object(run,'Builder',Builder))
                stack.enter_context(patch.object(run.oc,'run_epoch',side_effect=fake_epoch))
                stack.enter_context(patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No optimizer updates allowed')))
                m,enc,head,parent=run.construct(meta,meta['experiments'][0],'cpu',True)
                self.assertEqual(enc.param_groups[0]['lr'],3e-6);self.assertEqual(len(enc.param_groups),2)
                for state in enc.state.values():self.assertEqual(state['step'].item(),17.);self.assertTrue(torch.all(state['exp_avg']==.03))
                ev.preflight(meta,out,'cpu')
                calls=0
                def stop(*args):
                    nonlocal calls
                    calls+=1
                    if calls==2:raise RuntimeError('interrupt')
                    return fake_epoch(*args)
                with patch.object(run.oc,'run_epoch',side_effect=stop):
                    with self.assertRaisesRegex(RuntimeError,'interrupt'):run.worker(out,meta['experiments'][0]['name'],'cpu')
                for job in meta['experiments']:run.worker(out,job['name'],'cpu')
                locked=run.lock_selection(meta,out);self.assertEqual(len(locked['trials']),4)
                for seed in (42,43):
                    a=run.read_json(out/f'control_s{seed}/history.json');b=run.read_json(out/f'consistent_s{seed}/history.json');self.assertEqual([v['sampling'] for v in a],[v['sampling'] for v in b])
                ck=torch.load(out/'control_s42/last.pt',weights_only=True);self.assertEqual(ck['best_epoch'],0)
                for n in ('completion.json','best.pt','history.json'):(out/'control_s42'/n).unlink()
                run.worker(out,'control_s42','cpu');run.lock_selection(meta,out)
                # Real evaluation path on synthetic arrays and manually specified frozen heads.
                reader=Path(meta['reader']['root']);(reader/'cache').mkdir(parents=True);atomic_json({},reader/'manifest.json')
                width=model.core.encoder(torch.tensor(data['x'][:1])).shape[-1]
                target_stats=dict(mean=[0.]*13,scale=[1.]*13)
                fitted=dict(target_stats=target_stats,heads={f'control_s{s}':dict(targets=[dict(statistics=dict(mean=[0.]*width,scale=[1.]*width)) for _ in range(13)]) for s in (42,43)})
                atomic_json(fitted,reader/'fit.json');heads={}
                for seed in (42,43):heads.update({f'control_s{seed}_weights':np.zeros((width,13)),f'control_s{seed}_intercepts':np.zeros(13)})
                np.savez(reader/'heads.npz',**heads)
                target,mask=run.ur.up.targets(run.ur.up.restore_raw(data['x'],stats));prior={};inventories={}
                for split in odr_splits():
                    np.savez(reader/f'cache/{split}.npz',raw=data['x'].reshape(len(target),-1),targets=target,mask=mask)
                    atomic_json(dict(manifest_sha256=sha256(reader/'manifest.json'),files={f'{split}.npz':sha256(reader/f'cache/{split}.npz')}),reader/f'cache/{split}_index.json')
                    score,_=run.ur.up.measure(np.zeros_like(target),target,mask,target_stats);prior[split]=dict(scores={f'control_s{s}':score for s in (42,43)})
                    inventories[split]=[dict(key='A',symbol='A',period=15,month='2026-01',week=f'w{i}',end=f'2026-01-{i+1:02d}') for i in range(len(target))];atomic_json(inventories[split],out/f'{split}_inventory.json')
                atomic_json(dict(datasets=prior),reader/'utility_metrics.json')
                paired=fixture_data(stats)[3]
                with patch.object(run.odr,'prepare',return_value=({s:paired for s in odr_splits()},stats)):
                    ev.evaluate(meta,out,'cpu')
                self.assertEqual(run.read_json(out/'decision.json')['status'],'no_full_upgrade')
                self.assertEqual(len(run.read_json(out/'trained_validation_causality.json')),10)
                self.assertTrue((out/'test_consistent_s43_last.json').exists())
                ck['history'][0]['views']+=1
                with self.assertRaisesRegex(ValueError,'budget'):run.verify_history(meta,meta['experiments'][0],ck,8)

    def test_nested_frozen_reader_replay_and_real_error_gate(self):
        value=dict(targets=[dict(name='x',r2=None,nmse=.1)],direction=dict(all_classes_present=True,confusion=[[1,2],[3,4]]))
        ev.require_nested(value,copy.deepcopy(value))
        bad=copy.deepcopy(value);bad['targets'][0]['nmse']=.2
        with self.assertRaises(ValueError):ev.require_nested(value,bad)
        # Build the full decision matrix; zero agreement alone cannot hide bad reconstruction.
        rows=[dict(key='A',symbol='A',period=15,month='2026-01',week=f'w{i//10}') for i in range(60)];y=[.01]*60
        def record(gap,err):
            errors={t:{k:[err]*60 for k in ('primary','path','changes','body','activity')} for t in ('global','held','recent')}
            utility=dict(errors={'utility':[err]*60}|{n:y for n in run.ur.up.NAMES[6:]},scores=dict(targets=[dict(r2=.99)]*13))
            vals={'combined_gap':[gap]*60,'path_shape_gap_bps':[gap]*60}|{f'{k}_{side}'+s:[err]*60 for k,s in [('combined_error',''),('path_native_mae','_bps')] for side in ('a','b')}
            return dict(reconstruction=dict(errors=errors),utility=utility,overlap={str(d):dict(per_pair=copy.deepcopy(vals),valid={k:[True]*60 for k in vals}) for d in (1,16,64)})
        meta=dict(decision=dict(gap_gain=.1,reconstruction_retention=.05,family_retention=.1,utility_retention=.05,current_nmse=.1,current_r2=.8),evaluation_shifts=[1,16,64])
        bank={f'frozen_s{s}':record(1.,.1) for s in (42,43)}|{f'{mode}_s{s}/{kind}':record(.5 if mode=='consistent' else 1.,.1) for s in (42,43) for kind in ('best','last') for mode in ('control','consistent')}
        records={s:copy.deepcopy(bank) for s in odr_splits()};inventory={s:rows for s in records};masks={s:np.ones((60,13),bool) for s in records}
        self.assertEqual(ev.checks(meta,records,inventory,inventory,masks)['status'],'candidate_for_review')
        records['test']['consistent_s42/best']['reconstruction']['errors']['global']['primary']=[.4]*60
        self.assertEqual(ev.checks(meta,records,inventory,inventory,masks)['status'],'no_full_upgrade')

    def test_dense_prepare_replay_builder_and_cache_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,_,stats,_=small_model();raw,specs,rows,data=fixture_data(stats)
            bank=root/'bank';bank.mkdir();np.save(bank/'train_x.npy',raw);atomic_json(specs,bank/'train_sequences.json')
            sampling=root/'sampling';(sampling/'candidates').mkdir(parents=True);atomic_json(rows,sampling/'candidates/inventory.json')
            out=root/'out';out.mkdir();meta=dict(budget=2,packed=dict(train=dict(directory=str(bank))));atomic_json(meta,out/'manifest.json')
            pool={k:v.copy() for k,v in data['views'][0].items()}
            with ExitStack() as stack:
                stack.enter_context(patch.object(run,'context',return_value={}))
                stack.enter_context(patch.object(run,'alignment',return_value=dict(sampling_source=str(sampling))))
                stack.enter_context(patch.object(run,'statistics',return_value=(stats,{})))
                stack.enter_context(patch.object(run.ur,'inventories',return_value={s:rows for s in ('train','val','test','cross_research')}))
                stack.enter_context(patch.object(run.ur.up.probe,'inventory_audit',return_value={}))
                stack.enter_context(patch.object(run.ur.bb,'load_data',return_value=pool))
                run.prepare(meta,out);run.prepare(meta,out);self.assertEqual(run.read_json(out/'train_plan.json')['replay'],dict(x=0.,y=0.,mask=0.))
                builder=run.Builder(meta,out);a,b=builder(np.array([0,1]),np.array([1,16]));np.testing.assert_array_equal(a['x'][0],data['views'][1]['x'][0]);np.testing.assert_array_equal(a['x'][1],data['views'][16]['x'][1])
                plan=out/'train_plan.json';plan.write_text(plan.read_text()+' ')
                with self.assertRaises(ValueError):run.prepare(meta,out)
                pool['x'][0,0,0]+=1;other=root/'other';other.mkdir();atomic_json(meta,other/'manifest.json')
                with self.assertRaisesRegex(ValueError,'Dense packed replay'):run.prepare(meta,other)

    def test_export_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';out.mkdir();atomic_json(dict(status='failed'),out/'failure.json');(out/'model.pt').write_bytes(b'omit');fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_CONSISTENCY_RUN=str(out),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN=str(fake),BABEL_CONSISTENCY_LOG=str(root/'none'))
            script=Path(__file__).resolve().parents[2]/'scripts/babel_consistency768_autodl.sh'
            for mode,code in [('all',7),('export',0)]:
                p=subprocess.run(['bash',str(script),mode],env=env,capture_output=True,text=True);self.assertEqual(p.returncode,code,p.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:self.assertFalse(any(n.endswith('.pt') for n in t.getnames()));self.assertIn('run_status=failed',t.extractfile('run/run_status.txt').read().decode())


def odr_splits():return ('test','cross_research')

if __name__=='__main__':unittest.main()
