"""Synthetic cache, freeze, budget, resume and locked evaluation tests; no neural updates."""
import copy
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch,MagicMock
import numpy as np
import torch
from obson.babel import local_readapt as lr,local_readapt_run as r,capacity_growth as growth
from obson.babel.ae_extend import atomic_json,atomic_save
from obson.babel.dual_state import sha256
import test_capacity_growth as growth_tests


class LocalReadaptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_schedule_train_only_scales_masks_and_gradients(self):
        order,ps=lr.schedule(13,42,1);self.assertEqual(sorted(order.tolist()),list(range(13)));self.assertFalse(np.isin(ps,lr.ba.HELD_PREFIXES).any())
        for epoch in range(1,101):
            a,b=lr.schedule(13,42,epoch);c,d=lr.schedule(13,42,epoch);np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
            self.assertFalse(np.isin(b,lr.ba.HELD_PREFIXES).any());self.assertTrue((np.diff(b,axis=1)>0).all())
        rng=np.random.default_rng(42);x=rng.normal(size=(13,94,8)).astype('float32');stats=lr.scales(x,batch=3)
        np.testing.assert_allclose(stats['mean'],x.astype(float).mean((0,1)),atol=1e-12);np.testing.assert_allclose(stats['scale'],x.astype(float).std((0,1)))
        data={'x':torch.tensor(x),'y':torch.tensor(rng.normal(size=(13,94,16,7)).astype('float32')),'mask':torch.ones(13,94,16,7,dtype=torch.bool)}
        data['mask'][...,4]=False;head=lr.reader(8,8,42,'cpu');before=r.ur.bb.state_signature(head);opt=torch.optim.AdamW(head.parameters())
        with patch.object(opt,'step') as step:
            report=lr.epoch(head,data,stats,lr.ba.TRAIN_PREFIXES,order,ps,4,opt);self.assertEqual(step.call_count,4)
        self.assertEqual(report['prefix_exposures'],52);self.assertEqual(before,r.ur.bb.state_signature(head));self.assertIsNone(data['x'].grad)
        self.assertTrue(all(p.grad is not None for p in head.parameters()))
        old=lr.epoch(head,data,stats,lr.ba.TRAIN_PREFIXES,order,ps,4)['smooth'];data['y'][...,4]=10000
        self.assertEqual(old,lr.epoch(head,data,stats,lr.ba.TRAIN_PREFIXES,order,ps,4)['smooth'])
        bad=ps.copy();bad[:,0]=48
        with self.assertRaises(ValueError):lr.epoch(head,data,stats,lr.ba.TRAIN_PREFIXES,order,bad,4)
        data['x'].requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'frozen'):lr.epoch(head,data,stats,lr.ba.TRAIN_PREFIXES,order,ps,4)

    def setup_pipeline(self,root,stack):
        parent,data,stats,local=growth_tests.small();source=root/'source';source.mkdir();out=root/'run';out.mkdir()
        jobs=[dict(name=f'{v}_s{s}',variant=v,seed=s) for s in (42,43) for v in ('base','deep')]
        meta=dict(schema=r.SCHEMA,code_sha256=r.code_identity(),source=str(source),identity=dict(manifest={}),experiments=jobs,epochs=6,batch=4,extract_batch=4,width=8,hidden=8,lr=.001,train_prefixes=list(lr.ba.TRAIN_PREFIXES),val_prefixes=list(lr.ba.VAL_PREFIXES),held_prefixes=list(lr.ba.HELD_PREFIXES))
        atomic_json(meta,out/'manifest.json');atomic_json({'status':'no_capacity_upgrade','original_utility_protocol':{}},source/'decision.json')
        models={};endpoints={}
        for j in jobs:
            m=copy.deepcopy(parent)
            if j['variant']=='deep':growth.install(m,4,32,j['seed'])
            m.eval().requires_grad_(False);models[j['name']]=m
            endpoints[j['name']+'_best']=m.core.encoder(torch.tensor(data['x'])).detach().numpy()[:,-1]
        stack.enter_context(patch.object(r.gr,'parent_meta',return_value={}))
        stack.enter_context(patch.object(r.gr.cr,'alignment',return_value={}))
        stack.enter_context(patch.object(r.gr,'statistics',return_value=(stats,local)))
        stack.enter_context(patch.object(r.ur.bb,'load_data',return_value=data))
        stack.enter_context(patch.object(r.gr.rr.old,'verify_cache'))
        stack.enter_context(patch.object(r.gr.rr.old,'arrays',return_value=endpoints))
        stack.enter_context(patch.object(r.gr,'validation',return_value={'primary':1.}))
        meta['identity']['manifest']['experiments']=jobs;atomic_json(meta,out/'manifest.json')
        stack.enter_context(patch.object(r,'original_model',side_effect=lambda meta,j,dev:(copy.deepcopy(models[j['name']]),{'primary':1.})))
        for s in ('train','val','test','cross_research'):
            rows=[dict(index=i,symbol='x',period=15,month='m1',week=f'w{i//2}') for i in range(12)]
            atomic_json(rows,source/f'{s}_inventory.json');atomic_json(rows,out/f'{s}_inventory.json')
            if s in ('test','cross_research'):
                for j in jobs:
                    scores,_,_,_=r.ur.pf.score(models[j['name']],data,stats,local,4,'cpu');atomic_json(dict(reconstruction=dict(scores=scores)),source/f'{s}_{j["name"]}_best.json')
        return meta,out,models,data

    def test_cache_endpoints_targets_and_no_held_selection(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            meta,out,models,data=self.setup_pipeline(Path(td),stack)
            with self.assertRaises(FileNotFoundError):r.prepare(meta,out,'test','cpu')
            r.prepare(meta,out,'train','cpu');r.prepare(meta,out,'val','cpu')
            self.assertEqual(r.verify_cache(meta,out,'train')['positions'],list(lr.ba.TRAIN_PREFIXES));self.assertEqual(r.verify_cache(meta,out,'val')['positions'],list(lr.ba.VAL_PREFIXES))
            x=r.arrays(out,'train','base_s42')['x'];m=models['base_s42'];p=37;i=meta['train_prefixes'].index(p)
            with torch.no_grad():z=m.core.encoder(torch.tensor(data['x'][:,:p]))[:,-1]
            np.testing.assert_allclose(x[:,i],z.numpy(),rtol=2e-5,atol=5e-5)
            with patch.object(r,'original_model',side_effect=AssertionError('No repeated inference')):r.prepare(meta,out,'train','cpu')
            (out/'cache/val_y.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):r.verify_cache(meta,out,'val')

    def test_four_workers_resume_equal_initialization_lock_and_full_evaluation(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            meta,out,models,data=self.setup_pipeline(Path(td),stack)
            for s in ('train','val'):r.prepare(meta,out,s,'cpu')
            signatures={k:r.ur.bb.state_signature(v) for k,v in models.items()};original_epoch=lr.epoch;calls=[]
            def interrupted(*a,**k):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('simulated interruption')
                return original_epoch(*a,**k)
            stack.enter_context(patch.object(torch.optim.AdamW,'step',return_value=None))
            with patch.object(lr,'epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interruption'):r.worker(out,'base_s42','cpu')
            self.assertEqual(torch.load(out/'base_s42/last.pt',weights_only=True)['epoch'],1)
            # Nonzero moments are injected directly: verify restoration without
            # taking any neural optimizer step on the local machine.
            resume=torch.load(out/'base_s42/last.pt',weights_only=True)
            for ident,value in zip(resume['optimizer']['param_groups'][0]['params'],resume['model'].values()):
                resume['optimizer']['state'][ident]=dict(step=torch.tensor(1.),exp_avg=torch.full_like(value,.001),exp_avg_sq=torch.full_like(value,.002))
            r.publish(resume,out/'base_s42')
            for j in meta['experiments']:r.worker(out,j['name'],'cpu')
            restored=torch.load(out/'base_s42/last.pt',weights_only=True)['optimizer']
            for ident,value in resume['optimizer']['state'].items():
                for k,tensor in value.items():torch.testing.assert_close(restored['state'][ident][k],tensor)
            lock=r.lock_selection(meta,out);r.check_selection(meta,out)
            for j in meta['experiments']:
                q=r.read_json(out/j['name']/'training_summary.json');self.assertEqual(q['selected_epoch'],0);self.assertEqual(q['steps'],18);self.assertEqual(q['prefix_exposures'],288)
                with patch.object(lr,'epoch',side_effect=AssertionError('No repeats')):r.worker(out,j['name'],'cpu')
            for seed in (42,43):
                a=r.read_json(out/f'base_s{seed}/initial_validation.json');b=r.read_json(out/f'deep_s{seed}/initial_validation.json');self.assertEqual(a['initial_signature'],b['initial_signature'])
                a=r.read_json(out/f'base_s{seed}/history.json');b=r.read_json(out/f'deep_s{seed}/history.json');self.assertEqual([v['sampling'] for v in a],[v['sampling'] for v in b])
            for s in ('test','cross_research'):r.prepare(meta,out,s,'cpu')
            r.evaluate(meta,out,'cpu');d=r.read_json(out/'decision.json');self.assertFalse(d['automatic_promotion']);self.assertEqual(len(d['checks']),120)
            self.assertEqual(len(r.read_json(out/'frozen_replay.json')),8);self.assertEqual(signatures,{k:r.ur.bb.state_signature(v) for k,v in models.items()})
            self.assertEqual(len(r.read_json(out/'local_readapt_metrics.json')['summary']['test']),12)
            path=out/'deep_s43/best.pt';path.write_bytes(b'bad')
            with self.assertRaises(ValueError):r.check_selection(meta,out)

    def test_decision_requires_both_seeds_sets_readers_and_old_heads(self):
        rows={s:[dict(week=f'w{i//10}',symbol='x',period=15,month='m1') for i in range(60)] for s in ('test','cross_research')};records={}
        for s in rows:
            records[s]={}
            for seed in (42,43):
                for mode in ('base','deep'):
                    for kind in ('original','best','last'):
                        records[s][f'{mode}_s{seed}_{kind}']=dict(errors={task:{key:[.9 if mode=='deep' and kind!='original' else 1.]*60 for key in ('primary','path','body','activity')} for task in ('trained','held')})
        self.assertEqual(r.decide(records,rows)['status'],'local_readout_retention_recovered')
        records['test']['deep_s43_last']['errors']['held']['body']=[1.3]*60
        self.assertEqual(r.decide(records,rows)['status'],'local_readout_retention_unresolved')
        self.assertEqual(r.decide({},rows)['status'],'local_readout_retention_unresolved')

    def test_output_ancestry_and_worker_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';out=root/'out';source.mkdir();out.mkdir()
            with self.assertRaises(ValueError):r.check_output(source,source/'nested')
            atomic_json({'experiments':[dict(name='a'),dict(name='b')]},out/'manifest.json')
            bad=MagicMock();bad.poll.return_value=1;bad.returncode=1;active=MagicMock();active.poll.return_value=None
            with patch.object(r.subprocess,'Popen',side_effect=[bad,active]),patch.object(r,'progress'):
                with self.assertRaises(RuntimeError):r.run_jobs(out,2)
            active.terminate.assert_called_once();active.wait.assert_called_once()

    def test_controller_locks_before_research_and_completed_run_skips(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            meta,out,_,_=self.setup_pipeline(Path(td),stack);source=Path(meta['source']);events=[]
            atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__),source/'runtime.json')
            stack.enter_context(patch.object(r,'source_identity',return_value=meta['identity']))
            stack.enter_context(patch.object(r,'check_output'))
            stack.enter_context(patch.object(r,'make_manifest',return_value=meta))
            stack.enter_context(patch.object(r.gr.rr.up.probe,'inventory_audit',return_value={'passed':True}))
            stack.enter_context(patch.object(r,'prepare',side_effect=lambda meta,out,s,device:events.append(s)))
            stack.enter_context(patch.object(r,'run_jobs',side_effect=lambda *a:events.append('training')))
            stack.enter_context(patch.object(r,'lock_selection',side_effect=lambda *a:events.append('lock')))
            stack.enter_context(patch.object(r,'evaluate',side_effect=lambda *a:events.append('evaluate')))
            stack.enter_context(patch.object(r,'check_selection',side_effect=lambda *a:events.append('verify_lock')))
            r.run(source,out,2,'cpu')
            self.assertEqual(events,['train','val','training','lock','test','cross_research','evaluate'])
            r.run(source,out,2,'cpu');self.assertEqual(events[-1],'verify_lock');self.assertEqual(events.count('training'),1)

    def test_failure_archive_contains_reports_not_binaries(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'run/base_s42').mkdir(parents=True);(root/'run/base_s42/best.pt').write_bytes(b'omit');atomic_json([],root/'run/base_s42/history.json')
            env=os.environ|dict(BABEL_LOCAL_READAPT_RUN=str(root/'run'),BABEL_DOWNLOAD_DIR=str(root/'download'),BABEL_LOCAL_READAPT_LOG=str(root/'missing'),PYTHON_BIN='/usr/bin/false')
            result=subprocess.run(['bash','scripts/babel_local_readapt768_autodl.sh','all'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            with tarfile.open(root/'download/run_reports.tar.gz') as t:
                self.assertIn('run/base_s42/history.json',t.getnames());self.assertNotIn('run/base_s42/best.pt',t.getnames());self.assertIn('run_status=failed',t.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
