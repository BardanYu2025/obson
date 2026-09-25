"""Growth math, gradients, matched budgets and lifecycle without neural updates."""
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
from obson.babel import capacity_growth as g,capacity_growth_run as r,capacity_growth_evaluate as ev
from obson.babel import bar_alignment as ba,architecture as ar,architecture_benchmark as ab,pca_teacher as pt,prefix_readout as pr
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features
from test_overlap_diagnostic import fixture_data
import test_consistency_readout as reader_tests


def small():
    x=features(12);y,m=ar.ordered_targets(x);stats=ar.fit_scales(x,y,m);xx,yy,mm=ar.normalize(x,y,m,stats);data=dict(x=xx,y=yy,mask=mm)
    pca=ab.fit_pca(data,8);scales=pt.fit_coordinate_scales(pt.coefficients(pca,data));config=dict(pt.CONFIG,latent=8,heads=8,attention_layers=2,attention_ff=16,residual_width=8)
    model=ba.AlignedStudent(config,42,pca,scales,hidden=8);local=pr.fit_target_scales([pr.targets(data,stats,p) for p in ba.VAL_PREFIXES]);return model,data,stats,local


class GrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_widen_identity_outputs_causality_and_live_new_gradients(self):
        parent,data,stats,local=small();parent.eval();x=torch.tensor(data['x'][:3]);ps=torch.tensor([[32,64,96,128]]*3);base=parent.core.encoder(x).detach();a,b=parent(x,ps,True)
        for layers,ff in ((2,16),(2,32),(4,32)):
            model=copy.deepcopy(parent);info=g.install(model,layers,ff,42);model.eval();z=model.core.encoder(x)
            torch.testing.assert_close(z,base,atol=2e-6,rtol=2e-5);aa,bb=model(x,ps,True);torch.testing.assert_close(a,aa);torch.testing.assert_close(b,bb)
            self.assertEqual(info['decoder_trainable_parameters'],0);self.assertEqual(len(model.core.encoder.backbone.layers),layers)
            model.train();model.zero_grad();z=model.core.encoder(x);z.square().mean().backward()
            if ff>16:self.assertGreater(float(model.core.encoder.backbone.layers[0].linear2.weight.grad[:,16:].abs().sum()),0)
            if layers>2:
                layer=model.core.encoder.backbone.layers[-1];self.assertGreater(float(layer.self_attn.out_proj.weight.grad.abs().sum()),0);self.assertGreater(float(layer.linear2.weight.grad.abs().sum()),0)
                # Activate the new residual paths directly, without an optimizer step,
                # to test causality and gradients beyond the zero-output start.
                with torch.no_grad():layer.self_attn.out_proj.weight.fill_(.01);layer.linear2.weight.fill_(.01)
                model.zero_grad();model.core.encoder(x).square().mean().backward();self.assertGreater(float(layer.self_attn.in_proj_weight.grad.abs().sum()),0)
            model.eval();future=x.clone();future[:,64:]+=17
            torch.testing.assert_close(model.core.encoder(x)[:,:64],model.core.encoder(future)[:,:64],atol=2e-6,rtol=2e-5)
            self.assertTrue(all(p.grad is None for p in model.core.decoder.parameters()))

    def fixture(self,root):
        parent,data,stats,local=small();source=root/'source';source.mkdir();out=root/'out';out.mkdir()
        jobs=[dict(name=f'{name}_s{s}',seed=s,variant=name,layers=depth,ff=ff) for s in (42,43) for name,depth,ff in [('base',2,16),('wide',2,32),('deep',4,32)]]
        meta=dict(schema=r.SCHEMA,code_sha256=r.code_identity(),source=str(source),identity=dict(manifest=dict(source=str(source),identity=dict(manifest={}))),experiments=jobs,epochs=6,warmup=5,encoder_lr=1e-5,head_lr=3e-5,budget=4,batch=2,micro=1,weight=.1,evaluation_batch=2,state_width=8)
        atomic_json(meta,out/'manifest.json');atomic_json(dict(eligible=8),source/'train_plan.json');return parent,data,stats,local,meta,out

    def test_six_workers_reset_equal_schedule_interrupt_resume_and_epoch0(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            parent,data,stats,local,meta,out=self.fixture(Path(td));val=dict(selection=1.,primary=.5,path=.2);_,_,_,views=fixture_data(stats)
            class Builder:
                plan=list(range(8))
                def __init__(self,*a):pass
                def __call__(self,ids,ds):return ({k:np.stack([views['views'][int(d)][k][i%3] for i,d in zip(ids,ds)]) for k in ('x','y','mask')},{k:views['views'][0][k][np.asarray(ids)%3] for k in ('x','y','mask')})
            original=r.cr.oc.run_epoch
            def no_update(model,builder,ids,ds,ps,stats,local,batch,micro,device,weight,enc,head):
                result,_=original(model,builder,ids,ds,ps,stats,local,batch,micro,device,weight)
                return result,(len(ids)+batch-1)//batch
            stack.enter_context(patch.object(r.rr,'load_selected',side_effect=lambda *a:(copy.deepcopy(parent),{},val)))
            stack.enter_context(patch.object(r,'validation',return_value=val));stack.enter_context(patch.object(r,'statistics',return_value=(stats,local)));stack.enter_context(patch.object(r.cr,'Builder',Builder));stack.enter_context(patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural update allowed')))
            for job in meta['experiments']:
                model,enc,head,expected,info=r.construct(meta,job,'cpu',True);self.assertFalse(enc.state);self.assertFalse(head.state)
            calls=[]
            def interrupt(*a):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('interrupted')
                return no_update(*a)
            with patch.object(r.cr.oc,'run_epoch',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):r.worker(out,'base_s42','cpu')
            with patch.object(r.cr.oc,'run_epoch',side_effect=no_update):
                for job in meta['experiments']:r.worker(out,job['name'],'cpu')
            lock=r.lock_selection(meta,out);self.assertEqual(len(lock['trials']),6)
            for seed in (42,43):
                hs=[r.read_json(out/f'{n}_s{seed}/history.json') for n in ('base','wide','deep')]
                self.assertEqual([[v['sampling'] for v in h] for h in hs],[ [v['sampling'] for v in hs[0]]]*3)
            for job in meta['experiments']:
                s=lock['trials'][job['name']];self.assertEqual(s['selected_epoch'],0);self.assertEqual(s['encoder_steps'],12);self.assertEqual(s['pairs'],24);self.assertTrue(s['decoder_unchanged'])
            with patch.object(r.cr.oc,'run_epoch',side_effect=AssertionError('Must not retrain')):r.worker(out,'deep_s43','cpu')
            (out/'deep_s43/best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):r.lock_selection(meta,out)

    def test_schedule_lr_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            _,_,_,_,meta,out=self.fixture(Path(td));a=r.plan(meta,meta['experiments'][0],1,8);b=r.plan(meta,meta['experiments'][2],1,8)
            for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
            self.assertEqual(a[3]['absolute_epoch'],401);self.assertEqual(set(a[1]),{1,16});self.assertEqual(len(set(a[0])),4)
            self.assertAlmostEqual(g.learning_rate(1,100,1e-5),2e-6);self.assertEqual(g.learning_rate(5,100,1e-5),1e-5);self.assertAlmostEqual(g.learning_rate(100,100,1e-5),1e-6)
            state=dict(epoch=1,history=[dict(epoch=1,sampling=a[3],pairs=4,views=8,encoder_steps=2,head_steps=2,encoder_lr=2e-6,head_lr=6e-6,validation=dict(selection=1))],initial_validation=dict(selection=2),best_epoch=1,best_validation=dict(selection=1))
            r.verify_history(meta,meta['experiments'][0],state,8)
            state['history'][0]['views']=4
            with self.assertRaises(ValueError):r.verify_history(meta,meta['experiments'][0],state,8)

    def test_decision_needs_gain_retention_both_seeds_and_last(self):
        rows=[dict(symbol='A',period=15,month='m',week=f'w{i//10}') for i in range(60)];mask=np.ones((60,13),bool)
        def record(error):
            errors={t:{k:[error]*60 for k in ('primary','path','changes','body','activity')} for t in ('global','held','recent')}
            utility=dict(scores=dict(targets=[dict(r2=.95) for _ in range(13)]),errors={k:[.03]*60 for k in list(rr.up.NAMES)+list(rr.up.GROUPS)})
            keys=('combined_gap','combined_error_a','combined_error_b','path_shape_gap_bps','path_native_mae_a_bps','path_native_mae_b_bps');overlap={str(d):dict(per_pair={k:[.03]*60 for k in keys},valid={k:[True]*60 for k in keys}) for d in (1,16,64)}
            return dict(reconstruction=dict(errors=errors),utility=utility,overlap=overlap)
        rr=r.rr;bank={f'parent_s{s}':record(.12) for s in (42,43)}
        for s in (42,43):
            for k in ('best','last'):
                for mode,err in [('base',.1),('wide',.08),('deep',.06)]:bank[f'{mode}_s{s}_{k}']=record(err)
        records={s:copy.deepcopy(bank) for s in ('test','cross_research')};rs={s:rows for s in records};ms={s:mask for s in records}
        self.assertEqual(ev.decide({},records,rs,rs,ms)['status'],'deeper_candidate')
        records['test']['deep_s43_last']['utility']['errors']['utility']=[1.]*60
        self.assertEqual(ev.decide({},records,rs,rs,ms)['status'],'wider_candidate')
        records['test']['wide_s42_best']['overlap']['64']['per_pair']['combined_gap']=[1.]*60
        self.assertEqual(ev.decide({},records,rs,rs,ms)['status'],'no_capacity_upgrade')

    def test_readout_fit_locks_all12_without_research_and_rejects_changed_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            oldmeta,parent,rows,banks=reader_tests.ReadoutTests().fixture(Path(td));r.rr.fit(oldmeta,parent,'cpu')
            out=Path(td)/'growth';(out/'cache').mkdir(parents=True);jobs=[dict(name=f'{v}_s{s}') for s in (42,43) for v in ('base','wide','deep')];meta=dict(source=str(parent),experiments=jobs)
            atomic_json(meta,out/'manifest.json');weights={}
            for job in jobs:
                p=out/job['name'];p.mkdir();weights[job['name']]={}
                for k in ('best','last'):(p/f'{k}.pt').write_bytes(b'locked');weights[job['name']][k]=sha256(p/f'{k}.pt')
            atomic_json(dict(manifest=meta,trials={j['name']:{} for j in jobs},weights=weights),out/'model_selection_lock.json')
            for split in ('train','val'):
                bank=banks[split];values={k:bank[k] for k in ('targets','mask')}
                for n,_,_ in ev.entries(meta):values[n]=bank['consistent_s42']
                np.savez(out/f'cache/{split}.npz',**values);atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),files={f'{split}.npz':sha256(out/f'cache/{split}.npz')}),out/f'cache/{split}_index.json')
            with self.assertRaises(FileNotFoundError):ev.prepare(meta,out,'test','cpu')
            ev.fit(meta,out,'cpu');self.assertEqual(r.read_json(out/'fit.json')['candidates'],780);ev.check_readouts(out)
            with patch.object(ev.up,'fit_heads',side_effect=AssertionError('No repeated fits')):ev.fit(meta,out,'cpu')
            (out/'cache/train.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):ev.check_readouts(out)
            selection=r.read_json(out/'model_selection_lock.json');selection['weights'].pop(jobs[0]['name']);atomic_json(selection,out/'model_selection_lock.json')
            with self.assertRaisesRegex(ValueError,'All model selections'):ev.check_models(out)

    def test_manager_progress_and_failed_worker_cleanup(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as td:
            _,_,_,_,meta,out=self.fixture(Path(td));children=[]
            for job in meta['experiments']:
                path=out/job['name'];path.mkdir();atomic_json(dict(epoch=1,epochs=6,best_epoch=0,seconds=2.),path/'progress.json')
                child=MagicMock();child.poll.return_value=0;child.returncode=0;children.append(child)
            with patch.object(r.subprocess,'Popen',side_effect=children),patch.object(r,'progress') as log:
                r.run_jobs(out,2)
                self.assertEqual(sum('epoch=1/6' in c.args[0] for c in log.call_args_list),6)
            failed=MagicMock();failed.poll.return_value=1;failed.returncode=1
            running=MagicMock();running.poll.return_value=None
            with patch.object(r.subprocess,'Popen',side_effect=[failed,running]),patch.object(r,'progress'):
                with self.assertRaisesRegex(RuntimeError,'exit1'):r.run_jobs(out,2)
            running.terminate.assert_called_once();running.wait.assert_called_once()

    def test_complete14_state_evaluation_parent_and_baseline_replay(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            oldmeta,parent,rows,banks=reader_tests.ReadoutTests().fixture(Path(td));r.rr.fit(oldmeta,parent,'cpu');r.rr.evaluate(oldmeta,parent,rows)
            original_source=Path(oldmeta['source']);cm=copy.deepcopy(oldmeta['identity']['manifest']);cm.update(source='unused',identity={},packed={s:{} for s in ('test','cross_research')})
            rm=copy.deepcopy(oldmeta);rm['identity']['manifest']=cm
            out=Path(td)/'growth';(out/'cache').mkdir(parents=True);jobs=[dict(name=f'{v}_s{s}',seed=s) for s in (42,43) for v in ('base','wide','deep')]
            meta=dict(source=str(parent),identity=dict(manifest=rm),experiments=jobs,evaluation_batch=128)
            atomic_json(meta,out/'manifest.json');weights={}
            for j in jobs:
                cell=out/j['name'];cell.mkdir();weights[j['name']]={}
                for kind in ('best','last'):(cell/f'{kind}.pt').write_bytes(b'locked');weights[j['name']][kind]=sha256(cell/f'{kind}.pt')
            atomic_json(dict(manifest=meta,trials={j['name']:{} for j in jobs},weights=weights),out/'model_selection_lock.json')
            for split,bank in banks.items():
                values={k:bank[k] for k in ('raw','current','targets','mask')}
                for seed in (42,43):values[f'parent_s{seed}']=bank[f'consistent_s{seed}']
                for name,j,kind in ev.entries(meta):values[name]=bank[f'consistent_s{j["seed"]}']
                np.savez(out/f'cache/{split}.npz',**values);atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),files={f'{split}.npz':sha256(out/f'cache/{split}.npz')}),out/f'cache/{split}_index.json');atomic_json(rows[split],out/f'{split}_inventory.json')
            ev.fit(meta,out,'cpu')
            def model(name):
                m=torch.nn.Module();m.label=name;return m
            def score(m,*args):
                value=.12 if m.label=='parent' else .1 if m.label.startswith('base') else .08 if m.label.startswith('wide') else .06
                err={t:{k:np.full(72,value) for k in ('primary','path','changes','body','activity')} for t in ('global','held','recent')}
                scores={t:dict(metrics={k:float(v.mean()) for k,v in e.items()}) for t,e in err.items()}
                return scores,err,{'path':np.zeros(72)},np.zeros((72,128,7))
            def overlap(*args):
                keys=('combined_gap','combined_error_a','combined_error_b','path_shape_gap_bps','path_native_mae_a_bps','path_native_mae_b_bps')
                return {str(d):dict(per_pair={k:[.03]*72 for k in keys},valid={k:[True]*72 for k in keys},summary={'all':{'combined_gap':{'mean':.03}}}) for d in (1,16,64)}
            for split in ('test','cross_research'):
                for seed in (42,43):atomic_json(dict(reconstruction={'scores':score(model('parent'))[0]},overlap=overlap()),original_source/f'{split}_consistent_s{seed}_best.json')
            stack.enter_context(patch.object(r,'statistics',return_value=({},{})));stack.enter_context(patch.object(r.cr,'alignment',return_value={}))
            stack.enter_context(patch.object(r.cr.odr,'make_manifest',return_value={}));stack.enter_context(patch.object(r.cr.odr,'prepare',return_value=({s:{'rows':rows[s]} for s in ('test','cross_research')},{})))
            stack.enter_context(patch.object(r.ur.bb,'load_data',return_value={'y':np.zeros((72,128,7)),'mask':np.ones((72,128,7),bool)}))
            stack.enter_context(patch.object(r.rr,'load_selected',side_effect=lambda *a:(model('parent'),{},{})))
            stack.enter_context(patch.object(ev,'load',side_effect=lambda meta,out,j,kind,device:(model(j['name']),{})))
            stack.enter_context(patch.object(r.ur.pf,'score',side_effect=score));stack.enter_context(patch.object(ev.ce,'overlap',side_effect=overlap))
            ev.evaluate(meta,out,'cpu');d=r.read_json(out/'decision.json');self.assertEqual(d['status'],'deeper_candidate');self.assertFalse(d['automatic_promotion'])
            self.assertEqual(len(list(out.glob('test_*_best.json'))),6);self.assertEqual(len(list(out.glob('cross_research_*_last.json'))),6)
            report=r.read_json(out/'growth_metrics.json');self.assertEqual(len(report['summary']['test']),14)
            # Repeated scoring recomputes, and a mismatched legacy capability
            # certificate is rejected rather than silently accepted.
            prior=r.read_json(parent/'readout_metrics.json');prior['datasets']['test']['scores']['consistent_s42']['groups']['utility']+=1;atomic_json(prior,parent/'readout_metrics.json')
            with self.assertRaises(ValueError):ev.evaluate(meta,out,'cpu')

    def test_failure_export_has_worker_history_no_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'run/deep_s42').mkdir(parents=True);(root/'run/deep_s42/best.pt').write_bytes(b'omit');atomic_json([{'epoch':1}],root/'run/deep_s42/history.json')
            env=os.environ|dict(BABEL_GROWTH_RUN=str(root/'run'),BABEL_DOWNLOAD_DIR=str(root/'download'),BABEL_GROWTH_LOG=str(root/'none'),PYTHON_BIN='/usr/bin/false')
            res=subprocess.run(['bash','scripts/babel_growth768_autodl.sh','all'],env=env,capture_output=True,text=True);self.assertEqual(res.returncode,1)
            with tarfile.open(root/'download/run_reports.tar.gz') as t:
                self.assertIn('run/deep_s42/history.json',t.getnames());self.assertNotIn('run/deep_s42/best.pt',t.getnames());self.assertIn('run_status=failed',t.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
