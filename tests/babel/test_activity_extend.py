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

from test_activity_alignment import fixture as alignment_fixture
from obson.babel import activity_extend as ex
from obson.babel import activity_alignment as al
from obson.babel import activity_ablation as aa
from obson.babel import detail_alignment as da
from obson.babel.ae_extend import atomic_save,rng_state
from obson.babel.dual_state import sha256


def assert_tree(test,a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a:assert_tree(test,a[k],b[k])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_tree(test,x,y)
    else:test.assertEqual(a,b)


def fixture(root):
    _,old,parent,_,scales,_=alignment_fixture(root)
    old.update(schema=al.SCHEMA,epochs=30,upstream={'test':'identity'},fusion_run='synthetic')
    (parent/'manifest.json').write_text(json.dumps(old));index=json.loads((parent/'cache/index.json').read_text());index['manifest']=old;(parent/'cache/index.json').write_text(json.dumps(index))
    (parent/'completion.json').write_text('{"status":"complete"}');(parent/'alignment_metrics.json').write_text('{}')
    for job in old['experiments']:
        torch.manual_seed(123);model=al.initial_model(old,'cpu',job);model.configure(True);opt=ex.optimizer_for(model,old)
        # Explicit synthetic accumulators exercise restore without any optimizer step.
        for group in opt.param_groups:
            for param in group['params']:
                opt.state[param]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(param,.003),exp_avg_sq=torch.full_like(param,.01))
        score=da.run_epoch(model,al.arrays_for(old,parent,'val','cpu'),aa.ActivityStreams(parent/'cache','val',job),True,True,scales,2,latent_objective_fn=al.latent_objective(model,job['aux_weight']))[0]
        state=dict(metadata=dict(manifest=old,job=job),epoch=30,best_epoch=22,
            history=[dict(epoch=i,train=score,validation=score) for i in range(1,31)],best_validation=score,
            best_model=copy.deepcopy(model.state_dict()),model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        folder=parent/job['name'];folder.mkdir();da.publish(state,folder);(folder/'per_window_metrics.json').write_text('[]')
    with patch.object(al,'source_identity',return_value=({},old['upstream'])):read,identity=ex.parent_identity(parent)
    out=root/'extended';out.mkdir();meta=ex.make_manifest(parent,read,identity,100,10)
    (out/'manifest.json').write_text(json.dumps(meta));ex.prepare(meta,out)
    return old,parent,meta,out


class ActivityExtendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_import_preserves_optimizer_rng_history_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,parent,meta,out=fixture(Path(tmp));job=meta['experiments'][0]
            src=torch.load(parent/job['name']/'last.pt',weights_only=True);state=ex.import_state(meta,job)
            for k in ('epoch','model','optimizer','rng','history','best_epoch','best_model','best_validation'):assert_tree(self,src[k],state[k])
            self.assertEqual(state['metadata']['manifest']['epochs'],100);self.assertFalse(state['selectors_initialized'])
            self.assertEqual(sha256(parent/job['name']/'last.pt'),meta['parent_identity'][f'{job["name"]}/last.pt'])
            ex.prepare(meta,out);self.assertTrue((out/'cache/train_x.npy').is_symlink())
            (out/'cache/train_activity.npy').write_bytes(b'bad')
            with self.assertRaises(ValueError):ex.prepare(meta,out)

    def test_parent_hash_and_incomplete_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,parent,meta,out=fixture(Path(tmp));job=meta['experiments'][0]
            path=parent/job['name']/'last.pt';state=torch.load(path,weights_only=True);state['epoch']=29;atomic_save(state,path)
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):ex.import_state(meta,job)
            meta['parent_identity'][f'{job["name"]}/last.pt']=sha256(path)
            with self.assertRaisesRegex(ValueError,'declared budget'):ex.import_state(meta,job)
            with self.assertRaises(ValueError):ex.make_manifest(parent,old,{},30,10)

    def test_selector_past_only_and_price_retention(self):
        rng=np.random.default_rng(8);tr=rng.normal(size=(60,5));va=rng.normal(size=(20,5));w=rng.normal(size=(5,20))
        ty=tr@w;vy=va@w;tm=np.ones_like(ty,bool);vm=np.ones_like(vy,bool)
        score=ex.ridge_selection(tr,va,ty,vy,tm,vm)
        changed=vy.copy();changed[:,:5]=1e10
        self.assertEqual(score,ex.ridge_selection(tr,va,ty,changed,tm,vm));self.assertLess(score['validation_past_mse'],.01)
        model=torch.nn.Linear(2,2);reference=dict(base=1.,close_bps=10.,change16_mse=2.,detail=.1)
        state=dict(activity_best=None,selection_history=[])
        ex.consider_activity(state,model,40,reference,score,reference,'test')
        bad=dict(reference,close_bps=11.);better=dict(score,validation_past_mse=0.)
        ex.consider_activity(state,model,50,bad,better,reference,'test')
        self.assertEqual(state['activity_best']['epoch'],40)
        self.assertFalse(ex.activity_eligible(dict(reference,detail=.103),reference))
        tm[:,5]=False
        with self.assertRaisesRegex(ValueError,'support'):ex.ridge_selection(tr,va,ty,vy,tm,vm)

    def test_worker_restore_diagnostics_and_rng_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,parent,meta,out=fixture(Path(tmp));meta['epochs']=30;(out/'manifest.json').write_text(json.dumps(meta))
            job=meta['experiments'][0];source=torch.load(parent/job['name']/'last.pt',weights_only=True)
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):ex.worker(out,job['name'],'cpu')
            state=torch.load(out/job['name']/'last.pt',weights_only=True)
            for k in ('model','optimizer','rng','history','best_model'):assert_tree(self,state[k],source[k])
            self.assertTrue(state['selectors_initialized']);self.assertEqual(len(state['selection_history']),2)
            self.assertTrue((out/job['name']/'resume_validation.json').exists())
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):ex.worker(out,job['name'],'cpu')
            again=torch.load(out/job['name']/'last.pt',weights_only=True);assert_tree(self,again,state)

    def test_two_reports_fallback_and_budget_summary_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,parent,meta,out=fixture(Path(tmp));meta['epochs']=30;(out/'manifest.json').write_text(json.dumps(meta))
            for job in meta['experiments']:
                with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):ex.worker(out,job['name'],'cpu')
            view=ex.build_activity_view(out)
            with patch.object(al,'nonlinear_probe',side_effect=lambda z,y,m,*args:al.ridge_probe(z,y,m)),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                al.evaluate(out,'cpu');al.evaluate(view,'cpu')
            # Explicitly failed secondary selection must never be reported as qualified.
            path=view/'aux005_s42/best.pt';ck=torch.load(path,weights_only=True);ck['selection_qualified']=False;ck['selection_role']='price_fallback_no_qualified_history_candidate';atomic_save(ck,path)
            r=ex.annotate_activity_view(out);self.assertFalse(r['stage_decisions']['aux005']['evidence_pass'])
            self.assertFalse(r['variants']['aux005_s42']['selection_qualified'])
            (parent/'alignment_metrics.json').write_text((out/'alignment_metrics.json').read_text())
            ex.budget_report(out);self.assertTrue((out/'budget_summary.md').exists())
            budget=json.loads((out/'budget_comparison.json').read_text());self.assertEqual(budget['variants']['control_s42']['validation_trends']['detail']['best_epoch'],1)

    def test_continuation_uses_absolute_epochs_without_local_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,parent,meta,out=fixture(Path(tmp));meta['epochs']=33;(out/'manifest.json').write_text(json.dumps(meta))
            seeds=[];real=da.run_epoch
            def no_update(*args,**kwargs):
                args=list(args)
                if args[7] is not None:
                    seeds.append(args[8]);self.assertEqual([g['lr'] for g in args[7].param_groups],[3e-5 if i==1 else 1e-4 for i in range(3)])
                    args[7]=None
                return real(*args,**kwargs)
            with patch.object(da,'run_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                ex.worker(out,'control_s42','cpu')
            self.assertEqual(seeds,[73,74,75])
            state=torch.load(out/'control_s42/last.pt',weights_only=True);self.assertEqual([r['epoch'] for r in state['history']],list(range(1,34)))
            for value in state['optimizer']['state'].values():self.assertEqual(value['step'].item(),7.)
            self.assertEqual(state['selection_history'][-1]['epoch'],33)
            partial=dict(meta,epochs=34)
            with self.assertRaisesRegex(ValueError,'incomplete'):ex.verify_finished(out,partial)

    def test_export_nested_reports_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';(run/'activity_selection').mkdir(parents=True);(run/'activity_selection/metrics.json').write_text('{}');(run/'last.pt').write_bytes(b'x')
            env=dict(os.environ,BABEL_EXTEND_RUN=str(run),BABEL_EXTEND_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                p=subprocess.run(['bash','scripts/babel_activity_extend_autodl.sh',mode],env=env,capture_output=True,text=True);self.assertEqual(p.returncode,code,p.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertIn('run/activity_selection/metrics.json',t.getnames());self.assertFalse(any(n.endswith('.pt') for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())
