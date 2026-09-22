"""Schedule, immutable controls, selection, resume and raw-target audits without neural updates."""
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
import pandas as pd
import torch

from obson.babel import architecture as ar, architecture_benchmark as ab
from obson.babel import pca_teacher as pt, pca_teacher_benchmark as tb, pca_anneal_benchmark as an, path_audit
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
import test_capacity_benchmark as capacity_tests
from test_pca_teacher import SMALL


def read(p):return json.loads(p.read_text())


def no_update(model,data,stats,batch,micro,device,weight,opt=None,order_seed=0):
    rows,_=tb.run_epoch(model,data,stats,batch,micro,device,weight,None,order_seed)
    return rows,(len(data['x'])+batch-1)//batch if opt is not None else 0


class AnnealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capacity_tests.CapacityPipelineTests.setUpClass()
        cls.temp=tempfile.TemporaryDirectory();cls.teacher=Path(cls.temp.name)/'teacher';cls.teacher.mkdir()
        cls.source=capacity_tests.CapacityPipelineTests.source
        with patch.object(pt,'CONFIG',SMALL):
            meta=tb.make_manifest(cls.source,capacity_tests.CapacityPipelineTests.identity,3,2,4,2)
        meta['experiments']=[j for j in meta['experiments'] if j['phase']=='base']
        atomic_json(meta,cls.teacher/'manifest.json');tb.prepare(meta,cls.teacher)
        original=tb.run_epoch
        def fake(model,data,stats,batch,micro,device,weight,opt=None,order_seed=0):
            row,_=original(model,data,stats,batch,micro,device,weight,None,order_seed)
            return row,(len(data['x'])+batch-1)//batch if opt is not None else 0
        with patch.object(tb,'run_epoch',side_effect=fake),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
            for j in meta['experiments']:tb.worker(cls.teacher,j['name'],'cpu')
        tb.lock_selection(meta,cls.teacher,True);tb.lock_selection(meta,cls.teacher)
        stats=read(cls.source/'cache/statistics.json');report=dict(datasets={})
        for split in ('test','cross_research'):
            data=ab.load_arrays(cls.source,split);scores={};errors={}
            for j in meta['experiments']:
                model=tb.model_for(meta,cls.teacher,j,'cpu')
                model.load_state_dict(torch.load(cls.teacher/j['name']/'best.pt',weights_only=True)['model'])
                pred,_=ab.predictions(model,data['x'],2,'cpu');scores[j['name']],errors[j['name']]=ab.measure(pred,data,stats)
            atomic_json({n:{k:v.tolist() for k,v in r.items()} for n,r in errors.items()},cls.teacher/f'{split}_per_window_errors.json')
            report['datasets'][split]=dict(variants=scores,fixed_last=scores)
        atomic_json(report,cls.teacher/'teacher_metrics.json')
        files={p.name:sha256(p) for p in cls.teacher.iterdir() if p.is_file() and p.suffix=='.json'}
        files['cache/index.json']=sha256(cls.teacher/'cache/index.json')
        atomic_json(dict(status='complete',source_unchanged=True,files=files),cls.teacher/'completion.json')
        cls.identity=an.source_identity(cls.teacher)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup();capacity_tests.CapacityPipelineTests.tearDownClass()

    def make_run(self,td):
        out=Path(td)/'anneal';out.mkdir()
        with patch.object(pt,'CONFIG',SMALL):meta=an.make_manifest(self.teacher,self.identity,1,2)
        atomic_json(meta,out/'manifest.json');an.prepare(meta,out);return meta,out

    def test_exact_schedule_and_budget_guards(self):
        j=dict(schedule='early_anneal',teacher_weight=.25,hold_epochs=50,zero_epoch=100,epochs=200)
        self.assertEqual([an.teacher_weight(j,e) for e in (0,1,50,75,99,100,101,200)],[.25,.25,.25,.125,.0050000000000000044,0.,0.,0.])
        with patch.object(pt,'CONFIG',SMALL):
            with self.assertRaises(ValueError):an.make_manifest(self.teacher,self.identity,2,2)
        with self.assertRaises(ValueError):an.teacher_weight(dict(j,zero_epoch=201),1)

    def test_controls_remain_immutable_cache_reuse_and_all_results(self):
        before={str(p):sha256(p) for p in self.teacher.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            loader=ab.load_arrays
            def train_only(root,split):self.assertIn(split,('train','val'));return loader(root,split)
            with patch.object(ab,'load_arrays',side_effect=train_only),patch.object(pt,'fit_coordinate_scales',side_effect=AssertionError('No refit')):
                meta,out=self.make_run(td);an.preflight(meta,out,'cpu')
                with patch.object(an,'run_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
                    for j in meta['experiments']:an.worker(out,j['name'],'cpu')
            def evaluation(root,split):
                if split in ('test','cross_research'):self.assertTrue((out/'selection_lock.json').exists())
                return loader(root,split)
            with patch.object(ab,'load_arrays',side_effect=evaluation):report=an.evaluate(meta,out,'cpu')
            for split,rows in report['datasets'].items():
                self.assertEqual(len(rows['variants']),8);self.assertEqual(len(rows['fixed_last']),6);self.assertEqual(len(rows['paired']),4)
            for j in meta['experiments']:
                summary=read(out/j['name']/'training_summary.json')
                self.assertEqual(summary['selected_epoch'],0)
                self.assertFalse(summary['fully_released_selection'])
                self.assertEqual(summary['selected_teacher_weight'],.25)
                self.assertTrue(read(out/j['name']/'initialization.json')['matched'])
            self.assertIn('anneal_s42',(out/'examples.html').read_text())
            for job in self.identity['controls']:
                self.assertEqual(read(out/'controls'/job['name']/'history.json'),read(self.teacher/job['name']/'history.json'))
            self.assertNotIn('NaN',(out/'anneal_metrics.json').read_text())
            with patch.object(an,'run_epoch',side_effect=AssertionError('No rerun')):an.worker(out,'anneal_s42','cpu')
        self.assertEqual(before,{str(p):sha256(p) for p in self.teacher.rglob('*') if p.is_file()})

    def test_resume_retains_optimizer_and_absolute_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);name='anneal_s42';calls=[]
            def interrupted(*args,**kwargs):
                if len(args)>7 and args[7] is not None:
                    calls.append(args[6])
                    if len(calls)==2:raise RuntimeError('interrupt')
                    opt=args[7];param=opt.param_groups[0]['params'][0]
                    opt.state[param]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(param,.123),exp_avg_sq=torch.full_like(param,.456))
                return no_update(*args,**kwargs)
            with patch.object(an,'run_epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):an.worker(out,name,'cpu')
            first=read(out/name/'history.json')[0]
            load_opt=torch.optim.AdamW.load_state_dict;restored=[];calls=[]
            def load(self,state):restored.append(copy.deepcopy(state));return load_opt(self,state)
            def resumed(*args,**kwargs):
                if len(args)>7 and args[7] is not None:calls.append(args[6])
                return no_update(*args,**kwargs)
            with patch.object(an,'run_epoch',side_effect=resumed),patch.object(torch.optim.AdamW,'load_state_dict',new=load):an.worker(out,name,'cpu')
            self.assertEqual(calls,[0.,0.]);self.assertEqual(len(restored),1)
            moment=next(iter(restored[0]['state'].values()))
            self.assertEqual(float(moment['step']),7.);self.assertTrue(torch.all(moment['exp_avg']==.123))
            self.assertEqual(read(out/name/'history.json')[0],first)
            self.assertEqual(read(out/name/'resume_validation.json')['epoch'],1)
            history=read(out/name/'history.json');self.assertEqual([r['lr'] for r in history],[ar.learning_rate(e,3,3e-4) for e in (1,2,3)])

    def test_corrupted_control_and_coordinate_source_fail(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td)
            (out/'cache/val_q.npy').write_bytes(b'broken')
            with self.assertRaises(ValueError):an.verify_cache(meta,out)
            bad=dict(self.identity,files=dict(self.identity['files'],**{'plain_s42/best.pt':'invalid'}))
            with self.assertRaises(ValueError):an.prepare(dict(meta,teacher_identity=bad),out)
        with tempfile.TemporaryDirectory() as td:
            import shutil
            clone=Path(td)/'teacher';shutil.copytree(self.teacher,clone)
            (clone/'plain_s42/best.pt').write_bytes(b'broken')
            with self.assertRaises(ValueError):an.source_identity(clone)

    def test_history_budget_and_schedule_rejection(self):
        job=self.identity['controls'][0]
        history=read(self.teacher/job['name']/'history.json')
        n=read(self.teacher/'cache/coordinate_scales.json')['count']
        for key,value in [('teacher_weight',.9),('optimizer_steps',999),('windows',999),('lr',.99)]:
            changed=copy.deepcopy(history);changed[1][key]=value
            with self.assertRaises(ValueError):an.verify_history(changed,job,4,n,lambda e:job['teacher_weight'])

    def test_export_on_failure_and_reexport_keep_status(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download'
            atomic_json(dict(status='partial'),run/'metrics.json')
            for suffix in ('pt','npy','npz'):(run/f'data.{suffix}').write_bytes(b'weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGUMENT_LOG"\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,ARGUMENT_LOG=str(root/'arguments'),BABEL_ANNEAL_RUN=str(run),BABEL_ANNEAL_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            repo=Path(__file__).resolve().parents[2];cmd=['bash',str(repo/'scripts/babel_pca_anneal_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                self.assertIn('obson.babel.pca_anneal_benchmark',(root/'arguments').read_text())
                with tarfile.open(download/'run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


class PathAuditTests(unittest.TestCase):
    def test_telescope_and_cached_target_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'raw'/'ag';raw.mkdir(parents=True);source=root/'source';cache=source/'cache';cache.mkdir(parents=True);out=root/'out';out.mkdir()
            c=np.exp(np.linspace(4,4.4,140));o=np.r_[c[0],c[:-1]]
            frame=pd.DataFrame(dict(datetime=pd.date_range('2026-01-01',periods=140,freq='h'),open=o,close=c,high=np.maximum(o,c),low=np.minimum(o,c),volume=np.ones(140)))
            path=raw/'SHFE.ag2606_60m.csv';frame.to_csv(path,index=False);end=str(frame.datetime.iloc[-1])
            report,g,y=path_audit.raw_window(path,end)
            self.assertLess(report['codec_roundtrip_max_abs'],1e-4)
            from test_architecture_benchmark import features
            x=features(1);x[0,:,:2]=np.arcsinh(g);target,mask=ar.ordered_targets(x)
            stats=ar.fit_scales(x,target,mask);xx,yy,mask=ar.normalize(x,target,mask,stats)
            for n,a in [('x',xx),('y',yy),('mask',mask)]:np.save(cache/f'test_{n}.npy',a)
            atomic_json(stats,cache/'statistics.json');atomic_json([dict(key='ag/60/SHFE.ag2606',end=end)],cache/'test_inventory.json')
            with patch.object(path_audit,'CASES',(('test','ag/60/SHFE.ag2606',end),)):
                r=path_audit.audit(source,root/'raw',out);self.assertEqual(r['cases'][0]['status'],'matched')
                yy[0,10,0]+=1;np.save(cache/'test_y.npy',yy)
                with self.assertRaisesRegex(ValueError,'Cached target'):path_audit.audit(source,root/'raw',out)


if __name__=='__main__':unittest.main()
