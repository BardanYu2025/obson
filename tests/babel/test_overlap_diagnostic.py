"""Overlap boundaries, independent target geometry, source symlinks and frozen workflow."""
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

from obson.babel import overlap_diagnostic as od, overlap_diagnostic_run as run
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features
from test_bar_alignment import small_model


def fixture_data(stats):
    raw=features(6).reshape(-1,28);raw[:,0]=np.arcsinh(np.linspace(-.2,.3,len(raw)));raw[:,1]=np.arcsinh(.05)
    ends=[639,667,699];rows=[dict(key='A/15/C',symbol='A',period=15,row=e,end=f'2026-01-{i+10:02d} 10:00:00',month='2026-01',week=f'w{i}') for i,e in enumerate(ends)]
    specs=[dict(series=0,lo=0,offset=0,length=len(raw),endpoints=[[e,i] for i,e in enumerate(ends)])]
    plan,_=od.pair_plan(specs,rows,len(raw));views={}
    for d in (0,)+od.SHIFTS:
        x,y,m=run.normalized(od.windows(raw,plan,d),stats);views[d]=dict(x=x,y=y,mask=m)
    return raw,specs,rows,dict(rows=plan,views=views)


class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_common_target_alignment_masks_and_predicted_anchor_only(self):
        _,_,stats,_=small_model();_,_,_,data=fixture_data(stats);z=np.ones((3,4))
        for d in od.SHIFTS:
            va,vb=data['views'][d],data['views'][0];a,b=run.physical(va,stats),run.physical(vb,stats)
            score,valid=od.pair_scores(a,b,a,b,va['mask'],vb['mask'],z,z,d,stats)
            self.assertEqual(len(a[:,d:127]),3);self.assertEqual(a[:,d:127].shape[1],127-d)
            for k in ('path_shape_gap_bps','change1_gap_nmse','body_gap_nmse','activity_gap_nmse'):self.assertLess(max(score[k]),1e-3)
            aa=a.copy();bb=b.copy();aa[:,:,0]+=2;bb[:,:,0]-=3
            changed,_=od.pair_scores(aa,bb,a,b,va['mask'],vb['mask'],z,z,d,stats)
            np.testing.assert_allclose(changed['path_shape_gap_bps'],score['path_shape_gap_bps'],atol=1e-10)
            np.testing.assert_allclose(changed['path_native_mae_a_bps'],200,atol=1e-10)
            np.testing.assert_allclose(changed['path_native_mae_b_bps'],300,atol=1e-10)
            # All unshared/new/current bars must be excluded, including A's current bar.
            aa=a.copy();bb=b.copy();aa[:,:d]=9e6;aa[:,127]=9e6;bb[:,127-d:]=9e6
            again,_=od.pair_scores(aa,bb,a,b,va['mask'],vb['mask'],z,z,d,stats)
            for k in score:np.testing.assert_array_equal(score[k],again[k])
            aa=a.copy();bb=b.copy();aa[:,:,0]+=np.arange(128)*.1;bb[:,:,0]+=np.arange(128)*.1
            biased,_=od.pair_scores(aa,bb,a,b,va['mask'],vb['mask'],z,z,d,stats)
            self.assertLess(max(biased['path_shape_gap_bps']),1e-3)
            self.assertGreater(min(biased['path_shape_error_a_bps']),10)  # agreement alone is not success
        ma=va['mask'].copy();mb=vb['mask'].copy();ma[:,:,2:]=False;mb[:,:,2:]=False
        result,valid=od.pair_scores(a,b,a,b,ma,mb,z,z,d,stats)
        self.assertFalse(valid['activity_gap_nmse'].any());self.assertIsNone(od.summarize(result['activity_gap_nmse'],valid['activity_gap_nmse'])['mean'])
        mb[:,0,3]=True
        with self.assertRaisesRegex(ValueError,'masks'):od.pair_scores(a,b,a,b,ma,mb,z,z,d,stats)

    def test_exact_linear_reference_and_state_metrics(self):
        n=2;change=.2;stats=dict(delta_scale=[.5],y_scale=[1.]*7)
        a=np.zeros((n,128,7));b=a.copy();a[:,:,0]=np.arange(128)*change;b[:,:,0]=(np.arange(128)+16)*change
        mask=np.ones_like(a,bool);mask[:,-1]=False;z=np.array([[1.,0],[0,0]])
        v,valid=od.pair_scores(a,b,a,b,mask,mask,z,z,16,stats)
        self.assertLess(v['path_shape_gap_bps'].max(),1e-10);self.assertEqual(valid['state_cosine'].tolist(),[True,False])
        wrong=b.copy();wrong[:,1,1]=1
        with self.assertRaisesRegex(ValueError,'targets differ'):od.pair_scores(a,b,a,wrong,mask,mask,z,z,16,stats)

    def test_pair_plan_never_crosses_partition_contract_or_warmup(self):
        rows=[dict(key='A',row=511),dict(key='A',row=639),dict(key='B',row=1300)]
        specs=[dict(series=0,lo=0,offset=0,length=700,endpoints=[[511,0],[639,1]]),dict(series=1,lo=1173,offset=700,length=128,endpoints=[[127,2]])]
        plan,excluded=od.pair_plan(specs,rows,828)
        self.assertEqual([x['index'] for x in plan],[1]);self.assertEqual([x['index'] for x in excluded],[0,2])
        bad=copy.deepcopy(rows);bad[1]['row']+=1
        with self.assertRaises(ValueError):od.pair_plan(specs,bad,828)
        bad=copy.deepcopy(rows);bad[1]['key']='B'
        with self.assertRaises(ValueError):od.pair_plan(specs,bad,828)
        bad=copy.deepcopy(specs);bad[1]['offset']-=1
        with self.assertRaises(ValueError):od.pair_plan(bad,rows,828)

    def test_source_bank_identity_resolves_actual_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);original=root/'original';original.mkdir();bank=root/'bank';bank.mkdir();real=root/'real';real.mkdir();cross=root/'cross';(cross/'cache').mkdir(parents=True)
            (real/'test_x.npy').write_bytes(b'fixture');(bank/'test_x.npy').symlink_to(real/'test_x.npy');atomic_json([],bank/'test_sequences.json')
            for name in ('test_x.npy','test_sequences.json'):(cross/'cache'/name).write_bytes(b'cross fixture')
            atomic_json(dict(files={n:sha256(cross/'cache'/n) for n in ('test_x.npy','test_sequences.json')}),cross/'cache/index.json')
            pinned={str(p.resolve()):sha256(p) for p in (bank/'test_x.npy',bank/'test_sequences.json',cross/'cache/index.json')}
            atomic_json(dict(bank=str(bank),cross_run=str(cross),source_identity=pinned),original/'manifest.json')
            identity=dict(manifest=dict(identity=dict(manifest=dict(original_source=str(original)))))
            result=run.banks(identity);self.assertEqual(result['test']['directory'],str(bank))
            (real/'test_x.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):run.banks(identity)

    def test_complete_pipeline_resume_and_mutation_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model,_,stats,_=small_model();model.eval().requires_grad_(False);source=root/'source';source.mkdir();alignment=root/'alignment';(alignment/'cache').mkdir(parents=True)
            atomic_json(stats,alignment/'cache/statistics.json');atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__),source/'runtime.json')
            jobs=[dict(name=f'{mode}_s{s}',seed=s,enabled=mode=='path') for s in (42,43) for mode in ('control','path')]
            sm=dict(source=str(alignment),experiments=jobs,parents={'42':dict(width=768),'43':dict(width=768)},identity=dict(manifest={}))
            identity=dict(manifest=sm,coverage='fixture',files={});packed={};inventories={};cached={}
            for split in run.SPLITS:
                raw,specs,rows,data=fixture_data(stats);directory=root/split;directory.mkdir();np.save(directory/'test_x.npy',raw);atomic_json(specs,directory/'test_sequences.json')
                packed[split]=dict(directory=str(directory),files={});inventories[split]=rows;cached[split]=data['views'][0]
            out=root/'out';original={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
            def preflight(meta,destination,device):
                result=run.ur.pf.er.trained_causality(model,torch.tensor(cached['test']['x'][:2]));self.assertNotEqual(result['status'],'failed');atomic_json(result,destination/'trained_validation_causality.json')
            with ExitStack() as stack:
                for obj,name,value in [(run.ur,'check_output',lambda *a:None),(run.ur,'source_identity',lambda *a:identity),(run,'banks',lambda *a:packed),(run.ur,'inventories',lambda *a:inventories),(run.ur.up.probe,'inventory_audit',lambda *a:{}),(run.ur.bb,'load_data',lambda m,s:cached[s]),(run.ur,'preflight',preflight),(run.ur,'load_model',lambda *a:(copy.deepcopy(model),{},{}))]:stack.enter_context(patch.object(obj,name,side_effect=value))
                stack.enter_context(patch.object(torch.optim,'AdamW',side_effect=AssertionError('No optimizer allowed')))
                run.run(source,out,2,'cpu');done=run.read_json(out/'completion.json');self.assertEqual(done['status'],'complete');self.assertEqual(done['fits'],0)
                for split in run.SPLITS:
                    for name in run.MODELS:
                        c=run.read_json(out/f'{split}_{name}.json');self.assertEqual(set(c['shifts']),{'1','16','64'});self.assertTrue(c['numeric_control']['passed'])
                        for shift,v in c['shifts'].items():
                            for example in v['examples']:
                                i=example['index'];a=np.array(example['prediction_a'])[int(shift):127,0];truth=np.array(example['target_a'])[int(shift):127,0]
                                error=np.sqrt(np.mean(((a-a[0])-(truth-truth[0]))**2))*100
                                self.assertAlmostEqual(error,v['per_pair']['path_shape_error_a_bps'][i],places=9)
                            self.assertAlmostEqual(np.mean(v['per_pair']['body_gap_nmse']),v['summary']['all']['body_gap_nmse']['mean'])
                            self.assertEqual(v['shared_target_support'][0][0],127-int(shift))
                with patch.object(run,'predict',side_effect=AssertionError('Completed run cannot infer')):run.run(source,out,2,'cpu')
                (out/'completion.json').unlink()
                with patch.object(run,'predict',side_effect=AssertionError('Partial resume must reuse complete cells')):run.run(source,out,2,'cpu')
                with self.assertRaisesRegex(ValueError,'config changed'):run.run(source,out,3,'cpu')
                p=out/'test_control_s42.json';p.write_text(p.read_text()+' ')
                with self.assertRaises(ValueError):run.run(source,out,2,'cpu')
            self.assertEqual(original,{str(p):sha256(p) for p in source.rglob('*') if p.is_file()})

    def test_failure_export_keeps_status_and_excludes_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';out.mkdir();atomic_json(dict(status='failed'),out/'failure.json');(out/'weights.pt').write_bytes(b'omit');(out/'cache.npy').write_bytes(b'omit')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_OVERLAP_RUN=str(out),BABEL_OVERLAP_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN=str(fake))
            script=Path(__file__).resolve().parents[2]/'scripts/babel_overlap768_autodl.sh'
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(['bash',str(script),mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    names=t.getnames();self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in names));self.assertIn('run_status=failed',t.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
