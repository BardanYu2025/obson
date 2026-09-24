"""Synthetic masked readouts and immutable pipeline; no neural optimizer updates."""
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

from obson.babel import utility_probe as up, utility_probe_run as run
from obson.babel.ae_extend import atomic_json, atomic_save
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features
from test_bar_alignment import small_model
from test_window_state_probe import inventories
import test_path_feature as path_fixture


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_current_targets_keep_masks_and_have_no_future(self):
        raw=features(8).astype(float);raw[0,-1,27]=0;raw[1,-1,23:26]=0
        raw[2,-1,22]=0;raw[2,-1,20]=.2
        y,m=up.targets(raw)
        np.testing.assert_allclose(y[:,:6],up.probe.descriptors(raw))
        np.testing.assert_allclose(y[:,6:8],raw[:,-1,:2])
        self.assertFalse(m[0,9]);self.assertFalse(m[1,10:13].any());self.assertFalse(m[2,10:12].any())
        altered=raw.copy();altered[:,-1,0]+=.1
        self.assertFalse(np.array_equal(y,up.targets(altered)[0]))
        bank=np.concatenate((raw,features(8)),axis=1);before=up.targets(bank[:,:128]);bank[:,128:]=9
        for a,b in zip(before,up.targets(bank[:,:128])):np.testing.assert_array_equal(a,b)

    def test_float32_zero_roundtrip_preserves_suspect_oi_mask(self):
        raw=features(8).astype(float);raw[0,-1,22]=0;raw[0,-1,20]=.2;raw[1,-1,20]=0;raw[1,-1,22]=0
        mean=np.full(28,.123456789);scale=np.full(28,.912345678)
        for i in (11,23,24,25,26,27):mean[i]=0;scale[i]=1
        x=((raw-mean)/scale).astype(np.float32)
        self.assertNotEqual(float((x.astype(float)*scale+mean)[0,-1,22]),0.)
        restored=up.restore_raw(x,dict(x_mean=mean.tolist(),x_scale=scale.tolist()))
        self.assertEqual(restored[0,-1,22],0.);self.assertEqual(restored[1,-1,20],0.)
        np.testing.assert_array_equal(up.targets(restored)[1],up.targets(raw)[1])

    def test_masked_fit_matches_normal_equations_and_ignores_missing_values(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(40,8));v=rng.normal(size=(18,8))
        y=rng.normal(size=(40,13));vy=rng.normal(size=(18,13));mask=np.ones_like(y,bool);vm=np.ones_like(vy,bool)
        mask[:11,8:]=False;vm[:4,8:]=False;stats=up.target_scales(y,mask)
        a,w,b=up.fit_heads(x,y,mask,v,vy,vm,stats,'cpu')
        changed=y.copy();changed[~mask]=1e12;cv=vy.copy();cv[~vm]=-1e12
        a2,w2,b2=up.fit_heads(x,changed,mask,v,cv,vm,stats,'cpu')
        self.assertEqual(a,a2);np.testing.assert_array_equal(w,w2);np.testing.assert_array_equal(b,b2)
        for i in (0,8):
            ids=mask[:,i];xn=up.probe.normalize(x[ids],a[i]['statistics']);target=(y[ids,i]-stats['mean'][i])/stats['scale'][i]
            expect=np.linalg.solve(xn.T@xn/len(xn)+a[i]['alpha']*np.eye(8),xn.T@(target-target.mean())/len(xn))
            np.testing.assert_allclose(w[:,i],expect,atol=1e-12)
            self.assertEqual(a[i]['alpha'],min(a[i]['candidates'],key=lambda z:z['validation_nmse'])['alpha'])
        pred=up.predict(a,w,b,v,stats);s,e=up.measure(pred,vy,vm,stats)
        self.assertEqual(s['targets'][8]['support'],14)
        np.testing.assert_array_equal(e[up.NAMES[8]][~vm[:,8]],0)
        changed_pred=pred.copy();changed_pred[:,6:]+=1e4
        self.assertEqual(s['groups']['utility'],up.measure(changed_pred,vy,vm,stats)[0]['groups']['utility'])

    def test_gate_rejects_copy_only_bad_reference_or_missing_targets(self):
        rows=[dict(symbol='A',period=15,month='2020-01',week=f'w{i//10}') for i in range(60)]
        meta=run.make_manifest(Path('/source'),dict(manifest=dict(experiments=[dict(name=f'{mode}_s{s}') for mode in ('control','path') for s in (42,43)],parents={'42':dict(width=768)})))
        names=meta['variants'];scores={};errors={};mask=np.ones((60,13),bool)
        for name in names:
            scores[name]=dict(targets=[dict(r2=.9) for _ in up.NAMES])
            errors[name]={k:np.full(60,.4 if name in meta['models'] else 1.) for k in up.GROUPS}
            errors[name].update({k:np.full(60,.05) for k in up.NAMES})
        self.assertTrue(run.decide(meta,scores,errors,mask,rows)['control_s42']['utility_passed'])
        self.assertTrue(run.decide(meta,scores,errors,mask,rows)['control_s42']['current_passed'])
        errors['control_s42']['utility'][:]=1.2
        self.assertFalse(run.decide(meta,scores,errors,mask,rows)['control_s42']['utility_passed'])
        self.assertTrue(run.decide(meta,scores,errors,mask,rows)['control_s42']['current_passed'])
        scores['raw3584']['targets'][0]['r2']=.1
        self.assertFalse(run.decide(meta,scores,errors,mask,rows)['path_s43']['utility_passed'])
        mask[:,12]=False;result=run.decide(meta,scores,errors,mask,rows)
        self.assertFalse(result['path_s43']['current_passed']);self.assertIsNone(result['path_s43']['current_checks'][-1]['interval']['delta'])


class PipelineTests(unittest.TestCase):
    def test_selected_checkpoint_adapter_freezes_actual_path_module_and_rejects_reselection(self):
        with tempfile.TemporaryDirectory() as td:
            model,data,stats,local,sm,source=path_fixture.ContinuationTests().fixture(Path(td));job=sm['experiments'][1]
            path=source/job['name'];path.mkdir();meta=dict(source=str(source),identity=dict(manifest=sm))
            with patch.object(run.bb,'model_for',side_effect=lambda *args:copy.deepcopy(model)):
                candidate=run.pf.construct(sm,job,'cpu').requires_grad_(False).eval()
                atomic_save(dict(metadata=dict(manifest=sm,job=job),epoch=2,validation={'selection':1.},model=candidate.state_dict()),path/'best.pt')
                atomic_json(dict(trials={job['name']:dict(selected_epoch=2,validation={'selection':1.})}),source/'selection_lock.json')
                frozen,actual_job,expected=run.load_model(meta,job['name'],'cpu')
                self.assertFalse(any(p.requires_grad for p in frozen.parameters()));self.assertFalse(frozen.training)
                self.assertEqual(actual_job,job);self.assertEqual(expected,{'selection':1.})
                self.assertEqual(float(frozen.core.encoder.backbone.input.enabled),1.)
                torch.testing.assert_close(frozen.core(torch.tensor(data['x'])),candidate.eval().core(torch.tensor(data['x'])),atol=0,rtol=0)
                atomic_json(dict(trials={job['name']:dict(selected_epoch=1,validation={'selection':1.})}),source/'selection_lock.json')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):run.load_model(meta,job['name'],'cpu')

    def test_source_adapter_checks_all_workers_and_stays_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model,data,stats,local,meta,source=path_fixture.ContinuationTests().fixture(root)
            sampling=root/'sampling';sampling.mkdir();coverage=root/'coverage';coverage.mkdir()
            meta['identity']['manifest'].update(sampling_source=str(sampling),original_source=str(root/'original'))
            atomic_json(dict(coverage_source=str(coverage),coverage_identity={'test':True}),sampling/'manifest.json')
            atomic_json(meta,source/'manifest.json');trials={};weights={};allfiles={'manifest.json':sha256(source/'manifest.json')}
            for job in meta['experiments']:
                path=source/job['name'];path.mkdir();history=[dict(epoch=e,validation={'selection':1/e},windows=4,encoder_steps=2,head_steps=2) for e in (1,2)]
                summary=dict(selected_epoch=2,validation=history[-1]['validation'],windows=8,encoder_steps=4,head_steps=4)
                atomic_json(history,path/'history.json');atomic_json(dict(validation={'selection':2.}),path/'initial_validation.json');atomic_json(summary,path/'training_summary.json')
                for kind in ('best','last'):(path/f'{kind}.pt').write_bytes(b'fingerprint-only fixture')
                files={p.name:sha256(p) for p in path.iterdir() if p.is_file()}
                atomic_json(dict(status='complete',metadata=dict(manifest=meta,job=job),files=files),path/'completion.json')
                trials[job['name']]=summary;weights[job['name']]={k:sha256(path/f'{k}.pt') for k in ('best','last')}
                allfiles.update({job['name']+'/'+n:v for n,v in files.items()})
            atomic_json(dict(manifest=meta,trials=trials,weights=weights),source/'selection_lock.json');allfiles['selection_lock.json']=sha256(source/'selection_lock.json')
            atomic_json(dict(status='complete',source_unchanged=True,cells=4,encoder_steps=16,files=allfiles),source/'completion.json')
            before={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
            # External ancestral audits are substituted; this source's own files/locks are real.
            with patch.object(run.pf.ea,'source_identity',return_value=meta['identity']),patch.object(run.bb.ws,'coverage_identity',return_value={'test':True}):
                identity=run.source_identity(source);self.assertEqual(identity['manifest'],meta)
                self.assertEqual(before,{str(p):sha256(p) for p in source.rglob('*') if p.is_file()})
                (source/meta['experiments'][0]['name']/'best.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError):run.source_identity(source)

    def fixture(self,root):
        model,_,stats,local=small_model();source=root/'source';alignment=root/'alignment';(alignment/'cache').mkdir(parents=True);source.mkdir()
        atomic_json(stats,alignment/'cache/statistics.json');atomic_json(local,alignment/'cache/local_scales.json')
        atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__),source/'runtime.json')
        jobs=[dict(name=f'{mode}_s{s}',seed=s,enabled=mode=='path') for mode in ('control','path') for s in (42,43)]
        sm=dict(source=str(alignment),experiments=jobs,parents={'42':dict(width=768),'43':dict(width=768)},identity=dict(manifest={}))
        identity=dict(manifest=sm,coverage=str(root/'coverage'),files={})
        banks={};sizes=dict(train=20,val=10,test=10,cross_research=10)
        for i,(split,n) in enumerate(sizes.items()):
            x=features(n,10+i);x[...,22]=np.linspace(.1,.5,n)[:,None]
            x[::3,-1,27]=0
            y,mask=run.bb.ar.ordered_targets(x);xx,yy,mm=run.bb.ar.normalize(x,y,mask,stats);banks[split]=dict(x=xx,y=yy,mask=mm)
        rows=inventories(sizes);out=root/'run'
        return model,source,identity,banks,rows,out

    def test_frozen_extraction_all_heads_lock_resume_and_fingerprint_failures(self):
        with tempfile.TemporaryDirectory() as td:
            model,source,identity,banks,rows,out=self.fixture(Path(td));make=run.make_manifest;load_events=[]
            def manifest(*a,**kw):
                m=make(*a,**kw);m['pca_rank']=4;m['variants']=[n.replace('pca768','pca4') for n in m['variants']];return m
            def loader(sm,split):
                if split in run.SPLITS[2:]:self.assertTrue((out/'selection_lock.json').exists())
                load_events.append(split);return banks[split]
            def models(meta,name,device):
                job=next(j for j in meta['identity']['manifest']['experiments'] if j['name']==name)
                frozen=copy.deepcopy(model).requires_grad_(False).eval();return frozen,job,{}
            before={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
            with ExitStack() as stack:
                stack.enter_context(patch.object(run,'source_identity',return_value=identity))
                stack.enter_context(patch.object(run,'make_manifest',side_effect=manifest))
                stack.enter_context(patch.object(run,'check_output'))
                stack.enter_context(patch.object(run,'inventories',return_value=rows))
                stack.enter_context(patch.object(run,'load_model',side_effect=models))
                stack.enter_context(patch.object(run.bb,'load_data',side_effect=loader))
                stack.enter_context(patch.object(run.pf,'validation',return_value={}))
                stack.enter_context(patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No neural optimizer')))
                run.run(source,out,batch=4,device='cpu')
                self.assertEqual(load_events,['val','train','val','test','cross_research'])
                fit=run.read_json(out/'fit.json');self.assertEqual(len(fit['heads']),7)
                self.assertTrue(all(len(row['targets'])==13 for row in fit['heads'].values()))
                self.assertEqual(run.read_json(out/'runtime.json')['ridge_candidates'],455)
                self.assertEqual(run.read_json(out/'completion.json')['encoder_updates'],0)
                report=run.read_json(out/'utility_metrics.json')['datasets']
                for split in run.SPLITS[2:]:
                    data=run.read_json(out/f'{split}_predictions.json')
                    for name,pred in data['predictions'].items():
                        scored,errors=up.measure(pred,data['targets'],data['mask'],fit['target_stats'])
                        self.assertEqual(scored,report[split]['scores'][name])
                        for key in errors:np.testing.assert_array_equal(errors[key],data['per_window_errors'][name][key])
                # Completed resume performs no neural inference or refitting.
                with patch.object(run,'prepare',side_effect=AssertionError('No extraction')),patch.object(run,'fit',side_effect=AssertionError('No refit')):
                    run.run(source,out,batch=4,device='cpu')
                # Partial scoring resumes from the locked readouts and cached arrays.
                (out/'completion.json').unlink();lock_before=sha256(out/'selection_lock.json')
                with patch.object(up,'fit_heads',side_effect=AssertionError('Locked fits unchanged')):run.run(source,out,batch=4,device='cpu')
                self.assertEqual(lock_before,sha256(out/'selection_lock.json'))
                with self.assertRaisesRegex(ValueError,'configuration'):run.run(source,out,batch=8,device='cpu')
                (out/'heads.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError):run.run(source,out,batch=4,device='cpu')
            self.assertEqual(before,{str(p):sha256(p) for p in source.rglob('*') if p.is_file()})

    def test_research_without_lock_and_inventory_anchor_extra(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(FileNotFoundError):run.prepare({},root,'test',[],'cpu')
            source=root/'alignment';coverage=root/'coverage';source.mkdir();coverage.mkdir();rows=inventories(dict(train=3,val=2,test=2,cross_research=2))
            # Actual alignment inventories additionally contain an anchor field.
            for split,v in rows.items():
                normalized=[dict(x,week=str(run.pd.Timestamp(x['end']).to_period('W'))) for x in v]
                atomic_json(normalized,coverage/f'{split}_windows.json')
                if split in run.SPLITS[2:]:atomic_json([dict(x,anchor=100.) for x in normalized],source/f'{split}_inventory.json')
            result=run.inventories(dict(identity=dict(coverage=str(coverage),manifest=dict(source=str(source)))))
            self.assertEqual(len(result['test']),2)

    def test_failure_and_repeat_export(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';out.mkdir();atomic_json(dict(status='failed'),out/'failure.json')
            for suffix in ('pt','npy','npz'):(out/f'weights.{suffix}').write_bytes(b'private weights')
            fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_UTILITY_RUN=str(out),BABEL_UTILITY_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN=str(fake))
            script=Path(__file__).resolve().parents[2]/'scripts/babel_utility768_autodl.sh'
            for mode,code in [('all',7),('export',0)]:
                p=subprocess.run(['bash',str(script),mode],env=env,capture_output=True,text=True);self.assertEqual(p.returncode,code,p.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
