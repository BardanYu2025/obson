"""Sampling, provenance, recovery and evaluation gates; no neural optimizer updates."""
import copy
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from obson.babel import architecture as ar, architecture_benchmark as ab, pca_teacher as pt
from obson.babel import sampling_benchmark as sb, window_sampling as ws, coverage_audit as cov
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
import test_pca_anneal as anneal_tests
from test_pca_anneal import no_update, read
from test_pca_teacher import SMALL
from test_architecture_benchmark import features


class SamplingTests(unittest.TestCase):
    def test_strict_coverage_gate_checks_partitions_and_source_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';(source/'cache').mkdir(parents=True);audit=root/'audit';audit.mkdir()
            files={};replay={}
            for split in ('train','val','test','cross_research'):
                replay[split]={}
                for name in ('x','y','mask'):
                    key=f'{split}_{name}.npy';np.save(source/'cache'/key,np.zeros(1));files[key]=sha256(source/'cache'/key)
                    replay[split][name]=dict(source_binary_available=True,numeric_replay_passed=True,expected_sha256=files[key])
                atomic_json(replay[split],audit/f'{split}_cache_replay.json')
            atomic_json(dict(files=files),source/'cache/index.json')
            metrics=dict(schema=cov.SCHEMA,reports_only=False,all_cached_arrays_verified=True,
                sources={'architecture':str(source)},code_sha256={'coverage_audit.py':sha256(cov.__file__)},
                consumed_report_sha256={str(source/'cache/index.json'):sha256(source/'cache/index.json')},
                cache_replay=replay,profiles={'train':{'coverage':{'windows':12}}},
                train_candidates={'history512_stride16':{'coverage':{'windows':24}}},boundaries={})
            def write_report(m):
                atomic_json(m,audit/'coverage_metrics.json')
                atomic_json(dict(status='complete',training_updates=0,cache_mutations=0,
                    files={p.name:sha256(p) for p in audit.iterdir() if p.name!='completion.json'}),audit/'completion.json')
            write_report(metrics);self.assertEqual(ws.coverage_identity(audit,source)['train_count'],12)
            with self.assertRaisesRegex(ValueError,'sources differ'):ws.coverage_identity(audit,root/'different')
            for field,value in [('reports_only',True),('all_cached_arrays_verified',False),('cache_replay',{})]:
                bad=copy.deepcopy(metrics);bad[field]=value;write_report(bad)
                with self.assertRaises(ValueError):ws.coverage_identity(audit,source)
            write_report(metrics);(source/'cache/index.json').write_text('{}')
            with self.assertRaises(ValueError):ws.coverage_identity(audit,source)

    def test_membership_budget_random_state_and_resume_reproducibility(self):
        state = np.random.get_state()
        a = ws.sample_ids(38549, 4789, 42, 12)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        self.assertEqual(len(set(a)), 4789)
        self.assertTrue(np.all(np.diff(a)>0))
        self.assertTrue((a>=0).all() and (a<38549).all())
        for seed, epoch in [(42,13),(43,12)]:
            self.assertFalse(np.array_equal(a, ws.sample_ids(38549,4789,seed,epoch)))
        np.testing.assert_array_equal(a, ws.sample_ids(38549,4789,42,12))
        np.testing.assert_array_equal(ws.sample_ids(12,12,42,1),np.arange(12))
        for count,budget,epoch in [(3,4,1),(3,0,1),(3,2,0)]:
            with self.assertRaises(ValueError):ws.sample_ids(count,budget,42,epoch)

    def test_sample_plan_matches_explicit_bar_union_and_exposure_counts(self):
        inventory=[dict(key=f'A/15/C{i//4}',row=511+(i%4)*16,symbol='A',period=15,month='2020-01') for i in range(12)]
        plan=ws.sampling_plan(inventory,42,5,7);exposures=np.zeros(12,dtype=int);seen=set()
        for r in plan['epochs']:
            ids=ws.sample_ids(12,7,42,r['epoch']);exposures[ids]+=1
            bars={(inventory[i]['key'], b) for i in ids for b in range(inventory[i]['row']-127,inventory[i]['row']+1)}
            seen.update(bars)
            self.assertEqual(r['unique_input_bars'],len(bars))
            self.assertEqual(r['cumulative_unique_input_bars'],len(seen))
            self.assertEqual(r['cumulative_unique_windows'],int((exposures>0).sum()))
            self.assertEqual(r['strata']['symbol'],{'A':7})
        self.assertEqual(plan['exposure_counts'],exposures.tolist())
        self.assertEqual(plan['total_window_exposures'],35)
        self.assertEqual(plan['cumulative_strata']['period'],{'15':35})

    def test_subset_mmap_only_returns_requested_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.npy';a=np.arange(60).reshape(20,3);np.save(p,a)
            mmap=np.load(p,mmap_mode='r');ids=np.array([0,6,11,19]);view=ws.subset({'x':mmap},ids)['x']
            self.assertEqual(len(view),4);np.testing.assert_array_equal(view[[1,3]],a[[6,19]])
            self.assertIs(view.array,mmap);self.assertFalse(mmap.flags.writeable)

    def test_candidate_targets_replay_and_no_refitting(self):
        with tempfile.TemporaryDirectory() as td:
            cache=Path(td);x=features(3).reshape(-1,28)
            s=SimpleNamespace(key='A/15/C1',code='A',period=15,
                frame=pd.DataFrame({'datetime':pd.date_range('2020-01-01',periods=len(x),freq='15min')}))
            keys=[(0,127),(0,143),(0,255),(0,383)]
            old_x=np.stack([x[:128],x[128:256],x[256:384]])
            y,mask=ar.ordered_targets(old_x);stats=ar.fit_scales(old_x,y,mask)
            xx,yy,mm=ar.normalize(old_x,y,mask,stats);old=dict(x=xx,y=yy,mask=mm)
            oldinv=[dict(key=s.key,row=end,end=str(s.frame.datetime.iloc[end])) for end in (127,255,383)]
            pca=dict(mean=np.zeros(896),components=np.eye(896)[:8]);scales=dict(mean=[0.]*8,scale=[1.]*8)
            with patch.object(ar,'fit_scales',side_effect=AssertionError('No new stats')),patch.object(pt,'fit_coordinate_scales',side_effect=AssertionError('No PCA scaling refit')):
                inv,replay=ws.write_candidates([s],[x],keys,stats,pca,scales,oldinv,old,cache)
            self.assertEqual(replay['original_windows_matched'],3)
            self.assertFalse(any(replay['max_abs'].values()))
            values={k:np.load(cache/f'train_{k}.npy') for k in ('x','y','mask','q')}
            self.assertEqual(len(inv),4);self.assertFalse(values['mask'][:,-1].any())
            # New endpoint targets must be rebuilt relative to their own pre-window anchor.
            raw=x[16:144][None];ty,tm=ar.ordered_targets(raw);_,ty,tm=ar.normalize(raw,ty,tm,stats)
            np.testing.assert_array_equal(values['y'][1],ty[0])
            broken=dict(old,y=old['y'].copy());broken['y'][0,0,0]+=1
            with self.assertRaisesRegex(ValueError,'does not reproduce'):
                ws.write_candidates([s],[x],keys,stats,pca,scales,oldinv,broken,cache)
            with self.assertRaisesRegex(ValueError,'lost original'):
                ws.write_candidates([s],[x],keys[:-1],stats,pca,scales,oldinv,old,cache)

    def test_candidate_preparation_filters_and_pins_raw_source(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'coverage').mkdir();(out/'cache').mkdir();(out/'long').mkdir()
            raw=features(5).reshape(-1,28);dates=pd.date_range('2020-01-01',periods=len(raw),freq='h')
            series=SimpleNamespace(key='A/15/C1',code='A',period=15,frame=pd.DataFrame({'datetime':dates}),
                sessions=dates.normalize().to_numpy(),main=np.ones(len(raw),dtype=bool))
            bounds=dict(train_until='2020-01-25',val_until='2020-01-27',test_until='2020-01-30')
            old_x=raw[384:512][None];y,mask=ar.ordered_targets(old_x);stats=ar.fit_scales(old_x,y,mask)
            xx,yy,mm=ar.normalize(old_x,y,mask,stats);old=dict(x=xx,y=yy,mask=mm)
            inv=[dict(key=series.key,row=511,end=str(dates[511]))];atomic_json(inv,out/'coverage/train_windows.json')
            atomic_json(stats,out/'statistics.json');atomic_json(dict(mean=[0.]*8,scale=[1.]*8),out/'cache/coordinate_scales.json')
            ref=dict(sources=[dict(key='A/15/C1')]);atomic_json(dict(manifest=ref),out/'long/manifest.json')
            ends=cov.endpoints(series,bounds,'train',16,512)
            expected_bars=len({i for e in ends for i in range(e-127,e+1)})
            ci=dict(files={'train_windows.json':sha256(out/'coverage/train_windows.json')},boundaries=bounds,
                sources={'long':str(out/'long')},candidate=dict(coverage=dict(windows=len(ends),unique_contract_period_bars=expected_bars),counts_by_series={series.key:len(ends)}))
            meta=dict(source=str(out/'source'),raw_root=str(out/'raw'),coverage_identity=ci,windows_per_epoch=1,
                experiments=[dict(seed=42,epochs=2)])
            pca=dict(mean=np.zeros(896),components=np.eye(896)[:8])
            with patch.object(ws.data,'load_series',return_value=([series],None)),patch.object(ws.data,'manifest',return_value=ref),patch.object(ws.ae_context,'encode_context',return_value={'x':raw[:,:18]}),patch.object(ws.aa,'activity_features',return_value=(raw[:,18:],None)),patch.object(ws.cb,'source_pca',return_value=pca),patch.object(ab,'load_arrays',return_value=old),patch.object(ar,'fit_scales',side_effect=AssertionError('No refitting')):
                ws.prepare_candidates(meta,out)
                rows=read(out/'candidates/inventory.json');self.assertEqual([r['row'] for r in rows],list(ends))
                self.assertEqual(read(out/'candidates/data_audit.json')['original_replay']['original_windows_matched'],1)
                # Cache reuse must not touch raw data or fit anything again.
                with patch.object(ws.data,'load_series',side_effect=AssertionError('No rebuild')):ws.prepare_candidates(meta,out)
                (out/'candidates/index.json').unlink()
                with patch.object(ws.data,'manifest',return_value={'changed':True}):
                    with self.assertRaisesRegex(ValueError,'Raw candidate source'):ws.prepare_candidates(meta,out)

    def test_train_endpoints_never_cross_validation_or_history_gate(self):
        dates=pd.date_range('2020-01-01',periods=1100,freq='h')
        s=SimpleNamespace(frame=pd.DataFrame({'datetime':dates}),main=np.ones(len(dates),dtype=bool),sessions=dates.normalize().to_numpy())
        bounds=dict(train_until='2020-01-31',val_until='2020-02-10',test_until='2020-02-20')
        ids=cov.endpoints(s,bounds,'train',16,512)
        self.assertTrue(len(ids)>0)
        self.assertTrue((ids>=511).all());self.assertTrue(((ids-127)%16==0).all())
        mask=cov.rp.time_mask(s,bounds,'train')
        for e in ids:self.assertTrue(mask[e-511:e+1].all())


class SamplingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        anneal_tests.AnnealTests.setUpClass();cls.teacher=anneal_tests.AnnealTests.teacher;cls.identity=anneal_tests.AnnealTests.identity
        cls.source=anneal_tests.AnnealTests.source

    @classmethod
    def tearDownClass(cls):
        anneal_tests.AnnealTests.tearDownClass()

    def make_run(self, td):
        root=Path(td);coverage=root/'coverage';coverage.mkdir();out=root/'sampled';out.mkdir()
        # A deterministic synthetic pool exercises the complete worker/evaluation path.
        old=ab.load_arrays(self.source,'train');n=len(old['x'])
        inv=[dict(key=f'A/15/C{i}',row=511,end='2020-01-01',symbol='A',period=15,month='2020-01') for i in range(n)]
        atomic_json(inv,coverage/'train_windows.json')
        ci=dict(files={'train_windows.json':sha256(coverage/'train_windows.json')},train_count=n)
        with patch.object(pt,'CONFIG',SMALL):meta=sb.make_manifest(self.teacher,self.identity,coverage,ci,root/'raw')
        atomic_json(meta,out/'manifest.json')
        def prepare_candidates(meta,out):
            cache=out/'candidates';cache.mkdir()
            pool={k:np.concatenate([np.asarray(v),np.asarray(v)]) for k,v in old.items()}
            q=np.load(out/'cache/train_q.npy');pool['q']=np.concatenate([q,q])
            for k,v in pool.items():np.save(cache/f'train_{k}.npy',v)
            inventory=inv+[dict(r,row=527) for r in inv];atomic_json(inventory,cache/'inventory.json')
            for j in meta['experiments']:
                atomic_json(ws.sampling_plan(inventory,j['seed'],j['epochs'],n),cache/f'sampling_s{j["seed"]}.json')
            atomic_json(dict(manifest=meta,files={p.name:sha256(p) for p in cache.iterdir()}),cache/'index.json')
        with patch.object(ws,'prepare_candidates',side_effect=prepare_candidates):sb.prepare(meta,out)
        return meta,out

    def test_complete_budget_control_replay_selection_and_fixed_last(self):
        before={str(p):sha256(p) for p in self.teacher.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as td:
            loader=ab.load_arrays
            def restricted(root,split):self.assertIn(split,('train','val'));return loader(root,split)
            with patch.object(ab,'load_arrays',side_effect=restricted),patch.object(pt,'fit_coordinate_scales',side_effect=AssertionError('No refit')):
                meta,out=self.make_run(td);sb.preflight(meta,out,'cpu')
                with patch.object(sb,'run_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No neural updates')):
                    for j in meta['experiments']:sb.worker(out,j['name'],'cpu')
            def gated(root,split):
                if split in ('test','cross_research'):self.assertTrue((out/'selection_lock.json').exists())
                return loader(root,split)
            with patch.object(ab,'load_arrays',side_effect=gated):report=sb.evaluate(meta,out,'cpu')
            for rows in report['datasets'].values():
                self.assertEqual(len(rows['variants']),6);self.assertEqual(len(rows['fixed_last']),4)
                self.assertEqual(len(rows['paired']),2);self.assertEqual(len(rows['fixed_last_paired']),2)
            for j in meta['experiments']:
                summary=read(out/j['name']/'training_summary.json');self.assertEqual(summary['selected_epoch'],0)
                self.assertEqual(summary['window_exposures'],36);self.assertEqual(summary['optimizer_steps'],9)
                self.assertTrue(read(out/j['name']/'initialization.json')['matched'])
            self.assertIn('sampled_s42',(out/'examples.html').read_text())
            self.assertNotIn('NaN',(out/'sampling_metrics.json').read_text())
            with patch.object(sb,'run_epoch',side_effect=AssertionError('No rerun')):sb.worker(out,'sampled_s42','cpu')
            with patch.object(ws,'prepare_candidates',wraps=ws.prepare_candidates):sb.prepare(meta,out)
        self.assertEqual(before,{str(p):sha256(p) for p in self.teacher.rglob('*') if p.is_file()})

    def test_resume_restores_moments_and_exact_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);name='sampled_s42';calls=[]
            def interrupted(*args,**kwargs):
                if len(args)>7 and args[7] is not None:
                    calls.append(args[1]['x'].ids.copy())
                    if len(calls)==2:raise RuntimeError('interrupt')
                    opt=args[7];param=opt.param_groups[0]['params'][0]
                    opt.state[param]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(param,.123),exp_avg_sq=torch.full_like(param,.456))
                return no_update(*args,**kwargs)
            with patch.object(sb,'run_epoch',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):sb.worker(out,name,'cpu')
            first=read(out/name/'history.json')[0];load_opt=torch.optim.AdamW.load_state_dict;restored=[];after=[]
            def load(self,state):restored.append(copy.deepcopy(state));return load_opt(self,state)
            def resumed(*args,**kwargs):
                if len(args)>7 and args[7] is not None:after.append(args[1]['x'].ids.copy())
                return no_update(*args,**kwargs)
            with patch.object(sb,'run_epoch',side_effect=resumed),patch.object(torch.optim.AdamW,'load_state_dict',new=load):sb.worker(out,name,'cpu')
            np.testing.assert_array_equal(after[0],calls[1]);self.assertEqual(len(after),2)
            moment=next(iter(restored[0]['state'].values()));self.assertEqual(float(moment['step']),7.)
            self.assertTrue(torch.all(moment['exp_avg']==.123))
            self.assertEqual(read(out/name/'history.json')[0],first)
            self.assertEqual(read(out/name/'resume_validation.json')['epoch'],1)

    def test_sampling_tamper_and_budget_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);job=meta['experiments'][0]
            with patch.object(sb,'run_epoch',side_effect=no_update):sb.worker(out,job['name'],'cpu')
            history=read(out/job['name']/'history.json')
            for key,value in [('windows',13),('optimizer_steps',4),('lr',.2),('teacher_weight',.25),('sampling',{})]:
                bad=copy.deepcopy(history);bad[0][key]=value
                with self.assertRaises(ValueError):sb.verify_sampling_history(meta,out,job,bad)
            (out/'candidates/train_x.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):sb.verify_cache(meta,out)

    def test_resume_rejects_rehashed_candidate_cache(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);calls=[]
            def stop(*args,**kwargs):
                if len(args)>7 and args[7] is not None:raise RuntimeError('stop before update')
                return no_update(*args,**kwargs)
            with patch.object(sb,'run_epoch',side_effect=stop):
                with self.assertRaisesRegex(RuntimeError,'stop before'):sb.worker(out,'sampled_s42','cpu')
            index=read(out/'candidates/index.json')
            x=np.load(out/'candidates/train_x.npy');x[0,0,0]+=1;np.save(out/'candidates/train_x.npy',x)
            index['files']['train_x.npy']=sha256(out/'candidates/train_x.npy');atomic_json(index,out/'candidates/index.json')
            with self.assertRaisesRegex(ValueError,'changed since checkpoint'):sb.worker(out,'sampled_s42','cpu')

    def test_scheduler_dispatches_two_distinct_sampling_workers(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out=self.make_run(td);commands=[]
            def launch(command,**kwargs):
                commands.append(command)
                return SimpleNamespace(pid=100+len(commands),poll=lambda:0)
            with patch.object(sb.subprocess,'Popen',side_effect=launch):sb.run_phase(out,2,'base')
            self.assertEqual(len(commands),2)
            self.assertEqual({c[c.index('--name')+1] for c in commands},{'sampled_s42','sampled_s43'})
            self.assertTrue(all(c[1:3]==['-m','obson.babel.sampling_benchmark'] for c in commands))

    def test_export_failure_and_success_include_plans_exclude_weights(self):
        repo=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';(run/'candidates').mkdir(parents=True);download=root/'download'
            atomic_json({'seed':42},run/'candidates/sampling_s42.json')
            for suffix in ('pt','npy','npz'):(run/f'data.{suffix}').write_bytes(b'large binary')
            fake=root/'fail';fake.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGUMENT_LOG"\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,ARGUMENT_LOG=str(root/'arguments'),BABEL_SAMPLING_RUN=str(run),BABEL_SAMPLING_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            cmd=['bash',str(repo/'scripts/babel_sampling_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                args=(root/'arguments').read_text();self.assertIn('obson.babel.sampling_benchmark',args);self.assertIn('--coverage',args)
                with tarfile.open(download/'run_reports.tar.gz') as archive:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in archive.getnames()))
                    self.assertIn('run/candidates/sampling_s42.json',archive.getnames())
                    self.assertIn('run_status=failed',archive.extractfile('run/run_status.txt').read().decode())
            fake.write_text('#!/bin/sh\nexit 0\n');atomic_json(dict(status='complete'),run/'completion.json')
            result=subprocess.run(cmd+['all'],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
            with tarfile.open(download/'run_reports.tar.gz') as archive:
                self.assertIn('run_status=complete',archive.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
