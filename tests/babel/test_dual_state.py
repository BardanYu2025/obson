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
from torch.utils.data import DataLoader

from test_babel import series
from test_large_history import SMALL
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import rng_state
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import HistoryWindows
from obson.babel.large_history import LargeHistory, HierWindows
from obson.babel.reconstruction_fusion import ReconstructionFusion
from obson.babel.dual_state import (DualState, DualWindows, MODES, TASKS, parts_from_output,
    selection_key, experiment_jobs, sha256, verify_files, prepare, run_jobs, fit_latent_readout,
    publish, worker, source_identity)


def stats(width=16):
    return dict(long_mean=[.5]*width,long_scale=[2.]*width,short_mean=[-.2]*width,short_scale=[.5]*width,
                structure_mean=[1.,2.,.5],structure_scale=[2.,3.,.3])


def batch(width=16):
    return dict(long=torch.randn(2,width),short=torch.randn(2,width),
        y=torch.randn(2,4,128,7),mask=torch.ones(2,4,128,7,dtype=torch.bool),
        valid=torch.tensor([[False,False,True,True],[False,True,True,True]]),
        recent_y=torch.randn(2,64,7),recent_mask=torch.ones(2,64,7,dtype=torch.bool),structure=torch.randn(2,3))


class DualStateTests(unittest.TestCase):
    def test_single_vector_readout_and_gradient_without_optimizer(self):
        b=batch(); before={k:v.clone() for k,v in b.items()}; common=[]
        for mode in MODES:
            torch.manual_seed(42); model=DualState(mode,stats(),16,16,4).eval()
            r=model(b['long'],b['short']); common.append(model.recent_decoder.state_dict())
            self.assertEqual(r['z'].shape,(2,32 if mode=='concat' else 16))
            self.assertEqual(r['history'].shape,(2,4,128,7)); self.assertEqual(r['recent'].shape,(2,64,7))
            decoded=model.decode(r['z'])
            for key in ('history','recent','structure'): torch.testing.assert_close(decoded[key],r[key])
            if mode=='concat':
                torch.testing.assert_close(r['z'][:,:16],(b['long']-.5)/2)
                torch.testing.assert_close(r['z'][:,16:],(b['short']+.2)/.5)
            loss=parts_from_output(model,r,b,dict(recent=1.,history=2.))['total'].mean()
            loss.backward()
            if mode=='compress':
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.project.parameters()))
                self.assertGreater(model.project[0].weight.grad.abs().sum().item(),0)
            # Input cache/state tensors are read-only; no update is propagated into frozen encoders.
            for key in b: torch.testing.assert_close(before[key],b[key])
        for weights in common:
            for key in weights: torch.testing.assert_close(weights[key],common[0][key])

    def test_padding_excluded_and_structure_not_hidden(self):
        b=batch(); model=DualState('concat',stats(),16,16,4).eval(); r=model(b['long'],b['short'])
        original=parts_from_output(model,r,b,dict(recent=1.,history=1.))
        changed={k:v.clone() for k,v in b.items()}; changed['y'][~b['valid']]=1e6
        altered=parts_from_output(model,r,changed,dict(recent=1.,history=1.))
        for k in original: torch.testing.assert_close(original[k],altered[k])
        ref={k:1. for k in TASKS}; good={k:.99 for k in TASKS}
        bad={**good,'recent':.1,'structure_1':1.021}
        self.assertEqual(selection_key(good,ref)[0][0],0)
        self.assertEqual(selection_key(bad,ref)[0][0],1)
        self.assertLess(selection_key(good,ref)[0],selection_key(bad,ref)[0])
        for k in TASKS:
            worse={**good,k:2.}; self.assertEqual(selection_key(worse,ref)[0][0],1)
        with self.assertRaises(ValueError): selection_key({**good,'recent':float('nan')},ref)

    def test_plan_and_hashed_source_reject_corruption(self):
        jobs=experiment_jobs([42,43]); self.assertEqual(len(jobs),6)
        self.assertEqual([j['name'] for j in jobs],['concat_s42','compress_s42','concat_s43','compress_s43','long_s42','short_s42'])
        with self.assertRaises(ValueError): experiment_jobs([42,42])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source'; long=p/'long'; cache=source/'target_cache'
            cache.mkdir(parents=True); (source/'short').mkdir(); (long/'joint').mkdir(parents=True)
            (long/'joint/best.pt').write_bytes(b'long'); (long/'manifest.json').write_text('{}')
            (source/'short/best.pt').write_bytes(b'short')
            identity={'long_best':sha256(long/'joint/best.pt')}; (source/'manifest.json').write_text(json.dumps({'sources':identity}))
            files={}
            for split in ('train','val','test'):
                for k in ('long','short','y','mask','valid','teacher','recent_y','recent_mask','labels','keys'):
                    f=cache/f'{split}_{k}.npy'; f.write_bytes(b'test'); files[f.name]=sha256(f)
            (cache/'index.json').write_text(json.dumps(dict(identity=identity,files=files)))
            (source/'artifacts.json').write_text(json.dumps(dict(short=sha256(source/'short/best.pt'))))
            self.assertEqual(source_identity(source,long)['long_best'],identity['long_best'])
            (cache/'test_long.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'fingerprint'): source_identity(source,long)
            with self.assertRaisesRegex(ValueError,'fingerprint'): verify_files(cache,{'../manifest.json':'bad'})

    def test_prepare_shared_cache_alignment_statistics_and_reuse(self):
        data=series(4000); encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data]; bounds=split_boundaries(data)
        sets=[HistoryWindows(data,encoded,bounds,s) for s in ('train','val','test')]
        dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in data]
        hier=[HierWindows(ds,dummy,bounds,s) for ds,s in zip(sets,('train','val','test'))]
        long_model=LargeHistory(SMALL,aggregate_layers=1).eval().requires_grad_(False)
        short_model=ReconstructionFusion('short',16,16).eval()
        rng=np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source'; out=p/'out'; cache=source/'target_cache'
            cache.mkdir(parents=True); out.mkdir(); (source/'short').mkdir()
            torch.save(dict(model=short_model.state_dict()),source/'short/best.pt')
            for split,ds in zip(('train','val','test'),hier):
                rows=list(DataLoader(ds,batch_size=len(ds)))[0]
                for key in ('y','mask','valid'): np.save(cache/f'{split}_{key}.npy',rows[key].numpy())
                # Validation/test locations are deliberately shifted; input statistics must remain train-only.
                m=rng.normal(size=(len(ds),16)).astype(np.float32)+(0 if split=='train' else 50)
                s=rng.normal(size=(len(ds),16)).astype(np.float32)
                for key,value in dict(long=m,short=s,keys=np.array([(i,end) for i,end,_ in ds.items]),
                    recent_y=rows['y'][:,-1,-64:].numpy(),recent_mask=rows['mask'][:,-1,-64:].numpy(),
                    teacher=rows['y'].numpy()+.1,labels=np.ones(len(ds),np.int64)).items(): np.save(cache/f'{split}_{key}.npy',value)
            fake_ck=dict(metadata=dict(target_mean=[0.,0.,0.],target_scale=[1.,1.,1.]))
            with patch('obson.babel.dual_state.load_data',return_value=({'boundaries':bounds},data,encoded,sets)), \
                 patch('obson.babel.dual_state.load_long',return_value=(long_model,fake_ck)), \
                 patch('obson.babel.dual_state.ReconstructionFusion',return_value=short_model):
                prepare(source,source,'unused',out,{'source':'test'},4,'cpu')
                info=json.loads((out/'targets/statistics.json').read_text())
                np.testing.assert_allclose(info['stats']['long_mean'],np.load(cache/'train_long.npy').mean(0),atol=1e-6)
                np.testing.assert_allclose(np.load(out/'targets/test_structure.npy'),np.stack([hier[2][i]['targets'] for i in range(len(hier[2]))]))
                self.assertTrue(all(np.isfinite(v) and v>0 for v in info['normalizers'].values()))
                self.assertEqual(set(info['reference']),set(TASKS))
                self.assertNotIn('labels',DualWindows(cache,out/'targets','test')[0])
                with patch('obson.babel.dual_state.load_long',side_effect=AssertionError('cache not reused')):
                    prepare(source,source,'unused',out,{'source':'test'},4,'cpu')
                np.save(out/'targets/test_structure.npy',np.zeros((1,3)))
                with self.assertRaisesRegex(ValueError,'fingerprint'): prepare(source,source,'unused',out,{'source':'test'},4,'cpu')
            (out/'targets/index.json').unlink(); keys=np.load(cache/'test_keys.npy'); keys[0,1]+=1; np.save(cache/'test_keys.npy',keys)
            with patch('obson.babel.dual_state.load_data',return_value=({'boundaries':bounds},data,encoded,sets)):
                with self.assertRaisesRegex(ValueError,'endpoints'): prepare(source,source,'unused',out,{'source':'test'},4,'cpu')

    def test_latent_readout_does_not_select_on_test(self):
        rng=np.random.default_rng(7); xs=[rng.normal(size=(n,8)).astype(np.float32) for n in (80,20,20)]
        w=rng.normal(size=(8,4)); ys=[(x@w).astype(np.float32) for x in xs]
        recovered,info=fit_latent_readout(xs,ys)
        changed=[ys[0],ys[1],ys[2]+10000]
        again,new_info=fit_latent_readout(xs,changed)
        self.assertEqual(info,new_info)
        for a,b in zip(recovered,again): np.testing.assert_array_equal(a,b)
        self.assertLess(np.square(recovered[2]-ys[2]).mean(),.1)

    def test_parallel_limit_and_owned_process_cleanup(self):
        class Process:
            live=0; peak=0; instances=[]; fail=False
            def __init__(self,cmd,**kwargs):
                self.name=cmd[-1]; self.pid=len(self.instances)+1; self.ticks=0; self.done=False; self.terminated=False
                type(self).live+=1; type(self).peak=max(type(self).peak,type(self).live); type(self).instances.append(self)
            def poll(self):
                if self.done: return 1 if self.fail and self.name=='concat_s42' else 0
                if self.fail and self.name!='concat_s42': return None
                self.ticks+=1
                if self.ticks>=2: self.done=True; type(self).live-=1; return self.poll()
                return None
            def terminate(self):
                self.terminated=True
                if not self.done: type(self).live-=1; self.done=True
            def wait(self,timeout=None): return 0
            def kill(self): self.terminate()
        with tempfile.TemporaryDirectory() as tmp,patch('obson.babel.dual_state.subprocess.Popen',Process),patch('obson.babel.dual_state.time.sleep'):
            p=Path(tmp); (p/'manifest.json').write_text(json.dumps(dict(experiments=experiment_jobs([42,43]))))
            run_jobs(p,2); self.assertEqual(Process.peak,2); self.assertEqual(len(Process.instances),6)
            Process.instances=[]; Process.fail=True
            with self.assertRaisesRegex(RuntimeError,'concat_s42 failed'): run_jobs(p,2)
            self.assertTrue(any(x.terminated for x in Process.instances if x.name!='concat_s42'))

    def test_completed_resume_never_updates_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'targets').mkdir(); cache=p/'target_cache'; cache.mkdir()
            meta=dict(experiments=experiment_jobs([42]),source=str(p),lr=3e-4,batch=4,micro=4,epochs=2)
            job=meta['experiments'][0]; path=p/job['name']; path.mkdir()
            (p/'manifest.json').write_text(json.dumps(meta)); (p/'targets/statistics.json').write_text(json.dumps(dict(stats=stats(512))))
            (p/'targets/index.json').write_text('{}')
            for split in ('train','val'):
                for key in ('long','short','y','mask','valid','recent_y','recent_mask'): np.save(cache/f'{split}_{key}.npy',np.zeros((1,1),np.float32))
                np.save(p/f'targets/{split}_structure.npy',np.zeros((1,3),np.float32))
            model=DualState('concat',stats(512)); opt=torch.optim.AdamW(model.parameters(),lr=meta['lr'],weight_decay=.01)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2,eta_min=3e-5)
            state=dict(metadata=dict(manifest=meta,job=job,target_index=sha256(p/'targets/index.json')),
                epoch=2,best_epoch=1,best_key=(1,2.,2.),best_validation={},model=model.state_dict(),best_model=model.state_dict(),
                optimizer=opt.state_dict(),schedule=schedule.state_dict(),rng=rng_state(),history=[])
            publish(state,path); (path/'best.pt').unlink()
            threads=torch.get_num_threads()
            try:
                with patch('torch.optim.AdamW.step',side_effect=AssertionError('unexpected optimizer update')):
                    worker(p,job['name'],'cpu')
            finally: torch.set_num_threads(threads)
            ck=torch.load(path/'best.pt',weights_only=True)
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,ck['model'][k])

    def test_export_is_fresh_excludes_weights_and_distinguishes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); run=p/'dual'; (run/'concat_s42').mkdir(parents=True); (run/'targets').mkdir()
            (run/'concat_s42/metrics.json').write_text('{}'); (run/'concat_s42/best.pt').write_text('weights')
            (run/'targets/train_structure.npy').write_text('data'); (run/'targets/statistics.json').write_text('{}')
            env={**os.environ,'BABEL_DUAL_RUN':str(run),'BABEL_DOWNLOAD_DIR':str(p/'download'),'PYTHON_BIN':'/usr/bin/false'}
            def export(mode,code):
                r=subprocess.run(['bash','scripts/babel_dual_state_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,code,r.stderr)
                return tarfile.open(p/'download/dual_reports.tar.gz')
            with export('export',0) as t:
                names=t.getnames(); self.assertIn('dual/concat_s42/metrics.json',names)
                self.assertFalse(any('/targets/' in n or n.endswith('.pt') or n.endswith('.npy') for n in names))
                self.assertIn('training_status=partial',t.extractfile('dual/run_status.txt').read().decode())
            (run/'concat_s42/metrics.json').unlink(); (run/'completion.json').write_text('{"status":"complete"}')
            with export('export',0) as t:
                self.assertNotIn('dual/concat_s42/metrics.json',t.getnames())
                self.assertIn('training_status=complete',t.extractfile('dual/run_status.txt').read().decode())
            with export('all',1) as t:
                self.assertIn('command_exit_code=1',t.extractfile('dual/run_status.txt').read().decode())
                self.assertIn('training_status=failed',t.extractfile('dual/run_status.txt').read().decode())

    def test_full_evaluation_pipeline_on_synthetic_frozen_states(self):
        from obson.babel.dual_state import evaluate, validate
        rng=np.random.default_rng(44)
        original_model=LargeHistory(SMALL,blocks=4,aggregate_layers=1).eval().requires_grad_(False)
        short=ReconstructionFusion('short',16,16).eval()
        fake_ck=dict(metadata=dict(target_mean=[0.,0.,0.],target_scale=[1.,1.,1.]))
        make=lambda mode,st: DualState(mode,st,16,16,4)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'targets').mkdir(); source=p/'source'; cache=source/'target_cache'; cache.mkdir(parents=True)
            (source/'short').mkdir(); torch.save(dict(model=short.state_dict()),source/'short/best.pt')
            meta=dict(source=str(source),seeds=[42],experiments=experiment_jobs([42]))
            (p/'manifest.json').write_text(json.dumps(meta))
            info=dict(stats=stats(),normalizers=dict(recent=1.,history=1.),reference={k:1. for k in TASKS})
            (p/'targets/statistics.json').write_text(json.dumps(info))
            for split,n in (('train',24),('val',8),('test',8)):
                arrays=dict(long=rng.normal(size=(n,16)).astype(np.float32),short=rng.normal(size=(n,16)).astype(np.float32),
                    y=rng.normal(size=(n,4,128,7)).astype(np.float32),mask=np.ones((n,4,128,7),bool),valid=np.ones((n,4),bool),
                    recent_y=rng.normal(size=(n,64,7)).astype(np.float32),recent_mask=np.ones((n,64,7),bool),labels=np.arange(n)%3+1)
                arrays['teacher']=arrays['y']+.1
                for k,x in arrays.items(): np.save(cache/f'{split}_{k}.npy',x)
                np.save(p/f'targets/{split}_structure.npy',rng.normal(size=(n,3)).astype(np.float32))
            inventory=[dict(source='test/15/contract',row=i*128,blocks=4,end=str(i),anchor=100.,recent_anchor=101.) for i in range(8)]
            (p/'targets/test_inventory.json').write_text(json.dumps(inventory))
            val=DualWindows(cache,p/'targets','val')
            for job in meta['experiments']:
                model=make(job['mode'],stats()); score=validate(model,val,4,'cpu',info['normalizers'])
                path=p/job['name']; path.mkdir()
                torch.save(dict(metadata=dict(manifest=meta),epoch=0,model=model.state_dict(),validation=score),path/'best.pt')
            with patch('obson.babel.dual_state.load_long',return_value=(original_model,fake_ck)), \
                 patch('obson.babel.dual_state.ReconstructionFusion',return_value=short), \
                 patch('obson.babel.dual_state.DualState',side_effect=make):
                evaluate(p,p,4,'cpu')
            report=json.loads((p/'dual_state_metrics.json').read_text())
            self.assertEqual(len(report['variants']),4); self.assertIn('42',report['paired_comparison'])
            self.assertIn('frozen_references',report)
            self.assertTrue((p/'summary.md').exists())
            for job in meta['experiments']:
                r=report['variants'][job['name']]
                self.assertEqual(r['reconstruction']['history']['windows'],8)
                self.assertEqual(len(json.loads((p/job['name']/'recent_examples.json').read_text())),6)
                self.assertTrue(np.isfinite(r['adapted_legacy']['test_latent_standardized_mse']))
