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

from test_babel import series
from test_large_history import SMALL
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import rng_state
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import HistoryWindows
from obson.babel.large_history import LargeHistory,HierWindows
from obson.babel.reconstruction_fusion import (MODES,ReconstructionFusion,history_values,eligible,loss_parts,
                                               prepare_cache,CachedWindows,run_jobs,save_state,worker)


class ReconstructionFusionTests(unittest.TestCase):
    def test_matched_decoder_and_identity(self):
        m=torch.randn(3,16);s=torch.randn(3,16);weights=[]
        for mode in MODES:
            torch.manual_seed(42);model=ReconstructionFusion(mode,16,16).eval()
            r=model(m,s);torch.testing.assert_close(r['z'],s if mode=='short' else m)
            self.assertEqual(r['recent'].shape,(3,64,7));weights.append(model.decoder.state_dict())
        for w in weights:
            for key,value in w.items():torch.testing.assert_close(value,weights[0][key])

    def test_frozen_decoder_gradient_and_padding(self):
        long=LargeHistory(SMALL,blocks=4,aggregate_layers=1).eval().requires_grad_(False)
        model=ReconstructionFusion('add',16,16)
        b=dict(long=torch.randn(2,16),short=torch.randn(2,16),y=torch.randn(2,4,128,7),
               valid=torch.tensor([[False,False,True,True],[False,True,True,True]]),mask=torch.ones(2,4,128,7,dtype=torch.bool),
               recent_y=torch.randn(2,64,7),recent_mask=torch.ones(2,64,7,dtype=torch.bool))
        with torch.no_grad():b['teacher']=long.decode_history(b['long'])['reconstruction']
        parts=loss_parts(model,b,long)
        torch.testing.assert_close(parts['teacher'],torch.zeros(2))
        parts['total'].mean().backward()
        self.assertGreater(model.project.weight.grad.abs().sum().item(),0.)
        self.assertTrue(all(p.grad is None for p in long.parameters()));self.assertFalse(long.training)
        pred=b['teacher'].clone();before=list(history_values(pred,b['y'],b['mask'],b['valid']))
        pred[~b['valid']]+=10000;after=list(history_values(pred,b['y'],b['mask'],b['valid']))
        for a,c in zip(before,after):torch.testing.assert_close(a,c)

    def test_retention_selection_uses_both_metrics(self):
        baseline=dict(history_loss=1.,history_close_bps=100.)
        self.assertTrue(eligible(dict(history_loss=1.01,history_close_bps=101.),baseline,.02))
        self.assertFalse(eligible(dict(history_loss=.9,history_close_bps=103.),baseline,.02))
        self.assertFalse(eligible(dict(history_loss=1.03,history_close_bps=99.),baseline,.02))
        self.assertFalse(eligible(dict(history_loss=float('nan'),history_close_bps=100.),baseline,.02))

    def test_shared_target_cache_alignment_and_reuse(self):
        data=series(4000);encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data];bounds=split_boundaries(data)
        sets=[HistoryWindows(data,encoded,bounds,s) for s in ('train','val','test')]
        dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in data]
        hier=[HierWindows(ds,dummy,bounds,s) for ds,s in zip(sets,('train','val','test'))]
        arrays={s:dict(long=np.ones((len(ds),16),np.float32),short=np.zeros((len(ds),16),np.float32),
                       labels=np.ones(len(ds),np.int64),keys=np.array([(i,end) for i,end,_ in ds.items],np.int64))
                for s,ds in zip(('train','val','test'),hier)}
        long=LargeHistory(SMALL,aggregate_layers=1).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);source=path/'source';source.mkdir();out=path/'out';out.mkdir()
            (source/'manifest.json').write_text(json.dumps(dict(sources={})))
            with patch('obson.babel.reconstruction_fusion.load_data',return_value=({'boundaries':bounds},data,encoded,sets)), \
                 patch('obson.babel.reconstruction_fusion.load_long',return_value=(long,{})), \
                 patch('obson.babel.reconstruction_fusion.load_cache',side_effect=lambda _,s,__: (arrays[s],'sha')):
                prepare_cache(source,source,'unused',out,{'source':'test'},8,'cpu')
            ds=CachedWindows(out/'target_cache','test');item=ds[0];i,end=arrays['test']['keys'][0]
            expected=100*np.log(float(data[i].frame.close.iloc[end])/float(data[i].frame.close.iloc[end-64]))
            self.assertAlmostEqual(float(item['recent_y'][-1,1]),expected,places=3)
            self.assertNotIn('labels',item)
            with patch('obson.babel.reconstruction_fusion.load_data',side_effect=AssertionError('cache not reused')):
                prepare_cache(source,source,'unused',out,{'source':'test'},8,'cpu')

    def test_parallel_limit_and_failure_cleanup(self):
        class FakeProcess:
            live=0;peak=0;instances=[];fail=False
            def __init__(self,cmd,**kwargs):
                self.mode=cmd[-1];self.ticks=0;self.done=False;self.terminated=False
                type(self).live+=1;type(self).peak=max(type(self).peak,type(self).live);type(self).instances.append(self)
            def poll(self):
                if self.done:return 1 if type(self).fail and self.mode=='add' else 0
                if type(self).fail and self.mode!='add':return None
                self.ticks+=1
                if self.ticks>=2:
                    self.done=True;type(self).live-=1;return self.poll()
                return None
            def terminate(self):
                self.terminated=True
                if not self.done:type(self).live-=1;self.done=True
            def wait(self,timeout=None):return 0
            def kill(self):self.terminate()
        with tempfile.TemporaryDirectory() as tmp,patch('obson.babel.reconstruction_fusion.subprocess.Popen',FakeProcess),patch('obson.babel.reconstruction_fusion.time.sleep'):
            path=Path(tmp);run_jobs(path,path,2);self.assertEqual(FakeProcess.peak,2)
            self.assertEqual([p.mode for p in FakeProcess.instances],list(MODES))
            FakeProcess.instances=[];FakeProcess.fail=True
            with self.assertRaisesRegex(RuntimeError,'add failed'):run_jobs(path,path,2)
            self.assertTrue(any(p.terminated for p in FakeProcess.instances if p.mode!='add'))

    def test_completed_worker_resume_without_updates(self):
        model=ReconstructionFusion('long');opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        meta=dict(lr=3e-4,batch=4,fusion_micro=2,epochs=2,tolerance=.02)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'manifest.json').write_text(json.dumps(meta));path=out/'long';path.mkdir();cache=out/'target_cache';cache.mkdir()
            for split in ('train','val'):
                for key in ('long','short','y','mask','valid','teacher','recent_y','recent_mask'):np.save(cache/f'{split}_{key}.npy',np.zeros((1,1),np.float32))
            state=dict(metadata={**meta,'mode':'long','micro':4},epoch=2,best_epoch=1,best=1.,history=[],baseline={},
                       model=model.state_dict(),best_model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
            save_state(state,path);(path/'best.pt').unlink()
            threads=torch.get_num_threads()
            try:worker(out,'long',out,'cpu')
            finally:torch.set_num_threads(threads)
            ck=torch.load(path/'best.pt',weights_only=True)
            for k,v in ck['model'].items():torch.testing.assert_close(v,model.state_dict()[k])

    def test_automatic_archive_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);run=path/'experiment';(run/'add').mkdir(parents=True);(run/'target_cache').mkdir()
            (run/'add/metrics.json').write_text('{}');(run/'add/best.pt').write_text('weights');(run/'target_cache/index.json').write_text('{}')
            env={**os.environ,'BABEL_RECON_RUN':str(run),'BABEL_DOWNLOAD_DIR':str(path/'download'),'PYTHON_BIN':'/usr/bin/false'}
            for mode,expected in [('export',0),('all',1)]:
                r=subprocess.run(['bash','scripts/babel_reconstruction_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,expected,r.stderr)
                with tarfile.open(path/'download/experiment_reports.tar.gz') as archive:
                    names=archive.getnames();self.assertIn('experiment/add/metrics.json',names)
                    self.assertFalse(any('target_cache' in name or name.endswith('.pt') for name in names))
                    self.assertEqual(archive.extractfile('experiment/run_status.txt').read().decode(),f'exit_code={expected}\n')
