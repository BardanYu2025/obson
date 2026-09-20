import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_babel import series
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import rng_state
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import HistoryWindows
from obson.babel.representation import time_mask
from obson.babel.residual_fusion import (MODES,ResidualFusion,class_weights,losses,extract_short,
                                         score,save_training,train_variant,feature_cache,retention)
from obson.babel.large_history import LargeHistory,HierWindows
from obson.babel.short_state import ShortState
from test_large_history import SMALL


class ResidualFusionTests(unittest.TestCase):
    def test_identity_initialization_and_matched_readout(self):
        m=torch.randn(5,16);s=torch.randn(5,16);heads=[]
        for mode in MODES:
            torch.manual_seed(42);model=ResidualFusion(mode,16);r=model(m,s)
            torch.testing.assert_close(r['z'],s if mode=='short' else m)
            heads.append(model.head[1].weight.detach())
            if mode in ('gated','long_gated'):
                torch.testing.assert_close(r['gate'],torch.full_like(m,.1))
        for head in heads:torch.testing.assert_close(head,heads[0])

    def test_gradient_and_long_only_capacity_control(self):
        frozen=torch.nn.Linear(18,16).requires_grad_(False)
        m=frozen(torch.randn(6,18));s=frozen(torch.randn(6,18));m_before=m.clone();s_before=s.clone()
        labels=torch.tensor([1,1,2,2,3,3]);weights=class_weights(labels)
        model=ResidualFusion('gated',16)
        loss=losses(model(m,s),m,labels,weights,.1,'gated')[0].mean();loss.backward()
        self.assertGreater(model.project.weight.grad.abs().sum().item(),0.)
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        torch.testing.assert_close(m,m_before);torch.testing.assert_close(s,s_before)
        control=ResidualFusion('long_gated',16)
        with torch.no_grad():control.project.weight.copy_(torch.eye(16))
        torch.testing.assert_close(control(m,s)['z'],control(m,s+100)['z'])
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in control.parameters()))
        with torch.no_grad():model.project.weight.copy_(torch.eye(16))
        self.assertFalse(torch.allclose(model(m,s)['z'],model(m,-s)['z']))

    def test_short_cache_exact_alignment_causality_and_batching(self):
        data=series(1600);encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data]
        bounds=split_boundaries(data);base=HistoryWindows(data,encoded,bounds,'test')
        keys=np.array([base.items[-1],base.items[0],base.items[len(base)//2]],np.int64)
        model=ShortState(16,2).eval()
        a=extract_short(model,base,bounds,'test',keys,1,'cpu')
        b=extract_short(model,base,bounds,'test',keys,3,'cpu')
        np.testing.assert_allclose(a,b,atol=1e-6)
        for j,(i,end) in enumerate(keys):
            lo=np.flatnonzero(time_mask(data[i],bounds,'test'))[0]
            with torch.no_grad():expected=model(torch.tensor(encoded[i]['x'][lo:end+1])[None])[0][0,-1].numpy()
            np.testing.assert_allclose(a[j],expected,atol=1e-6)
        i,end=keys[1];original=encoded[i]['x'].copy();encoded[i]['x'][end+1:]+=100
        changed=extract_short(model,base,bounds,'test',keys[1:2],1,'cpu')
        np.testing.assert_allclose(a[1],changed[0],atol=1e-6)
        encoded[i]['x']=original

    def test_scores_batch_invariant(self):
        torch.manual_seed(9);model=ResidualFusion('gated',16)
        with torch.no_grad():model.project.weight.normal_(std=.1)
        data=dict(long=torch.randn(11,16),short=torch.randn(11,16),labels=torch.tensor([1,2,3,1,1,2,3,1,2,3,3]))
        weights=class_weights(data['labels'])
        a=score(model,data,weights,.1,1);b=score(model,data,weights,.1,7)
        self.assertEqual(a['state'],b['state'])
        for key in ('objective','weighted_ce','relative_squared_drift','mean_gate','mean_relative_correction'):
            self.assertAlmostEqual(a[key],b[key],places=6)

    def test_cache_and_frozen_decoder_retention_pipeline(self):
        data=series(4000);encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data]
        bounds=split_boundaries(data);base=HistoryWindows(data,encoded,bounds,'test')
        dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in data]
        ds=HierWindows(base,dummy,bounds,'test',blocks=4,minimum=2);ds.items=ds.items[:3]
        long=LargeHistory(SMALL,blocks=4,aggregate_layers=1).eval().requires_grad_(False)
        short=ShortState(16,1).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            features=feature_cache(long,short,ds,bounds,'test',path,{'source':'test'},2,2,'cpu')
            cached=feature_cache(long,short,ds,bounds,'test',path,{'source':'test'},1,1,'cpu')
            for key,value in features.items():np.testing.assert_array_equal(value,cached[key])
            self.assertEqual(features['long'].shape,(len(ds),16))
            first=retention(ResidualFusion('long',16),long,ds,features,torch.zeros(3),torch.ones(3),2,'cpu',path)
            second=retention(ResidualFusion('gated',16),long,ds,features,torch.zeros(3),torch.ones(3),2,'cpu',path,True)
            self.assertEqual(first,second)
            self.assertTrue((path/'examples.html').exists())
            self.assertTrue(all(p.grad is None for p in long.parameters()))

    def test_completed_recovery_recreates_reports_without_training(self):
        torch.manual_seed(42);model=ResidualFusion('add');opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        meta=dict(seed=42,lr=3e-4,batch=4,preserve=.1,epochs=2)
        state=dict(metadata={**meta,'mode':'add'},epoch=2,best_epoch=1,best=.8,history=[dict(epoch=2)],
                   model=model.state_dict(),best_model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
        data=dict(long=np.ones((4,512),np.float32),short=np.zeros((4,512),np.float32),labels=np.array([1,2,3,1]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);save_training(state,path);(path/'best.pt').unlink();(path/'history.jsonl').unlink()
            recovered,epoch=train_variant('add',data,data,path,meta,'cpu')
            self.assertEqual(epoch,1);self.assertTrue((path/'best.pt').exists());self.assertTrue((path/'history.jsonl').exists())
            for key,value in recovered.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key])
            with self.assertRaisesRegex(ValueError,'metadata mismatch'):train_variant('add',data,data,path,{**meta,'batch':8},'cpu')

    def test_report_export_excludes_weights_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);run=path/'fusion';(run/'gated').mkdir(parents=True);(run/'feature_cache').mkdir()
            (run/'gated/metrics.json').write_text('{}');(run/'gated/best.pt').write_text('weights')
            (run/'feature_cache/train.json').write_text('{}')
            env={**os.environ,'BABEL_FUSION_RUN':str(run),'BABEL_DOWNLOAD_DIR':str(path/'download')}
            result=subprocess.run(['bash','scripts/babel_fusion_autodl.sh','export'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            dest=path/'download/fusion'
            self.assertTrue((dest/'gated/metrics.json').exists());self.assertFalse((dest/'gated/best.pt').exists())
            self.assertFalse((dest/'feature_cache').exists())
