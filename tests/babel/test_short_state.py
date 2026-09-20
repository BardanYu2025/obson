import unittest

import numpy as np
import torch

from test_babel import series
from obson.babel.ae_context import encode_context
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import HistoryWindows
from obson.babel.short_state import ShortState, Streams, multiscale_loss, run_epoch
from obson.babel.large_history import LargeHistory, HierWindows
from obson.babel.stage_audit import components
from test_large_history import SMALL


class ShortStateTests(unittest.TestCase):
    def test_chunked_state_matches_full_and_is_causal(self):
        torch.manual_seed(42);model=ShortState(16,2).eval();x=torch.randn(2,150,18)
        full,state=model(x)
        left,s=model(x[:,:73]);right,carried=model(x[:,73:],s.detach())
        torch.testing.assert_close(torch.cat((left,right),1),full)
        torch.testing.assert_close(carried,state)
        changed=x.clone();changed[:,100:]+=100
        torch.testing.assert_close(model(changed)[0][:,:100],full[:,:100])
        # Separate lanes do not mix; reset is explicitly None.
        torch.testing.assert_close(model(x[:1])[0],full[:1])
        torch.testing.assert_close(model(x[:,73:],None)[0],model(x[:,73:])[0])
        _,s=model(x[:,:73]);s=s.detach();h,_=model(x[:,73:],s)
        y=torch.randn(2,64,7);y[...,2:]=y[...,2:].abs()
        loss=multiscale_loss(model.decode(h[:,-1]),y,torch.ones_like(y,dtype=torch.bool)).mean()
        loss.backward();self.assertTrue(torch.isfinite(model.rnn.weight_ih_l0.grad).all())
        self.assertIsNone(s.grad_fn)

    def test_stream_endpoints_targets_and_partition_reset(self):
        data=series(1600);encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data]
        bounds=split_boundaries(data)
        for split in ('train','val','test'):
            base=HistoryWindows(data,encoded,bounds,split);ds=Streams(base,bounds,split)
            seen=[];last={};resets=0
            for reset,x,indices,samples in ds.batches(2):
                if reset:last={};resets+=1
                for (lane,step),sample in zip(indices,samples):
                    key=(sample['series'],sample['row']);seen.append(key)
                    if lane in last:self.assertGreater(sample['row'],last[lane])
                    last[lane]=sample['row']
                    i,end=key
                    np.testing.assert_array_equal(x[lane,step],encoded[i]['x'][end])
                    expected=100*np.log(float(data[i].frame.close.iloc[end])/sample['anchor'])
                    self.assertAlmostEqual(float(sample['y'][-1,1]),expected,places=3)
                    self.assertEqual(sample['y'].shape,(64,7))
            self.assertEqual(sorted(seen),sorted(base.items));self.assertGreater(resets,0)
        # Frozen-model evaluation independent of stream grouping / trailing padding.
        model=ShortState(16,1).eval()
        self.assertAlmostEqual(run_epoch(model,ds,1,'cpu'),run_epoch(model,ds,2,'cpu'),places=5)

    def test_audit_components_batch_invariant(self):
        data=series(4000);encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data]
        bounds=split_boundaries(data);base=HistoryWindows(data,encoded,bounds,'test')
        banks=[np.zeros((len(range(127,len(s.frame),128)),16),np.float32) for s in data]
        ds=HierWindows(base,banks,bounds,'test',blocks=4,minimum=2)
        model=LargeHistory(SMALL,blocks=4,aggregate_layers=1).eval()
        a=components(model,ds,torch.zeros(3),torch.ones(3),1,'cpu')
        b=components(model,ds,torch.zeros(3),torch.ones(3),3,'cpu')
        self.assertEqual(a['windows'],b['windows'])
        for k in a['means']:self.assertAlmostEqual(a['means'][k],b['means'][k],places=4)
