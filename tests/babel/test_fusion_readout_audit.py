import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from obson.babel.ae_diagnostics import fingerprint
from obson.babel.ae_extend import atomic_json
from obson.babel.fusion_readout_audit import load_cache,geometry,vectors
from obson.babel.residual_fusion import ResidualFusion


class FusionReadoutAuditTests(unittest.TestCase):
    def test_cache_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);(source/'feature_cache').mkdir();path=source/'feature_cache/train.npz'
            np.savez(path,long=np.ones((4,512),np.float32),short=np.zeros((4,512),np.float32),labels=np.array([1,2,3,1]),keys=np.zeros((4,2),np.int64))
            atomic_json(dict(identity={'source':'frozen'},sha256=fingerprint(path)),path.with_suffix('.json'))
            data,_=load_cache(source,'train',{'source':'frozen'});self.assertEqual(data['long'].shape,(4,512))
            with self.assertRaisesRegex(ValueError,'fingerprint'):load_cache(source,'train',{'source':'other'})
            with path.open('ab') as file:file.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'fingerprint'):load_cache(source,'train',{'source':'frozen'})

    def test_vectors_do_not_modify_checkpoint_and_are_batch_invariant(self):
        model=ResidualFusion('gated',16)
        with torch.no_grad():model.project.weight.normal_(std=.1)
        original={k:v.clone() for k,v in model.state_dict().items()}
        data=dict(long=np.random.default_rng(42).normal(size=(11,16)).astype(np.float32),short=np.ones((11,16),np.float32),labels=np.ones(11,int))
        a=vectors(model,data,1);b=vectors(model,data,7)
        np.testing.assert_allclose(a,b,atol=1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k])
        self.assertAlmostEqual(geometry(np.ones((8,4)))['common_mean_energy_fraction'],1.)
        self.assertEqual(geometry(np.ones((8,4)))['fraction_std_below_001'],1.)
