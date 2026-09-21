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
from obson.babel.ae_extend import atomic_save, rng_state
from obson.babel.ar_reconstruction import read_arrays, device_arrays
from obson.babel.detail_alignment import (DetailModel, PackedStreams, detail_scales, detail_values, values,
    eligible, improves, sequence_specs, run_epoch, worker, evaluate, preflight)


def fixture(root):
    torch.manual_seed(13)
    model = DetailModel(16, 16).eval()
    source, short, out = root/'source', root/'short', root/'out'
    (source/'target_cache').mkdir(parents=True); (source/'short').mkdir(); short.mkdir(); (out/'cache').mkdir(parents=True)
    atomic_save(dict(model=model.encoder.state_dict()), short/'best.pt')
    atomic_save(dict(model=model.head.state_dict()), source/'short/best.pt')
    rng = np.random.default_rng(42)
    for split in ('train', 'val', 'test'):
        x = rng.normal(size=(768, 18)).astype(np.float32)
        specs = [dict(series=i, lo=0, offset=i*384, length=384, endpoints=[[127+j*128,i*3+j] for j in range(3)]) for i in range(2)]
        np.save(out/f'cache/{split}_x.npy', x)
        (out/f'cache/{split}_sequences.json').write_text(json.dumps(specs))
        with torch.no_grad():
            z = np.concatenate([model.encoder(torch.tensor(x[i*384:(i+1)*384])[None])[0][0,[127,255,383]].numpy() for i in range(2)])
        y = rng.normal(size=(6,64,7)).astype(np.float32)*.1; y[...,2:] = np.abs(y[...,2:])
        for name, a in [('short',z),('recent_y',y),('recent_mask',np.ones_like(y,dtype=bool)),('labels',np.arange(6)%4)]:
            np.save(source/f'target_cache/{split}_{name}.npy',a)
    data = device_arrays(read_arrays(source,'val'),'cpu'); streams = PackedStreams(out/'cache','val')
    scales = detail_scales(read_arrays(source,'train')['y'])
    reference = run_epoch(model,data,streams,False,False,scales,2)[0]
    (out/'cache/statistics.json').write_text(json.dumps(dict(scales=scales,reference=reference)))
    inventory = [dict(key='rb/60/rb2505',symbol='rb',period=60,end=str(j),anchor=100.,week=str(j)) for j in range(6)]
    (out/'cache/test_inventory.json').write_text(json.dumps(inventory))
    meta = dict(config=dict(width=16,decoder_width=16),source=str(source),short_run=str(short),streams=2,
                encoder_lr=3e-5,decoder_lr=1e-4,detail_weight=.25,epochs=1,seeds=[42],
                experiments=[dict(name=f'{m}_s42',mode=m,seed=42) for m in ('frozen_base','frozen_detail','joint_base','joint_detail')])
    (out/'manifest.json').write_text(json.dumps(meta))
    return model, meta, out, data, scales, reference


class DetailAlignmentTests(unittest.TestCase):
    def test_detail_scale_offset_invariance_and_retention_gates(self):
        y = torch.randn(3,64,7)
        scales = detail_scales(y.numpy())
        shifted = y.clone(); shifted[...,1] += 10
        self.assertLess(float(detail_values(shifted,y,scales).max()),1e-10)
        base = dict(base=1.,close_bps=10.,change16_mse=2.,detail=.5)
        self.assertTrue(improves(dict(base=1.01,close_bps=10.1,change16_mse=2.01,detail=.4),base,base))
        for key, bad in [('base',1.06),('close_bps',10.21),('change16_mse',2.11),('detail',float('nan'))]:
            candidate = dict(base,detail=.1); candidate[key] = bad
            self.assertFalse(eligible(candidate,base))

    def test_sequence_packing_boundary_and_future_exclusion(self):
        ss = series(800)[:1]
        bounds = dict(train_until=str(ss[0].sessions[299]),val_until=str(ss[0].sessions[699]),test_until='2030-01-01')
        x = np.arange(800*18,dtype=np.float32).reshape(800,18)
        keys = np.array([[0,511],[0,639]])
        packed,spec = sequence_specs(ss,[dict(x=x)],bounds,'val',keys)
        lo = spec[0]['lo']; self.assertEqual(lo,300)
        np.testing.assert_array_equal(packed,x[lo:640])
        self.assertEqual(spec[0]['endpoints'],[[211,0],[339,1]])
        changed = x.copy(); changed[640:] = -999
        other,_ = sequence_specs(ss,[dict(x=changed)],bounds,'val',keys)
        np.testing.assert_array_equal(packed,other)
        with self.assertRaisesRegex(ValueError,'crosses partition'):
            sequence_specs(ss,[dict(x=x)],bounds,'val',np.array([[0,350]]))

    def test_cached_and_live_alignment_same_endpoint_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,meta,out,data,scales,_ = fixture(Path(tmp)); streams = PackedStreams(out/'cache','val')
            a,z,p = run_epoch(model,data,streams,False,False,scales,2,collect=True)
            b,zz,pp = run_epoch(model,data,streams,True,False,scales,2,collect=True)
            np.testing.assert_allclose(z,zz,atol=2e-5,rtol=2e-5)
            np.testing.assert_allclose(p,pp,atol=2e-5,rtol=2e-5)
            self.assertAlmostEqual(a['detail'],b['detail'],places=5)
            frozen = [(reset,pick) for reset,x,pick in streams.batches(1,42,False)]
            joint = [(reset,pick) for reset,x,pick in streams.batches(1,42,True)]
            self.assertEqual(frozen,joint)
            # No optimizer executes during synthetic full joint-backward preflight.
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no local training')):
                self.assertTrue(np.isfinite(preflight(meta,out,'cpu')['gradient_norm']))

    def test_configure_and_detail_gradient_reaches_encoder(self):
        m = DetailModel(16,16); m.configure(True)
        self.assertFalse(any(p.requires_grad for p in m.encoder.decoder.parameters()))
        x = torch.randn(2,128,18); h,_ = m.encoder(x); pred = m.head.decode_recent(h[:,-1])
        target = torch.randn_like(pred); loss = detail_values(pred,target,[.1,.2,.3]).mean(); loss.backward()
        self.assertGreater(float(m.encoder.rnn.weight_ih_l0.grad.abs().sum()),0.)
        m.configure(False)
        self.assertFalse(any(p.requires_grad for p in m.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.head.parameters()))

    def test_completed_resume_preserves_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,meta,out,data,scales,reference = fixture(Path(tmp))
            job = meta['experiments'][1]; path = out/job['name']; path.mkdir()
            opt = torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr'])],weight_decay=.01)
            state = dict(metadata=dict(manifest=meta,job=job),epoch=1,best_epoch=0,history=[],best_validation=reference,
                         model=model.state_dict(),best_model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
            atomic_save(state,path/'last.pt')
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('no training')): worker(out,job['name'],'cpu')
            ck = torch.load(path/'best.pt',weights_only=True)
            self.assertEqual(ck['epoch'],0)
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,ck['model'][k],rtol=0,atol=0)

    def test_full_report_with_unchanged_encoders_and_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,meta,out,data,scales,reference = fixture(Path(tmp))
            for job in meta['experiments']:
                path = out/job['name']; path.mkdir()
                score = run_epoch(model,data,PackedStreams(out/'cache','val'),job['mode'].startswith('joint'),False,scales,2)[0]
                atomic_save(dict(metadata=dict(manifest=meta,job=job),epoch=0,model=model.state_dict(),validation=score),path/'best.pt')
            evaluate(out,'cpu')
            result = json.loads((out/'detail_metrics.json').read_text())
            self.assertEqual(len(result['variants']),5); self.assertEqual(len(result['paired']),4)
            self.assertEqual(result['variants']['original']['relative_latent_mse'],0.)
            for r in result['variants'].values():
                self.assertTrue(r['validation_retained']); self.assertIn('legacy_head',r); self.assertIn('state_probe',r)
            same = result['paired']['frozen_detail_s42_minus_frozen_base_s42']['normalized_detail']
            self.assertEqual(same['delta_mean'],0.)
            self.assertTrue((out/'joint_detail_s42/examples.html').exists())

    def test_export_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            (run/'metrics.json').write_text('{}'); (run/'best.pt').write_bytes(b'weights'); (run/'train_x.npy').write_bytes(b'cache')
            env = dict(os.environ,BABEL_DETAIL_RUN=str(run),BABEL_DETAIL_LOG=str(root/'none.log'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                r = subprocess.run(['bash','scripts/babel_detail_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,code,r.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertFalse(any(n.endswith(('.pt','.npy')) for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())


if __name__ == '__main__': unittest.main()
