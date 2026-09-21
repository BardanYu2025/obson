import copy
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_babel import series, frame
from test_large_history import SMALL
from obson.babel.ae_context import encode_context
from obson.babel.ae_extend import atomic_save
from obson.babel.dual_state import sha256
from obson.babel.large_history import LargeHistory
from obson.babel.reconstruction_fusion import ReconstructionFusion
from obson.babel.short_state import ShortState
from obson.babel.streaming import CausalFeatures, DualStream, frame_bars, build_bundle, load_bundle
from obson.babel.stream_audit import audit, comparison


def models():
    torch.manual_seed(17)
    return (LargeHistory(SMALL,blocks=4,aggregate_layers=1).eval(),
            ShortState(16,2).eval(),ReconstructionFusion('short',16,16).eval())


def stats():
    return dict(long_mean=[.1]*16,long_scale=[2.]*16,short_mean=[.2]*16,short_scale=[.7]*16)


def engine(mods=None):
    return DualStream(*(mods or models()),stats(),[0.,0.,0.],[1.,1.,1.],'rb/60/rb2505',60,'test-source')


class StreamingTests(unittest.TestCase):
    def test_incremental_features_match_batch_and_are_causal(self):
        df=frame(700); df.loc[100:150,'oi_available']=False
        expected=encode_context(df,60,'ema8_32')['x']
        codec=CausalFeatures(); actual=[]
        for b in frame_bars(df,'rb',60): actual.append(codec.advance(b)[0])
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
        changed=df.copy(); changed.loc[200:,['open','high','low','close']]*=3
        changed_codec=CausalFeatures(); altered=[changed_codec.advance(b)[0] for b in frame_bars(changed,'rb',60)]
        np.testing.assert_array_equal(np.asarray(actual)[:200],np.asarray(altered)[:200])
        saved=codec.snapshot(); b=next(frame_bars(df.iloc[-1:],'rb',60))
        for bad in (b,replace(b,closed=False),replace(b,high=0),replace(b,volume=float('nan'))):
            with self.assertRaises(ValueError): codec.advance(bad)
            self.assertEqual(codec.snapshot(),saved)

    def test_stream_batch_equivalence_grid_and_partition_reset(self):
        df=frame(780); encoded=encode_context(df,60,'ema8_32')['x']; bars=list(frame_bars(df,'rb/60/rb2505',60))
        stream=engine(); lo=17
        for b in bars[:lo]: stream.warm_features(b)
        stream.start_partition('test')
        for j,b in enumerate(bars[lo:768],lo):
            result=stream.push(b)
            if j<639: self.assertIsNone(result['embedding'])
            if j==639:
                self.assertTrue(result['long_refreshed']); self.assertEqual(result['metadata']['long_age_bars'],0)
                snap=result
            if j==640:
                self.assertEqual(result['metadata']['long_age_bars'],1)
                torch.testing.assert_close(result['long'],snap['long'],rtol=0,atol=0)
        with torch.no_grad():
            short,_=stream.short_model(torch.tensor(encoded[lo:768])[None])
            # Last4 complete blocks 256..767 after sliding capacity4.
            local=stream.long_model.local.encode(torch.tensor(encoded[256:768]).reshape(4,128,18))[:,-1][None]
            anchor=float(df.close.iloc[639]); anchors=[float(df.close.iloc[k-1]) for k in (256,384,512,640)]
            offsets=torch.tensor([[100*(np.log(x)-np.log(anchor)) for x in anchors]],dtype=torch.float32)
            long=stream.long_model.summarize(local,offsets,torch.ones(1,4,dtype=torch.bool))
        torch.testing.assert_close(stream.short,short[:,-1],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(stream.long,long,atol=2e-5,rtol=2e-5)
        self.assertEqual(stream.long_row,767); self.assertEqual(len(stream.blocks),4)
        self.assertEqual(stream.long_anchor,anchor)
        decoded=stream.decode(); current=stream.current()
        with torch.no_grad():
            torch.testing.assert_close(decoded['recent']['coordinates'],stream.recent_decoder.decode_recent(short[:,-1]),atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(decoded['history']['coordinates'],stream.long_model.decode_history(long)['reconstruction'],atol=2e-5,rtol=2e-5)
        self.assertAlmostEqual(decoded['recent']['anchor'],float(df.close.iloc[703]))
        # Stateless decoder remains correct after the stream itself advances.
        stream.push(bars[768])
        again=stream.decode_embedding(current['embedding'],decoded['recent']['anchor'],decoded['history']['anchor'],4,current['metadata'])
        torch.testing.assert_close(again['recent']['coordinates'],decoded['recent']['coordinates'])
        self.assertEqual(stream.current()['metadata']['long_age_bars'],1)
        features=stream.features.snapshot(); stream.start_partition('next')
        self.assertEqual(features,stream.features.snapshot()); self.assertIsNone(stream.current()['long']); self.assertIsNone(stream.current()['short'])

    def test_snapshots_resume_partial_blocks_and_reset_contract(self):
        bars=list(frame_bars(frame(650),'rb/60/rb2505',60)); mods=models(); a=engine(mods); b=engine(mods)
        for bar in bars[:601]: a.push(bar)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'; atomic_save(a.snapshot(),path)
            b.restore(torch.load(path,weights_only=True))
        for bar in bars[601:]:
            x,y=a.push(bar),b.push(bar)
            torch.testing.assert_close(x['embedding'],y['embedding'],rtol=0,atol=0)
            self.assertEqual(x['metadata'],y['metadata'])
        before=a.snapshot()
        with self.assertRaisesRegex(ValueError,'explicitly reset'): a.push(replace(bars[-1],key='another'))
        self.assertEqual(a.features.snapshot(),before['features'])
        corrupted=copy.deepcopy(before); corrupted['identity']='other-weights'
        with self.assertRaisesRegex(ValueError,'mismatch'): a.restore(corrupted)
        # Explicit reset removes all old contract prices and hidden state.
        a.reset_contract('next',60); self.assertEqual(a.features.row,-1); self.assertIsNone(a.hidden); self.assertFalse(a.blocks)
        fresh=engine(mods); fresh.reset_contract('next',60)
        x=a.push(replace(bars[0],key='next')); y=fresh.push(replace(bars[0],key='next'))
        torch.testing.assert_close(x['short'],y['short'],rtol=0,atol=0)
        self.assertIsNone(x['embedding']); self.assertIsNone(a.decode()['recent'])

    def test_future_extension_cannot_mutate_exported_vectors(self):
        bars=list(frame_bars(frame(520),'rb/60/rb2505',60)); stream=engine()
        for bar in bars[:512]: saved=stream.push(bar)
        values={k:v.clone() for k,v in saved.items() if torch.is_tensor(v)}
        for bar in bars[512:]: stream.push(bar)
        for k,v in values.items(): torch.testing.assert_close(saved[k],v,rtol=0,atol=0)
        self.assertEqual(saved['metadata']['long_age_bars'],0)
        self.assertEqual(stream.current()['metadata']['long_age_bars'],8)

    def test_bundle_verifies_lineage_and_loads_without_training(self):
        mods=models()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); long=p/'long'; short=p/'short'; recon=p/'recon'; fusion=p/'fusion'
            for path in (long/'joint',short,recon/'short',fusion): path.mkdir(parents=True)
            (long/'manifest.json').write_text('{}'); (short/'manifest.json').write_text('{}')
            torch.save(dict(epoch=20,model=mods[0].state_dict(),metadata=dict(target_mean=[0.,0.,0.],target_scale=[1.,1.,1.])),long/'joint/best.pt')
            torch.save(dict(epoch=21,model=mods[1].state_dict()),short/'best.pt')
            torch.save(dict(epoch=33,model=mods[2].state_dict(),metadata=dict(mode='short')),recon/'short/best.pt')
            hashes={k:sha256(path) for k,path in dict(long_manifest=long/'manifest.json',long_best=long/'joint/best.pt',short_manifest=short/'manifest.json',short_best=short/'best.pt').items()}
            (fusion/'manifest.json').write_text(json.dumps(dict(sources=hashes)))
            (recon/'manifest.json').write_text(json.dumps(dict(sources=dict(fusion_manifest=sha256(fusion/'manifest.json'),long_best=hashes['long_best']))))
            (recon/'artifacts.json').write_text(json.dumps(dict(short=sha256(recon/'short/best.pt'))))
            rng=np.random.default_rng(8); train={k:rng.normal(size=(20,16)).astype(np.float32) for k in ('long','short')}
            with patch('obson.babel.streaming.load_cache',return_value=(train,'trainhash')):
                meta,identity=build_bundle(long,short,recon,fusion,p/'bundle.pt')
            with patch('obson.babel.streaming.LargeHistory',return_value=mods[0]),patch('obson.babel.streaming.ShortState',return_value=mods[1]),patch('obson.babel.streaming.ReconstructionFusion',return_value=mods[2]):
                instance=load_bundle(p/'bundle.pt','rb/60/rb2505',60)
            self.assertEqual(instance.identity,identity); self.assertTrue(all(not x.requires_grad for model in mods for x in model.parameters()))
            np.testing.assert_allclose(meta['statistics']['long_mean'],train['long'].mean(0))
            (short/'best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'differ'): build_bundle(long,short,recon,fusion,p/'other.pt')

    def test_readonly_audit_end_to_end_synthetic_data(self):
        from obson.babel.data import split_boundaries
        from obson.babel.history_autoencoder import HistoryWindows
        from obson.babel.large_history import HierWindows
        from obson.babel.representation import time_mask
        data=series(4000); bounds=split_boundaries(data); encoded=[encode_context(s.frame,s.period,'ema8_32') for s in data]
        sets=[HistoryWindows(data,encoded,bounds,s) for s in ('train','val','test')]
        dummy=[np.zeros((len(range(127,len(s.frame),128)),1),np.float32) for s in data]
        ds=HierWindows(sets[2],dummy,bounds,'test',blocks=4)
        keys=np.array([(i,end) for i,end,_ in ds.items]); mods=models(); stream=engine(mods)
        values=dict(keys=keys,long=[],short=[],targets=[])
        with torch.no_grad():
            for j,(i,end) in enumerate(keys):
                b=ds[j]; local=mods[0].local.encode(torch.tensor(b['x']))[:,-1][None]
                m=mods[0].summarize(local,torch.tensor(b['offsets'])[None],torch.tensor(b['valid'])[None])
                lo=int(np.flatnonzero(time_mask(data[i],bounds,'test'))[0])
                s,_=mods[1](torch.tensor(encoded[i]['x'][lo:end+1])[None])
                values['long'].append(m[0].numpy()); values['short'].append(s[0,-1].numpy()); values['targets'].append(b['targets'])
        for k in ('long','short','targets'): values[k]=np.asarray(values[k])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); cache=p/'target_cache'; cache.mkdir(); (p/'manifest.json').write_text('{"sources":{}}'); (p/'bundle.pt').write_bytes(b'weights')
            samples=[ds[j] for j in range(len(ds))]
            arrays={k:np.stack([v[k] for v in samples]) for k in ('y','mask','valid')}
            arrays.update(long=values['long'],short=values['short'],keys=keys,teacher=arrays['y'].copy(),
                recent_y=arrays['y'][:,-1,-64:].copy(),recent_mask=arrays['mask'][:,-1,-64:].copy())
            for j,(i,end) in enumerate(keys): arrays['recent_y'][j,:,:2]-=100*np.log(float(data[i].frame.close.iloc[end-64])/samples[j]['anchor'])
            for k,v in arrays.items(): np.save(cache/f'test_{k}.npy',v)
            with patch('obson.babel.stream_audit.load_cache',return_value=(values,'testhash')),patch('obson.babel.stream_audit.load_bundle',return_value=stream), \
                 patch('obson.babel.stream_audit.load_data',return_value=({'boundaries':bounds},data,encoded,sets)):
                audit(p/'bundle.pt',p,p,p,'unused',p,batch=2,replays=2,device='cpu')
            report=json.loads((p/'stream_metrics.json').read_text()); self.assertTrue(report['passed']); self.assertFalse(report['training'])
            self.assertEqual(report['windows'],len(keys)); self.assertEqual(len(report['stream_replays']),2)
            self.assertTrue(all(x['checks']['long']['passed'] and x['checks']['short']['passed'] for x in report['stream_replays']))
            self.assertTrue((p/'summary.md').exists())
            # A numerical failure is recorded for every selected endpoint before the audit exits.
            with patch('obson.babel.stream_audit.load_cache',return_value=(values,'testhash')),patch('obson.babel.stream_audit.load_bundle',return_value=stream), \
                 patch('obson.babel.stream_audit.load_data',return_value=({'boundaries':bounds},data,encoded,sets)), \
                 patch('obson.babel.stream_audit.short_numerics',return_value=dict(passed=False,checks=dict(decoder_price=dict(passed=False)))):
                with self.assertRaisesRegex(ValueError,'full diagnostics saved'):
                    audit(p/'bundle.pt',p,p,p,'unused',p,batch=2,replays=2,device='cpu')
            failed=json.loads((p/'stream_metrics.json').read_text())
            self.assertFalse(failed['passed']); self.assertEqual(len(failed['stream_replays']),2)
            self.assertTrue(all(not x['passed'] for x in failed['stream_replays']))

    def test_reports_exclude_bundle_and_report_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); run=p/'stream'; run.mkdir(); (run/'stream_metrics.json').write_text('{}'); (run/'inference_bundle.pt').write_bytes(b'weights')
            env={**os.environ,'BABEL_STREAM_RUN':str(run),'BABEL_DOWNLOAD_DIR':str(p/'download'),'PYTHON_BIN':'/usr/bin/false'}
            for mode,code in [('export',0),('all',1)]:
                r=subprocess.run(['bash','scripts/babel_stream_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,code,r.stderr)
                with tarfile.open(p/'download/stream_reports.tar.gz') as t:
                    self.assertIn('stream/stream_metrics.json',t.getnames()); self.assertNotIn('stream/inference_bundle.pt',t.getnames())
                    text=t.extractfile('stream/run_status.txt').read().decode()
                    self.assertIn('audit_status='+('partial' if code==0 else 'failed'),text)

    def test_right_aligned_padding_matches_original_full_history_encoder(self):
        mods=list(models()); mods[0]=LargeHistory(SMALL,blocks=16,aggregate_layers=1).eval()
        stream=engine(mods); df=frame(513); encoded=encode_context(df,60,'ema8_32')['x']
        for bar in frame_bars(df.iloc[:512],'rb/60/rb2505',60): result=stream.push(bar)
        self.assertEqual(result['metadata']['completed_blocks'],4)
        x=torch.zeros(16,128,18); x[-4:]=torch.tensor(encoded[:512]).reshape(4,128,18)
        anchors=[float(df.open.iloc[0])]+[float(df.close.iloc[k]) for k in (127,255,383)]
        offsets=torch.zeros(1,16); offsets[0,-4:]=torch.tensor([100*np.log(a/anchors[-1]) for a in anchors])
        valid=torch.zeros(1,16,dtype=torch.bool); valid[:,-4:]=True
        with torch.no_grad():
            local=mods[0].local.encode(x)[:,-1][None]
            reference=mods[0].summarize(local,offsets,valid)
        torch.testing.assert_close(stream.long,reference,atol=2e-5,rtol=2e-5)
        self.assertEqual(result['metadata']['history_anchor'],float(df.close.iloc[383]))
        self.assertEqual(result['metadata']['recent_anchor'],float(df.close.iloc[447]))

    def test_short_numerics_controls_and_effect_limits(self):
        from obson.babel.stream_audit import bounded_difference, short_numerics
        # Sparse near-zero coordinate drift can fail old allclose but satisfy the declared RMS budget.
        ref=np.zeros(512,np.float32); delta=ref.copy(); delta[0]=.000666
        self.assertFalse(comparison(delta,ref)['passed'])
        self.assertTrue(bounded_difference(delta,ref)['passed'])
        self.assertFalse(bounded_difference(ref+.0003,ref)['passed'])  # RMS catches widespread drift.
        delta[0]=.0021; self.assertFalse(bounded_difference(delta,ref)['passed'])
        stream=engine(); x=np.random.default_rng(9).normal(size=(260,18)).astype(np.float32)
        with torch.no_grad():
            h=None
            for row in x: z,h=stream.short_model(torch.tensor(row)[None,None],h)
            state=z[0,-1].numpy(); decoded=stream.recent_decoder.decode_recent(z[:,-1]).numpy()
        result=short_numerics(stream,x,state,state,decoded,decoded)
        self.assertTrue(result['passed'])
        wrong_state=state.copy(); wrong_state[0]+=.001
        self.assertFalse(short_numerics(stream,x,wrong_state,state,decoded,decoded)['checks']['stream_vs_cached_input_step']['passed'])
        wrong_price=decoded.copy(); wrong_price[...,1]+=.0003
        result=short_numerics(stream,x,state,state,wrong_price,decoded)
        self.assertFalse(result['checks']['decoder_price']['passed'])  # .03bp mean exceeds .02bp.
        wrong_price=decoded.copy(); wrong_price[0,0,1]+=.0011
        self.assertFalse(short_numerics(stream,x,state,state,wrong_price,decoded)['checks']['decoder_price']['passed'])
