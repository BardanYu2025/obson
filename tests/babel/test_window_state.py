"""Frozen interface: causal feature replay, exact restore, immutable delivery."""
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
import pandas as pd
import torch

from obson.babel import window_state as ws, window_state_delivery as delivery, window_state_cli as cli
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_babel import frame
from test_bar_alignment import small_model
import test_endpoint_readout as endpoint_fixture


def batch_features(df,period=15):
    return np.column_stack((ws.ae_context.encode_context(df,period,'ema8_32')['x'],ws.aa.activity_features(df,period)[0]))


def engine():
    model,_,stats,local=small_model()
    return ws.WindowState(model.core,stats,local,dict(test=True,supported_periods=[15]),'A/15/contract',15)


def bars(df):return list(ws.frame_bars(df,'A/15/contract',15))
def asof(bar):return ws.streaming.timestamp(bar.datetime)+pd.Timedelta(minutes=bar.period)


class StateTests(unittest.TestCase):
    def test_incremental_matches_batch_with_gaps_availability_and_future_suffix(self):
        df=frame(700);df['datetime']=pd.date_range('2025-01-01',periods=700,freq='15min')
        df.loc[350:,'datetime']+=pd.Timedelta(days=1)
        df.loc[::9,'volume']=0.;df.loc[::11,'close_oi']=0.
        df['open_oi']=df.close_oi-3;df.loc[::3,'open_oi']=np.nan;df.loc[::7,'open_oi']=0.
        df=ws.data.validate_frame(df);features=ws.Features()
        actual=np.stack([features.advance(b)[0] for b in bars(df)])
        np.testing.assert_allclose(actual,batch_features(df),atol=1e-6,rtol=2e-5)
        np.testing.assert_array_equal(batch_features(df)[:550],batch_features(df.iloc[:550]))
        no_oi=ws.data.validate_frame(df.drop(columns=['close_oi','open_oi']))
        f=ws.Features();np.testing.assert_allclose(np.stack([f.advance(b)[0] for b in bars(no_oi)]),batch_features(no_oi),atol=1e-6,rtol=2e-5)

    def test_warmup_final_state_and_own_anchor_crop(self):
        e=engine();df=frame(514);bb=bars(df)
        for b in bb[:511]:e.warm(b,as_of=asof(b))
        self.assertFalse(e.current()['metadata']['ready']);self.assertIsNone(e.current()['embedding'])
        output=e.push(bb[511],as_of=asof(bb[511]));self.assertTrue(output['metadata']['ready'])
        x=batch_features(df)[:512][-128:];norm=((x-e.statistics['x_mean'])/e.statistics['x_scale']).astype(np.float32)
        with torch.inference_mode():z=e.model.encoder(torch.tensor(norm[None]))[:,-1]
        np.testing.assert_allclose(output['embedding'],z[0].numpy(),atol=5e-5,rtol=2e-4)
        self.assertEqual(len(output['history']['values']),127);self.assertEqual(len(output['recent']['values']),16)
        self.assertEqual(output['history']['bar_starts'][-1],str(df.datetime.iloc[510]))
        self.assertEqual(output['recent']['bar_starts'][0],str(df.datetime.iloc[495]))
        self.assertEqual(output['history']['price_anchor'],df.close.iloc[383])
        self.assertEqual(output['recent']['price_anchor'],output['history']['close_prices'][110])
        np.testing.assert_allclose(output['recent']['close_prices'],output['history']['close_prices'][111:127],rtol=2e-6)
        self.assertFalse(any(p.requires_grad for p in e.model.parameters()))

    def test_restore_is_exact_and_invalid_inputs_are_transactional(self):
        e=engine();bb=bars(frame(521))
        for b in bb[:512]:e.warm(b,as_of=asof(b))
        snap=json.loads(json.dumps(e.snapshot()));other=engine();other.restore(snap)
        for b in bb[512:]:self.assertEqual(e.push(b,as_of=asof(b)),other.push(b,as_of=asof(b)))
        before=e.snapshot()
        for b,clock in [(bb[-1],asof(bb[-1])),(replace(bb[-1],key='B/15/other'),asof(bb[-1])),
                        (replace(bb[-1],closed=False),asof(bb[-1])),(bb[-1],bb[-1].datetime)]:
            with self.assertRaises(ValueError):e.push(b,as_of=clock)
            self.assertEqual(before,e.snapshot())
        bad=copy.deepcopy(before);bad['rows'][-1]['time']=bad['rows'][0]['time']
        with self.assertRaises(ValueError):e.restore(bad)
        self.assertEqual(before,e.snapshot())
        with self.assertRaises(ValueError):e.reset_contract('A/60/contract',60)
        self.assertEqual(before,e.snapshot())
        e.reset_contract('B/15/other',15);self.assertFalse(e.current()['metadata']['ready']);self.assertEqual(e.count,0)

    def test_inference_failure_does_not_consume_bar(self):
        e=engine();bb=bars(frame(512))
        for b in bb[:511]:e.warm(b,as_of=asof(b))
        before=e.snapshot()
        with patch.object(e.model.encoder,'forward',side_effect=RuntimeError('GPU failure')):
            with self.assertRaises(RuntimeError):e.push(bb[-1],as_of=asof(bb[-1]))
        self.assertEqual(before,e.snapshot())

    def test_cli_ignores_unclosed_values_and_restores_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);csv=root/'raw.csv';df=frame(514);df.loc[513,'close']=np.nan;df.to_csv(csv,index=False)
            out=root/'states.jsonl';snapshot=root/'snapshots/state.json';e=engine()
            with patch.object(ws,'load_bundle',return_value=e):
                cli.run(root/'bundle',42,e.key,15,csv,asof(bars(df.iloc[:513])[-1]),out,last_only=True,snapshot_out=snapshot)
            rows=out.read_text().splitlines();self.assertEqual(len(rows),1);self.assertEqual(json.loads(rows[0])['metadata']['observed_bars'],513)
            with patch.object(ws,'load_bundle',return_value=engine()):
                with self.assertRaisesRegex(ValueError,'later than as_of'):
                    cli.run(root/'bundle',42,e.key,15,csv,df.datetime.iloc[0],root/'early',snapshot_in=snapshot)
                with self.assertRaisesRegex(ValueError,'append-only'):
                    cli.run(root/'bundle',42,e.key,15,csv,asof(bars(df.iloc[:513])[-1]),root/'overlap',snapshot_in=snapshot)
            df=frame(514);df.iloc[513:].to_csv(root/'new.csv',index=False)
            with patch.object(ws,'load_bundle',return_value=engine()):
                cli.run(root/'bundle',42,e.key,15,root/'new.csv',asof(bars(df)[-1]),root/'next',snapshot_in=snapshot)
            self.assertEqual(json.loads((root/'next').read_text())['metadata']['observed_bars'],514)
            with self.assertRaises(FileExistsError):cli.run(root/'bundle',42,e.key,15,csv,'2020',out)


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        endpoint_fixture.EndpointPipelineTests.setUpClass();cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.source=cls.root/'endpoint';ws.ea.run(endpoint_fixture.EndpointPipelineTests.source,cls.source,'cpu',batch=2)
        cls.meta=dict(schema=delivery.SCHEMA,source=str(cls.source),identity=delivery.source_identity(cls.source))
        cls.out=cls.root/'delivery';cls.out.mkdir();cls.bundle=delivery.build_bundle(cls.meta,cls.out)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup();endpoint_fixture.EndpointPipelineTests.tearDownClass()

    def certify(self):atomic_json(dict(status='passed',index_sha256=sha256(self.bundle/'index.json')),self.bundle/'validation.json')

    def test_portable_replays_without_training_or_fitting_and_tamper_rejected(self):
        before={str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()}
        with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer')),patch.object(ws.ea.bb.ab,'fit_pca',side_effect=AssertionError('No fitting')):
            report=delivery.cached_replay(self.meta,self.bundle,self.out,'cpu')
        self.assertEqual(len(report),4);self.assertTrue(all(v['global_replay'] and v['recent_replay'] for v in report.values()))
        self.assertEqual(before,{str(p):sha256(p) for p in self.source.rglob('*') if p.is_file()})
        (self.bundle/'validation.json').unlink(missing_ok=True)
        with self.assertRaises(FileNotFoundError):ws.load_bundle(self.bundle,42,'A/15/contract',15)
        self.certify();e=ws.load_bundle(self.bundle,42,'A/15/contract',15);self.assertEqual(e.identity['seed'],42)
        path=self.bundle/'baseline_s42.pt';original=path.read_bytes()
        try:
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):ws.load_bundle(self.bundle,42,'A/15/contract',15)
        finally:path.write_bytes(original)

    def test_raw_rolling_replay_and_snapshot_on_synthetic_contract(self):
        # Full raw pipeline with an independent batch-feature cache. No model updates.
        df=frame(540);path=self.root/'raw.csv';df.to_csv(path,index=False);end=530
        row=dict(key='A/15/contract',period=15,row=end,end=str(df.datetime.iloc[end]))
        plan=[dict(path=str(path),sha256=sha256(path),inventory=row,index=0,split='test')]
        e=ws.load_bundle(self.bundle,42,row['key'],15,_audit=True)
        x=batch_features(df)[end-127:end+1];norm=((x-e.statistics['x_mean'])/e.statistics['x_scale']).astype(np.float32)
        with patch.object(ws.ea.bb,'load_data',return_value=dict(x=norm[None])),patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer')):
            result=delivery.raw_replay(self.meta,self.bundle,self.out,plan,'cpu')
        self.assertEqual(len(result),2);self.assertTrue(all(v['snapshot_exact'] and v['raw_cache_replay'] for v in result))
        self.assertTrue(all(v['rolling_states']==9 for v in result))

    def test_failure_export_and_separate_weight_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download';atomic_json(dict(status='partial'),run/'metrics.json')
            (run/'data.pt').write_bytes(b'weights');fake=root/'fail';fake.write_text('#!/bin/sh\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,BABEL_STATE_RUN=str(run),BABEL_STATE_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(fake))
            cmd=['bash',str(Path(__file__).resolve().parents[2]/'scripts/babel_window_state_autodl.sh')]
            for mode,code in [('all',7),('export',0)]:
                result=subprocess.run(cmd+[mode],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,code,result.stderr)
                with tarfile.open(download/'run_reports.tar.gz') as f:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in f.getnames()))
                    self.assertIn('run_status=failed',f.extractfile('run/run_status.txt').read().decode())
            bundle=run/'bundle';bundle.mkdir();atomic_json(dict(status='passed'),bundle/'validation.json');(bundle/'weights.pt').write_bytes(b'weights')
            atomic_json(dict(status='complete'),run/'completion.json');(run/'run_status.txt').write_text('run_status=complete\n')
            result=subprocess.run(cmd+['export'],env=env,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
            with tarfile.open(download/'run_bundle.tar.gz') as f:self.assertIn('bundle/weights.pt',f.getnames())

    def test_orchestration_certifies_only_after_all_checks_and_rejects_changed_source(self):
        # Raw consistency has an independent real test above; mock only that
        # stage here because upstream fixture caches contain synthetic features.
        out=self.root/'orchestrated';raw=self.root/'raw_root';raw.mkdir()
        with patch.object(delivery,'raw_plan',return_value=[]),patch.object(delivery,'raw_replay',return_value=[]):
            delivery.run(self.source,out,raw,'cpu')
            done=ws.read_json(out/'completion.json');self.assertEqual(done['encoder_updates'],0)
            self.assertEqual(ws.load_bundle(out/'bundle',42,'A/15/contract',15).identity['seed'],42)
            with patch.object(delivery,'cached_replay',side_effect=ValueError('failed replay')):
                with self.assertRaisesRegex(ValueError,'failed replay'):delivery.run(self.source,out,raw,'cpu')
            self.assertFalse((out/'completion.json').exists());self.assertFalse((out/'bundle/validation.json').exists())
            with self.assertRaises(FileNotFoundError):ws.load_bundle(out/'bundle',42,'A/15/contract',15)
            delivery.run(self.source,out,raw,'cpu')
            with patch.object(delivery,'raw_plan',return_value=[dict(changed=True)]):
                with self.assertRaisesRegex(ValueError,'Pinned raw plan'):delivery.run(self.source,out,raw,'cpu')
        with self.assertRaisesRegex(ValueError,'Separate'):delivery.run(self.source,self.source/'bad',raw,'cpu')
        self.assertFalse((self.source/'bad').exists())


if __name__=='__main__':unittest.main()
