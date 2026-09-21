import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from obson.babel import state_holdout as sh, holdout_audit as ha
from obson.babel import activity_ablation as aa, activity_alignment as al, detail_alignment as da
from obson.babel.ae_extend import atomic_json, atomic_save
from obson.babel.data import validate_frame
from obson.babel.dual_state import sha256
from test_state_transfer import bank


def frame(n=400,base=100.):
    rng=np.random.default_rng(41);c=base+np.cumsum(rng.normal(0,.04,n));o=np.r_[c[0],c[:-1]]
    return pd.DataFrame(dict(datetime=pd.date_range('2026-01-01 09:00',periods=n,freq='15min'),open=o,
        close=c,high=np.maximum(o,c)+.01,low=np.minimum(o,c)-.01,volume=np.arange(n)+100.,
        close_oi=np.arange(n)+1000.,open_oi=np.arange(n)+999.))


def raw_fixture(root):
    raw=root/'raw';(raw/'old').mkdir(parents=True);(raw/'new').mkdir()
    a=raw/'old/EX.old2601_15m.csv';frame().to_csv(a,index=False)
    b=raw/'new/EX.new2601_15m.csv';frame(base=200).to_csv(b,index=False)
    df=validate_frame(pd.read_csv(a));r=dict(key='old/15/EX.old2601',start=str(df.datetime.iloc[0]),end=str(df.datetime.iloc[-1]),
        sha256=hashlib.sha256(pd.util.hash_pandas_object(df,index=False).values.tobytes()).hexdigest())
    return raw,a,b,dict(source_records=[r],verified=True)


class StateHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_lineage_resolves_path_hash_and_unknown_ancestor_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';short=root/'short';head=root/'head'
            for p in (base,short,head):p.mkdir()
            atomic_json(dict(manifest=dict(sources=[dict(key='x/15/C',sha256='a'*64,start='2020',end='2021')])),base/'manifest.json')
            atomic_json(dict(source_manifest=sha256(base/'manifest.json')),short/'manifest.json')
            atomic_json(dict(short_run=str(short),parent_run=str(base)),head/'manifest.json')
            a=ha.scan_lineage(root,head,root/'out');self.assertTrue(a['verified']);self.assertEqual(len(a['ancestors']),3)
            atomic_json(dict(source_manifest='b'*64),short/'manifest.json')
            a=ha.scan_lineage(root,head,root/'out');self.assertFalse(a['verified']);self.assertIn('unresolved_manifest_hash',[r['reason'] for r in a['issues']])

    def test_raw_quality_overlap_and_source_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw,old,new,lineage=raw_fixture(Path(tmp))
            a=ha.audit_raw(raw,lineage,('old','new'));self.assertEqual(a['eligible_symbols'],['new'])
            # Re-exported alias: same timestamp/OHLCV, changed OI and formatting.
            df=pd.read_csv(old);df.close_oi+=1;df.to_csv(new,index=False,float_format='%.12g')
            # Exact decimal roundtrip fixture, no formatting truncation for equality hash.
            df.to_csv(new,index=False)
            a=ha.audit_raw(raw,lineage,('new',));self.assertFalse(a['eligible_symbols'])
            frame(base=200).to_csv(new,index=False);df=pd.read_csv(old);df.volume+=3;df.to_csv(old,index=False)
            a=ha.audit_raw(raw,lineage,('new',));self.assertFalse(a['eligible_symbols']);self.assertTrue(a['missing_source_inventory'])

    def test_prior_holdout_and_cross_candidate_copies_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,old,new,lineage=raw_fixture(root)
            lineage['previous_holdout_symbols']=['new']
            self.assertFalse(ha.audit_raw(raw,lineage,('new',))['eligible_symbols'])
            lineage['previous_holdout_symbols']=[];(raw/'alias').mkdir();(raw/'alias/EX.alias2601_15m.csv').write_bytes(new.read_bytes())
            a=ha.audit_raw(raw,lineage,('new','alias'));self.assertFalse(a['eligible_symbols'])
            self.assertTrue(all('content_overlap_with_another_candidate_symbol' in a['symbols'][s]['reasons'] for s in ('new','alias')))

    def test_bad_file_excludes_whole_symbol_and_missing_inventory_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw,old,new,lineage=raw_fixture(Path(tmp));df=pd.read_csv(new);df.loc[3,'close']=-1;df.to_csv(new,index=False)
            a=ha.audit_raw(raw,lineage,('new',));self.assertIn('invalid_candidate_file',a['symbols']['new']['reasons'])
            frame(base=200).to_csv(new,index=False);old.unlink()
            a=ha.audit_raw(raw,lineage,('new',));self.assertIn('provenance_unresolved',a['symbols']['new']['reasons'])

    def test_ridge_restores_original_fit_and_never_fits_holdout(self):
        rng=np.random.default_rng(7);xs=[rng.normal(size=(n,5)) for n in (40,20,16)]
        ys=[x[:,:3]+rng.normal(0,.1,(len(x),3)) for x in xs];mask=[np.ones_like(y,bool) for y in ys]
        mask[0][2,1]=False;ys[0][2,1]=1e9
        report,expected=aa.fit_activity_probe(xs,ys,mask)
        fit=sh.locked_ridge(xs[0],ys[0],mask[0],report['targets'])
        sh.check_ridge(fit,xs[2],ys[2],mask[2],report['targets'])
        np.testing.assert_allclose(sh.ridge_errors(fit,xs[2],ys[2],mask[2]),expected,atol=1e-12)
        saved=json.dumps(fit);sh.ridge_errors(fit,xs[2]*100,ys[2]*30,mask[2]);self.assertEqual(saved,json.dumps(fit))
        with self.assertRaisesRegex(ValueError,'replay failed'):sh.check_ridge(fit,xs[2]*2,ys[2],mask[2],report['targets'])

    def test_common_support_and_small_group_not_positive_evidence(self):
        a=np.array([[1.,2.],[np.nan,0.],[3.,4.]]);b=np.array([[2.,3.],[1.,2.],[np.nan,2.]])
        inv=[dict(symbol='x',period=15,month='2026-01',week=f'w{i}') for i in range(3)]
        r=sh.contrast(a,b,inv,[0,1])['all'];self.assertEqual(r['support'],1);self.assertFalse(r['supported']);self.assertIsNone(r['high'])
        self.assertEqual(r['delta_mse'],-1.)
        self.assertIsNone(sh.safe({'x':float('nan')})['x'])

    def test_packed_replay_causal_and_contract_batch_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x=bank(768,9);specs=[dict(offset=0,length=384,endpoints=[[127,0],[255,1]]),
                dict(offset=384,length=384,endpoints=[[127,2],[255,3]])]
            np.save(root/'test_x.npy',x);atomic_json(specs,root/'test_sequences.json')
            model=da.DetailModel(width=8,decoder_width=8,input_dim=28).eval();data=dict(z=torch.zeros(4,8),y=torch.zeros(4,64,7),mask=torch.ones(4,64,7,dtype=torch.bool))
            with torch.no_grad():
                a=da.run_epoch(model,data,aa.ActivityStreams(root,'test',dict(activity='features')),True,True,[1]*3,1,collect=True)
                b=da.run_epoch(model,data,aa.ActivityStreams(root,'test',dict(activity='features')),True,True,[1]*3,2,collect=True)
                np.testing.assert_allclose(a[1],b[1],atol=2e-6)
                x[256:384]+=10;x[640:]+=10;np.save(root/'test_x.npy',x)
                c=da.run_epoch(model,data,aa.ActivityStreams(root,'test',dict(activity='features')),True,True,[1]*3,2,collect=True)
                np.testing.assert_allclose(b[1],c[1],atol=2e-6)

    def test_raw_preparation_and_all_heads_synthetic_no_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';(raw/'new').mkdir(parents=True);frame(800,200).to_csv(raw/'new/EX.new2601_15m.csv',index=False)
            out=root/'out';out.mkdir();transfer=root/'transfer';transfer.mkdir();price=root/'price';price.mkdir();encoder=root/'encoder';encoder.mkdir();frozen=out/'frozen';frozen.mkdir()
            meta=dict(root=str(raw),boundaries=dict(train_until='2025-01-01',val_until='2025-12-31',test_until='2026-02-01'),cap=10,streams=2,
                transfer=str(transfer),encoder_names=['control_s42','aux020_s42'])
            jobs=[dict(name=n,activity='features',seed=42,aux_weight=w) for n,w in zip(meta['encoder_names'],(0,.2))]
            em=dict(config=dict(width=8,decoder_width=8),epochs=100);atomic_json(em,encoder/'manifest.json')
            pm=dict(experiments=jobs);atomic_json(pm,price/'manifest.json');(price/'cache').mkdir();atomic_json(dict(scales=[1]*3),price/'cache/replay.json');atomic_json(dict(stage_screen=[]),price/'readapt_metrics.json')
            names=['current','statistics']+meta['encoder_names']+[n+'_plus_statistics' for n in meta['encoder_names']]
            tm=dict(parent_run=str(price),encoder_run=str(encoder),seeds=[42],hidden=8,experiments=[dict(name=n) for n in names]);atomic_json(tm,transfer/'manifest.json');(transfer/'cache').mkdir()
            atomic_json(dict(mean=[0]*12,scale=[1]*12,active=[True]*12),transfer/'cache/target_statistics.json')
            sh.prepare_holdout(meta,out,dict(eligible_symbols=['new'],raw_hashes={}))
            for job in jobs:
                model=da.DetailModel(**em['config'],input_dim=28);model.activity_head=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,20))
                (encoder/job['name']).mkdir();(price/job['name']).mkdir()
                atomic_save(dict(metadata=dict(manifest=em,job=job),epoch=100,model=model.state_dict()),encoder/job['name']/'last.pt')
                atomic_save(dict(metadata=dict(manifest=pm,job=job),epoch=1,model=model.head.state_dict()),price/job['name']/'best.pt')
            def ridge(dim,tasks):return [dict(mean=[0]*dim,scale=[1]*dim,ym=0,ys=1,weight=[0]*(dim+1),alpha=1) for _ in range(tasks)]
            for n in names:
                dim=28 if n=='current' else 61 if n=='statistics' else 69 if n.endswith('_plus_statistics') else 8
                atomic_json(ridge(dim,12),frozen/f'{n}_ridge.json')
                for kind in sh.KINDS[1:]:
                    model=nn.Sequential(nn.Linear(dim,8),nn.GELU(),nn.Linear(8,12))
                    atomic_save(dict(model=model.state_dict(),normalizers=dict(x_mean=[0]*dim,x_scale=[1]*dim,y_mean=[0]*12,y_scale=[1]*12)),frozen/f'{n}_{kind}.pt')
            for n in meta['encoder_names']+['current_input']:atomic_json(ridge(10 if n=='current_input' else 8,20),frozen/f'{n}_activity_ridge.json')
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No training')):sh.evaluate(meta,out,'cpu')
            report=ha.read_json(out/'holdout_metrics.json');self.assertEqual(len(report['transfer']),6);self.assertEqual(len(report['price']),2)
            self.assertFalse(report['automatic_promotion']);self.assertEqual(report['encoder_updates'],0)
            # Serialized JSON is finite; unavailable scores must be null.
            self.assertNotIn('NaN',(out/'holdout_metrics.json').read_text())

    def test_freeze_readouts_replays_existing_models_without_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);transfer=root/'transfer';price=root/'price';encoder=root/'encoder';out=root/'out'
            for p in (transfer/'cache',price/'cache',encoder/'cache',encoder/'last_diagnostic',out):p.mkdir(parents=True)
            names=['current','statistics','control_s42','aux020_s42','control_s42_plus_statistics','aux020_s42_plus_statistics']
            tm=dict(parent_run=str(price),encoder_run=str(encoder),hidden=8,experiments=[dict(name=n) for n in names])
            atomic_json(tm,transfer/'manifest.json');meta=dict(transfer=str(transfer),encoder_names=['control_s42','aux020_s42'])
            rng=np.random.default_rng(28);ys=[rng.normal(size=(n,12)).astype(np.float32) for n in (24,12,10)]
            masks=[np.ones_like(y,bool) for y in ys];ays=[rng.normal(size=(len(y),20)).astype(np.float32) for y in ys]
            ams=[np.ones_like(y,bool) for y in ays];report=dict(variants={});prior=dict(variants={})
            for split,y,m,ay,am in zip(da.SPLITS,ys,masks,ays,ams):
                np.save(transfer/f'cache/{split}_y.npy',y);np.save(transfer/f'cache/{split}_mask.npy',m)
                np.save(encoder/f'cache/{split}_activity.npy',ay);np.save(encoder/f'cache/{split}_activity_mask.npy',am)
            for name in names:
                dim=28 if name=='current' else 61 if name=='statistics' else 69 if name.endswith('_plus_statistics') else 8
                xs=[rng.normal(size=(len(y),dim)).astype(np.float32) for y in ys];folder=transfer/name;folder.mkdir()
                for split,x in zip(da.SPLITS,xs):np.save(transfer/f'cache/{split}_{name}.npy',x)
                value,_=aa.fit_activity_probe(xs,ys,masks);report['variants'][name]=dict(linear=value);errors={}
                for kind,seed in zip(sh.KINDS[1:],(1701,1702)):
                    r=dict(selected_epoch=1,probe_seed=seed,decay=.01);report['variants'][name][kind]=r
                    model=nn.Sequential(nn.Linear(dim,8),nn.GELU(),nn.Linear(8,12));ck=dict(report=r,
                        metadata=dict(manifest=tm,name=name,probe_seed=seed,decay=.01),model=model.state_dict(),
                        normalizers=dict(x_mean=[0]*dim,x_scale=[1]*dim,y_mean=[0]*12,y_scale=[1]*12))
                    path=folder/f'p{seed}_wd0.01';path.mkdir();atomic_save(ck,path/'best.pt')
                    errors[kind]=sh.mlp_errors(ck,xs[2],ys[2],masks[2],8,'cpu').tolist()
                atomic_json(dict(errors=errors),folder/'per_window_errors.json')
                if name in meta['encoder_names']:
                    for split,x in zip(da.SPLITS,xs):np.save(price/f'cache/{name}_{split}_z.npy',x)
                    value,_=aa.fit_activity_probe(xs,ays,ams);prior['variants'][name]=dict(linear_activity_readout=value)
            xs=[rng.normal(size=(len(y),10)) for y in ys]
            for split,x in zip(da.SPLITS,xs):np.save(encoder/f'cache/{split}_current.npy',x)
            value,_=aa.fit_activity_probe(xs,ays,ams);prior['current_input_control']=dict(linear=value)
            atomic_json(prior,encoder/'last_diagnostic/alignment_metrics.json');atomic_json(report,transfer/'transfer_metrics.json')
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No training')):sh.freeze_readouts(meta,out,'cpu')
            index=ha.read_json(out/'frozen/index.json');self.assertEqual(len(index['files']),21)
            with patch.object(sh,'locked_ridge',side_effect=AssertionError('Do not refit completed bundle')):sh.freeze_readouts(meta,out,'cpu')
            (out/'frozen/current_ridge.json').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'fingerprint'):sh.freeze_readouts(meta,out,'cpu')

    def test_export_blocked_status_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();atomic_json(dict(status='blocked'),run/'audit_status.json');(run/'best.pt').write_text('exclude')
            env=dict(os.environ,BABEL_HOLDOUT_RUN=str(run),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN=str(Path('.venv/bin/python').resolve()))
            result=subprocess.run(['bash','scripts/babel_state_holdout_autodl.sh','export'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with tarfile.open(root/'download/run_reports.tar.gz') as tar:
                self.assertFalse(any(n.endswith('.pt') for n in tar.getnames()))
                self.assertIn('run_status=blocked',tar.extractfile('run/run_status.txt').read().decode())
            (run/'run_status.txt').write_text('command=all\ncommand_exit_code=1\nrun_status=failed\n')
            result=subprocess.run(['bash','scripts/babel_state_holdout_autodl.sh','export'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with tarfile.open(root/'download/run_reports.tar.gz') as tar:
                self.assertIn('run_status=failed',tar.extractfile('run/run_status.txt').read().decode())
