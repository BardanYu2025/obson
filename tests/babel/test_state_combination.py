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
from torch import nn

from obson.babel import state_combination as sc, state_holdout as sh, state_transfer as st
from obson.babel import activity_ablation as aa, detail_alignment as da
from obson.babel.ae_extend import atomic_json, atomic_save
from obson.babel.dual_state import sha256
from test_state_transfer import no_update, bank


def cache_fixture(root):
    out=root/'combination';cache=out/'cache';cache.mkdir(parents=True)
    meta=sc.make_manifest(root/'transfer',root/'cross',dict(seeds=[42,43]),{},2,8,2);meta['hidden']=8
    atomic_json(meta,out/'manifest.json');rng=np.random.default_rng(26);files={}
    for split,n in zip(sc.SPLITS,(24,12,12,10)):
        features={k:rng.normal(size=(n,3 if k=='current' else 5 if k=='statistics' else 4)).astype(np.float32)
            for k in ('current','statistics','control_s42','control_s43','aux020_s42','aux020_s43')}
        for name,parts in meta['representations'].items():
            f=cache/f'{split}_{name}.npy';np.save(f,sc.combine(parts,features));files[f.name]=sha256(f)
        for task,names in meta['tasks'].items():
            y=rng.normal(size=(n,len(names))).astype(np.float32);mask=np.ones_like(y,bool);mask[0,1]=False
            for suffix,value in [('y',y),('mask',mask)]:
                f=cache/f'{split}_{task}_{suffix}.npy';np.save(f,value);files[f.name]=sha256(f)
        if split in ('test','cross_research'):
            f=cache/f'{split}_inventory.json';atomic_json([dict(symbol='A' if i%2 else 'B',period=15,month='2026-01',week=f'w{i}',end=f'2026-01-{i+1:02d}') for i in range(n)],f);files[f.name]=sha256(f)
    atomic_json(dict(manifest=meta,files=files),cache/'index.json');atomic_json(dict(price_routing={},prior_aux_price_screen=[]),out/'preparation_audit.json')
    return meta,out


class StateCombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_matrix_dimensions_order_and_invalid_alignment(self):
        reps=sc.representations([42,43]);self.assertEqual(len(reps),18)
        base={k:np.full((3,61 if k=='statistics' else 28 if k=='current' else 512),i,dtype=np.float32) for i,k in enumerate(('current','statistics','control_s42','control_s43','aux020_s42','aux020_s43'))}
        for seed in (42,43):
            for suffix in ('','_plus_statistics'):
                a=sc.combine(reps[f'dual_s{seed}{suffix}'],base);b=sc.combine(reps[f'price_pair_s{seed}{suffix}'],base)
                self.assertEqual(a.shape,b.shape);np.testing.assert_array_equal(a[:,:512],base[f'control_s{seed}']);np.testing.assert_array_equal(b[:,:512],a[:,:512])
        with self.assertRaises(ValueError):sc.representations([42])
        with self.assertRaisesRegex(ValueError,'implementation changed'):sc.verify_code(dict(code_sha256={}))
        with self.assertRaises(ValueError):sc.combine(['a','b'],{'a':np.zeros((2,4)),'b':np.zeros((3,4))})
        with self.assertRaises(ValueError):sc.combine(['a'],{'a':np.full((2,4),np.nan)})

    def test_grouped_ridge_matches_individual_masked_selection(self):
        rng=np.random.default_rng(77);xs=[rng.normal(size=(n,7)) for n in (40,22,13)]
        ys=[rng.normal(size=(len(x),15)) for x in xs];masks=[np.ones_like(y,bool) for y in ys]
        for y in ys:y[:,14]=1
        for m in masks:m[::3,2]=False;m[::4,5]=False
        old,expected=aa.fit_activity_probe(xs,ys,masks);model,rows=sc.fit_ridge(xs[:2],ys[:2],masks[:2])
        for a,b in zip(rows,old['targets']):
            self.assertEqual(a.get('alpha'),b.get('alpha'))
            if 'alpha' in a:self.assertAlmostEqual(a['validation_normalized_mse'],b['validation_normalized_mse'],places=10)
        np.testing.assert_allclose(sh.ridge_errors(model,xs[2],ys[2],masks[2]),expected,atol=1e-11,equal_nan=True)
        self.assertIsNone(model[14])
        with self.assertRaisesRegex(ValueError,'only'):sc.fit_ridge(xs,ys,masks)

    def test_protected_price_route_does_not_use_second_state(self):
        head=da.ReconstructionFusion('short',width=8,decoder_width=8)
        rng=np.random.default_rng(1);c=rng.normal(size=(7,8)).astype(np.float32);a=rng.normal(size=(7,8)).astype(np.float32)
        before={k:v.clone() for k,v in head.state_dict().items()};r=sc.price_route_audit(head,c,a,3,'cpu')
        self.assertEqual(r['max_abs_coordinate_difference'],0);self.assertTrue(r['aux_perturbation_invariant'])
        for k,v in head.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        with self.assertRaises(ValueError):sc.decode_price(head,torch.zeros(2,15),8)

    def test_selection_never_loads_evaluation_arrays_and_freezes_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out=cache_fixture(Path(tmp));job=meta['experiments'][-1];original_load=sc.load_arrays
            def guarded(folder,name,task,split):
                if split in ('test','cross_research'):self.assertTrue((folder/job['name']/'selection_lock.json').exists())
                return original_load(folder,name,task,split)
            original_trial=st.mlp_trial
            def trial(xs,ys,masks,*args,**kwargs):
                self.assertEqual(len(xs),2);self.assertEqual(len(ys),2);self.assertEqual(len(masks),2)
                return original_trial(xs,ys,masks,*args,**kwargs)
            with patch.object(st,'train_probe_epoch',side_effect=no_update),patch.object(st,'mlp_trial',side_effect=trial),patch.object(sc,'load_arrays',side_effect=guarded),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                sc.worker(out,job['name'],'cpu')
            row=json.loads((out/job['name']/'metrics.json').read_text());self.assertEqual(len(row['target_names']),15)
            self.assertFalse(row['selection']['mlp_s1701']['trained_selection'])
            with patch.object(st,'mlp_trial',side_effect=AssertionError('Completed worker trained again')):sc.worker(out,job['name'],'cpu')
            # A crash after selection but before completion must preserve the lock.
            lock=(out/job['name']/'selection_lock.json').read_bytes()
            (out/job['name']/'completion.json').unlink()
            with patch.object(st,'train_probe_epoch',side_effect=AssertionError('Completed trials updated')):sc.worker(out,job['name'],'cpu')
            self.assertEqual(lock,(out/job['name']/'selection_lock.json').read_bytes())
            (out/job['name']/'p1701_wd0.001/best.pt').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'fingerprint'):sc.worker(out,job['name'],'cpu')

    def test_complete_matrix_reports_both_research_sets_and_all_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out=cache_fixture(Path(tmp));before=sha256(out/'cache/index.json')
            with patch.object(st,'train_probe_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                for job in meta['experiments']:sc.worker(out,job['name'],'cpu')
            report=sc.evaluate(out);self.assertFalse(report['independent_holdout']);self.assertFalse(report['automatic_promotion'])
            for split in ('test','cross_research'):
                for task in sc.TASKS:
                    r=report['datasets'][split][task];self.assertEqual(len(r['variants']),18);self.assertEqual(len(r['paired']),16)
                    self.assertFalse(any(v['all_readouts_supported'] for v in r['exploratory_signals']))
                    self.assertIn('dual_s43_plus_statistics_minus_price_pair_s43_plus_statistics',r['paired'])
            self.assertEqual(before,sha256(out/'cache/index.json'))
            self.assertNotIn('NaN',(out/'combination_metrics.json').read_text())

    def test_prepare_replays_pinned_encoders_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);transfer=root/'transfer';cross=root/'cross';price=root/'price';encoder=root/'encoder';out=root/'out'
            for p in (transfer/'cache',cross/'cache',price/'cache',encoder/'cache',out):p.mkdir(parents=True)
            tm=dict(seeds=[42,43],parent_run=str(price),encoder_run=str(encoder));atomic_json(tm,transfer/'manifest.json')
            jobs=[dict(name=f'{mode}_s{s}',seed=s,activity='features',aux_weight=.2 if mode=='aux020' else 0.) for s in (42,43) for mode in ('control','aux020')]
            em=dict(config=dict(width=8,decoder_width=8),epochs=100);pm=dict(experiments=jobs,config=em['config'])
            atomic_json(em,encoder/'manifest.json');atomic_json(pm,price/'manifest.json');atomic_json(dict(scales=[1]*3),price/'cache/replay.json')
            rng=np.random.default_rng(2);inv=[dict(symbol='X',period=15,week=f'w{i}',end=f'2026-01-{i+1:02d}') for i in range(4)]
            atomic_json(inv,cross/'cache/test_inventory.json');atomic_json(inv,transfer/'cache/test_inventory.json')
            for split,n in [('train',8),('val',6),('test',4)]:
                for k,d in [('current',28),('statistics',61),('y',12)]+[(j['name'],8) for j in jobs]:np.save(transfer/f'cache/{split}_{k}.npy',rng.normal(size=(n,d)).astype(np.float32))
                np.save(transfer/f'cache/{split}_mask.npy',np.ones((n,12),bool));np.save(encoder/f'cache/{split}_activity.npy',rng.normal(size=(n,20)).astype(np.float32));np.save(encoder/f'cache/{split}_activity_mask.npy',np.ones((n,20),bool))
            for k,d in [('current',28),('statistics',61),('y',12),('activity',20)]:np.save(cross/f'cache/test_{k}.npy',rng.normal(size=(4,d)).astype(np.float32))
            for k,d in [('mask',12),('activity_mask',20)]:np.save(cross/f'cache/test_{k}.npy',np.ones((4,d),bool))
            np.save(cross/'cache/test_x.npy',bank(512,7));atomic_json([dict(offset=0,length=256,endpoints=[[127,0],[255,1]]),dict(offset=256,length=256,endpoints=[[127,2],[255,3]])],cross/'cache/test_sequences.json')
            y=np.zeros((4,64,7),np.float32);mask=np.ones_like(y,bool);np.save(cross/'cache/test_price_y.npy',y);np.save(cross/'cache/test_price_mask.npy',mask)
            old=dict(price={},evidence_screen={})
            for job in jobs:
                model=da.DetailModel(**em['config'],input_dim=28);model.activity_head=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,20));model.eval()
                (encoder/job['name']).mkdir();(price/job['name']).mkdir()
                atomic_save(dict(metadata=dict(manifest=em,job=job),epoch=100,model=model.state_dict()),encoder/job['name']/'last.pt')
                atomic_save(dict(metadata=dict(manifest=pm,job=job),model=model.head.state_dict()),price/job['name']/'best.pt')
                with torch.no_grad():score,_,_=da.run_epoch(model,dict(z=torch.zeros(4,8),y=torch.tensor(y),mask=torch.tensor(mask)),aa.ActivityStreams(cross/'cache','test',job),True,True,[1]*3,2,collect=True)
                old['price'][job['name']]=dict(objectives=score)
            atomic_json(old,cross/'holdout_metrics.json')
            meta=sc.make_manifest(transfer,cross,tm,{},2,8,2)
            sc.prepare(meta,out,'cpu');info=json.loads((out/'preparation_audit.json').read_text())
            self.assertEqual(len(info['source_replay']),4);self.assertEqual(info['price_routing']['test']['control_s42']['max_abs_coordinate_difference'],0)
            np.testing.assert_array_equal(np.load(out/'cache/test_dual_s42.npy')[:,:8],np.load(transfer/'cache/test_control_s42.npy'))
            with patch.object(sc,'replay_cross',side_effect=AssertionError('Cache replayed again')):sc.prepare(meta,out,'cpu')
            (out/'cache/test_dual_s42.npy').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'fingerprint'):sc.prepare(meta,out,'cpu')

    def test_source_identity_rejects_wrong_ancestry_and_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);transfer=root/'transfer';cross=root/'cross';(cross/'cache').mkdir(parents=True);(cross/'frozen').mkdir()
            tm=dict(seeds=[42,43]);identity={'original':'fixed'};hm=dict(schema=sh.SCHEMA,transfer=str(transfer),source_identity=identity)
            atomic_json(hm,cross/'manifest.json');atomic_json(dict(status='complete'),cross/'completion.json')
            np.save(cross/'cache/x.npy',np.zeros((2,3)));atomic_json(dict(manifest=hm,files={'x.npy':sha256(cross/'cache/x.npy')}),cross/'cache/index.json')
            for name in ('lineage_audit.json','raw_audit.json','holdout_metrics.json','frozen/index.json'):atomic_json({},cross/name)
            atomic_json([],cross/'cache/test_inventory.json');lock=dict(manifest=hm)
            for f,k in [('lineage_audit.json','lineage_sha256'),('raw_audit.json','raw_audit_sha256'),('frozen/index.json','frozen_index_sha256')]:lock[k]=sha256(cross/f)
            atomic_json(lock,cross/'evaluation_lock.json')
            with patch.object(sh,'sources',side_effect=lambda p:(tm,dict(identity))):
                _,first=sc.source_identity(transfer,cross);self.assertIn(str((cross/'cache/index.json').resolve()),first)
                changed=dict(hm,transfer=str(root/'another'));atomic_json(changed,cross/'manifest.json')
                with self.assertRaisesRegex(ValueError,'sources differ'):sc.source_identity(transfer,cross)
                atomic_json(hm,cross/'manifest.json');(cross/'cache/x.npy').write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError,'fingerprint'):sc.source_identity(transfer,cross)

    def test_nested_worker_fingerprints_reject_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);worker=root/'worker';(worker/'trial').mkdir(parents=True);file=worker/'trial/checkpoint';file.write_text('fixed')
            sc.verify_worker_files(worker,{'trial/checkpoint':sha256(file)})
            external=root/'outside';external.write_text('fixed');(worker/'link').symlink_to(external)
            for name in ('../outside','link',str(external)):
                with self.assertRaisesRegex(ValueError,'fingerprint'):sc.verify_worker_files(worker,{name:sha256(external)})

    def test_export_success_and_failure_never_includes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();atomic_json({},run/'combination_metrics.json');(run/'best.pt').write_bytes(b'omit')
            env=dict(os.environ,BABEL_COMBINATION_RUN=str(run),BABEL_COMBINATION_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,expected in [('export',0),('all',1),('export',0)]:
                result=subprocess.run(['bash','scripts/babel_state_combination_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,expected,result.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as tar:
                    self.assertIn('run/combination_metrics.json',tar.getnames());self.assertFalse(any(n.endswith('.pt') for n in tar.getnames()))
                    if (run/'run_status.txt').exists():self.assertIn('run_status=failed',tar.extractfile('run/run_status.txt').read().decode())
