"""Synthetic linear arithmetic and lifecycle checks; no neural optimizer updates."""
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
from obson.babel import consistency_readout as r
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,root):
        out=root/'run';reader=root/'reader';source=root/'source'
        for d in (out,reader): (d/'cache').mkdir(parents=True)
        source.mkdir();rng=np.random.default_rng(42);coef=rng.normal(size=(4,13));rows={};banks={}
        for split in r.old.SPLITS:
            z=rng.normal(size=(72,4));y=z@coef+.003*rng.normal(size=(72,13));mask=np.ones_like(y,bool);mask[::5,11]=False;y[~mask]=0
            bank=dict(raw=z,current=z,targets=y,mask=mask)
            for seed in (42,43):
                bank[f'frozen_s{seed}']=z.copy();bank[f'control_s{seed}']=z.copy()
                bank[f'consistent_s{seed}']=z@np.diag([2.,-.5,1.3,-2.])+3
            banks[split]=bank;rows[split]=[dict(symbol='A',period=15,month='m',week=f'w{i//12}') for i in range(72)]
        ts=r.up.target_scales(banks['train']['targets'],banks['train']['mask']);rawstats=r.up.probe.scales(banks['train']['raw']);pca=np.eye(4);prior=dict(target_stats=ts,raw_stats=rawstats,heads={});tensors={}
        for name in ('control_s42','control_s43')+r.BASELINES:
            key=name.replace('control','frozen') if name.startswith('control') else name
            tr=r.old.representation(key,banks['train'],pca,rawstats);va=r.old.representation(key,banks['val'],pca,rawstats)
            head,w,b=r.up.fit_heads(tr,banks['train']['targets'],banks['train']['mask'],va,banks['val']['targets'],banks['val']['mask'],ts,'cpu')
            prior['heads'][name]=dict(targets=head);tensors[name+'_weights']=w;tensors[name+'_intercepts']=b
        atomic_json(prior,reader/'fit.json');np.savez(reader/'heads.npz',**tensors);np.savez(reader/'pca.npz',components=pca)
        config=dict(raw_retention=.05,family_retention=.10,baseline_gain=.05,minimum_utility_r2=.5,current_nmse=.1,current_r2=.8,min_windows=50,min_weeks=5)
        meta=dict(source=str(source),identity=dict(manifest=dict(reader=dict(root=str(reader),manifest=dict(decision=config)))))
        atomic_json(meta,out/'manifest.json')
        for split,bank in banks.items():
            np.savez(out/f'cache/{split}.npz',**bank)
            atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),files={f'{split}.npz':sha256(out/f'cache/{split}.npz')}),out/f'cache/{split}_index.json')
        legacy_report={}
        for split in r.old.SPLITS[2:]:
            bank=banks[split];legacy_report[split]=dict(scores={})
            for name in r.MODELS+r.BASELINES:
                oldname='control_s'+name[-2:] if name in r.MODELS else name
                x=bank[name] if name in r.MODELS else r.old.representation(name,bank,pca,rawstats)
                pred=r.up.predict(prior['heads'][oldname]['targets'],tensors[oldname+'_weights'],tensors[oldname+'_intercepts'],x,ts)
                score,err=r.up.measure(pred,bank['targets'],bank['mask'],ts)
                if name in r.MODELS:atomic_json(dict(utility=dict(scores=score,predictions=pred.tolist())),source/f'{split}_{name}{"" if name.startswith("frozen") else "_best"}.json')
                else:legacy_report[split]['scores'][name]=score
        atomic_json(dict(datasets=legacy_report),reader/'utility_metrics.json')
        return meta,out,rows,banks

    def test_refit_replays_and_recovers_rotated_coordinates_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as td, patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('Forbidden neural update')):
            meta,out,rows,banks=self.fixture(Path(td));r.fit(meta,out,'cpu');fit=r.read_json(out/'fit.json')
            self.assertEqual(len(fit['heads']),6)
            self.assertEqual(sum(len(h['candidates']) for v in fit['heads'].values() for h in v['targets']),390)
            for v in fit['heads'].values():
                for h in v['targets']:
                    self.assertEqual(h['alpha'],min(h['candidates'],key=lambda x:x['validation_nmse'])['alpha'])
            r.evaluate(meta,out,rows);report=r.read_json(out/'readout_metrics.json')
            self.assertEqual(report['decision']['status'],'readout_recovered_stability_candidate')
            self.assertFalse(report['decision']['automatic_promotion'])
            for split in r.old.SPLITS[2:]:
                s=report['datasets'][split]['scores'];self.assertLess(s['consistent_s42']['groups']['utility'],.001*s['legacy_consistent_s42']['groups']['utility'])
                pred=r.read_json(out/f'{split}_predictions.json');y=np.array(pred['targets']);mask=np.array(pred['mask']);sc=np.array(fit['target_stats']['scale'])
                for name,values in pred['predictions'].items():
                    err=np.where(mask,((np.array(values)-y)/sc)**2,0)
                    np.testing.assert_allclose(err[:,[0,2,3,5]].mean(1),pred['per_window_errors'][name]['utility'])
            with patch.object(r.up,'fit_heads',side_effect=AssertionError('No duplicate fitting')):r.fit(meta,out,'cpu')
            (out/'heads.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):r.fit(meta,out,'cpu')

    def test_research_requires_all_heads_locked_and_partial_fit_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            meta,out,rows,banks=self.fixture(Path(td))
            with self.assertRaises(FileNotFoundError):r.prepare(meta,out,'test',rows['test'],'cpu')
            original=r.up.fit_heads;calls=[]
            def interrupted(*a):
                calls.append(1)
                if len(calls)==3:raise RuntimeError('interruption')
                return original(*a)
            with patch.object(r.up,'fit_heads',side_effect=interrupted):
                with self.assertRaises(RuntimeError):r.fit(meta,out,'cpu')
            self.assertFalse((out/'selection_lock.json').exists());r.fit(meta,out,'cpu')
            r.old.check_selection(out)
            index=out/'cache/train_index.json';index.write_text(index.read_text()+' ')
            with self.assertRaises(ValueError):r.old.check_selection(out)

    def test_retention_rejects_utility_loss_even_if_current_is_preserved(self):
        rows=[dict(symbol='A',period=15,month='m',week=f'w{i//10}') for i in range(60)];mask=np.ones((60,13),bool)
        errors={n:{k:np.full(60,.03) for k in list(r.up.GROUPS)+list(r.up.NAMES)} for n in r.MODELS}
        scores={n:dict(targets=[dict(r2=.95) for _ in r.up.NAMES]) for n in r.MODELS}
        self.assertTrue(all(c['passed'] for c in r.retention(scores,errors,mask,rows)))
        errors['consistent_s42']['utility']=np.full(60,.1)
        checks=r.retention(scores,errors,mask,rows);self.assertFalse(all(c['passed'] for c in checks))
        self.assertTrue(all(c['passed'] for c in checks if c['metric'].startswith('current/')))
        mask[:,11]=False;checks=r.retention(scores,errors,mask,rows)
        self.assertTrue(any(not c['passed'] and c['interval']['support']==0 for c in checks))

    def test_target_masks_current_inputs_and_order_checked(self):
        rng=np.random.default_rng(6);x=rng.normal(0,.1,(6,128,28)).astype('float32');x[:,:,23:]=1
        stats=dict(x_mean=[0.]*28,x_scale=[1.]*28);y,mask=r.up.targets(r.up.restore_raw(x,stats));bank=dict(raw=x.reshape(6,-1),current=x[:,-1],targets=y,mask=mask)
        r.check_inputs(bank,x,stats,[{}]*6)
        for key in ('raw','current','targets','mask'):
            bad=copy.deepcopy(bank)
            if key=='mask':bad[key][0,0]=False
            else:bad[key].flat[0]+=1
            with self.assertRaises(ValueError):r.check_inputs(bad,x,stats,[{}]*6)
        with self.assertRaises(ValueError):r.check_inputs(bank,x[::-1],stats,[{}]*6)

    def test_frozen_extraction_no_updates_and_cache_tampering(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.core=torch.nn.Module();self.core.encoder=torch.nn.Linear(28,768,bias=False)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';reader=root/'reader';out.mkdir();(reader/'cache').mkdir(parents=True)
            x=np.zeros((4,128,28),np.float32);x[:,:,23:]=1;stats=dict(x_mean=[0.]*28,x_scale=[1.]*28);y,mask=r.up.targets(x)
            bank=dict(raw=x.reshape(4,-1),current=x[:,-1],targets=y,mask=mask,control_s42=np.zeros((4,768)),control_s43=np.zeros((4,768)))
            np.savez(reader/'cache/train.npz',**bank);atomic_json({},reader/'manifest.json');atomic_json(dict(manifest_sha256=sha256(reader/'manifest.json'),files={'train.npz':sha256(reader/'cache/train.npz')}),reader/'cache/train_index.json')
            rows=[{}]*4;atomic_json(rows,reader/'train_inventory.json');sm=dict(reader=dict(root=str(reader),manifest={}));meta=dict(batch=2,identity=dict(manifest=sm));atomic_json(meta,out/'manifest.json')
            model=Tiny().eval().requires_grad_(False);before=r.ur.bb.state_signature(model)
            with patch.object(r.cr,'alignment',return_value={}),patch.object(r.cr,'statistics',return_value=(stats,{})),patch.object(r.ur.bb,'load_data',return_value={'x':x}),patch.object(r,'load_selected',return_value=(model,None,None)):
                r.prepare(meta,out,'train',rows,'cpu');self.assertEqual(before,r.ur.bb.state_signature(model));r.prepare(meta,out,'train',rows,'cpu')
                (out/'cache/train.npz').write_bytes(b'bad')
                with self.assertRaises(ValueError):r.prepare(meta,out,'train',rows,'cpu')

    def test_orchestration_locks_before_research_and_completed_run_does_not_refit(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as td:
            meta,out,rows,banks=self.fixture(Path(td));source=Path(meta['source']);reader=Path(meta['identity']['manifest']['reader']['root'])
            identity=meta['identity'];meta=r.make_manifest(source.resolve(),identity,128);atomic_json(meta,out/'manifest.json')
            atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__),source/'runtime.json')
            for split in r.old.SPLITS:
                atomic_json(rows[split],reader/f'{split}_inventory.json')
                index=r.read_json(out/f'cache/{split}_index.json');index['manifest_sha256']=sha256(out/'manifest.json');atomic_json(index,out/f'cache/{split}_index.json')
            events=[]
            def prepare(meta,out,split,rows,device):
                if split in r.old.SPLITS[2:]:r.check_selection(out)
                events.append(split)
            with ExitStack() as stack:
                stack.enter_context(patch.object(r,'source_identity',return_value=identity))
                stack.enter_context(patch.object(r,'check_output'))
                stack.enter_context(patch.object(r.cr,'context',return_value={}))
                stack.enter_context(patch.object(r.ur,'inventories',return_value=rows))
                stack.enter_context(patch.object(r.up.probe,'inventory_audit',return_value={'passed':True}))
                stack.enter_context(patch.object(r,'preflight'))
                stack.enter_context(patch.object(r,'prepare',side_effect=prepare))
                r.run(source,out,128,'cpu');self.assertEqual(events,list(r.old.SPLITS))
                with patch.object(r,'fit',side_effect=AssertionError('No refit')):r.run(source,out,128,'cpu')
                self.assertEqual(events,list(r.old.SPLITS));self.assertEqual(r.read_json(out/'completion.json')['encoder_updates'],0)
                (out/'fit.json').write_text('{}')
                with self.assertRaises(ValueError):r.run(source,out,128,'cpu')

    def test_directory_guard_uses_explicit_ancestors_not_old_schema_depth(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';out=root/'new';a={k:str(root/k) for k in ('source','original_source','sampling_source')}
            identity=dict(manifest=dict(source=str(root/'parent'),reader=dict(root=str(root/'reader')),packed=dict(train=dict(directory=str(root/'bank')))))
            with patch.object(r.cr,'alignment',return_value=a):
                r.check_output(source,out,identity)
                for path in (source,source/'child',root,root/'reader',root/'parent',root/'bank/child',root/'sampling_source'):
                    with self.assertRaises(ValueError):r.check_output(source,path,identity)

    def test_shell_failure_export(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);env=os.environ|dict(BABEL_RECAL_RUN=str(root/'run'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false',BABEL_RECAL_LOG=str(root/'missing.log'))
            cmd=['bash','scripts/babel_consistency_readout768_autodl.sh'];res=subprocess.run(cmd+['all'],env=env,capture_output=True,text=True)
            self.assertEqual(res.returncode,1);archive=root/'download/run_reports.tar.gz';self.assertTrue(archive.exists())
            with tarfile.open(archive) as t:self.assertIn('run_status=failed',t.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
