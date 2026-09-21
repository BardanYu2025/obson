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

from test_activity_extend import fixture as extend_fixture, assert_tree
from obson.babel import price_readapt as pr
from obson.babel import activity_extend as ex, activity_alignment as al
from obson.babel.ae_extend import atomic_json, atomic_save
from obson.babel.dual_state import sha256


def fixture(root):
    old,grand,meta,parent=extend_fixture(root)
    for job in meta['experiments']:
        state=ex.import_state(meta,job);score=state['history'][-1]['validation']
        state.update(epoch=100,history=[dict(epoch=e,train=score,validation=score) for e in range(1,101)],
            activity_reference=score,selectors_initialized=True)
        (parent/job['name']).mkdir();ex.publish(state,parent/job['name'])
    with patch.object(al,'source_identity',return_value=({},old['upstream'])),\
         patch.object(al,'nonlinear_probe',side_effect=lambda z,y,m,*args:al.ridge_probe(z,y,m)),\
         patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
        al.evaluate(parent,'cpu');ex.diagnose_last(parent,'cpu')
    atomic_json(dict(status='complete'),parent/'completion.json')
    # Supply an actual fixed-reference report and refresh all pinned metadata.
    (grand/'alignment_metrics.json').write_text((parent/'alignment_metrics.json').read_text())
    for job in meta['experiments']:
        (grand/job['name']/'per_window_metrics.json').write_text((parent/job['name']/'per_window_metrics.json').read_text())
    with patch.object(al,'source_identity',return_value=({},old['upstream'])):
        meta['parent_identity']=ex.parent_identity(grand)[1]
    atomic_json(meta,parent/'manifest.json')
    index=json.loads((parent/'cache/index.json').read_text());index['manifest']=meta;atomic_json(index,parent/'cache/index.json')
    for job in meta['experiments']:
        for file in ('last.pt','best.pt','activity_best.pt'):
            path=parent/job['name']/file;ck=torch.load(path,weights_only=True);ck['metadata']['manifest']=meta;atomic_save(ck,path)
    with patch.object(al,'source_identity',return_value=({},old['upstream'])):oldmeta,identity=pr.source_identity(parent)
    out=root/'readapt';out.mkdir();new=pr.make_manifest(parent,oldmeta,identity,2,2)
    atomic_json(new,out/'manifest.json');pr.prepare(new,out,'cpu')
    return new,out,parent,old


class PriceReadaptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_common_head_cache_replay_and_corruption_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out,parent,old=fixture(Path(tmp));a=pr.initial_head(meta,'cpu');b=pr.initial_head(meta,'cpu')
            assert_tree(self,a.state_dict(),b.state_dict())
            with patch.object(al,'initial_model',side_effect=AssertionError('Cached reuse')):pr.prepare(meta,out,'cpu')
            for job in meta['experiments']:
                data=pr.arrays(out,job['name'],'train','cpu');self.assertFalse(data['z'].requires_grad)
                self.assertEqual(len(data['z']),len(data['y']))
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No updates')):
                self.assertGreater(pr.preflight(meta,'cpu')['gradient_norm'],0)
            p=out/f'cache/{meta["experiments"][0]["name"]}_train_z.npy';p.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'fingerprint'):pr.prepare(meta,out,'cpu')
            Path(meta['common_head']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'Common head changed'):pr.initial_head(meta,'cpu')

    def test_selector_qualifies_before_minimizing_and_marks_failure(self):
        model=torch.nn.Linear(2,2);ref=dict(base=1.,detail=1.,close_bps=1.,change16_mse=1.)
        state={};bad=dict(ref,close_bps=2.,detail=.1)
        pr.consider(state,model,0,bad,ref);self.assertFalse(state['qualified'])
        pr.consider(state,model,1,ref,ref);self.assertTrue(state['qualified']);self.assertEqual(state['best_epoch'],1)
        pr.consider(state,model,2,bad,ref);self.assertEqual(state['best_epoch'],1)
        pr.consider(state,model,3,dict(ref,detail=.9),ref);self.assertEqual(state['best_epoch'],3)
        cfg=dict(epochs=100,decoder_lr=1e-4,min_lr=1e-5)
        self.assertAlmostEqual(pr.learning_rate(cfg,1),1e-4);self.assertAlmostEqual(pr.learning_rate(cfg,100),1e-5)

    def test_workers_resume_frozen_cache_reports_and_fixed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,out,parent,old=fixture(Path(tmp));source_hashes={p:sha256(p) for p in parent.rglob('*') if p.is_file()}
            real=pr.head_epoch;calls=[]
            def no_update(head,data,scales,batch,weight=.25,opt=None,seed=0,collect=False):
                if opt is not None:
                    self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},{id(p) for p in head.parameters()})
                    calls.append((seed,opt.param_groups[0]['lr']))
                return real(head,data,scales,batch,weight,None,seed,collect)
            with patch.object(pr,'head_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                for job in meta['experiments']:pr.worker(out,job['name'],'cpu')
            self.assertEqual(calls,[(43,1e-4),(44,1e-5)]*2)
            for job in meta['experiments']:
                path=out/job['name']/'last.pt';before=sha256(path)
                with patch.object(pr,'head_epoch',side_effect=AssertionError('Complete resume retrained')):pr.worker(out,job['name'],'cpu')
                self.assertEqual(sha256(path),before)
            # Restore nonzero synthetic Adam moments mid-run without taking any step.
            job=meta['experiments'][0];path=out/job['name']/'last.pt';state=torch.load(path,weights_only=True)
            state['epoch']=1;state['history']=state['history'][:1]
            head=pr.initial_head(meta,'cpu')
            for idx,param in enumerate(head.parameters()):
                state['optimizer']['state'][idx]=dict(step=torch.tensor(7.),exp_avg=torch.full_like(param,.003),exp_avg_sq=torch.full_like(param,.01))
            saved=copy.deepcopy(state);atomic_save(state,path)
            with patch.object(pr,'head_epoch',side_effect=no_update),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No local training')):
                pr.worker(out,job['name'],'cpu')
            restored=torch.load(path,weights_only=True)
            assert_tree(self,restored['optimizer']['state'],saved['optimizer']['state'])
            assert_tree(self,restored['rng'],saved['rng']);assert_tree(self,restored['model'],saved['model'])
            self.assertEqual([r['epoch'] for r in restored['history']],[1,2])
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('Evaluation trained')):report=pr.evaluate(out,'cpu')
            self.assertFalse(report['automatic_promotion']);self.assertFalse(report['evidence_pass'])
            self.assertFalse(report['stage_screen'][0]['trained_selection'])
            for job in meta['experiments']:
                row=report['variants'][job['name']]
                self.assertEqual(row['final']['epoch'],2);self.assertTrue(row['activity_and_state_unchanged'])
                self.assertIn(job['name']+'_minus_fixed_reference',report['paired'])
            for p,digest in source_hashes.items():self.assertEqual(sha256(p),digest)
            path=out/meta['experiments'][0]['name']/'last.pt';ck=torch.load(path,weights_only=True);ck['epoch']=1;atomic_save(ck,path)
            with self.assertRaisesRegex(ValueError,'Incomplete head training'):pr.evaluate(out,'cpu')

    def test_no_training_collection_and_endpoint_pair_checks(self):
        with self.assertRaisesRegex(ValueError,'chronological'):pr.head_epoch(None,None,None,2,opt=object(),collect=True)
        with self.assertRaisesRegex(ValueError,'length'):pr.paired([{}],[],[{}])
        with self.assertRaisesRegex(ValueError,'replay'):pr.assert_score(dict(base=1.),dict(base=2.))

    def test_shell_failure_and_report_export_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();(run/'readapt_metrics.json').write_text('{}');(run/'last.pt').write_bytes(b'x')
            env=dict(os.environ,BABEL_READAPT_RUN=str(run),BABEL_READAPT_LOG=str(root/'none'),BABEL_DOWNLOAD_DIR=str(root/'download'),PYTHON_BIN='/usr/bin/false')
            for mode,code in [('export',0),('all',1)]:
                p=subprocess.run(['bash','scripts/babel_price_readapt_autodl.sh',mode],env=env,capture_output=True,text=True)
                self.assertEqual(p.returncode,code,p.stderr)
                with tarfile.open(root/'download/run_reports.tar.gz') as t:
                    self.assertIn('run/readapt_metrics.json',t.getnames());self.assertFalse(any(n.endswith('.pt') for n in t.getnames()))
                    self.assertIn(f'command_exit_code={code}',t.extractfile('run/run_status.txt').read().decode())
