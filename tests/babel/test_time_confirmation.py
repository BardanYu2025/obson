"""Later-time eligibility, frozen paired inference, immutable recovery and export."""
import copy
import hashlib
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from obson.babel import time_confirmation as tc
from obson.babel.ae_extend import atomic_json
from obson.babel.data import validate_frame
from obson.babel.dual_state import sha256
from test_bar_alignment import small_model


def make_raw(root, n=1100, old=200):
    directory=root/'raw'/'X'; directory.mkdir(parents=True)
    values=np.arange(n)%17+100
    frame=pd.DataFrame(dict(datetime=pd.date_range('2024-01-01 09:00',periods=n,freq='15min'),
        open=values,high=values+2,low=values-2,close=values+1,volume=np.arange(n)+100,close_oi=np.arange(n)+1000))
    path=directory/'EX.X2401_15m.csv';frame.to_csv(path,index=False)
    history=validate_frame(pd.read_csv(path)).iloc[:old].reset_index(drop=True)
    row=dict(key='X/15/EX.X2401',start=str(history.datetime.iloc[0]),end=str(history.datetime.iloc[-1]),
        sha256=hashlib.sha256(pd.util.hash_pandas_object(history,index=False).values.tobytes()).hexdigest())
    return directory.parent,path,row


class TimeConfirmationTests(unittest.TestCase):
    def test_source_close_cutoff_all128_inputs_and_warmup(self):
        t=pd.date_range('2024-01-01',periods=900,freq='15min')
        s=SimpleNamespace(frame=pd.DataFrame(dict(datetime=t)), ends=(t+pd.Timedelta(minutes=15)).to_numpy(),
                          main=np.ones(900,bool),sessions=t.to_numpy('datetime64[D]'))
        cutoff=t[500];asof=t[899]
        got=tc.eligible_ends(s,cutoff,asof)
        np.testing.assert_array_equal(got,[639,767,895])
        self.assertTrue(all(t[e-127]>cutoff and e>=511 for e in got))
        s.main[767]=False
        np.testing.assert_array_equal(tc.eligible_ends(s,cutoff,t[895]),[639])
        rows=[dict(key='X/60/C',end='2024-01-01 10:00'),dict(key='Y/15/D',end='2024-01-01 10:30')]
        self.assertEqual(tc.cutoff_from_records(rows),pd.Timestamp('2024-01-01 11:00'))

    def test_raw_append_prefix_and_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            root,path,row=make_raw(Path(td));cutoff=tc.cutoff_from_records([row])
            audit=tc.audit_snapshot(root,[row],cutoff)
            self.assertEqual(audit['status'],'eligible');self.assertEqual(audit['later_rows'],899)
            df=pd.read_csv(path);df.loc[0,'volume']+=1;df.to_csv(path,index=False)
            self.assertEqual(tc.audit_snapshot(root,[row],cutoff)['reason'],'raw_provenance_failed')
            path.unlink()
            self.assertEqual(tc.audit_snapshot(root,[row],cutoff)['issues'][0]['reason'],'registered_contract_missing')

    def test_prior_raw_audits_extend_boundary_even_for_cross_symbols(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(dict(files=[dict(symbol='Y',period=60,contract='Y2609',start='2026-09-01',
                end='2026-09-15',source_hash='b'*64)]),root/'raw_audit.json')
            lineage=dict(source_records=[dict(key='X/15/C',start='2020',end='2026-09-12',sha256='a'*64)],
                         manifests={str(root/'manifest.json'):'c'*64})
            rows,extra=tc.past_records(lineage)
            self.assertEqual(tc.cutoff_from_records(rows),pd.Timestamp('2026-09-15 01:00'))
            self.assertIn(str(root/'raw_audit.json'),extra)

    def test_zero_later_data_exits_before_gpu_and_preserves_blocked_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,path,row=make_raw(base,n=200,old=200)
            registry=base/'checkpoints';source=registry/'alignment';source.mkdir(parents=True)
            atomic_json(dict(sources=[row]),source/'manifest.json');out=registry/'new'
            with patch.object(tc.ea,'source_identity',return_value={}),patch.object(tc,'evaluate',side_effect=AssertionError('No inference')):
                self.assertEqual(tc.run(source,registry,root,out),3)
                self.assertEqual(tc.run(source,registry,root,out),3)
            self.assertEqual(tc.read_json(out/'run_state.json')['reason'],'no_later_raw_history')
            self.assertFalse((out/'manifest.json').exists())

    def test_gain_does_not_hide_path_retention_failure_and_low_support(self):
        n=60;inventory=[dict(symbol='X',period=15,month='2024-01',week=f'week{i//10}') for i in range(n)]
        metrics=('primary','path','changes','body','activity','close_mae_bps')
        errors={}
        for job in tc.JOBS:
            for kind in tc.KINDS:
                value=.7 if '768' in job else 1.
                errors[f'{job}/{kind}']={task:{m:np.full(n,value) for m in metrics} for task in ('global','recent')}
        _,decision=tc.comparison_checks(errors,inventory)
        self.assertEqual(decision['status'],'gain_and_retention_confirmed')
        errors['w768_joint_s43/best']['global']['path'][:]=1.15
        _,decision=tc.comparison_checks(errors,inventory)
        self.assertEqual(decision['status'],'mixed_or_unconfirmed')
        self.assertEqual(sum(not r['passed'] for r in decision['checks']),1)
        short={n:{t:{k:v[:5] for k,v in e.items()} for t,e in tasks.items()} for n,tasks in errors.items()}
        self.assertEqual(tc.comparison_checks(short,inventory[:5])[1]['status'],'insufficient_support')

    def test_complete_frozen_pipeline_replay_and_tamper_rejection(self):
        model,data,stats,local=small_model();model.eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,path,row=make_raw(base)
            registry=base/'checkpoints';source=registry/'alignment';(source/'cache').mkdir(parents=True)
            jobs=[dict(name=n,joint='_joint_' in n) for n in tc.JOBS]
            sm=dict(sources=[row],experiments=jobs)
            atomic_json(sm,source/'manifest.json');atomic_json(stats,source/'cache/statistics.json')
            atomic_json(local,source/'cache/local_scales.json');atomic_json(dict(status='retain_baseline_and_stop'),source/'decision.json')
            atomic_json(dict(torch=str(torch.__version__),numpy=np.__version__),source/'runtime.json')
            for w in (512,768):
                np.savez(source/f'cache/pca{w}.npz',components=model.core.decoder.basis.numpy(),mean=model.core.decoder.target_mean.numpy())
            out=registry/'new';expected={'selection':1.}
            identity={'manifest':sm}
            patches=[patch.object(tc.ea,'source_identity',return_value=identity),
                     patch.object(tc.ea,'load_model',side_effect=lambda *a:(copy.deepcopy(model),expected)),
                     patch.object(tc.ea.bb,'load_data',return_value=data),
                     patch.object(tc.ea.bb,'validation',return_value=expected),
                     patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer')),
                     patch.object(tc.ea.bb.ab,'fit_pca',side_effect=AssertionError('No PCA fitting'))]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in patches:stack.enter_context(p)
                self.assertEqual(tc.run(source,registry,root,out,'cpu',2),0)
                self.assertTrue(tc.read_json(out/'completion.json')['source_unchanged'])
                result=tc.read_json(out/'confirmation_metrics.json');self.assertEqual(len(result['scores']),10)
                self.assertEqual(len(tc.read_json(out/'validation_replay.json')),8)
                self.assertEqual(tc.read_json(out/'decision.json')['status'],'insufficient_support')
                # Interrupt after a locked inference, then recover without moving asof.
                (out/'completion.json').unlink()
                old_asof=tc.read_json(out/'manifest.json')['asof']
                with patch.object(tc,'prepare_data',side_effect=AssertionError('Locked cache must be reused')):
                    self.assertEqual(tc.run(source,registry,root,out,'cpu',2),0)
                self.assertEqual(old_asof,tc.read_json(out/'manifest.json')['asof'])
                # Completed resume performs no second inference and keeps source fingerprints.
                before=sha256(out/'completion.json')
                with patch.object(tc,'evaluate',side_effect=AssertionError('No re-evaluation')):
                    self.assertEqual(tc.run(source,registry,root,out,'cpu',2),0)
                self.assertEqual(before,sha256(out/'completion.json'))
                with self.assertRaisesRegex(ValueError,'configuration'):
                    tc.run(source,registry,root,out,'cpu',3)
                (out/'cache/x.npy').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Completed report changed'):
                    tc.run(source,registry,root,out,'cpu',2)

    def test_shell_export_excludes_arrays_and_keeps_blocked_status(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'attempt';out.mkdir();download=root/'download'
            atomic_json(dict(status='blocked',reason='no_later_raw_history'),out/'run_state.json')
            (out/'weights.pt').write_bytes(b'no');(out/'cache.npy').write_bytes(b'no')
            env=os.environ|dict(BABEL_TIME_RUN=str(out),BABEL_DOWNLOAD_DIR=str(download),PYTHON_BIN=str(Path('.venv/bin/python').resolve()))
            result=subprocess.run(['bash','scripts/babel_time_confirmation_autodl.sh','export'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with tarfile.open(download/'attempt_reports.tar.gz') as archive:
                self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in archive.getnames()))
                self.assertIn(b'run_status=blocked',archive.extractfile('attempt/run_status.txt').read())
            self.assertFalse((out/'run_status.txt').exists())

    def test_shell_failure_still_exports(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);registry=root/'registry';registry.mkdir();out=registry/'attempt'
            env=os.environ|dict(BABEL_TIME_RUN=str(out),BABEL_TIME_SOURCE=str(registry/'missing'),
                BABEL_TIME_REGISTRY=str(registry),BABEL_TIME_ROOT=str(root),BABEL_DOWNLOAD_DIR=str(root/'download'),
                PYTHON_BIN=str(Path('.venv/bin/python').resolve()))
            result=subprocess.run(['bash','scripts/babel_time_confirmation_autodl.sh','all'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            with tarfile.open(root/'download/attempt_reports.tar.gz') as archive:
                self.assertIn(b'run_status=failed',archive.extractfile('attempt/run_status.txt').read())


if __name__=='__main__':
    unittest.main()
