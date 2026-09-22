"""Endpoint lineage, overlap, causal distributions and cache verification; no training."""
import tempfile
import os
import subprocess
import tarfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from obson.babel import coverage_audit as ca, architecture as ar
from obson.babel.data import validate_frame
from obson.babel.history_autoencoder import HistoryWindows
from obson.babel.large_history import HierWindows
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256
from test_architecture_benchmark import features


def series(n=1500):
    close=np.exp(np.linspace(4,4.3,n));op=np.r_[close[0],close[:-1]]
    frame=validate_frame(pd.DataFrame(dict(datetime=pd.date_range('2020-01-01',periods=n,freq='h'),
        open=op,close=close,high=np.maximum(op,close),low=np.minimum(op,close),volume=np.ones(n))))
    return SimpleNamespace(frame=frame,period=60,code='x',contract='x01',key='x/60/x01',
        sessions=frame.datetime.to_numpy('datetime64[D]'),main=np.ones(n,bool),x=np.zeros((n,12)))

BOUNDS=dict(train_until='2020-01-25',val_until='2020-02-15',test_until='2020-04-01')


class CoverageTests(unittest.TestCase):
    def test_original_hierarchy_rule_exactly_reproduces_and_local_gate_is_separate(self):
        s=series();s.main[::11]=False;s.sessions[-1]=np.datetime64('NaT','D')
        for split in ('train','val','test'):
            base=HistoryWindows([s],[dict(x=np.zeros((len(s.frame),18)))],BOUNDS,split)
            try:hier=HierWindows(base,[np.zeros((20,1))],BOUNDS,split)
            except ValueError:
                self.assertEqual(len(ca.endpoints(s,BOUNDS,split,128,512)),0);continue
            self.assertEqual([(i,end) for i,end,n in hier.items],[(0,int(e)) for e in ca.endpoints(s,BOUNDS,split,128,512)])
        old=ca.endpoints(s,BOUNDS,'train',128,512);local=ca.endpoints(s,BOUNDS,'train',128,128)
        self.assertTrue(set(old)<set(local))
        self.assertTrue(set(old)<=set(ca.endpoints(s,BOUNDS,'train',16,512)))

    def test_exact_overlap_union_and_duplicate_rejection(self):
        row,counts=ca.overlap_stats(400,[127,255]);self.assertEqual(row['unique_contract_period_bars'],256);self.assertEqual(row['overlapping_adjacent_pairs'],0)
        row,counts=ca.overlap_stats(400,[127,143]);self.assertEqual(row['unique_contract_period_bars'],144);self.assertEqual(row['adjacent_overlap_bars'],112)
        self.assertEqual(int(counts.sum()),256);self.assertEqual(row['max_multiplicity'],2)
        with self.assertRaises(ValueError):ca.overlap_stats(400,[127,127])
        with self.assertRaises(ValueError):ca.overlap_stats(400,[126])

    def test_specs_bijection_and_partition_overlap_rejection(self):
        s=series();spec=[dict(series=0,lo=0,length=256,offset=0,endpoints=[[127,1],[255,0]])]
        self.assertEqual(ca.unpack_specs(spec,[s],BOUNDS,'train',2),[(0,255),(0,127)])
        spec[0]['endpoints'][1][1]=1
        with self.assertRaises(ValueError):ca.unpack_specs(spec,[s],BOUNDS,'train',2)
        good=dict(train=[(0,127)],val=[(0,255)],test=[(0,383)])
        self.assertEqual(set(ca.prove_disjoint([s],good).values()),{0})
        with self.assertRaises(ValueError):ca.prove_disjoint([s],dict(good,val=[(0,128)]))

    def test_descriptor_current_row_is_excluded_and_reference_is_not_refit(self):
        x=features(1)[0];y,mask=ar.ordered_targets(x[None]);stats=ar.fit_scales(x[None],y,mask)
        before=ca.describe_windows(None,x,[127],stats);changed=x.copy();changed[127]=100
        np.testing.assert_array_equal(before,ca.describe_windows(None,changed,[127],stats))
        ref=np.arange(100,dtype=float);r=ca.distribution(np.array([-100.,200.]),ref)
        self.assertEqual(r['outside_train_min_max_fraction'],1.)
        np.testing.assert_array_equal(ref,np.arange(100))

    def test_cached_replay_and_missing_binary_scope(self):
        x=features(1)[0];y,mask=ar.ordered_targets(x[None]);stats=ar.fit_scales(x[None],y,mask)
        xx,yy,mm=ar.normalize(x[None],y,mask,stats)
        with tempfile.TemporaryDirectory() as td:
            source=Path(td);cache=source/'cache';cache.mkdir();files={}
            for name,a in [('x',xx),('y',yy),('mask',mm)]:
                p=cache/f'train_{name}.npy';np.save(p,a);files[p.name]=sha256(p)
                self.assertEqual(ca.ndarray_hash(a),files[p.name])
            atomic_json(dict(files=files),cache/'index.json')
            result=ca.replay_windows([], [x],[(0,127)],stats,source,'train',True)
            self.assertTrue(all(r['exact_reconstruction'] and r['numeric_replay_passed'] for r in result.values()))
            (cache/'train_x.npy').unlink();files['train_x.npy']='invalid';atomic_json(dict(files=files),cache/'index.json')
            with self.assertRaisesRegex(ValueError,'absent binary'):ca.replay_windows([],[x],[(0,127)],stats,source,'train',True)
            result=ca.replay_windows([],[x],[(0,127)],stats,source,'train',False)
            self.assertFalse(result['x']['exact_reconstruction']);self.assertFalse(result['x']['source_binary_available'])

    def test_cpu_dispatch_export_failure_and_reexport(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';run.mkdir();download=root/'download'
            atomic_json(dict(status='partial'),run/'metrics.json')
            for ext in ('pt','npy','npz'):(run/f'binary.{ext}').write_bytes(b'not for reports')
            fake=root/'fake'
            fake.write_text('#!/bin/sh\necho "$@" > "$ARGUMENT_LOG"\nexit 7\n');fake.chmod(0o755)
            env=dict(os.environ,PYTHON_BIN=str(fake),ARGUMENT_LOG=str(root/'args'),BABEL_COVERAGE_RUN=str(run),
                BABEL_COVERAGE_LOG=str(root/'no_log'),BABEL_DOWNLOAD_DIR=str(download))
            repo=Path(__file__).resolve().parents[2];cmd=['bash',str(repo/'scripts/babel_coverage_autodl.sh')]
            for action,code in [('audit',7),('export',0)]:
                r=subprocess.run(cmd+[action],env=env,capture_output=True,text=True);self.assertEqual(r.returncode,code,r.stderr)
                self.assertIn('obson.babel.coverage_audit',(root/'args').read_text())
                self.assertNotIn('--reports-only',(root/'args').read_text())
                with tarfile.open(download/'run_reports.tar.gz') as archive:
                    self.assertFalse(any(n.endswith(('.pt','.npy','.npz')) for n in archive.getnames()))
                    self.assertIn('run_status=failed',archive.extractfile('run/run_status.txt').read().decode())


if __name__=='__main__':unittest.main()
