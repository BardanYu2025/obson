import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_babel import series
from obson.babel.memory_benchmark import distant_targets,feature_row,fit_readout,projection


class MemoryBenchmarkTests(unittest.TestCase):
    def test_targets_past_dependence_future_isolation_and_scale(self):
        frame=series(2400)[0].frame.copy();end=2207;sigma=.01
        original=distant_targets(frame,end,sigma)
        modified=frame.copy()
        modified.loc[end+1:,["open","high","low","close"]]*=5
        np.testing.assert_array_equal(original,distant_targets(modified,end,sigma))
        modified=frame.copy();modified.loc[end-127:end-1,"high"]*=5
        np.testing.assert_array_equal(original,distant_targets(modified,end,sigma))
        modified=frame.copy();modified.loc[end-2047,"high"]*=5
        changed=distant_targets(modified,end,sigma)
        self.assertLess(changed[0],original[0]);self.assertEqual(changed[2],1.)
        self.assertEqual(changed[1],original[1])
        modified=frame.copy();modified[["open","high","low","close"]]*=7
        np.testing.assert_allclose(original,distant_targets(modified,end,sigma),atol=1e-12)
        with self.assertRaises(ValueError):distant_targets(frame,2046,sigma)

    def test_controls_preserve_current_and_do_not_read_future(self):
        frame=series(2400)[0].frame;end=2207
        bank=np.random.default_rng(7).normal(size=(len(range(127,len(frame),16)),256)).astype(np.float32)
        row=feature_row(bank,frame,end,.01,projection(),42)
        self.assertEqual(row["ordered"].shape,(511,))
        np.testing.assert_array_equal(row["ordered"][:256],row["masked"][:256])
        self.assertTrue((row["masked"][256:]==0).all())
        a=row["ordered"][256:].reshape(15,17);b=row["shuffled"][256:].reshape(15,17)
        self.assertEqual(sorted(map(tuple,a)),sorted(map(tuple,b)))
        altered=bank.copy();altered[(end-127)//16+1:]*=100
        other=feature_row(altered,frame,end,.01,projection(),42)
        for key in row:np.testing.assert_array_equal(row[key],other[key])

    def test_linear_readout(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(150,4));y=x[:,:3]*2+1
        result=fit_readout(x[:80],y[:80],x[80:110],y[80:110],x[110:],y[110:])
        self.assertEqual(result["alpha"],1.)
        for value in result["test"].values():self.assertGreater(value["r2"],.99)

    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"benchmark";run.mkdir()
            (run/"benchmark_metrics.json").write_text('{}')
            env=dict(os.environ,BABEL_MEMORY_BENCHMARK_RUN=str(run),BABEL_DOWNLOAD_DIR=str(root/"download"))
            subprocess.run(["bash","scripts/babel_memory_benchmark_autodl.sh","export"],env=env,check=True,capture_output=True)
            self.assertTrue((root/"download/benchmark/benchmark_metrics.json").exists())
