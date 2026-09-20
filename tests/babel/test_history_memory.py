import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_babel import series
from obson.babel.ae_context import encode_context
from obson.babel.data import split_boundaries
from obson.babel.history_autoencoder import HistoryAE, HistoryWindows
from obson.babel.history_memory import bank_index, encode_bank, examples, history_indices, read_memory, retrieve_past, state_features, write_review


class MemoryTests(unittest.TestCase):
    def test_nonoverlapping_past_reads_and_future_isolation(self):
        bank=np.arange(80*8,dtype=np.float32).reshape(80,8)+1
        end=127+16*32
        ids=history_indices(end,len(bank))
        self.assertEqual(ids.tolist(),[32,24,16,8,0])
        current,count=read_memory(bank,end)
        self.assertEqual(count,5)
        np.testing.assert_array_equal(current["local256"],bank[32])
        np.testing.assert_allclose(current["mean4_256"],bank[[32,24,16,8]].mean(0))
        future=bank.copy();future[33:]*=-100
        for name,value in current.items():
            np.testing.assert_array_equal(value,read_memory(future,end)[0][name])
        self.assertEqual(retrieve_past(bank,end),retrieve_past(future,end))
        for item in retrieve_past(bank,end):
            self.assertLess(item["end_row"],end-127)
        self.assertEqual(retrieve_past(bank,127),[])
        with self.assertRaises(ValueError): history_indices(128,len(bank))

    def test_cached_local_matches_direct_and_future_changes(self):
        s=series(400)[0]
        index=bank_index(s)
        self.assertEqual(index["end_rows"][:2],[127,143])
        self.assertEqual(index["price_anchors"][0],s.frame.open.iloc[0])
        self.assertEqual(index["price_anchors"][1],s.frame.close.iloc[15])
        e=encode_context(s.frame,s.period,"ema8_32")
        model=HistoryAE(hidden=16,layers=1,latent=8,context="ema8_32").eval()
        bank=encode_bank(model,e,"cpu",8)
        end=255
        with torch.no_grad(): direct=model.encode(torch.tensor(e["x"][128:256])[None])[:,-1].numpy()[0]
        np.testing.assert_allclose(read_memory(bank,end)[0]["local256"],direct,atol=1e-6)
        changed={**e,"x":e["x"].copy()};changed["x"][256:]+=10
        other=encode_bank(model,changed,"cpu",8)
        np.testing.assert_allclose(bank[:9],other[:9],atol=1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_state_alignment_and_standalone_review(self):
        data=series(1600)
        encoded=[encode_context(s.frame,s.period,"ema8_32") for s in data]
        banks=[np.full((len(range(127,len(s.frame),16)),256),i+1,dtype=np.float32) for i,s in enumerate(data)]
        ds=HistoryWindows(data,encoded,split_boundaries(data),"test")
        features,labels,counts=state_features(ds,banks)
        self.assertEqual(features["multiscale768"].shape,(len(ds),768))
        for j,(i,end) in enumerate(ds.items):
            np.testing.assert_array_equal(labels[j],data[i].labels["state"][end])
            self.assertTrue((features["local256"][j]==i+1).all())
        cards=examples(ds,banks)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"review.html";write_review(path,cards)
            self.assertIn('<svg',path.read_text())
            self.assertNotIn('<script',path.read_text())
        self.assertTrue((counts>=1).all())

    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"memory";run.mkdir()
            (run/"memory_metrics.json").write_text('{}')
            (run/"cache").mkdir();(run/"cache/large.npy").write_bytes(b'not for download')
            env=dict(os.environ,BABEL_MEMORY_RUN=str(run),BABEL_DOWNLOAD_DIR=str(root/"download"))
            subprocess.run(["bash","scripts/babel_memory_autodl.sh","export"],env=env,check=True,capture_output=True)
            self.assertTrue((root/"download/memory/memory_metrics.json").exists())
            self.assertFalse((root/"download/memory/cache").exists())
