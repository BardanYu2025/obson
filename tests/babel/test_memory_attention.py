import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from obson.babel.ae_extend import atomic_save,rng_state
from obson.babel.memory_attention import MemoryAttention,normalization,predict,tensors,train_arm


class MemoryAttentionTests(unittest.TestCase):
    def test_anchor_control_removes_only_historical_embeddings(self):
        torch.manual_seed(42);model=MemoryAttention().eval()
        x=torch.randn(3,511);changed=x.clone()
        changed[:,256:].reshape(-1,15,17)[...,:16]+=10
        torch.testing.assert_close(model(x,True),model(changed,True))
        self.assertGreater((model(x)-model(changed)).abs().sum().item(),1e-5)
        pred=model(x)
        self.assertEqual(pred.shape,(3,3))
        pred.square().mean().backward()
        self.assertGreater(model.token.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.current.weight.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(model.read.in_proj_weight.grad).all())

    def test_normalization_and_eval_no_updates(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(20,511)).astype(np.float32);y=rng.normal(size=(20,3))
        stats=normalization(x,y);before={k:v.clone() for k,v in stats.items()}
        ds=tensors(x+100,y,stats)
        model=MemoryAttention();pred=predict(model,ds,"cpu",8,False)
        self.assertEqual(pred.shape,(20,3))
        for k,v in stats.items():torch.testing.assert_close(v,before[k])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_completed_recovery_without_optimizer_updates(self):
        x=np.random.default_rng(42).normal(size=(12,511)).astype(np.float32);y=np.zeros((12,3))
        values=[({"ordered":x},y,[]) for _ in range(3)]
        model=MemoryAttention();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        metadata={"epochs":50,"batch_size":256}
        state={"schema":"babel-memory-attention-resume-v1","metadata":{**metadata,"condition":"full"},
               "epoch":50,"model":model.state_dict(),"optimizer":opt.state_dict(),"rng":rng_state(),
               "normalization":normalization(x,y),"history":[{"epoch":50}],
               "best_epoch":40,"best_loss":.2,"best_model":model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);atomic_save(state,path/"last.pt")
            train_arm(values,path,metadata,"full","cpu")
            self.assertEqual(torch.load(path/"best.pt",weights_only=True)["epoch"],40)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                train_arm(values,path,{**metadata,"epochs":51},"full","cpu")

    def test_export_both_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"attention";run.mkdir()
            for mode in ("full","anchors_only"):
                (run/mode).mkdir();(run/mode/"history.jsonl").write_text('{"epoch":1}\n')
            env=dict(os.environ,BABEL_ATTENTION_RUN=str(run),BABEL_DOWNLOAD_DIR=str(root/"download"))
            subprocess.run(["bash","scripts/babel_memory_attention_autodl.sh","export"],env=env,check=True,capture_output=True)
            for mode in ("full","anchors_only"):self.assertTrue((root/"download/attention"/f"{mode}_history.jsonl").exists())
