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
from obson.babel.history_autoencoder import HistoryAE,HistoryWindows
from obson.babel.large_history import CheckpointLocal,HierWindows,LargeHistory,objective,preflight_batch,publish,set_stage,train_stage,write_global_review
from obson.babel.ae_extend import atomic_save,rng_state
from obson.babel.representation import time_mask

SMALL=dict(hidden=16,layers=1,heads=4,latent=16,window=128,dropout=0.,context="ema8_32")


class LargeHistoryTests(unittest.TestCase):
    def test_preflight_batch_device(self):
        # Meta catches accidental CPU allocations even on hosts without CUDA.
        devices=["cpu","meta"]+(["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            batch=preflight_batch(2,device)
            for name,value in batch.items():
                self.assertEqual(value.device.type,device,name)
            self.assertEqual(batch["valid"].dtype,torch.bool)
            self.assertEqual(batch["valid"].shape,(2,16))

    def test_checkpoint_local_equivalent_and_causal(self):
        a=HistoryAE(**SMALL).eval();b=CheckpointLocal(**SMALL).eval();b.load_state_dict(a.state_dict())
        x=torch.randn(2,128,18)
        torch.testing.assert_close(a(x)["reconstruction"],b(x)["reconstruction"])
        changed=x.clone();changed[:,80:]+=10
        torch.testing.assert_close(b.encode(x)[:,:80],b.encode(changed)[:,:80])
        b.train();out=b(x);out["reconstruction"].square().mean().backward()
        self.assertTrue(torch.isfinite(b.input.weight.grad).all())

    def test_global_decode_masking_and_joint_backward(self):
        model=LargeHistory(SMALL,blocks=4,aggregate_layers=1).eval()
        b=dict(x=torch.randn(1,4,128,18),y=torch.randn(1,4,128,7),mask=torch.ones(1,4,128,7,dtype=torch.bool),
               teacher=torch.randn(1,4,16),offsets=torch.tensor([[0.,-1.,-.5,0.]]),
               valid=torch.tensor([[False,True,True,True]]),targets=torch.randn(1,3))
        b["mask"][:,0]=False;b["y"][...,2:]=b["y"][...,2:].abs()
        out=model(b)
        torch.testing.assert_close(out["reconstruction"],model.decode_history(out["z"])["reconstruction"])
        changed={k:v.clone() for k,v in b.items()};changed["teacher"][:,0]+=100;changed["offsets"][:,0]=100
        torch.testing.assert_close(out["z"],model(changed)["z"])
        self.assertEqual(out["anchors"][0,-1].item(),0.)
        set_stage(model,"aggregate",True)
        self.assertFalse(any(p.requires_grad for p in model.local.parameters()))
        self.assertFalse(model.local.training)
        frozen_loss,_=objective(model,b,model(b),torch.zeros(3),torch.ones(3),False)
        frozen_loss.backward()
        self.assertTrue(all(p.grad is None for p in model.local.parameters()))
        self.assertGreater(model.history_decoder[0].weight.grad.abs().sum().item(),0.)
        model.zero_grad(set_to_none=True)
        set_stage(model,"joint",True)
        loss,_=objective(model,b,model(b,joint=True),torch.zeros(3),torch.ones(3),True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(model.local.input.weight.grad.abs().sum().item(),0.)
        self.assertGreater(model.history_decoder[0].weight.grad.abs().sum().item(),0.)

    def test_hierarchical_split_and_physical_targets(self):
        data=series(4000);encoded=[encode_context(s.frame,s.period,"ema8_32") for s in data]
        banks=[np.ones((len(range(127,len(s.frame),128)),16),np.float32) for s in data]
        bounds=split_boundaries(data)
        for split in ("train","val","test"):
            base=HistoryWindows(data,encoded,bounds,split);ds=HierWindows(base,banks,bounds,split)
            for i,end,n in ds.items:self.assertTrue(time_mask(data[i],bounds,split)[end+1-n*128:end+1].all())
            sample=ds[0];i,end,n=ds.items[0]
            self.assertEqual(sample["valid"].sum(),n);self.assertEqual(sample["offsets"][-1],0.)
            actual=np.log(data[i].frame.close.iloc[end])*100-np.log(sample["anchor"])*100
            self.assertAlmostEqual(float(sample["y"][-1,-1,1]),actual,places=5)
            # Future input changes cannot alter this sample or its historical targets.
            e=encoded[i]["x"].copy();encoded[i]["x"][end+1:]+=10
            np.testing.assert_array_equal(sample["x"],ds[0]["x"])
            np.testing.assert_array_equal(sample["targets"],ds[0]["targets"])
            encoded[i]["x"]=e

    def test_best_artifact_stable_across_recovery(self):
        state=dict(metadata={"epochs":2},best_epoch=1,best_model={"w":torch.ones(2)},history=[])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);publish(state,path);before=(path/"best.pt").read_bytes()
            state["best_model"]={"w":state["best_model"]["w"].clone()}
            publish(state,path);self.assertEqual(before,(path/"best.pt").read_bytes())

    def test_completed_stage_recovery_without_training(self):
        model=LargeHistory(SMALL,blocks=4,aggregate_layers=1);set_stage(model,"local",True)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.01)
        metadata=dict(epochs=2,micro=1,effective=2,lr=3e-4,target_mean=[0.,0.,0.],target_scale=[1.,1.,1.])
        state=dict(metadata={**metadata,"stage":"local"},epoch=2,model=model.state_dict(),optimizer=opt.state_dict(),
                   rng=rng_state(),best_model=model.state_dict(),best_epoch=1,best_loss=.1,history=[{"epoch":2}])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);atomic_save(state,path/"last.pt")
            train_stage(model,([],[]),path,metadata,"local","cpu")
            self.assertEqual(torch.load(path/"best.pt",weights_only=True)["epoch"],1)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                train_stage(model,([],[]),path,{**metadata,"epochs":3},"local","cpu")

    def test_export_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"large";run.mkdir();(run/"joint").mkdir()
            (run/"joint/history.jsonl").write_text('{}\n')
            env=dict(os.environ,BABEL_LARGE_RUN=str(run),BABEL_DOWNLOAD_DIR=str(root/"download"))
            subprocess.run(["bash","scripts/babel_large_autodl.sh","export"],env=env,check=True,capture_output=True)
            self.assertTrue((root/"download/large/joint_history.jsonl").exists())
            bars=[[100,101,99,100],[100,102,99,101]]
            write_global_review(root/"review.html",[dict(source="fixture",end="2025",blocks=4,truth_ohlc=bars,reconstructed_ohlc=bars)])
            self.assertIn('<polyline',(root/"review.html").read_text())
