"""Function-preserving FFN widening and identity depth extension; fixed768 states."""
import math
import torch
from torch import nn

VARIANTS=(('base',2,512),('wide',2,3072),('deep',4,3072))


def install(model,layers,ff,seed):
    backbone=model.core.encoder.backbone;old=list(backbone.layers)
    if len(old)!=2 or layers not in (2,4) or ff<old[0].linear1.out_features:
        raise ValueError('Expected two-layer parent and noncontracting FFN')
    width=old[0].self_attn.embed_dim;device=old[0].linear1.weight.device;dtype=old[0].linear1.weight.dtype
    if any(not layer.norm_first or layer.self_attn.num_heads!=8 or layer.dropout.p!=0 for layer in old):
        raise ValueError('Identity expansion requires original prenorm8-head dropout0 parent')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+20260925)
        for layer in old:
            n=layer.linear1.out_features
            if ff==n:continue
            first=nn.Linear(width,ff).to(device=device,dtype=dtype);second=nn.Linear(ff,width).to(device=device,dtype=dtype)
            with torch.no_grad():
                first.weight[:n].copy_(layer.linear1.weight);first.bias[:n].copy_(layer.linear1.bias)
                second.weight.zero_();second.weight[:,:n].copy_(layer.linear2.weight);second.bias.copy_(layer.linear2.bias)
            layer.linear1=first;layer.linear2=second
        for _ in range(layers-len(old)):
            layer=nn.TransformerEncoderLayer(width,8,ff,dropout=0.,activation='gelu',batch_first=True,norm_first=True).to(device=device,dtype=dtype)
            with torch.no_grad():
                layer.self_attn.out_proj.weight.zero_();layer.self_attn.out_proj.bias.zero_()
                layer.linear2.weight.zero_();layer.linear2.bias.zero_()
            old.append(layer)
    backbone.layers=nn.ModuleList(old)
    model.core.encoder.requires_grad_(True);model.local_head.requires_grad_(True)
    model.core.decoder.requires_grad_(False)
    return dict(width=width,layers=layers,heads=8,ff=ff,encoder_parameters=sum(p.numel() for p in model.core.encoder.parameters()),
                local_parameters=sum(p.numel() for p in model.local_head.parameters()),decoder_trainable_parameters=sum(p.numel() for p in model.core.decoder.parameters() if p.requires_grad))


def learning_rate(epoch,epochs,peak,warmup=5):
    if not 1<=epoch<=epochs or epochs<=warmup or peak<=0:raise ValueError('Invalid fixed warmup/cosine schedule')
    if epoch<=warmup:return peak*epoch/warmup
    return peak*(.1+.9*(1+math.cos(math.pi*(epoch-warmup)/(epochs-warmup)))/2)


def compare(a,b):
    diff=(a-b).abs();tol=5e-5+2e-4*b.abs()
    return dict(passed=bool((diff<=tol).all()),max_abs=float(diff.max()),max_tolerance_ratio=float((diff/tol).max()),atol=5e-5,rtol=2e-4)
