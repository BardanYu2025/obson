"""Read-only checkpoint comparison; evaluation batch sizes do not alter training manifests."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .ae_context import encode_context
from .ae_diagnostics import fingerprint
from .ae_extend import atomic_json
from .data import load_series, manifest
from .history_autoencoder import HistoryWindows, reconstruction_loss
from .large_history import LargeHistory, HierWindows, teacher_banks, evaluate_large, move
from .progress import progress


def load_data(root, source):
    ref=json.loads((Path(source)/'manifest.json').read_text())['manifest']
    keys=[s['key'].split('/') for s in ref['sources']]
    series,_=load_series(root,sorted({k[0] for k in keys}),sorted({int(k[1]) for k in keys}))
    if manifest(series,ref['boundaries'])!=ref:raise ValueError('Data fingerprint mismatch')
    encoded=[encode_context(s.frame,s.period,'ema8_32') for s in series]
    sets=[HistoryWindows(series,encoded,ref['boundaries'],s) for s in ('train','val','test')]
    return ref,series,encoded,sets


@torch.no_grad()
def components(model,ds,mean,scale,batch,device='cuda'):
    model.eval();totals={};count=0
    for data in DataLoader(ds,batch_size=batch):
        b=move(data,device);r=model(b,joint=True);valid=b['valid']
        direct=model.local.decode(r['local_z'][valid])
        target=b['y'][valid].clone();target[...,:2]-=b['offsets'][valid][:,None,None]
        # Per-window averages, invariant to padding count and evaluation batch size.
        local_loss=reconstruction_loss(direct,target,b['mask'][valid])
        global_loss=reconstruction_loss(r['reconstruction'][valid],b['y'][valid],b['mask'][valid])
        start=0
        for j in range(len(valid)):
            n=int(valid[j].sum());sl=slice(start,start+n);v=valid[j]
            values=dict(global_reconstruction=global_loss[sl].mean(),local_reconstruction=local_loss[sl].mean(),
                teacher_latent=torch.nn.functional.smooth_l1_loss(r['latents'][j,v],b['teacher'][j,v]),
                anchor=torch.nn.functional.smooth_l1_loss(r['anchors'][j,v],b['offsets'][j,v]),
                structure=((r['structure'][j]-(b['targets'][j]-mean)/scale)**2).mean())
            for name,value in values.items():totals[name]=totals.get(name,0.)+float(value)
            count+=1;start+=n
    return dict(windows=count,means={k:v/count for k,v in totals.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--local-batch',type=int,default=64)
    a=p.parse_args()
    if not torch.cuda.is_available() or min(a.batch,a.local_batch)<1:raise ValueError('CUDA and positive batches required')
    source=Path(a.source);out=Path(a.out)
    if out.resolve()==source.resolve():raise ValueError('Use a separate report directory')
    out.mkdir(parents=True,exist_ok=True)
    hashes={s:fingerprint(source/s/'best.pt') for s in ('local','aggregate','joint')}
    ref,series,encoded,sets=load_data(a.root,source)
    model=LargeHistory().cuda()
    ck=torch.load(source/'local/best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
    banks=teacher_banks(model,series,encoded,out/'teacher_cache',hashes['local'],'cuda',a.local_batch)
    hier=[HierWindows(ds,banks,ref['boundaries'],s) for ds,s in zip(sets,('train','val','test'))]
    summary=dict(schema='babel-stage-audit-v1',source_hashes=hashes,batch=a.batch,local_batch=a.local_batch,stages={},
                 interpretation='No optimizer updates. Aggregate local module equals stage-1 local module. Components use per-window means; not identical to original variable-block micro-batch loss weighting.')
    for stage in ('aggregate','joint'):
        ck=torch.load(source/stage/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
        if stage=='aggregate':
            original=torch.load(source/'local/best.pt',map_location='cpu',weights_only=True)['model']
            if any(not torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items() if k.startswith('local.')):
                raise ValueError('Aggregate local module differs from stage-1 checkpoint')
        mean=torch.tensor(ck['metadata']['target_mean'],device='cuda');scale=torch.tensor(ck['metadata']['target_scale'],device='cuda')
        dest=out/stage;dest.mkdir(exist_ok=True)
        evaluate_large(model,sets,hier,dest,mean,scale,a.batch,local_micro=a.local_batch)
        summary['stages'][stage]=dict(selected_epoch=ck['epoch'],components={s:components(model,ds,mean,scale,a.batch) for s,ds in zip(('val','test'),hier[1:])})
        atomic_json(summary,out/'stage_comparison.json');progress(f'Stage audit complete: {stage}')
    if hashes!={s:fingerprint(source/s/'best.pt') for s in hashes}:raise ValueError('Source checkpoints changed during audit')
    progress(f'Stage audit complete: {out}')


if __name__=='__main__':main()
