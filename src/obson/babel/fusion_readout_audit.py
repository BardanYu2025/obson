"""Re-read cached fusion vectors with the same train-standardized ridge probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ae_diagnostics import fingerprint
from .ae_extend import atomic_json
from .autoregressive import probe
from .progress import progress
from .residual_fusion import MODES, ResidualFusion


def load_cache(source,split,sources):
    path=source/'feature_cache'/f'{split}.npz'
    index=json.loads(path.with_suffix('.json').read_text())
    if index['identity']!=sources or index['sha256']!=fingerprint(path):raise ValueError('Feature cache fingerprint mismatch')
    with np.load(path,allow_pickle=False) as file:data={k:file[k] for k in file.files}
    n=len(data['labels'])
    if data['long'].shape!=(n,512) or data['short'].shape!=(n,512) or data['keys'].shape!=(n,2):raise ValueError('Invalid cache shape')
    if any(not np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite cached values')
    if not np.isin(data['labels'],np.arange(4)).all():raise ValueError('Invalid state labels')
    return data,index['sha256']


def geometry(x):
    std=x.std(0);total=float(np.square(x).sum(1).mean())
    return dict(std_quantiles=np.quantile(std,[0,.1,.5,.9,1]).tolist(),fraction_std_below_001=float((std<.01).mean()),
                common_mean_energy_fraction=float(np.square(x.mean(0)).sum()/max(total,1e-12)))


@torch.no_grad()
def vectors(model,data,batch=512):
    rows=[];model.eval()
    for start in range(0,len(data['labels']),batch):
        m=torch.from_numpy(data['long'][start:start+batch]);s=torch.from_numpy(data['short'][start:start+batch])
        rows.append(model(m,s)['z'].numpy())
    return np.concatenate(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'manifest.json').read_text());artifacts=json.loads((source/'artifacts.json').read_text())
    if meta['sources']!=artifacts['sources']:raise ValueError('Source identity mismatch')
    original=json.loads((source/'fusion_metrics.json').read_text());data=[];cache_hashes={}
    for split in ('train','val','test'):
        values,sha=load_cache(source,split,meta['sources']);data.append(values);cache_hashes[split]=sha
    result=dict(schema='babel-fusion-readout-audit-v1',source_manifest=fingerprint(source/'manifest.json'),cache_hashes=cache_hashes,
                method='Existing balanced ridge probe; per-coordinate mean/std from train only, std floor .01, alpha in [1,10,100] selected by validation BA. No neural updates or checkpoint reselection.',
                interpretation='Same cached endpoints for all modes. Concatenation is a 1024-dimensional information-retention reference, not an equal-size bottleneck. Previously seen research test period; no new holdout claims.',variants={})
    for mode in (*MODES,'concat'):
        if mode=='concat':zs=[np.concatenate((d['long'],d['short']),axis=1) for d in data];epoch=None;sha=None
        else:
            file=source/mode/'best.pt';sha=fingerprint(file)
            if sha!=artifacts['checkpoints'][mode]:raise ValueError('Fusion checkpoint changed')
            ck=torch.load(file,map_location='cpu',weights_only=True);epoch=ck['epoch']
            if epoch!=original['variants'][mode]['selected_epoch']:raise ValueError('Selected epoch changed')
            model=ResidualFusion(mode);model.load_state_dict(ck['model']);zs=[vectors(model,d) for d in data]
        evaluation=probe(*(item for z,d in zip(zs,data) for item in (z,d['labels'])),4)
        result['variants'][mode]=dict(dimensions=zs[0].shape[1],selected_epoch=epoch,checkpoint_sha256=sha,ridge=evaluation,
            training_geometry=geometry(zs[0]),original_head=original['variants'][mode]['classification'] if mode!='concat' else None)
        atomic_json(result,out/'readout_audit.json');progress(f'Readout audit {mode}: val BA={evaluation["validation_ba"]:.4f}, test BA={evaluation["test"]["ba"]:.4f}')
    progress(f'Readout audit complete: {out}')


if __name__=='__main__':main()
