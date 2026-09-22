"""Read-only localization of two predeclared extreme historical paths; never trim data."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import architecture as ar, architecture_benchmark as ab
from .data import validate_frame
from .dual_state import sha256
from .holdout_audit import read_json
from .ae_extend import atomic_json

CASES = (('test','ag/60/SHFE.ag2606','2026-02-11 14:00:00'),
         ('cross_research','au/60/SHFE.au2606','2026-02-25 00:00:00'))


def raw_window(path, end, window=128):
    frame=validate_frame(pd.read_csv(path),str(path))
    ids=np.flatnonzero(frame.datetime.eq(pd.Timestamp(end)).to_numpy())
    if len(ids)!=1 or ids[0]<window: raise ValueError('Endpoint/preceding anchor unavailable in raw snapshot')
    row=int(ids[0]);start=row-window+1
    w=frame.iloc[start:row+1];anchor=float(frame.close.iloc[start-1])
    o=w.open.to_numpy(float);c=w.close.to_numpy(float);prev=np.r_[anchor,c[:-1]]
    if min(o.min(),c.min(),anchor)<=0: raise ValueError('Nonpositive raw price')
    geometry=100*np.column_stack((np.log(o/prev),np.log(c/o)))
    target=np.column_stack((100*np.log(c/anchor),100*np.log(c/o)))
    replay=np.sinh(np.arcsinh(geometry).astype(np.float32).astype(float))
    reconstructed=np.column_stack((replay.sum(-1).cumsum(),replay[:,1]))
    # Last row is input-only, excluded from reconstruction scores in this benchmark.
    diff=reconstructed[:-1]-target[:-1]
    if not np.allclose(reconstructed[:-1],target[:-1],atol=2e-4,rtol=2e-5):
        raise ValueError('Raw log-price/asinh round trip failed')
    step=100*np.log(c/prev);order=np.argsort(-np.abs(step[:-1]),kind='stable')[:5]
    gaps=100*np.log(o/prev)
    report=dict(path=str(Path(path).resolve()),sha256=sha256(path),row=row,start=str(w.datetime.iloc[0]),end=end,
        prewindow_anchor=anchor,current_close=float(c[-1]),scored_bars=127,
        raw_log_path_min=float(target[:-1,0].min()),raw_log_path_max=float(target[:-1,0].max()),
        raw_close_min=float(c[:-1].min()),raw_close_max=float(c[:-1].max()),
        codec_roundtrip_max_abs=float(np.abs(diff).max()),
        largest_scored_changes=[dict(datetime=str(w.datetime.iloc[i]),close=float(c[i]),previous_close=float(prev[i]),
            change_log_percent=float(step[i]),gap_log_percent=float(gaps[i]),volume=float(w.volume.iloc[i])) for i in order])
    return report,geometry,target


def audit(source, root, out):
    stats=read_json(source/'cache/statistics.json');results=[]
    for split,key,end in CASES:
        inv=read_json(source/f'cache/{split}_inventory.json')
        matches=[i for i,r in enumerate(inv) if r['key']==key and r['end']==end]
        item=dict(dataset=split,key=key,end=end)
        if len(matches)!=1:
            results.append(dict(item,status='endpoint_unavailable'));continue
        idx=matches[0];data=ab.load_arrays(source,split)
        x=np.asarray(data['x'][idx:idx+1])*np.array(stats['x_scale'])+np.array(stats['x_mean'])
        target,mask=ar.ordered_targets(x)
        stored=np.asarray(data['y'][idx])*np.array(stats['y_scale'])+np.array(stats['y_mean'])
        valid=np.asarray(data['mask'][idx]);delta=target[0][valid]-stored[valid]
        # Normalization round trip includes FP32 rounding, especially accumulated prices.
        cached_ok=bool(np.array_equal(mask[0],valid) and np.allclose(target[0][valid],stored[valid],atol=2e-4,rtol=2e-5))
        item.update(index=idx,cached_target_replay_passed=cached_ok,cached_target_max_abs=float(np.abs(delta).max()))
        if not cached_ok:
            atomic_json(dict(status='failed',cases=results+[dict(item,status='cached_target_mismatch')],
                policy='No target changes or exclusions'),out/'path_audit.json')
            raise ValueError(f'Cached target construction mismatch: {item}')
        symbol,period,contract=key.split('/');path=root/symbol/f'{contract}_{period}m.csv'
        if not path.exists():
            results.append(dict(item,status='raw_file_unavailable',path=str(path)));continue
        try: raw,geometry,raw_target=raw_window(path,end)
        except ValueError as exc:
            results.append(dict(item,status='raw_snapshot_invalid',reason=str(exc),sha256=sha256(path)));continue
        feature_diff=np.sinh(x[0,:,:2])-geometry
        target_diff=stored[:-1,:2]-raw_target[:-1]
        matched=bool(np.allclose(feature_diff,0,atol=2e-4,rtol=0) and np.allclose(target_diff,0,atol=2e-4,rtol=0))
        results.append(dict(item,status='matched' if matched else 'raw_cache_mismatch',raw=raw,
            feature_max_abs=float(np.abs(feature_diff).max()),price_target_max_abs=float(np.abs(target_diff).max()),
            largest_cached_log_path=float(np.abs(stored[:-1,0]).max())))
    report=dict(cases=results,selection='Two endpoints fixed before new training, identified from prior path-error report.',
        scope='Raw OHLCV geometry, log/asinh telescoping and cache-target consistency; not exchange-tick verification.',
        policy='No cache changes, no excluded windows, no new ranking. Missing/different raw snapshot remains explicit; cannot infer data error or authenticity from price magnitude.')
    atomic_json(report,out/'path_audit.json');return report
