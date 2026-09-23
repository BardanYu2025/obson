"""Frozen per-contract rolling128 endpoint states; no neural hidden-state reuse."""
import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import architecture as ar, ae_context, activity_ablation as aa, ar_codec, data
from . import endpoint_readout as er, endpoint_readout_audit as ea, streaming
from .dual_state import sha256, verify_files
from .holdout_audit import read_json

SCHEMA = 'babel-window-state-v1'
WARMUP = 512
FEATURES = tuple(ae_context.feature_names('ema8_32'))+tuple(aa.ACTIVITY)


@dataclass(frozen=True)
class MarketBar(streaming.ClosedBar):
    open_oi: float | None = None


def code_identity():
    modules = (streaming, ae_context, aa, ar_codec, data)
    return ea.code_identity() | {Path(m.__file__).name: sha256(m.__file__) for m in modules} | {p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('window_state_cli.py'))}


def check_key(key, period):
    if type(period) is not int or period <= 0 or not isinstance(key, str): raise ValueError('Explicit contract/period required')
    parts = key.split('/')
    if len(parts) != 3 or not all(parts) or parts[1] != str(period): raise ValueError('Key must be symbol/period/contract')


class Features:
    """Incremental equivalent of the exact28 training channels, with causal EMA carry."""
    def __init__(self):
        self.price = streaming.CausalFeatures()
        self.log_volume = self.volume_ema = None
        self.oi = 0.; self.oi_valid = False

    def advance(self, bar):
        old = self.price.last_time
        price = copy.deepcopy(self.price)
        x, anchor, time = price.advance(bar)
        lv = float(np.log1p(bar.volume)); first = self.log_volume is None
        prior = lv if first else self.volume_ema
        opening = float(bar.open_oi) if bar.open_oi is not None else float('nan')
        valid_oi = bool(bar.oi_available and bar.oi > 0)
        use_open = bool(valid_oi and np.isfinite(opening) and opening > 0)
        consecutive = old is not None and (time-old).total_seconds() == bar.period*60
        use_previous = not use_open and valid_oi and self.oi_valid and consecutive
        valid = use_open or use_previous; base = (opening if use_open else self.oi) if valid else 1.
        delta = bar.oi-base if valid else 0.; ratio = valid and bar.volume > 0
        activity = np.array([lv-prior, 0. if first else lv-self.log_volume, np.arcsinh(100*delta/base),
            np.arcsinh(delta/max(bar.volume,1.)) if ratio else 0., np.arcsinh(100*bar.volume/base) if valid else 0.,
            valid, ratio, valid, use_open, not first], dtype=np.float32)
        result = np.r_[x,activity].astype(np.float32)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite state features')
        self.price = price; self.log_volume = lv
        self.volume_ema = lv if first else (31/33)*prior+(2/33)*lv
        self.oi = float(bar.oi); self.oi_valid = valid_oi
        return result, anchor, time

    def snapshot(self):
        return dict(price=self.price.snapshot(), log_volume=self.log_volume, volume_ema=self.volume_ema,
                    oi=self.oi, oi_valid=self.oi_valid)

    @classmethod
    def restore(cls, value):
        obj = cls(); obj.price = streaming.CausalFeatures.restore(value['price'])
        if obj.price.row >= 0 and (not np.isfinite([value['log_volume'],value['volume_ema'],value['oi']]).all()
                                   or min(value['log_volume'],value['volume_ema'],value['oi']) < 0):
            raise ValueError('Invalid feature snapshot')
        obj.log_volume = value['log_volume']; obj.volume_ema = value['volume_ema']
        obj.oi = value['oi']; obj.oi_valid = value['oi_valid']
        return obj


def frame_bars(frame, key, period):
    check_key(key,period)
    for column, expected in [('key',key),('contract',key.split('/')[2])]:
        if column in frame and not (frame[column].astype(str)==expected).all(): raise ValueError('Mixed contract file')
    for _, row in frame.iterrows():
        opening = pd.to_numeric(row.get('open_oi',np.nan),errors='coerce')
        yield MarketBar(key,period,row.datetime,float(row.open),float(row.high),float(row.low),float(row.close),
                        float(row.volume),float(row.oi),bool(row.oi_available),True,
                        float(opening) if np.isfinite(opening) else None)


class WindowState:
    """One contract per instance; every ready call recomputes a fresh128-row window.

    Features carry causal pre-window history. Emit only after512 observed bars;
    no automatic main-contract selection, roll stitching, or neural KV reuse.
    """
    def __init__(self, model, statistics, local, identity, key, period, device='cpu'):
        check_key(key,period)
        self.model = model.to(device).eval().requires_grad_(False); self.device = torch.device(device)
        self.statistics = copy.deepcopy(statistics); self.local = copy.deepcopy(local); self.identity = copy.deepcopy(identity)
        for name, width in [('x_mean',28),('x_scale',28),('y_mean',7),('y_scale',7)]:
            value = np.asarray(statistics[name])
            if value.shape != (width,) or not np.isfinite(value).all() or (name.endswith('scale') and (value<=0).any()):
                raise ValueError('Invalid pinned state normalization')
        for name in ('mean','scale'):
            value=np.asarray(local[name])
            if value.shape!=(7,) or not np.isfinite(value).all() or (name=='scale' and (value<=0).any()):
                raise ValueError('Invalid pinned recent normalization')
        self.reset_contract(key,period)

    def reset_contract(self, key, period):
        check_key(key,period)
        if 'supported_periods' in self.identity and period not in self.identity['supported_periods']:
            raise ValueError('Period outside the pinned training source')
        self.key, self.period = key,period
        self.features = Features(); self.rows = deque(maxlen=128); self.count = 0
        self.origin = None

    def _metadata(self, rows, count):
        last = rows[-1] if rows else None
        return dict(schema=SCHEMA, model=copy.deepcopy(self.identity), key=self.key, period=self.period,
            bar_start=None if last is None else last['time'],
            available_at=None if last is None else str(streaming.timestamp(last['time'])+pd.Timedelta(minutes=self.period)),
            timezone='Asia/Shanghai', observed_bars=count, required_warmup=WARMUP, remaining_warmup=max(0,WARMUP-count),
            ready=count>=WARMUP, window=128, feature_context='Causal EMA/volatility since this contract stream origin',
            stream_origin=self.origin if self.origin is not None else (rows[0]['time'] if rows else None),
            window_start=rows[0]['time'] if len(rows)==128 else None,
            state_semantics='Final state of a fresh rolling128 window; includes the current completed bar',
            quality_scope='Frozen baseline. Original research used selected endpoints; all rolling bars are not newly quality-validated.',
            automatic_contract_selection=False, neural_cache_reused=False, forecast=False)

    @torch.inference_mode()
    def _result(self, rows, count):
        metadata = self._metadata(rows,count)
        if count < WARMUP: return dict(metadata=metadata,embedding=None,history=None,recent=None)
        raw = np.stack([row['x'] for row in rows])
        normalized = ((raw-np.asarray(self.statistics['x_mean']))/np.asarray(self.statistics['x_scale'])).astype(np.float32)
        z = self.model.encoder(torch.tensor(normalized[None],device=self.device))[:, -1]
        pred = self.model.decoder(z)
        recent_normalized = er.crop_prediction(pred,self.statistics,self.local)[0].cpu().numpy()
        values = pred[0].cpu().numpy()*np.asarray(self.statistics['y_scale'])+np.asarray(self.statistics['y_mean'])
        recent = recent_normalized*np.asarray(self.local['scale'])+np.asarray(self.local['mean'])
        _,mask = ar.ordered_targets(raw[None]); mask = mask[0]
        anchor = rows[0]['anchor']
        with np.errstate(over='raise',invalid='raise'):
            try: prices = anchor*np.exp(values[:,0]/100)
            except FloatingPointError as exc: raise ValueError('Nonfinite decoded prices') from exc
        if not np.isfinite(values).all() or not np.isfinite(recent).all() or not torch.isfinite(z).all():
            raise ValueError('Nonfinite decoded state')
        times = [row['time'] for row in rows]
        history = dict(channels=list(ar.NAMES), values=values[:127].tolist(), valid_mask=mask[:127].tolist(),
            bar_starts=times[:127], price_anchor=float(anchor), anchor_source='Observed close before the128-row input window',
            close_prices=prices[:127].tolist(), current_bar_included=False)
        # Do not recompute a recent anchor from observed prices: retain the path
        # confirmed by endpoint_readout, even when absolute display prices are used.
        recent_prices = prices[110]*np.exp(recent[:,0]/100)
        if not np.isfinite(recent_prices).all(): raise ValueError('Nonfinite recent prices')
        return dict(metadata=metadata, embedding=z[0].cpu().tolist(), history=history,
            recent=dict(channels=list(ar.NAMES),values=recent.tolist(),valid_mask=mask[111:127].tolist(),
                bar_starts=times[111:127],price_anchor=float(prices[110]),anchor_source='Predicted global close at input bar111',
                close_prices=recent_prices.tolist(),current_bar_included=False))

    def _advance(self, bar, as_of, emit):
        if bar.key != self.key or bar.period != self.period: raise ValueError('Contract/period changed; explicitly reset_contract')
        check_key(bar.key,bar.period)
        time = streaming.timestamp(bar.datetime); observed_at = streaming.timestamp(as_of)
        if not bar.closed or time+pd.Timedelta(minutes=self.period)>observed_at: raise ValueError('Bar has not closed at as_of')
        candidate = copy.deepcopy(self.features)
        x,anchor,time = candidate.advance(bar)
        rows = deque(self.rows,maxlen=128); rows.append(dict(x=x,time=str(time),anchor=float(anchor)))
        result = self._result(rows,self.count+1) if emit else dict(metadata=self._metadata(rows,self.count+1))
        if self.count==0: self.origin=str(time)
        result['metadata']['stream_origin']=self.origin
        self.features,self.rows,self.count = candidate,rows,self.count+1
        return result

    def push(self, bar, *, as_of):
        return self._advance(bar,as_of,True)

    def warm(self, bar, *, as_of):
        """Consume closed history without neural computation; no embedding returned."""
        return self._advance(bar,as_of,False)['metadata']

    def current(self): return self._result(self.rows,self.count)

    def snapshot(self):
        return dict(schema=SCHEMA,identity=copy.deepcopy(self.identity),key=self.key,period=self.period,count=self.count,
            origin=None if self.count==0 else self.origin,features=self.features.snapshot(),
            rows=[dict(x=r['x'].tolist(),time=r['time'],anchor=r['anchor']) for r in self.rows])

    def restore(self, snapshot):
        if (snapshot['schema']!=SCHEMA or snapshot['identity']!=self.identity or snapshot['key']!=self.key
                or snapshot['period']!=self.period): raise ValueError('Snapshot model/contract identity mismatch')
        count=snapshot['count']; rows=snapshot['rows']; features=Features.restore(snapshot['features'])
        if type(count) is not int or count<0 or len(rows)!=min(count,128) or features.price.row!=count-1:
            raise ValueError('Invalid snapshot history length')
        if count==0 and (snapshot['features']!=Features().snapshot() or snapshot['origin'] is not None):
            raise ValueError('Nonempty feature state for empty snapshot')
        previous=None
        for row in rows:
            x=np.asarray(row['x'],dtype=np.float32); time=streaming.timestamp(row['time'])
            if (x.shape!=(28,) or not np.isfinite(x).all() or not np.isfinite(row['anchor']) or row['anchor']<=0
                    or (previous is not None and (time-previous).total_seconds()<self.period*60)):
                raise ValueError('Invalid snapshot row')
            previous=time
        if count and (previous!=features.price.last_time or streaming.timestamp(snapshot['origin'])>streaming.timestamp(rows[0]['time'])):
            raise ValueError('Snapshot time mismatch')
        restored=deque([dict(x=np.asarray(v['x'],np.float32),time=v['time'],anchor=v['anchor']) for v in rows],maxlen=128)
        self.features,self.rows,self.count=features,restored,count
        self.origin=snapshot['origin']


def load_bundle(bundle, seed, key, period, device='cpu', *, _audit=False):
    if torch.device(device).type=='cuda':
        ea.bb.ab.configure_runtime()
    bundle=Path(bundle); index=read_json(bundle/'index.json')
    if index['schema']!=SCHEMA or index['code_sha256']!=code_identity(): raise ValueError('Bundle implementation differs')
    verify_files(bundle,index['files'])
    if not _audit:
        certificate=read_json(bundle/'validation.json')
        if certificate['status']!='passed' or certificate['index_sha256']!=sha256(bundle/'index.json'):
            raise ValueError('Unvalidated state bundle')
    if str(seed) not in index['models']: raise ValueError('Choose retained seed42 or43; no automatic ensemble')
    name=index['models'][str(seed)]; ck=torch.load(bundle/name,map_location='cpu',weights_only=True)
    if ck['schema']!=SCHEMA or ck['seed']!=seed: raise ValueError('Invalid bundle model')
    state=ck['model']; pca=dict(components=state['decoder.basis'].numpy(),mean=state['decoder.target_mean'].numpy())
    scales=dict(mean=state['decoder.coordinate_mean'].tolist(),scale=state['decoder.coordinate_scale'].tolist())
    model=ea.bb.pt.Student(ck['config'],seed,pca,scales,residual=False);model.load_state_dict(state)
    identity=dict(bundle_index=sha256(bundle/'index.json'),model_sha256=index['files'][name],**ck['identity'])
    return WindowState(model,ck['statistics'],ck['local'],identity,key,period,device)
