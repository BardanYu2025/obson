"""Causal dual-state inference using frozen specialist encoders and decoders."""
import argparse
import copy
import json
import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .ae_extend import atomic_save
from .ar_codec import State, consume
from .dual_state import sha256
from .fusion_readout_audit import load_cache
from .large_history import LargeHistory
from .reconstruction_fusion import ReconstructionFusion
from .short_state import ShortState

SCHEMA = 'babel-causal-dual-stream-v1'


def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError('Timestamp must be valid')
    if t.tzinfo is not None:
        t = t.tz_convert('Asia/Shanghai').tz_localize(None)
    return t


@dataclass(frozen=True)
class ClosedBar:
    key: str
    period: int
    datetime: object
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float = 0.
    oi_available: bool = False
    closed: bool = True


class CausalFeatures:
    """Incremental equivalent of encode_context(..., 'ema8_32')."""
    def __init__(self):
        self.codec = None
        self.fast = self.slow = self.last_time = None
        self.row = -1
        self.last_x = None

    def advance(self, bar):
        if not bar.closed:
            raise ValueError('Only completed bars may be consumed')
        if not isinstance(bar.period, int) or bar.period <= 0:
            raise ValueError('Positive integer period in minutes required')
        t = timestamp(bar.datetime)
        elapsed = 1. if self.last_time is None else (t - self.last_time).total_seconds() / (bar.period * 60)
        if not math.isfinite(elapsed) or elapsed < 1.:
            raise ValueError('Duplicate, out-of-order, or sub-period bar')
        # Work on a copy so invalid bars do not partly modify feature state.
        codec = copy.copy(self.codec) if self.codec else State(float(bar.open))
        anchor, sigma = codec.close, codec.sigma
        x, _, _ = consume(codec, [bar.open, bar.high, bar.low, bar.close], bar.volume,
                          bar.oi, bar.oi_available, bar.period, elapsed)
        price = np.log(bar.close)
        fast = price if self.fast is None else self.fast + (2/9) * (price-self.fast)
        slow = price if self.slow is None else self.slow + (2/33) * (price-self.slow)
        context = np.arcsinh(np.array([price-fast, fast-slow,
            0. if self.fast is None else fast-self.fast, 0. if self.slow is None else slow-self.slow]) / sigma)
        result = np.r_[x, context].astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite streaming features')
        self.codec, self.fast, self.slow, self.last_time = codec, float(fast), float(slow), t
        self.row += 1
        self.last_x = result.copy()
        return result, float(anchor), t

    def snapshot(self):
        return dict(row=self.row, close=None if self.codec is None else self.codec.close,
                    variance=None if self.codec is None else self.codec.variance,
                    fast=self.fast, slow=self.slow, last_time=None if self.last_time is None else str(self.last_time))

    @classmethod
    def restore(cls, value):
        obj = cls(); obj.row = int(value['row'])
        if obj.row >= 0:
            fields = [value[k] for k in ('close','variance','fast','slow')]
            if not np.isfinite(fields).all() or fields[0] <= 0 or fields[1] < 0:
                raise ValueError('Invalid saved feature state')
            obj.codec = State(float(value['close']), float(value['variance']))
            obj.fast, obj.slow = float(value['fast']), float(value['slow'])
            obj.last_time = timestamp(value['last_time'])
        return obj


class DualStream:
    """One contract/period, one neural partition; original two states remain separate.

    The standardized 1024-vector is optional until four complete global-grid blocks exist.
    Consumers must retain metadata: the long half may precede the short half by up to127 bars.
    """
    def __init__(self, long_model, short_model, recent_decoder, statistics, target_mean, target_scale,
                 key, period, identity, device='cpu'):
        self.device = torch.device(device)
        self.long_model = long_model.to(self.device).eval().requires_grad_(False)
        self.short_model = short_model.to(self.device).eval().requires_grad_(False)
        self.recent_decoder = recent_decoder.to(self.device).eval().requires_grad_(False)
        self.width = self.long_model.config['latent']; self.window = self.long_model.config['window']
        if self.window != 128 or self.short_model.width != self.width:
            raise ValueError('Streaming interface expects aligned128-bar source models')
        self.statistics = {k: torch.tensor(statistics[k],dtype=torch.float32,device=self.device)
                           for k in ('long_mean','long_scale','short_mean','short_scale')}
        for k,v in self.statistics.items():
            if v.shape != (self.width,) or not torch.isfinite(v).all() or (k.endswith('scale') and not (v>0).all()):
                raise ValueError('Invalid training normalization')
        self.target_mean = torch.tensor(target_mean,dtype=torch.float32,device=self.device)
        self.target_scale = torch.tensor(target_scale,dtype=torch.float32,device=self.device)
        self.identity = identity
        self.reset_contract(key, period)

    def reset_contract(self, key, period):
        if not isinstance(key,str) or not key or not isinstance(period,int) or period<=0:
            raise ValueError('Explicit contract key and positive integer period required')
        self.key, self.period = key, period
        self.features = CausalFeatures()
        self.start_partition('live')

    def start_partition(self, name):
        """Reset neural memory only, preserving past-only codec/EMA state and absolute bar grid.

        Used to reproduce train/val/test resets. Not a session/night-gap reset.
        """
        self.partition = str(name); self.partition_start = self.features.row + 1; self.count = 0
        self.hidden = self.short = self.long = None
        self.block_x = []; self.block_anchor = None
        self.blocks = deque(maxlen=self.long_model.blocks)
        self.closes = deque(maxlen=65)
        if self.features.codec is not None:
            self.closes.append(self.features.codec.close)
        self.long_row = self.long_time = self.long_anchor = None

    def _check(self, bar):
        if bar.key != self.key or bar.period != self.period:
            raise ValueError('Contract/period changed; explicitly reset_contract or use a separate stream')

    def warm_features(self, bar):
        """Replay a historical prefix for codec/EMA only; then call start_partition.

        No neural output is produced. Cannot warm after neural inference has begun.
        """
        self._check(bar)
        if self.count:
            raise ValueError('Feature warmup requires a fresh neural partition')
        x, _, _ = self.features.advance(bar)
        return x

    @torch.no_grad()
    def push(self, bar):
        self._check(bar)
        self.long_model.eval(); self.short_model.eval(); self.recent_decoder.eval()
        x, previous_close, t = self.features.advance(bar)
        row = self.features.row
        if self.count == 0:
            self.partition_start = row
            self.closes.clear(); self.closes.append(previous_close)
        self.closes.append(float(bar.close)); self.count += 1
        values, self.hidden = self.short_model(torch.tensor(x,device=self.device)[None,None],self.hidden)
        self.short = values[:, -1].clone()
        if row % self.window == 0:
            self.block_x = []; self.block_anchor = previous_close
        # Initial partial absolute-grid blocks are excluded, matching HierWindows.
        if self.block_anchor is not None:
            self.block_x.append(torch.tensor(x,device=self.device))
        refreshed = False
        if row % self.window == self.window-1 and len(self.block_x) == self.window:
            latent = self.long_model.local.encode(torch.stack(self.block_x)[None])[:, -1]
            self.blocks.append(dict(z=latent.clone(), anchor=self.block_anchor, row=row, time=str(t)))
            self.block_x = []; self.block_anchor = None
            if len(self.blocks) >= 4:
                d, k, n = self.width, self.long_model.blocks, len(self.blocks)
                local = torch.zeros(1,k,d,device=self.device); offsets = torch.zeros(1,k,device=self.device)
                valid = torch.zeros(1,k,dtype=torch.bool,device=self.device)
                current_anchor = self.blocks[-1]['anchor']
                for j,block in enumerate(self.blocks,k-n):
                    local[:,j] = block['z']; offsets[:,j] = 100*(np.log(block['anchor'])-np.log(current_anchor)); valid[:,j] = True
                self.long = self.long_model.summarize(local,offsets,valid).clone()
                self.long_row, self.long_time, self.long_anchor = row, str(t), current_anchor
                refreshed = True
        return dict(**self.current(), long_refreshed=refreshed)

    def metadata(self):
        return dict(schema=SCHEMA, key=self.key, period=self.period, partition=self.partition,
                    row=self.features.row, bars_since_neural_reset=self.count,
                    as_of=None if self.features.last_time is None else str(self.features.last_time),
                    available_at=None if self.features.last_time is None else str(self.features.last_time+pd.Timedelta(minutes=self.period)),
                    short_ready=self.count>=128, long_ready=self.long is not None,
                    long_as_of=self.long_time, long_row=self.long_row,
                    long_age_bars=None if self.long_row is None else self.features.row-self.long_row,
                    completed_blocks=len(self.blocks), dimensions=self.width*2,
                    recent_anchor=float(self.closes[0]) if self.count>=128 else None,
                    history_anchor=self.long_anchor,
                    quality_scope='Endpoint quality verified on complete128-bar grid; between-grid quality not yet validated. Long history/structure readouts retain their own older as_of.')

    @torch.no_grad()
    def current(self):
        embedding = None
        if self.long is not None:
            m = (self.long-self.statistics['long_mean'])/self.statistics['long_scale']
            s = (self.short-self.statistics['short_mean'])/self.statistics['short_scale']
            embedding = torch.cat((m,s),dim=-1)
        return dict(metadata=self.metadata(),embedding=None if embedding is None else embedding.clone(),
                    long=None if self.long is None else self.long.clone(), short=None if self.short is None else self.short.clone())

    @torch.no_grad()
    def decode_embedding(self, embedding, recent_anchor, history_anchor, blocks, metadata):
        """Stateless readout: the exported vector plus explicit physical anchors and timestamps."""
        z = torch.as_tensor(embedding,device=self.device,dtype=torch.float32)
        if z.shape != (1,2*self.width) or not torch.isfinite(z).all():
            raise ValueError('Expected one finite concatenated vector')
        if not 4<=blocks<=self.long_model.blocks or min(recent_anchor,history_anchor)<=0 or not np.isfinite([recent_anchor,history_anchor]).all():
            raise ValueError('Invalid history coverage or physical anchors')
        m=z[:,:self.width]*self.statistics['long_scale']+self.statistics['long_mean']
        s=z[:,self.width:]*self.statistics['short_scale']+self.statistics['short_mean']
        return dict(metadata=copy.deepcopy(metadata),
            recent=dict(as_of=metadata['as_of'],anchor=float(recent_anchor),coordinates=self.recent_decoder.decode_recent(s).clone()),
            history=dict(as_of=metadata['long_as_of'],anchor=float(history_anchor),coordinates=self.long_model.decode_history(m)['reconstruction'][:,-blocks:].clone()),
            structure=dict(as_of=metadata['long_as_of'],values=(self.long_model.structure(m)*self.target_scale+self.target_mean).clone()))

    @torch.no_grad()
    def decode(self):
        """Decode observed history; an older long state keeps its older time/price anchor."""
        current=self.current()
        if current['embedding'] is not None:
            return self.decode_embedding(current['embedding'],self.closes[0],self.long_anchor,len(self.blocks),current['metadata'])
        result=dict(metadata=self.metadata(),recent=None,history=None,structure=None)
        if self.count>=128:
            result['recent']=dict(as_of=str(self.features.last_time),anchor=float(self.closes[0]),
                coordinates=self.recent_decoder.decode_recent(self.short).clone())
        return result

    def snapshot(self):
        cpu = lambda x: None if x is None else x.detach().cpu().clone()
        return dict(schema=SCHEMA,identity=self.identity,key=self.key,period=self.period,features=self.features.snapshot(),
                    partition=self.partition,partition_start=self.partition_start,count=self.count,
                    hidden=cpu(self.hidden),short=cpu(self.short),long=cpu(self.long),
                    block_x=[cpu(x) for x in self.block_x],block_anchor=self.block_anchor,
                    blocks=[{**b,'z':cpu(b['z'])} for b in self.blocks],closes=list(self.closes),
                    long_row=self.long_row,long_time=self.long_time,long_anchor=self.long_anchor)

    def restore(self, state):
        if (state['schema']!=SCHEMA or state['identity']!=self.identity or state['key']!=self.key or state['period']!=self.period):
            raise ValueError('Stream snapshot schema/model/contract/period mismatch')
        def tensor(value,shape):
            if value is None: return None
            if tuple(value.shape)!=shape or not torch.isfinite(value).all(): raise ValueError('Invalid saved tensor')
            return value.to(self.device).clone()
        features = CausalFeatures.restore(state['features'])
        count = int(state['count']); start = int(state['partition_start'])
        if count<0 or count and count!=features.row-start+1:
            raise ValueError('Invalid stream row coverage')
        if len(state['blocks'])>self.long_model.blocks or len(state['block_x'])>=128 or len(state['closes'])>65:
            raise ValueError('Invalid stream buffers')
        hidden=tensor(state['hidden'],(self.short_model.layers,1,self.width))
        short=tensor(state['short'],(1,self.width)); long=tensor(state['long'],(1,self.width))
        if count and (hidden is None or short is None): raise ValueError('Missing short state')
        blocks=[]
        for b in state['blocks']:
            if not math.isfinite(b['anchor']) or b['anchor']<=0: raise ValueError('Invalid block anchor')
            if b['row']%128!=127 or b['row']-127<start or b['row']>features.row or b['z'] is None:
                raise ValueError('Invalid block row coverage')
            blocks.append({**b,'z':tensor(b['z'],(1,self.width))})
        block_x=[tensor(x,(18,)) for x in state['block_x']]
        if (long is not None) != (state['long_row'] is not None) or (long is not None)!=(len(blocks)>=4): raise ValueError('Missing long timestamp/state')
        if any(b['row']-a['row']!=128 for a,b in zip(blocks,blocks[1:])): raise ValueError('Noncontiguous cached blocks')
        if long is not None and (state['long_row']!=blocks[-1]['row'] or state['long_anchor']!=blocks[-1]['anchor']): raise ValueError('Long anchor/row mismatch')
        if bool(block_x)!=(state['block_anchor'] is not None): raise ValueError('Partial block anchor mismatch')
        if state['block_anchor'] is not None and (not math.isfinite(state['block_anchor']) or state['block_anchor']<=0): raise ValueError('Invalid partial block anchor')
        if long is not None and (len(blocks)<4 or not 0<=features.row-state['long_row']<128): raise ValueError('Invalid long age')
        if not np.isfinite(state['closes']).all() or any(x<=0 for x in state['closes']): raise ValueError('Invalid recent anchors')
        self.features=features; self.count=count; self.partition_start=start; self.partition=state['partition']
        self.hidden,self.short,self.long=hidden,short,long
        self.block_x=block_x; self.block_anchor=state['block_anchor']; self.blocks=deque(blocks,maxlen=self.long_model.blocks)
        self.closes=deque(state['closes'],maxlen=65)
        self.long_row,self.long_time,self.long_anchor=state['long_row'],state['long_time'],state['long_anchor']


def build_bundle(long_run,short_run,recon_run,fusion_run,out):
    """Read-only source verification; package weights locally on the GPU machine."""
    paths=dict(long_manifest=long_run/'manifest.json',long_best=long_run/'joint/best.pt',
               short_manifest=short_run/'manifest.json',short_best=short_run/'best.pt')
    hashes={k:sha256(v) for k,v in paths.items()}
    fusion=json.loads((fusion_run/'manifest.json').read_text())
    if fusion['sources']!=hashes: raise ValueError('Long/short weights differ from original aligned feature cache')
    recon=json.loads((recon_run/'manifest.json').read_text())
    if recon['sources']['fusion_manifest']!=sha256(fusion_run/'manifest.json') or recon['sources']['long_best']!=hashes['long_best']:
        raise ValueError('Recent decoder uses different source encoders')
    artifact=json.loads((recon_run/'artifacts.json').read_text())
    recent_path=recon_run/'short/best.pt'
    if sha256(recent_path)!=artifact['short']: raise ValueError('Recent specialist checkpoint changed')
    train,train_hash=load_cache(fusion_run,'train',hashes)
    stats={}
    for name in ('long','short'):
        stats[name+'_mean']=train[name].mean(0).tolist(); stats[name+'_scale']=train[name].std(0).clip(.01).tolist()
    long=torch.load(paths['long_best'],map_location='cpu',weights_only=True)
    short=torch.load(paths['short_best'],map_location='cpu',weights_only=True)
    recent=torch.load(recent_path,map_location='cpu',weights_only=True)
    if recent['metadata']['mode']!='short': raise ValueError('Expected the selected short-only recent decoder')
    metadata=dict(schema=SCHEMA,sources={**hashes,'recent_best':artifact['short'],'train_cache':train_hash},
                  source_epochs=dict(long=long['epoch'],short=short['epoch'],recent=recent['epoch']),statistics=stats,
                  target_mean=long['metadata']['target_mean'],target_scale=long['metadata']['target_scale'],
                  policy='Closed bars only; explicit contract reset; short every bar, long every128 absolute-grid bars, 4-16 completed blocks. Separate specialist decoders. No fitted parameters added.')
    identity=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    atomic_save(dict(metadata=metadata,identity=identity,long=long['model'],short=short['model'],recent=recent['model']),out)
    return metadata,identity


def load_bundle(path, key, period, device='cpu'):
    bundle=torch.load(path,map_location='cpu',weights_only=True); meta=bundle['metadata']
    if meta['schema']!=SCHEMA: raise ValueError('Unknown inference bundle')
    identity=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    if identity!=bundle['identity']: raise ValueError('Inference metadata changed')
    long=LargeHistory(); long.load_state_dict(bundle['long'])
    short=ShortState(); short.load_state_dict(bundle['short'])
    recent=ReconstructionFusion('short'); recent.load_state_dict(bundle['recent'])
    return DualStream(long,short,recent,meta['statistics'],meta['target_mean'],meta['target_scale'],key,period,identity,device)


def frame_bars(frame,key,period):
    for row in frame.itertuples(index=False):
        yield ClosedBar(key,int(period),row.datetime,float(row.open),float(row.high),float(row.low),float(row.close),
                        float(row.volume),float(getattr(row,'oi',0.)),bool(getattr(row,'oi_available',False)))


def json_result(result):
    return {k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in result.items()}


def main():
    p=argparse.ArgumentParser(description='Encode already-closed contract bars; no optimizer or training.')
    p.add_argument('--bundle',required=True); p.add_argument('--csv',required=True); p.add_argument('--key',required=True)
    p.add_argument('--period',required=True,type=int); p.add_argument('--out',required=True); p.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    p.add_argument('--emit-every',default=128,type=int); p.add_argument('--state-in'); p.add_argument('--state-out')
    a=p.parse_args()
    if a.emit_every<1: p.error('--emit-every must be positive')
    from .data import validate_frame
    frame=validate_frame(pd.read_csv(a.csv),a.csv)
    engine=load_bundle(Path(a.bundle),a.key,a.period,a.device)
    if a.state_in: engine.restore(torch.load(a.state_in,map_location='cpu',weights_only=True))
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise ValueError('Output already exists; use a new filename')
    start=time.perf_counter(); count=0
    with out.open('x') as f:
        for bar in frame_bars(frame,a.key,a.period):
            result=engine.push(bar); count+=1
            if count%a.emit_every==0 or count==len(frame):
                f.write(json.dumps(json_result(result),allow_nan=False)+'\n')
    if a.state_out:
        state_path=Path(a.state_out); state_path.parent.mkdir(parents=True,exist_ok=True)
        atomic_save(engine.snapshot(),state_path)
    print(f'Encoded {count} bars in {time.perf_counter()-start:.1f}s; output={out}')


if __name__=='__main__': main()
