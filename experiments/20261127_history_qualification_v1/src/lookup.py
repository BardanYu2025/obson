"""Look up a real contract and close time using the immutable local qualification index."""
from pathlib import Path
import argparse,gzip,json
import numpy as np
from qualification import CLASS_NAMES,ORDER_NAMES,PERIODS

E=Path(__file__).resolve().parents[1]
RID=json.loads((E/'configs/protocol.json').read_text())['run_id']


def lookup(contract,when):
    with gzip.open(E/'reports'/RID/'source_index.json.gz','rt') as f:rows=json.load(f)
    t=int(np.datetime64(when,'m').astype(int))-1
    stamp=lambda m:str(np.datetime64(int(m),'m'))
    matches=[r for r in rows if r['contract']==contract and r['first'][:16]<=stamp(t).replace('T',' ')<=r['last'][:16]]
    if len(matches)!=1:raise ValueError('No unique registered contract window at that close time')
    source=matches[0]
    with np.load(E/'runs'/RID/f"{source['id']:04d}.npz") as z:
        clock=z['minute_start_local'];i=int(np.searchsorted(clock,t))
        if i==len(clock) or clock[i]!=t:raise ValueError('No observed minute at this time')
        result={'contract':contract,'source':source['path'],'data_sha256':source['sha256'],'source_row':i,
                'close_time_shanghai':stamp(t+1),'class64':CLASS_NAMES[int(z['class64'][i])],
                'order37':ORDER_NAMES[int(z['order37'][i])],'strict_label_calendar_64':bool(z['strict64'][i]),
                'strict_label_calendar_37':bool(z['strict37'][i]),'split_safe_model_history':bool(z['input_split_safe'][i]),
                'label64_within_split':bool(z['label64_in_split'][i]),'label37_within_split':bool(z['label37_in_split'][i]),
                'all18_source_units':bool(z['all18_source'][i]),'price_reachable':{}}
        for p in PERIODS:
            ends=z[f'bar_end_{p}'];starts=z[f'bar_start_{p}'];last=np.searchsorted(ends,i,side='right')-1
            result['price_reachable'][str(p)]={}
            for tier in ('strict','screen'):
                n=int(z[f'{tier}{p}'][i])
                result['price_reachable'][str(p)][tier]={'bars':n,'first_open':stamp(clock[starts[last-n+1]]) if n else None,
                                                       'last_close':stamp(clock[ends[last]]+1) if n else None}
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('contract');ap.add_argument('close_time',help='Shanghai YYYY-MM-DDTHH:MM, bar CLOSE')
    args=ap.parse_args();print(json.dumps(lookup(args.contract,args.close_time),ensure_ascii=False,indent=2))

