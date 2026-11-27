"""Predeclared sampled counterfactual checks; actual daily legal bands are not in this delivery."""
import math
import struct
import numpy as np

from qualification import PERIODS, CAPS


def facts(block):
    if len(block)!=16 or not np.isfinite(block).all() or np.any(block[:,4:]<0):
        return (0.,)*5
    prices=[0.]
    for j in range(1,16):
        prices.append(prices[-1]+math.log(block[j,0]/block[j-1,3])+math.log(block[j,3]/block[j,0]))
    return (math.fsum((1 if j>=8 else -1)*v/4 for j,v in enumerate(prices)),
            math.fsum((1 if 4<=j<12 else -1)*v/4 for j,v in enumerate(prices)),
            math.log1p(math.fsum(block[:,4])),math.asinh(block[-1,6]-block[0,5]),1.)


def descriptors(bars):
    values=[]
    for h in (16,64,256):
        c=np.log(bars[-h-1:,3]); r=np.diff(c); net=c[-1]-c[0]
        total=math.fsum(np.abs(r))
        values.extend((net,abs(net)/total if total else 0.,math.sqrt(math.fsum(r*r)),
                       max(np.maximum.accumulate(c)-c),math.log1p(math.fsum(bars[-h:,4])),
                       math.asinh(bars[-1,6]-bars[-h,5])))
    return struct.pack('>18d',*values)


def probe(i,raw,views,reach,newest,labels,tick):
    if not labels['order37'][i]:return 'no_order_support',False
    if not labels['all18_source'][i]:return 'no_full_descriptors',False
    if not all(reach['screen'+str(p)][i]==cap for p,cap in zip(PERIODS,CAPS)):
        return 'incomplete_four_view_chains',False
    if not tick or not math.isfinite(tick) or tick<=0:return 'no_contract_matched_tick',False
    start=i-256; window=raw[start:i+1].copy()
    loc=None
    for k in range(i-36,i-1):
        quarter=(i-1-k)%16//4
        if quarter!=(i-2-k)%16//4 or not np.all(raw[k:k+3,4]>0):continue
        parents={}
        for p in PERIODS[1:]:
            bs,ends,_=views[p];j=np.searchsorted(ends,k)
            if j<len(ends) and bs[j]<=k and ends[j]>=k+2 and ends[j]<=i:
                parents[p]=j
        if len(parents)==3:
            loc=k;break
    if loc is None:return 'no_first_legal_position',False
    remaining=np.r_[raw[i-36:loc,3],raw[loc+2:i+1,3]]
    high,low=float(remaining.max()+tick),float(remaining.min()-tick)
    if low<=0 or any(abs(v/tick-round(v/tick))>1e-6 for v in (high,low)):
        return 'price_grid_unresolved',False
    variants=[]; triple=[]; k=loc-start
    for first,second in [(high,low),(low,high)]:
        x=window.copy();x[k:k+2,3]=first,second
        for j in range(k,k+3):
            x[j,0]=x[j-1,3];x[j,1]=max(x[j,0],x[j,3]);x[j,2]=min(x[j,0],x[j,3])
        v=x[-37:,3];variants.append(x);triple.append(v.argmax()<v.argmin())
    if triple!=[True,False]:return 'no_order_flip',False
    # The three affected minutes belong to one complete parent at each resolution.
    # Equality of their ordered OHLC aggregate and unchanged activity proves equality of all those parents.
    def aggregate(x):
        q=x[k:k+3]
        return (q[0,0],max(q[:,1]),min(q[:,2]),q[-1,3],sum(q[:,4]),q[0,5],q[-1,6])
    if aggregate(variants[0])!=aggregate(variants[1]):return 'unequal_parents',False
    coarse=[]
    for p,cap in zip(PERIODS[1:],CAPS[1:]):
        _,ends,coarse_raw=views[p]; j=int(newest[p][i]);current=int(ends[j]==i)
        stop=j+1-current
        for g in range(cap//16):
            if 16*(g+1)>cap-current:coarse.extend((0.,)*5)
            else:coarse.extend(facts(coarse_raw[stop-16*(g+1):stop-16*g]))
    buffers=[]
    for x in variants:
        a=[]
        for g in range(16):
            if g==15:a.extend((0.,)*5)
            else:a.extend(facts(x[256-16*(g+1):256-16*g]))
        buffers.append(struct.pack('>160f',*(a+coarse)))
    if buffers[0]!=buffers[1]:return 'unequal_packed_facts',False
    same=descriptors(variants[0])==descriptors(variants[1])
    return ('matched_160_and_18' if same else 'matched_160_only'),True

