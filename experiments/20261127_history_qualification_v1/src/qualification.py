"""CPU qualifications on a single immutable, chronological delivery window."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

PERIODS = (1, 5, 15, 60)
CAPS = (256, 128, 64, 64)
# 0 file start, 1 wall-minute, 2 witnessed daytime recess, 3 witnessed session change,
# 4 witnessed intervening product minutes, 5 unsupported boundary, 6 invalid price.
CLASS_NAMES = ('unavailable', 'down', 'low_direction', 'up')
ORDER_NAMES = ('unavailable', 'high_first', 'low_first', 'flat')


def ages(links, valid):
    """Number of consecutive usable observations ending at each row, not elapsed minutes."""
    i = np.arange(len(valid), dtype=np.int32)
    starts = np.maximum.accumulate(np.where(~links | ~valid, i, 0))
    return np.where(valid, i - starts + 1, 0).astype(np.int32)


def boundaries(times, main_clock, map_days, financial, good):
    n = len(times)
    kind = np.ones(n, dtype=np.uint8)
    kind[0] = 0
    gaps = np.flatnonzero(np.diff(times) != 1) + 1
    if len(gaps):
        left, right = times[gaps - 1], times[gaps]
        a, b = np.searchsorted(main_clock, left), np.searchsorted(main_clock, right)
        def matches(k, x):
            return (k < len(main_clock)) & (main_clock[np.minimum(k, len(main_clock)-1)] == x)
        paired = matches(a, left) & matches(b, right)
        witness = paired & (b == a + 1)
        day0, hm0 = np.divmod(left, 1440)
        day1, hm1 = np.divmod(right, 1440)
        is_day0 = np.isin(day0, map_days)
        is_day1 = np.isin(day1, map_days)
        morning = 570 if financial else 540
        mids = (hm0 == 689) & (hm1 == (780 if financial else 810))
        if not financial:
            mids |= (hm0 == 614) & (hm1 == 630)
        recess = (day0 == day1) & is_day0 & mids
        # Official mapping trading dates pair with actual product-clock neighbours.
        # In particular, a weekend is never inferred from elapsed wall time alone.
        after0 = np.searchsorted(map_days, day0, side='right')
        nextday = (after0 < len(map_days)) & (map_days[np.minimum(after0,len(map_days)-1)] == day1)
        day_to_night = (hm0 == 899) & (hm1 == 1260) & (day0 == day1) & is_day0
        day_to_day = (hm0 == 899) & (hm1 == morning) & nextday & is_day0
        eve_to_day = (hm0 >= 1260) & np.isin(hm0, [1379,1409,1439]) & (hm1 == morning) & nextday & is_day0
        # A post-midnight night close belongs to the next mapped trading day
        # after the evening's civil date, including a Friday evening.
        evening_day = day0 - 1
        ni = np.searchsorted(map_days, evening_day, side='right')
        after_evening = (ni < len(map_days)) & (map_days[np.minimum(ni,len(map_days)-1)] == day1)
        late_to_day = np.isin(hm0, [59,89,149]) & (hm1 == morning) & after_evening & np.isin(evening_day,map_days)
        night = day_to_night | eve_to_day | late_to_day
        session = day_to_day | (night if not financial else False)
        kind[gaps] = np.select([witness & recess, witness & is_day1 & session,
                                 paired & (b > a + 1)], [2,3,4], default=5)
        # The evening date itself is mapped for a day-to-night transition.
    if n > 1:
        kind[1:][~(good[1:] & good[:-1])] = 6
    strict = (kind == 1) | (kind == 2)
    screened = strict | (kind == 3)
    return kind, strict, screened


def split_codes(times, hold, cutoff):
    # Local-clock minute starts. The old source cutoff is a minute START, hence <= cutoff is development.
    y24 = int(np.datetime64('2024-01-01','m').astype(int))
    y25 = int(np.datetime64('2025-01-01','m').astype(int))
    return np.select([times > cutoff, times >= y25, times >= y24], [3,2,1], default=0).astype(np.uint8) + 4 * int(hold)


def labels(close, active, quantity_good, strict_age, screened_age, part):
    n = len(close)
    ii = np.arange(n)
    same_part = np.r_[False, part[1:] == part[:-1]]
    part_age = ages(same_part, np.ones(n, bool))
    price_ok = np.isfinite(close) & (close > 0)
    logc = np.log(np.where(price_ok,close,1.0))
    class64 = np.zeros(n,np.uint8)
    order37 = np.zeros(n,np.uint8)
    if n >= 65:
        diff = np.abs(np.diff(logc, prepend=logc[0]))
        travel = pd.Series(diff).rolling(64,min_periods=64).sum().to_numpy()[64:]
        net = logc[64:] - logc[:-64]
        # Direct summation for boundaries sensitive to rolling round-off.
        near = np.flatnonzero((travel > 0) & (np.abs(2*np.abs(net)-travel) < 1e-12))
        for j in near:
            travel[j] = math.fsum(diff[j+1:j+65])
        codes = np.where((travel>0) & (2*net>=travel),3,np.where((travel>0)&(-2*net>=travel),1,2))
        class64[64:] = np.where(screened_age[64:]>=65,codes,0)
    if n >= 37:
        sw = np.lib.stride_tricks.sliding_window_view(close,37)
        # argmin/argmax choose the FIRST occurrence of repeated extrema.
        hi,lo = sw.argmax(axis=1),sw.argmin(axis=1)
        order37[36:] = np.where(screened_age[36:]>=37,np.where(hi==lo,3,np.where(hi<lo,1,2)),0)
    consecutive = np.r_[False,np.ones(n-1,bool)]
    active_age = ages(consecutive & np.r_[False,active[:-1]], active)
    quant_age = ages(consecutive & np.r_[False,quantity_good[:-1]], quantity_good)
    return {'class64':class64,'order37':order37,'strict64':strict_age>=65,'strict37':strict_age>=37,
            'active64':active_age>=65,'active37':active_age>=37,'label64_in_split':part_age>=65,
            'label37_in_split':part_age>=37,'all18_source':(screened_age>=257)&(quant_age>=256),
            'all18_strict':(strict_age>=257)&(quant_age>=256),'label18_in_split':part_age>=257}


def session_spans(times,financial):
    """Local minute grid, restarting each session. Night grid crosses civil midnight."""
    day,hm=np.divmod(times,1440)
    spans=[(570,690),(780,900)] if financial else [(540,615),(630,690),(810,900)]
    origin=np.full(len(times),-1,np.int64)
    end=np.full(len(times),-1,np.int64)
    for a,b in spans:
        match=(hm>=a)&(hm<b)
        origin[match]=day[match]*1440+a
        end[match]=day[match]*1440+b
    if not financial:
        night=(hm>=1260)|(hm<150)
        origin[night]=(day[night]-(hm[night]<150))*1440+1260
        # Only full night bins. No present-day night close is backcast to an earlier year.
        end[night]=origin[night]+600
    return origin,end


def bars_for_period(times, raw, good, spans, p):
    n=len(times)
    if p==1:
        idx=np.flatnonzero(good)
        return idx,idx,raw[idx]
    origin,limit=spans
    bin_start=origin+((times-origin)//p)*p
    bin_end=np.minimum(bin_start+p,limit)
    # Group by clock grid, never compress a missing source minute.
    start=np.r_[0,np.flatnonzero(bin_start[1:]!=bin_start[:-1])+1]
    end=np.r_[start[1:]-1,n-1]
    bad=np.r_[0,np.cumsum(~good)]
    ok=(origin[start]>=0)&(times[start]==bin_start[start])&(times[end]+1==bin_end[start])
    ok &= (end-start+1==bin_end[start]-bin_start[start])&(bad[end+1]==bad[start])
    h=np.maximum.reduceat(raw[:,1],start);l=np.minimum.reduceat(raw[:,2],start)
    v=np.add.reduceat(raw[:,4],start)
    values=np.column_stack((raw[start,0],h,l,raw[end,3],v,raw[start,5],raw[end,6]))
    return start[ok],end[ok],values[ok]


def price_reach(views, strict, screened, strict_age, screened_age, part):
    """Provide a constructive path to C_t at each resolution. Stale/disconnected bars are not certified."""
    n=len(strict); row=np.arange(n)
    out={}; indexes={}
    # Conservative purge for a natural rolling input (including 65 pre-window closes).
    boundary=np.maximum.accumulate(np.where(np.r_[True,part[1:]!=part[:-1]],row,0))
    pure=np.ones(n,bool)
    for period,cap in zip(PERIODS,CAPS):
        starts,ends,raw=views[period]
        newest=np.searchsorted(ends,row,side='right')-1
        inside_count=newest+1-np.searchsorted(starts,boundary)
        pure &= (boundary==0)|(inside_count>=cap+65)
        indexes[period]=newest
        for typ,link,fine_age in [('strict',strict,strict_age),('screen',screened,screened_age)]:
            count=np.zeros(n,np.uint16)
            if len(ends):
                edge=np.r_[False, (starts[1:]==ends[:-1]+1) & link[starts[1:]]]
                run=ages(edge,np.ones(len(ends),bool))
                j=np.maximum(newest,0)
                # The latest closed coarse C is literally a visible 1m C.
                # Cap the one-minute bridge as well as the target view itself.
                reachable=(newest>=0)&(row-ends[j]<np.minimum(256,fine_age))&(fine_age>0)
                count[reachable]=np.minimum(cap,run[j[reachable]])
            out[f'{typ}{period}']=count
    return out,indexes,pure


def roundtrip(raw,starts,ends,links,cap):
    """Recover each complete bar once, in bounded same-view chains with their own known terminal close."""
    if not len(raw):return [0.0]*4
    n=len(raw); idx=np.arange(n)
    connected=np.r_[False,(starts[1:]==ends[:-1]+1)&links[starts[1:]]]
    before=np.maximum.accumulate(np.where(~connected,idx,0))
    reset=~connected|((idx-before)%cap==0)
    begin=np.flatnonzero(reset);last=np.r_[begin[1:]-1,n-1]
    o,h,l,c=raw[:,:4].T
    gap=np.log(o/np.r_[c[0],c[:-1]])
    body=np.log(c/o); u=np.log(h/np.maximum(o,c));d=np.log(np.minimum(o,c)/l)
    jumps=np.where(reset,0.0,gap+body)
    cum=np.cumsum(jumps)
    terminal=np.repeat(last,last-begin+1)
    cc=c[terminal]*np.exp(cum-cum[terminal]);oo=cc*np.exp(-body)
    reconstructed=np.column_stack((oo,np.maximum(oo,cc)*np.exp(u),np.minimum(oo,cc)*np.exp(-d),cc))
    return np.max(np.abs(reconstructed-raw[:,:4]),axis=0).tolist()

