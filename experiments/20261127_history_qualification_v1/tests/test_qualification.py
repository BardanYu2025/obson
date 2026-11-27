import numpy as np
import pytest

from qualification import (ages,boundaries,labels,session_spans,bars_for_period,
                           price_reach,roundtrip,split_codes,PERIODS,CAPS)
from run_audit import full_identity,minute_int,mask_ok
from order_pairs import descriptors,facts,probe


def tm(s):return int(np.datetime64(s,'m').astype(int))


def bars(close):
    c=np.array(close,dtype=float);o=np.r_[c[0],c[:-1]]
    return np.column_stack((o,np.maximum(o,c),np.minimum(o,c),c,np.ones(len(c)),np.full(len(c),100),np.full(len(c),100)))


def test_civil_morning_breaks_and_missing_minute():
    b=tm('2020-01-02T00:00'); times=b+np.array([613,614,630,632,689,810,899,1260])
    main=np.sort(np.r_[times,b+631])
    k,s,wide=boundaries(times,main,np.array([b//1440]),False,np.ones(len(times),bool))
    assert list(k)==[0,1,2,4,5,2,5,3]
    assert list(np.where(s)[0])==[1,2,5] and wide[-1]


def test_no_reference_no_lunch_certificate():
    t=np.array([tm('2020-01-02T11:29'),tm('2020-01-02T13:30')])
    k,_,_=boundaries(t,t[:1],np.array([t[0]//1440]),False,np.array([1,1],bool))
    assert k.tolist()==[0,5]


def test_friday_night_weekend_and_extended_missing():
    ts=np.array([tm(x) for x in ['2024-01-06T00:59','2024-01-08T09:00','2024-01-08T14:59','2024-01-10T09:00']])
    days=np.array([tm(x)//1440 for x in ['2024-01-05','2024-01-08','2024-01-09','2024-01-10']])
    k,_,_=boundaries(ts,ts,days,False,np.ones(4,bool));assert k[1]==3 and k[3]==5


def test_horizon_boundaries_first_ties_flat_and_signs():
    c=np.ones(260)*100;c[0]=102;c[1]=102;c[35]=98
    a=np.arange(1,261);v=np.ones(260,bool);p=np.zeros(260,np.uint8)
    out=labels(c,v,v,a,a,p)
    assert out['order37'][35]==0 and out['order37'][36]==1 and out['order37'][-1]==3
    assert out['class64'][63]==0 and out['class64'][-1]==2 and out['all18_source'][256]
    for c,expected in [(np.arange(260)+100,3),(400-np.arange(260),1)]:
        assert labels(c,v,v,a,a,p)['class64'][64]==expected


def test_price_views_do_not_jump_over_missing_rows():
    t=tm('2020-01-02T09:00')+np.delete(np.arange(65),34)
    g=np.ones(len(t),bool);raw=bars(np.arange(len(t))*.1+100)
    link=np.r_[False,np.diff(t)==1];aa=ages(link,g)
    views={p:bars_for_period(t,raw,g,session_spans(t,False),p) for p in PERIODS}
    out,_,_=price_reach(views,link,link,aa,aa,np.zeros(len(t)))
    assert out['strict1'][-1]==30 and out['strict5'][-1]==6
    assert out['strict60'][-1]==0
    for p,cap in zip(PERIODS,CAPS):
        s,e,b=views[p];assert max(roundtrip(b,s,e,link,cap))<1e-10


def test_sessions_night_crosses_midnight_and_short_day_tail():
    t=tm('2024-01-02T21:00')+np.arange(330)
    s,e,_=bars_for_period(t,bars([100]*len(t)),np.ones(len(t),bool),session_spans(t,False),60)
    assert s.tolist()==[0,60,120,180,240] and list(e)==[59,119,179,239,299]
    t=tm('2024-01-02T09:00')+np.arange(75)
    s,e,_=bars_for_period(t,bars([100]*len(t)),np.ones(len(t),bool),session_spans(t,False),60)
    assert list(e)==[59,74]


def test_split_disjoint_and_zero_activity():
    n=300;part=np.zeros(n,int);part[200:]=1
    age=np.arange(1,n+1);g=np.ones(n,bool);active=g.copy();active[242]=False
    lab=labels(100+np.arange(n)*.01,active,g,age,age,part)
    assert not lab['label64_in_split'][263] and lab['label64_in_split'][264]
    assert not lab['active37'][265] and lab['active37'][279]
    assert split_codes(np.array([minute_int('2026-09-12T03:00'),minute_int('2026-09-12T03:01')]),True,minute_int('2026-09-12T03:00')).tolist()==[6,7]


def test_alias_and_mask_positions():
    assert full_identity('CZCE.CF609',('2016-01-01','2016-08-01'),['2016-03-01'])=='CZCE.CF201609'
    assert full_identity('CZCE.CF609',('2026-01-01','2026-08-01'),['2026-03-01'])=='CZCE.CF202609'
    with pytest.raises(ValueError):full_identity('CZCE.CF609',('2016','2017'),[])
    n=20;g=np.ones(n,bool);a=np.arange(1,n+1);valid=mask_ok(np.r_[False,g[:-1]],g,a)
    assert not valid[15] and valid[16]


def test_packed_equal_description_collision():
    a=bars([100]*300);b=a.copy()
    i=299
    for x,pair in [(a,[101,99]),(b,[99,101])]:
        prices=np.full(300,100.);prices[i-50:i-48]=[102,98];prices[i-34:i-32]=pair
        x[:]=bars(prices)
    assert descriptors(a)==descriptors(b)
    start=i-48;assert np.array_equal(np.array(facts(a[start:start+16]),np.float32),np.array(facts(b[start:start+16]),np.float32))
    assert np.argmax(a[-37:,3])<np.argmin(a[-37:,3]) and np.argmax(b[-37:,3])>np.argmin(b[-37:,3])



def test_full_synthetic_source_read_receipt_and_lookup(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import run_audit as audit
    t=tm('2022-01-05T09:00')+np.r_[np.arange(75),np.arange(90,150)]
    raw=bars(100+np.arange(len(t))%7);cols={k:raw[:,j] for j,k in enumerate(audit.FIELDS)}
    cols['datetime']=[audit.fmt(x)+':00' for x in t]
    cols['datetime_nano']=(t-480)*60_000_000_000
    path=tmp_path/'test.parquet';pq.write_table(pa.table(cols),path)
    rec={'id':0,'path':path.name,'sha256':audit.digest(path.read_bytes()),'bytes':path.stat().st_size,
         'product':'rb','contract':'SHFE.rb202205','symbol':'SHFE.rb2205','meta_tick':1,
         'rows':len(t),'first':cols['datetime'][0],'last':cols['datetime'][-1]}
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    inv,stats,weeks,samples,pairs,gaps=audit.analyze_file(rec,t,np.unique(t//1440),tmp_path)
    assert stats[0]['class_strict_purged']==len(t)-64 and stats[0]['order_strict_purged']==len(t)-36
    assert stats[0]['edge_day_recess']==1 and inv['rows']==len(t)
    with np.load(tmp_path/'0000.npz') as z:assert z['strict1'][-1]==len(t) and z['class64'][63]==0


def test_real_challenge_logic_on_synthetic_grid():
    from order_pairs import probe
    n=8400;i=n-1
    close=np.full(n,100.);close[i-50:i-48]=[102,98];raw=bars(close)
    views={}
    for p in PERIODS:
        st=np.arange(0,n,p);en=st+p-1
        grouped=np.array([[raw[a,0],max(raw[a:b+1,1]),min(raw[a:b+1,2]),raw[b,3],
                           sum(raw[a:b+1,4]),100,100] for a,b in zip(st,en)])
        views[p]=(st,en,grouped)
    reach={'screen'+str(p):np.full(n,cap) for p,cap in zip(PERIODS,CAPS)}
    latest={p:np.searchsorted(views[p][1],np.arange(n),side='right')-1 for p in PERIODS}
    lab={'order37':np.ones(n,dtype=bool),'all18_source':np.ones(n,dtype=bool)}
    assert probe(i,raw,views,reach,latest,lab,1)==('matched_160_and_18',True)
