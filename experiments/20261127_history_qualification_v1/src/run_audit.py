"""Immutable input reconciliation, scoped as-of qualifications and local endpoint lookup."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from qualification import (PERIODS,CAPS,ages,boundaries,split_codes,labels,session_spans,
                           bars_for_period,price_reach,roundtrip)
from order_pairs import probe

EXPERIMENT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((EXPERIMENT/'configs/protocol.json').read_text())
ROOT=Path(CONFIG['root'])
SPLITS=['train_time','validation_time','development_time','later_time',
        'held_product_train_era','held_product_validation_era','held_product_development_era','held_product_later_time']
FIELDS=['open','high','low','close','volume','open_oi','close_oi']
EDGE=['file_start','wall_minute','day_recess','source_session','main_intervening','unproven','invalid_quote']


def digest(data):return hashlib.sha256(data).hexdigest()


def verified(path,expected):
    before=path.stat(); data=path.read_bytes(); after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or digest(data)!=expected:
        raise ValueError(f'Input changed or SHA mismatch: {path}')
    return data


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def minute_int(text):return int(np.datetime64(text,'m').astype(np.int64))


def fmt(t):return str(np.datetime64(int(t),'m')).replace('T',' ')


def full_identity(symbol, date_range, dominant_dates):
    exchange,code=symbol.split('.'); product,digits=re.fullmatch(r'([A-Za-z]+)(\d{3,4})',code).groups()
    month=int(digits[-2:]); year_digits=int(digits[:-2]);mod=10**(len(digits)-2)
    # A symbol must appear in its own registered main window. Do not resolve a
    # short CZCE name against the latest catalogue's decade.
    days=[d for d in dominant_dates if date_range[0]<=d<=date_range[1]]
    if not days or month not in range(1,13):raise ValueError(f'Unwitnessed identity: {symbol} {date_range}')
    possible=set()
    for day in days:
        yr,mo=map(int,day[:7].split('-'))
        for y in range(yr,yr+3):
            if y%mod==year_digits and 0<=(y-yr)*12+month-mo<=24:possible.add(y)
    if len(possible)!=1:raise ValueError(f'Ambiguous maturity: {symbol} {date_range} {possible}')
    return f'{exchange}.{product}{possible.pop():04d}{month:02d}'


def prepare():
    for rel,expected in CONFIG['source_documents'].items():verified(ROOT/rel,expected)
    manifest=csv_rows(ROOT/'_incoming_p2_from_i/BOX_MONTH_FILE_SHA256.csv')
    expected={r['relpath'] for r in manifest}
    actual={str(p.relative_to(ROOT)) for p in (ROOT/'month_contracts/bars').glob('*/*.parquet')}
    if expected!=actual or len(expected)!=len(manifest):raise ValueError('Month path inventory differs from delivery')
    checkpoint=csv_rows(ROOT/'ops/p2_checkpoint.csv')
    registry=csv_rows(ROOT/'month_contracts/window_registry.csv')
    reg_key=lambda v:(v['product'],v['symbol'],v['window_start'],v['window_end'])
    if len(set(map(reg_key,registry)))!=len(registry) or Counter(map(reg_key,registry))!=Counter(map(reg_key,checkpoint)):
        raise ValueError('Registry and checkpoint windows disagree')
    cp={r['path']:r for r in checkpoint if r['ok']=='True'}
    if set(cp)!=expected:raise ValueError('Successful delivery records differ from month files')
    main_sha={}
    for line in (ROOT/'ops/SHA256SUMS_main_p0').read_text().splitlines():
        h,name=line.split();main_sha[Path(name).stem]=h
    products=sorted({r['product'] for r in manifest})
    if set(products)!=set(main_sha) or {p.stem for p in (ROOT/'main').glob('*.parquet')}!=set(products):
        raise ValueError('Main timestamp reference differs from delivery')
    metadata={x['instrument_id']:x for x in csv_rows(ROOT/'metadata/futures_symbol_info.csv')}
    by_product=defaultdict(list);maps={};all_ranges=defaultdict(list)
    for product in products:
        mapping=csv_rows(ROOT/'mapping'/f'{product}.csv')
        dates=defaultdict(list)
        for r in mapping:dates[r['underlying']].append(r['date'])
        maps[product]=(dates,sorted({r['date'] for r in mapping}))
    for sid,r in enumerate(sorted(manifest,key=lambda x:x['relpath'])):
        q=cp[r['relpath']]; product=r['product'];symbol=q['symbol']
        identity=full_identity(symbol,(q['window_start'],q['window_end']),maps[product][0][symbol])
        meta=metadata.get(symbol,{})
        ym=identity[-6:]; tick=None
        if meta.get('delivery_year') and meta.get('delivery_month'):
            if int(float(meta['delivery_year']))==int(ym[:4]) and int(float(meta['delivery_month']))==int(ym[4:]):
                tick=float(meta['price_tick']) if meta.get('price_tick') else None
        rec={'id':sid,'path':r['relpath'],'sha256':r['sha256'],'bytes':int(r['bytes']),'product':product,
             'symbol':symbol,'contract':identity,'first':q['first'],'last':q['last'],'rows':int(q['rows']),
             'window_start':q['window_start'],'window_end':q['window_end'],'meta_tick':tick}
        by_product[product].append(rec);all_ranges[identity].append((q['first'],q['last']))
    for identity,ranges in all_ranges.items():
        seq=sorted(ranges)
        if any(a[1]>=b[0] for a,b in zip(seq,seq[1:])):raise ValueError('Overlapping physical source keys: '+identity)
    jobs=[(p,by_product[p],maps[p][1],main_sha[p]) for p in products]
    return jobs,Counter(q['reason'] or 'delivered' for q in checkpoint)


def mask_ok(strict_clock,good,screen_age):
    n=len(good);i=np.arange(n)
    end8=ages(strict_clock,good)>=8
    last=np.maximum.accumulate(np.where(end8,i,-10000))
    second=np.r_[-10000,last[:-1]];first=last[np.maximum(0,second-8)]
    return (second>=8)&(first-7>=np.maximum(i-255,i-screen_age+1))


def analyze_file(rec,main,days,output):
    data=verified(ROOT/rec['path'],rec['sha256'])
    if len(data)!=rec['bytes']:raise ValueError('Manifest byte count mismatch')
    tb=pq.read_table(pa.BufferReader(data),use_threads=False)
    ns=tb['datetime_nano'].to_numpy()
    times=ns//60_000_000_000+480
    textual=pc.cast(tb['datetime'],pa.timestamp('s')).cast(pa.int64()).to_numpy()
    if (len(times)!=rec['rows'] or np.any(ns%60_000_000_000) or np.any(np.diff(times)<=0)
        or not np.array_equal(times*60,textual) or fmt(times[0])!=rec['first'][:16] or fmt(times[-1])!=rec['last'][:16]):
        raise ValueError('Clock or registry failed: '+rec['path'])
    raw=np.column_stack([tb[c].to_numpy() for c in FIELDS]).astype(np.float64,copy=False)
    o,h,l,c,v,so,eo=raw.T
    good=np.isfinite(raw[:,:4]).all(axis=1)&(l>0)&(l<=np.minimum(o,c))&(np.maximum(o,c)<=h)
    qgood=np.isfinite(raw[:,4:]).all(axis=1)&(np.min(raw[:,4:],axis=1)>=0)
    positive=np.isfinite(v)&(v>0)
    financial=rec['symbol'].startswith('CFFEX.')
    kind,strict,screen=boundaries(times,main,days,financial,good)
    strict_age,screen_age=ages(strict,good),ages(screen,good)
    product=rec['product'];part=split_codes(times,product in CONFIG['held_products'],minute_int(CONFIG['cutoff_last_old_minute_start_shanghai']))
    lab=labels(c,positive,qgood,strict_age,screen_age,part)
    mask=mask_ok(kind==1,good,screen_age)
    spans=session_spans(times,financial);views={p:bars_for_period(times,raw,good,spans,p) for p in PERIODS}
    reach,latest,pure=price_reach(views,strict,screen,strict_age,screen_age,part)
    ids=np.arange(len(times)); stats=[]; weeks=[]; samples=[]; pairs=[]
    for k in np.unique(part):
        select=part==k; cl=lab['class64'];od=lab['order37']
        cc=lambda x:int(np.count_nonzero(x&select))
        ct=cc((cl>0)&lab['strict64']&lab['label64_in_split']&pure)
        ot=(od>0)&(od<3);st=ot&lab['strict37']&lab['label37_in_split']&pure
        entry={'product':product,'split':SPLITS[k],'source_id':rec['id'],'rows':int(select.sum()),
               'price_valid':cc(good),'source_zero_flat_unchanged_oi':cc((v==0)&(h==l)&(so==eo)),
               'source_positive_volume':cc(positive),'quantity_fields_valid':cc(qgood),'clean_endpoint':cc(good),
               'two_mask_segments':cc(mask),'natural_input_split_safe':cc(pure),
               'class_screen':cc(cl>0),'class_strict':cc((cl>0)&lab['strict64']),
               'order_screen':cc(ot),'order_strict':cc(ot&lab['strict37']),
               'order_flat_screen':cc(od==3),'order_flat_strict':cc((od==3)&lab['strict37']),
               'class_strict_purged':ct,'order_strict_purged':cc(st),
               'class_screen_purged':cc((cl>0)&lab['label64_in_split']&pure),
               'order_screen_purged':cc(ot&lab['label37_in_split']&pure),
               'class_strict_active_purged':cc((cl>0)&lab['strict64']&lab['label64_in_split']&pure&lab['active64']),
               'order_strict_active_purged':cc(st&lab['active37']),
               'all18_screen_source_units':cc(lab['all18_source']),'all18_strict_source_units':cc(lab['all18_strict']),
               'all18_screen_purged_source_units':cc(lab['all18_source']&lab['label18_in_split']&pure),
               'class_screen_outside_all18':cc((cl>0)&~lab['all18_source'])}
        for code,name in enumerate(EDGE):entry['edge_'+name]=cc(kind==code)
        jump=np.r_[False,o[1:]!=c[:-1]]
        entry['day_recess_nonzero_open_gap']=cc((kind==2)&jump)
        entry['source_session_nonzero_open_gap']=cc((kind==3)&jump)
        for code,name in ((1,'down'),(2,'low_direction'),(3,'up')):
            entry['class_strict_purged_'+name]=cc((cl==code)&lab['strict64']&lab['label64_in_split']&pure)
        for code,name in ((1,'high_first'),(2,'low_first')):
            entry['order_strict_purged_'+name]=cc(st&(od==code))
        for p,cap in zip(PERIODS,CAPS):
            _,ends,_=views[p];j=latest[p];is_current=np.zeros(len(times),bool)
            if len(ends):is_current=(j>=0)&(ends[np.maximum(0,j)]==ids)
            for typ in ['strict','screen']:
                reach_p=reach[typ+str(p)]
                entry[f'{typ}_{p}m_observed_bar_reads']=int(reach_p[select].astype(np.int64).sum())
                entry[f'{typ}_{p}m_past_reads']=entry[f'{typ}_{p}m_observed_bar_reads']-cc(is_current&(reach_p>0))
                entry[f'{typ}_{p}m_full_cap']=cc(reach_p==cap)
        stats.append(entry)
        week_ids=(times//1440+3)//7
        for week in np.unique(week_ids[select]):
            group=(week_ids==week)&select
            weeks.append([product,SPLITS[k],int(week),int(np.count_nonzero(group&(cl>0)&lab['strict64']&lab['label64_in_split']&pure)),
                          int(np.count_nonzero(group&st)),int(np.count_nonzero(group&st&(od==1))),int(np.count_nonzero(group&st&(od==2)))])
        if k==0 and len(samples)==0:
            chosen=np.flatnonzero(select&(cl>0)&st&lab['strict64']&lab['active64'])
            if len(chosen):
                m=chosen[0];samples.append({'source_id':rec['id'],'contract':rec['contract'],'close_time':fmt(times[m]+1),
                    'row':int(m),'class_code':int(cl[m]),'order_code':int(od[m]),'reachable_bars_strict':[int(reach[f'strict{p}'][m]) for p in PERIODS]})
    for i in range(CONFIG['pair_offset'],len(times),CONFIG['pair_stride']):
        if not pure[i] or not lab['label18_in_split'][i]:
            result='partition_purge'
        else:
            result,_=probe(i,raw,views,reach,latest,lab,rec['meta_tick'])
        pairs.append([rec['id'],int(i),fmt(times[i]+1),SPLITS[part[i]],result])
    out={'minute_start_local':times.astype(np.int32),'boundary':kind,'split':part,'input_split_safe':pure,
         'two_mask_segments':mask,'positive_volume':positive,'price_good':good,**lab,**reach}
    for p in PERIODS:
        out[f'bar_start_{p}']=views[p][0].astype(np.int32)
        out[f'bar_end_{p}']=views[p][1].astype(np.int32)
    dest=output/f"{rec['id']:04d}.npz";np.savez_compressed(dest,**out)
    rr={str(p):roundtrip(views[p][2],views[p][0],views[p][1],screen,cap) for p,cap in zip(PERIODS,CAPS)}
    inv=dict(rec,index_sha256=digest(dest.read_bytes()),index_bytes=dest.stat().st_size,
             source_roundtrip_max_quote_error=rr,observed_complete_bars={str(p):len(views[p][0]) for p in PERIODS})
    unproven=[{'source_id':rec['id'],'full_contract':rec['contract'],'left_row':int(i-1),'right_row':int(i),
               'prior_source_close':fmt(times[i-1]+1),'next_source_open':fmt(times[i]),
               'wall_missing_minutes':int(times[i]-times[i-1]-1),'evidence':EDGE[kind[i]]}
              for i in np.flatnonzero(kind>=4) if i>0]
    return inv,stats,weeks,samples,pairs,unproven


def worker(job):
    product,inputs,mapping,expected,out=job
    data=verified(ROOT/'main'/f'{product}.parquet',expected)
    clock=pq.read_table(pa.BufferReader(data),columns=['datetime_nano'],use_threads=False).column(0).to_numpy()
    if np.any(clock%60_000_000_000) or np.any(np.diff(clock)<=0):raise ValueError('Invalid reference clock '+product)
    main=clock//60_000_000_000+480;days=np.array(mapping,dtype='datetime64[D]').astype(np.int64)
    return [analyze_file(rec,main,days,out) for rec in inputs]


def write_csv(path,rows,fields):
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def atomic_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def release_check():
    repo=EXPERIMENT.parents[1];commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,'origin/codex/babel-next'],check=True)
    seal_path=EXPERIMENT/'release.json';rel=seal_path.relative_to(repo)
    if subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:{rel}'])!=seal_path.read_bytes():raise ValueError('Release SHA drift')
    seal=json.loads(seal_path.read_text())
    for path,sha in seal['files'].items():verified(EXPERIMENT/path,sha)
    return commit,digest(seal_path.read_bytes())


def run(workers):
    commit,release=release_check();jobs,checkpoints=prepare();run_id=CONFIG['run_id']
    runs=EXPERIMENT/'runs';runs.mkdir(exist_ok=True)
    target=runs/run_id;report=EXPERIMENT/'reports'/run_id
    if target.exists() or report.exists():raise FileExistsError('Existing run or report; never overwrite')
    lock=runs/(run_id+'.lock')
    lock.mkdir()
    temp=Path(tempfile.mkdtemp(prefix='.qualification-',dir=runs))
    assets=temp/'assets';assets.mkdir();rp=temp/'report';rp.mkdir()
    inventory=[];stats=[];weeks=[];samples=[];pairs=[];gaps=[]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job,parts in zip(jobs,pool.map(worker,[(*j,assets) for j in jobs])):
                for a,b,c,d,e,f in parts:inventory.append(a);stats.extend(b);weeks.extend(c);samples.extend(d);pairs.extend(e);gaps.extend(f)
                print(f'{job[0]}: {len(parts)} source files verified and qualified',flush=True)
        # Reconcile the closing inventory and small mutable source documents too.
        jobs_end,_=prepare()
        if jobs_end!=jobs:raise ValueError('Source inventory changed during qualification')
        by=defaultdict(Counter); by_file=defaultdict(Counter)
        for row in stats:
            counts={k:v for k,v in row.items() if k not in ('source_id','product','split')}
            by[row['product'],row['split']].update(counts);by_file[row['source_id']].update(counts)
        fields=list(counts)
        compact=[dict(product=k[0],split=k[1],**v) for k,v in sorted(by.items())]
        write_csv(rp/'by_product_split.csv.gz',compact,['product','split',*fields])
        write_csv(rp/'by_contract_split.csv.gz',stats,['source_id','product','split',*fields])
        market=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        for product,part,week,*vals in weeks:market[part,week]+=np.array(vals)
        weekrows=[dict(split=k[0],week_monday=fmt((k[1]*7-3)*1440)[:10],class_endpoints=int(v[0]),order_endpoints=int(v[1]),
                       order_high_first=int(v[2]),order_low_first=int(v[3])) for k,v in sorted(market.items())]
        write_csv(rp/'market_weeks.csv.gz',weekrows,list(weekrows[0]))
        totals=Counter();split_totals=defaultdict(Counter)
        for (_,part),v in by.items():totals.update(v);split_totals[part].update(v)
        write_csv(rp/'unproven_boundaries.csv.gz',gaps,['source_id','full_contract','left_row','right_row','prior_source_close','next_source_open','wall_missing_minutes','evidence'])
        pair_names=['source_id','source_row','close_time','split','reason']
        write_csv(assets/'counterfactual_sample.csv.gz',[dict(zip(pair_names,row)) for row in pairs],pair_names)
        index_rows=sorted(inventory,key=lambda x:x['id'])
        with gzip.open(rp/'source_index.json.gz','wt',encoding='utf-8') as f:json.dump(index_rows,f,ensure_ascii=False,separators=(',',':'))
        atomic_json(rp/'examples.json',samples[::max(1,len(samples)//24)][:24])
        pair_counts={part:dict(Counter(x[4] for x in pairs if x[3]==part)) for part in sorted(split_totals)}
        structure={str(p):[max(x['source_roundtrip_max_quote_error'][str(p)][j] for x in inventory) for j in range(4)] for p in PERIODS}
        dense={str(p):sum(x['observed_complete_bars'][str(p)] for x in inventory) for p in PERIODS}
        summary={'run_id':run_id,'source_commit':commit,'release_sha256':release,'data_scope':'month quotes; only main timestamps and mapping dates/symbols used',
                 'products':len(jobs),'contract_files':len(inventory),'distinct_full_contracts':len({x['contract'] for x in inventory}),
                 'month_bytes':sum(x['bytes'] for x in inventory),'checkpoint_outcomes':dict(checkpoints),
                 'first_minute_start':min(x['first'] for x in inventory),'last_minute_start':max(x['last'] for x in inventory),
                 'totals':dict(totals),'split_totals':{k:dict(v) for k,v in sorted(split_totals.items())},
                 'observed_bars_by_period':dense,'source_roundtrip_max_native_quote_error_OHLC':structure,
                 'eligible_market_weeks':{k:int(sum(bool(v[0]>0 and v[1]>0) for (sp,_),v in market.items() if sp==k)) for k in split_totals},
                 'counterfactual_screen_by_split':pair_counts,'exchange_legal_pairs_signed':None,'volume_OI_units_signed':False,'independent_V13_signed':False,
                 'index_bytes':sum(x['index_bytes'] for x in inventory),'pair_lookup_sha256':digest((assets/'counterfactual_sample.csv.gz').read_bytes()),
                 'neural_model_run':False}
        atomic_json(rp/'summary.json',summary)
        atomic_json(rp/'receipt.json',{'run_id':run_id,'source_commit':commit,'release_sha256':release,
                    'reports':{p.name:digest(p.read_bytes()) for p in sorted(rp.iterdir())},'local_index':'runs/'+run_id,
                    'per_source_asset_SHA':'source_index.json.gz','input_manifest_SHA':CONFIG['source_documents'],'execution':'local CPU arithmetic only'})
        os.rename(assets,target)
        try:os.rename(rp,report)
        except BaseException:
            os.rename(target,assets)
            raise
    finally:
        shutil.rmtree(temp,ignore_errors=True)
        lock.rmdir()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=CONFIG['workers'])
    run(ap.parse_args().workers)

