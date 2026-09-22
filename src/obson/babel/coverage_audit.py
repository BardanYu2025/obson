"""CPU-only audit of inherited endpoints, coverage and distribution; no fitting or training."""
import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import architecture as ar, architecture_benchmark as ab
from . import data, ae_context, ar_codec, activity_ablation as aa, representation as rp
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA='babel-endpoint-coverage-v1'
DESCRIPTORS=('change_rms_logpct','max_abs_path_logpct','end_path_logpct','body_rms_logpct',
             'mean_log1p_volume','zero_volume_fraction','oi_valid_fraction','oi_suspect_fraction',
             'path_train_scaled_energy','body_train_scaled_energy','activity_train_scaled_energy')


def ndarray_hash(value):
    stream=io.BytesIO();np.save(stream,value,allow_pickle=False)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def file_check(path,digest):
    if not path.is_file() or sha256(path)!=digest:raise ValueError(f'Pinned report changed: {path}')


def endpoints(s,bounds,split,stride=1,minimum=128):
    """Original contract-relative grid; require every minimum-history bar in partition."""
    if stride<1 or minimum<128:raise ValueError('Positive stride and at least128 history required')
    same=rp.time_mask(s,bounds,split);bad=np.r_[0,np.cumsum(~same)]
    ids=np.arange(127,len(s.frame),stride)
    ids=ids[ids>=minimum-1]
    return ids[s.main[ids]&same[ids]&(bad[ids+1]==bad[ids+1-minimum])]


def unpack_specs(specs,series,bounds,split,n):
    keys=[None]*n;offset=0;seen_series=set()
    for spec in specs:
        i=spec['series'];lo=spec['lo'];length=spec['length']
        if i in seen_series or not 0<=i<len(series) or spec['offset']!=offset or length<128:
            raise ValueError('Invalid packed sequence identity')
        seen_series.add(i);offset+=length
        partition=np.flatnonzero(rp.time_mask(series[i],bounds,split))
        if not len(partition) or lo!=partition[0] or not np.all(np.diff(partition)==1) or lo+length-1>partition[-1]:
            raise ValueError('Packed sequence crosses partition')
        for end,idx in spec['endpoints']:
            if not 0<=idx<n or keys[idx] is not None or not 127<=end<length:
                raise ValueError('Duplicate/outside endpoint')
            keys[idx]=(i,lo+end)
    if any(k is None for k in keys):raise ValueError('Incomplete endpoint index')
    return keys


def overlap_stats(length,ends):
    ends=np.asarray(ends,dtype=int)
    if len(np.unique(ends))!=len(ends) or (len(ends) and (ends.min()<127 or ends.max()>=length)):
        raise ValueError('Invalid or duplicate128-row endpoints')
    delta=np.zeros(length+1,np.int64)
    np.add.at(delta,ends-127,1);np.add.at(delta,ends+1,-1)
    counts=delta.cumsum()[:-1];covered=int((counts>0).sum())
    gaps=np.diff(np.sort(ends));overlap=np.maximum(0,128-gaps)
    return dict(windows=len(ends),input_bar_exposures=len(ends)*128,unique_contract_period_bars=covered,
        max_multiplicity=int(counts.max()) if len(counts) else 0,
        adjacent_pairs=len(gaps),overlapping_adjacent_pairs=int((overlap>0).sum()),
        adjacent_overlap_bars=int(overlap.sum())),counts


def describe_windows(s,x,ends,stats):
    result=[]
    for start in range(0,len(ends),256):
        e=np.asarray(ends[start:start+256]);w=x[e[:,None]+np.arange(-127,1)[None,:]]
        y,mask=ar.ordered_targets(w);g=np.sinh(w[:,:127,:2].astype(float));change=g.sum(-1)
        _,scaled,_=ar.normalize(w,y,mask,stats)
        def energy(ids):
            values=np.asarray(scaled[:,:,ids],float)**2;valid=mask[:,:,ids]
            count=valid.sum(1);means=(values*valid).sum(1)/count.clip(1)
            return (means*(count>0)).sum(-1)/(count>0).sum(-1).clip(1)
        result.append(np.column_stack((np.sqrt((change**2).mean(1)),np.abs(y[:,:127,0]).max(1),y[:,126,0],
            np.sqrt((g[:,:,1]**2).mean(1)),(w[:,:127,9]*10).mean(1),(w[:,:127,9]==0).mean(1),
            mask[:,:127,4].mean(1),((w[:,:127,23]>0)&~mask[:,:127,4]).mean(1),
            energy(slice(0,1)),energy(slice(1,2)),energy(slice(2,7)))))
    return np.concatenate(result) if result else np.empty((0,len(DESCRIPTORS)))


def distribution(values,reference=None):
    values=np.asarray(values,float)
    if not len(values):return dict(count=0)
    if not np.isfinite(values).all():raise ValueError('Nonfinite distribution statistic')
    result=dict(count=len(values),minimum=float(values.min()),maximum=float(values.max()),mean=float(values.mean()),
        quantiles={str(q):float(np.quantile(values,q)) for q in (.01,.05,.25,.5,.75,.95,.99)})
    if reference is not None and len(reference):
        lo,hi=np.quantile(reference,[.01,.99])
        result.update(outside_train_p01_p99_fraction=float(((values<lo)|(values>hi)).mean()),
            outside_train_min_max_fraction=float(((values<np.min(reference))|(values>np.max(reference))).mean()))
    return result


def profile(series,keys,encoded,stats):
    n=len(keys);values=np.empty((n,len(DESCRIPTORS)));inventory=[None]*n;groups={};per_series=[];totals=Counter()
    for j,(i,end) in enumerate(keys):groups.setdefault(i,[]).append((j,end))
    for i,pairs in groups.items():
        s=series[i];ids,ends=map(np.array,zip(*pairs));values[ids]=describe_windows(s,encoded[i],ends,stats)
        row,_=overlap_stats(len(s.frame),ends);totals.update({k:v for k,v in row.items() if k!='max_multiplicity'})
        per_series.append(dict(key=s.key,**row))
        for j,end in pairs:
            dt=s.frame.datetime.iloc[end];inventory[j]=dict(key=s.key,symbol=s.code,period=s.period,row=int(end),
                end=str(dt),session=str(s.sessions[end]),month=str(dt.to_period('M')),year=int(dt.year))
    strata={field:dict(sorted(Counter(str(v[field]) for v in inventory).items())) for field in ('symbol','period','year','month')}
    count=len({r['key'] for r in inventory})
    total=dict(totals);total.update(contract_period_series=count,max_multiplicity=max((r['max_multiplicity'] for r in per_series),default=0),
        mean_multiplicity=totals['input_bar_exposures']/max(totals['unique_contract_period_bars'],1),
        first_end=min((r['end'] for r in inventory),default=None),last_end=max((r['end'] for r in inventory),default=None))
    return dict(coverage=total,strata=strata,per_series=per_series),values,inventory


def population_profile(series,bounds,split,stride,minimum,selected,encoded,stats):
    totals=Counter();groups=Counter();months=Counter();changes=[];same_seen=set(selected);descriptors=[];unique_extra=0
    for i,s in enumerate(series):
        ends=endpoints(s,bounds,split,stride,minimum);old=np.array([e for k,e in selected if k==i],int)
        row,cover=overlap_stats(len(s.frame),ends);_,before=overlap_stats(len(s.frame),old)
        totals.update({k:v for k,v in row.items() if k!='max_multiplicity'})
        unique_extra+=int(((cover>0)&(before==0)).sum())
        groups[s.key]+=len(ends)
        if len(ends):
            months.update(str(t.to_period('M')) for t in s.frame.datetime.iloc[ends])
            descriptors.append(describe_windows(s,encoded[i],ends,stats))
        changes.append(dict(key=s.key,windows=len(ends),new_endpoints=sum((i,int(e)) not in same_seen for e in ends),
            new_contract_period_bars=int(((cover>0)&(before==0)).sum())))
    return dict(coverage=dict(totals),extra_unique_contract_period_bars=unique_extra,contract_period_series=sum(v>0 for v in groups.values()),
        counts_by_series=dict(groups),counts_by_month=dict(sorted(months.items())),changes=changes),np.concatenate(descriptors)


def prove_disjoint(series,all_keys):
    result={}
    for a,b in (('train','val'),('train','test'),('val','test')):
        shared=0
        for i,s in enumerate(series):
            _,x=overlap_stats(len(s.frame),[e for j,e in all_keys[a] if j==i])
            _,y=overlap_stats(len(s.frame),[e for j,e in all_keys[b] if j==i])
            shared+=int(((x>0)&(y>0)).sum())
        result[a+'_vs_'+b]=shared
        if shared:raise ValueError('Raw input windows overlap chronological partitions')
    return result


def replay_windows(series,encoded,keys,stats,source,split,strict,diagnostic_path=None):
    x=np.stack([encoded[i][end-127:end+1] for i,end in keys]);y,mask=ar.ordered_targets(x);statistics_replay=None
    if split=='train':
        fitted=ar.fit_scales(x,y,mask)
        for k,v in stats.items():
            if isinstance(v,list) and not np.allclose(fitted[k],v,atol=1e-6,rtol=2e-5):raise ValueError(f'Train statistics replay mismatch: {k}')
        statistics_replay=dict(passed=True,atol=1e-6,rtol=2e-5,
            max_abs_by_field={k:float(np.max(np.abs(np.asarray(fitted[k])-np.asarray(v)))) for k,v in stats.items() if isinstance(v,list)})
    x,y,mask=ar.normalize(x,y,mask,stats);index=read_json(source/'cache/index.json');rows={}
    for k,array in [('x',x),('y',y),('mask',mask)]:
        name=f'{split}_{k}.npy';expected=index['files'][name];digest=ndarray_hash(array);path=source/'cache'/name
        row=dict(expected_sha256=expected,reconstructed_sha256=digest,exact_reconstruction=digest==expected,source_binary_available=path.exists())
        if path.exists():
            file_check(path,expected);original=np.load(path,mmap_mode='r',allow_pickle=False)
            same_shape=original.shape==array.shape
            passed=same_shape and (np.array_equal(original,array) if k=='mask' else np.allclose(original,array,atol=1e-6,rtol=2e-5))
            row.update(numeric_replay_passed=bool(passed),original_shape=list(original.shape),reconstructed_shape=list(array.shape))
            if same_shape:
                difference=np.abs(original.astype(float)-array.astype(float))
                row.update(max_abs=float(difference.max()),per_channel_max_abs=difference.max(axis=(0,1)).tolist(),
                    channels=list(ae_context.feature_names('ema8_32'))+list(aa.ACTIVITY) if k=='x' else list(ar.NAMES))
            if not passed:
                if diagnostic_path:atomic_json(dict(rows,**{k:row}),diagnostic_path)
                raise ValueError(f'Cached window replay mismatch: {name}')
        elif strict and digest!=expected:
            if diagnostic_path:atomic_json(dict(rows,**{k:row}),diagnostic_path)
            raise ValueError(f'Cannot exactly reproduce absent binary: {name}; audit on AutoDL with originals')
        if k=='x' and statistics_replay is not None:row['original_train_statistics_replay']=statistics_replay
        rows[k]=row
        if diagnostic_path:atomic_json(rows,diagnostic_path)
    return rows


def audit(root,source,bank,long_run,cross,out,strict=True):
    out.mkdir(parents=True,exist_ok=True)
    consumed=[source/'manifest.json',source/'completion.json',source/'data_audit.json',source/'cache/index.json',
        source/'cache/statistics.json',source/'cache/test_inventory.json',source/'cache/cross_research_inventory.json',
        bank/'manifest.json',long_run/'manifest.json',cross/'manifest.json',cross/'raw_audit.json',cross/'cache/index.json',
        cross/'cache/test_sequences.json',cross/'cache/test_inventory.json']+[bank/f'cache/{s}_sequences.json' for s in ('train','val','test')]
    report_hashes={str(p):sha256(p) for p in consumed}
    am=read_json(source/'manifest.json');bm=read_json(bank/'manifest.json');lm=read_json(long_run/'manifest.json');cm=read_json(cross/'manifest.json')
    ab.verify_code(am)
    if am['schema']!=ab.SCHEMA:raise ValueError('Expected original architecture report/cache')
    # Report-only execution is allowed, but every consumed report must be bound to the same experiment lineage.
    file_check(source/'cache/index.json',read_json(source/'completion.json')['files']['cache/index.json'])
    index=read_json(source/'cache/index.json')
    if index['manifest']!=am:raise ValueError('Architecture cache manifest mismatch')
    file_check(source/'data_audit.json',read_json(source/'completion.json')['files']['data_audit.json'])
    for name in ('statistics.json','test_inventory.json','cross_research_inventory.json'):file_check(source/'cache'/name,index['files'][name])
    expected_long=bm['upstream']['source_weights']['long_manifest'];file_check(long_run/'manifest.json',expected_long)
    for split in ('train','val','test'):
        file_check(bank/f'cache/{split}_sequences.json',am['source_identity'][am['bank']+f'/{split}_sequences.json'])
    for name in ('manifest.json','raw_audit.json','cache/index.json','cache/test_inventory.json'):
        file_check(cross/name,am['source_identity'][am['cross_run']+'/'+name])
    ci=read_json(cross/'cache/index.json')
    if ci['manifest']!=cm or cm['boundaries']!=lm['manifest']['boundaries']:raise ValueError('Cross manifest/boundaries mismatch')
    file_check(cross/'cache/test_sequences.json',ci['files']['test_sequences.json'])
    ref=lm['manifest'];bounds=ref['boundaries'];keys=[r['key'].split('/') for r in ref['sources']]
    series,_=data.load_series(root,sorted({r[0] for r in keys}),sorted({int(r[1]) for r in keys}))
    if data.manifest(series,bounds)!=ref:raise ValueError('Original raw source fingerprint/manifest mismatch')
    progress('Original raw manifest matched; encoding causal context/activity without model inference')
    encoded=[np.column_stack((ae_context.encode_context(s.frame,s.period,'ema8_32')['x'],aa.activity_features(s.frame,s.period)[0])) for s in series]
    stats=read_json(source/'cache/statistics.json');all_keys={};profiles={};values={};replay={};inventories={};gate_counts={}
    coverage=read_json(source/'data_audit.json')['coverage']
    for split in ('train','val','test'):
        selected=unpack_specs(read_json(bank/f'cache/{split}_sequences.json'),series,bounds,split,coverage[split]['windows']);all_keys[split]=selected
        expected=[(i,int(e)) for i,s in enumerate(series) for e in endpoints(s,bounds,split,128,512)]
        if selected!=expected:raise ValueError('Inherited endpoint policy does not reproduce packed index')
        gate_counts[split]={f'history{minimum}_stride{stride}':sum(len(endpoints(s,bounds,split,stride,minimum)) for s in series)
            for minimum,stride in ((128,1),(128,16),(128,128),(512,16),(512,128))}
        replay[split]=replay_windows(series,encoded,selected,stats,source,split,strict,out/f'{split}_cache_replay.json')
        profiles[split],values[split],inventories[split]=profile(series,selected,encoded,stats)
        if split=='test':
            old=read_json(source/'cache/test_inventory.json')
            if any(any(r[k]!=v[k] for k in ('key','row','end')) for r,v in zip(old,inventories[split])):raise ValueError('Test inventory mismatch')
        progress(f'{split}: {len(selected)} endpoints reproduced; coverage and distributions measured')
    disjoint=prove_disjoint(series,all_keys)
    candidates={}
    for minimum in (512,128):
        name=f'history{minimum}_stride16'
        candidates[name],vals=population_profile(series,bounds,'train',16,minimum,all_keys['train'],encoded,stats)
        candidates[name]['distributions']={n:distribution(vals[:,j],values['train'][:,j]) for j,n in enumerate(DESCRIPTORS)}
        progress(f'Train-only hypothetical candidate {name}: {len(vals)} endpoints, no cache changes')
    # Cross-symbol research data are audited separately; never enter candidate training populations.
    raw_audit=read_json(cross/'raw_audit.json');cross_series,_=data.load_series(root,raw_audit['eligible_symbols'],(15,30,60))
    oldraw={(v['symbol'],v['period'],v['contract']):v for v in raw_audit['files'] if v['symbol'] in raw_audit['eligible_symbols']}
    if set(oldraw)!={(s.code,s.period,s.contract) for s in cross_series}:raise ValueError('Cross-symbol raw inventory changed')
    for s in cross_series:
        v=oldraw[s.code,s.period,s.contract];path=root/s.code/f'{s.contract}_{s.period}m.csv'
        file_check(path,v['sha256'])
        if s.source_hash!=v['source_hash']:raise ValueError('Cross normalized raw source changed')
    cencoded=[np.column_stack((ae_context.encode_context(s.frame,s.period,'ema8_32')['x'],aa.activity_features(s.frame,s.period)[0])) for s in cross_series]
    ck=unpack_specs(read_json(cross/'cache/test_sequences.json'),cross_series,bounds,'test',coverage['cross_research']['windows'])
    profiles['cross_research'],values['cross_research'],inventories['cross_research']=profile(cross_series,ck,cencoded,stats)
    oldinv=read_json(source/'cache/cross_research_inventory.json')
    if any(any(r[k]!=v[k] for k in ('key','row','end')) for r,v in zip(oldinv,inventories['cross_research'])):raise ValueError('Cross inventory mismatch')
    replay['cross_research']=replay_windows(cross_series,cencoded,ck,stats,source,'cross_research',strict,out/'cross_research_cache_replay.json')
    for split in profiles:
        profiles[split]['distributions']={n:distribution(values[split][:,j],values['train'][:,j]) for j,n in enumerate(DESCRIPTORS)}
        by_period={}
        for period in (15,30,60):
            use=np.array([r['period']==period for r in inventories[split]])
            reference=np.array([r['period']==period for r in inventories['train']])
            by_period[str(period)]={n:distribution(values[split][use,j],values['train'][reference,j]) for j,n in enumerate(DESCRIPTORS)}
        profiles[split]['distributions_by_period']=by_period
        rows=[dict(v,descriptors=dict(zip(DESCRIPTORS,map(float,values[split][i])))) for i,v in enumerate(inventories[split])]
        atomic_json(rows,out/f'{split}_windows.json')
    report=dict(schema=SCHEMA,goal='Audit endpoint provenance, raw-bar coverage, overlap and distributions before any training expansion.',
        consumed_report_sha256=report_hashes,
        code_sha256={Path(mod.__file__).name:sha256(mod.__file__) for mod in (data,ae_context,ar_codec,aa,ar,rp)}|{Path(__file__).name:sha256(__file__)},
        sources={name:str(path.resolve()) for name,path in [('architecture',source),('bank',bank),('long',long_run),('cross',cross)]},
        boundaries=bounds,original_raw_manifest_matched=True,reports_only=not strict,
        all_cached_arrays_verified=all(v['exact_reconstruction'] or v.get('numeric_replay_passed',False) for row in replay.values() for v in row.values()),raw_contract_period_files=len(series),raw_bars=sum(len(s.frame) for s in series),
        inherited_policy='Main contract at endpoint from lagged-session volume, contract-relative grid end=127 mod128; >=512 in-partition historical bars inherited from hierarchical model.',
        cache_replay=replay,partition_input_bar_overlap=disjoint,gate_counts=gate_counts,profiles=profiles,train_candidates=candidates,
        limitations='No neural inference, optimizer updates, PCA or normalizer fitting for a new model. Original train-statistic replay only. Descriptive repeated research sets; no independence/causal-error-source claim. Unique bars count contract-period rows, not unique trades or cross-frequency information.',
        policy='No cache mutations, new train/val splits, cross-symbol training ingestion, sample deletion or new evaluation ranking.')
    for file,digest in report_hashes.items():file_check(Path(file),digest)
    atomic_json(report,out/'coverage_metrics.json')
    summary=['# Endpoint coverage audit','','No training or source changes. Counts are contract-period rows, not independent trades.','',
        '| Dataset | Windows | Unique input bars | Contract-period series |','|---|---:|---:|---:|']
    for name,profile_row in profiles.items():
        v=profile_row['coverage'];summary.append(f'| {name} | {v["windows"]} | {v["unique_contract_period_bars"]} | {v["contract_period_series"]} |')
    summary+=['','| Train candidate | Windows | Unique input bars | New input bars |','|---|---:|---:|---:|']
    for name,v in candidates.items():summary.append(f'| {name} | {v["coverage"]["windows"]} | {v["coverage"]["unique_contract_period_bars"]} | {v["extra_unique_contract_period_bars"]} |')
    summary+=['',f'All cached arrays verified: {report["all_cached_arrays_verified"]}. Reports-only: {report["reports_only"]}.',
        'Chronological input-window intersection: '+json.dumps(disjoint)+'.',
        'More windows are mostly overlapping views; do not automatically multiply the epoch budget or claim new independent history.']
    (out/'summary.md').write_text('\n'.join(summary))
    atomic_json(dict(status='complete',training_updates=0,cache_mutations=0,
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.md') and p.name!='completion.json'}),out/'completion.json')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--source',required=True);parser.add_argument('--bank',required=True)
    parser.add_argument('--long-run',required=True);parser.add_argument('--cross',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--reports-only',action='store_true',help='Allow absent cache binaries; report reconstructed distributions without claiming binary equivalence')
    args=parser.parse_args();paths={k:Path(v).resolve() for k,v in vars(args).items() if k!='reports_only'}
    out=paths['out']
    if any(out==src or out in src.parents or src in out.parents for k,src in paths.items() if k not in ('out','root')):raise ValueError('Use a separate output directory')
    invocation=dict(paths={k:str(v) for k,v in paths.items()},reports_only=args.reports_only,implementation_sha256=sha256(__file__),
        architecture_index_sha256=sha256(paths['source']/'cache/index.json'),long_manifest_sha256=sha256(paths['long_run']/'manifest.json'),
        cross_manifest_sha256=sha256(paths['cross']/'manifest.json'))
    if out.exists() and any(out.iterdir()):
        if not (out/'audit_manifest.json').exists() or read_json(out/'audit_manifest.json')!=invocation:
            raise ValueError('Existing audit has a different identity; use a new output directory')
    out.mkdir(parents=True,exist_ok=True);atomic_json(invocation,out/'audit_manifest.json')
    (out/'completion.json').unlink(missing_ok=True)
    audit(paths['root'],paths['source'],paths['bank'],paths['long_run'],paths['cross'],out,strict=not args.reports_only)
    progress('Coverage audit complete: no fitting, training, or source changes')


if __name__=='__main__':main()
