"""Frozen common-history readback, with explicit position, anchor and mask boundaries."""
import numpy as np

SHIFTS=(1,16,64)


def pair_plan(specs,rows,bank_length,shifts=SHIFTS):
    """Earlier windows stay in the same packed contract/partition, with512 raw warmup.

    Every shift uses the same intersection cohort. Preserve original row order;
    shifted endpoints need not satisfy the old main-contract sampling gate.
    """
    if not shifts or any(not 0<d<126 for d in shifts):raise ValueError('Invalid overlap shifts')
    indexed={};offset=0;series=set()
    for spec in specs:
        lo,length,start=spec['lo'],spec['length'],spec['offset']
        if min(lo,start)<0 or length<128 or start!=offset or start+length>bank_length or spec['series'] in series:
            raise ValueError('Invalid packed partition')
        series.add(spec['series']);offset+=length;keys=set()
        for end,index in spec['endpoints']:
            if index in indexed or not 0<=index<len(rows) or not 127<=end<length:raise ValueError('Invalid packed endpoint')
            row=rows[index];keys.add(row['key'])
            if row['row']!=lo+end:raise ValueError('Global endpoint row differs from packed index')
            indexed[index]=(start+end, end, lo+end)
        if len(keys)!=1:raise ValueError('Packed sequence crosses contract or is empty')
    if offset!=bank_length or set(indexed)!=set(range(len(rows))):raise ValueError('Incomplete packed inventory')
    selected=[];excluded=[]
    for i in range(len(rows)):
        absolute,local,global_row=indexed[i]
        if local-max(shifts)<127 or global_row-max(shifts)<511:
            excluded.append(dict(index=i,reason='earlier_window_partition_or_512_warmup_boundary'));continue
        selected.append(dict(index=i,bank_end=absolute,earlier_rows={str(d):global_row-d for d in shifts},**rows[i]))
    return selected,excluded


def windows(bank,plan,shift=0):
    return np.stack([bank[p['bank_end']-shift-127:p['bank_end']-shift+1] for p in plan]).astype(np.float32)


def pair_scores(a,b,ya,yb,ma,mb,za,zb,shift,stats):
    """Inputs are physical seven-channel outputs. Never correct a prediction by truth.

    Shape reanchors each predicted path to its OWN first shared predicted close.
    Native errors retain the original target anchor and expose constant offsets.
    """
    if not 0<shift<126:raise ValueError('Invalid overlap shift')
    for v in (a,b,ya,yb,ma,mb):
        if v.ndim!=3 or v.shape!=a.shape or v.shape[1:]!=(128,7):raise ValueError('Matched128x7 windows required')
    aa=np.asarray(a[:,shift:127],float);bb=np.asarray(b[:,:127-shift],float)
    ta=np.asarray(ya[:,shift:127],float);tb=np.asarray(yb[:,:127-shift],float)
    mask=ma[:,shift:127]&mb[:,:127-shift]
    if not np.array_equal(ma[:,shift:127],mb[:,:127-shift]) or not mask[:,:,:2].all():raise ValueError('Shared target masks differ or price missing')
    if not np.isfinite(np.concatenate((aa,bb,ta,tb))).all():raise ValueError('Nonfinite shared outputs')
    # The same bars have identical non-path targets and identical path differences.
    if not np.allclose(ta[:,:,1:][mask[:,:,1:]],tb[:,:,1:][mask[:,:,1:]],atol=1e-6,rtol=2e-5):raise ValueError('Shared targets differ')
    if not np.allclose(np.diff(ta[:,:,0]),np.diff(tb[:,:,0]),atol=2e-5,rtol=2e-5):raise ValueError('Shared path chronology differs')
    pa,pb=aa[:,:,0],bb[:,:,0];qa,qb=ta[:,:,0],tb[:,:,0]
    shape=lambda v:v-v[:,:1]
    rms=lambda v:np.sqrt(np.mean(v*v,axis=1))
    scale=float(stats['delta_scale'][0]);ys=np.array(stats['y_scale'])
    result=dict(path_shape_gap_bps=rms(shape(pa)-shape(pb))*100,
        path_shape_error_a_bps=rms(shape(pa)-shape(qa))*100,path_shape_error_b_bps=rms(shape(pb)-shape(qb))*100,
        path_native_mae_a_bps=np.abs(pa-qa).mean(1)*100,path_native_mae_b_bps=np.abs(pb-qb).mean(1)*100,
        path_signed_offset_a_bps=(pa-qa).mean(1)*100,path_signed_offset_b_bps=(pb-qb).mean(1)*100,
        change1_gap_nmse=((np.diff(pa)-np.diff(pb))/scale)**2,
        change1_error_a_nmse=((np.diff(pa)-np.diff(qa))/scale)**2,
        change1_error_b_nmse=((np.diff(pb)-np.diff(qb))/scale)**2)
    for k in ('change1_gap_nmse','change1_error_a_nmse','change1_error_b_nmse'):result[k]=result[k].mean(1)
    support={k:np.ones(len(a),bool) for k in result}
    for name,channels in [('body',[1]),('activity',[2,3,4,5,6])]:
        valid=mask[:,:,channels];count=valid.sum(1);ok=count>0
        for label,x,y in [('gap',aa,bb),('error_a',aa,ta),('error_b',bb,tb)]:
            e=((x[:,:,channels]-y[:,:,channels])/ys[channels])**2
            per_channel=(e*valid).sum(1)/count.clip(1)
            key=f'{name}_{label}_nmse';result[key]=(per_channel*ok).sum(1)/ok.sum(1).clip(1);support[key]=ok.any(1)
    za,zb=np.asarray(za,float),np.asarray(zb,float)
    if za.shape!=zb.shape or za.ndim!=2 or len(za)!=len(a) or not np.isfinite(za).all() or not np.isfinite(zb).all():raise ValueError('Invalid paired states')
    denom=np.linalg.norm(za,axis=1)*np.linalg.norm(zb,axis=1);valid=denom>1e-12
    result['state_cosine']=np.divide((za*zb).sum(1),denom,out=np.zeros(len(za)),where=valid);support['state_cosine']=valid
    result['state_rms_change']=rms(za-zb);support['state_rms_change']=np.ones(len(a),bool)
    return result,support


def summarize(values,valid):
    x=np.asarray(values)[np.asarray(valid,bool)]
    if not len(x):return dict(support=0,mean=None,p50=None,p90=None,p99=None,maximum=None)
    if not np.isfinite(x).all():raise ValueError('Nonfinite score')
    return dict(support=len(x),mean=float(x.mean()),p50=float(np.quantile(x,.5)),p90=float(np.quantile(x,.9)),p99=float(np.quantile(x,.99)),maximum=float(x.max()))
