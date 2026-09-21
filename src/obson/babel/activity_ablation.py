"""Price/activity features versus split input processing with matched warm starts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import detail_alignment as da
from .ae_extend import atomic_json
from .ar_reconstruction import read_arrays, device_arrays, run_jobs, paired_error_interval, describe_predictions
from .dual_state import sha256, verify_files
from .ema_ablation import identity_for
from .progress import progress
from .stage_audit import load_data
from .stream_audit import bounded_difference

SCHEMA = 'babel-activity-ablation-v1'
MODES = ('baseline','features','branches')
ACTIVITY = ('log_volume_vs_prior_ema32','delta_log_volume','oi_change_percent_asinh',
            'oi_change_per_volume_asinh','turnover_percent_asinh','oi_change_available',
            'oi_volume_ratio_available','turnover_available','oi_source_open','previous_volume_available')
PRICE_IDS = tuple(range(9))+(12,13,14,15,16,17)
ACTIVITY_IDS = (9,10,11)+tuple(range(18,28))
CONTRASTS = (('features','baseline'),('branches','features'),('branches','baseline'))


def activity_features(frame, period):
    """Observable quantities only; no inferred buy/sell aggressor or open/close labels."""
    v = frame.volume.to_numpy(float); oi = frame.oi.to_numpy(float); n = len(frame)
    lv = np.log1p(v); prev_lv = np.r_[lv[0],lv[:-1]]
    prior = pd.Series(lv).ewm(span=32,adjust=False).mean().shift(1).fillna(lv[0]).to_numpy()
    prev = np.r_[0.,oi[:-1]]
    valid_oi = frame.oi_available.to_numpy(bool)&(oi>0)
    prev_valid = np.r_[False,valid_oi[:-1]]
    consecutive = frame.datetime.diff().dt.total_seconds().to_numpy()==period*60
    opening = pd.to_numeric(frame['open_oi'],errors='coerce').to_numpy(float) if 'open_oi' in frame else np.full(n,np.nan)
    use_open = valid_oi&np.isfinite(opening)&(opening>0)
    use_previous = ~use_open&valid_oi&prev_valid&consecutive
    valid = use_open|use_previous
    base = np.where(use_open,opening,prev); safe = np.where(valid,base,1.)
    delta = np.where(valid,oi-safe,0.)
    ratio_valid = valid&(v>0)
    turnover_valid = valid
    x = np.column_stack((lv-prior,lv-prev_lv,np.arcsinh(100*delta/safe),
        np.where(ratio_valid,np.arcsinh(delta/np.maximum(v,1.)),0.),
        np.where(turnover_valid,np.arcsinh(100*v/safe),0.),valid,ratio_valid,turnover_valid,use_open,np.arange(n)>0)).astype(np.float32)
    if x.shape!=(n,10) or not np.isfinite(x).all(): raise ValueError('Invalid activity features')
    audit = dict(bars=n,zero_volume=int((v==0).sum()),oi_unavailable_or_zero=int((~valid_oi).sum()),
        opening_oi_used=int(use_open.sum()),previous_close_fallback=int(use_previous.sum()),delta_unavailable=int((~valid).sum()),
        gap_without_open_oi=int((~use_open&~consecutive&(np.arange(n)>0)).sum()),
        ratio_abs_gt1=int((ratio_valid&(np.abs(delta)>v)).sum()),ratio_abs_gt2=int((ratio_valid&(np.abs(delta)>2*v)).sum()),
        delta_with_zero_volume=int((valid&(v==0)&(delta!=0)).sum()),
        opening_oi_nonpositive=int((np.isfinite(opening)&(opening<=0)).sum()),
        invalid_open_oi=int((~np.isfinite(opening)|(opening<0)).sum()) if 'open_oi' in frame else n)
    return x,audit


def select_features(bank, mode):
    if mode not in MODES: raise ValueError('Unknown activity mode')
    if bank.shape[-1]!=28: raise ValueError('Expected28 channels')
    x = np.array(bank,copy=True)
    if mode=='baseline': x[...,18:] = 0
    return x


class ActivityStreams(da.PackedStreams):
    def __init__(self,path,split,job):
        super().__init__(path,split); self.mode = job.get('activity','original')

    def batches(self,batch,seed=None,inputs=True):
        for reset,x,pick in super().batches(batch,seed,inputs):
            if x is not None: x = x[...,:18] if self.mode=='original' else select_features(x,self.mode)
            yield reset,x,pick


class SplitProjection(nn.Module):
    """Same linear parameter budget plus two zero-start nonlinear residual gates."""
    def __init__(self,width):
        super().__init__()
        self.price = nn.Linear(len(PRICE_IDS),width,bias=False)
        self.activity = nn.Linear(len(ACTIVITY_IDS),width,bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        self.gates = nn.Parameter(torch.zeros(2))
        self.norm = nn.LayerNorm(width,elementwise_affine=False)
        self.activation = nn.GELU()

    def forward(self,x):
        p = self.price(x[...,PRICE_IDS]); a = self.activity(x[...,ACTIVITY_IDS])
        g = self.gates.tanh()
        return p+a+self.bias+g[0]*(self.activation(self.norm(p))-p)+g[1]*(self.activation(self.norm(a))-a)


def initial_model(meta,device,job):
    if job.get('name')=='original': return da.initial_model(meta,device)
    mode = job['activity']
    if mode not in MODES: raise ValueError('Unknown activity model')
    model = da.DetailModel(**meta['config'],input_dim=28)
    # All jobs consume the same initialization RNG draws, including the unused
    # split module for the two flat controls, so dropout starts from the same RNG.
    split = SplitProjection(meta['config']['width'])
    state = dict(torch.load(Path(meta['short_run'])/'best.pt',map_location='cpu',weights_only=True)['model'])
    old = state['input.0.weight']; w = old.new_zeros((old.shape[0],28)); w[:,:18] = old
    state['input.0.weight'] = w
    model.encoder.load_state_dict(state)
    if mode=='branches':
        with torch.no_grad():
            split.price.weight.copy_(w[:,PRICE_IDS]); split.activity.weight.copy_(w[:,ACTIVITY_IDS]); split.bias.copy_(state['input.0.bias'])
        model.encoder.input[0] = split
    model.head.load_state_dict(torch.load(Path(meta['source'])/'short/best.pt',map_location='cpu',weights_only=True)['model'])
    return model.to(device)


def probe_targets(features, keys):
    targets = []; masks = []
    for i,end in keys:
        x = features[i][end-15:end+1]
        if len(x)!=16: raise ValueError('Insufficient activity target history')
        valid = np.column_stack((np.ones(16,bool),x[:,9]>0,x[:,5]>0,x[:,6]>0,x[:,7]>0))
        count = valid.sum(0); mean = (x[:,:5]*valid).sum(0)/np.maximum(count,1)
        targets.append(np.r_[x[-1,:5],mean]); masks.append(np.r_[valid[-1],count==16])
    return np.array(targets,np.float32),np.array(masks,bool)


def prepare(meta,out,device):
    cache = out/'cache'; cache.mkdir(exist_ok=True); index = cache/'index.json'
    if index.exists():
        info = json.loads(index.read_text())
        if info['manifest']!=meta: raise ValueError('Prepared data identity changed')
        verify_files(cache,info['files']); progress('Verified activity bank and audit'); return
    ref,series,encoded,_ = load_data(meta['root'],meta['long_run'])
    activities = []; audit = []
    for s in series:
        x,summary = activity_features(s.frame,s.period); activities.append(x); audit.append(dict(key=s.key,**summary))
    banks = [dict(x=np.column_stack((e['x'],x))) for e,x in zip(encoded,activities)]
    atomic_json(dict(contracts=audit,totals={k:sum(a[k] for a in audit) for k in audit[0] if k!='key'},
        scope='Per-contract/period observed bar audit, not unique transactions. Ratios are descriptive; no buy/sell or opening/closing inference. Missing/zero OI masked, no strict delta<=volume assumption.',
        source='TqSdk Kline volume is bar volume; open_oi/close_oi are start/end OI. Downloader may fill missing OI columns with zero; zero is treated conservatively as unavailable for new features.'),cache/'activity_audit.json')
    files = {'activity_audit.json':sha256(cache/'activity_audit.json')}; coverage = {}; source = Path(meta['source'])
    for split in da.SPLITS:
        keys = np.load(source/f'target_cache/{split}_keys.npy',allow_pickle=False)
        x,specs = da.sequence_specs(series,banks,ref['boundaries'],split,keys)
        with (cache/f'{split}_x.tmp').open('wb') as f: np.save(f,x,allow_pickle=False)
        (cache/f'{split}_x.tmp').replace(cache/f'{split}_x.npy'); atomic_json(specs,cache/f'{split}_sequences.json')
        target,mask = probe_targets(activities,keys)
        np.save(cache/f'{split}_activity.npy',target,allow_pickle=False); np.save(cache/f'{split}_activity_mask.npy',mask,allow_pickle=False)
        for name in (f'{split}_x.npy',f'{split}_sequences.json',f'{split}_activity.npy',f'{split}_activity_mask.npy'): files[name] = sha256(cache/name)
        coverage[split] = dict(windows=len(keys),streams=len(specs),replayed_bars=len(x),activity_probe_support=mask.sum(0).tolist())
        if split=='test':
            inventory = [dict(key=series[i].key,symbol=series[i].code,period=series[i].period,row=int(end),end=str(series[i].frame.datetime.iloc[end]),
                anchor=float(series[i].frame.close.iloc[end-64]),week=str(pd.Timestamp(series[i].sessions[end]).to_period('W-SUN'))) for i,end in keys]
            atomic_json(inventory,cache/'test_inventory.json'); files['test_inventory.json'] = sha256(cache/'test_inventory.json')
    scales = da.detail_scales(read_arrays(source,'train')['y']); va = device_arrays(read_arrays(source,'val'),device)
    old = da.initial_model(meta,device).eval()
    reference = da.run_epoch(old,va,ActivityStreams(cache,'val',{}),False,False,scales,meta['streams'])[0]
    checks = {}
    for mode in MODES:
        job = dict(activity=mode); model = initial_model(meta,device,job).eval()
        score,z,_ = da.run_epoch(model,va,ActivityStreams(cache,'val',job),True,True,scales,meta['streams'],collect=True)
        numeric = bounded_difference(z,va['z'].cpu().numpy())
        checks[mode] = dict(comparison=numeric,score=score,retained=da.eligible(score,reference),parameters=sum(p.numel() for p in model.parameters()))
        atomic_json(dict(reference=reference,variants=checks),out/'warm_start_validation.json')
        if not numeric['passed'] or not da.eligible(score,reference): raise ValueError('Activity initial model differs from original cache')
    atomic_json(dict(scales=scales,reference=reference),cache/'statistics.json'); files['statistics.json'] = sha256(cache/'statistics.json')
    atomic_json(dict(boundaries=ref['boundaries'],splits=coverage),out/'coverage.json'); atomic_json(dict(manifest=meta,files=files),index)


def fit_activity_probe(zs, targets, masks):
    """Frozen linear readout; all normalizers fit train, alpha selected on val."""
    results = []; test_errors = np.full(targets[2].shape,np.nan)
    for j in range(targets[0].shape[1]):
        valid = [m[:,j] for m in masks]; counts = [int(v.sum()) for v in valid]
        if min(counts)<2:
            results.append(dict(support=counts,skipped='insufficient support')); continue
        x = [z[v].astype(float) for z,v in zip(zs,valid)]; y = [a[v,j].astype(float) for a,v in zip(targets,valid)]
        mean,scale = x[0].mean(0),x[0].std(0).clip(1e-5); ym,ys = y[0].mean(),max(y[0].std(),1e-5)
        if y[0].std()<1e-8:
            results.append(dict(support=counts,skipped='constant training target')); continue
        x = [np.column_stack(((a-mean)/scale,np.ones(len(a)))) for a in x]; yy = [(a-ym)/ys for a in y]
        lhs,rhs = x[0].T@x[0],x[0].T@yy[0]; best = None
        for alpha in (1.,10.,100.,1000.):
            penalty = np.eye(lhs.shape[0])*alpha; penalty[-1,-1] = 0
            weight = np.linalg.solve(lhs+penalty,rhs); score = float(np.mean((x[1]@weight-yy[1])**2))
            if best is None or score<best[0]: best = score,alpha,weight
        pred = x[2]@best[2]; errors = (pred-yy[2])**2; test_errors[valid[2],j] = errors
        total = float(np.square(yy[2]-yy[2].mean()).sum())
        results.append(dict(support=counts,alpha=best[1],validation_normalized_mse=best[0],test_normalized_mse=float(errors.mean()),
            test_r2=1-float(errors.sum())/total if total>1e-12 else None,train_mean_baseline_mse=float(np.square(yy[2]).mean())))
    return dict(targets=results,names=[f'{h}/{k}' for h in ('current','mean16') for k in ACTIVITY[:5]],
                scope='Observed-history descriptor readout, not prediction or reconstruction of true order flow.'),test_errors


def reconstruction_diagnostics(pred, data):
    y = data['y'].cpu().numpy(); mask = data['mask'].cpu().numpy(); metrics = {}
    def add(name,error,valid):
        values = error[valid]
        metrics[name] = dict(mae=float(values.mean()) if len(values) else None,support=int(valid.sum()))
    add('body_mae_bps',np.abs((pred[...,1]-pred[...,0])-(y[...,1]-y[...,0]))*100,mask[...,0]&mask[...,1])
    pr = np.abs(pred[...,1]-pred[...,0])+pred[...,2]+pred[...,3]
    yr = np.abs(y[...,1]-y[...,0])+y[...,2]+y[...,3]
    add('range_mae_bps',np.abs(pr-yr)*100,mask[...,:4].all(-1))
    for j,name in ((2,'upper_bps'),(3,'lower_bps'),(4,'encoded_volume'),(5,'encoded_oi')):
        factor = 100 if j<4 else 1
        add(name,np.abs(pred[...,j]-y[...,j])*factor,mask[...,j])
        if j>=4:
            for h in (1,4,16):
                error = np.abs((pred[:,h:,j]-pred[:,:-h,j])-(y[:,h:,j]-y[:,:-h,j]))
                add(f'{name}_change{h}',error,mask[:,h:,j]&mask[:,:-h,j])
    return metrics


@torch.no_grad()
def evaluate(out,device):
    da.evaluate(out,device,model_factory=initial_model,streams_factory=ActivityStreams,contrasts=CONTRASTS,
        schema=SCHEMA,report_name='activity_metrics.json',title='价格与交易活动输入对照',
        scope='Same old18 inputs versus10 added activity descriptors versus split nonlinear input branches; matched initialization, branch model has only2 extra parameters. All joint original+detail objective. Reused research test period.')
    meta = json.loads((out/'manifest.json').read_text()); aux = json.loads((out/'cache/statistics.json').read_text())
    report = json.loads((out/'activity_metrics.json').read_text()); errors = {}
    targets = [np.load(out/f'cache/{s}_activity.npy',allow_pickle=False) for s in da.SPLITS]
    masks = [np.load(out/f'cache/{s}_activity_mask.npy',allow_pickle=False) for s in da.SPLITS]
    data = {s:device_arrays(read_arrays(meta['source'],s),device) for s in da.SPLITS}
    inventory = json.loads((out/'cache/test_inventory.json').read_text())
    for job in [dict(name='original')]+meta['experiments']:
        name = job['name']; progress(f'Activity descriptor readout: {name}'); model = initial_model(meta,device,job).eval()
        if name!='original':
            ck = torch.load(out/name/'best.pt',map_location='cpu',weights_only=True); model.load_state_dict(ck['model'])
        zs = []
        for split in da.SPLITS:
            _,z,pred = da.run_epoch(model,data[split],ActivityStreams(out/'cache',split,job),name!='original',True,aux['scales'],meta['streams'],collect=True)
            zs.append(z)
        readout,err = fit_activity_probe(zs,targets,masks); errors[name] = err
        row = report['variants'][name]; row['activity_readout'] = readout
        row['candle_activity_reconstruction'] = reconstruction_diagnostics(pred,data['test'])
        if name!='original' and job['activity']=='branches': row['branch_gates'] = model.encoder.input[0].gates.tanh().cpu().tolist()
        if name!='original':
            diagnostic = torch.load(out/name/'diagnostic_best.pt',map_location='cpu',weights_only=True)
            if diagnostic['metadata'] != dict(manifest=meta,job=job): raise ValueError('Diagnostic identity mismatch')
            model.load_state_dict(diagnostic['model'])
            score = da.run_epoch(model,data['val'],ActivityStreams(out/'cache','val',job),True,True,aux['scales'],meta['streams'])[0]
            if not np.allclose(list(score.values()),list(diagnostic['validation'].values()),rtol=1e-4,atol=1e-5): raise ValueError('Diagnostic validation failed to reproduce')
            test,z,pred = da.run_epoch(model,data['test'],ActivityStreams(out/'cache','test',job),True,True,aux['scales'],meta['streams'],collect=True)
            metrics,_ = describe_predictions(data['test']['y'].cpu().numpy(),pred,inventory)
            row['diagnostic_candidate'] = dict(epoch=diagnostic['epoch'],validation=score,validation_retained=da.eligible(score,aux['reference']),
                test_objectives=test,reconstruction=metrics,candle_activity_reconstruction=reconstruction_diagnostics(pred,data['test']),scope='Validation-best detail WITHOUT retention gate; diagnostic, not deployment candidate.')
        atomic_json(row,out/name/'metrics.json')
    weeks = np.array([x['week'] for x in json.loads((out/'cache/test_inventory.json').read_text())]); report['paired_activity'] = {}
    for seed in meta['seeds']:
        for a,b in CONTRASTS:
            an,bn = f'{a}_s{seed}',f'{b}_s{seed}'; paired = {}
            for j,name in enumerate(report['variants'][an]['activity_readout']['names']):
                valid = np.isfinite(errors[an][:,j])&np.isfinite(errors[bn][:,j])
                if not valid.any(): paired[name] = dict(support=0); continue
                row = paired_error_interval(errors[an][valid,j],errors[bn][valid,j],weeks[valid]); row['delta_normalized_mse'] = row.pop('delta_close_mae_bps')
                row['interpretation'] = 'Candidate minus reference normalized MSE; lower is better. Paired weeks.'; paired[name] = row
            report['paired_activity'][an+'_minus_'+bn] = paired
    report['initialization'] = json.loads((out/'warm_start_validation.json').read_text())
    report['limits'] = 'Activity quantities are descriptive proxies, not buy/sell/open/close labels or measured liquidity. Same pretraining and two finetuning seeds; no new holdout. Gates near zero indicate little structural adaptation, not proof that every split architecture fails.'
    atomic_json(report,out/'activity_metrics.json')


def preflight(meta,out,device):
    aux = json.loads((out/'cache/statistics.json').read_text()); result = {}
    for mode in MODES:
        model = initial_model(meta,device,dict(activity=mode)); model.configure(True); model.train()
        bank = np.random.default_rng(0).normal(size=(meta['streams'],128,28)).astype(np.float32)
        x = torch.tensor(select_features(bank,mode),device=device); y = torch.randn(meta['streams'],64,7,device=device)*.1; y[...,2:] = y[...,2:].abs()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        h,_ = model.encoder(x); pred = model.head.decode_recent(h[:,-1]); parts = da.values(pred,dict(y=y,mask=torch.ones_like(y,dtype=torch.bool)),aux['scales'])
        loss = (parts['base']+.25*parts['detail']).mean(); loss.backward(); norm = nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if mode=='branches':
            gate_grad = model.encoder.input[0].gates.grad
            if not torch.isfinite(gate_grad).all() or not torch.count_nonzero(gate_grad): raise ValueError('Split gates cannot learn')
        else:
            extra_grad = model.encoder.input[0].weight.grad[:,18:]
            if mode=='baseline' and torch.count_nonzero(extra_grad): raise ValueError('Baseline uses extra features')
            if mode=='features' and not torch.count_nonzero(extra_grad): raise ValueError('New features cannot learn')
        result[mode] = dict(parameters=sum(p.numel() for p in model.parameters()),gradient_norm=float(norm),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None)
        del model,x,y,h,pred,parts,loss
    return dict(variants=result,scope='Synthetic backward only, no optimizer updates; excludes optimizer states and concurrent workers.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('all','worker','evaluate')); p.add_argument('--out',required=True)
    for name in ('source','short-run','long-run','fusion-run','root','name'): p.add_argument('--'+name)
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--streams',type=int,default=64); p.add_argument('--jobs',type=int,default=2)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise ValueError('Formal activity training requires CUDA')
    torch.set_num_threads(4); out = Path(a.out).resolve()
    if a.action=='worker':
        if not a.name: p.error('--name required')
        da.worker(out,a.name,model_factory=initial_model,streams_factory=ActivityStreams,track_diagnostic=True); return
    if not all((a.source,a.short_run,a.long_run,a.fusion_run,a.root)): p.error('All source paths required')
    if min(a.epochs,a.streams,a.jobs)<1 or a.jobs>4: raise ValueError('Invalid runtime settings')
    source,short,long,fusion = map(lambda x:Path(x).resolve(),(a.source,a.short_run,a.long_run,a.fusion_run))
    identity = identity_for(source,short,long,fusion)
    meta = dict(schema=SCHEMA,source=str(source),short_run=str(short),long_run=str(long),root=str(Path(a.root).resolve()),sources=identity,
        config=dict(width=512,decoder_width=256),seeds=[42,43],epochs=a.epochs,streams=a.streams,encoder_lr=3e-5,decoder_lr=1e-4,detail_weight=.25,
        experiments=[dict(name=f'{mode}_s{seed}',activity=mode,mode='joint_detail',seed=seed) for seed in (42,43) for mode in MODES],
        features=list(ACTIVITY),price_indices=list(PRICE_IDS),activity_indices=list(ACTIVITY_IDS),
        initialization='Original encoder/head; old18 columns preserved, new10 zero. Split copies disjoint weight columns, shared bias and norm;2 tanh residual gates zero.',
        selection='Same validation detail selection with original close<=1.02, base<=1.05, change16MSE<=1.05; epoch0 included, report actual selected epoch.',
        protocol='Matched dimensions, budget, chronological groups, TBPTT128 and dropout RNG start. Split has2 extra scalar parameters; baseline padded columns have no effective capacity. No extra training target or classification head.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta: raise ValueError('Settings changed; choose new output directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty directory lacks manifest')
    out.mkdir(parents=True,exist_ok=True); atomic_json(meta,out/'manifest.json')
    for name in ('completion.json','activity_metrics.json','summary.md'): (out/name).unlink(missing_ok=True)
    progress('Preparing price/activity bank and auditing OI source availability')
    prepare(meta,out,'cuda'); torch.cuda.empty_cache()
    atomic_json(dict(jobs=a.jobs,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)),out/'runtime.json')
    if a.action=='all':
        atomic_json(preflight(meta,out,'cuda'),out/'preflight.json'); torch.cuda.empty_cache(); run_jobs(out,a.jobs,'obson.babel.activity_ablation')
    evaluate(out,'cuda')
    if identity_for(source,short,long,fusion)!=identity: raise ValueError('Source artifacts changed during experiment')
    report = json.loads((out/'activity_metrics.json').read_text())
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),source_weights_unchanged=True,
        improved_candidates=[j['name'] for j in meta['experiments'] if report['variants'][j['name']]['selected_epoch']>0 and report['variants'][j['name']]['validation_retained']]),out/'completion.json')
    progress(f'Activity ablation complete: {out}')


if __name__=='__main__': main()
