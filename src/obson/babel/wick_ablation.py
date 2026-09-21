"""Controlled wick precision tradeoff: fixed architecture, data and non-wick coefficients."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import detail_alignment as da
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import read_arrays, device_arrays, run_jobs, paired_error_interval
from .ema_ablation import identity_for
from .dual_state import sha256
from .history_autoencoder import WEIGHTS, CHANNELS
from .progress import progress

SCHEMA = 'babel-wick-ablation-v1'
MODES = ('original_loss','wick_low','wick_coarse')
DESCRIPTORS = ('upper_volatility','lower_volatility','upper_fraction','lower_fraction')
CONTRASTS = (('wick_low','original_loss'),('wick_coarse','original_loss'),('wick_coarse','wick_low'))


def descriptors(y, sigma):
    body = (y[...,1]-y[...,0]).abs()
    span = (body+y[...,2]+y[...,3]).clamp_min(1e-6)
    return torch.stack((torch.log1p(y[...,2]/sigma),torch.log1p(y[...,3]/sigma),
                        y[...,2]/span,y[...,3]/span),dim=-1)


def descriptor_mask(mask):
    all_price = mask[...,:4].all(-1)
    return torch.stack((mask[...,2],mask[...,3],all_price,all_price),dim=-1)


def fit_statistics(y, mask, sigma):
    y = torch.as_tensor(y); mask = torch.as_tensor(mask); sigma = torch.as_tensor(sigma)
    d = descriptors(y,sigma).numpy(); valid = descriptor_mask(mask).numpy()
    bins = []; scales = []; counts = []
    for j in range(4):
        x = d[...,j][valid[...,j]]
        if not len(x): raise ValueError('No training support for wick descriptor')
        thresholds = sorted(set(float(q) for q in np.quantile(x,[1/3,2/3]) if q>1e-8))
        bins.append(thresholds); scales.append(float(max(np.sqrt(np.mean(x.astype(float)**2)),.1)))
        counts.append(np.bincount(np.searchsorted(thresholds,x,side='left'),minlength=len(thresholds)+1).tolist())
    body = (y[...,1]-y[...,0]).abs().numpy(); body_mask = (mask[...,0]&mask[...,1]).numpy()
    return dict(bins=bins,scales=scales,class_counts=counts,descriptors=list(DESCRIPTORS),
                body_deadband=float(max(np.quantile(body[body_mask],.1),1e-5)),
                scope='Train-only descriptor tertiles; duplicate/zero boundaries removed. Volatility is per-bar prior RMS in log-percent. Fractions use own OHLC range; no target body passed to prediction.')


def coarse_values(pred, b, stats):
    a = descriptors(pred,b['sigma']); target = descriptors(b['y'],b['sigma']); mask = descriptor_mask(b['mask'])
    terms = []
    for j,thresholds in enumerate(stats['bins']):
        cuts = target.new_tensor(thresholds)
        labels = torch.bucketize(target[...,j].contiguous(),cuts)
        edges = torch.cat((target.new_zeros(1),cuts,target.new_tensor([float('inf')])))
        low,high = edges[labels],edges[labels+1]
        distance = (F.relu(low-a[...,j])+F.relu(a[...,j]-high))/stats['scales'][j]
        loss = F.smooth_l1_loss(distance,torch.zeros_like(distance),reduction='none')
        terms.append((loss*mask[...,j]).sum(1)/mask[...,j].sum(1).clamp_min(1))
    return torch.stack(terms).mean(0)


def loss_components(pred, b, scales, stats):
    # Keep ORIGINAL denominator in every variant: lowering wick weights must not
    # silently increase open/close/volume/OI/time coefficients.
    channels = []; adjacent = []
    for n in (16,32,64):
        p,y,mask = pred[:,-n:],b['y'][:,-n:],b['mask'][:,-n:]
        w = y.new_tensor(WEIGHTS)*mask; denom = w.sum((1,2)).clamp_min(1e-12)
        channels.append((F.smooth_l1_loss(p,y,reduction='none')*w).sum(1)/denom[:,None])
        adjacent.append(.5*F.smooth_l1_loss(p[:,1:,1]-p[:,:-1,1],y[:,1:,1]-y[:,:-1,1],reduction='none').mean(1))
    ch = torch.stack(channels).mean(0)
    changes = .25*torch.stack([F.smooth_l1_loss(pred[:,h:,1]-pred[:,:-h,1],b['y'][:,h:,1]-b['y'][:,:-h,1],reduction='none').mean(1) for h in (4,16)]).mean(0)
    parts = {name:ch[:,i] for i,name in enumerate(CHANNELS)}
    parts.update(change1=torch.stack(adjacent).mean(0),change4_16=changes,
                 detail=.25*da.detail_values(pred,b['y'],scales),coarse=coarse_values(pred,b,stats))
    return parts


def objective(pred, b, scales, stats, mode):
    if mode not in MODES: raise ValueError('Unknown wick objective')
    parts = loss_components(pred,b,scales,stats)
    original = sum(parts[k] for k in parts if k!='coarse')
    if mode == 'original_loss': return original
    # 0.5 -> 0.1 in the original channel numerator; other terms unchanged.
    low = original-.8*(parts['upper_log_percent']+parts['lower_log_percent'])
    return low+(.05*parts['coarse'] if mode=='wick_coarse' else 0.)


def shape_values(pred, b, stats):
    y = b['y']; mask = b['mask']; result = {}
    for j,name in enumerate(CHANNELS):
        factor = 100 if j<4 else 1
        result[name+'_mae'] = ((pred[...,j]-y[...,j]).abs()*mask[...,j]).sum(1)/mask[...,j].sum(1).clamp_min(1)*factor
    pm = mask[...,:4].all(-1); count = pm.sum(1).clamp_min(1)
    pb,tb = pred[...,1]-pred[...,0],y[...,1]-y[...,0]
    result['body_mae_bps'] = ((pb-tb).abs()*pm).sum(1)/count*100
    dead = stats['body_deadband']
    pclass = (pb>dead).long()-(pb < -dead).long(); tclass = (tb>dead).long()-(tb < -dead).long()
    result['body_direction_accuracy'] = ((pclass==tclass)*pm).sum(1)/count
    prange = pb.abs()+pred[...,2]+pred[...,3]; trange = tb.abs()+y[...,2]+y[...,3]
    result['range_mae_bps'] = ((prange-trange).abs()*pm).sum(1)/count*100
    a,t = descriptors(pred,b['sigma']),descriptors(y,b['sigma']); dm = descriptor_mask(mask)
    for j,name in enumerate(DESCRIPTORS):
        cuts = y.new_tensor(stats['bins'][j]); pa = torch.bucketize(a[...,j].contiguous(),cuts); ta = torch.bucketize(t[...,j].contiguous(),cuts)
        result[name+'_bin_accuracy'] = ((pa==ta)*dm[...,j]).sum(1)/dm[...,j].sum(1).clamp_min(1)
        result[name+'_bin_error'] = ((pa-ta).abs()*dm[...,j]).sum(1)/dm[...,j].sum(1).clamp_min(1)
    result['coarse'] = coarse_values(pred,b,stats)
    return result


def shape_summary(pred, b, scales, stats):
    row = {k:float(v.mean()) for k,v in shape_values(pred,b,stats).items()}
    parts = loss_components(pred,b,scales,stats)
    row['original_loss_contributions'] = {k:float(v.mean()) for k,v in parts.items() if k!='coarse'}
    row['coarse_weighted_loss'] = .05*float(parts['coarse'].mean())
    row['channel_support_bars'] = {}
    for j,name in enumerate(CHANNELS):
        valid = b['mask'][...,j]; support = int(valid.sum())
        row['channel_support_bars'][name] = support
        row[name+'_mae'] = float((pred[...,j]-b['y'][...,j]).abs()[valid].mean())*(100 if j<4 else 1) if support else None
    a,t = descriptors(pred,b['sigma']),descriptors(b['y'],b['sigma']); dm = descriptor_mask(b['mask'])
    row['descriptor_classes'] = {}
    for j,name in enumerate(DESCRIPTORS):
        cuts = a.new_tensor(stats['bins'][j]); pa = torch.bucketize(a[...,j].contiguous(),cuts); ta = torch.bucketize(t[...,j].contiguous(),cuts)
        classes = len(cuts)+1
        cm = torch.bincount((ta[dm[...,j]]*classes+pa[dm[...,j]]),minlength=classes**2).reshape(classes,classes)
        support = cm.sum(1); valid = support>0
        recall = [float(cm[c,c]/support[c]) if support[c] else None for c in range(classes)]
        row['descriptor_classes'][name] = dict(confusion=cm.cpu().tolist(),recall=recall,
            balanced_accuracy=float((cm.diag()[valid]/support[valid]).mean()))
    return row


def retention(score, reference):
    limits = dict(close_bps=1.02,body_mae_bps=1.05,change16_mse=1.05,coarse=1.10)
    return all(np.isfinite(v) for v in score.values()) and all(score[k] <= reference[k]*v+1e-8 for k,v in limits.items())


def rank(score): return score['detail'],score['close_bps']


def save_sigmas(cache, split, keys, encoded):
    sigma = np.stack([np.exp(encoded[i]['x'][end-63:end+1,8])*100 for i,end in keys]).astype(np.float32)
    if sigma.shape != (len(keys),64) or not np.isfinite(sigma).all() or (sigma<=0).any(): raise ValueError('Invalid prior bar volatility')
    path = cache/f'{split}_sigma.npy'; np.save(path,sigma,allow_pickle=False)
    return {path.name:sha256(path)}


def arrays_for(meta, out, split, device):
    data = device_arrays(read_arrays(meta['source'],split),device)
    data['sigma'] = torch.tensor(np.load(out/f'cache/{split}_sigma.npy',allow_pickle=False),device=device)
    return data


def measure(model, data, streams, scales, stats, batch, collect=False, joint=True):
    score,z,pred = da.run_epoch(model,data,streams,joint,True,scales,batch,collect=True)
    shapes = shape_values(torch.tensor(pred,device=data['z'].device),data,stats)
    selection = dict(score,body_mae_bps=float(shapes['body_mae_bps'].mean()),coarse=float(shapes['coarse'].mean()))
    return score,selection,z if collect else None,pred if collect else None


def prepare(meta, out, device):
    da.prepare(meta,out,device,extra_builder=save_sigmas)
    train = arrays_for(meta,out,'train','cpu')
    stats = fit_statistics(train['y'],train['mask'],train['sigma'])
    aux = json.loads((out/'cache/statistics.json').read_text())
    model = da.initial_model(meta,device).eval(); va = arrays_for(meta,out,'val',device)
    _,reference,_,_ = measure(model,va,da.PackedStreams(out/'cache','val'),aux['scales'],stats,meta['streams'],joint=False)
    stats['reference'] = reference
    atomic_json(stats,out/'wick_statistics.json')


def publish(state, path):
    da.publish(state,path)
    atomic_save(dict(metadata=state['metadata'],epoch=state['diagnostic_epoch'],model=state['diagnostic_model'],
                     validation=state['diagnostic_validation'],selection=state['diagnostic_selection']),path/'diagnostic_best.pt')
    # Keep compatibility with existing evaluation; custom retention uses selection.
    atomic_save(dict(metadata=state['metadata'],epoch=state['best_epoch'],model=state['best_model'],
                     validation=state['best_validation'],selection=state['best_selection']),path/'best.pt')


def worker(out, name, device='cuda'):
    meta = json.loads((out/'manifest.json').read_text()); aux = json.loads((out/'cache/statistics.json').read_text())
    stats = json.loads((out/'wick_statistics.json').read_text()); reference = stats['reference']
    job = next(j for j in meta['experiments'] if j['name']==name); path = out/name; path.mkdir(exist_ok=True)
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = da.initial_model(meta,device); model.configure(True)
    opt = torch.optim.AdamW([dict(params=model.head.parameters(),lr=meta['decoder_lr']),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad],lr=meta['encoder_lr'])],weight_decay=.01)
    arrays = {s:arrays_for(meta,out,s,device) for s in ('train','val')}; streams = {s:da.PackedStreams(out/'cache',s) for s in arrays}
    metadata = dict(manifest=meta,job=job)
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata: raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    else:
        score,sel,_,_ = measure(model,arrays['val'],streams['val'],aux['scales'],stats,meta['streams'])
        if not retention(sel,reference): raise ValueError('Initial model fails common retention')
        weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state = dict(metadata=metadata,epoch=0,best_epoch=0,history=[],best_model=weights,best_validation=score,best_selection=sel,
                     diagnostic_epoch=0,diagnostic_model=weights,diagnostic_validation=score,diagnostic_selection=sel)
    for epoch in range(state['epoch']+1,meta['epochs']+1):
        started = time.perf_counter()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        loss_fn = lambda pred,b,parts: objective(pred,b,aux['scales'],stats,job['objective'])
        train = da.run_epoch(model,arrays['train'],streams['train'],True,True,aux['scales'],meta['streams'],
                            opt,job['seed']+epoch,objective_fn=loss_fn)[0]
        score,sel,_,_ = measure(model,arrays['val'],streams['val'],aux['scales'],stats,meta['streams'])
        if retention(sel,reference) and rank(sel)<rank(state['best_selection']):
            state.update(best_epoch=epoch,best_validation=score,best_selection=sel,best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if rank(sel)<rank(state['diagnostic_selection']):
            state.update(diagnostic_epoch=epoch,diagnostic_validation=score,diagnostic_selection=sel,diagnostic_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        seconds = time.perf_counter()-started
        state['history'].append(dict(epoch=epoch,train=train,validation=score,selection=sel,eligible=retention(sel,reference),seconds=seconds,
            remaining_minutes=seconds*(meta['epochs']-epoch)/60,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None))
        state.update(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state()); publish(state,path)
        progress(f'{name} epoch={epoch}/{meta["epochs"]} detail={sel["detail"]:.5f} body={sel["body_mae_bps"]:.2f}bp eligible={retention(sel,reference)} best={state["best_epoch"]} diagnostic={state["diagnostic_epoch"]} seconds={seconds:.1f}')
    publish(state,path)


def gradient_audit(meta, out, device):
    """Fixed training examples: gradient at decoder input, not full encoder gradients."""
    stats = json.loads((out/'wick_statistics.json').read_text()); aux = json.loads((out/'cache/statistics.json').read_text())
    data = arrays_for(meta,out,'train',device); ids = np.linspace(0,len(data['z'])-1,min(128,len(data['z'])),dtype=int).tolist()
    b = {k:v[ids] for k,v in data.items()}; model = da.initial_model(meta,device).eval()
    with torch.enable_grad():
        z = b['z'].detach().clone().requires_grad_(True); pred = model.head.decode_recent(z)
        parts = loss_components(pred,b,aux['scales'],stats)
        terms = dict(price=parts[CHANNELS[0]]+parts[CHANNELS[1]]+parts['change1']+parts['change4_16']+parts['detail'],
                     exact_wicks=parts[CHANNELS[2]]+parts[CHANNELS[3]],coarse_weighted=.05*parts['coarse'])
        gs = {k:torch.autograd.grad(v.mean(),z,retain_graph=True)[0] for k,v in terms.items()}
        result = dict(rows=ids,scope='Fixed original-model training endpoints in eval mode; dL/d embedding at decoder input. Norms do not measure full encoder gradient or prove a bottleneck.',terms={})
        for k,g in gs.items():
            base = gs['price']; cosine = float((g*base).sum()/(g.norm()*base.norm()).clamp_min(1e-12))
            result['terms'][k] = dict(loss=float(terms[k].mean().detach()),latent_gradient_norm=float(g.norm()),cosine_with_price=cosine)
        result['objectives'] = {mode:float(objective(pred,b,aux['scales'],stats,mode).mean().detach()) for mode in MODES}
    atomic_json(result,out/'gradient_audit.json')


@torch.no_grad()
def evaluate(out, device):
    da.evaluate(out,device,contrasts=CONTRASTS,schema=SCHEMA,report_name='wick_metrics.json',title='影线精度取舍',
        scope='All variants use identical EMA8/32 inputs and joint detail training. Change only wick objective. Common path/body/coarse retention, old exact loss still reported; two finetuning seeds, reused research test period.')
    meta = json.loads((out/'manifest.json').read_text()); stats = json.loads((out/'wick_statistics.json').read_text())
    aux = json.loads((out/'cache/statistics.json').read_text()); report = json.loads((out/'wick_metrics.json').read_text())
    data = {s:arrays_for(meta,out,s,device) for s in ('val','test')}; streams = {s:da.PackedStreams(out/'cache',s) for s in data}
    inventory = json.loads((out/'cache/test_inventory.json').read_text()); weeks = np.array([x['week'] for x in inventory]); rows_by_name = {}
    for job in [dict(name='original')]+meta['experiments']:
        name = job['name']; model = da.initial_model(meta,device).eval(); record = report['variants'][name]
        if name != 'original':
            ck = torch.load(out/name/'best.pt',map_location='cpu',weights_only=True); model.load_state_dict(ck['model'])
        _,selection,_,_ = measure(model,data['val'],streams['val'],aux['scales'],stats,meta['streams'],joint=name!='original')
        if name != 'original' and not np.allclose(list(selection.values()),list(ck['selection'].values()),rtol=1e-4,atol=1e-5):
            raise ValueError('Wick selection failed to reproduce')
        _,_,_,pred = measure(model,data['test'],streams['test'],aux['scales'],stats,meta['streams'],collect=True,joint=name!='original')
        pred = torch.tensor(pred,device=device)
        record.update(validation_retained=retention(selection,stats['reference']),selection=selection,
                      shape=shape_summary(pred,data['test'],aux['scales'],stats))
        per = {k:v.cpu().tolist() for k,v in shape_values(pred,data['test'],stats).items()}
        rows_by_name[name] = per; atomic_json(per,out/name/'shape_per_window.json')
        if name != 'original':
            ck = torch.load(out/name/'diagnostic_best.pt',map_location='cpu',weights_only=True)
            if ck['metadata'] != dict(manifest=meta,job=job): raise ValueError('Diagnostic checkpoint identity mismatch')
            model.load_state_dict(ck['model'])
            _,sel,_,_ = measure(model,data['val'],streams['val'],aux['scales'],stats,meta['streams'])
            if not np.allclose(list(sel.values()),list(ck['selection'].values()),rtol=1e-4,atol=1e-5): raise ValueError('Diagnostic validation failed to reproduce')
            score,_,_,pred = measure(model,data['test'],streams['test'],aux['scales'],stats,meta['streams'],collect=True)
            record['diagnostic_candidate'] = dict(epoch=ck['epoch'],selection=sel,validation_retained=retention(sel,stats['reference']),
                test_objectives=score,shape=shape_summary(torch.tensor(pred,device=device),data['test'],aux['scales'],stats),
                scope='Selected on validation detail WITHOUT retention gates; diagnostic only, not a replacement candidate.')
        atomic_json(record,out/name/'metrics.json')
        atomic_json(report,out/'wick_metrics.json')
    report['paired_shape'] = {}
    for seed in meta['seeds']:
        for a,b in CONTRASTS:
            an,bn = f'{a}_s{seed}',f'{b}_s{seed}'; paired = {}
            for k in rows_by_name[an]:
                item = paired_error_interval(np.array(rows_by_name[an][k]),np.array(rows_by_name[bn][k]),weeks)
                item['delta_mean'] = item.pop('delta_close_mae_bps')
                item['interpretation'] = 'Candidate minus reference; errors lower, accuracy higher. Weekly paired research bootstrap.'
                paired[k] = item
            report['paired_shape'][an+'_minus_'+bn] = paired
    report['wick_statistics'] = stats
    report['limits'] = 'Reduced exact-wick loss is an intentional tradeoff, not evidence of better embeddings. Judge unchanged price metrics, body/range/coarse morphology, state readout and both seeds jointly. Coarse bins describe candle geometry, not validated trading patterns. No automatic deployment.'
    report['improved_candidates'] = [j['name'] for j in meta['experiments'] if report['variants'][j['name']]['selected_epoch']>0 and report['variants'][j['name']]['validation_retained']]
    atomic_json(report,out/'wick_metrics.json')
    lines = ['# 影线精度取舍：固定输入和非影线系数','','| 组 | 选轮 | 达标 | 收盘MAE bp | 实体MAE bp | 上影MAE bp | 下影MAE bp | 状态BA |','|---|---:|---|---:|---:|---:|---:|---:|']
    for name,r in report['variants'].items():
        sh = r['shape']; close = r['groups']['all']['metrics']['close_mae_bps']['mean']
        lines.append(f'| {name} | {r["selected_epoch"]} | {r["validation_retained"]} | {close:.3f} | {sh["body_mae_bps"]:.3f} | {sh["upper_log_percent_mae"]:.3f} | {sh["lower_log_percent_mae"]:.3f} | {r["state_probe"]["test"]["ba"]:.2%} |')
    lines += ['','diagnostic_candidate另列按验证细节误差选出的候选，即使未达标也保留真实训练后的指标。第0轮表示选回原模型，不代表整组未训练。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('all','worker','evaluate')); p.add_argument('--out',required=True)
    for name in ('source','short-run','long-run','fusion-run','root','name'): p.add_argument('--'+name)
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--streams',type=int,default=64); p.add_argument('--jobs',type=int,default=2)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise ValueError('Formal wick training requires CUDA')
    torch.set_num_threads(4); out = Path(a.out).resolve()
    if a.action=='worker':
        if not a.name: p.error('--name required')
        worker(out,a.name); return
    if not all((a.source,a.short_run,a.long_run,a.fusion_run,a.root)): p.error('All source paths required')
    if min(a.epochs,a.streams,a.jobs)<1 or a.jobs>4: raise ValueError('Invalid runtime settings')
    source,short,long,fusion = map(lambda x:Path(x).resolve(),(a.source,a.short_run,a.long_run,a.fusion_run))
    identity = identity_for(source,short,long,fusion)
    meta = dict(schema=SCHEMA,source=str(source),short_run=str(short),long_run=str(long),root=str(Path(a.root).resolve()),sources=identity,
        config=dict(width=512,decoder_width=256),seeds=[42,43],epochs=a.epochs,streams=a.streams,encoder_lr=3e-5,decoder_lr=1e-4,detail_weight=.25,
        experiments=[dict(name=f'{mode}_s{seed}',objective=mode,mode='joint_detail',seed=seed) for seed in (42,43) for mode in MODES],
        input='Unchanged18 features including original EMA8/32; original encoder21 and parallel head33, identical initialization.',
        objectives=dict(original_loss='recent_loss + .25 normalized1/4/16 detail',wick_low='Original minus .8 exact-wick channel contribution; denominator unchanged',wick_coarse='wick_low + .05 train-tertile interval loss for wick volatility units and fractions'),
        selection='Min validation detail, tie closeMAE. Gates vs original: close<=1.02, bodyMAE<=1.05, change16MSE<=1.05, coarse<=1.10. Old exact loss reported but not gated. Save unqualified diagnostic best separately.',
        protocol='Fixed data/endpoints/TBPTT128, same seeds/dropout/grouping and train budget. No extra head or width. Test reused research period.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta: raise ValueError('Settings changed; choose new output directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty directory lacks manifest')
    out.mkdir(parents=True,exist_ok=True); atomic_json(meta,out/'manifest.json')
    for name in ('completion.json','wick_metrics.json','summary.md'): (out/name).unlink(missing_ok=True)
    progress('Preparing chronological data, train-only wick bins and common validation reference')
    prepare(meta,out,'cuda'); torch.cuda.empty_cache()
    atomic_json(dict(jobs=a.jobs,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)),out/'runtime.json')
    gradient_audit(meta,out,'cuda'); torch.cuda.empty_cache()
    if a.action=='all':
        atomic_json(preflight(meta,out,'cuda'),out/'preflight.json'); torch.cuda.empty_cache()
        run_jobs(out,a.jobs,'obson.babel.wick_ablation')
    evaluate(out,'cuda')
    if identity_for(source,short,long,fusion)!=identity: raise ValueError('Source artifacts changed during experiment')
    report = json.loads((out/'wick_metrics.json').read_text())
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),source_weights_unchanged=True,improved_candidates=report['improved_candidates']),out/'completion.json')
    progress(f'Wick ablation complete: {out}')


def preflight(meta, out, device):
    stats = json.loads((out/'wick_statistics.json').read_text()); scales = json.loads((out/'cache/statistics.json').read_text())['scales']; results = {}
    for mode in MODES:
        model = da.initial_model(meta,device); model.configure(True); model.train()
        x = torch.randn(meta['streams'],128,18,device=device); y = torch.randn(meta['streams'],64,7,device=device)*.1; y[...,2:] = y[...,2:].abs()
        b = dict(y=y,mask=torch.ones_like(y,dtype=torch.bool),sigma=torch.ones_like(y[...,0])*.1)
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        h,_ = model.encoder(x); pred = model.head.decode_recent(h[:,-1]); loss = objective(pred,b,scales,stats,mode).mean(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss): raise ValueError('Nonfinite wick preflight')
        results[mode] = dict(loss=float(loss.detach()),gradient_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None)
        del model,x,y,b,h,pred,loss
    return dict(variants=results,scope='Synthetic full-chunk backward only; no optimizer step; excludes other workers and optimizer states.')


if __name__=='__main__': main()
