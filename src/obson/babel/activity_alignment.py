"""Preserve observed activity history in causal embeddings; no future targets."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import activity_ablation as aa
from . import detail_alignment as da
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import read_arrays, device_arrays, run_jobs, paired_error_interval
from .dual_state import sha256, verify_files
from .ema_ablation import identity_for
from .progress import progress

SCHEMA = 'babel-activity-alignment-v1'
MODES = {'control': 0., 'aux005': .05, 'aux020': .20}
CONTRASTS = (('aux005', 'control'), ('aux020', 'control'), ('aux020', 'aux005'))
NAMES = [f'{h}/{k}' for h in ('current', 'past1_16', 'lag4', 'lag16') for k in aa.ACTIVITY[:5]]
GOAL = dict(overall='Reusable causal embedding compressing observed price/activity history, suitable for later readouts and historical reconstruction.',
    stage='Test whether explicit activity-history supervision improves retained information without damaging price representation.',
    success='Both seeds improve past-only activity readouts, with paired weekly evidence and price/state retention; current copying or lower training loss alone does not count.',
    exclusions='No future prediction, simulated aggressor flow, opening/closing labels, larger encoder, or automatic model replacement.',
    limitations='Same pretrained encoder and reused research test period; two finetuning seeds, not a fresh final holdout.')


def targets_from_bank(x, specs, n):
    """Only observed rows from this partition/contract, and no current row in past1_16."""
    y = np.zeros((n, 20), np.float32); mask = np.zeros((n, 20), bool)
    clean = np.zeros(n, bool); current = np.zeros((n, 10), np.float32); seen = np.zeros(n, bool)
    for s in specs:
        bank = x[s['offset']:s['offset']+s['length'], 18:28]
        for end, idx in s['endpoints']:
            if end < 16 or end >= len(bank) or seen[idx]: raise ValueError('Invalid activity endpoint')
            seen[idx] = True; a = np.array(bank[end-16:end+1], copy=True)
            valid = np.column_stack((np.ones(17, bool), a[:, 9]>0, a[:, 5]>0, a[:, 6]>0, a[:, 7]>0))
            # Observed inconsistencies, not inferred trade labels. Keep raw inputs;
            # mask suspect OI-change targets and report a separate clean subset.
            zero_volume_delta = (a[:, 5]>0)&(a[:, 4]==0)&(a[:, 2]!=0)
            extreme_ratio = (a[:, 6]>0)&(np.abs(a[:, 3])>np.arcsinh(1.)+1e-6)
            suspect = zero_volume_delta|extreme_ratio
            valid[suspect, 2:4] = False
            rows = [a[-1, :5], a[:-1, :5].mean(0), a[-5, :5], a[0, :5]]
            masks = [valid[-1], valid[:-1].all(0), valid[-5], valid[0]]
            y[idx] = np.concatenate(rows); mask[idx] = np.concatenate(masks)
            clean[idx] = not suspect.any()
            current[idx] = a[-1]  # Values and explicit missing/source masks.
    if not seen.all(): raise ValueError('Incomplete activity target coverage')
    return y, mask, clean, current


def fit_target_stats(y, mask):
    count = mask.sum(0)
    if (count < 2).any(): raise ValueError('Insufficient activity training support')
    mean = np.where(mask, y, 0).sum(0)/count
    std = np.sqrt(np.where(mask, (y-mean)**2, 0).sum(0)/count)
    active = std > 1e-6
    if not active.any(): raise ValueError('All activity targets constant')
    return dict(mean=mean.tolist(), scale=np.maximum(std, 1e-5).tolist(), active=active.tolist(), support=count.tolist())


def verify_alignment_cache(cache, files, upstream):
    """Only the three explicitly pinned input banks may resolve outside cache."""
    linked = {f'{s}_x.npy' for s in da.SPLITS}
    for name in linked:
        path = cache/name
        if (name not in files or not path.is_symlink() or
                path.resolve() != (upstream/'cache'/name).resolve() or sha256(path) != files[name]):
            raise ValueError(f'Activity input bank identity mismatch: {name}')
    verify_files(cache, {k:v for k,v in files.items() if k not in linked})


def prepare(meta, out):
    upstream = Path(meta['activity_run']); src = upstream/'cache'; dst = out/'cache'; dst.mkdir(exist_ok=True)
    index = dst/'index.json'
    if index.exists():
        info = json.loads(index.read_text())
        if info['manifest'] != meta: raise ValueError('Target cache identity mismatch')
        verify_alignment_cache(dst, info['files'], upstream); return
    info = json.loads((src/'index.json').read_text()); verify_files(src, info['files'])
    files = {}; supports = {}
    for split in da.SPLITS:
        # Read-only symlinks avoid duplicating the multi-million-bar feature bank.
        name = f'{split}_x.npy'; target = dst/name
        if not target.exists(): target.symlink_to((src/name).resolve())
        if target.resolve() != (src/name).resolve(): raise ValueError('Feature bank destination mismatch')
        specs = json.loads((src/f'{split}_sequences.json').read_text())
        shutil.copyfile(src/f'{split}_sequences.json', dst/f'{split}_sequences.json')
        n = len(read_arrays(meta['source'], split)['z'])
        y, mask, clean, current = targets_from_bank(np.load(target, mmap_mode='r'), specs, n)
        for suffix, values in [('activity', y), ('activity_mask', mask), ('clean', clean), ('current', current)]:
            np.save(dst/f'{split}_{suffix}.npy', values, allow_pickle=False)
            files[f'{split}_{suffix}.npy'] = sha256(dst/f'{split}_{suffix}.npy')
        files[name] = sha256(target); files[f'{split}_sequences.json'] = sha256(dst/f'{split}_sequences.json')
        supports[split] = dict(endpoints=n, clean_endpoints=int(clean.sum()), target_support=mask.sum(0).tolist())
        if split == 'train': stats = fit_target_stats(y, mask)
    for name in ('statistics.json', 'test_inventory.json', 'activity_audit.json'):
        shutil.copyfile(src/name, dst/name); files[name] = sha256(dst/name)
    atomic_json(stats, dst/'activity_statistics.json'); files['activity_statistics.json'] = sha256(dst/'activity_statistics.json')
    shutil.copyfile(upstream/'coverage.json', out/'coverage.json')
    atomic_json(dict(names=NAMES, splits=supports, anomaly_policy='Mask OI-change targets if zero volume with nonzero delta or abs(delta)/volume>1; also report endpoints whose entire17-bar target span is clear. No raw input removal.'), out/'target_audit.json')
    atomic_json(dict(manifest=meta, files=files), index)


def initial_model(meta, device, job):
    if job.get('name') == 'original': return da.initial_model(meta, device)
    model = aa.initial_model(meta, device, dict(activity='features'))
    # Same auxiliary head and RNG consumption in every cell, even lambda=0.
    model.activity_head = nn.Sequential(nn.LayerNorm(meta['config']['width']), nn.Linear(meta['config']['width'], 20)).to(device)
    return model


def arrays_for(meta, out, split, device):
    data = device_arrays(read_arrays(meta['source'], split), device)
    stats = json.loads((out/'cache/activity_statistics.json').read_text())
    y = np.load(out/f'cache/{split}_activity.npy'); mask = np.load(out/f'cache/{split}_activity_mask.npy')
    mask = mask & np.array(stats['active'])
    data['activity'] = torch.tensor((y-np.array(stats['mean']))/np.array(stats['scale']), dtype=torch.float32, device=device)
    data['activity_mask'] = torch.tensor(mask, device=device)
    return data


def activity_loss(pred, y, mask):
    err = nn.functional.smooth_l1_loss(pred, y, reduction='none')
    # Equal weighting per target group; a row with no valid entries contributes0.
    losses = (err*mask).reshape(-1, 4, 5).sum(-1)/mask.reshape(-1, 4, 5).sum(-1).clamp_min(1)
    return losses.mean(-1)


def latent_objective(model, weight):
    def objective(z, b, parts):
        pred = model.activity_head(z)
        loss = activity_loss(pred, b['activity'], b['activity_mask'])
        parts['activity'] = loss
        parts['past_activity'] = activity_loss(pred, b['activity'], b['activity_mask'] &
            (torch.arange(20, device=z.device)[None, :]>=5)) * (4/3)
        return weight*loss
    return objective


def worker(out, name, device='cuda'):
    meta = json.loads((out/'manifest.json').read_text()); stats = json.loads((out/'cache/statistics.json').read_text())
    job = next(j for j in meta['experiments'] if j['name']==name); path = out/name; path.mkdir(exist_ok=True)
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    model = initial_model(meta, device, job); model.configure(True)
    groups = [dict(params=model.head.parameters(), lr=meta['decoder_lr']),
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=meta['encoder_lr']),
        dict(params=model.activity_head.parameters(), lr=meta['decoder_lr'])]
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    data = {s:arrays_for(meta, out, s, device) for s in ('train', 'val')}
    streams = {s:aa.ActivityStreams(out/'cache', s, job) for s in ('train', 'val')}
    callback = latent_objective(model, job['aux_weight']); metadata = dict(manifest=meta, job=job)
    def epoch(split, training=False, seed=None):
        return da.run_epoch(model, data[split], streams[split], True, True, stats['scales'], meta['streams'],
            opt if training else None, seed, latent_objective_fn=callback)[0]
    if (path/'last.pt').exists():
        state = torch.load(path/'last.pt', map_location='cpu', weights_only=True)
        if state['metadata'] != metadata: raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    else:
        score = epoch('val')
        if not da.eligible(score, stats['reference']): raise ValueError('Warm start fails retention')
        state = dict(metadata=metadata, epoch=0, best_epoch=0, history=[], best_validation=score,
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    def save(): da.publish(state, path)
    for e in range(state['epoch']+1, meta['epochs']+1):
        start = time.perf_counter()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        train = epoch('train', True, job['seed']+e); score = epoch('val')
        if da.improves(score, state['best_validation'], stats['reference']):
            state.update(best_epoch=e, best_validation=score, best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        seconds = time.perf_counter()-start
        state['history'].append(dict(epoch=e, train=train, validation=score, eligible=da.eligible(score,stats['reference']), seconds=seconds,
            remaining_minutes=seconds*(meta['epochs']-e)/60, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None))
        state.update(epoch=e, model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state()); save()
        progress(f'{name} epoch={e}/{meta["epochs"]} close={score["close_bps"]:.2f} detail={score["detail"]:.5f} activity={score["activity"]:.4f} best={state["best_epoch"]}')
    save()


def score_probe(pred, targets, masks, stats):
    errors = np.full(targets.shape, np.nan); rows = []
    means = np.array(stats['mean']); scales = np.array(stats['scale'])
    for j in range(targets.shape[1]):
        valid = masks[:, j] & stats['active'][j]; y = (targets[valid, j]-means[j])/scales[j]
        if len(y)<2: rows.append(dict(support=int(len(y)), skipped='insufficient support')); continue
        err = (pred[valid, j]-y)**2; errors[valid, j] = err
        total = np.square(y-y.mean()).sum()
        rows.append(dict(support=len(y), test_normalized_mse=float(err.mean()), test_r2=1-float(err.sum()/total) if total>1e-12 else None,
            train_mean_baseline_mse=float(np.square(y).mean())))
    return dict(names=NAMES, targets=rows), errors


def nonlinear_probe(zs, ys, masks, device, epochs=80):
    """Frozen inputs, train-only normalization, validation-selected epoch/decay; GPU only."""
    if device!='cuda': raise ValueError('Nonlinear readout training requires CUDA')
    stats = fit_target_stats(ys[0], masks[0]); mean = zs[0].mean(0); scale = zs[0].std(0).clip(1e-5)
    x = [torch.tensor((z-mean)/scale, dtype=torch.float32, device=device) for z in zs]
    y = [torch.tensor((a-np.array(stats['mean']))/np.array(stats['scale']), dtype=torch.float32, device=device) for a in ys]
    mask = [torch.tensor(a & np.array(stats['active']), device=device) for a in masks]
    best = None
    # Same fixed probe seed and search budget for every representation.
    with torch.enable_grad():
        for decay in (.001, .01):
            torch.manual_seed(1701)
            model = nn.Sequential(nn.Linear(x[0].shape[1], 128), nn.GELU(), nn.Linear(128,20)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay)
            for e in range(epochs+1):
                if e:
                    model.train(); order = torch.randperm(len(x[0]),device=device)
                    for ids in order.split(256):
                        loss = activity_loss(model(x[0][ids]), y[0][ids], mask[0][ids]).mean()
                        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                model.eval()
                with torch.no_grad():
                    err = (model(x[1])-y[1]).square()
                    valid = mask[1].sum(0)>0
                    per_target = (err*mask[1]).sum(0)/mask[1].sum(0).clamp_min(1)
                    score = float(per_target[valid].mean())
                if best is None or score<best['score']:
                    best = dict(score=score, epoch=e, decay=decay, state={k:v.detach().clone() for k,v in model.state_dict().items()})
                if e % 20 == 0:
                    progress(f'Frozen MLP readout decay={decay} epoch={e}/{epochs} validation_mse={score:.4f}')
    model.load_state_dict(best['state']); model.eval()
    with torch.no_grad(): pred = model(x[2]).cpu().numpy()
    report, errors = score_probe(pred, ys[2], masks[2], stats)
    report.update(validation_normalized_mse=best['score'], selected_epoch=best['epoch'], weight_decay=best['decay'], probe_seed=1701,
        scope='Frozen encoder, newly fitted128-hidden MLP; train-only normalizers and validation-only model selection.')
    return report, errors


def ridge_probe(zs, ys, masks):
    report, errors = aa.fit_activity_probe(zs, ys, masks); report['names'] = NAMES
    return report, errors


def error_summary(errors, clean):
    def summarize(a):
        valid = np.isfinite(a); count = valid.sum(0); means = np.nansum(a,axis=0)/np.maximum(count,1)
        return dict(per_target=[float(v) if n else None for v,n in zip(means,count)],support=count.tolist(),
            current=float(means[:5][count[:5]>0].mean()) if (count[:5]>0).any() else None,
            past_only=float(means[5:][count[5:]>0].mean()) if (count[5:]>0).any() else None)
    return dict(all=summarize(errors), clean=summarize(errors[clean]))


def paired_errors(a, b, weeks, clean):
    result = {}
    for subset, keep in [('all', np.ones(len(a),bool)), ('clean',clean)]:
        rows = {}
        for j,name in enumerate(NAMES):
            ok = keep & np.isfinite(a[:,j]) & np.isfinite(b[:,j])
            if ok.sum()<2: rows[name] = dict(support=int(ok.sum())); continue
            row = paired_error_interval(a[ok,j], b[ok,j], weeks[ok]); row['delta_normalized_mse'] = row.pop('delta_close_mae_bps')
            row['support'] = int(ok.sum()); rows[name] = row
        # Predeclared primary endpoint: equal-weight past-only descriptors,
        # using windows where all15 are valid (avoids changing target mixtures).
        ok = keep & np.isfinite(a[:,5:]).all(1) & np.isfinite(b[:,5:]).all(1)
        if ok.sum()>1:
            row = paired_error_interval(a[ok,5:].mean(1),b[ok,5:].mean(1),weeks[ok])
            row['delta_normalized_mse'] = row.pop('delta_close_mae_bps'); row['support'] = int(ok.sum()); rows['past_only_primary'] = row
        result[subset] = rows
    return result


def evaluate(out, device):
    meta = json.loads((out/'manifest.json').read_text()); cache = out/'cache'
    da.evaluate(out, device, model_factory=initial_model, streams_factory=aa.ActivityStreams, contrasts=CONTRASTS,
        schema=SCHEMA, report_name='alignment_metrics.json', title='量仓历史保留：训练目标对齐', scope=meta.get('goal',GOAL)['stage'])
    report = json.loads((out/'alignment_metrics.json').read_text()); report['goal'] = meta.get('goal',GOAL)
    ys = [np.load(cache/f'{s}_activity.npy') for s in da.SPLITS]; masks = [np.load(cache/f'{s}_activity_mask.npy') for s in da.SPLITS]
    clean = np.load(cache/'test_clean.npy'); weeks = np.array([r['week'] for r in json.loads((cache/'test_inventory.json').read_text())])
    stats = json.loads((cache/'statistics.json').read_text()); astats = json.loads((cache/'activity_statistics.json').read_text())
    errors = {}
    for job in [dict(name='original')]+meta['experiments']:
        name = job['name']; progress(f'Readouts for {name}: linear and nonlinear, current and past-only')
        model = initial_model(meta,device,job).eval()
        if name!='original':
            ck = torch.load(out/name/'best.pt', map_location='cpu',weights_only=True); model.load_state_dict(ck['model'])
        zs = []
        for split in da.SPLITS:
            data = device_arrays(read_arrays(meta['source'],split),device)
            _,z,pred = da.run_epoch(model,data,aa.ActivityStreams(cache,split,job),name!='original',True,stats['scales'],meta['streams'],collect=True)
            zs.append(z)
        row = report['variants'][name]; errors[name] = {}
        row['candle_activity_reconstruction'] = aa.reconstruction_diagnostics(pred,data)
        for kind,fn in [('linear',ridge_probe),('nonlinear',lambda z,y,m:nonlinear_probe(z,y,m,device,meta['probe_epochs']))]:
            value,err = fn(zs,ys,masks); value['normalized_error_summary'] = error_summary(err,clean)
            row[kind+'_activity_readout'] = value; errors[name][kind] = err
        if name!='original' and job['aux_weight']>0:
            with torch.no_grad(): pred = model.activity_head(torch.tensor(zs[2],device=device)).cpu().numpy()
            value,err = score_probe(pred,ys[2],masks[2],astats); value['normalized_error_summary'] = error_summary(err,clean)
            row['trained_auxiliary_head'] = value
        atomic_json(row,out/name/'metrics.json'); atomic_json(report,out/'alignment_metrics.json')
    # Instantaneous inputs quantify how much past-only readout can be explained
    # by persistence/current observations without a recurrent state.
    current = [np.load(cache/f'{s}_current.npy') for s in da.SPLITS]
    report['current_input_control'] = {}; errors['current_input'] = {}
    for kind,fn in [('linear',ridge_probe),('nonlinear',lambda z,y,m:nonlinear_probe(z,y,m,device,meta['probe_epochs']))]:
        value,err = fn(current,ys,masks); value['normalized_error_summary'] = error_summary(err,clean)
        report['current_input_control'][kind] = value; errors['current_input'][kind] = err
    report['paired_activity'] = {}
    for seed in meta['seeds']:
        for a,b in CONTRASTS:
            an,bn = f'{a}_s{seed}',f'{b}_s{seed}'
            report['paired_activity'][an+'_minus_'+bn] = {kind:paired_errors(errors[an][kind],errors[bn][kind],weeks,clean) for kind in ('linear','nonlinear')}
    report['versus_current_input'] = {name:{kind:paired_errors(err[kind],errors['current_input'][kind],weeks,clean) for kind in ('linear','nonlinear')} for name,err in errors.items() if name!='current_input'}
    report['stage_decisions'] = {}
    for mode in ('aux005','aux020'):
        rows = []
        for seed in meta['seeds']:
            name,base = f'{mode}_s{seed}',f'control_s{seed}'; a,b = report['variants'][name],report['variants'][base]
            price = a['test_objectives']['close_bps']<=b['test_objectives']['close_bps']*1.02 and a['test_objectives']['detail']<=b['test_objectives']['detail']*1.02
            state = a['state_probe']['test']['ba']>=b['state_probe']['test']['ba']-.01
            pair = report['paired_activity'][name+'_minus_'+base]
            bounds = [pair[k][subset].get('past_only_primary',{}).get('high') for k in ('linear','nonlinear') for subset in ('all','clean')]
            supported = all(v is not None and v<0 for v in bounds)
            current = report['versus_current_input'][name]
            bounds = [current[k][subset].get('past_only_primary',{}).get('high') for k in ('linear','nonlinear') for subset in ('all','clean')]
            beyond_current = all(v is not None and v<0 for v in bounds)
            rows.append(dict(seed=seed, selected_epoch=a['selected_epoch'], price_retained=price,state_retained=state,
                past_only_supported=supported, beyond_current_input=beyond_current))
        report['stage_decisions'][mode] = dict(seeds=rows, evidence_pass=all(r['selected_epoch']>0 and r['price_retained'] and r['state_retained'] and r['past_only_supported'] and r['beyond_current_input'] for r in rows),
            interpretation='Research evidence screen only, not automatic promotion. Both seeds, linear+nonlinear past-only paired upper bounds<0 versus control AND current inputs, clean sensitivity, price/detail<=1.02 of control and state BA drop<=1 percentage point.')
    atomic_json(report,out/'alignment_metrics.json')
    with (out/'summary.md').open('a') as f:
        f.write('\n## Stage goal\n'+json.dumps(meta.get('goal',GOAL),ensure_ascii=False,indent=2)+'\n\n## Evidence screen\n'+json.dumps(report['stage_decisions'],indent=2)+'\n')


def source_identity(activity_run, fusion_run):
    meta = json.loads((activity_run/'manifest.json').read_text())
    complete = json.loads((activity_run/'completion.json').read_text())
    if complete.get('status')!='complete': raise ValueError('Upstream activity experiment incomplete')
    index = json.loads((activity_run/'cache/index.json').read_text())
    if index['manifest'] != meta: raise ValueError('Upstream cache manifest mismatch')
    verify_files(activity_run/'cache',index['files'])
    weights = identity_for(Path(meta['source']),Path(meta['short_run']),Path(meta['long_run']),fusion_run)
    if weights != meta['sources']: raise ValueError('Upstream warm-start artifacts changed')
    return meta, dict(manifest_sha256=sha256(activity_run/'manifest.json'),cache_index_sha256=sha256(activity_run/'cache/index.json'),source_weights=weights)


def preflight(meta, out, device):
    job = meta['experiments'][-1]; model = initial_model(meta,device,job); model.configure(True)
    x = torch.randn(meta['streams'],128,28,device=device); h,_ = model.encoder(x); z = h[:,-1]
    y = torch.randn(len(z),64,7,device=device)*.1; y[...,2:] = y[...,2:].abs()
    b = dict(y=y,mask=torch.ones_like(y,dtype=torch.bool),activity=torch.randn(len(z),20,device=device),activity_mask=torch.ones(len(z),20,dtype=torch.bool,device=device))
    stats = json.loads((out/'cache/statistics.json').read_text()); parts = da.values(model.head.decode_recent(z),b,stats['scales'])
    aux = latent_objective(model,job['aux_weight'])(z,b,parts)
    loss = (parts['base']+.25*parts['detail']+aux).mean(); loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not model.activity_head[-1].weight.grad.abs().sum(): raise ValueError('No auxiliary gradient')
    return dict(gradient_norm=float(norm),parameters=sum(p.numel() for p in model.parameters()),scope='Synthetic backward only; no optimizer step.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--activity-run');p.add_argument('--name');p.add_argument('--fusion-run',default='checkpoints/babel_fusion512_s42')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--streams',type=int,default=64);p.add_argument('--jobs',type=int,default=2);p.add_argument('--probe-epochs',type=int,default=80)
    a=p.parse_args();out=Path(a.out).resolve()
    if not torch.cuda.is_available(): raise ValueError('Formal training/readout experiments require CUDA')
    torch.set_num_threads(4)
    if a.action=='worker':
        if not a.name: p.error('--name required')
        worker(out,a.name);return
    if not a.activity_run:p.error('--activity-run required')
    if min(a.epochs,a.streams,a.jobs,a.probe_epochs)<1 or a.jobs>4:raise ValueError('Invalid runtime settings')
    activity=Path(a.activity_run).resolve();fusion=Path(a.fusion_run).resolve();up,identity=source_identity(activity,fusion)
    meta=dict(schema=SCHEMA,activity_run=str(activity),fusion_run=str(fusion),upstream=identity,source=up['source'],short_run=up['short_run'],long_run=up['long_run'],
        config=up['config'],seeds=[42,43],epochs=a.epochs,streams=a.streams,encoder_lr=3e-5,decoder_lr=1e-4,detail_weight=.25,probe_epochs=a.probe_epochs,goal=GOAL,
        experiments=[dict(name=f'{mode}_s{seed}',activity='features',mode='joint_detail',seed=seed,aux_weight=weight) for seed in (42,43) for mode,weight in MODES.items()],
        selection='Identical validation price-detail selection with original retention gates for all weights, including epoch0. Auxiliary/test readouts never select encoder checkpoints.',
        auxiliary='Train-standardized SmoothL1,20 targets: current/past1..16 excluding current/lag4/lag16 x5descriptors. New head only receives z; no input bypass.',
        anomaly='Mask suspicious OI-change targets; report all and clean endpoint sensitivity. Keep encoder inputs identical across cells.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Settings changed; choose a new output directory')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output lacks manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');atomic_json(GOAL,out/'experiment_goal.json')
    for name in ('completion.json','alignment_metrics.json'):(out/name).unlink(missing_ok=True)
    progress('Stage: preserve observed activity history while retaining price representation')
    prepare(meta,out);atomic_json(dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),jobs=a.jobs),out/'runtime.json')
    if a.action=='all':
        atomic_json(preflight(meta,out,'cuda'),out/'preflight.json');torch.cuda.empty_cache();run_jobs(out,a.jobs,'obson.babel.activity_alignment')
    evaluate(out,'cuda')
    if source_identity(activity,fusion)[1]!=identity:raise ValueError('Source artifacts changed during experiment')
    atomic_json(dict(status='complete',experiments=6,source_artifacts_unchanged=True,goal=GOAL),out/'completion.json')
    progress('Activity alignment complete; evidence screen is not automatic model promotion')


if __name__=='__main__':main()
