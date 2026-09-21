"""Matched-initial-output ablation of EMA context for short-state embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import detail_alignment as da
from .ae_extend import atomic_json
from .ar_codec import FEATURES, INITIAL_SIGMA, SIGMA_FLOOR
from .ae_context import CONTEXT
from .ar_reconstruction import read_arrays, device_arrays, run_jobs
from .dual_state import source_identity, sha256, verify_files
from .progress import progress
from .stage_audit import load_data
from .stream_audit import bounded_difference

SCHEMA = 'babel-ema-ablation-v1'
CONTEXT_MODES = ('none', 'ema8_32', 'multiscale', 'retained_ema8_32')
MULTISCALE = ('close_minus_ema4', 'ema4_minus_ema8', 'ema8_minus_ema16',
              'ema16_minus_ema32', 'ema32_minus_ema64')
CONTRASTS = (('ema8_32', 'none'), ('multiscale', 'none'), ('multiscale', 'ema8_32'),
             ('ema8_32', 'retained_ema8_32'), ('multiscale', 'retained_ema8_32'))


def feature_bank(frame, encoded):
    """Full-contract preprocessing, never reset EMA at window or partition boundaries."""
    p = pd.Series(np.log(frame.close.to_numpy()))
    ema = [p.ewm(span=n, adjust=False).mean().to_numpy() for n in (4, 8, 16, 32, 64)]
    sigma = np.sqrt(np.maximum(np.r_[INITIAL_SIGMA**2, encoded['variance'][:-1]], SIGMA_FLOOR**2))
    raw = np.column_stack([p.to_numpy()-ema[0]]+[a-b for a,b in zip(ema[:-1],ema[1:])])
    multi = np.arcsinh(raw/sigma[:,None]).astype(np.float32)
    x = np.column_stack((encoded['x'], multi))
    if x.shape != (len(frame),23) or not np.isfinite(x).all(): raise ValueError('Invalid EMA feature bank')
    return x


def select_features(bank, mode):
    if mode not in CONTEXT_MODES: raise ValueError('Unknown EMA context')
    if bank.shape[-1] != 23: raise ValueError('Expected shared 23-channel bank')
    x = np.zeros((*bank.shape[:-1],19),np.float32)
    x[...,:14] = bank[...,:14]
    if mode in ('ema8_32','retained_ema8_32'): x[...,14:18] = bank[...,14:18]
    elif mode == 'multiscale': x[...,14:] = bank[...,18:23]
    return x


class ContextStreams(da.PackedStreams):
    def __init__(self, path, split, job):
        super().__init__(path,split)
        self.mode = job.get('context','original')

    def batches(self, batch, seed=None, inputs=True):
        for reset,x,pick in super().batches(batch,seed,inputs):
            if x is not None:
                x = x[...,:18] if self.mode == 'original' else select_features(x,self.mode)
            yield reset,x,pick


def initial_model(meta, device, job):
    if job.get('name') == 'original': return da.initial_model(meta,device)
    mode = job['context']
    if mode not in CONTEXT_MODES: raise ValueError('Unknown EMA initialization')
    model = da.DetailModel(**meta['config'],input_dim=19).to(device)
    state = torch.load(Path(meta['short_run'])/'best.pt',map_location='cpu',weights_only=True)['model']
    old = state['input.0.weight']; weight = old.new_zeros((old.shape[0],19))
    weight[:,:14] = old[:,:14]
    if mode == 'retained_ema8_32': weight[:,14:18] = old[:,14:18]
    state = dict(state); state['input.0.weight'] = weight
    model.encoder.load_state_dict(state)
    model.head.load_state_dict(torch.load(Path(meta['source'])/'short/best.pt',map_location='cpu',weights_only=True)['model'])
    return model


def prepare(meta, out, device):
    cache = out/'cache'; cache.mkdir(exist_ok=True); index = cache/'index.json'
    if index.exists():
        info = json.loads(index.read_text())
        if info['manifest'] != meta: raise ValueError('Prepared data identity changed')
        verify_files(cache,info['files']); progress('Verified prepared EMA bank'); return
    ref,series,encoded,_ = load_data(meta['root'],meta['long_run'])
    banks = [dict(x=feature_bank(s.frame,e)) for s,e in zip(series,encoded)]
    source = Path(meta['source']); files = {}; coverage = {}
    for split in da.SPLITS:
        keys = np.load(source/f'target_cache/{split}_keys.npy',allow_pickle=False)
        x,specs = da.sequence_specs(series,banks,ref['boundaries'],split,keys)
        with (cache/f'{split}_x.tmp').open('wb') as f: np.save(f,x,allow_pickle=False)
        (cache/f'{split}_x.tmp').replace(cache/f'{split}_x.npy')
        atomic_json(specs,cache/f'{split}_sequences.json')
        for name in (f'{split}_x.npy',f'{split}_sequences.json'): files[name] = sha256(cache/name)
        coverage[split] = dict(windows=len(keys),streams=len(specs),replayed_bars=len(x))
        if split == 'test':
            inventory = [dict(key=series[i].key,symbol=series[i].code,period=series[i].period,row=int(end),
                end=str(series[i].frame.datetime.iloc[end]),anchor=float(series[i].frame.close.iloc[end-64]),
                week=str(pd.Timestamp(series[i].sessions[end]).to_period('W-SUN'))) for i,end in keys]
            atomic_json(inventory,cache/'test_inventory.json'); files['test_inventory.json'] = sha256(cache/'test_inventory.json')
    del banks, encoded, series, x
    scales = da.detail_scales(read_arrays(source,'train')['y'])
    va = device_arrays(read_arrays(source,'val'),device)
    old = da.initial_model(meta,device).eval()
    reference = da.run_epoch(old,va,ContextStreams(cache,'val',{}),False,False,scales,meta['streams'])[0]
    validation = {}; neutral_z = None
    for mode in CONTEXT_MODES:
        job = dict(context=mode); model = initial_model(meta,device,job).eval()
        score,z,_ = da.run_epoch(model,va,ContextStreams(cache,'val',job),True,True,scales,meta['streams'],collect=True)
        comparison = bounded_difference(z,va['z'].cpu().numpy()) if mode == 'retained_ema8_32' else None
        if mode == 'none': neutral_z = z
        same_start = bool(np.array_equal(z,neutral_z)) if mode != 'retained_ema8_32' else None
        validation[mode] = dict(score=score,retained=da.eligible(score,reference),
                               cached_comparison=comparison,identical_neutral_output=same_start)
        atomic_json(dict(reference=reference,variants=validation),out/'warm_start_validation.json')
        if mode == 'retained_ema8_32':
            if not comparison['passed'] or not da.eligible(score,reference):
                raise ValueError('Retained EMA replay differs from original cache')
        elif not same_start: raise ValueError('Neutral context variants have different initial outputs')
    atomic_json(dict(scales=scales,reference=reference,scale_units='train-only close-change RMS in log-percent'),cache/'statistics.json')
    files['statistics.json'] = sha256(cache/'statistics.json')
    atomic_json(dict(boundaries=ref['boundaries'],splits=coverage),out/'coverage.json')
    atomic_json(dict(manifest=meta,files=files),index)


def preflight(meta, out, device):
    results = {}; scales = json.loads((out/'cache/statistics.json').read_text())['scales']
    for mode in CONTEXT_MODES:
        model = initial_model(meta,device,dict(context=mode)); model.configure(True); model.train()
        bank = np.random.default_rng(123).normal(size=(meta['streams'],128,23)).astype(np.float32)
        x = torch.tensor(select_features(bank,mode),device=device)
        y = torch.randn(meta['streams'],64,7,device=device)*.1; y[...,2:] = y[...,2:].abs()
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        h,_ = model.encoder(x); pred = model.head.decode_recent(h[:,-1])
        parts = da.values(pred,dict(y=y,mask=torch.ones_like(y,dtype=torch.bool)),scales)
        loss = (parts['base']+meta['detail_weight']*parts['detail']).mean(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss): raise ValueError('Nonfinite synthetic preflight')
        grad = model.encoder.input[0].weight.grad[:,14:]
        if mode == 'none' and torch.count_nonzero(grad): raise ValueError('No-EMA receives context gradients')
        if mode != 'none' and not torch.count_nonzero(grad): raise ValueError('EMA input cannot learn')
        results[mode] = dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            gradient_norm=float(norm),context_gradient_norm=float(grad.norm()),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device == 'cuda' else None)
        del model,x,y,h,pred,parts,loss,grad
    return dict(variants=results,scope='Disposable synthetic backward only; excludes optimizer states and other workers.')


def evaluate(out, device):
    da.evaluate(out,device,model_factory=initial_model,streams_factory=ContextStreams,contrasts=CONTRASTS,
        schema=SCHEMA,report_name='ema_metrics.json',title='EMA上下文：共同初始输出的输入对照',
        scope='Three neutral-context variants share initial output; retained EMA is an adaptation-cost control. All use joint detail training. Epoch0 may fail original retention gates: validation_retained=false means no acceptable candidate, not success. Seeds share one pretrained backbone; reused research test period.')
    meta = json.loads((out/'manifest.json').read_text())
    report = json.loads((out/'ema_metrics.json').read_text())
    report['initialization'] = json.loads((out/'warm_start_validation.json').read_text())
    report['limits'] += ' Neutralizing context does not erase EMA pretraining from recurrent weights. Equal stored parameter counts include zero-input columns with no effective capacity. No-EMA retains base volatility and previous-close features.'
    report['accepted_candidates'] = [j['name'] for j in meta['experiments'] if report['variants'][j['name']]['validation_retained']]
    atomic_json(report,out/'ema_metrics.json')


def identity_for(source, short, long, fusion):
    identity = source_identity(source,long)
    fm = json.loads((fusion/'manifest.json').read_text()); rm = json.loads((source/'manifest.json').read_text())
    if rm['sources']['fusion_manifest'] != sha256(fusion/'manifest.json') or fm['sources']['short_best'] != sha256(short/'best.pt'):
        raise ValueError('Short encoder is not the source of cached states')
    identity.update(short_best=sha256(short/'best.pt'),fusion_manifest=sha256(fusion/'manifest.json'))
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('all','worker','evaluate')); p.add_argument('--out',required=True)
    for name in ('source','short-run','long-run','fusion-run','root','name'): p.add_argument('--'+name)
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--streams',type=int,default=64); p.add_argument('--jobs',type=int,default=2)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise ValueError('Formal EMA training requires CUDA')
    torch.set_num_threads(4); out = Path(a.out).resolve()
    if a.action == 'worker':
        if not a.name: p.error('--name required')
        da.worker(out,a.name,model_factory=initial_model,streams_factory=ContextStreams,allow_initial_regression=True); return
    if not all((a.source,a.short_run,a.long_run,a.fusion_run,a.root)): p.error('All source paths required')
    if min(a.epochs,a.streams,a.jobs)<1 or a.jobs>4: raise ValueError('Invalid runtime settings')
    source,short,long,fusion = map(lambda s:Path(s).resolve(),(a.source,a.short_run,a.long_run,a.fusion_run))
    identity = identity_for(source,short,long,fusion)
    meta = dict(schema=SCHEMA,source=str(source),short_run=str(short),long_run=str(long),root=str(Path(a.root).resolve()),sources=identity,
        config=dict(width=512,decoder_width=256),seeds=[42,43],epochs=a.epochs,streams=a.streams,
        encoder_lr=3e-5,decoder_lr=1e-4,detail_weight=.25,
        experiments=[dict(name=f'{mode}_s{seed}',context=mode,mode='joint_detail',seed=seed) for seed in (42,43) for mode in CONTEXT_MODES],
        features=dict(base=list(FEATURES),legacy=list(CONTEXT),multiscale=list(MULTISCALE),input_dim=19,bank_dim=23,
            spans=[4,8,16,32,64],normalization='asinh(log-price differences / prior RMS volatility)',
            clock='observed bars, full original contract prefix, no window/partition resets'),
        initialization='Preserve base14 projection, bias, recurrent weights and head. Zero all5 context columns in three main variants; retained control preserves old4 and pads one zero. Same neutral initial outputs; original pretraining still used EMA.',
        selection='Minimize validation detail subject to original close<=1.02, base<=1.05, change16 MSE<=1.05. If no epoch qualifies, save rejected epoch0 and mark validation_retained=false.',
        protocol='Same chronological endpoint groups, TBPTT128, loss, budget and seeds; state reset only at contract/period/partition boundary; eight joint jobs.')
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != meta: raise ValueError('Settings changed; choose new output directory')
    elif out.exists() and any(out.iterdir()): raise ValueError('Nonempty directory lacks manifest')
    out.mkdir(parents=True,exist_ok=True); atomic_json(meta,out/'manifest.json')
    for name in ('completion.json','ema_metrics.json','summary.md'): (out/name).unlink(missing_ok=True)
    progress('Preparing shared EMA bank, original replay audit and matched initialization audit')
    prepare(meta,out,'cuda'); torch.cuda.empty_cache()
    atomic_json(dict(jobs=a.jobs,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)),out/'runtime.json')
    if a.action == 'all':
        atomic_json(preflight(meta,out,'cuda'),out/'preflight.json'); torch.cuda.empty_cache()
        run_jobs(out,a.jobs,'obson.babel.ema_ablation')
    evaluate(out,'cuda')
    if identity_for(source,short,long,fusion) != identity: raise ValueError('Source artifacts changed during experiment')
    accepted = json.loads((out/'ema_metrics.json').read_text())['accepted_candidates']
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),source_weights_unchanged=True,
                     accepted_candidates=accepted),out/'completion.json')
    progress(f'EMA ablation complete: {out}; {len(accepted)}/{len(meta["experiments"])} candidates meet validation retention')


if __name__ == '__main__': main()
