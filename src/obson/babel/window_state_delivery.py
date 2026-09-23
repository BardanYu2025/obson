"""Package and validate the frozen512 baseline interface, without fitting or training."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import window_state as ws, endpoint_readout_audit as ea
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA='babel-window-state-delivery-v1'


def code_identity():
    return ws.code_identity() | {Path(__file__).name:sha256(__file__)}


def source_identity(source):
    meta=read_json(source/'manifest.json');done=read_json(source/'completion.json')
    if meta['schema']!=ea.SCHEMA or meta['code_sha256']!=ea.code_identity() or done['status']!='complete' or not done['source_unchanged']:
        raise ValueError('Completed immutable endpoint audit required')
    if any(done[k]!=0 for k in ('encoder_updates','head_updates','statistics_fits')):raise ValueError('Unexpected fitted source')
    ea.bb.ab.sc.verify_worker_files(source,done['files'])
    if ea.source_identity(Path(meta['source']))!=meta['identity']:raise ValueError('Upstream source changed')
    return dict(manifest=meta,files=done['files']|{'completion.json':sha256(source/'completion.json')})


def jobs(meta):
    sm=meta['identity']['manifest']['identity']['manifest']
    width=sm['widths'][0]
    return [j for j in sm['experiments'] if j['width']==width and j['mode']=='endpoint']


def build_bundle(meta,out):
    source=Path(meta['source']);parent=meta['identity']['manifest'];alignment=Path(parent['source'])
    sm=parent['identity']['manifest'];bundle=out/'bundle';bundle.mkdir(exist_ok=True)
    if (bundle/'index.json').exists():
        idx=read_json(bundle/'index.json')
        if idx['delivery_manifest']!=meta or idx['code_sha256']!=ws.code_identity():raise ValueError('Existing bundle differs')
        verify_files(bundle,idx['files']);return bundle
    names={}
    inventory=read_json(Path(sm['sampling_source'])/'candidates/inventory.json')
    periods=sorted({int(r['period']) for r in inventory})
    for job in jobs(meta):
        aligned,_=ea.load_model(parent,job,'best','cpu')
        row=read_json(alignment/job['name']/'training_summary.json');name=f'baseline_s{job["seed"]}.pt'
        ck=dict(schema=ws.SCHEMA,seed=job['seed'],config=dict(sm['config'],latent=job['width']),
            model=ea.bb.ab.cpu_state(aligned.core),statistics=read_json(alignment/'cache/statistics.json'),
            local=read_json(alignment/'cache/local_scales.json'),identity=dict(job=job['name'],seed=job['seed'],checkpoint='best',
                epoch=row['selected_epoch'],source_weight_sha256=sha256(alignment/job['name']/'best.pt'),
                supported_periods=periods,baseline_retained=True,automatic_promotion=False))
        atomic_save(ck,bundle/name);names[str(job['seed'])]=name
    if set(names)!={'42','43'}:raise ValueError('Both fixed baseline seeds required')
    atomic_json(dict(schema=ws.SCHEMA,delivery_manifest=meta,models=names,code_sha256=ws.code_identity(),
        files={name:sha256(bundle/name) for name in names.values()},minimum_history=ws.WARMUP,
        feature_names=list(ws.FEATURES),selection='Original best epochs, both baseline seeds. No new seed selection; no automatic embedding ensemble.'),bundle/'index.json')
    return bundle


@torch.inference_mode()
def cached_replay(meta,bundle,out,device):
    parent=meta['identity']['manifest'];sm=parent['identity']['manifest'];alignment=Path(parent['source']);report={}
    inventory=read_json(Path(sm['sampling_source'])/'candidates/inventory.json')
    period=int(inventory[0]['period'])
    for job in jobs(meta):
        engine=ws.load_bundle(bundle,job['seed'],f'TEST/{period}/contract',period,device,_audit=True)
        for split in ea.SPLITS:
            data=ea.bb.load_data(sm,split);gs=[];rs=[]
            for start in range(0,len(data['x']),64):
                z=engine.model.encoder(torch.tensor(np.asarray(data['x'][start:start+64]),device=device))[:,-1]
                g=engine.model.decoder(z);gs.append(g.cpu().numpy());rs.append(ws.er.crop_prediction(g,engine.statistics,engine.local).cpu().numpy())
            pred=np.concatenate(gs);recent=np.concatenate(rs)
            score,errors=ea.bb.ab.measure(pred,data,engine.statistics)
            previous=read_json(alignment/f'{split}_best_errors.json')[job['name']+'/global'];ea.require_replay(errors,previous,'bundle/global')
            y=torch.tensor(np.asarray(data['y']));mask=torch.tensor(np.asarray(data['mask']));ps=torch.full((len(y),1),128)
            target,valid=ea.bb.ba.local_targets(y,mask,ps,engine.statistics,engine.local)
            local_score,local_errors=ea.bb.pr.measure(recent,target[:,0].numpy(),valid[:,0].numpy(),engine.local)
            old=read_json(Path(meta['source'])/f'trials/{job["name"]}_best/{split}_errors.json')['global_crop']
            ea.require_replay(local_errors,old,'bundle/recent')
            report[job['name']+'/'+split]=dict(global_replay=True,recent_replay=True,global_score=score,recent_score=local_score,windows=len(y))
        progress(f'Portable baseline replay passed: seed{job["seed"]}')
    atomic_json(report,out/'cached_replay.json')
    return report


def raw_plan(meta,root):
    """Two predetermined distinct eligible contracts per research set, no score selection."""
    source=Path(meta['source']);result=[]
    for split in ea.SPLITS:
        inv=read_json(source/f'{split}_inventory.json');eligible=[];seen=set()
        for i,row in enumerate(inv):
            if row['row']>=ws.WARMUP+8 and row['key'] not in seen:
                eligible.append((i,row));seen.add(row['key'])
        if len(eligible)<2:raise ValueError('Two complete raw replay contracts per set required')
        for choice in (eligible[0],eligible[-1]):
            idx,row=choice;symbol,period,contract=row['key'].split('/')
            path=(root/symbol/f'{contract}_{period}m.csv').resolve()
            result.append(dict(split=split,index=idx,inventory=row,path=str(path),sha256=sha256(path)))
    return result


@torch.inference_mode()
def raw_replay(meta,bundle,out,plan,device):
    parent=meta['identity']['manifest'];sm=parent['identity']['manifest'];results=[]
    for n,item in enumerate(plan):
        row=item['inventory'];period=int(row['period']);end=row['row'];path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Raw replay file changed')
        df=ws.data.validate_frame(pd.read_csv(path),str(path))
        if end>=len(df) or str(df.datetime.iloc[end])!=row['end']:raise ValueError('Raw endpoint timestamp changed')
        batch_x=np.column_stack((ws.ae_context.encode_context(df,period,'ema8_32')['x'],ws.aa.activity_features(df,period)[0]))
        prefix=df.iloc[:end+1];prefix_x=np.column_stack((ws.ae_context.encode_context(prefix,period,'ema8_32')['x'],ws.aa.activity_features(prefix,period)[0]))
        if not np.array_equal(batch_x[:end+1],prefix_x):raise ValueError('Batch features changed with future suffix')
        bars=list(ws.frame_bars(prefix,row['key'],period));cached=ea.bb.load_data(sm,item['split'])['x'][item['index']]
        for job in jobs(meta):
            engine=ws.load_bundle(bundle,job['seed'],row['key'],period,device,_audit=True)
            for bar in bars[:end-8]:engine.warm(bar,as_of=ws.streaming.timestamp(bar.datetime)+pd.Timedelta(minutes=period))
            restored=ws.load_bundle(bundle,job['seed'],row['key'],period,device,_audit=True);restored.restore(engine.snapshot())
            examples=[];max_features=0.;max_state=0.;max_resume=0.
            for i in range(end-8,end+1):
                bar=bars[i];asof=ws.streaming.timestamp(bar.datetime)+pd.Timedelta(minutes=period)
                output=engine.push(bar,as_of=asof);again=restored.push(bar,as_of=asof)
                current=np.stack([r['x'] for r in engine.rows]);expected=batch_x[i-127:i+1]
                max_features=max(max_features,float(np.abs(current-expected).max()))
                if not np.allclose(current,expected,atol=1e-6,rtol=2e-5):raise ValueError('Incremental28-channel mismatch')
                normalized=((expected-engine.statistics['x_mean'])/engine.statistics['x_scale']).astype(np.float32)
                z=engine.model.encoder(torch.tensor(normalized[None],device=device))[:,-1]
                z_actual=torch.tensor([output['embedding']],device=device);numeric=ws.er.comparison(z_actual,z,ws.er.TOLERANCES['fp32'])
                if not numeric['passed']:raise ValueError(f'Rolling endpoint differs from batch reference: {numeric}')
                max_state=max(max_state,numeric['max_abs'])
                diff=float(np.abs(np.asarray(output['embedding'])-again['embedding']).max());max_resume=max(max_resume,diff)
                if output!=again:raise ValueError('Snapshot continuation differs')
                if i==end:
                    if not np.allclose(normalized,cached,atol=1e-6,rtol=2e-5):raise ValueError('Raw window differs from pinned source cache')
                if i in (end-8,end-1,end):examples.append(output)
            result=dict(key=row['key'],seed=job['seed'],split=item['split'],end=row['end'],rolling_states=9,
                max_feature_abs=max_features,max_state_abs=max_state,max_resume_abs=max_resume,
                raw_cache_replay=True,future_feature_prefix_equal=True,snapshot_exact=True)
            atomic_json(dict(result=result,examples=examples),out/f'raw_{n}_s{job["seed"]}.json');results.append(result)
            progress(f'Raw rolling-window replay passed: {row["key"]}, seed{job["seed"]}')
        if sha256(path)!=item['sha256']:raise ValueError('Raw file mutated during replay')
    atomic_json(results,out/'raw_replay.json');return results


def run(source,out,root,device='cuda'):
    source,out,root=source.resolve(),out.resolve(),root.resolve()
    for dep in (source,root):
        if out==dep or out in dep.parents or dep in out.parents:raise ValueError('Separate delivery output required')
    progress('Verifying frozen baseline lineage; no training')
    identity=source_identity(source);parent=identity['manifest'];ea.check_output(Path(parent['source']),out)
    runtime=read_json(source/'runtime.json')
    if runtime['torch']!=str(torch.__version__) or runtime['numpy']!=np.__version__:
        raise ValueError('Restore source Torch/NumPy environment')
    meta=dict(schema=SCHEMA,source=str(source),identity=identity,raw_root=str(root),code_sha256=code_identity(),
        encoder_updates=0,head_updates=0,statistics_fits=0,minimum_history=ws.WARMUP,
        interface='Explicit per-contract rolling128 endpoint; both original512 endpoint best checkpoints. No candidate promotion or seed ranking.')
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Delivery source/configuration changed')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty delivery output')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    bundle=build_bundle(meta,out);(bundle/'validation.json').unlink(missing_ok=True)
    plan=raw_plan(meta,root)
    if (out/'raw_plan.json').exists() and read_json(out/'raw_plan.json')!=plan:raise ValueError('Pinned raw plan changed')
    atomic_json(plan,out/'raw_plan.json')
    cached_replay(meta,bundle,out,device);raw_replay(meta,bundle,out,plan,device)
    if source_identity(source)!=identity:raise ValueError('Source changed during delivery')
    atomic_json(dict(status='passed',index_sha256=sha256(bundle/'index.json'),checks=['cached_global','cached_recent','raw_features','rolling_endpoint','snapshot_restore'],
        scope='Source research endpoints and fixed raw sequences; no new all-bars quality or forecasting claim'),bundle/'validation.json')
    (out/'summary.md').write_text('# Frozen window-state interface\n\nTwo retained512 endpoint seeds. Zero training and fitting. Cached global/recent results and raw rolling/restore checks passed.\n')
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p!=out/'completion.json'}
    atomic_json(dict(status='complete',source_unchanged=True,encoder_updates=0,head_updates=0,statistics_fits=0,files=files),out/'completion.json')
    progress('Validated portable bundle ready; source models unchanged')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--out',required=True);parser.add_argument('--root',required=True)
    a=parser.parse_args();ea.bb.ab.configure_runtime()
    if not torch.cuda.is_available():raise ValueError('Formal delivery verification runs on AutoDL CUDA')
    import fcntl
    source,out,root=Path(a.source).resolve(),Path(a.out).resolve(),Path(a.root).resolve()
    parent=read_json(source/'manifest.json')
    if parent['identity']['manifest']['widths'][0]!=512:
        raise ValueError('Formal delivery requires the retained512 baseline')
    for dep in (source,root):
        if out==dep or out in dep.parents or dep in out.parents:raise ValueError('Separate output required')
    ea.check_output(Path(read_json(source/'manifest.json')['source']),out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/('.'+out.name+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Delivery output already running')
        run(source,out,root)


if __name__=='__main__':main()
