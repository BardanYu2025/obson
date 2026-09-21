"""Read-only bundle audit: every cached test endpoint plus fixed raw streaming replays."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .ae_diagnostics import metrics, summarize
from .ae_extend import atomic_json
from .dual_state import sha256, source_identity
from .fusion_readout_audit import load_cache
from .history_autoencoder import to_ohlc
from .large_history import write_global_review
from .memory_benchmark import regression_metrics
from .progress import progress
from .reconstruction_fusion import CachedWindows, history_values
from .representation import time_mask
from .stage_audit import load_data
from .streaming import build_bundle, load_bundle, frame_bars


def comparison(actual,expected,atol=5e-4,rtol=2e-4):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False,reason='shape/nonfinite',actual_shape=list(a.shape),expected_shape=list(b.shape))
    return dict(passed=bool(np.allclose(a,b,atol=atol,rtol=rtol)),max_abs=float(np.max(np.abs(a-b))),
                rms=float(np.sqrt(np.mean((a-b)**2))),atol=atol,rtol=rtol)


def bounded_difference(actual,expected,max_abs=2e-3,rms=2e-4):
    value=comparison(actual,expected)
    value['strict_passed']=value['passed']
    value['max_abs_limit'],value['rms_limit']=max_abs,rms
    value['passed']=bool('max_abs' in value and value['max_abs']<=max_abs and value['rms']<=rms)
    return value


@torch.no_grad()
def short_numerics(engine,features,stream_state,cached_state,stream_decoded,cached_decoded):
    """Independent cached-input replays isolate input/reset errors from execution layout drift.

    Use the same unmodified frozen model. No rounding, cache substitution or threshold fitting.
    """
    model=engine.short_model; model.eval(); x=torch.as_tensor(features,device=engine.device,dtype=torch.float32)
    hidden=None
    for row in range(len(x)):
        h,hidden=model(x[row:row+1][None],hidden)
    step=h[0,-1].cpu().numpy()
    hidden=None
    for start in range(0,len(x),128):
        n=min(128,len(x)-start)
        block=x.new_zeros(1,128,x.shape[1]); block[0,:n]=x[start:start+n]
        h,hidden=model(block,hidden)
    chunk=h[0,n-1].cpu().numpy()
    checks=dict(stream_vs_cached_input_step=comparison(stream_state,step,atol=2e-5,rtol=2e-5),
                step_vs_chunk128=bounded_difference(step,chunk),
                chunk128_vs_original_cache=bounded_difference(chunk,cached_state),
                stream_vs_original_cache=bounded_difference(stream_state,cached_state),
                decoder_all_channels=bounded_difference(stream_decoded,cached_decoded))
    a,b=np.asarray(stream_decoded),np.asarray(cached_decoded)
    if a.shape==b.shape and a.shape[-1]==7 and np.isfinite(a).all() and np.isfinite(b).all():
        delta=np.abs(a[...,:2]-b[...,:2])*100
        channel_means=delta.reshape(-1,2).mean(0)
        price=dict(max_abs_bps=float(delta.max()),mean_abs_bps=float(delta.mean()),
                   open_mean_abs_bps=float(channel_means[0]),close_mean_abs_bps=float(channel_means[1]),
                   max_channel_mean_abs_bps=float(channel_means.max()),max_limit_bps=.1,mean_limit_bps=.02)
        price['passed']=price['max_abs_bps']<=.1 and price['max_channel_mean_abs_bps']<=.02
    else: price=dict(passed=False,reason='invalid decoder prices')
    checks['decoder_price']=price
    return dict(passed=all(v['passed'] for v in checks.values()),checks=checks,
        interpretation='Same-input single-step versus128-step singleton replay, plus original cache (possibly different batch size). Bounds establish engineering numerical equivalence, not bitwise equality or proof of a particular CUDA kernel cause.')


@torch.no_grad()
def audit(bundle,source,fusion,long_run,root,out,batch=16,replays=6,device='cuda'):
    (out/'summary.md').unlink(missing_ok=True)
    started=time.perf_counter(); source_meta=json.loads((fusion/'manifest.json').read_text())['sources']
    values,cache_hash=load_cache(fusion,'test',source_meta)
    # Match reconstruction targets to the same exact vector rows, rather than trusting counts.
    keys=np.load(source/'target_cache/test_keys.npy',allow_pickle=False)
    if not np.array_equal(keys,values['keys']): raise ValueError('Cached endpoint order mismatch')
    ref,series,encoded,sets=load_data(root,long_run)
    engine=load_bundle(bundle,'audit',60,device)
    ds=CachedWindows(source/'target_cache','test')
    records={}; structures=[]; recent_cards=[]; history_cards=[]; max_roundtrip=0.; hist_loss=0.; offset=0
    chosen=set(np.linspace(0,len(keys)-1,min(replays,len(keys)),dtype=int))
    for data in DataLoader(ds,batch_size=batch):
        n=len(data['long']); m=data['long'].to(device); s=data['short'].to(device)
        if not np.array_equal(data['long'].numpy(),values['long'][offset:offset+n]) or not np.array_equal(data['short'].numpy(),values['short'][offset:offset+n]):
            raise ValueError('Reconstruction cache vectors differ from original fusion cache')
        z=torch.cat(((m-engine.statistics['long_mean'])/engine.statistics['long_scale'],
                     (s-engine.statistics['short_mean'])/engine.statistics['short_scale']),-1)
        recovered_m=z[:,:engine.width]*engine.statistics['long_scale']+engine.statistics['long_mean']
        recovered_s=z[:,engine.width:]*engine.statistics['short_scale']+engine.statistics['short_mean']
        max_roundtrip=max(max_roundtrip,float((m-recovered_m).abs().max()),float((s-recovered_s).abs().max()))
        recent=engine.recent_decoder.decode_recent(recovered_s).cpu().numpy()
        history=engine.long_model.decode_history(recovered_m)['reconstruction']
        structure=(engine.long_model.structure(recovered_m)*engine.target_scale+engine.target_mean).cpu().numpy()
        structures.extend(structure)
        loss,_=history_values(history,data['y'].to(device),data['mask'].to(device),data['valid'].to(device))
        hist_loss+=float(loss.sum())
        for j in range(n):
            valid=data['valid'][j].numpy(); y=data['y'][j].numpy()[valid]; p=history[j].cpu().numpy()[valid]
            yr=data['recent_y'][j].numpy()
            for span in (16,32,64): records.setdefault(f'recent/{span}',[]).append(metrics(yr[-span:],recent[j][-span:]))
            records.setdefault('history',[]).append(metrics(y.reshape(-1,7),p.reshape(-1,7)))
            if offset+j in chosen:
                i,end=keys[offset+j]; original=series[i]
                for cards,truth,pred,anchor,blocks in ((recent_cards,yr,recent[j],original.frame.close.iloc[end-64],1),
                        (history_cards,y.reshape(-1,7),p.reshape(-1,7),original.frame.close.iloc[end-128],int(valid.sum()))):
                    cards.append(dict(source=original.key,end=str(original.frame.datetime.iloc[end]),blocks=blocks,
                        truth_ohlc=to_ohlc(truth,float(anchor)).tolist(),reconstructed_ohlc=to_ohlc(pred,float(anchor)).tolist()))
        offset+=n
    if max_roundtrip>1e-5: raise ValueError('Concatenated representation failed invertibility check')
    report=dict(schema='babel-dual-stream-audit-v1',training=False,windows=len(keys),cache_sha256=cache_hash,
        bundle_sha256=sha256(bundle),roundtrip_max_abs=max_roundtrip,
        frozen_specialists=dict(reconstruction=summarize(records),history_loss=hist_loss/len(keys),
                               structure=regression_metrics(values['targets'],np.asarray(structures))),
        scope='All cached test endpoints use the original frozen specialist heads. Raw per-bar replay uses fixed endpoints only. No new holdout, training or forecasting.',
        numerical_policy='Strict cache allclose reported separately. Short-state numerical equivalence additionally requires same-input step control <=2e-5 atol/rtol; layout/cache differences max<=.002 RMS<=.0002; decoder price max<=.1bp mean<=.02bp. Long/history thresholds unchanged.',
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                     matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
        stream_replays=[])
    atomic_json(report,out/'stream_metrics.json')
    for label,cards in (('recent',recent_cards),('history',history_cards)):
        atomic_json(cards,out/f'{label}_examples.json'); write_global_review(out/f'{label}_examples.html',cards)
        page=out/f'{label}_examples.html'
        page.write_text(page.read_text().replace('单个512维综合向量重建历史','1024维双状态，经原专用头重建').replace('全局向量解码','双状态专用头解码'))
    progress(f'All {len(keys)} cached endpoints evaluated; starting {len(chosen)} fixed raw replays')
    for number,idx in enumerate(sorted(chosen),1):
        i,end=map(int,keys[idx]); original=series[i]
        rows=np.flatnonzero(time_mask(original,ref['boundaries'],'test')); lo=int(rows[0])
        if np.any(np.diff(rows)!=1): raise ValueError('Noncontiguous test partition')
        engine.reset_contract(original.key,original.period)
        replay_start=time.perf_counter(); max_feature=0.; restart_check=None; trace=[]
        for row,bar in enumerate(frame_bars(original.frame.iloc[:end+1],original.key,original.period)):
            if row<lo:
                feature=engine.warm_features(bar)
                max_feature=max(max_feature,float(np.max(np.abs(feature-encoded[i]['x'][row]))))
                continue
            if row==lo: engine.start_partition('test')
            # Decode and restore shortly before the target to exercise partial-block state.
            if row==end-7:
                saved=engine.snapshot(); previous=engine.current()
                engine.restore(saved)
                restart_check=comparison(engine.current()['short'].cpu().numpy(),previous['short'].cpu().numpy(),atol=0.,rtol=0.)
            result=engine.push(bar)
            max_feature=max(max_feature,float(np.max(np.abs(engine.features.last_x-encoded[i]['x'][row]))))
            if result['long_refreshed'] or row in (lo,end): trace.append(result['metadata'])
            if (row-lo+1)%2048==0: progress(f'Replay {number}/{len(chosen)} {original.key}: {row-lo+1}/{end-lo+1} bars')
        current=engine.current(); decoded=engine.decode()
        checks=dict(long=comparison(current['long'].cpu().numpy()[0],values['long'][idx]),
                    short=comparison(current['short'].cpu().numpy()[0],values['short'][idx]))
        m=torch.tensor(values['long'][idx:idx+1],device=device); s=torch.tensor(values['short'][idx:idx+1],device=device)
        stream_recent=decoded['recent']['coordinates'].cpu().numpy()
        cached_recent=engine.recent_decoder.decode_recent(s).cpu().numpy()
        checks['recent_decoder']=comparison(stream_recent,cached_recent)
        expected_history=engine.long_model.decode_history(m)['reconstruction'][:,-len(engine.blocks):].cpu().numpy()
        checks['history_decoder']=comparison(decoded['history']['coordinates'].cpu().numpy(),expected_history)
        numerics=short_numerics(engine,encoded[i]['x'][lo:end+1],current['short'].cpu().numpy()[0],
                               values['short'][idx],stream_recent,cached_recent)
        passed=(max_feature<=2e-5 and checks['long']['passed'] and checks['history_decoder']['passed']
                and numerics['passed'] and current['metadata']['long_age_bars']==0 and bool(restart_check and restart_check['passed']))
        row_report=dict(key=original.key,end_row=end,partition_start=lo,bars_replayed=end-lo+1,
            codec_warmup_bars=lo,codec_all_replayed_max_abs=max_feature,checks=checks,restart_check=restart_check,
            seconds=time.perf_counter()-replay_start,final_metadata=current['metadata'],
            short_numerics=numerics,strict_cache_match=all(v['passed'] for v in checks.values()),passed=bool(passed))
        # One subsequent observed bar verifies stale long metadata and immutable exported tensors.
        if end+1<len(original.frame) and end+1 in rows:
            old=current; new=engine.push(next(frame_bars(original.frame.iloc[end+1:end+2],original.key,original.period)))
            stale=comparison(new['long'].cpu().numpy(),old['long'].cpu().numpy(),atol=0.,rtol=0.)
            if not stale['passed'] or new['metadata']['long_age_bars']!=1 or new['metadata']['long_row']!=end:
                row_report['passed']=False
            row_report['next_bar']=dict(metadata=new['metadata'],long_unchanged=stale,
                short_changed=not torch.equal(new['short'],old['short']))
        atomic_json(trace,out/f'replay_{number}_timeline.json')
        report['stream_replays'].append(row_report); atomic_json(report,out/'stream_metrics.json')
        progress(f'Replay {number}/{len(chosen)} passed={row_report["passed"]} strict={row_report["strict_cache_match"]}: {original.key}, long maxdiff={checks["long"].get("max_abs")}, short maxdiff={checks["short"].get("max_abs")}, price={numerics["checks"]["decoder_price"]}')
    report['seconds']=time.perf_counter()-started; report['passed']=all(r['passed'] for r in report['stream_replays'])
    atomic_json(report,out/'stream_metrics.json')
    if not report['passed']:
        failures=[r['key'] for r in report['stream_replays'] if not r['passed']]
        raise ValueError(f'Stream audit failed for {failures}; full diagnostics saved in {out / "stream_metrics.json"}')
    r=report['frozen_specialists']; lines=['# 双状态推理验收','',f'无训练；全量缓存测试窗口 {len(keys)}，原始逐bar复现 {len(chosen)} 个固定端点。',
        f'- 近期64根收盘MAE：{r["reconstruction"]["recent/64"]["metrics"]["close_mae_bps"]["mean"]:.4f} bp',
        f'- 历史收盘MAE：{r["reconstruction"]["history"]["metrics"]["close_mae_bps"]["mean"]:.4f} bp',
        f'- 拼接/还原最大误差：{max_roundtrip:.3g}',
        '- 长状态按原严格阈值验收；短状态通过独立单步/分块对照与解码误差上限检查。strict_cache_match单独报告。',
        '- 长状态每128根刷新，间隔内时间戳保持旧值；不宣称中间bar质量已验证。',
        '- 本轮没有重新评估趋势分类、预测能力或交易收益。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--long-run',required=True); p.add_argument('--short-run',required=True)
    p.add_argument('--recon-run',required=True); p.add_argument('--fusion-run',required=True)
    p.add_argument('--root',required=True); p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=16); p.add_argument('--replays',type=int,default=6)
    a=p.parse_args()
    if not torch.cuda.is_available() or min(a.batch,a.replays)<1: raise ValueError('CUDA and positive settings required for formal audit')
    torch.set_num_threads(4)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for name in ('completion.json','stream_metrics.json','summary.md'):
        (out/name).unlink(missing_ok=True)
    long,short,recon,fusion=map(Path,(a.long_run,a.short_run,a.recon_run,a.fusion_run))
    progress('Verifying source cache; no optimizer or training will run')
    identity=source_identity(recon,long)
    bundle=out/'inference_bundle.pt'; metadata,bundle_id=build_bundle(long,short,recon,fusion,bundle)
    atomic_json(dict(metadata=metadata,identity=bundle_id,source_identity=identity,batch=a.batch,replays=a.replays),out/'manifest.json')
    audit(bundle,recon,fusion,long,a.root,out,a.batch,a.replays,'cuda')
    if sha256(long/'joint/best.pt')!=metadata['sources']['long_best'] or sha256(short/'best.pt')!=metadata['sources']['short_best']:
        raise ValueError('Frozen encoder source changed during audit')
    atomic_json(dict(status='complete',training=False),out/'completion.json')
    progress(f'Stream inference audit complete: {out}')


if __name__=='__main__': main()
