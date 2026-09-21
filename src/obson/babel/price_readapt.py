"""Matched price-head adaptation on immutable final activity states."""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import activity_alignment as al, activity_ablation as aa, activity_extend as ex
from . import detail_alignment as da
from .ae_extend import atomic_json, atomic_save, rng_state, restore_rng
from .ar_reconstruction import read_arrays, device_arrays, run_jobs, paired_error_interval, derangement
from .dual_state import sha256, verify_files
from .progress import progress

SCHEMA = 'babel-price-readapt-v1'
GOAL = dict(overall=al.GOAL['overall'],
    stage='Test recoverable price decoding on fixed final control/strong activity representations.',
    success='Both seeds retain price against matched adapted control AND fixed pre-extension control, while frozen activity evidence remains intact.',
    limits='Shared pretrained head, 100-epoch adaptation budget, reused research test. Failure is not proof that information is absent. No automatic promotion.')


def source_identity(parent):
    meta, identity = ex.parent_identity(parent)
    if meta['schema'] != ex.SCHEMA: raise ValueError('Expected completed activity continuation')
    ex.verify_finished(parent, meta)
    done = json.loads((parent/'last_diagnostic/completion.json').read_text())
    if done.get('status') != 'complete' or done.get('epoch') != meta['epochs']:
        raise ValueError('Complete fixed-final diagnostic required')
    fixed = Path(meta['parent_run'])
    if ex.parent_identity(fixed)[1] != meta['parent_identity']: raise ValueError('Fixed parent changed')
    if (parent/'cache/test_inventory.json').read_bytes()!=(fixed/'cache/test_inventory.json').read_bytes():
        raise ValueError('Fixed-reference endpoints differ')
    diagnostic=json.loads((parent/'last_diagnostic/alignment_metrics.json').read_text())
    if any(diagnostic['variants'][f'{mode}_s{s}']['selected_epoch']!=meta['epochs']
           for s in meta['seeds'] for mode in ('control','aux020')):
        raise ValueError('Diagnostic does not cover final encoder states')
    paths = [parent/'last_diagnostic/completion.json', parent/'last_diagnostic/alignment_metrics.json',
             Path(meta['source'])/'short/best.pt', fixed/'alignment_metrics.json']
    for seed in meta['seeds']:
        paths.append(fixed/f'control_s{seed}/per_window_metrics.json')
        for mode in ('control', 'aux020'):
            paths.append(parent/f'last_diagnostic/{mode}_s{seed}/per_window_metrics.json')
    return meta, dict(parent=identity, extra={str(p.resolve()):sha256(p) for p in paths})


def make_manifest(parent, old, identity, epochs=100, batch=256):
    if epochs < 1 or batch < 1: raise ValueError('Positive epochs and batch required')
    jobs = [copy.deepcopy(j) for j in old['experiments'] if j['name'].split('_s')[0] in ('control','aux020')]
    if len(jobs) != 2*len(old['seeds']): raise ValueError('Missing matched control/strong cells')
    return dict(schema=SCHEMA, parent_run=str(parent.resolve()), parent_identity=identity,
        fixed_reference_run=old['parent_run'], source=old['source'], config=old['config'],
        encoder_epoch=old['epochs'], seeds=old['seeds'], experiments=jobs, epochs=epochs, batch=batch,
        decoder_lr=1e-4, min_lr=1e-5, weight_decay=.01, detail_weight=.25,
        common_head=str((Path(old['source'])/'short/best.pt').resolve()), goal=GOAL,
        schedule='Cosine by absolute adaptation epoch from1e-4 to1e-5; fresh head-only AdamW, exact resume thereafter.',
        selection='Every validation epoch including0; minimum detail then base among fixed same-seed price gates. No qualified candidate: diagnostic minimum-detail fallback, explicitly disqualified.',
        frozen='Replay each final encoder once into immutable endpoint caches. Workers contain ONLY the price decoder; no encoder/auxiliary parameters or state labels in optimizer/loss.')


def initial_head(meta, device):
    head = da.ReconstructionFusion('short', **meta['config']).to(device)
    expected = meta['parent_identity']['extra'][meta['common_head']]
    if sha256(Path(meta['common_head'])) != expected: raise ValueError('Common head changed')
    head.load_state_dict(torch.load(meta['common_head'], map_location='cpu', weights_only=True)['model'])
    return head


def assert_score(actual, expected):
    if not all(np.isclose(v, expected[k], rtol=1e-4, atol=1e-5) for k,v in actual.items()):
        raise ValueError('Frozen endpoint replay mismatch')


@torch.no_grad()
def prepare(meta, out, device):
    cache=out/'cache';cache.mkdir(exist_ok=True);index=cache/'index.json'
    if index.exists():
        info=json.loads(index.read_text())
        if info['manifest']!=meta:raise ValueError('Cache metadata changed')
        verify_files(cache,info['files']);return
    parent=Path(meta['parent_run']);old=json.loads((parent/'manifest.json').read_text())
    report=json.loads((parent/'last_diagnostic/alignment_metrics.json').read_text())
    stats=json.loads((parent/'cache/statistics.json').read_text())
    files={};audit={}
    def save(name,value):
        np.save(cache/name,value,allow_pickle=False);files[name]=sha256(cache/name)
    for split in da.SPLITS:
        data=read_arrays(meta['source'],split)
        for k in ('y','mask'):save(f'{split}_{k}.npy',data[k])
    for job in meta['experiments']:
        name=job['name'];progress(f'Caching fixed encoder {name} at epoch {meta["encoder_epoch"]}')
        state=torch.load(parent/name/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=dict(manifest=old,job=job) or state['epoch']!=meta['encoder_epoch']:
            raise ValueError('Wrong fixed encoder state')
        model=al.initial_model(old,device,job).eval().requires_grad_(False)
        model.load_state_dict(state['model']);checks={}
        for split in da.SPLITS:
            data=device_arrays(read_arrays(meta['source'],split),device)
            score,z,_=da.run_epoch(model,data,aa.ActivityStreams(parent/'cache',split,job),True,False,
                stats['scales'],old['streams'],collect=True)
            if split!='train':
                expected=state['history'][-1]['validation'] if split=='val' else report['variants'][name]['test_objectives']
                assert_score(score,expected)
                checks[split]=dict(replayed=score,expected={k:expected[k] for k in score})
            save(f'{name}_{split}_z.npy',z)
        audit[name]=dict(epoch=state['epoch'],reference=state['activity_reference'],replays=checks,
            encoder_checkpoint_sha256=sha256(parent/name/'last.pt'))
        del model,state
    atomic_json(dict(scales=stats['scales'],variants=audit),cache/'replay.json');files['replay.json']=sha256(cache/'replay.json')
    atomic_json(dict(manifest=meta,files=files),index)


def arrays(out,name,split,device):
    return device_arrays({k:np.load(out/'cache'/f'{name+"_" if k=="z" else ""}{split}_{k}.npy',allow_pickle=False)
                          for k in ('z','y','mask')},device)


def head_epoch(head,data,scales,batch,weight=.25,opt=None,seed=0,collect=False):
    if collect and opt is not None:raise ValueError('Only chronological evaluation collects predictions')
    head.train(opt is not None);n=len(data['z']);sums={};predictions=[]
    order=np.random.default_rng(seed).permutation(n) if opt is not None else np.arange(n)
    with torch.set_grad_enabled(opt is not None):
        for left in range(0,n,batch):
            ids=torch.as_tensor(order[left:left+batch],device=data['z'].device)
            b={k:v[ids] for k,v in data.items()};pred=head.decode_recent(b['z']);parts=da.values(pred,b,scales)
            loss=(parts['base']+weight*parts['detail']).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite head objective')
            if opt is not None:
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
            for k,v in parts.items():sums[k]=sums.get(k,0.)+float(v.detach().sum())
            if collect:predictions.append(pred.detach().cpu().numpy())
    return {k:v/n for k,v in sums.items()},np.concatenate(predictions) if collect else None


def consider(state,head,epoch,score,reference):
    qualified=ex.activity_eligible(score,reference)
    old=state.get('best_validation')
    if old is None or (qualified and not state['qualified']) or (qualified==state['qualified'] and
            (score['detail'],score['base'])<(old['detail'],old['base'])):
        state.update(best_epoch=epoch,best_validation=score,qualified=qualified,
            best_model={k:v.detach().cpu().clone() for k,v in head.state_dict().items()})


def learning_rate(meta,epoch):
    fraction=(epoch-1)/max(meta['epochs']-1,1)
    return meta['min_lr']+(meta['decoder_lr']-meta['min_lr'])*.5*(1+math.cos(math.pi*fraction))


def worker(out,name,device='cuda'):
    meta=json.loads((out/'manifest.json').read_text());job=next(j for j in meta['experiments'] if j['name']==name)
    index=json.loads((out/'cache/index.json').read_text())
    if index['manifest']!=meta:raise ValueError('Cache metadata mismatch')
    verify_files(out/'cache',index['files'])
    audit=json.loads((out/'cache/replay.json').read_text());reference=audit['variants'][name]['reference']
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
    head=initial_head(meta,device);opt=torch.optim.AdamW(head.parameters(),lr=meta['decoder_lr'],weight_decay=meta['weight_decay'])
    data={s:arrays(out,name,s,device) for s in ('train','val')}
    path=out/name;path.mkdir(exist_ok=True);metadata=dict(manifest=meta,job=job)
    def run(split,training=False,epoch=0):
        return head_epoch(head,data[split],audit['scales'],meta['batch'],meta['detail_weight'],
                          opt if training else None,job['seed']+epoch)[0]
    def save():
        state.update(model=head.state_dict(),optimizer=opt.state_dict(),rng=rng_state());da.publish(state,path)
        atomic_json(dict(qualified=state['qualified'],epoch=state['best_epoch'],reference=reference,
            role='validation_price' if state['qualified'] else 'diagnostic_no_qualified_candidate'),path/'selection.json')
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=metadata:raise ValueError('Resume identity mismatch')
        if state['epoch']>meta['epochs'] or [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)):raise ValueError('Incomplete history')
        head.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    else:
        score=run('val');state=dict(metadata=metadata,epoch=0,history=[])
        consider(state,head,0,score,reference)
        atomic_json(dict(common_head_sha256=meta['parent_identity']['extra'][meta['common_head']],
            trainable_parameters=sum(p.numel() for p in head.parameters()),encoder_parameters_in_optimizer=0,
            validation=score,optimizer='fresh head-only AdamW'),path/'initialization.json');save()
    for e in range(state['epoch']+1,meta['epochs']+1):
        start=time.perf_counter();lr=learning_rate(meta,e)
        for group in opt.param_groups:group['lr']=lr
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        train=run('train',True,e);score=run('val');consider(state,head,e,score,reference)
        seconds=time.perf_counter()-start
        state['history'].append(dict(epoch=e,train=train,validation=score,lr=lr,
            eligible=ex.activity_eligible(score,reference),seconds=seconds,remaining_minutes=seconds*(meta['epochs']-e)/60,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None))
        state['epoch']=e;save()
        progress(f'{name} head epoch={e}/{meta["epochs"]} close={score["close_bps"]:.2f} detail={score["detail"]:.5f} qualified={state["qualified"]}')


def paired(a,b,inventory):
    if len(a)!=len(b) or len(a)!=len(inventory):raise ValueError('Paired endpoint length mismatch')
    weeks=np.array([x['week'] for x in inventory]);result={}
    for metric in ('close_mae_bps','normalized_detail','change_1_correlation','change_1_std_ratio'):
        x=np.array([r.get(metric,np.nan) for r in a],dtype=float);y=np.array([r.get(metric,np.nan) for r in b],dtype=float)
        valid=np.isfinite(x)&np.isfinite(y)
        if not valid.any():result[metric]=dict(supported_windows=0);continue
        item=paired_error_interval(x[valid],y[valid],weeks[valid]);item['delta_mean']=item.pop('delta_close_mae_bps')
        result[metric]=item
    return result


def preflight(meta,device):
    """Disposable full-batch backward, without optimizer updates."""
    head=initial_head(meta,device)
    z=torch.randn(meta['batch'],meta['config']['width'],device=device)
    y=torch.randn(meta['batch'],64,7,device=device)*.1;y[...,2:]=y[...,2:].abs()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    parts=da.values(head.decode_recent(z),dict(y=y,mask=torch.ones_like(y,dtype=torch.bool)),[1.,1.,1.])
    loss=(parts['base']+meta['detail_weight']*parts['detail']).mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
    return dict(batch=meta['batch'],head_parameters=sum(p.numel() for p in head.parameters()),gradient_norm=float(norm),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,
        scope='Disposable head-only backward; no optimizer step, excludes AdamW states and concurrent workers.')


@torch.no_grad()
def evaluate(out,device):
    meta=json.loads((out/'manifest.json').read_text());parent=Path(meta['parent_run'])
    prior=json.loads((parent/'last_diagnostic/alignment_metrics.json').read_text())
    fixed=json.loads((Path(meta['fixed_reference_run'])/'alignment_metrics.json').read_text())
    audit=json.loads((out/'cache/replay.json').read_text())
    inventory=json.loads((parent/'cache/test_inventory.json').read_text())
    report=dict(schema=SCHEMA,goal=GOAL,automatic_promotion=False,variants={},paired={},
        inherited_activity=dict(scope='Unchanged encoder: reuse hash-pinned fixed-final readouts, not a new replication.',
            current_input_control=prior['current_input_control'],paired_activity=prior['paired_activity'],
            versus_current_input=prior['versus_current_input']),stage_screen=[])
    per={}
    for job in meta['experiments']:
        name=job['name'];path=out/name;state=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        if state['metadata']!=dict(manifest=meta,job=job) or state['epoch']!=meta['epochs'] or [r['epoch'] for r in state['history']]!=list(range(1,meta['epochs']+1)):
            raise ValueError('Incomplete head training')
        head=initial_head(meta,device);data={s:arrays(out,name,s,device) for s in ('val','test')};variants={}
        for label,weights,epoch,validation in [('selected',state['best_model'],state['best_epoch'],state['best_validation']),
                ('final',state['model'],state['epoch'],state['history'][-1]['validation'])]:
            head.load_state_dict(weights)
            score,_=head_epoch(head,data['val'],audit['scales'],meta['batch']);assert_score(score,validation)
            test,pred=head_epoch(head,data['test'],audit['scales'],meta['batch'],collect=True)
            summary,rows=da.describe_predictions(data['test']['y'].cpu().numpy(),pred,inventory)
            details=da.detail_values(torch.tensor(pred,device=device),data['test']['y'],audit['scales']).cpu().tolist()
            for row,value in zip(rows,details):row['normalized_detail']=value
            summary.update(epoch=epoch,validation=score,test_objectives=test,
                qualified=ex.activity_eligible(score,audit['variants'][name]['reference']))
            variants[label]=summary;atomic_json(rows,path/f'{label}_per_window_metrics.json')
            if label=='selected':
                per[name]=rows
                shuffled=dict(data['test'],z=data['test']['z'][torch.as_tensor(derangement(len(rows)),device=device)])
                shuffled_score,_=head_epoch(head,shuffled,audit['scales'],meta['batch'])
                summary['shuffled_state_test_objectives']=shuffled_score
                cards=[dict(source=inventory[i]['key'],end=inventory[i]['end'],blocks=1,
                    truth_ohlc=da.to_ohlc(data['test']['y'][i].cpu().numpy(),inventory[i]['anchor']).tolist(),
                    reconstructed_ohlc=da.to_ohlc(pred[i],inventory[i]['anchor']).tolist())
                    for i in sorted(set(np.linspace(0,len(inventory)-1,min(6,len(inventory)),dtype=int)))]
                atomic_json(cards,path/'examples.json');da.write_global_review(path/'examples.html',cards)
        if variants['selected']['qualified']!=state['qualified']:raise ValueError('Selection qualification mismatch')
        base=prior['variants'][name];variants.update(before=base,
            fixed_reference=fixed['variants'][f'control_s{job["seed"]}'],
            activity_and_state_unchanged=True,selection_role='validation_price' if state['qualified'] else 'diagnostic_no_qualified_candidate')
        report['variants'][name]=variants;atomic_json(variants,path/'metrics.json')
        for label,root,other in [('before',parent/'last_diagnostic',name),('fixed_reference',Path(meta['fixed_reference_run']),f'control_s{job["seed"]}')]:
            rows=json.loads((root/other/'per_window_metrics.json').read_text())
            report['paired'][name+'_minus_'+label]=paired(per[name],rows,inventory)
    for seed in meta['seeds']:
        an,bn=f'aux020_s{seed}',f'control_s{seed}';a,b=report['variants'][an],report['variants'][bn]
        report['paired'][an+'_minus_'+bn]=paired(per[an],per[bn],inventory)
        price=lambda x,y:all(x[k]<=y[k]*1.02+1e-8 for k in ('close_bps','detail'))
        inherited=next(x for x in prior['diagnostic_stage_screen']['aux020']['seeds'] if x['seed']==seed)
        report['stage_screen'].append(dict(seed=seed,qualified=a['selected']['qualified'] and b['selected']['qualified'],
            trained_selection=a['selected']['epoch']>0 and b['selected']['epoch']>0,
            price_vs_adapted_control=price(a['selected']['test_objectives'],b['selected']['test_objectives']),
            price_vs_fixed_reference=price(a['selected']['test_objectives'],a['fixed_reference']['test_objectives']),
            state_retained=a['before']['state_probe']['test']['ba']>=a['fixed_reference']['state_probe']['test']['ba']-.01 and inherited['state_retained'],
            past_supported=inherited['past_only_supported'],beyond_current=inherited['beyond_current_input']))
    report['evidence_pass']=all(all(v for k,v in row.items() if k!='seed') for row in report['stage_screen'])
    atomic_json(report,out/'readapt_metrics.json')
    lines=['# Frozen-state price decoder adaptation','',GOAL['stage'],'',
        'Epochs below refer to head adaptation; every encoder remains at the same parent final epoch. Unqualified rows are diagnostic only.',
        '', '| variant | view | head epoch | qualified | close bp | detail |','|---|---|---:|---|---:|---:|']
    for name,r in report['variants'].items():
        for label in ('before','selected','final','fixed_reference'):
            x=r[label];lines.append(f'| {name} | {label} | {x.get("epoch", "—")} | {x.get("qualified", "—")} | {x["test_objectives"]["close_bps"]:.3f} | {x["test_objectives"]["detail"]:.5f} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n\n## Research screen (no promotion)\n'+json.dumps(report['stage_screen'],indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--parent');p.add_argument('--name')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=256);p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();out=Path(a.out).resolve()
    if not torch.cuda.is_available():raise ValueError('Formal head training requires CUDA on AutoDL')
    torch.set_num_threads(4)
    if a.action=='worker':
        if not a.name:p.error('--name required')
        worker(out,a.name);return
    if not a.parent:p.error('--parent required')
    if not 1<=a.jobs<=4:raise ValueError('jobs must be1..4')
    parent=Path(a.parent).resolve();old,identity=source_identity(parent)
    if parent==out or parent in out.parents or out in parent.parents:raise ValueError('Use separate sibling directory')
    meta=make_manifest(parent,old,identity,a.epochs,a.batch)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=meta:raise ValueError('Settings changed; use another directory')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output without manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    atomic_json(dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),jobs=a.jobs),out/'runtime.json')
    prepare(meta,out,'cuda');torch.cuda.empty_cache()
    if a.action=='all':
        atomic_json(preflight(meta,'cuda'),out/'preflight.json');torch.cuda.empty_cache()
        run_jobs(out,a.jobs,'obson.babel.price_readapt')
    evaluate(out,'cuda')
    verify_files(out/'cache',json.loads((out/'cache/index.json').read_text())['files'])
    if source_identity(parent)[1]!=identity:raise ValueError('Sources changed during adaptation')
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),encoder_epoch=meta['encoder_epoch'],
        head_epochs=meta['epochs'],source_unchanged=True,automatic_promotion=False,goal=GOAL),out/'completion.json')
    progress('Price adaptation complete; all four cells and fixed final heads reported')


if __name__=='__main__':main()
