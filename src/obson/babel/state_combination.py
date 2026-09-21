"""Matched readout experiments on fixed price/activity state combinations."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import state_transfer as st, state_holdout as sh, activity_alignment as al
from . import activity_ablation as aa, detail_alignment as da, price_readapt as pr
from .ae_extend import atomic_json, atomic_save
from .ar_reconstruction import run_jobs
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = 'babel-state-combination-v1'
TASKS = {'transfer':st.NAMES, 'activity_past':al.NAMES[5:]}
SPLITS = ('train','val','test','cross_research')
GOAL = dict(overall='Reusable causal compression of observed price and activity history.',
    stage='Test complementary frozen states against single-state, statistical and equally wide two-price-state controls.',
    scope='Old research test and already-opened five-symbol research sample; no independent confirmation.',
    automatic_promotion=False, encoder_updates=0,
    price='Original control decoder reads ONLY the first control state. Retention is guaranteed by routing, not learned by this experiment.')


def code_identity():
    return {Path(p).name:sha256(p) for p in (__file__,st.__file__,sh.__file__)}


def verify_code(meta):
    if meta['code_sha256']!=code_identity():raise ValueError('Readout implementation changed during this run')


def representations(seeds):
    if len(seeds)!=2 or seeds[0]==seeds[1]:raise ValueError('Exactly two distinct encoder seeds required')
    result={'current':['current'],'statistics':['statistics']}
    for seed in seeds:
        for mode in ('control','aux020'):
            name=f'{mode}_s{seed}';result[name]=[name];result[name+'_plus_statistics']=[name,'statistics']
    for seed,other in zip(seeds,seeds[::-1]):
        for label,parts in [('dual',[f'control_s{seed}',f'aux020_s{seed}']),
                            ('price_pair',[f'control_s{seed}',f'control_s{other}'])]:
            result[f'{label}_s{seed}']=parts
            result[f'{label}_s{seed}_plus_statistics']=parts+['statistics']
    return result


def verify_worker_files(directory,files):
    """Trial checkpoints are nested; still reject traversal/symlinks outside worker."""
    root=directory.resolve()
    if not files:raise ValueError('Empty worker fingerprint index')
    for name,digest in files.items():
        rel=Path(name);path=(root/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Worker fingerprint mismatch: {name}')


def combine(parts,features):
    values=[features[p] for p in parts]
    if any(v.ndim!=2 or len(v)!=len(values[0]) or not np.isfinite(v).all() for v in values):
        raise ValueError('State alignment/nonfinite feature error')
    return np.concatenate(values,axis=1).astype(np.float32)


def fit_ridge(xs,ys,masks):
    """Two splits ONLY; share solves for targets with identical valid-row masks."""
    if not len(xs)==len(ys)==len(masks)==2:raise ValueError('Ridge selection accepts train/validation only')
    groups={};rows=[None]*ys[0].shape[1];models=[None]*ys[0].shape[1]
    for j in range(ys[0].shape[1]):
        valid=[m[:,j] for m in masks];count=[int(v.sum()) for v in valid]
        if min(count)<2 or np.std(ys[0][valid[0],j].astype(float))<1e-8:
            rows[j]=dict(support=count,skipped='insufficient support or constant training target');continue
        key=tuple(v.tobytes() for v in valid);groups.setdefault(key,[]).append(j)
    for ids in groups.values():
        valid=[m[:,ids[0]] for m in masks];x=[v[ok].astype(float) for v,ok in zip(xs,valid)]
        y=[v[ok][:,ids].astype(float) for v,ok in zip(ys,valid)]
        mean,scale=x[0].mean(0),x[0].std(0).clip(1e-5);ym,yscale=y[0].mean(0),y[0].std(0).clip(1e-5)
        x=[np.column_stack(((v-mean)/scale,np.ones(len(v)))) for v in x];y=[(v-ym)/yscale for v in y]
        lhs,rhs=x[0].T@x[0],x[0].T@y[0];best=np.full(len(ids),np.inf)
        for alpha in (1.,10.,100.,1000.):
            penalty=np.eye(lhs.shape[0])*alpha;penalty[-1,-1]=0
            weight=np.linalg.solve(lhs+penalty,rhs);score=((x[1]@weight-y[1])**2).mean(0)
            for k,j in enumerate(ids):
                if score[k]<best[k]:
                    best[k]=score[k];rows[j]=dict(alpha=alpha,validation_normalized_mse=float(score[k]),support=[len(v) for v in y])
                    models[j]=dict(mean=mean.tolist(),scale=scale.tolist(),ym=float(ym[k]),ys=float(yscale[k]),
                        weight=weight[:,k].tolist(),alpha=alpha)
    return models,rows


def source_identity(transfer,cross):
    tm,identity=sh.sources(transfer);hm=read_json(cross/'manifest.json')
    if hm['schema']!=sh.SCHEMA or read_json(cross/'completion.json').get('status')!='complete':
        raise ValueError('Completed prior cross-symbol run required')
    if Path(hm['transfer']).resolve()!=transfer.resolve() or hm['source_identity']!=identity:
        raise ValueError('Cross-symbol and transfer sources differ')
    index=read_json(cross/'cache/index.json')
    if index['manifest']!=hm:raise ValueError('Cross-symbol cache metadata mismatch')
    verify_files(cross/'cache',index['files'])
    lock=read_json(cross/'evaluation_lock.json')
    if lock['manifest']!=hm:raise ValueError('Cross-symbol evaluation lock mismatch')
    for file,key in [('lineage_audit.json','lineage_sha256'),('raw_audit.json','raw_audit_sha256'),('frozen/index.json','frozen_index_sha256')]:
        if sha256(cross/file)!=lock[key]:raise ValueError('Prior audit lock changed')
    files=('manifest.json','completion.json','cache/index.json','cache/test_inventory.json',
           'holdout_metrics.json','evaluation_lock.json','lineage_audit.json','raw_audit.json','frozen/index.json')
    identity.update({str((cross/f).resolve()):sha256(cross/f) for f in files})
    return tm,identity


def make_manifest(transfer,cross,tm,identity,epochs=100,batch=256,streams=32):
    if min(epochs,batch,streams)<1:raise ValueError('Positive epochs, batch and streams required')
    reps=representations(tm['seeds'])
    return dict(schema=SCHEMA,transfer=str(transfer.resolve()),cross_run=str(cross.resolve()),source_identity=identity,
        seeds=tm['seeds'],representations=reps,tasks=TASKS,epochs=epochs,batch=batch,streams=streams,
        hidden=128,lr=1e-3,decays=[.001,.01],probe_seeds=[1701,1702],goal=GOAL,
        code_sha256=code_identity(),
        experiments=[dict(name=f'{task}__{name}',task=task,representation=name) for task in TASKS for name in reps],
        selection='Per-task train-only normalization; Ridge alpha or MLP epoch/decay selected on original validation only. Both probe seeds reported.100 epochs is a budget, not convergence.',
        controls='dual_s42=[control42,aux42], price_pair_s42=[control42,control43]; reversed control order for s43. Shared pretraining/price controls; not four independent encoders.',
        primary='Per-task complete-case standardized MSE; both research datasets reported separately. At least50 common endpoints and5 weeks for paired signals. No new holdout claim.')


def load_arrays(out,name,task,split):
    cache=out/'cache'
    return tuple(np.load(cache/f'{split}_{k}.npy',allow_pickle=False) for k in (name,task+'_y',task+'_mask'))


def decode_price(head,joined,width):
    if joined.ndim!=2 or joined.shape[1]!=2*width:raise ValueError('Expected two equally wide states')
    return head.decode_recent(joined[:,:width].contiguous())


@torch.no_grad()
def price_route_audit(head,control,aux,batch,device):
    if control.shape!=aux.shape:raise ValueError('Price/activity states misaligned')
    maximum=0.
    head.eval().requires_grad_(False)
    for left in range(0,len(control),batch):
        c=torch.tensor(control[left:left+batch],device=device);a=torch.tensor(aux[left:left+batch],device=device)
        direct=head.decode_recent(c);joined=torch.cat((c,a),1)
        routed=decode_price(head,joined,c.shape[1]);perturbed=decode_price(head,torch.cat((c,a*100+73),1),c.shape[1])
        if not torch.equal(direct,routed) or not torch.equal(routed,perturbed):raise ValueError('Auxiliary state altered protected price output')
        maximum=max(maximum,float((direct-routed).abs().max()))
    return dict(windows=len(control),max_abs_coordinate_difference=maximum,aux_perturbation_invariant=True,
        interpretation='Same frozen control path by construction, not newly learned retention.')


@torch.no_grad()
def replay_cross(meta,tm,out,device):
    cross=Path(meta['cross_run']);price=Path(tm['parent_run']);pm=read_json(price/'manifest.json');encoder=Path(tm['encoder_run']);em=read_json(encoder/'manifest.json')
    old=read_json(cross/'holdout_metrics.json');cache=cross/'cache';n=len(read_json(cache/'test_inventory.json'))
    data={k:torch.tensor(v,device=device) for k,v in dict(z=np.zeros((n,em['config']['width']),np.float32),
        y=np.load(cache/'test_price_y.npy'),mask=np.load(cache/'test_price_mask.npy')).items()}
    scales=read_json(price/'cache/replay.json')['scales'];features={};audit={}
    for job in pm['experiments']:
        name=job['name'];progress(f'Replaying known research states: {name}')
        ck=torch.load(encoder/name/'last.pt',map_location='cpu',weights_only=True)
        if ck['metadata']!=dict(manifest=em,job=job) or ck['epoch']!=em['epochs']:raise ValueError('Fixed encoder mismatch')
        model=da.DetailModel(**em['config'],input_dim=28)
        model.activity_head=nn.Sequential(nn.LayerNorm(em['config']['width']),nn.Linear(em['config']['width'],20));model.load_state_dict(ck['model'])
        head=torch.load(price/name/'best.pt',map_location='cpu',weights_only=True)
        if head['metadata']!=dict(manifest=pm,job=job):raise ValueError('Fixed price decoder mismatch')
        model.head.load_state_dict(head['model']);model.to(device).eval().requires_grad_(False)
        score,z,_=da.run_epoch(model,data,aa.ActivityStreams(cache,'test',job),True,True,scales,meta['streams'],collect=True)
        pr.assert_score(score,old['price'][name]['objectives']);features[name]=z;audit[name]=dict(replayed=score,expected=old['price'][name]['objectives'])
        del model,ck,head
    return features,audit


def prepare(meta,out,device):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/'index.json').exists():
        info=read_json(cache/'index.json')
        if info['manifest']!=meta:raise ValueError('Combination cache settings changed')
        verify_files(cache,info['files']);return
    transfer=Path(meta['transfer']);cross=Path(meta['cross_run']);tm=read_json(transfer/'manifest.json');encoder=Path(tm['encoder_run']);price=Path(tm['parent_run']);pm=read_json(price/'manifest.json')
    features,replays=replay_cross(meta,tm,out,device);files={};counts={};routing={}
    def save(name,value):
        np.save(cache/name,value,allow_pickle=False);files[name]=sha256(cache/name)
    for split in SPLITS:
        cross_split=split=='cross_research';source=cross if cross_split else transfer;prefix='test' if cross_split else split
        base={k:np.load(source/f'cache/{prefix}_{k}.npy') for k in ('current','statistics')}
        for seed in meta['seeds']:
            for mode in ('control','aux020'):
                name=f'{mode}_s{seed}';base[name]=features[name] if cross_split else np.load(transfer/f'cache/{split}_{name}.npy')
        n=len(base['current'])
        for task in TASKS:
            if task=='transfer':y=np.load(source/f'cache/{prefix}_y.npy');mask=np.load(source/f'cache/{prefix}_mask.npy')
            else:
                location=cross if cross_split else encoder
                y=np.load(location/f'cache/{prefix}_activity.npy')[:,5:];mask=np.load(location/f'cache/{prefix}_activity_mask.npy')[:,5:]
            if y.shape!=(n,len(TASKS[task])) or mask.shape!=y.shape:raise ValueError('Task alignment mismatch')
            save(f'{split}_{task}_y.npy',y);save(f'{split}_{task}_mask.npy',mask)
            counts[f'{split}/{task}']=dict(endpoints=n,target_support=mask.sum(0).tolist(),complete_cases=int(mask.all(1).sum()))
        for name,parts in meta['representations'].items():save(f'{split}_{name}.npy',combine(parts,base))
        if split in ('test','cross_research'):
            inventory=read_json(source/'cache/test_inventory.json')
            if len(inventory)!=n:raise ValueError('Inventory alignment mismatch')
            for r in inventory:r['month']=r['end'][:7]
            atomic_json(inventory,cache/f'{split}_inventory.json');files[f'{split}_inventory.json']=sha256(cache/f'{split}_inventory.json')
            routing[split]={}
            for seed in meta['seeds']:
                name=f'control_s{seed}';job=next(j for j in pm['experiments'] if j['name']==name)
                ck=torch.load(price/name/'best.pt',map_location='cpu',weights_only=True)
                if ck['metadata']!=dict(manifest=pm,job=job):raise ValueError('Protected price head mismatch')
                head=da.ReconstructionFusion('short',**pm['config']).to(device);head.load_state_dict(ck['model'])
                routing[split][name]=price_route_audit(head,base[name],base[f'aux020_s{seed}'],meta['batch'],device)
                del head,ck
    atomic_json(dict(coverage=counts,source_replay=replays,price_routing=routing,prior_aux_price_screen=read_json(cross/'holdout_metrics.json')['evidence_screen'],goal=GOAL),out/'preparation_audit.json')
    atomic_json(dict(manifest=meta,files=files),cache/'index.json')


def worker(out,name,device='cuda'):
    meta=read_json(out/'manifest.json');verify_code(meta);job=next(j for j in meta['experiments'] if j['name']==name)
    index=read_json(out/'cache/index.json')
    if index['manifest']!=meta:raise ValueError('Cache manifest mismatch')
    verify_files(out/'cache',index['files']);path=out/name;path.mkdir(exist_ok=True)
    if (path/'completion.json').exists():
        done=read_json(path/'completion.json')
        if done['manifest']!=meta or done['status']!='complete':raise ValueError('Worker completion identity mismatch')
        verify_worker_files(path,done['files']);return
    arrays=[load_arrays(out,job['representation'],job['task'],s) for s in ('train','val')]
    xs,ys,masks=map(list,zip(*arrays));mean,scale,active=st.target_stats(ys[0],masks[0]);report=dict(task=job['task'],representation=job['representation'],
        input_dimensions=xs[0].shape[1],target_names=meta['tasks'][job['task']],scope=GOAL['scope'],selection={},datasets={})
    ridge,rows=fit_ridge(xs,ys,masks);atomic_json(ridge,path/'ridge.json');report['selection']['linear']=dict(targets=rows)
    selected={}
    for seed in meta['probe_seeds']:
        best=None
        for decay in meta['decays']:
            trial=path/f'p{seed}_wd{decay}';metadata=dict(manifest=meta,job=job,probe_seed=seed,decay=decay)
            value=st.mlp_trial(xs,ys,masks,trial,metadata,seed,decay,meta['epochs'],meta['batch'],meta['hidden'],meta['lr'],device)
            if best is None or value['validation_mse']<best[0]['validation_mse']:best=(value,trial)
        kind=f'mlp_s{seed}';report['selection'][kind]=dict(best[0],selected_at_budget_end=best[0]['selected_epoch']==meta['epochs'])
        selected[kind]=best[1]/'best.pt'
    # Freeze all choices before opening either research evaluation feature/target array.
    lock=dict(manifest=meta,job=job,ridge_sha256=sha256(path/'ridge.json'),selected={k:dict(path=str(v),sha256=sha256(v)) for k,v in selected.items()},selection=report['selection'])
    if (path/'selection_lock.json').exists() and read_json(path/'selection_lock.json')!=lock:raise ValueError('Readout selection changed after evaluation')
    atomic_json(lock,path/'selection_lock.json')
    atomic_json(dict(mean=mean.tolist(),scale=scale.tolist(),active=active.tolist()),path/'target_statistics.json')
    for split in ('test','cross_research'):
        x,y,mask=load_arrays(out,job['representation'],job['task'],split);mask=mask&active;truth=(y-mean)/scale
        inventory=read_json(out/f'cache/{split}_inventory.json');report['datasets'][split]={};errors={}
        for kind in sh.KINDS:
            err=(sh.ridge_errors(ridge,x,y,mask) if kind=='linear' else sh.mlp_errors(torch.load(selected[kind],map_location='cpu',weights_only=True),x,y,mask,meta['hidden'],device))
            errors[kind]=err;np.save(path/f'{split}_{kind}_errors.npy',err)
            report['datasets'][split][kind]=sh.grouped_errors(err,truth,inventory,report['target_names'])
        atomic_json(sh.safe(errors),path/f'{split}_per_window_errors.json')
    atomic_json(sh.safe(report),path/'metrics.json')
    files={str(p.relative_to(path)):sha256(p) for p in path.rglob('*') if p.is_file() and p.name not in ('completion.json','progress.json','run.log') and not p.name.endswith('.tmp')}
    atomic_json(dict(status='complete',manifest=meta,files=files),path/'completion.json')
    atomic_json(dict(status='complete',representation=job['representation'],task=job['task']),path/'progress.json')


def comparisons(seed):
    pairs=[]
    for suffix in ('','_plus_statistics'):
        a=f'dual_s{seed}{suffix}'
        pairs.extend((a,f'{base}_s{seed}{suffix}') for base in ('control','aux020','price_pair'))
    pairs.extend([(f'dual_s{seed}_plus_statistics','statistics'),(f'dual_s{seed}_plus_statistics',f'dual_s{seed}')])
    return pairs


def evaluate(out):
    meta=read_json(out/'manifest.json');report=dict(schema=SCHEMA,goal=GOAL,datasets={},price_routing=read_json(out/'preparation_audit.json')['price_routing'],
        automatic_promotion=False,independent_holdout=False,prior_aux_price_screen=read_json(out/'preparation_audit.json')['prior_aux_price_screen'],
        limits='Both evaluation sets previously opened. Larger state costs two encoders. Price unchanged by routing, not learned preservation. Two price-pair orders share the same encoders; probe seeds are not encoder replications.')
    workers={}
    for job in meta['experiments']:
        path=out/job['name'];done=read_json(path/'completion.json')
        if done['status']!='complete' or done['manifest']!=meta:raise ValueError('Incomplete combination worker')
        verify_worker_files(path,done['files']);workers[job['name']]=read_json(path/'metrics.json')
    def favorable(row):return row.get('supported',False) and row.get('high') is not None and row['high']<0
    for split in ('test','cross_research'):
        inventory=read_json(out/f'cache/{split}_inventory.json');report['datasets'][split]={}
        for task,names in meta['tasks'].items():
            variants={};errors={};contrasts={};signals=[]
            groups={'primary':list(range(len(names)))}
            groups.update({family:list(range(i*3,i*3+3)) for i,family in enumerate(st.FAMILIES)} if task=='transfer' else
                {family:list(range(i*5,i*5+5)) for i,family in enumerate(('past1_16','lag4','lag16'))})
            for name in meta['representations']:
                worker_name=f'{task}__{name}';row=workers[worker_name]
                variants[name]=dict(input_dimensions=row['input_dimensions'],selection=row['selection'],metrics=row['datasets'][split])
                errors[name]={k:np.load(out/worker_name/f'{split}_{k}_errors.npy') for k in sh.KINDS}
                if any(e.shape!=(len(inventory),len(names)) for e in errors[name].values()):raise ValueError('Evaluation error alignment mismatch')
            for seed in meta['seeds']:
                for a,b in comparisons(seed):
                    key=a+'_minus_'+b;contrasts[key]={}
                    for kind in sh.KINDS:
                        contrasts[key][kind]={}
                        for group,ids in groups.items():
                            values=sh.contrast(errors[a][kind],errors[b][kind],inventory,ids)
                            for row in values.values():row['interpretation']='Negative favors candidate; paired weekly bootstrap on previously opened research data. Group analyses uncorrected.'
                            contrasts[key][kind][group]=values
                    # Report each contrast, rather than hiding failures in one score.
                    signals.append(dict(seed=seed,candidate=a,reference=b,
                        all_readouts_supported=all(favorable(contrasts[key][k]['primary']['all']) and
                            (k=='linear' or (variants[a]['selection'][k]['trained_selection'] and variants[b]['selection'][k]['trained_selection'])) for k in sh.KINDS)))
            report['datasets'][split][task]=dict(variants=variants,paired=contrasts,exploratory_signals=signals)
    atomic_json(sh.safe(report),out/'combination_metrics.json')
    lines=['# Frozen price/activity state combinations','',GOAL['scope'],'','Price output is unchanged by routing; this is not evidence of learned preservation.','']
    for split,tasks in report['datasets'].items():
        for task,values in tasks.items():
            lines.extend([f'## {split} / {task}','','| representation | dimensions | Ridge | MLP1701 | MLP1702 | complete cases |','|---|---:|---:|---:|---:|---:|'])
            for name,row in values['variants'].items():
                vals=[row['metrics'][k]['all'] for k in sh.KINDS]
                lines.append(f'| {name} | {row["input_dimensions"]} | '+' | '.join('—' if v['primary_mse'] is None else f'{v["primary_mse"]:.4f}' for v in vals)+f' | {vals[0]["primary_support"]} |')
            lines.append('')
    (out/'summary.md').write_text('\n'.join(lines)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('all','worker','evaluate'))
    p.add_argument('--out',required=True);p.add_argument('--transfer');p.add_argument('--cross-run');p.add_argument('--name')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=256);p.add_argument('--streams',type=int,default=32);p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args();out=Path(a.out).resolve()
    if not torch.cuda.is_available():raise ValueError('Formal readout fitting runs on AutoDL CUDA')
    torch.set_num_threads(4)
    if a.action=='worker':
        if not a.name:p.error('--name required')
        worker(out,a.name);return
    if not a.transfer or not a.cross_run:p.error('--transfer and --cross-run required')
    if not 1<=a.jobs<=4:raise ValueError('jobs must be1..4')
    transfer=Path(a.transfer).resolve();cross=Path(a.cross_run).resolve()
    if any(out==src or out in src.parents or src in out.parents for src in (transfer,cross)):raise ValueError('Separate sibling output required')
    tm,identity=source_identity(transfer,cross);meta=make_manifest(transfer,cross,tm,identity,a.epochs,a.batch,a.streams)
    if (out/'manifest.json').exists():
        if read_json(out/'manifest.json')!=meta:raise ValueError('Settings changed; use a separate run')
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output lacks manifest')
    out.mkdir(parents=True,exist_ok=True);atomic_json(meta,out/'manifest.json');(out/'completion.json').unlink(missing_ok=True)
    atomic_json(dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),numpy=np.__version__,jobs=a.jobs),out/'runtime.json')
    prepare(meta,out,'cuda');torch.cuda.empty_cache()
    if a.action=='all':run_jobs(out,a.jobs,'obson.babel.state_combination')
    verify_code(meta);evaluate(out);verify_files(out/'cache',read_json(out/'cache/index.json')['files'])
    if source_identity(transfer,cross)[1]!=identity:raise ValueError('Upstream sources changed')
    atomic_json(dict(status='complete',experiments=len(meta['experiments']),encoder_updates=0,source_unchanged=True,
        independent_holdout=False,automatic_promotion=False),out/'completion.json')
    progress('Frozen-state combination matrix complete; both datasets remain research evaluations')


if __name__=='__main__':main()
