"""Locked12-checkpoint readout fit, reconstruction, stability and capacity decisions."""
from pathlib import Path
import numpy as np
import torch
from . import capacity_growth_run as run
from .ae_extend import atomic_json
from .dual_state import sha256,verify_files
from .holdout_audit import read_json
from .progress import progress

rr=run.rr;ur=run.ur;cr=run.cr;ce=run.ce;up=rr.up;old=rr.old


def entries(meta):return [(j['name']+'_'+kind,j,kind) for j in meta['experiments'] for kind in ('best','last')]


def check_models(out):
    lock=read_json(out/'model_selection_lock.json');meta=read_json(out/'manifest.json')
    names={j['name'] for j in meta['experiments']}
    if lock['manifest']!=meta or set(lock['trials'])!=names or set(lock['weights'])!=names or any(set(w)!={'best','last'} for w in lock['weights'].values()):raise ValueError('All model selections required')
    for name,weights in lock['weights'].items():
        for kind,digest in weights.items():
            if sha256(out/name/f'{kind}.pt')!=digest:raise ValueError('Selected model changed')
    return lock


def check_readouts(out):
    lock=read_json(out/'readout_selection_lock.json')
    if lock['manifest_sha256']!=sha256(out/'manifest.json') or lock['model_selection_sha256']!=sha256(out/'model_selection_lock.json'):raise ValueError('Readout/model lock changed')
    verify_files(out,lock['files'])
    for split,h in lock['cache_indexes'].items():
        if sha256(out/f'cache/{split}_index.json')!=h:raise ValueError('Readout fitting input changed')
        old.verify_cache(read_json(out/'manifest.json'),out,split)
    return lock


def load(meta,out,job,kind,device):
    model=run.construct(meta,job,device);ck=torch.load(out/job['name']/f'{kind}.pt',map_location='cpu',weights_only=True);lock=read_json(out/'model_selection_lock.json')['trials'][job['name']]
    if ck['metadata']!=dict(manifest=meta,job=job):raise ValueError('Evaluation checkpoint identity differs')
    if kind=='best':
        if ck['epoch']!=lock['selected_epoch'] or ck['validation']!=lock['validation']:raise ValueError('Best checkpoint changed')
        expected=ck['validation']
    else:
        if ck['epoch']!=meta['epochs']:raise ValueError('Incomplete fixed last')
        expected=ck['history'][-1]['validation']
    model.load_state_dict(ck['model']);model.eval().requires_grad_(False);return model,expected


@torch.inference_mode()
def prepare(meta,out,split,device):
    check_models(out)
    if split in old.SPLITS[2:]:check_readouts(out)
    elif split not in old.SPLITS:raise ValueError('Unknown split')
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/f'{split}_index.json').exists():old.verify_cache(meta,out,split);return
    source=Path(meta['source']);old.verify_cache(meta['identity']['manifest'],source,split);bank=old.arrays(source,split)
    rows=read_json(out/f'{split}_inventory.json')
    if rows!=read_json(source/f'{split}_inventory.json'):raise ValueError('Original row/week identity changed')
    data=ur.bb.load_data(cr.alignment(run.parent_meta(meta)),split);x=np.asarray(data['x']);stats,_=run.statistics(meta);rr.check_inputs(bank,x,stats,rows)
    values={k:bank[k] for k in ('raw','current','targets','mask')}
    for seed in (42,43):values[f'parent_s{seed}']=bank[f'consistent_s{seed}']
    for name,job,kind in entries(meta):
        model,expected=load(meta,out,job,kind,device);before=ur.bb.state_signature(model)
        if split=='train':
            actual=run.validation(meta,job,model,device);ur.pf.ea.require_replay(actual,expected,name+'/trained validation')
            val=ur.bb.load_data(cr.alignment(run.parent_meta(meta)),'val');causal=ur.pf.er.trained_causality(model,torch.tensor(np.asarray(val['x'][:4]),device=device))
            if causal['status']=='failed':raise ValueError('Trained expanded model noncausal')
            path=out/'trained_validation_causality.json';checks=read_json(path) if path.exists() else {};checks[name]=dict(validation=actual,causality=causal);atomic_json(checks,path)
        zs=[]
        for start in range(0,len(x),meta['evaluation_batch']):zs.append(model.core.encoder(torch.tensor(x[start:start+meta['evaluation_batch']],device=device))[:,-1].cpu().numpy())
        values[name]=np.concatenate(zs)
        if values[name].shape!=(len(rows),meta['state_width']) or before!=ur.bb.state_signature(model) or any(p.requires_grad for p in model.parameters()):raise ValueError('State shape/frozen weight failure')
        del model
    if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite extraction')
    np.savez(cache/f'{split}.npz',**values);atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),windows=len(rows),files={f'{split}.npz':sha256(cache/f'{split}.npz')}),cache/f'{split}_index.json')
    progress(f'Extracted12 checkpoints: {split}, {len(rows)} windows')


def fit(meta,out,device):
    check_models(out)
    if (out/'readout_selection_lock.json').exists():check_readouts(out);return
    for s in ('train','val'):old.verify_cache(meta,out,s)
    train,val=(old.arrays(out,s) for s in ('train','val'));stats=up.target_scales(train['targets'],train['mask']);ce.require_nested(stats,read_json(Path(meta['source'])/'fit.json')['target_stats'],'original train target scales')
    heads={};arrays={}
    for name,job,kind in entries(meta):
        head,w,b=up.fit_heads(train[name],train['targets'],train['mask'],val[name],val['targets'],val['mask'],stats,device)
        heads[name]=dict(targets=head,input_width=train[name].shape[1]);arrays[name+'_weights']=w;arrays[name+'_intercepts']=b;progress(f'{name}: selected13 readouts using original5 alphas')
    np.savez(out/'heads.npz',**arrays);atomic_json(dict(target_stats=stats,heads=heads,alphas=list(up.probe.ALPHAS),fits_use_train_only=True,candidates=len(heads)*13*5),out/'fit.json')
    atomic_json(dict(manifest_sha256=sha256(out/'manifest.json'),model_selection_sha256=sha256(out/'model_selection_lock.json'),files={n:sha256(out/n) for n in ('heads.npz','fit.json')},cache_indexes={s:sha256(out/f'cache/{s}_index.json') for s in ('train','val')}),out/'readout_selection_lock.json')


def decide(meta,records,rows,overlap_rows,masks):
    checks=[]
    for split,bank in records.items():
        for seed in (42,43):
            for kind in ('best','last'):
                for mode,reference in (('wide','base'),('deep','wide'),('deep','base')):
                    a=f'{mode}_s{seed}_{kind}';b=f'{reference}_s{seed}_{kind}';parent=f'parent_s{seed}';pair=mode+'_vs_'+reference
                    def check(metric,x,y,factor=1.,extra=True,cohort=None):
                        ci=ur.interval(np.asarray(x),factor*np.asarray(y),rows[split] if cohort is None else cohort)
                        checks.append(dict(dataset=split,seed=seed,checkpoint=kind,comparison=pair,metric=metric,factor=factor,interval=ci,extra_condition=bool(extra),passed=bool(extra and ci['supported'] and ci['high'] is not None and ci['high']<=0)))
                    check('global_gain',bank[a]['reconstruction']['errors']['global']['primary'],bank[b]['reconstruction']['errors']['global']['primary'],.95)
                    for ref in (b,parent):
                        for task,metric in [('global','primary'),('global','path'),('global','changes'),('global','body'),('global','activity'),('held','primary'),('recent','primary')]:
                            factor=1.10 if task=='global' and metric!='primary' else 1.05
                            check(f'{task}/{metric} vs {ref}',bank[a]['reconstruction']['errors'][task][metric],bank[ref]['reconstruction']['errors'][task][metric],factor)
                        readable=all(bank[n]['utility']['scores']['targets'][i]['r2'] is not None and bank[n]['utility']['scores']['targets'][i]['r2']>=.5 for n in (a,ref) for i in (0,2,3,5))
                        for task,factor in [('utility',1.05),('direction',1.10),('volatility',1.10)]:check(f'{task} vs {ref}',bank[a]['utility']['errors'][task],bank[ref]['utility']['errors'][task],factor,readable)
                        for d in ('1','16','64'):
                            av,bv=bank[a]['overlap'][d],bank[ref]['overlap'][d]
                            for metric in ('combined_gap','combined_error_a','combined_error_b','path_shape_gap_bps','path_native_mae_a_bps','path_native_mae_b_bps'):
                                ids=np.flatnonzero(np.array(av['valid'][metric])&np.array(bv['valid'][metric]));cohort=[overlap_rows[split][i] for i in ids]
                                check(f'overlap{d}/{metric} vs {ref}',np.array(av['per_pair'][metric])[ids],np.array(bv['per_pair'][metric])[ids],1.05,cohort=cohort)
                    for i,name in enumerate(up.NAMES[6:],6):
                        ids=np.flatnonzero(masks[split][:,i]);r2=bank[a]['utility']['scores']['targets'][i]['r2']
                        check(name,np.array(bank[a]['utility']['errors'][name])[ids],np.full(len(ids),.1),extra=r2 is not None and r2>=.8,cohort=[rows[split][j] for j in ids])
    passed={name:all(c['passed'] for c in checks if c['comparison']==name) for name in ('wide_vs_base','deep_vs_wide','deep_vs_base')}
    status='deeper_candidate' if passed['deep_vs_wide'] and passed['deep_vs_base'] else 'wider_candidate' if passed['wide_vs_base'] else 'no_capacity_upgrade'
    return dict(status=status,comparisons=passed,checks=checks,automatic_promotion=False,scope='Both seeds, both reused research sets, best AND last; paired weekly intervals uncorrected for multiplicity. Equal exposures/updates, different compute. Fixed100-epoch grown-model adaptation, not a universal depth optimum or convergence proof.')


@torch.inference_mode()
def evaluate(meta,out,device):
    check_models(out);check_readouts(out);cm=run.parent_meta(meta);stats,local=run.statistics(meta);source=Path(meta['source']);original_reader=Path(cm['reader']['root'])
    om=cr.odr.make_manifest(Path(cm['source']),cm['identity'],{s:cm['packed'][s] for s in cr.odr.SPLITS},meta['evaluation_batch']);paired,_=cr.odr.prepare(om,out)
    fitted=read_json(out/'fit.json');parentfit=read_json(source/'fit.json');originalfit=read_json(original_reader/'fit.json')
    atomic_json(dict(reconstruction=stats,local=local,utility_targets=fitted['target_stats']),out/'evaluation_scales.json')
    def arrays(path):
        with np.load(path,allow_pickle=False) as f:return dict(f)
    heads=arrays(out/'heads.npz');pheads=arrays(source/'heads.npz');oheads=arrays(original_reader/'heads.npz');pca=arrays(original_reader/'pca.npz')['components']
    records={};inventories={};masks={};absolute={};reader_rows={}
    all_entries=[(f'parent_s{s}',None,'best',s) for s in (42,43)]+[(n,j,k,j['seed']) for n,j,k in entries(meta)]
    for split in ('test','cross_research'):
        old.verify_cache(meta,out,split);bank=old.arrays(out,split);rows=read_json(out/f'{split}_inventory.json');inventories[split]=rows;masks[split]=bank['mask'];records[split]={};scores={};errors={};predictions={};groups={}
        data=ur.bb.load_data(cr.alignment(cm),split)
        def score_reader(name,pred):
            score,err=up.measure(pred,bank['targets'],bank['mask'],fitted['target_stats']);scores[name]=score;errors[name]=err;predictions[name]=pred.tolist();groups[name]={}
            for field in ('symbol','period'):
                for value in sorted({str(r[field]) for r in rows}):
                    ids=np.array([str(r[field])==value for r in rows]);groups[name][field+'/'+value]=up.measure(pred[ids],bank['targets'][ids],bank['mask'][ids],fitted['target_stats'])[0]
            return dict(scores=score,errors={k:v.tolist() for k,v in err.items()},predictions=pred.tolist())
        for name,job,kind,seed in all_entries:
            if job is None:
                model,_,_=rr.load_selected(meta['identity']['manifest'],f'consistent_s{seed}',device);hn=f'consistent_s{seed}';fit=parentfit;weights=pheads
            else:model,_=load(meta,out,job,kind,device);hn=name;fit=fitted;weights=heads
            model.eval().requires_grad_(False);before=ur.bb.state_signature(model)
            recon,err,decomp,pred=ur.pf.score(model,data,stats,local,meta['evaluation_batch'],device)
            reader_pred=up.predict(fit['heads'][hn]['targets'],weights[hn+'_weights'],weights[hn+'_intercepts'],bank[name],fit['target_stats']);utility=score_reader(name,reader_pred)
            record=dict(reconstruction=dict(scores=recon,errors={t:{k:v.tolist() for k,v in e.items()} for t,e in err.items()},path_decomposition={k:v.tolist() for k,v in decomp.items()}),utility=utility,overlap=ce.overlap(model,paired[split],stats,meta['evaluation_batch'],device),examples=[dict(index=int(i),prediction=pred[i].tolist(),target=data['y'][i].tolist(),mask=data['mask'][i].tolist()) for i in np.linspace(0,len(rows)-1,4,dtype=int)])
            if job is None:
                oldrecon=read_json(run.parent_root(meta)/f'{split}_consistent_s{seed}_best.json');ce.require_nested(recon,oldrecon['reconstruction']['scores'],'frozen400 reconstruction replay')
                ce.require_nested(utility['scores'],read_json(source/'readout_metrics.json')['datasets'][split]['scores'][hn],'parent calibrated utility replay')
                for d in ('1','16','64'):ce.require_nested(record['overlap'][d]['summary'],oldrecon['overlap'][d]['summary'],'parent stability replay')
            if before!=ur.bb.state_signature(model):raise ValueError('Scoring mutated weights')
            atomic_json(record,out/f'{split}_{name}.json');records[split][name]=record;del model
            progress(f'Evaluated {split}/{name}: reconstruction, stability and recalibrated utility')
        for name in rr.BASELINES:
            x=old.representation(name,bank,pca,originalfit['raw_stats']);score_reader(name,up.predict(originalfit['heads'][name]['targets'],oheads[name+'_weights'],oheads[name+'_intercepts'],x,originalfit['target_stats']))
            ce.require_nested(scores[name],read_json(source/'readout_metrics.json')['datasets'][split]['scores'][name],name+'/baseline replay')
        score_reader('train_mean',np.tile(fitted['target_stats']['mean'],(len(rows),1)))
        absolute[split]=ur.decide(dict(models=[n for n,_,_ in entries(meta)],pca_rank=768,decision=cm['reader']['manifest']['decision']),scores,errors,bank['mask'],rows)
        reader_rows[split]=dict(scores=scores,groups=groups)
        atomic_json(dict(target_names=list(up.NAMES),targets=bank['targets'].tolist(),mask=bank['mask'].tolist(),predictions=predictions,per_window_errors={n:{k:v.tolist() for k,v in e.items()} for n,e in errors.items()}),out/f'{split}_predictions.json')
    decision=decide(meta,records,inventories,{s:paired[s]['rows'] for s in paired},masks);decision['original_utility_protocol']=absolute
    atomic_json(decision,out/'decision.json');atomic_json(dict(datasets=reader_rows,decision=decision),out/'readout_metrics.json')
    summary={s:{n:dict(global_primary=v['reconstruction']['scores']['global']['metrics']['primary'],utility=v['utility']['scores']['groups']['utility'],gap1=v['overlap']['1']['summary']['all']['combined_gap']['mean']) for n,v in bank.items()} for s,bank in records.items()}
    atomic_json(dict(summary=summary,decision=decision),out/'growth_metrics.json')
    lines=['# Fixed768 FFN/depth growth','',f'Decision: {decision["status"]}; no automatic promotion.','', '| dataset/model | global primary | calibrated utility | overlap gap1 |','|---|---:|---:|---:|']
    for s,bank in summary.items():
        for n,v in bank.items():lines.append(f'| {s}/{n} | {v["global_primary"]:.6f} | {v["utility"]:.6f} | {v["gap1"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
