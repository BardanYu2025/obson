"""Locked best/last reconstruction, overlap and frozen-reader retention checks."""
from pathlib import Path
import numpy as np
import torch
from . import overlap_consistency_run as run
from .ae_extend import atomic_json
from .holdout_audit import read_json
from .progress import progress

ur=run.ur;odr=run.odr;pf=run.pf


@torch.no_grad()
def preflight(meta,out,device):
    data=ur.bb.load_data(run.alignment(meta),'val');x=torch.tensor(np.asarray(data['x'][:4]),device=device);rows=[]
    for seed in (42,43):
        jobs=[j for j in meta['experiments'] if j['seed']==seed];models=[run.construct(meta,j,device).eval() for j in jobs]
        if ur.bb.state_signature(models[0])!=ur.bb.state_signature(models[1]) or not torch.equal(models[0].core(x),models[1].core(x)):raise ValueError('Paired initialization differs')
        causal=pf.er.trained_causality(models[0],x)
        rows.append(dict(seed=seed,paired_initial_exact=True,causality=causal));atomic_json(rows,out/'preflight.json')
        if causal['status']=='failed':raise ValueError('Parent causal failure')



def require_nested(actual,expected,path='frozen utility'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError(path+': keys differ')
        for k in expected:require_nested(actual[k],expected[k],path+'/'+k)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError(path+': length differs')
        for i,(a,b) in enumerate(zip(actual,expected)):require_nested(a,b,path+'/'+str(i))
    elif expected is None or isinstance(expected,(str,bool)):
        if actual!=expected:raise ValueError(path+': value differs')
    elif not np.isclose(actual,expected,atol=1e-6,rtol=2e-5):raise ValueError(path+': numerical replay differs')


def utility(model,data,stats,reader,seed,batch,device):
    bank=reader['bank'];fitted=reader['fit'];name=f'control_s{seed}'
    z,_=odr.predict(model,data['x'],stats,batch,device)
    pred=ur.up.predict(fitted['heads'][name]['targets'],reader['heads'][name+'_weights'],reader['heads'][name+'_intercepts'],z,fitted['target_stats'])
    scores,errors=ur.up.measure(pred,bank['targets'],bank['mask'],fitted['target_stats'])
    return dict(scores=scores,errors={k:v.tolist() for k,v in errors.items()},predictions=pred.tolist())


def overlap(model,data,stats,batch,device):
    views=data['views'];zb,b=odr.predict(model,views[0]['x'],stats,batch,device);truth_b=odr.physical(views[0],stats);result={}
    for d in odr.od.SHIFTS:
        za,a=odr.predict(model,views[d]['x'],stats,batch,device)
        values,valid=odr.od.pair_scores(a,b,odr.physical(views[d],stats),truth_b,views[d]['mask'],views[0]['mask'],za,zb,d,stats)
        for label in ('gap','error_a','error_b'):
            keys=[f'{family}_{label}_nmse' for family in ('change1','body','activity')]
            count=sum(valid[k].astype(int) for k in keys)
            values[f'combined_{label}']=sum(values[k]*valid[k] for k in keys)/count.clip(1);valid[f'combined_{label}']=count>0
        result[str(d)]=dict(per_pair={k:v.tolist() for k,v in values.items()},valid={k:v.tolist() for k,v in valid.items()},summary=odr.summaries(values,valid,data['rows']))
    return result


def checks(meta,records,rows,overlap_rows,masks):
    config=meta['decision'];all_checks=[]
    for split in odr.SPLITS:
        for kind in ('best','last'):
            for seed in (42,43):
                a=f'consistent_s{seed}/{kind}';b=f'control_s{seed}/{kind}';f=f'frozen_s{seed}';bank=records[split]
                def check(name,x,y,cohort,factor=1.,extra=True):
                    ci=ur.interval(np.array(x),factor*np.array(y),cohort)
                    all_checks.append(dict(dataset=split,checkpoint=kind,seed=seed,metric=name,factor=factor,interval=ci,extra_condition=bool(extra),passed=bool(extra and ci['supported'] and ci['high'] is not None and ci['high']<=0)))
                for ref in (b,f):
                    for task,metric in [('global','primary'),('global','path'),('global','changes'),('global','body'),('global','activity'),('held','primary'),('recent','primary')]:
                        factor=1+(config['family_retention'] if task=='global' and metric!='primary' else config['reconstruction_retention'])
                        check(f'{task}/{metric} vs {ref}',bank[a]['reconstruction']['errors'][task][metric],bank[ref]['reconstruction']['errors'][task][metric],rows[split],factor)
                    readable=all(bank[a]['utility']['scores']['targets'][i]['r2'] is not None and bank[a]['utility']['scores']['targets'][i]['r2']>=.5 for i in (0,2,3,5))
                    check(f'utility vs {ref}',bank[a]['utility']['errors']['utility'],bank[ref]['utility']['errors']['utility'],rows[split],1+config['utility_retention'],readable)
                for d in meta['evaluation_shifts']:
                    av,bv=bank[a]['overlap'][str(d)],bank[b]['overlap'][str(d)]
                    for metric,factor in [('combined_gap',1-config['gap_gain']),('path_shape_gap_bps',1+config['reconstruction_retention'])]+[(f'combined_error_{side}',1+config['reconstruction_retention']) for side in ('a','b')]+[(f'path_native_mae_{side}_bps',1+config['reconstruction_retention']) for side in ('a','b')]:
                        ids=np.flatnonzero(np.array(av['valid'][metric])&np.array(bv['valid'][metric]));cohort=[overlap_rows[split][i] for i in ids]
                        check(f'overlap{d}/{metric}',np.array(av['per_pair'][metric])[ids],np.array(bv['per_pair'][metric])[ids],cohort,factor)
                for i,name in enumerate(ur.up.NAMES[6:],6):
                    ids=np.flatnonzero(masks[split][:,i]);r2=bank[a]['utility']['scores']['targets'][i]['r2']
                    check(name,np.array(bank[a]['utility']['errors'][name])[ids],np.full(len(ids),config['current_nmse']),[rows[split][j] for j in ids],extra=r2 is not None and r2>=config['current_r2'])
    return dict(status='candidate_for_review' if all(r['passed'] for r in all_checks) else 'no_full_upgrade',checks=all_checks,automatic_promotion=False,
        scope='Both seeds, both reused research sets, selected best and fixed last. Frozen reader retention tests compatibility, not a proof of information absence when failed. No old512 upgrade claim.')


@torch.no_grad()
def evaluate(meta,out,device):
    run.lock_selection(meta,out)  # All four fixed-budget jobs before any research readout.
    stats,local=run.statistics(meta);context=run.context(meta);context['models']=list(odr.MODELS)
    overlap_meta=odr.make_manifest(Path(meta['source']),meta['identity'],{s:meta['packed'][s] for s in odr.SPLITS},meta['evaluation_batch'])
    paired_data,_=odr.prepare(overlap_meta,out)
    root=Path(meta['reader']['root']);fitted=read_json(root/'fit.json')
    atomic_json(dict(reconstruction=stats,local=local,utility_targets=fitted['target_stats']),out/'evaluation_scales.json')
    with np.load(root/'heads.npz',allow_pickle=False) as archive:heads=dict(archive)
    rows={s:read_json(out/f'{s}_inventory.json') for s in odr.SPLITS};masks={};records={};causality={}
    validation=ur.bb.load_data(run.alignment(meta),'val');causal_x=torch.tensor(np.asarray(validation['x'][:4]),device=device)
    entries=[(f'frozen_s{s}',None,'best',s) for s in (42,43)]+[(j['name']+'/'+kind,j,kind,j['seed']) for j in meta['experiments'] for kind in ('best','last')]
    for split in odr.SPLITS:
        data=ur.bb.load_data(run.alignment(meta),split);ur.old.verify_cache(meta['reader']['manifest'],root,split);bank=ur.old.arrays(root,split)
        if not np.array_equal(bank['raw'],np.asarray(data['x']).reshape(len(rows[split]),-1)):raise ValueError('Frozen utility input order differs')
        targets,mask=ur.up.targets(ur.up.restore_raw(np.asarray(data['x']),stats))
        if not np.array_equal(mask,bank['mask']) or not np.allclose(targets,bank['targets'],atol=1e-10,rtol=1e-10):raise ValueError('Frozen utility targets changed')
        masks[split]=mask;records[split]={};atomic_json(dict(names=list(ur.up.NAMES),targets=targets.tolist(),mask=mask.tolist()),out/f'{split}_utility_targets.json')
        for label,job,kind,seed in entries:
            filename=f'{split}_{label.replace("/","_")}.json';path=out/filename
            # Results are intentionally recomputed after interruption: no unverified partial score reuse.
            if job is None:model,_,expected=ur.load_model(context,f'control_s{seed}',device)
            else:
                model=run.construct(meta,job,device);ck=torch.load(out/job['name']/f'{kind}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);expected=ck['validation'] if kind=='best' else ck['history'][-1]['validation']
            model.eval().requires_grad_(False)
            if label not in causality:
                if job is None:
                    pj=run.parent_job(meta,dict(seed=seed));actual=pf.validation(meta['identity']['manifest'],pj,model,validation,stats,local,device)
                else:actual=run.validation(meta,job,model,device)
                pf.ea.require_replay(actual,expected,label+'/selected validation');causal=pf.er.trained_causality(model,causal_x)
                causality[label]=dict(validation=actual,causality=causal);atomic_json(causality,out/'trained_validation_causality.json')
                if causal['status']=='failed':raise ValueError('Selected trained model noncausal')
            before=ur.bb.state_signature(model);scores,errors,decomp,pred=pf.score(model,data,stats,local,meta['evaluation_batch'],device)
            ids=np.linspace(0,len(rows[split])-1,min(4,len(rows[split])),dtype=int)
            result=dict(reconstruction=dict(scores=scores,errors={t:{k:v.tolist() for k,v in e.items()} for t,e in errors.items()},path_decomposition={k:v.tolist() for k,v in decomp.items()}),
                utility=utility(model,data,stats,dict(bank=bank,fit=fitted,heads=heads),seed,meta['evaluation_batch'],device),
                overlap=overlap(model,paired_data[split],stats,meta['evaluation_batch'],device),
                examples=[dict(index=int(i),prediction=pred[i].tolist(),target=data['y'][i].tolist(),mask=data['mask'][i].tolist()) for i in ids])
            if before!=ur.bb.state_signature(model):raise ValueError('Scoring mutated model')
            if job is None:
                prior=read_json(root/'utility_metrics.json')['datasets'][split]['scores'][f'control_s{seed}']
                require_nested(result['utility']['scores'],prior,label+'/frozen utility')
            atomic_json(result,path);records[split][label]=result;del model
            progress(f'Evaluated {split}/{label}: reconstruction, overlap and frozen readers')
    decision=checks(meta,records,rows,{s:paired_data[s]['rows'] for s in odr.SPLITS},masks)
    atomic_json(decision,out/'decision.json')
    summary={s:{n:dict(global_primary=v['reconstruction']['scores']['global']['metrics']['primary'],utility=v['utility']['scores']['groups']['utility'],overlap={d:x['summary']['all']['combined_gap']['mean'] for d,x in v['overlap'].items()}) for n,v in bank.items()} for s,bank in records.items()}
    atomic_json(dict(summary=summary,decision=decision,manifest=meta),out/'consistency_metrics.json')
    lines=['# Matched overlap consistency continuation','',f'Decision: {decision["status"]}; no automatic promotion.','', '| dataset/model | global error | frozen utility error | gap1 | gap16 | gap64 |','|---|---:|---:|---:|---:|---:|']
    for split,models in summary.items():
        for name,v in models.items():lines.append(f'| {split}/{name} | {v["global_primary"]:.6f} | {v["utility"]:.6f} | '+' | '.join(f'{v["overlap"][d]:.6f}' for d in ('1','16','64'))+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
