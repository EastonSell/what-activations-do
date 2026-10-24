"""Staged computation on explicitly supplied arrays. Importing never loads digits."""
import numpy as np
import core
from storage import write_json,write_npz


def validate_splits(splits,n):
    if set(splits)!={'train','validation','test'}: raise ValueError('Wrong split names')
    all_indices=np.concatenate([np.asarray(splits[k]) for k in ('train','validation','test')])
    if all_indices.dtype.kind not in 'iu' or sorted(all_indices.tolist())!=list(range(n)):
        raise ValueError('Splits must partition all examples exactly once')
    if any(len(splits[k])==0 for k in splits): raise ValueError('Empty split')

def train_stage(x,y,splits,cfg,out,budget):
    validate_splits(splits,len(y)); tr=splits['train']; va=splits['validation']
    accuracies={'A':[],'B':[]}; rows=[]
    write_npz(out/'splits.npz',**splits,train_x=x[tr],train_y=y[tr],validation_x=x[va],validation_y=y[va])
    for seed in cfg['seeds']:
        initial=core.initialize(seed,cfg['input_dim'],cfg['hidden_dim'],cfg['classes'])
        histories={}
        for arm in ('A','B'):
            budget.check()
            p,history=core.train(initial,x[tr],y[tr],seed,cfg,arm=='B',budget.check)
            histories[arm]=history
            train_h=core.hidden(p,x[tr]); val_h=core.hidden(p,x[va])
            mu,pcs,s=core.fit_geometry(train_h)
            val_logits=core.head(p,val_h); m=core.metrics(val_logits,y[va])
            accuracy=float(m['correct'].mean()); accuracies[arm].append(accuracy)
            rows.append({'seed':seed,'arm':arm,'validation_accuracy':accuracy,
                         'validation_ce':float(m['ce'].mean()),'epochs':cfg['epochs'],
                         'updates':history[-1]['updates'],'convergence':'not claimed; fixed epoch cap'})
            write_npz(out/f'model-{seed}-{arm}.npz',**p,
                      **{'initial_'+k:v for k,v in initial.items()},
                      mean=mu,pcs=pcs,singular_values=s,train_hidden=train_h,
                      validation_hidden=val_h,validation_logits=val_logits,
                      untrained_validation_logits=core.forward(initial,x[va]))
            write_json(out/f'history-{seed}-{arm}.json',history)
            budget.check()
        if [r['order_sha256'] for r in histories['A']]!=[r['order_sha256'] for r in histories['B']]:
            raise AssertionError('Paired batch schedule mismatch')
    result={'stage':'training','fit_count':len(rows),'rows':rows,
            'gates':core.gates(accuracies['A'],accuracies['B'],cfg['competence_floor'],cfg['competence_gap'])}
    write_json(out/'summary.json',result)
    return result

def load_model(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def test_stage(x,y,splits,cfg,training,out,budget):
    import json
    validate_splits(splits,len(y)); te=splits['test']; yy=y[te]
    training_summary=json.loads((training/'summary.json').read_text())
    if not training_summary['gates']['primary_pass']: raise ValueError('Primary validation gate failed')
    width=cfg['hidden_dim']; draws=cfg['random_draws']; ks=cfg['ranks']
    conditions=['pca']+[f'rank-{d}' for d in range(draws)]+[f'energy-{d}' for d in range(draws)]
    write_npz(out/'test-identity.npz',indices=te,labels=yy,test_x=x[te])
    effects={'A':[],'B':[]}; curve_rows=[]; integrity=[]
    for seed in cfg['seeds']:
        bases=[core.random_bases(seed,d,width) for d in range(draws)]
        q=np.array([b[0] for b in bases]); r=np.array([b[1] for b in bases])
        ortho=max(float(np.max(np.abs(v.T@v-np.eye(width)))) for v in [*q,*r])
        if ortho>cfg['numerical_atol']: raise AssertionError('Random basis not orthogonal')
        for arm in ('A','B'):
            budget.check(); model=load_model(training/f'model-{seed}-{arm}.npz')
            p={key:model[key] for key in core.KEYS}; initial={key:model['initial_'+key] for key in core.KEYS}
            h=core.hidden(p,x[te]); baseline=core.head(p,h); mu=model['mean']; v=model['pcs']
            logits=np.empty((len(ks),len(conditions),len(te),cfg['classes']),dtype=np.float64)
            norms=np.empty((len(ks),len(conditions),len(te))); negative=np.empty_like(norms)
            ce=np.empty_like(norms); max_energy_error=0.; max_identity_error=0.; max_endpoint_error=0.
            for ki,k in enumerate(ks):
                budget.check(); top=core.intervene(h,mu,v,k,'pca')
                target_norm=np.linalg.norm(h-top,axis=1)
                for ci,condition in enumerate(conditions):
                    if ci==0: edited=top
                    elif ci<=draws: edited=core.intervene(h,mu,q[ci-1],k,'rank')
                    else: edited=core.intervene(h,mu,v,k,'energy',r[ci-draws-1])
                    result=core.head(p,edited); logits[ki,ci]=result
                    norms[ki,ci]=np.linalg.norm(h-edited,axis=1)
                    negative[ki,ci]=(edited<0).mean(axis=1)
                    m=core.metrics(result,yy); ce[ki,ci]=m['ce']
                    if ci>draws: max_energy_error=max(max_energy_error,float(np.max(np.abs(norms[ki,ci]-target_norm))))
                    if k==0: max_identity_error=max(max_identity_error,float(np.max(np.abs(result-baseline))))
                    if k==width and ci<=draws: max_endpoint_error=max(max_endpoint_error,float(np.max(np.abs(edited-mu))))
                    curve_rows.append({'seed':seed,'arm':arm,'rank':k,'condition':condition,
                                       'accuracy':float(m['correct'].mean()),'ce':float(m['ce'].mean()),
                                       'margin':float(m['margin'].mean()),'norm':float(norms[ki,ci].mean()),
                                       'negative_fraction':float(negative[ki,ci].mean())})
            errors={'seed':seed,'arm':arm,'orthogonality':ortho,'energy':max_energy_error,
                    'identity':max_identity_error,'full_rank_endpoint':max_endpoint_error}
            if max(errors[k] for k in ('orthogonality','energy','identity','full_rank_endpoint'))>cfg['numerical_atol']:
                raise AssertionError('Intervention integrity failed: '+str(errors))
            integrity.append(errors)
            ix=[ks.index(k) for k in cfg['primary_ranks']]
            effect=(ce[ix,0,:]-ce[ix,draws+1:,:].mean(axis=1)).mean(axis=0)
            effects[arm].append(effect)
            write_npz(out/f'test-{seed}-{arm}.npz',logits=logits,baseline_logits=baseline,
                      untrained_logits=core.forward(initial,x[te]),perturbation_norm=norms,
                      negative_fraction=negative,test_hidden=h,rank_bases=q,energy_rotations=r,
                      mean=mu,pcs=v,W2=p['W2'],b2=p['b2'],primary_effect=effect)
            budget.check()
    a=np.array(effects['A']); b=np.array(effects['B'])
    primary=core.bootstrap_cases(a,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
    lower,upper=primary['ci95']; positive=int(np.sum(np.array(primary['per_seed'])>0))
    primary['positive_seeds']=positive
    primary['interpretation']=('supports_conditional_excess' if lower>0 and positive>=int(np.ceil(.8*len(a)))
                               else 'against_predicted_excess' if upper<=0 else 'inconclusive')
    comparison=core.bootstrap_cases(a-b,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
    comparison['eligible']=training_summary['gates']['comparison_pass']
    comparison['interpretation']='descriptive secondary only' if comparison['eligible'] else 'competence mismatch; no learning-specific conclusion'
    summary={'stage':'heldout','model_arm_count':2*len(cfg['seeds']),'examples':len(te),
             'ranks':ks,'conditions':conditions,'primary':primary,'secondary':comparison,
             'integrity':integrity,'reuse_limit':'Previously used benchmark; not new/writer-held-out data'}
    write_npz(out/'effects.npz',A=a,B=b)
    write_json(out/'curves.json',curve_rows); write_json(out/'summary.json',summary)
    return summary
