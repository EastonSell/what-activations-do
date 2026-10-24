"""Independent artifact audit: no core/pipeline imports, dataset load, or training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from storage import sha256,verify_manifest

ROOT=Path(__file__).resolve().parent
SOURCE_BASE=['PROTOCOL.md','README.md','config.json','core.py','pipeline.py','storage.py','study.py','run_stage.py','audit.py']

def read(path):
    return json.loads(Path(path).read_text())

def close(actual,expected,label,atol=1e-10):
    a=np.asarray(actual); b=np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=atol,rtol=0):
        raise ValueError(label+' mismatch')

def accepted(root):
    root=Path(root)
    try:
        if (root/'FAILED.json').exists(): raise ValueError('Stage failure record exists')
        watchdog=read(root/'WATCHDOG.json'); complete=read(root/'COMPLETE.json')
        if watchdog['status']!='completed' or watchdog['returncode']!=0 or not np.isfinite(watchdog['elapsed_seconds']) or watchdog['elapsed_seconds']<0 or watchdog['elapsed_seconds']>watchdog['limit_seconds']:
            raise ValueError('Watchdog did not accept execution')
        if sha256(root/'MANIFEST.json')!=complete['manifest_sha256']: raise ValueError('Completion hash mismatch')
        verify_manifest(root,read(root/'MANIFEST.json'))
    except (OSError,KeyError,json.JSONDecodeError) as exc:
        raise ValueError('Missing or invalid completion evidence') from exc

def validate_stage(root,stage,source_root=None,training_hash=None):
    root=Path(root); source_root=Path(source_root or ROOT); accepted(root)
    source=read(source_root/'SOURCE_MANIFEST.json')
    expected_source=set(SOURCE_BASE+[str(p.relative_to(source_root)) for p in (source_root/'tests').glob('*.py')])
    if set(source['files'])!=expected_source: raise ValueError('Source manifest coverage mismatch')
    verify_manifest(source_root,source)
    source_hash=sha256(source_root/'SOURCE_MANIFEST.json'); cfg=read(source_root/'config.json')
    meta=read(root/'metadata.json')
    if meta['stage']!=stage or meta['source_manifest_sha256']!=source_hash or meta['config']!=cfg or meta['dataset_sha256']!=cfg['data_sha256']:
        raise ValueError('Artifact config/source binding mismatch')
    clearance=meta['clearance']
    if clearance.get('accepted') is not True or clearance.get('stage')!=stage or clearance.get('source_manifest_sha256')!=source_hash or not clearance.get('review_reference'):
        raise ValueError('Clearance binding mismatch')
    if meta.get('training_manifest_sha256')!=training_hash or (stage=='heldout' and clearance.get('training_manifest_sha256')!=training_hash):
        raise ValueError('Training manifest binding mismatch')
    names={'metadata.json','resources.json','summary.json'}
    if stage=='training':
        names.add('splits.npz')
        for seed in cfg['seeds']:
            for arm in ('A','B'): names.update({f'model-{seed}-{arm}.npz',f'history-{seed}-{arm}.json'})
    else:
        names.update({'test-identity.npz','curves.json','effects.npz'})
        names.update(f'test-{seed}-{arm}.npz' for seed in cfg['seeds'] for arm in ('A','B'))
    if set(read(root/'MANIFEST.json')['files'])!=names: raise ValueError('Artifact manifest coverage mismatch')
    if {p.name for p in root.iterdir()}!=names|{'MANIFEST.json','COMPLETE.json','WATCHDOG.json'}:
        raise ValueError('Unexpected artifact files or directories')
    resources=read(root/'resources.json'); watchdog=read(root/'WATCHDOG.json')
    for key,limit in [('wall_seconds',cfg['wall_seconds']),('rss_peak_bytes',cfg['rss_bytes']),('retained_bytes',cfg['disk_bytes'])]:
        value=resources[key]
        if not np.isfinite(value) or value<0 or value>limit: raise ValueError('Resource bound exceeded: '+key)
    if watchdog['limit_seconds']!=cfg['wall_seconds'] or watchdog['elapsed_seconds']>cfg['wall_seconds']:
        raise ValueError('Watchdog configured bound mismatch')
    if sum(p.stat().st_size for p in root.iterdir())>cfg['disk_bytes']: raise ValueError('Actual output exceeds disk cap')
    return cfg

def rescore(z,y):
    z=np.asarray(z); y=np.asarray(y)
    if z.ndim!=2 or y.shape!=(len(z),) or y.dtype.kind not in 'iu' or not np.isfinite(z).all() or np.any(y<0) or np.any(y>=z.shape[1]):
        raise ValueError('Invalid logits or labels')
    true=z[np.arange(len(y)),y]
    loss=np.logaddexp.reduce(z,axis=-1)-true
    competitor=np.max(np.where(np.arange(z.shape[-1])==y[:,None],-np.inf,z),axis=-1)
    return {'ce':loss,'margin':true-competitor,'correct':np.argmax(z,axis=-1)==y}

def initial_parameters(cfg,seed):
    gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,0])))
    result={}
    for i,n,m in [(1,cfg['input_dim'],cfg['hidden_dim']),(2,cfg['hidden_dim'],cfg['classes'])]:
        bound=np.sqrt(6/(n+m))
        result[f'W{i}']=gen.uniform(-bound,bound,(n,m)); result[f'b{i}']=gen.uniform(-bound,bound,m)
    return result

def expected_basis(seed,draw,kind,width):
    gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,2,draw,kind])))
    q,r=np.linalg.qr(gen.normal(size=(width,width))); signs=np.where(np.diag(r)<0,-1.,1.)
    return q*signs

def audit_training(root,source_root=None):
    root=Path(root); cfg=validate_stage(root,'training',source_root)
    summary=read(root/'summary.json'); expected_models=2*len(cfg['seeds'])
    if summary['stage']!='training' or summary['fit_count']!=expected_models or len(summary['rows'])!=expected_models: raise ValueError('Training coverage mismatch')
    scores={'A':[],'B':[]}; total_fit_seconds=0.
    with np.load(root/'splits.npz',allow_pickle=False) as f:
        tr=f['train']; va=f['validation']; te=f['test']; all_indices=np.concatenate([tr,va,te])
        if all_indices.dtype.kind not in 'iu' or sorted(all_indices.tolist())!=list(range(len(all_indices))): raise ValueError('Split partition invalid')
        tx=f['train_x']; ty=f['train_y']; vx=f['validation_x']; vy=f['validation_y']
        if tx.shape!=(len(tr),cfg['input_dim']) or vx.shape!=(len(va),cfg['input_dim']) or len(ty)!=len(tr) or len(vy)!=len(va): raise ValueError('Split data shape mismatch')
    for seed in cfg['seeds']:
        initial=initial_parameters(cfg,seed)
        for arm in ('A','B'):
            with np.load(root/f'model-{seed}-{arm}.npz',allow_pickle=False) as f:
                for key,expected in initial.items():
                    if not np.array_equal(f['initial_'+key],expected): raise ValueError('Prescribed initialization mismatch')
                th=np.maximum(0,tx@f['W1']+f['b1']); vh=np.maximum(0,vx@f['W1']+f['b1'])
                vl=vh@f['W2']+f['b2']; tl=th@f['W2']+f['b2']
                close(th,f['train_hidden'],'Training activation'); close(vh,f['validation_hidden'],'Validation activation')
                close(vl,f['validation_logits'],'Validation output'); close(th.mean(axis=0),f['mean'],'Training mean')
                untrained=np.maximum(0,vx@initial['W1']+initial['b1'])@initial['W2']+initial['b2']
                close(untrained,f['untrained_validation_logits'],'Untrained validation output')
                v=f['pcs']; cov=(th-f['mean']).T@(th-f['mean']); eig=v.T@cov@v
                close(v.T@v,np.eye(v.shape[0]),'PCA orthogonality')
                if not np.allclose(eig,np.diag(f['singular_values']**2),atol=1e-8,rtol=1e-10): raise ValueError('PCA eigensystem mismatch')
                if np.any(np.diff(f['singular_values'])>1e-10): raise ValueError('PC order is not descending')
                if arm=='B':
                    for key in ('W1','b1'):
                        if f[key].tobytes()!=initial[key].tobytes(): raise ValueError('Frozen hidden weights changed')
                m=rescore(vl,vy); accuracy=float(m['correct'].mean()); scores[arm].append(accuracy)
                rows=[r for r in summary['rows'] if r['seed']==seed and r['arm']==arm]
                if len(rows)!=1 or rows[0]['validation_accuracy']!=accuracy: raise ValueError('Validation row mismatch')
                close(rows[0]['validation_ce'],m['ce'].mean(),'Validation CE')
                final_loss=rescore(tl,ty)['ce'].mean()+cfg['alpha']/(2*len(tr))*(np.sum(f['W1']**2)+np.sum(f['W2']**2))
            history=read(root/f'history-{seed}-{arm}.json')
            if len(history)!=cfg['epochs']: raise ValueError('Epoch count mismatch')
            prior=0.
            for e,row in enumerate(history):
                expected_updates=(e+1)*int(np.ceil(len(tr)/cfg['batch_size']))
                if row['updates']!=expected_updates or row['epoch']!=e+1: raise ValueError('Update count mismatch')
                order=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1,e]))).permutation(len(tr))
                expected_hash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
                if row['order_sha256']!=expected_hash: raise ValueError('Prescribed batch order mismatch')
                if not np.isfinite(row['loss']) or row['loss']<0: raise ValueError('Invalid training loss')
                elapsed=row['elapsed_seconds']; duration=row['epoch_seconds']
                if not np.isfinite(elapsed) or not np.isfinite(duration) or elapsed<prior or duration<0: raise ValueError('Invalid epoch elapsed time')
                close(duration,elapsed-prior,'Epoch time accounting'); prior=elapsed
            close(history[-1]['loss'],final_loss,'Final training loss'); total_fit_seconds+=prior
            if rows[0]['epochs']!=cfg['epochs'] or rows[0]['updates']!=history[-1]['updates']: raise ValueError('Training row schedule mismatch')
    if total_fit_seconds>read(root/'resources.json')['wall_seconds']+1e-8: raise ValueError('Fit time exceeds stage time')
    a=np.array(scores['A']); b=np.array(scores['B']); primary=bool(np.all(a>=cfg['competence_floor']))
    comparison=bool(primary and np.all(b>=cfg['competence_floor']) and np.all(np.abs(a-b)<=cfg['competence_gap']))
    gates={'primary_pass':primary,'comparison_pass':comparison}
    if summary['gates']!=gates: raise ValueError('Gate mismatch')
    return {'accepted':True,'models':expected_models,'gates':gates,
            'limit':'Verifies archived inputs, seeded initialization/orders, forward outputs and final loss; not independent retraining'}

def bootstrap_statistics(effects,cfg):
    effects=np.asarray(effects); per_seed=np.mean(effects,axis=1); case_mean=np.mean(effects,axis=0)
    gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([cfg['bootstrap_seed']])))
    draws=np.empty(cfg['bootstrap_draws'])
    for i in range(len(draws)):
        draw=gen.integers(0,len(case_mean),len(case_mean)); draws[i]=np.sum(case_mean[draw])/len(draw)
    draws.sort(); bounds=[]
    for q in (.025,.975):
        ix=q*(len(draws)-1); lower=int(np.floor(ix)); upper=int(np.ceil(ix))
        bounds.append(float(draws[lower]+(ix-lower)*(draws[upper]-draws[lower])))
    return {'mean':float(np.mean(case_mean)),'per_seed':per_seed.tolist(),'ci95':bounds}

def check_statistics(actual,expected,label):
    for key in ('mean','per_seed','ci95'): close(actual[key],expected[key],label+' '+key)

def audit_test(root,training=None,source_root=None):
    if training is None: raise ValueError('Held-out audit requires its sealed training directory')
    root=Path(root); training=Path(training)
    train_audit=audit_training(training,source_root)
    if not train_audit['gates']['primary_pass']: raise ValueError('Primary competence gate failed')
    cfg=validate_stage(root,'heldout',source_root,sha256(training/'MANIFEST.json'))
    summary=read(root/'summary.json'); curves=read(root/'curves.json'); n_draw=cfg['random_draws']; width=cfg['hidden_dim']
    conditions=['pca']+[f'rank-{d}' for d in range(n_draw)]+[f'energy-{d}' for d in range(n_draw)]
    if summary['stage']!='heldout' or summary['ranks']!=cfg['ranks'] or summary['conditions']!=conditions or summary['model_arm_count']!=2*len(cfg['seeds']): raise ValueError('Held-out coverage mismatch')
    curve_index={(r['seed'],r['arm'],r['rank'],r['condition']):r for r in curves}
    expected_cells={(seed,arm,k,c) for seed in cfg['seeds'] for arm in ('A','B') for k in cfg['ranks'] for c in conditions}
    if set(curve_index)!=expected_cells or len(curves)!=len(expected_cells): raise ValueError('Missing/duplicate curve cells')
    with np.load(root/'test-identity.npz',allow_pickle=False) as f: y=f['labels']; indices=f['indices']; test_x=f['test_x']
    with np.load(training/'splits.npz',allow_pickle=False) as f:
        if not np.array_equal(indices,f['test']) or test_x.shape!=(len(indices),cfg['input_dim']) or summary['examples']!=len(indices): raise ValueError('Held-out data identity mismatch')
        n=len(f['train'])+len(f['validation'])+len(indices); full_x=np.empty((n,cfg['input_dim']),dtype=np.float64); full_y=np.empty(n,dtype=np.int64)
        full_x[f['train']]=f['train_x']; full_y[f['train']]=f['train_y']
        full_x[f['validation']]=f['validation_x']; full_y[f['validation']]=f['validation_y']
        full_x[indices]=test_x; full_y[indices]=y
    data_hash=hashlib.sha256((full_x*16.).tobytes()+full_y.tobytes()).hexdigest()
    if data_hash!=cfg['data_sha256']: raise ValueError('Reconstructed complete dataset hash mismatch')
    effects={'A':[],'B':[]}; max_error=0.; observed_integrity=[]
    for seed in cfg['seeds']:
        expected_q=np.array([expected_basis(seed,d,0,width) for d in range(n_draw)])
        expected_r=np.array([expected_basis(seed,d,1,width) for d in range(n_draw)])
        for arm in ('A','B'):
            with np.load(training/f'model-{seed}-{arm}.npz',allow_pickle=False) as model, np.load(root/f'test-{seed}-{arm}.npz',allow_pickle=False) as f:
                for key in ('W2','b2','mean','pcs'):
                    if not np.array_equal(f[key],model[key]): raise ValueError('Sealed training head/geometry linkage mismatch')
                h=np.maximum(0,test_x@model['W1']+model['b1']); close(h,f['test_hidden'],'Held-out activation')
                mu=model['mean']; v=model['pcs']; z=h-mu; baseline=h@model['W2']+model['b2']
                close(baseline,f['baseline_logits'],'Held-out baseline')
                initial=initial_parameters(cfg,seed)
                untrained=np.maximum(0,test_x@initial['W1']+initial['b1'])@initial['W2']+initial['b2']
                close(untrained,f['untrained_logits'],'Wholly untrained held-out output')
                close(f['rank_bases'],expected_q,'Seeded rank bases'); close(f['energy_rotations'],expected_r,'Seeded energy rotations')
                ortho=max(float(np.max(np.abs(b.T@b-np.eye(width)))) for b in [*expected_q,*expected_r])
                errors={'seed':seed,'arm':arm,'orthogonality':ortho,'energy':0.,'identity':0.,'full_rank_endpoint':0.}
                if f['logits'].shape!=(len(cfg['ranks']),len(conditions),len(y),cfg['classes']): raise ValueError('Logit array shape mismatch')
                losses=[]
                for ki,k in enumerate(cfg['ranks']):
                    top=(z@v[:,:k])@v[:,:k].T; target_norm=np.linalg.norm(h-(h-top),axis=1); cell_losses=[]
                    for ci,name in enumerate(conditions):
                        if ci==0: delta=top
                        elif ci<=n_draw:
                            q=expected_q[ci-1,:,:k]; delta=(z@q)@q.T
                        else: delta=top@expected_r[ci-n_draw-1]
                        edited=h-delta; expected_logits=edited@model['W2']+model['b2']; saved=f['logits'][ki,ci]
                        error=float(np.max(np.abs(expected_logits-saved))); max_error=max(max_error,error)
                        close(saved,expected_logits,'Intervention logits',cfg['numerical_atol'])
                        norm=np.linalg.norm(h-edited,axis=1); negative=(edited<0).mean(axis=1)
                        close(norm,f['perturbation_norm'][ki,ci],'Perturbation norm'); close(negative,f['negative_fraction'][ki,ci],'Negative activation')
                        if ci>n_draw: errors['energy']=max(errors['energy'],float(np.max(np.abs(norm-target_norm))))
                        if k==0: errors['identity']=max(errors['identity'],float(np.max(np.abs(saved-baseline))))
                        if k==width and ci<=n_draw: errors['full_rank_endpoint']=max(errors['full_rank_endpoint'],float(np.max(np.abs(edited-mu))))
                        m=rescore(saved,y); cell_losses.append(m['ce']); row=curve_index[(seed,arm,k,name)]
                        for key,want in [('accuracy',m['correct'].mean()),('ce',m['ce'].mean()),('margin',m['margin'].mean()),('norm',norm.mean()),('negative_fraction',negative.mean())]: close(row[key],want,'Curve '+key)
                    losses.append(cell_losses)
                for key in ('orthogonality','energy','identity','full_rank_endpoint'):
                    if errors[key]>cfg['numerical_atol']: raise ValueError('Independent control failure: '+key)
                observed_integrity.append(errors)
                losses=np.array(losses); ix=[cfg['ranks'].index(k) for k in cfg['primary_ranks']]
                effect=(losses[ix,0]-losses[ix,n_draw+1:].mean(axis=1)).mean(axis=0)
                close(effect,f['primary_effect'],'Per-case primary effect'); effects[arm].append(effect)
    if len(summary['integrity'])!=len(observed_integrity): raise ValueError('Integrity coverage mismatch')
    for observed,reported in zip(observed_integrity,summary['integrity']):
        if reported['seed']!=observed['seed'] or reported['arm']!=observed['arm']: raise ValueError('Integrity identity mismatch')
        for key in ('orthogonality','energy','identity','full_rank_endpoint'): close(reported[key],observed[key],'Integrity '+key)
    a=np.array(effects['A']); b=np.array(effects['B'])
    with np.load(root/'effects.npz',allow_pickle=False) as f:
        close(f['A'],a,'Saved A effects'); close(f['B'],b,'Saved B effects')
    primary=bootstrap_statistics(a,cfg); check_statistics(summary['primary'],primary,'Primary')
    positive=int(np.sum(np.mean(a,axis=1)>0)); lower,upper=primary['ci95']
    decision=('supports_conditional_excess' if lower>0 and positive>=int(np.ceil(.8*len(a))) else 'against_predicted_excess' if upper<=0 else 'inconclusive')
    if summary['primary']['positive_seeds']!=positive or summary['primary']['interpretation']!=decision: raise ValueError('Primary sign/decision mismatch')
    secondary=bootstrap_statistics(a-b,cfg); check_statistics(summary['secondary'],secondary,'Secondary')
    eligible=train_audit['gates']['comparison_pass']
    interpretation='descriptive secondary only' if eligible else 'competence mismatch; no learning-specific conclusion'
    if summary['secondary']['eligible'] is not eligible or summary['secondary']['interpretation']!=interpretation: raise ValueError('Secondary eligibility/decision mismatch')
    return {'accepted':True,'reconstructed_cells':len(expected_cells),'max_logit_error':max_error,
            'primary':primary,'primary_decision':decision,'secondary':secondary,'comparison_eligible':eligible,
            'dataset_sha256':data_hash,'training_manifest_sha256':sha256(training/'MANIFEST.json'),
            'limit':'Independent rescore/bootstrap and seeded provenance checks, not independent retraining; conditional reused-benchmark evidence'}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('directory',type=Path); ap.add_argument('--training',type=Path); args=ap.parse_args()
    stage=read(args.directory/'metadata.json')['stage']
    result=audit_training(args.directory) if stage=='training' else audit_test(args.directory,args.training)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
