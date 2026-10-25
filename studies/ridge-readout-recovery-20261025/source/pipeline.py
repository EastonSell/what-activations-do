"""Staged real-array computation. Importing performs no I/O or experiment."""
from pathlib import Path
import time
import numpy as np
import custody
from protocol_v2 import readout
from stageio import verify_completed


def close(a,b,label,atol):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=atol,rtol=0):
        raise ValueError(label+' differs')


def load_npz(path,keys=None):
    custody.record_access('read_npz_without_pickle',path)
    with np.load(path,allow_pickle=False) as f:
        if keys is not None and set(f.files)!=set(keys): raise ValueError('NPZ key coverage differs')
        result={k:f[k].copy() for k in f.files}
    for k,v in result.items():
        if v.dtype.kind in 'fiu' and not np.isfinite(v).all(): raise ValueError('Nonfinite archive array: '+k)
        if v.dtype.kind not in 'fiuUb': raise ValueError('Unreviewed array dtype: '+k)
    return result


def historical_training(root,spec):
    cfg=spec['config']; root=Path(root)
    m=custody.verify_manifest(root,'MANIFEST.json',spec['training_manifest_sha256'],
      extra_allowed={'COMPLETE.json','WATCHDOG.json'})
    expected={'splits.npz','metadata.json','summary.json','resources.json'}
    for seed in cfg['seeds']:
        for arm in ['A','B']: expected|={f'model-{seed}-{arm}.npz',f'history-{seed}-{arm}.json'}
    if set(m['files'])!=expected: raise ValueError('Historical training coverage differs')
    if custody.read_json(root/'COMPLETE.json')['manifest_sha256']!=spec['training_manifest_sha256']:
        raise ValueError('Historical completion hash differs')
    watchdog=custody.read_json(root/'WATCHDOG.json')
    if watchdog['status']!='completed' or watchdog['returncode']!=0 or watchdog['limit_seconds']!=600 or not 0<=watchdog['elapsed_seconds']<=600:
        raise ValueError('Historical watchdog failed')
    meta=custody.read_json(root/'metadata.json')
    if (meta['stage']!='training' or meta['source_manifest_sha256']!=spec['historical_source_sha256'] or
        meta['config']!=spec['historical_config'] or meta['dataset_sha256']!=spec['data_sha256']):
        raise ValueError('Historical source/config identity differs')
    historical_summary=custody.read_json(root/'summary.json')
    if historical_summary['gates']['primary_pass'] is not True: raise ValueError('Historical primary competence failed')
    data=load_npz(root/'splits.npz',{'train','validation','test','train_x','train_y','validation_x','validation_y'})
    all_ids=np.concatenate([data['train'],data['validation'],data['test']])
    n=cfg['train_cases']+cfg['validation_cases']+cfg['test_cases']
    if all_ids.dtype.kind not in 'iu' or sorted(all_ids.tolist())!=list(range(n)):
        raise ValueError('Historical split does not partition cases')
    for split,count in [('train',cfg['train_cases']),('validation',cfg['validation_cases']),('test',cfg['test_cases'])]:
        if data[split].shape!=(count,): raise ValueError('Historical split count differs')
        if split!='test':
            if data[split+'_x'].shape!=(count,64) or data[split+'_x'].dtype!=np.float64:
                raise ValueError('Historical input shape/dtype differs')
            readout.labels(data[split+'_y'],count,10)
    models={}; original=[]
    for seed in cfg['seeds']:
        model=load_npz(root/f'model-{seed}-A.npz')
        for key,shape in [('W1',(64,32)),('b1',(32,)),('W2',(32,10)),('b2',(10,)),('mean',(32,)),('pcs',(32,32)),('singular_values',(32,))]:
            if model[key].shape!=shape or model[key].dtype!=np.float64: raise ValueError('Model shape/dtype differs: '+key)
        with np.errstate(over='raise',invalid='raise'):
            th=np.maximum(0,data['train_x']@model['W1']+model['b1'])
            vh=np.maximum(0,data['validation_x']@model['W1']+model['b1'])
            vl=vh@model['W2']+model['b2']
        tol=spec['numeric']['geometry_atol']
        close(th,model['train_hidden'],'Training hidden state',tol)
        close(vh,model['validation_hidden'],'Validation hidden state',tol)
        close(vl,model['validation_logits'],'Historical validation logits',spec['numeric']['score_atol'])
        close(th.mean(0),model['mean'],'Training mean',tol)
        V=model['pcs']; close(V.T@V,np.eye(32),'PCA orthogonality',tol)
        centered=th-model['mean']; covariance=centered.T@centered
        if not np.allclose(V.T@covariance@V,np.diag(model['singular_values']**2),
          atol=spec['numeric']['eigensystem_atol'],rtol=spec['numeric']['eigensystem_rtol']): raise ValueError('PCA eigensystem differs')
        if np.any(np.diff(model['singular_values'])>tol): raise ValueError('PCA ordering differs')
        _,newV=readout.geometry(th)
        close(V[:,:8]@V[:,:8].T,newV[:,:8]@newV[:,:8].T,'Recomputed top-eight projector',spec['numeric']['projector_atol'])
        accuracy=float((vl.argmax(1)==data['validation_y']).mean())
        rows=[r for r in historical_summary['rows'] if r['seed']==seed and r['arm']=='A']
        if len(rows)!=1 or rows[0]['validation_accuracy']!=accuracy: raise ValueError('Historical validation metric differs')
        original.append(accuracy); models[seed]=model
    return data,models,original


def conditions(seed,model):
    return [('untouched',model['pcs'],0),('pca8',model['pcs'],8)]+[(f'random8-{d:02}',readout.random_basis(seed,d,32),8) for d in range(16)]


def support(x,mu):
    value=readout.support_diagnostics(x,mu); restored=x+mu
    for prefix,threshold in [('strict',0.),('material',-1e-10)]:
        mask=restored<threshold
        value[prefix+'_negative_coordinate_count']=int(mask.sum())
        value[prefix+'_any_negative_case_count']=int(mask.any(axis=1).sum())
    return value


def score_row(seed,condition,split,scores,y,x,mu):
    if scores.shape!=(len(y),10) or not np.isfinite(scores).all(): raise ValueError('Invalid scores before argmax')
    pred=scores.argmax(1); confusion=np.zeros((10,10),dtype=np.int64)
    np.add.at(confusion,(y,pred),1)
    return {'seed':seed,'condition':condition,'split':split,'correct_count':int((pred==y).sum()),
      'cases':len(y),'accuracy':float((pred==y).mean()),'confusion_matrix':confusion.tolist(),
      'support':support(x,mu)}


def write_metadata(out,stage,spec,context,extra=None):
    custody.write_json(out/'metadata.json',dict(context,stage=stage,spec=spec,
      protocol_manifest_sha256=spec['protocol_manifest_sha256'],**(extra or {})))


def train_stage(archive,out,spec,context):
    out=Path(out); cfg=spec['config']; started=time.monotonic()
    receipt=custody.import_archive(archive,out/'historical_training',spec['training_archive_sha256'],
      spec['training_manifest_sha256'],None,cfg['max_output_bytes_per_stage']//2)
    data,models,original=historical_training(out/'historical_training',spec)
    write_metadata(out,'training',spec,context,{'historical_import':receipt})
    rows=[]; fits=[]; clean=[]
    for seed in cfg['seeds']:
        model=models[seed]; weights=[]; biases=[]; ts=[]; vs=[]; names=[]
        for name,basis,k in conditions(seed,model):
            before=time.monotonic(); names.append(name)
            tx=readout.project(model['train_hidden'],model['mean'],basis,k)
            vx=readout.project(model['validation_hidden'],model['mean'],basis,k)
            fit=readout.ridge(tx,data['train_y'],10,cfg['ridge_lambda'])
            if (fit['stationarity_max_abs']>cfg['normal_equation_gradient_atol'] or
                fit['normal_equation_relative_max_abs']>cfg['normal_equation_relative_max_abs_tolerance'] or
                fit['gram_condition_number']>cfg['gram_condition_number_max']):
                raise ValueError('Readout numerical validity gate failed')
            train_scores=readout.predict_scores(fit,tx); val_scores=readout.predict_scores(fit,vx)
            weights.append(fit['weights']); biases.append(fit['intercept']); ts.append(train_scores); vs.append(val_scores)
            rows.extend([score_row(seed,name,'train',train_scores,data['train_y'],tx,model['mean']),
                         score_row(seed,name,'validation',val_scores,data['validation_y'],vx,model['mean'])])
            singular=np.linalg.svd(tx-tx.mean(0),compute_uv=False)
            rank_tol=max(tx.shape)*np.finfo(np.float64).eps*singular[0]
            centered=model['train_hidden']-model['mean']; removed=centered-tx
            fits.append({'seed':seed,'condition':name,'elapsed_seconds':time.monotonic()-before,
              **{key:fit[key] for key in ['stationarity_max_abs','normal_equation_relative_max_abs','gram_condition_number','objective']},
              'actual_train_rank':int(np.sum(singular>rank_tol)),'rank_threshold':float(rank_tol),
              'maximum_feature_subspace_dimension':32-k,'removed_rank':k,
              'mean_removed_squared_norm':float(np.mean(np.sum(removed**2,axis=1))),
              'removed_variance_fraction':float(np.sum(removed**2)/max(np.sum(centered**2),np.finfo(float).tiny))})
            if name=='untouched': clean.append(rows[-1]['accuracy'])
        custody.write_npz(out/f'fitted-{seed}.npz',weights=np.array(weights),intercepts=np.array(biases),
          condition_names=np.array(names),train_scores=np.array(ts),validation_scores=np.array(vs))
    gates=readout.eligibility(original,clean)
    summary={'stage':'training','readout_count':len(fits),'rows':rows,'fit_rows':fits,'gates':gates,
      'original_validation_accuracy':original,'untouched_refit_validation_accuracy':clean,
      'elapsed_compute_seconds':time.monotonic()-started,'status':'completed' if gates['eligible'] else 'gate_failed'}
    custody.write_json(out/'summary.json',summary)
    custody.write_json(out/'WORKER_RESULT.json',{'stage':'training','status':summary['status'],'readout_count':len(fits)})
    return summary


def heldout_stage(archive,training,out,spec,context):
    training=Path(training); out=Path(out); cfg=spec['config']
    training_hash=context.get('clearance',{}).get('training_manifest_sha256')
    if not isinstance(training_hash,str) or len(training_hash)!=64 or any(c not in '0123456789abcdef' for c in training_hash):
        raise ValueError('Held-out clearance must bind an exact training manifest')
    verify_completed(training,'training',spec,context['source_manifest_sha256'],training_hash)
    data,models,_=historical_training(training/'historical_training',spec)
    # Load a bounded in-memory training snapshot before any held-out access, then
    # verify all its source files against the independently cleared manifest.
    all_heads={seed:load_npz(training/f'fitted-{seed}.npz') for seed in cfg['seeds']}
    verify_completed(training,'training',spec,context['source_manifest_sha256'],training_hash)
    receipt=custody.import_archive(archive,out/'historical_heldout',spec['heldout_archive_sha256'],
      spec['heldout_manifest_sha256'],{'test-identity.npz','metadata.json','COMPLETE.json','WATCHDOG.json'},cfg['max_output_bytes_per_stage']//2)
    history=custody.read_json(out/'historical_heldout/metadata.json')
    if (history['stage']!='heldout' or history['source_manifest_sha256']!=spec['historical_source_sha256'] or
        history['training_manifest_sha256']!=spec['training_manifest_sha256'] or history['config']!=spec['historical_config']):
        raise ValueError('Historical held-out metadata binding differs')
    if custody.read_json(out/'historical_heldout/COMPLETE.json')['manifest_sha256']!=spec['heldout_manifest_sha256']:
        raise ValueError('Historical held-out completion identity differs')
    identity=load_npz(out/'historical_heldout/test-identity.npz',{'indices','labels','test_x'})
    if not np.array_equal(identity['indices'],data['test']) or identity['test_x'].shape!=(360,64) or identity['test_x'].dtype!=np.float64:
        raise ValueError('Held-out case identity differs')
    y=readout.labels(identity['labels'],360,10)
    all_x=np.empty((1797,64),dtype=np.float64); all_y=np.empty(1797,dtype=np.int64)
    for split in ['train','validation']:
        all_x[data[split]]=data[split+'_x']; all_y[data[split]]=data[split+'_y']
    all_x[data['test']]=identity['test_x']; all_y[data['test']]=y
    if custody.digest_bytes((all_x*16.).tobytes()+all_y.tobytes())!=spec['data_sha256']:
        raise ValueError('Full reconstructed data identity differs')
    write_metadata(out,'heldout',spec,context,{'training_manifest_sha256':training_hash,'historical_import':receipt})
    rows=[]; primary_a=[]; primary_b=[]; gains=[]; random_contrasts=[]
    for seed in cfg['seeds']:
        model=models[seed]; heads=all_heads[seed]
        expected_names=[c[0] for c in conditions(seed,model)]
        if heads['condition_names'].tolist()!=expected_names or heads['weights'].shape!=(18,32,10) or heads['intercepts'].shape!=(18,10):
            raise ValueError('Refitted readout coverage differs')
        h=np.maximum(0,identity['test_x']@model['W1']+model['b1']); scores=[]; correctness=[]
        for ci,(name,basis,k) in enumerate(conditions(seed,model)):
            x=readout.project(h,model['mean'],basis,k)
            fit={'weights':heads['weights'][ci],'intercept':heads['intercepts'][ci]}
            score=readout.predict_scores(fit,x); scores.append(score); correctness.append(score.argmax(1)==y)
            rows.append(score_row(seed,name,'test',score,y,x,model['mean']))
        frozen=[]
        for name,k in [('original-clean',0),('original-pca8',8)]:
            x=readout.project(h,model['mean'],model['pcs'],k)
            score=readout.predict_scores({'weights':model['W2'],'intercept':model['b2']},x+model['mean'])
            frozen.append(score); rows.append(score_row(seed,name,'test',score,y,x,model['mean']))
        primary_a.append(correctness[1]); primary_b.append(correctness[0])
        gains.append(float(np.mean(correctness[1])-np.mean(frozen[1].argmax(1)==y)))
        random_contrasts.append(float(np.mean(correctness[1])-np.mean(correctness[2:])))
        custody.write_npz(out/f'test-{seed}.npz',scores=np.array(scores),original_scores=np.array(frozen),
          correctness=np.array(correctness),case_indices=identity['indices'],labels=y)
    primary=readout.primary(np.array(primary_a),np.array(primary_b))
    verify_completed(training,'training',spec,context['source_manifest_sha256'],training_hash)
    summary={'stage':'heldout','cases':360,'rows':rows,'primary':primary,
      'pca_refit_minus_original_pca_accuracy_pp_per_seed':(100*np.array(gains)).tolist(),
      'pca_refit_minus_mean_random_refit_accuracy_pp_per_seed':(100*np.array(random_contrasts)).tolist(),
      'status':'completed_pending_independent_audit','scientific_acceptance':False}
    custody.write_json(out/'summary.json',summary)
    custody.write_json(out/'WORKER_RESULT.json',{'stage':'heldout','status':'completed','test_conditions':len(rows)})
    return summary
