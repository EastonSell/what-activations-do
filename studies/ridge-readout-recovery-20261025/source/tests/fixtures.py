"""Invented one-hot arrays only. Never loads digits or historical archives."""
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import custody
import settings


def fixture_archives(root, bad_training_labels=False):
    root=Path(root); spec=settings.production_spec(); spec['kind']='synthetic_fixture_not_scientific'
    cfg=spec['config']; n=cfg['train_cases']+cfg['validation_cases']+cfg['test_cases']
    y=np.arange(n,dtype=np.int64)%10; x=np.eye(64,dtype=np.float64)[y]
    tr=np.arange(1078,dtype=np.int64); va=np.arange(1078,1437,dtype=np.int64); te=np.arange(1437,n,dtype=np.int64)
    if bad_training_labels: y[tr]=(y[tr]+1)%10
    spec['data_sha256']=hashlib.sha256((x*16.).tobytes()+y.tobytes()).hexdigest()
    spec['historical_config']=dict(spec['historical_config'],data_sha256=spec['data_sha256'])
    source=root/'historical_train_fixture'; source.mkdir()
    custody.write_npz(source/'splits.npz',train=tr,validation=va,test=te,
                     train_x=x[tr],train_y=y[tr],validation_x=x[va],validation_y=y[va])
    rows=[]
    for seed in cfg['seeds']:
        W1=np.eye(64,32,dtype=np.float64)*(1+seed/10000); b1=np.zeros(32)
        W2=np.zeros((32,10)); W2[:10]=5*np.eye(10); b2=np.zeros(10)
        th=np.maximum(0,x[tr]@W1+b1); vh=np.maximum(0,x[va]@W1+b1)
        mean=th.mean(0); _,singular,vt=np.linalg.svd(th-mean,full_matrices=False)
        for arm in ['A','B']:
            custody.write_npz(source/f'model-{seed}-{arm}.npz',W1=W1,b1=b1,W2=W2,b2=b2,
              initial_W1=W1,initial_b1=b1,initial_W2=W2,initial_b2=b2,
              train_hidden=th,validation_hidden=vh,validation_logits=vh@W2+b2,
              untrained_validation_logits=vh@W2+b2,mean=mean,pcs=vt.T,singular_values=singular)
            custody.write_json(source/f'history-{seed}-{arm}.json',[])
            rows.append({'seed':seed,'arm':arm,'validation_accuracy':1.})
    metadata={'stage':'training','source_manifest_sha256':spec['historical_source_sha256'],
      'config':spec['historical_config'],'dataset_sha256':spec['data_sha256'],'training_manifest_sha256':None,
      'clearance':{'accepted':True,'stage':'training','source_manifest_sha256':spec['historical_source_sha256'],'review_reference':'synthetic only'}}
    custody.write_json(source/'metadata.json',metadata)
    custody.write_json(source/'summary.json',{'stage':'training','fit_count':10,'rows':rows,'gates':{'primary_pass':True,'comparison_pass':True}})
    custody.write_json(source/'resources.json',{'wall_seconds':1.,'rss_peak_bytes':1000000,'retained_bytes':10000})
    custody.write_json(source/'MANIFEST.json',custody.manifest(source,custody.files_under(source)))
    spec['training_manifest_sha256']=custody.sha256(source/'MANIFEST.json')
    custody.write_json(source/'COMPLETE.json',{'manifest_sha256':spec['training_manifest_sha256']})
    custody.write_json(source/'WATCHDOG.json',{'status':'completed','returncode':0,'elapsed_seconds':1.,'limit_seconds':600})
    ta=root/'synthetic-training.tar'
    with tarfile.open(ta,'x') as t: t.add(source,arcname='training')
    spec['training_archive_sha256']=custody.sha256(ta)
    held=root/'historical_test_fixture'; held.mkdir()
    custody.write_npz(held/'test-identity.npz',indices=te,labels=y[te],test_x=x[te])
    custody.write_json(held/'metadata.json',dict(metadata,stage='heldout',training_manifest_sha256=spec['training_manifest_sha256']))
    custody.write_json(held/'MANIFEST.json',custody.manifest(held,custody.files_under(held)))
    spec['heldout_manifest_sha256']=custody.sha256(held/'MANIFEST.json')
    custody.write_json(held/'COMPLETE.json',{'manifest_sha256':spec['heldout_manifest_sha256']})
    custody.write_json(held/'WATCHDOG.json',{'status':'completed','returncode':0,'elapsed_seconds':1.,'limit_seconds':600})
    ha=root/'synthetic-heldout.tar'
    with tarfile.open(ha,'x') as t: t.add(held,arcname='heldout')
    spec['heldout_archive_sha256']=custody.sha256(ha)
    return ta,ha,spec


def synthetic_context():
    return {'source_manifest_sha256':'a'*64,'protocol_manifest_sha256':settings.PROTOCOL_HASH,
            'clearance':{'review_reference':'synthetic test, no scientific acceptance'}}


def seal_fixture(out,stage,spec,context):
    """Manufactured synthetic receipt, never usable by the production CLI."""
    from stageio import finalize
    import time
    size=sum((out/n).stat().st_size for n in custody.files_under(out))
    custody.write_json(out/'WATCHDOG.json',{'status':'success','success':True,'returncode':0,'violations':[],'errors':[],
      'elapsed_seconds':.1,'peak_sampled_rss_bytes':1000000,'output_bytes':size,'peak_sampled_output_bytes':size,
      'limits':{'wall_seconds':600,'rss_bytes':1073741824,'output_bytes':134217728}})
    finalize(out,stage,spec,context['source_manifest_sha256'],time.monotonic())
    if stage=='training': authorize_heldout_fixture(out,spec,context)


def authorize_heldout_fixture(training,spec,context):
    """Bind synthetic call context exactly as a separately reviewed clearance would."""
    if spec.get('kind')!='synthetic_fixture_not_scientific':
        raise ValueError('Fixture authorization is only for invented synthetic arrays')
    context['clearance']={'accepted':True,'stage':'heldout',
      'source_manifest_sha256':context['source_manifest_sha256'],
      'protocol_manifest_sha256':spec['protocol_manifest_sha256'],
      'training_manifest_sha256':custody.sha256(Path(training)/'MANIFEST.json'),
      'review_reference':'Synthetic fixture only, no scientific acceptance'}
