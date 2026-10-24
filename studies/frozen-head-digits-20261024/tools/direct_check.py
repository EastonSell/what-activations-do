"""Read-only audit statistics from sealed archives; does not train or run producer."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--training',type=Path,required=True)
parser.add_argument('--heldout',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
source=args.source;train=args.training;test=args.heldout
if args.out.exists() or args.out.is_symlink():raise FileExistsError(str(args.out))
cfg=json.loads((source/'config.json').read_text());summary=json.loads((test/'summary.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(test/'MANIFEST.json')=='0f4096e4ce89c6190f21a2c67ddb98f0233e0f735a7d9adbd820eec211fcc8f8'
assert sha(train/'MANIFEST.json')=='eee5fe23598de00c5960e2cae9557b7655401eeb0411a6553e3d80e96b771399'
assert sha(source/'SOURCE_MANIFEST.json')=='10a4ea2c6083e5f248cd4cd211b83ec6530f71e9e11591df9ee2f1d4e7ebf117'
manifest=json.loads((test/'MANIFEST.json').read_text())
for name,entry in manifest['files'].items():
 p=test/name;assert p.is_file() and not p.is_symlink();assert p.stat().st_size==entry['bytes'];assert sha(p)==entry['sha256']
with np.load(test/'test-identity.npz',allow_pickle=False) as f:
 te=f['indices'];yy=f['labels'];xx=f['test_x']
with np.load(train/'splits.npz',allow_pickle=False) as f:
 tr=f['train'];va=f['validation'];assert np.array_equal(te,f['test'])
 fullx=np.empty((1797,64));fully=np.empty(1797,dtype=np.int64)
 fullx[tr]=f['train_x'];fully[tr]=f['train_y'];fullx[va]=f['validation_x'];fully[va]=f['validation_y'];fullx[te]=xx;fully[te]=yy
assert hashlib.sha256((fullx*16).tobytes()+fully.tobytes()).hexdigest()==cfg['data_sha256']
rtr,rest=train_test_split(np.arange(1797),test_size=.4,stratify=fully,random_state=20261024)
rva,rte=train_test_split(rest,test_size=.5,stratify=fully[rest],random_state=20261025)
assert np.array_equal(tr,rtr) and np.array_equal(va,rva) and np.array_equal(te,rte)
rows=[];effects={'A':[],'B':[]};untrained=[]
for seed in cfg['seeds']:
 for arm in ['A','B']:
  with np.load(test/f'test-{seed}-{arm}.npz',allow_pickle=False) as f:
   logits=f['logits'];baseline=f['baseline_logits'];negative=f['negative_fraction'];norms=f['perturbation_norm'];untrained_logits=f['untrained_logits']
   assert logits.shape==(33,33,360,10) and logits.dtype==np.float64 and np.isfinite(logits).all()
   assert negative.shape==norms.shape==(33,33,360)
   assert np.isfinite(negative).all() and np.isfinite(norms).all()
   assert np.all((negative>=0)&(negative<=1))
   assert np.array_equal(negative*32,np.round(negative*32))
   # Independent stable cross entropy with a centering implementation separate from frozen auditor.
   centered=logits-logits.max(axis=-1,keepdims=True)
   true=np.take_along_axis(centered,yy[None,None,:,None],axis=-1)[...,0]
   losses=np.log(np.exp(centered).sum(axis=-1))-true
   effect=np.mean(losses[1:9,0,:]-np.mean(losses[1:9,17:,:],axis=1),axis=0)
   assert np.allclose(effect,f['primary_effect'],atol=1e-10,rtol=0)
   effects[arm].append(effect)
   row={'seed':seed,'arm':arm,'clean_accuracy':float(np.mean(np.argmax(baseline,axis=-1)==yy)),'clean_ce':float(np.mean(np.logaddexp.reduce(baseline,axis=1)-baseline[np.arange(360),yy])),'untrained_accuracy':float(np.mean(np.argmax(untrained_logits,axis=-1)==yy))}
   for name,select in [('pca',slice(0,1)),('rank',slice(1,17)),('energy',slice(17,33))]:
    chosen_logits=logits[1:9,select,:,:];neg=negative[1:9,select,:]
    row[name]={'primary_rank_mean_ce':float(np.mean(losses[1:9,select,:])),'primary_rank_mean_accuracy':float(np.mean(np.argmax(chosen_logits,axis=-1)==yy)),'negative_hidden_coordinate_fraction':float(np.mean(neg)),'intervention_case_fraction_with_any_negative':float(np.mean(neg>0)),'mean_perturbation_norm':float(np.mean(norms[1:9,select,:]))}
   rows.append(row)
   assert np.max(np.abs(norms[1:9,17:,:]-norms[1:9,0,None,:]))<1e-10
with np.load(test/'effects.npz',allow_pickle=False) as f:
 for arm in effects:assert np.allclose(effects[arm],f[arm],atol=1e-10,rtol=0)
def stats(values):
 values=np.array(values);bycase=values.mean(axis=0);gen=np.random.default_rng(20261026)
 ids=gen.integers(0,360,size=(10000,360));draws=bycase[ids].mean(axis=1)
 return {'mean':float(values.mean()),'per_seed':values.mean(axis=1).tolist(),'ci95':np.quantile(draws,[.025,.975]).tolist(),'positive_seeds':int(np.sum(values.mean(axis=1)>0))}
a=np.array(effects['A']);b=np.array(effects['B']);primary=stats(a);secondary=stats(a-b)
for label,value in [('primary',primary),('secondary',secondary)]:
 for key in ['mean','per_seed','ci95']:assert np.allclose(value[key],summary[label][key],atol=1e-10,rtol=0)
assert primary['positive_seeds']==5 and primary['ci95'][0]>0
assert summary['primary']['interpretation']=='supports_conditional_excess' and summary['secondary']['eligible'] is False
aggregates={}
for arm in ['A','B']:
 selected=[r for r in rows if r['arm']==arm]
 aggregates[arm]={'clean_mean_accuracy':float(np.mean([r['clean_accuracy'] for r in selected])),'clean_accuracy_range':[min(r['clean_accuracy'] for r in selected),max(r['clean_accuracy'] for r in selected)],'clean_mean_ce':float(np.mean([r['clean_ce'] for r in selected]))}
 for kind in ['pca','rank','energy']:
  aggregates[arm][kind]={key:float(np.mean([r[kind][key] for r in selected])) for key in selected[0][kind]}
result={'accepted':True,'heldout_manifest_sha256':sha(test/'MANIFEST.json'),'source_manifest_sha256':sha(source/'SOURCE_MANIFEST.json'),'training_manifest_sha256':sha(train/'MANIFEST.json'),'verified_heldout_files':len(manifest['files']),'complete_dataset_sha256':cfg['data_sha256'],'exact_seeded_split_regeneration':'PASS','primary':primary,'secondary_arithmetic_check':secondary,'secondary_eligible':False,'primary_rank_aggregates':aggregates,'rows':rows,'scope':'Read-only independently rescored stored outputs; no producer run, retraining, or outcome-driven adjustment.'}
with args.out.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
