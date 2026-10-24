"""Real-data worker. Requires matching review clearance and independent watchdog."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='2'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import core
import pipeline
from storage import Budget,check_clearance,fresh_directory,manifest,sha256,verify_manifest,write_json

ROOT=Path(__file__).resolve().parent

def load_data(cfg):
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    if np.__version__!='2.3.5' or sklearn.__version__!='1.8.0':
        raise ValueError('Unreviewed NumPy/sklearn runtime version')
    data=load_digits()
    digest=hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest()
    if digest!=cfg['data_sha256']: raise ValueError('Dataset hash mismatch')
    indices=np.arange(len(data.target))
    tr,rest=train_test_split(indices,test_size=.4,stratify=data.target,random_state=cfg['split_seed'])
    va,te=train_test_split(rest,test_size=.5,stratify=data.target[rest],random_state=cfg['split_seed']+1)
    splits={'train':tr,'validation':va,'test':te}; pipeline.validate_splits(splits,len(indices))
    return data.data.astype(np.float64)/16.,data.target.astype(np.int64),splits

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('training','heldout'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--clearance',type=Path,required=True)
    ap.add_argument('--training',type=Path)
    args=ap.parse_args()
    if sys.platform!='linux': raise ValueError('Only reviewed Linux resource accounting is supported')
    if os.environ.get('FROZEN_HEAD_WATCHDOG')!='1': raise ValueError('Run through run_stage.py, not directly')
    source=ROOT/'SOURCE_MANIFEST.json'
    verify_manifest(ROOT,json.loads(source.read_text())); source_hash=sha256(source)
    cfg=json.loads((ROOT/'config.json').read_text())
    training_hash=None
    if args.stage=='heldout':
        if args.training is None: raise ValueError('Held-out stage requires training directory')
        from audit import audit_training
        audit_training(args.training)
        training_hash=sha256(args.training/'MANIFEST.json')
        metadata=json.loads((args.training/'metadata.json').read_text())
        if metadata['source_manifest_sha256']!=source_hash: raise ValueError('Training source freeze differs')
        if not json.loads((args.training/'summary.json').read_text())['gates']['primary_pass']:
            raise ValueError('Primary competence gate failed')
    clearance=check_clearance(args.clearance,args.stage,source_hash,training_hash)
    out=fresh_directory(args.out)
    budget=Budget(out,cfg['wall_seconds'],cfg['rss_bytes'],cfg['disk_bytes']-65536)
    try:
        from threadpoolctl import threadpool_info,threadpool_limits
        with threadpool_limits(limits=cfg['threads']):
            x,y,splits=load_data(cfg); budget.check()
            if args.stage=='heldout':
                with np.load(args.training/'splits.npz',allow_pickle=False) as f:
                    for key in splits:
                        if not np.array_equal(splits[key],f[key]): raise ValueError('Split identity differs from training')
            write_json(out/'metadata.json',{'stage':args.stage,'source_manifest_sha256':source_hash,
                       'training_manifest_sha256':training_hash,'config':cfg,'clearance':clearance,
                       'python':platform.python_version(),'numpy':np.__version__,'sklearn':'1.8.0',
                       'threadpools':threadpool_info(),'dataset_sha256':cfg['data_sha256']})
            if args.stage=='training': pipeline.train_stage(x,y,splits,cfg,out,budget)
            else: pipeline.test_stage(x,y,splits,cfg,args.training,out,budget)
            write_json(out/'resources.json',budget.check())
            write_json(out/'MANIFEST.json',manifest(out,[p.name for p in out.iterdir() if p.is_file()]))
            budget.check()
            write_json(out/'COMPLETE.json',{'manifest_sha256':sha256(out/'MANIFEST.json')})
            budget.check()
    except BaseException as exc:
        if not (out/'FAILED.json').exists():
            write_json(out/'FAILED.json',{'type':type(exc).__name__,'reason':str(exc),'resources':budget.measure()})
        raise

if __name__=='__main__': main()
