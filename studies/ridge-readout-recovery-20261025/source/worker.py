"""Production worker: reviewed source and separate clearance precede array access."""
import os
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='2'
import argparse
from pathlib import Path
import platform
import sys
import numpy as np
import custody
import pipeline
import settings


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['training','heldout'],required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--clearance',type=Path,required=True)
    ap.add_argument('--training',type=Path)
    args=ap.parse_args(); out=args.out
    if os.environ.get('READOUT_STAGE_WATCHDOG')!='1':
        raise ValueError('Use reviewed independent runner, not worker directly')
    if sys.platform!='linux' or np.__version__!='2.3.5': raise ValueError('Unreviewed Linux/NumPy runtime')
    if not out.is_dir() or set(custody.files_under(out))-{'stdout.log','stderr.log'}:
        raise ValueError('Worker output must be fresh except supervisor logs')
    custody.begin_access_log()
    try:
        source_hash=settings.verify_source(); spec=settings.production_spec()
        training_hash=None
        if args.stage=='heldout':
            if args.training is None: raise ValueError('Missing new sealed training artifact')
            training_hash=custody.sha256(args.training/'MANIFEST.json')
        elif args.training is not None: raise ValueError('Unexpected stage input')
        cleared=custody.clearance(args.clearance,args.stage,source_hash,settings.PROTOCOL_HASH,training_hash)
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
        if any(p.get('num_threads',0)>2 for p in pools): raise ValueError('Numerical thread ceiling exceeded')
        context={'source_manifest_sha256':source_hash,'protocol_manifest_sha256':settings.PROTOCOL_HASH,
          'clearance':cleared,'clearance_sha256':custody.sha256(args.clearance),
          'runtime':{'python':platform.python_version(),'numpy':np.__version__,'threadpools':pools},
          'scope':'Execution receipt only; independent scientific audit still required'}
        result=(pipeline.train_stage(args.archive,out,spec,context) if args.stage=='training' else
                pipeline.heldout_stage(args.archive,args.training,out,spec,context))
        if settings.verify_source()!=source_hash: raise ValueError('Source identity changed during worker execution')
        custody.write_json(out/'ACCESS_LOG.json',{'scope':'Explicit source/scientific file-operation calls, not OS tracing; supervisor sealing is recorded by final manifest',
          'operations':custody.access_log()})
        return 0 if args.stage=='heldout' or result['gates']['eligible'] else 2
    except BaseException as exc:
        if not (out/'FAILED_WORKER.json').exists():
            custody.write_json(out/'FAILED_WORKER.json',{'type':type(exc).__name__,'reason':str(exc)})
        if not (out/'ACCESS_LOG.json').exists():
            custody.write_json(out/'ACCESS_LOG.json',{'scope':'Partial explicit file-operation record, not OS tracing','operations':custody.access_log()})
        raise


if __name__=='__main__': sys.exit(main())
