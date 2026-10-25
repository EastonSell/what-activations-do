"""Independent stage supervisor. Production CLI has no study-parameter overrides."""
import argparse
import os
from pathlib import Path
import sys
import time
import custody
import settings
import stageio
from watchdog import supervise


def run_command(command,out,stage,spec,source_hash,source_recheck=None):
    """Generic supervisor also exercised by explicitly synthetic subprocess tests."""
    started=time.monotonic(); out=Path(out).resolve()
    if out.exists() or out.is_symlink(): raise FileExistsError(str(out))
    if not out.parent.is_dir(): raise ValueError('Output parent must already exist')
    out.mkdir(exist_ok=False)
    cfg=spec['config']
    limits={'wall_seconds':cfg['max_seconds_per_stage'],'rss_bytes':cfg['max_rss_bytes'],
            'output_bytes':cfg['max_output_bytes_per_stage'],'poll_seconds':.02}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=str(settings.ROOT),READOUT_STAGE_WATCHDOG='1')
    receipt=supervise(command,out,limits,env)
    if receipt['status']!='success':
        return {'status':'execution_failed','watchdog':receipt,'scientific_acceptance':False}
    try:
        if source_recheck is not None and source_recheck()!=source_hash:
            raise ValueError('Source changed during supervised stage')
        stageio.finalize(out,stage,spec,source_hash,started)
    except BaseException as exc:
        failure=out/'FAILED_FINALIZATION.json'
        if not failure.exists(): custody.write_json(failure,{'type':type(exc).__name__,'reason':str(exc)})
        raise
    return {'status':'execution_complete_pending_independent_audit',
            'stage':stage,'manifest_sha256':custody.sha256(out/'MANIFEST.json'),
            'scientific_acceptance':False,'watchdog':receipt}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['training','heldout'],required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--clearance',type=Path,required=True)
    ap.add_argument('--training',type=Path)
    args=ap.parse_args()
    if sys.platform!='linux': raise ValueError('Only reviewed Linux supervision is supported')
    source_hash=settings.verify_source(); spec=settings.production_spec()
    output=args.out.resolve()
    if output.is_relative_to(settings.ROOT) or output==args.archive.resolve():
        raise ValueError('Outputs must be outside reviewed source and input archives')
    training_hash=None
    if args.stage=='heldout':
        if args.training is None: raise ValueError('Held-out stage requires sealed new training artifact')
        if output.is_relative_to(args.training.resolve()): raise ValueError('Output cannot be inside training artifact')
        training_hash=custody.sha256(args.training/'MANIFEST.json')
    elif args.training is not None: raise ValueError('Training stage accepts no held-out/training-stage directory argument')
    custody.clearance(args.clearance,args.stage,source_hash,settings.PROTOCOL_HASH,training_hash)
    command=[sys.executable,'-B',str(settings.ROOT/'worker.py'),'--stage',args.stage,
      '--archive',str(args.archive.resolve()),'--out',str(output),'--clearance',str(args.clearance.resolve())]
    if args.training is not None: command+=['--training',str(args.training.resolve())]
    result=run_command(command,output,args.stage,spec,source_hash,settings.verify_source)
    import json
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['status']=='execution_complete_pending_independent_audit' else 1


if __name__=='__main__': sys.exit(main())
