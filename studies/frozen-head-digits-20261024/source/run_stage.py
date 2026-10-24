"""Independent wall watchdog. No stage is accepted without WATCHDOG.json success."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def supervise(command,limit,env=None):
    started=time.monotonic(); proc=subprocess.Popen(command,env=env,start_new_session=True)
    status='completed'
    try: code=proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        status='timeout'; os.killpg(proc.pid,signal.SIGTERM)
        try: code=proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); code=proc.wait()
    elapsed=time.monotonic()-started
    if code!=0 and status=='completed': status='failed'
    if elapsed>limit: status='timeout'
    return {'status':status,'returncode':code,'elapsed_seconds':elapsed,'limit_seconds':limit}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=('training','heldout'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--clearance',type=Path,required=True)
    ap.add_argument('--training',type=Path)
    args=ap.parse_args()
    if os.path.lexists(args.out): raise FileExistsError(str(args.out))
    limit=json.loads((ROOT/'config.json').read_text())['wall_seconds']
    env=dict(os.environ,FROZEN_HEAD_WATCHDOG='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=[sys.executable,str(ROOT/'study.py'),'--stage',args.stage,'--out',str(args.out.resolve()),'--clearance',str(args.clearance.resolve())]
    if args.training: command+=['--training',str(args.training.resolve())]
    report=supervise(command,limit,env)
    if args.out.is_dir():
        with open(args.out/'WATCHDOG.json','x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(report)); return 0 if report['status']=='completed' and report['returncode']==0 else 1

if __name__=='__main__': sys.exit(main())
