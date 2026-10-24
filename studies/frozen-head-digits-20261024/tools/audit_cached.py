"""Read-only decompression cache around unchanged, separately frozen audit code."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

class CachedNpz:
    def __init__(self,wrapped):
        self.wrapped=wrapped; self.cache={}; self.files=wrapped.files
    def __getitem__(self,key):
        if key not in self.cache:
            value=self.wrapped[key]
            value.setflags(write=False)
            self.cache[key]=value
        return self.cache[key]
    def __enter__(self): return self
    def __exit__(self,*args):
        self.wrapped.close(); self.cache.clear()

@contextmanager
def cached_load():
    original=np.load
    def load(*args,**kwargs):
        value=original(*args,**kwargs)
        return CachedNpz(value) if hasattr(value,'files') else value
    np.load=load
    try: yield
    finally: np.load=original

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--training',type=Path,required=True)
    ap.add_argument('--heldout',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError(str(args.out))
    sys.path.insert(0,str(args.source.resolve())); auditor=importlib.import_module('audit')
    started=time.monotonic()
    with cached_load(): result=auditor.audit_test(args.heldout,args.training,args.source)
    result['audit_execution']={'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               'wall_seconds':time.monotonic()-started,
                               'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                               'scope':'Read-only decompression caching; frozen audit logic unchanged'}
    with args.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'accepted':result['accepted'],'output':str(args.out),'audit_execution':result['audit_execution']}))

if __name__=='__main__': main()
