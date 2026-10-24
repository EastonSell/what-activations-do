"""Exclusive artifact writes, explicit review guards, and bounded local accounting."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def fresh_directory(path):
    path=Path(path)
    if os.path.lexists(path): raise FileExistsError(str(path))
    path.mkdir(parents=True,exist_ok=False)
    return path

def write_json(path,value):
    with open(path,'x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')

def write_npz(path,**arrays):
    with open(path,'xb') as f: np.savez_compressed(f,**arrays)

def safe_file(root,relative):
    relative=Path(relative); root=Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe manifest path')
    path=root/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Manifest target must be a regular local file')
    return path

def manifest(root,names):
    return {'schema':1,'files':{name:{'sha256':sha256(safe_file(root,name)),
                                    'bytes':safe_file(root,name).stat().st_size}
                              for name in sorted(names)}}

def verify_manifest(root,value):
    if value.get('schema')!=1 or not value.get('files'): raise ValueError('Invalid manifest')
    for name,entry in value['files'].items():
        path=safe_file(root,name)
        if path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
            raise ValueError('Hash or size mismatch: '+name)

def check_clearance(path,stage,source_hash,training_hash=None):
    obj=json.loads(Path(path).read_text())
    if obj.get('accepted') is not True or obj.get('stage')!=stage or obj.get('source_manifest_sha256')!=source_hash or not obj.get('review_reference'):
        raise ValueError('Missing, mismatched, or unaccepted stage clearance')
    if stage=='heldout' and obj.get('training_manifest_sha256')!=training_hash:
        raise ValueError('Held-out clearance does not match training artifact')
    return obj

class Budget:
    def __init__(self,root,wall_seconds,rss_bytes,disk_bytes):
        self.root=Path(root); self.start=time.monotonic()
        self.wall=wall_seconds; self.rss=rss_bytes; self.disk=disk_bytes
    def measure(self):
        # Linux ru_maxrss is KiB. This runner targets Linux only.
        return {'wall_seconds':time.monotonic()-self.start,
                'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'retained_bytes':sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()),
                'accounting':'single process including threads; not process-tree/kernel containment'}
    def check(self):
        m=self.measure()
        if m['wall_seconds']>self.wall or m['rss_peak_bytes']>self.rss or m['retained_bytes']>self.disk:
            raise RuntimeError('Resource budget exceeded: '+json.dumps(m))
        return m
