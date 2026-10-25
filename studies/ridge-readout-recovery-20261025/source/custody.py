"""Exclusive writes and exact byte custody; no reads are performed at import."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

_ACCESS_LOG=None


def begin_access_log():
    global _ACCESS_LOG
    _ACCESS_LOG=[]


def record_access(operation,path):
    if _ACCESS_LOG is not None:
        _ACCESS_LOG.append({'operation':operation,'path':str(Path(path).resolve())})


def access_log(): return list(_ACCESS_LOG or [])


def digest_bytes(data): return hashlib.sha256(data).hexdigest()


def sha256(path):
    record_access('read_hash',path)
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def parse_json(raw):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(raw,parse_constant=reject)


def read_json(path):
    record_access('read_json',path)
    path=Path(path)
    if path.stat().st_size>4*1024*1024: raise ValueError('JSON exceeds bounded metadata size')
    return parse_json(path.read_text())


def write_json(path,value):
    encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    record_access('write_json_exclusive',path)
    with open(path,'x',encoding='utf-8') as f: f.write(encoded)


def write_npz(path,**arrays):
    import numpy as np
    record_access('write_npz_exclusive',path)
    with open(path,'xb') as f: np.savez_compressed(f,**arrays)


def safe_relative(name):
    if not isinstance(name,str) or '\\' in name: raise ValueError('Unsafe relative path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('Unsafe relative path')
    return p.as_posix()


def safe_file(root,name):
    root=Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('Root must be a real directory')
    name=safe_relative(name); path=root/name
    for node in [path,*path.parents]:
        if node==root.parent: break
        if node.is_symlink(): raise ValueError('Symlink in artifact path')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Missing or escaping regular file')
    return path


def files_under(root):
    root=Path(root)
    if root.is_symlink(): raise ValueError('Symlink root')
    result=[]
    for p in root.rglob('*'):
        if p.is_symlink(): raise ValueError('Symlink artifact')
        if p.is_file(): result.append(p.relative_to(root).as_posix())
    return sorted(result)


def manifest(root,names):
    entries={}
    for name in sorted(names):
        p=safe_file(root,name)
        entries[name]={'sha256':sha256(p),'bytes':p.stat().st_size}
    return {'schema':1,'files':entries}


def verify_manifest(root,manifest_name='MANIFEST.json',expected_hash=None,exact=True,extra_allowed=()):
    path=safe_file(root,manifest_name)
    if expected_hash is not None and sha256(path)!=expected_hash: raise ValueError('Manifest identity mismatch')
    value=read_json(path)
    if value.get('schema')!=1 or not value.get('files') or not isinstance(value['files'],dict):
        raise ValueError('Invalid manifest schema')
    for name,entry in value['files'].items():
        p=safe_file(root,name)
        if set(entry)!={'sha256','bytes'} or type(entry['bytes']) is not int or entry['bytes']<0:
            raise ValueError('Invalid manifest entry')
        if p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:
            raise ValueError('Artifact hash/size mismatch: '+name)
    if exact and set(files_under(root))!=set(value['files'])|{manifest_name}|set(extra_allowed):
        raise ValueError('Manifest file coverage mismatch')
    return value


def clearance(path,stage,source_hash,protocol_hash,training_hash=None):
    value=read_json(path)
    if (value.get('accepted') is not True or value.get('stage')!=stage or
        value.get('source_manifest_sha256')!=source_hash or
        value.get('protocol_manifest_sha256')!=protocol_hash or
        not isinstance(value.get('review_reference'),str) or not value['review_reference'].strip()):
        raise ValueError('Missing or mismatched stage clearance')
    if stage=='heldout' and value.get('training_manifest_sha256')!=training_hash:
        raise ValueError('Held-out clearance does not bind training artifact')
    return value


def import_archive(archive,out,archive_hash,manifest_hash,selected_names,max_bytes):
    """Authenticate whole archive before extracting bounded selected members.

    selected_names=None imports all manifest files plus historical completion
    and watchdog receipts. A held-out import selects only identity/metadata.
    This supports a large raw archive without expanding its unused logits.
    """
    archive=Path(archive); out=Path(out)
    record_access('read_authenticated_archive',archive)
    if archive.is_symlink() or not archive.is_file(): raise ValueError('Archive must be a regular file')
    if sha256(archive)!=archive_hash: raise ValueError('Archive identity mismatch')
    if out.exists() or out.is_symlink(): raise FileExistsError(str(out))
    with tarfile.open(archive,'r:*') as tar:
        members={}
        for member in tar:
            if len(members)>=1024: raise ValueError('Archive member limit exceeded')
            if member.name in ('','.','./') and member.isdir(): continue
            name=safe_relative(member.name)
            if name in members: raise ValueError('Duplicate normalized archive member')
            if not member.isfile() and not member.isdir(): raise ValueError('Non-regular archive member')
            members[name]=member
        candidates=[]
        for name,m in members.items():
            if m.isfile() and PurePosixPath(name).name=='MANIFEST.json':
                if m.size>2*1024*1024: raise ValueError('Archive manifest too large')
                raw=tar.extractfile(m).read()
                if digest_bytes(raw)==manifest_hash: candidates.append((name,raw))
        if len(candidates)!=1: raise ValueError('Unique authenticated historical manifest not found')
        manifest_name,raw=candidates[0]; value=parse_json(raw)
        if value.get('schema')!=1 or not isinstance(value.get('files'),dict) or not value['files']:
            raise ValueError('Invalid historical manifest')
        prefix=PurePosixPath(manifest_name).parent.as_posix()
        prefix='' if prefix=='.' else prefix+'/'
        names=(set(value['files'])|{'COMPLETE.json','WATCHDOG.json'} if selected_names is None else set(selected_names))|{'MANIFEST.json'}
        requested={}
        for name in names:
            safe_relative(name)
            member=members.get(prefix+name)
            if member is None or not member.isfile(): raise ValueError('Missing historical member: '+name)
            requested[name]=member
        if sum(m.size for m in requested.values())>max_bytes:
            raise ValueError('Selected archive expansion exceeds bound')
        out.mkdir(parents=False,exist_ok=False)
        for name,m in sorted(requested.items()):
            target=out/name; target.parent.mkdir(parents=True,exist_ok=True)
            record_access('write_archive_member_exclusive',target)
            h=hashlib.sha256(); count=0
            with tar.extractfile(m) as source,open(target,'xb') as dest:
                for block in iter(lambda:source.read(1024*1024),b''):
                    count+=len(block)
                    if count>m.size: raise ValueError('Archive member size changed')
                    h.update(block); dest.write(block)
            if count!=m.size: raise ValueError('Truncated archive member')
            if name in value['files']:
                expected=value['files'][name]
                if h.hexdigest()!=expected['sha256'] or count!=expected['bytes']:
                    raise ValueError('Selected historical member hash mismatch')
    if sha256(archive)!=archive_hash: raise ValueError('Archive changed during import')
    return {'archive_sha256':archive_hash,'manifest_sha256':manifest_hash,
            'selected_names':sorted(names),'selected_bytes':sum(m.size for m in requested.values())}
