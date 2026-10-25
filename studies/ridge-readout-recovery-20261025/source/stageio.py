"""Execution completion is distinct from independent scientific acceptance."""
from pathlib import Path
import math
import time
import custody


def spec_hash(spec):
    import json
    return custody.digest_bytes(json.dumps(spec,sort_keys=True,allow_nan=False,separators=(',',':')).encode())


def verify_watchdog(receipt,spec):
    cfg=spec['config']; limits=receipt['limits']
    expected={'wall_seconds':cfg['max_seconds_per_stage'],'rss_bytes':cfg['max_rss_bytes'],
              'output_bytes':cfg['max_output_bytes_per_stage']}
    if (receipt.get('status')!='success' or receipt.get('success') is not True or
        receipt.get('returncode')!=0 or receipt.get('violations')!=[] or receipt.get('errors')!=[] or
        any(limits.get(k)!=v for k,v in expected.items())):
        raise ValueError('Watchdog did not establish successful bounded execution')
    for field,limit in [('elapsed_seconds',expected['wall_seconds']),
                        ('peak_sampled_rss_bytes',expected['rss_bytes']),('output_bytes',expected['output_bytes']),
                        ('peak_sampled_output_bytes',expected['output_bytes'])]:
        v=receipt.get(field)
        if type(v) not in (int,float) or not math.isfinite(v) or v<0 or v>limit:
            raise ValueError('Watchdog observation outside bound: '+field)


def verify_completed(root,stage,spec,source_hash,expected_manifest_hash=None):
    root=Path(root)
    if any('FAILED' in Path(n).name for n in custody.files_under(root)):
        raise ValueError('Failure evidence invalidates stage')
    if not (root/'COMPLETE.json').is_file():
        raise ValueError('Stage is not sealed complete; failed or pending stages cannot proceed')
    complete=custody.read_json(root/'COMPLETE.json')
    if expected_manifest_hash is not None and complete.get('manifest_sha256')!=expected_manifest_hash:
        raise ValueError('Completed artifact differs from cleared training manifest')
    custody.verify_manifest(root,'MANIFEST.json',complete.get('manifest_sha256'),extra_allowed={'COMPLETE.json'})
    meta=custody.read_json(root/'metadata.json')
    if (meta.get('stage')!=stage or meta.get('spec')!=spec or
        meta.get('source_manifest_sha256')!=source_hash or
        meta.get('protocol_manifest_sha256')!=spec['protocol_manifest_sha256']):
        raise ValueError('Stage source/protocol/input specification binding differs')
    if complete.get('status')!='execution_complete_pending_independent_audit':
        raise ValueError('Invalid completion status')
    verify_watchdog(custody.read_json(root/'WATCHDOG.json'),spec)
    result=custody.read_json(root/'WORKER_RESULT.json')
    if result.get('status')!='completed' or result.get('stage')!=stage:
        raise ValueError('Worker did not complete stage')
    if stage=='training' and custody.read_json(root/'summary.json')['gates']['eligible'] is not True:
        raise ValueError('Training gate failed; held-out stage is forbidden')
    if sum((root/n).stat().st_size for n in custody.files_under(root))>spec['config']['max_output_bytes_per_stage']:
        raise ValueError('Actual sealed output exceeds bound')
    resources=custody.read_json(root/'STAGE_RESOURCES.json')
    elapsed=resources.get('elapsed_before_sealing_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=spec['config']['max_seconds_per_stage']:
        raise ValueError('Whole stage exceeded wall bound')
    return meta


def finalize(root,stage,spec,source_hash,started):
    root=Path(root)
    receipt=custody.read_json(root/'WATCHDOG.json'); verify_watchdog(receipt,spec)
    result=custody.read_json(root/'WORKER_RESULT.json')
    if result.get('status')!='completed' or result.get('stage')!=stage:
        raise ValueError('No successful worker result to seal')
    meta=custody.read_json(root/'metadata.json')
    if meta.get('spec')!=spec or meta.get('source_manifest_sha256')!=source_hash or meta.get('stage')!=stage:
        raise ValueError('Cannot seal mismatched worker metadata')
    if stage=='training' and not custody.read_json(root/'summary.json')['gates']['eligible']:
        raise ValueError('Failed competence gate cannot be accepted')
    def check():
        elapsed=time.monotonic()-started
        size=sum((root/n).stat().st_size for n in custody.files_under(root))
        if elapsed>spec['config']['max_seconds_per_stage'] or size>spec['config']['max_output_bytes_per_stage']:
            raise ValueError('Whole stage resource ceiling exceeded')
        return elapsed,size
    elapsed,size=check()
    custody.write_json(root/'STAGE_RESOURCES.json',{'elapsed_before_sealing_seconds':elapsed,
      'bytes_before_sealing':size,'peak_sampled_rss_bytes':receipt['peak_sampled_rss_bytes'],
      'scope':'Supervisor wall check spans run_command entry through terminal verification; excludes CLI preflight before run_command. RSS/output sampled, not kernel containment'})
    custody.write_json(root/'MANIFEST.json',custody.manifest(root,custody.files_under(root)))
    check()
    custody.write_json(root/'COMPLETE.json',{'manifest_sha256':custody.sha256(root/'MANIFEST.json'),
      'status':'execution_complete_pending_independent_audit'})
    try:
        check()
        verified=verify_completed(root,stage,spec,source_hash)
        check()
    except BaseException as exc:
        custody.write_json(root/'FAILED_FINALIZATION.json',{'reason':str(exc)})
        raise
    return verified
