"""Separately cleared, fixed-budget, read-only saved-evidence audit supervisor.

Production arguments cannot change the scientific specification or audit budget.
A successful mechanical audit is not scientific acceptance or permission for any
experiment. Source and clearance preflight precede all scientific artifact reads.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import custody
import settings
import stageio
from watchdog import supervise

STATUS = 'mechanical_audit_passed_pending_reviewer_decision'
SCOPE = ('Mechanical saved-evidence verification only; no scientific acceptance, '
         'execution authorization, adaptive option, or new experimental refit. '
         'The 90 SVD head recomputations deterministically verify existing heads.')


def audit_budget():
    """Independent audit allowance, not charged to or inherited from experiment stages."""
    return {'cpu_only': True, 'blas_threads_max': 2, 'wall_seconds': 600,
            'rss_bytes': 1024**3, 'output_bytes': 128*1024**2,
            'deterministic_head_verifications': 90}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=['training', 'heldout'])
    ap.add_argument('--artifact', required=True, type=Path)
    ap.add_argument('--artifact-manifest-sha256', required=True)
    ap.add_argument('--training', type=Path)
    ap.add_argument('--training-manifest-sha256')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--clearance', required=True, type=Path)
    return ap


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a lowercase SHA256 identity, not an inferred artifact hash')
    return value


def verify_clearance(args, source_hash):
    """Only reads clearance; expected identities are supplied, never inferred by reading inputs."""
    _digest(source_hash); _digest(args.artifact_manifest_sha256)
    if args.stage not in {'training', 'heldout'}:
        raise ValueError('Invalid audit stage')
    if args.stage == 'heldout':
        if args.training is None:
            raise ValueError('Held-out audit requires an exact training artifact')
        _digest(args.training_manifest_sha256)
    elif args.training is not None or args.training_manifest_sha256 is not None:
        raise ValueError('Training audit accepts no training-directory binding')
    value = custody.read_json(args.clearance)
    expected = {'stage': 'audit-' + args.stage,
                'source_manifest_sha256': source_hash,
                'protocol_manifest_sha256': settings.PROTOCOL_HASH,
                'artifact_manifest_sha256': args.artifact_manifest_sha256,
                'training_manifest_sha256': args.training_manifest_sha256,
                'audit_budget': audit_budget()}
    if (not isinstance(value, dict) or value.get('accepted') is not True or
        any(value.get(key) != item for key, item in expected.items()) or
        not isinstance(value.get('review_reference'), str) or not value['review_reference'].strip()):
        raise ValueError('Missing or mismatched separate audit clearance')
    return value


def validate_paths(args):
    out = args.out.resolve()
    protected = [settings.ROOT.resolve(), args.artifact.resolve(), args.clearance.resolve()]
    if args.training is not None:
        protected.append(args.training.resolve())
        if args.training.resolve() == args.artifact.resolve():
            raise ValueError('Held-out and training artifacts must be distinct')
    if any(out.is_relative_to(path) or path.is_relative_to(out) for path in protected):
        raise ValueError('Audit output must be outside, without overlap with source, inputs, or clearance')
    if args.out.is_symlink() or out.exists():
        raise FileExistsError(str(args.out))
    if not out.parent.is_dir():
        raise ValueError('Output parent must already exist')
    return out


def prepare(args):
    if sys.platform != 'linux':
        raise ValueError('Only reviewed Linux supervision is supported')
    source_hash = settings.verify_source()
    cleared = verify_clearance(args, source_hash)
    validate_paths(args)
    return settings.production_spec(), source_hash, cleared


def verify_inputs(args, spec, source_hash):
    """Hash exact sealed input trees after clearance; never write to them."""
    stageio.verify_completed(args.artifact, args.stage, spec, source_hash,
                             args.artifact_manifest_sha256)
    result = {'artifact': custody.manifest(args.artifact, custody.files_under(args.artifact))}
    if args.stage == 'heldout':
        stageio.verify_completed(args.training, 'training', spec, source_hash,
                                 args.training_manifest_sha256)
        metadata = custody.read_json(args.artifact/'metadata.json')
        if metadata.get('training_manifest_sha256') != args.training_manifest_sha256:
            raise ValueError('Audited held-out artifact binds a different training manifest')
        result['training'] = custody.manifest(args.training, custody.files_under(args.training))
    return result


def report_envelope(report, spec, source_hash, artifact_hash, training_hash=None):
    """Worker record used by both the production worker and invented-array test child."""
    import platform
    import numpy as np
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    if any(type(p.get('num_threads')) is not int or not 1 <= p['num_threads'] <= 2 for p in pools):
        raise ValueError('Numerical thread ceiling exceeded')
    return {'report': report, 'source_manifest_sha256': source_hash,
            'protocol_manifest_sha256': spec['protocol_manifest_sha256'],
            'spec_sha256': stageio.spec_hash(spec), 'artifact_manifest_sha256': artifact_hash,
            'training_manifest_sha256': training_hash, 'budget': audit_budget(),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'cpu_only': True, 'threadpools': pools},
            'scientific_acceptance': False, 'authorizes_execution': False, 'scope': SCOPE}


def _verify_report(value, args, spec, source_hash):
    expected = {'source_manifest_sha256': source_hash,
                'protocol_manifest_sha256': spec['protocol_manifest_sha256'],
                'spec_sha256': stageio.spec_hash(spec),
                'artifact_manifest_sha256': args.artifact_manifest_sha256,
                'training_manifest_sha256': args.training_manifest_sha256,
                'budget': audit_budget()}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Audit report binding differs')
    report = value.get('report', {})
    if (report.get('status') != 'passed' or report.get('stage') != args.stage or
        report.get('manifest_sha256') != args.artifact_manifest_sha256 or
        value.get('scientific_acceptance') is not False or value.get('authorizes_execution') is not False):
        raise ValueError('No successful correctly scoped audit report')
    if args.stage == 'training':
        if report.get('readout_count') != 90:
            raise ValueError('Audit did not verify exactly 90 saved heads')
    elif (report.get('training_manifest_sha256') != args.training_manifest_sha256 or
          report.get('verification_refits') != 90 or report.get('test_conditions') != 100 or report.get('cases') != 360):
        raise ValueError('Held-out audit coverage differs')
    runtime = value.get('runtime', {})
    pools = runtime.get('threadpools')
    if (runtime.get('numpy') != '2.3.5' or runtime.get('cpu_only') is not True or
        not isinstance(pools, list) or not pools or
        any(type(p.get('num_threads')) is not int or not 1 <= p['num_threads'] <= 2 for p in pools)):
        raise ValueError('Unreviewed audit numerical runtime or thread budget')


def _verify_watchdog(receipt):
    budget = audit_budget()
    if (receipt.get('status') != 'success' or receipt.get('success') is not True or
        receipt.get('returncode') != 0 or receipt.get('violations') != [] or receipt.get('errors') != [] or
        any(receipt.get('limits', {}).get(k) != budget[k] for k in ('wall_seconds', 'rss_bytes', 'output_bytes'))):
        raise ValueError('Audit watchdog did not establish bounded success')
    for name, maximum in [('elapsed_seconds', budget['wall_seconds']),
                          ('peak_sampled_rss_bytes', budget['rss_bytes']),
                          ('output_bytes', budget['output_bytes']), ('peak_sampled_output_bytes', budget['output_bytes'])]:
        number = receipt.get(name)
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= maximum:
            raise ValueError('Audit watchdog observation outside bound: ' + name)


def run_command(command, args, spec, source_hash, source_recheck=None):
    """Fixed-budget supervisor. Synthetic tests supply only invented-array commands/specs.

    The production CLI never accepts a command, synthetic spec, or budget override.
    Source/clearance preflight is outside the numerical audit budget. The recorded
    whole-audit timer spans input verification, child execution, and final sealing.
    """
    cleared = verify_clearance(args, source_hash)
    out = validate_paths(args)
    started = time.monotonic()
    clearance_hash = custody.sha256(args.clearance)
    before = verify_inputs(args, spec, source_hash)
    out.mkdir(exist_ok=False)
    budget = audit_budget()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', BLIS_NUM_THREADS='2',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(settings.ROOT), READOUT_AUDIT_WATCHDOG='1')
    limits = {k: budget[k] for k in ('wall_seconds', 'rss_bytes', 'output_bytes')}
    limits['poll_seconds'] = .02

    def check():
        elapsed = time.monotonic() - started
        size = sum(custody.safe_file(out, n).stat().st_size for n in custody.files_under(out))
        if elapsed > budget['wall_seconds'] or size > budget['output_bytes']:
            raise ValueError('Whole audit resource ceiling exceeded')
        return elapsed, size

    try:
        check()
        receipt = supervise(command, out, limits, env)
        if receipt['status'] != 'success':
            marker = out/'AUDIT_COMPLETE.json'
            if marker.is_file() or marker.is_symlink():
                marker.unlink()
            custody.write_json(out/'FAILED_AUDIT.json', {'reason': 'Audit child failed', 'watchdog_status': receipt['status']})
            return {'status': 'audit_failed', 'scientific_acceptance': False,
                    'authorizes_execution': False, 'watchdog': receipt}
        _verify_watchdog(receipt)
        if any('FAILED' in Path(n).name for n in custody.files_under(out)):
            raise ValueError('Failure evidence invalidates audit')
        report = custody.read_json(out/'AUDIT_REPORT.json')
        _verify_report(report, args, spec, source_hash)
        if (source_recheck or settings.verify_source)() != source_hash:
            raise ValueError('Source changed during audit')
        if verify_clearance(args, source_hash) != cleared or custody.sha256(args.clearance) != clearance_hash:
            raise ValueError('Audit clearance changed during audit')
        if verify_inputs(args, spec, source_hash) != before:
            raise ValueError('Audit input changed during audit')
        elapsed, size = check()
        custody.write_json(out/'AUDIT_INPUTS.json', {'artifact_manifest_sha256': args.artifact_manifest_sha256,
            'training_manifest_sha256': args.training_manifest_sha256, 'unchanged_after_audit': True,
            'snapshots': before, 'clearance': cleared, 'clearance_sha256': clearance_hash})
        custody.write_json(out/'AUDIT_RESOURCES.json', {'budget': budget,
            'elapsed_before_sealing_seconds': elapsed, 'bytes_before_sealing': size,
            'peak_sampled_rss_bytes': receipt['peak_sampled_rss_bytes'],
            'scope': 'Whole audit timer includes input verification through terminal checks; source/clearance preflight excluded. '
                     'Child-tree RSS and output are sampled, not kernel containment. Parent memory is excluded.',
            'scientific_acceptance': False})
        custody.write_json(out/'MANIFEST.json', custody.manifest(out, custody.files_under(out)))
        check()
        complete = {'status': STATUS, 'manifest_sha256': custody.sha256(out/'MANIFEST.json'),
                    'scientific_acceptance': False, 'authorizes_execution': False, 'scope': SCOPE}
        custody.write_json(out/'AUDIT_COMPLETE.json', complete)
        custody.verify_manifest(out, expected_hash=complete['manifest_sha256'], extra_allowed={'AUDIT_COMPLETE.json'})
        check()
        return dict(complete, watchdog=receipt)
    except BaseException as exc:
        # Only our fresh output is touched. Never leave a success marker after failure.
        marker = out/'AUDIT_COMPLETE.json'
        if marker.is_file() or marker.is_symlink():
            marker.unlink()
        if not (out/'FAILED_AUDIT.json').exists():
            custody.write_json(out/'FAILED_AUDIT.json', {'type': type(exc).__name__, 'reason': str(exc)})
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    spec, source_hash, _ = prepare(args)
    command = [sys.executable, '-B', str(settings.ROOT/'audit_worker.py'),
               '--stage', args.stage, '--artifact', str(args.artifact.resolve()),
               '--artifact-manifest-sha256', args.artifact_manifest_sha256,
               '--out', str(args.out.resolve()), '--clearance', str(args.clearance.resolve())]
    if args.training is not None:
        command += ['--training', str(args.training.resolve()),
                    '--training-manifest-sha256', args.training_manifest_sha256]
    result = run_command(command, args, spec, source_hash, settings.verify_source)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == STATUS else 1


if __name__ == '__main__':
    sys.exit(main())
