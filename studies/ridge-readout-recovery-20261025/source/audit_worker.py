"""Production audit child. Use audit_runner.py, never execute this child directly."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import custody
import settings
import audit_runner


def main(argv=None):
    args = audit_runner.parser().parse_args(argv)
    if os.environ.get('READOUT_AUDIT_WATCHDOG') != '1':
        raise ValueError('Use the supervised audit runner, not the audit worker directly')
    if sys.platform != 'linux':
        raise ValueError('Only reviewed Linux supervision is supported')
    # Source identity and explicit audit-only clearance precede all artifact access.
    source_hash = settings.verify_source()
    cleared = audit_runner.verify_clearance(args, source_hash)
    spec = settings.production_spec()
    out = args.out
    if not out.is_dir() or set(custody.files_under(out)) - {'stdout.log', 'stderr.log'}:
        raise ValueError('Audit output must be fresh except supervisor logs')
    custody.begin_access_log()
    try:
        before = audit_runner.verify_inputs(args, spec, source_hash)
        import numpy as np
        if np.__version__ != '2.3.5':
            raise ValueError('Unreviewed NumPy runtime')
        import audit
        # Establish numerical runtime/thread limits before running any SVDs.
        audit_runner.report_envelope({}, spec, source_hash, args.artifact_manifest_sha256,
                                     args.training_manifest_sha256)
        report = (audit.audit_training(args.artifact, spec, source_hash) if args.stage == 'training' else
                  audit.audit_heldout(args.artifact, args.training, spec, source_hash))
        if settings.verify_source() != source_hash:
            raise ValueError('Source changed during audit child')
        if audit_runner.verify_clearance(args, source_hash) != cleared:
            raise ValueError('Audit clearance changed during child')
        if audit_runner.verify_inputs(args, spec, source_hash) != before:
            raise ValueError('Audit input changed during child')
        custody.write_json(out/'AUDIT_REPORT.json', audit_runner.report_envelope(
            report, spec, source_hash, args.artifact_manifest_sha256, args.training_manifest_sha256))
        custody.write_json(out/'AUDIT_ACCESS_LOG.json', {
            'scope': 'Explicit file-operation calls only, not OS tracing. Input directories are never written.',
            'operations': custody.access_log()})
        return 0
    except BaseException as exc:
        if not (out/'FAILED_WORKER.json').exists():
            custody.write_json(out/'FAILED_WORKER.json', {'type': type(exc).__name__, 'reason': str(exc)})
        if not (out/'AUDIT_ACCESS_LOG.json').exists():
            custody.write_json(out/'AUDIT_ACCESS_LOG.json', {'scope': 'Partial explicit file-operation record, not OS tracing',
                                                          'operations': custody.access_log()})
        raise


if __name__ == '__main__':
    sys.exit(main())
