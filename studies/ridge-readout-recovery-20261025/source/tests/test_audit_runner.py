"""Audit execution tests: fabricated identities and invented arrays only."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import custody
import settings
import pipeline
from fixtures import fixture_archives, synthetic_context, seal_fixture


class AuditRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        ta, ha, cls.spec = fixture_archives(cls.root)
        cls.context = synthetic_context()
        cls.training = cls.root / 'training'
        cls.training.mkdir()
        pipeline.train_stage(ta, cls.training, cls.spec, cls.context)
        seal_fixture(cls.training, 'training', cls.spec, cls.context)
        cls.heldout = cls.root / 'heldout'
        cls.heldout.mkdir()
        pipeline.heldout_stage(ha, cls.training, cls.heldout, cls.spec, cls.context)
        seal_fixture(cls.heldout, 'heldout', cls.spec, cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))
        self.source_hash = self.context['source_manifest_sha256']

    def tearDown(self):
        shutil.rmtree(self.case)

    def api(self):
        import audit_runner
        return audit_runner

    def clearance(self, stage='training'):
        api = self.api()
        artifact = self.training if stage == 'training' else self.heldout
        value = {'accepted': True, 'stage': 'audit-' + stage,
                 'source_manifest_sha256': self.source_hash,
                 'protocol_manifest_sha256': settings.PROTOCOL_HASH,
                 'artifact_manifest_sha256': custody.sha256(artifact / 'MANIFEST.json'),
                 'training_manifest_sha256': None if stage == 'training' else custody.sha256(self.training / 'MANIFEST.json'),
                 'audit_budget': api.audit_budget(), 'review_reference': 'Invented fixture clearance only'}
        path = self.case / ('clearance-' + stage + '.json')
        custody.write_json(path, value)
        return path, value

    def options(self, stage='training'):
        path, value = self.clearance(stage)
        argv = ['--stage', stage, '--artifact', str(self.training if stage == 'training' else self.heldout),
                '--out', str(self.case / 'audit'), '--clearance', str(path),
                '--artifact-manifest-sha256', value['artifact_manifest_sha256']]
        if stage == 'heldout':
            argv += ['--training', str(self.training), '--training-manifest-sha256', value['training_manifest_sha256']]
        return self.api().parser().parse_args(argv), value

    def test_fixed_budget_has_no_cli_override(self):
        api = self.api()
        self.assertEqual(api.audit_budget(), {'cpu_only': True, 'blas_threads_max': 2,
            'wall_seconds': 600, 'rss_bytes': 1073741824, 'output_bytes': 134217728,
            'deterministic_head_verifications': 90})
        self.assertNotIn('spec', {a.dest for a in api.parser()._actions})
        self.assertNotIn('wall_seconds', {a.dest for a in api.parser()._actions})

    def test_clearance_exact_bindings(self):
        api = self.api()
        args, value = self.options('heldout')
        self.assertEqual(api.verify_clearance(args, self.source_hash), value)
        for field, bad in [('stage', 'heldout'), ('accepted', False), ('artifact_manifest_sha256', 'b'*64),
                           ('training_manifest_sha256', 'c'*64), ('source_manifest_sha256', 'd'*64),
                           ('protocol_manifest_sha256', 'e'*64), ('review_reference', ''), ('audit_budget', {})]:
            altered = dict(value, **{field: bad})
            args.clearance.write_text(json.dumps(altered))
            with self.subTest(field=field), self.assertRaises(ValueError):
                api.verify_clearance(args, self.source_hash)

    def test_training_rejects_unexpected_training_binding(self):
        args, value = self.options()
        value['training_manifest_sha256'] = 'b'*64
        args.clearance.write_text(json.dumps(value))
        with self.assertRaises(ValueError): self.api().verify_clearance(args, self.source_hash)

    def test_denied_clearance_never_reads_artifacts(self):
        api = self.api()
        args, value = self.options('heldout')
        value['accepted'] = False
        args.clearance.write_text(json.dumps(value))
        with patch.object(settings, 'verify_source', return_value=self.source_hash), \
             patch.object(api, 'verify_inputs', side_effect=AssertionError('premature artifact read')):
            with self.assertRaisesRegex(ValueError, 'clearance'):
                api.prepare(args)
        self.assertFalse(args.out.exists())

    def test_source_failure_precedes_clearance_or_artifact_reads(self):
        api = self.api(); args, _ = self.options()
        with patch.object(settings, 'verify_source', side_effect=ValueError('source changed')), \
             patch.object(api, 'verify_clearance', side_effect=AssertionError('source unchecked')):
            with self.assertRaisesRegex(ValueError, 'source changed'): api.prepare(args)

    def test_output_overlap_rejected(self):
        api = self.api(); args, _ = self.options('heldout')
        for output in (settings.ROOT/'audit-out', self.training/'audit-out', self.heldout/'audit-out', self.root):
            args.out = output
            with patch.object(settings, 'verify_source', return_value=self.source_hash):
                with self.assertRaisesRegex(ValueError, 'overlap|outside'):
                    api.prepare(args)

    def run_fixture(self, stage='training', code=None, recheck=None):
        api = self.api(); args, _ = self.options(stage)
        payload = self.case / 'fixture.json'
        custody.write_json(payload, {'spec': self.spec, 'source_hash': self.source_hash,
            'stage': stage, 'artifact': str(args.artifact), 'training': str(args.training) if args.training else None,
            'artifact_hash': args.artifact_manifest_sha256, 'training_hash': args.training_manifest_sha256})
        helper = self.case / 'audit_fixture.py'
        helper.write_text(code or (
            'import json,pathlib,sys,audit,custody,audit_runner\n'
            'p=json.loads(pathlib.Path(sys.argv[1]).read_text()); out=pathlib.Path(sys.argv[2])\n'
            'r=audit.audit_training(p["artifact"],p["spec"],p["source_hash"]) if p["stage"]=="training" else audit.audit_heldout(p["artifact"],p["training"],p["spec"],p["source_hash"])\n'
            'custody.write_json(out/"AUDIT_REPORT.json",audit_runner.report_envelope(r,p["spec"],p["source_hash"],p["artifact_hash"],p["training_hash"]))\n'))
        command = [sys.executable, '-B', str(helper), str(payload), str(args.out)]
        return api.run_command(command, args, self.spec, self.source_hash,
                               source_recheck=recheck or (lambda: self.source_hash))

    def test_real_subprocess_training_audit_read_only(self):
        api = self.api()
        before = custody.manifest(self.training, custody.files_under(self.training))
        result = self.run_fixture()
        self.assertEqual(result['status'], 'mechanical_audit_passed_pending_reviewer_decision')
        self.assertIs(result['scientific_acceptance'], False)
        out = self.case / 'audit'
        complete = custody.read_json(out/'AUDIT_COMPLETE.json')
        custody.verify_manifest(out, expected_hash=complete['manifest_sha256'], extra_allowed={'AUDIT_COMPLETE.json'})
        self.assertEqual(custody.read_json(out/'AUDIT_RESOURCES.json')['budget'], api.audit_budget())
        self.assertEqual(before, custody.manifest(self.training, custody.files_under(self.training)))

    def test_real_subprocess_heldout_audit_read_only(self):
        before = custody.manifest(self.heldout, custody.files_under(self.heldout))
        result = self.run_fixture('heldout')
        self.assertEqual(result['status'], 'mechanical_audit_passed_pending_reviewer_decision')
        self.assertEqual(before, custody.manifest(self.heldout, custody.files_under(self.heldout)))

    def test_nonzero_or_missing_report_has_no_accepted_receipt(self):
        result = self.run_fixture(code='raise SystemExit(7)\n')
        self.assertEqual(result['status'], 'audit_failed')
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())

    def test_failed_child_cannot_leave_a_success_marker(self):
        result = self.run_fixture(code='import pathlib,sys\npathlib.Path(sys.argv[2],"AUDIT_COMPLETE.json").write_text("{}")\nraise SystemExit(7)\n')
        self.assertEqual(result['status'], 'audit_failed')
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())

    def test_success_without_report_is_failure(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_fixture(code='pass\n')
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())
        self.assertTrue((self.case/'audit/FAILED_AUDIT.json').exists())

    def test_changed_source_after_child_has_no_accepted_receipt(self):
        with self.assertRaisesRegex(ValueError, 'Source'):
            self.run_fixture(recheck=lambda: 'b'*64)
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())

    def test_changed_input_after_child_has_no_accepted_receipt(self):
        api = self.api(); original = api.verify_inputs; calls = []
        def altered(*args, **kwargs):
            value = original(*args, **kwargs)
            calls.append(1)
            return value if len(calls) == 1 else {'changed': True}
        with patch.object(api, 'verify_inputs', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'input'):
                self.run_fixture()
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())

    def test_existing_output_is_never_overwritten(self):
        api = self.api(); args, _ = self.options(); args.out.mkdir()
        sentinel = args.out/'sentinel'; sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            api.run_command([sys.executable, '-c', 'pass'], args, self.spec, self.source_hash)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_wrong_manifest_identity_rejected_before_launch(self):
        api = self.api(); args, _ = self.options()
        args.artifact_manifest_sha256 = 'b'*64
        with patch.object(api, 'supervise', side_effect=AssertionError('launch before identity')):
            with self.assertRaises(ValueError):
                api.run_command([sys.executable, '-c', 'pass'], args, self.spec, self.source_hash)
        self.assertFalse((args.out/'AUDIT_COMPLETE.json').exists())

    def test_bad_watchdog_budget_is_not_accepted(self):
        api = self.api()
        value = {'status': 'success', 'success': True, 'returncode': 0, 'violations': [], 'errors': [],
                 'elapsed_seconds': 1., 'peak_sampled_rss_bytes': 1000000, 'output_bytes': 1000,
                 'peak_sampled_output_bytes': 1000,
                 'limits': {'wall_seconds': 600, 'rss_bytes': 1073741824, 'output_bytes': 134217728}}
        api._verify_watchdog(value)
        for field, bad in [('errors', ['lost child']), ('elapsed_seconds', 601),
                           ('peak_sampled_rss_bytes', 1073741825), ('output_bytes', 134217729),
                           ('peak_sampled_output_bytes', float('nan')), ('returncode', 1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                api._verify_watchdog(dict(value, **{field: bad}))
        changed = copy.deepcopy(value); changed['limits']['wall_seconds'] = 601
        with self.assertRaises(ValueError): api._verify_watchdog(changed)

    def test_report_cannot_change_identity_scope_or_numerical_budget(self):
        api = self.api(); args, _ = self.options()
        report = {'status': 'passed', 'stage': 'training', 'readout_count': 90,
                  'manifest_sha256': args.artifact_manifest_sha256}
        value = api.report_envelope(report, self.spec, self.source_hash, args.artifact_manifest_sha256)
        api._verify_report(value, args, self.spec, self.source_hash)
        for field, bad in [('scientific_acceptance', True), ('authorizes_execution', True),
                           ('source_manifest_sha256', 'b'*64), ('artifact_manifest_sha256', 'c'*64),
                           ('budget', {}), ('spec_sha256', 'd'*64), ('runtime', {})]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                api._verify_report(dict(value, **{field: bad}), args, self.spec, self.source_hash)
        changed = copy.deepcopy(value); changed['report']['readout_count'] = 91
        with self.assertRaises(ValueError): api._verify_report(changed, args, self.spec, self.source_hash)

    def test_changed_clearance_after_child_has_no_accepted_receipt(self):
        api = self.api(); original = api.verify_clearance; calls = []
        def altered(*args, **kwargs):
            value = original(*args, **kwargs); calls.append(1)
            return value if len(calls) == 1 else dict(value, review_reference='changed')
        with patch.object(api, 'verify_clearance', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'clearance'):
                self.run_fixture()
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())

    def test_direct_worker_without_watchdog_refuses_access(self):
        api = self.api(); args, _ = self.options()
        env = dict(os.environ, PYTHONPATH=str(settings.ROOT)); env.pop('READOUT_AUDIT_WATCHDOG', None)
        result = subprocess.run([sys.executable, '-B', str(settings.ROOT/'audit_worker.py'),
            '--stage', 'training', '--artifact', '/nonexistent/synthetic', '--out', str(self.case/'out'),
            '--clearance', '/nonexistent/clearance', '--artifact-manifest-sha256', 'a'*64],
            env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('supervised audit runner', result.stderr)


if __name__ == '__main__': unittest.main()
