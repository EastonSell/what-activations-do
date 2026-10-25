"""Saved-evidence tests use invented arrays and manufactured receipts ONLY."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import custody
import pipeline
from fixtures import fixture_archives, synthetic_context, seal_fixture


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ta, cls.ha, cls.spec = fixture_archives(cls.root)
        cls.context = synthetic_context()
        cls.training = cls.root / 'training'
        cls.training.mkdir()
        pipeline.train_stage(cls.ta, cls.training, cls.spec, cls.context)
        seal_fixture(cls.training, 'training', cls.spec, cls.context)
        cls.heldout = cls.root / 'heldout'
        cls.heldout.mkdir()
        pipeline.heldout_stage(cls.ha, cls.training, cls.heldout, cls.spec, cls.context)
        seal_fixture(cls.heldout, 'heldout', cls.spec, cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))
        self.train = self.case / 'training'
        self.test = self.case / 'heldout'
        shutil.copytree(self.training, self.train)
        shutil.copytree(self.heldout, self.test)

    def tearDown(self):
        shutil.rmtree(self.case)

    def auditor(self):
        module = importlib.util.find_spec('audit')
        self.assertIsNotNone(module, 'Independent saved-evidence auditor is missing')
        import audit
        return audit

    def json_change(self, root, name, change):
        path = root / name
        value = custody.read_json(path)
        change(value)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))

    def npz_change(self, root, name, change):
        path = root / name
        with np.load(path, allow_pickle=False) as value:
            data = {k: value[k].copy() for k in value.files}
        change(data)
        np.savez_compressed(path, **data)

    def reseal(self, root):
        # Adversarially recompute the outer hashes. Numerical audit must still fail.
        names = [n for n in custody.files_under(root) if n not in {'MANIFEST.json', 'COMPLETE.json'}]
        (root / 'MANIFEST.json').write_text(json.dumps(custody.manifest(root, names)))
        (root / 'COMPLETE.json').write_text(json.dumps({
            'manifest_sha256': custody.sha256(root / 'MANIFEST.json'),
            'status': 'execution_complete_pending_independent_audit'}))

    def audit_train(self, **kwargs):
        return self.auditor().audit_training(self.train, kwargs.get('spec', self.spec),
                                              kwargs.get('source_hash', self.context['source_manifest_sha256']))

    def audit_test(self):
        return self.auditor().audit_heldout(self.test, self.train, self.spec,
                                             self.context['source_manifest_sha256'])

    def test_valid_training_is_read_only_and_independent(self):
        before = custody.manifest(self.train, custody.files_under(self.train))
        with patch('pipeline.historical_training', side_effect=AssertionError('shared reconstruction')), \
             patch('pipeline.conditions', side_effect=AssertionError('shared projection')), \
             patch('protocol_v2.readout.ridge', side_effect=AssertionError('shared solver')):
            report = self.audit_train()
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['readout_count'], 90)
        self.assertEqual(before, custody.manifest(self.train, custody.files_under(self.train)))

    def test_valid_heldout_has_complete_denominators(self):
        before = custody.manifest(self.test, custody.files_under(self.test))
        with patch('protocol_v2.readout.primary', side_effect=AssertionError('shared bootstrap')):
            report = self.audit_test()
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['test_conditions'], 100)
        self.assertEqual(report['cases'], 360)
        self.assertEqual(before, custody.manifest(self.test, custody.files_under(self.test)))

    def test_resealed_head_change_rejected(self):
        self.npz_change(self.train, 'fitted-101.npz', lambda d: d['weights'].__setitem__((0, 0, 0), d['weights'][0, 0, 0] + .01))
        self.reseal(self.train)
        with self.assertRaisesRegex(ValueError, 'weight|head|ridge'):
            self.audit_train()

    def test_resealed_full_score_change_rejected(self):
        self.npz_change(self.train, 'fitted-101.npz', lambda d: d['validation_scores'].__setitem__((17, 4, 2), d['validation_scores'][17, 4, 2] + .01))
        self.reseal(self.train)
        with self.assertRaisesRegex(ValueError, 'score'):
            self.audit_train()

    def test_source_hash_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.audit_train(source_hash='b' * 64)

    def test_resealed_source_metadata_change_rejected(self):
        self.json_change(self.train, 'metadata.json', lambda d: d.__setitem__('source_manifest_sha256', 'b' * 64))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()

    def test_label_and_case_order_corruption_rejected(self):
        for key in ('labels', 'case_indices'):
            with self.subTest(key=key):
                path = self.test / 'test-101.npz'
                original = path.read_bytes()
                self.npz_change(self.test, 'test-101.npz', lambda d: d.__setitem__(key, d[key][::-1]))
                self.reseal(self.test)
                with self.assertRaises(ValueError):
                    self.audit_test()
                path.write_bytes(original)
                self.reseal(self.test)

    def test_missing_and_duplicate_conditions_rejected(self):
        for edit in (lambda d: d['rows'].pop(), lambda d: d['rows'].__setitem__(-1, d['rows'][0])):
            with self.subTest(edit=edit):
                path = self.test / 'summary.json'
                original = path.read_bytes()
                self.json_change(self.test, 'summary.json', edit)
                self.reseal(self.test)
                with self.assertRaises(ValueError):
                    self.audit_test()
                path.write_bytes(original)
                self.reseal(self.test)

    def test_resealed_test_score_corruption_rejected(self):
        self.npz_change(self.test, 'test-503.npz', lambda d: d['original_scores'].__setitem__((1, 10, 0), d['original_scores'][1, 10, 0] + .1))
        self.reseal(self.test)
        with self.assertRaisesRegex(ValueError, 'score'):
            self.audit_test()

    def test_numerical_gate_claim_rejected(self):
        self.json_change(self.train, 'summary.json', lambda d: d['fit_rows'][0].__setitem__('stationarity_max_abs', 1e-6))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()

    def test_competence_gate_claim_rejected(self):
        self.json_change(self.train, 'summary.json', lambda d: d['gates'].__setitem__('eligible', False))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()

    def test_support_integer_count_rejected_even_when_fraction_unchanged(self):
        self.json_change(self.train, 'summary.json', lambda d: d['rows'][0]['support'].__setitem__('strict_negative_coordinate_count', 1))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()

    def test_fractional_confusion_count_rejected(self):
        self.json_change(self.train, 'summary.json', lambda d: d['rows'][0]['confusion_matrix'][0].__setitem__(0, 108.0))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()

    def test_resealed_bootstrap_and_secondary_corruption_rejected(self):
        for edit in (lambda d: d['primary'].__setitem__('mean_pp', 17.),
                     lambda d: d['primary'].__setitem__('decision', 'inconclusive'),
                     lambda d: d['pca_refit_minus_mean_random_refit_accuracy_pp_per_seed'].__setitem__(4, 1.234)):
            with self.subTest(edit=edit):
                path = self.test / 'summary.json'
                original = path.read_bytes()
                self.json_change(self.test, 'summary.json', edit)
                self.reseal(self.test)
                with self.assertRaises(ValueError):
                    self.audit_test()
                path.write_bytes(original)
                self.reseal(self.test)

    def test_budget_gate_rejected(self):
        self.json_change(self.train, 'STAGE_RESOURCES.json', lambda d: d.__setitem__('peak_sampled_rss_bytes', 2**31))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()

    def test_relaxed_numeric_spec_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec['numeric']['score_atol'] = 10.
        self.json_change(self.train, 'metadata.json', lambda d: d.__setitem__('spec', spec))
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train(spec=spec)

    def test_missing_fitted_file_rejected(self):
        (self.train / 'fitted-503.npz').unlink()
        self.reseal(self.train)
        with self.assertRaises(ValueError):
            self.audit_train()


    def test_impossible_diagnostic_and_proportion_domains_rejected(self):
        edits = [
            lambda d: d['fit_rows'][0].__setitem__('stationarity_max_abs', -1e-12),
            lambda d: d['fit_rows'][0].__setitem__('normal_equation_relative_max_abs', -1e-12),
            lambda d: d['rows'][0].__setitem__('accuracy', 1.0 + 5e-10),
            lambda d: d['rows'][0]['support'].__setitem__('material_negative_coordinate_fraction', -1e-12),
        ]
        path = self.train / 'summary.json'
        original = path.read_bytes()
        for edit in edits:
            with self.subTest(edit=edit):
                self.json_change(self.train, 'summary.json', edit)
                self.reseal(self.train)
                with self.assertRaises(ValueError):
                    self.audit_train()
                path.write_bytes(original)
                self.reseal(self.train)


if __name__ == '__main__':
    unittest.main()
