"""Independent generated-only corruption tests. Never reads production archives/data."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

RUNNER = Path(__file__).resolve().parent.parent / 'source'
sys.path[:0] = [str(RUNNER), str(RUNNER / 'tests')]
import audit
import custody
from test_audit import AuditTests


class IndependentAuditTests(AuditTests):
    # Inherit synthetic fixture construction/helpers, but run independent names only.
    def reject_train(self, pattern=None):
        self.reseal(self.train)
        with self.assertRaisesRegex(ValueError, pattern or '.'):
            self.audit_train()

    def reject_test(self, pattern=None):
        self.reseal(self.test)
        with self.assertRaisesRegex(ValueError, pattern or '.'):
            self.audit_test()

    def test_independent_seed_swap(self):
        left, right = self.train/'fitted-101.npz', self.train/'fitted-503.npz'
        a,b=left.read_bytes(),right.read_bytes()
        left.write_bytes(b);right.write_bytes(a)
        self.reject_train('head|weight|ridge')

    def test_independent_control_draw_swap(self):
        def mutate(d):
            for key in ('weights','intercepts','train_scores','validation_scores'):
                d[key][[2,17]]=d[key][[17,2]]
        self.npz_change(self.train,'fitted-101.npz',mutate)
        self.reject_train('head|weight|ridge')

    def test_independent_condition_name_reorder(self):
        self.npz_change(self.train,'fitted-101.npz',lambda d:d['condition_names'].__setitem__(slice(2,4),d['condition_names'][2:4][::-1]))
        self.reject_train('condition order')

    def test_independent_intercept(self):
        self.npz_change(self.train,'fitted-101.npz',lambda d:d['intercepts'].__setitem__((17,9),d['intercepts'][17,9]+.01))
        self.reject_train('intercept|head')

    def test_independent_nan_score(self):
        self.npz_change(self.train,'fitted-101.npz',lambda d:d['train_scores'].__setitem__((0,0,0),float('nan')))
        self.reject_train('Nonfinite')

    def test_independent_lower_precision_score(self):
        self.npz_change(self.train,'fitted-101.npz',lambda d:d.__setitem__('train_scores',d['train_scores'].astype('float32')))
        self.reject_train('shape/dtype/finite')

    def test_independent_correctness_boolean(self):
        self.npz_change(self.test,'test-503.npz',lambda d:d['correctness'].__setitem__((17,359),not d['correctness'][17,359]))
        self.reject_test('correctness')

    def test_independent_test_refit_full_score(self):
        self.npz_change(self.test,'test-503.npz',lambda d:d['scores'].__setitem__((17,359,9),d['scores'][17,359,9]+.01))
        self.reject_test('score')

    def test_independent_confusion_metric(self):
        self.json_change(self.test,'summary.json',lambda d:d['rows'][-1]['confusion_matrix'][0].__setitem__(0,999))
        self.reject_test('confusion')

    def test_independent_seed_row_identity(self):
        self.json_change(self.train,'summary.json',lambda d:d['fit_rows'][0].__setitem__('seed',503))
        self.reject_train('seed')

    def test_independent_rank_diagnostic(self):
        self.json_change(self.train,'summary.json',lambda d:d['fit_rows'][0].__setitem__('actual_train_rank',32))
        self.reject_train('rank')

    def test_independent_head_normal_equation_gate(self):
        self.json_change(self.train,'summary.json',lambda d:d['fit_rows'][0].__setitem__('normal_equation_relative_max_abs',5e-10))
        # Difference is less than comparison atol; hard gate must still reject.
        self.reject_train('numerical gate|exceeds bound')

    def test_independent_stage_failure_marker(self):
        (self.train/'FAILED_AUDIT.json').write_text('{}')
        self.reject_train('Failure')

    def test_independent_watchdog_failure(self):
        self.json_change(self.train,'WATCHDOG.json',lambda d:d.update(success=False,status='failure'))
        self.reject_train('Watchdog')

    def test_independent_completion_pending(self):
        self.json_change(self.train,'COMPLETE.json',lambda d:d.update(status='pending'))
        with self.assertRaisesRegex(ValueError,'completion'):
            self.audit_train()

    def test_independent_missing_historical_identity(self):
        (self.test/'historical_heldout/test-identity.npz').unlink()
        self.reject_test('coverage|missing|Missing|mismatch')

    def test_independent_relaxed_config(self):
        spec=copy.deepcopy(self.spec);spec['config']['normal_equation_gradient_atol']=1.
        self.json_change(self.train,'metadata.json',lambda d:d.update(spec=spec))
        self.reseal(self.train)
        with self.assertRaisesRegex(ValueError,'Frozen specification'):
            self.audit_train(spec=spec)

    def test_independent_primary_bootstrap_seed_claim(self):
        self.json_change(self.test,'summary.json',lambda d:d['primary'].update(bootstrap_seed=42))
        self.reject_test('bootstrap_seed')

    def test_independent_primary_interval_claim(self):
        self.json_change(self.test,'summary.json',lambda d:d['primary']['case_resampling_interval95_pp'].__setitem__(0,100.))
        self.reject_test('interval')

    def test_independent_primary_scope_claim(self):
        self.json_change(self.test,'summary.json',lambda d:d['primary'].update(scope='Calibrated confidence interval'))
        self.reject_test('scope')

    def test_independent_no_hidden_evidence_writes(self):
        before=custody.manifest(self.case,custody.files_under(self.case))
        with patch('custody.write_json',side_effect=AssertionError('write')),patch('custody.write_npz',side_effect=AssertionError('write')):
            self.audit_test()
        self.assertEqual(before,custody.manifest(self.case,custody.files_under(self.case)))

    def test_independent_projector_semantics_after_fixture_rebind(self):
        # Rebind only synthetic hashes to reach semantic geometry checks, not fail on pin.
        historical=self.train/'historical_training'
        self.npz_change(historical,'model-101-A.npz',lambda d:d['pcs'].__setitem__((slice(None),[0,31]),d['pcs'][:,[31,0]]))
        names=[n for n in custody.files_under(historical) if n not in {'MANIFEST.json','COMPLETE.json','WATCHDOG.json'}]
        (historical/'MANIFEST.json').write_text(json.dumps(custody.manifest(historical,names)))
        spec=copy.deepcopy(self.spec);spec['training_manifest_sha256']=custody.sha256(historical/'MANIFEST.json')
        self.json_change(historical,'COMPLETE.json',lambda d:d.update(manifest_sha256=spec['training_manifest_sha256']))
        self.json_change(self.train,'metadata.json',lambda d:d.update(spec=spec))
        self.reseal(self.train)
        with self.assertRaisesRegex(ValueError,'PCA'):
            self.audit_train(spec=spec)


    def test_independent_dataset_hash_after_identity_rebind(self):
        historical=self.test/'historical_heldout'
        self.npz_change(historical,'test-identity.npz',lambda d:d['test_x'].__setitem__((0,63),0.25))
        names=['test-identity.npz','metadata.json']
        (historical/'MANIFEST.json').write_text(json.dumps(custody.manifest(historical,names)))
        spec=copy.deepcopy(self.spec);spec['heldout_manifest_sha256']=custody.sha256(historical/'MANIFEST.json')
        self.json_change(historical,'COMPLETE.json',lambda d:d.update(manifest_sha256=spec['heldout_manifest_sha256']))
        self.json_change(self.train,'metadata.json',lambda d:d.update(spec=spec))
        self.reseal(self.train)
        training_hash=custody.sha256(self.train/'MANIFEST.json')
        def metadata(d):
            d.update(spec=spec,training_manifest_sha256=training_hash)
            if 'training_manifest_sha256' in d.get('clearance',{}):
                d['clearance']['training_manifest_sha256']=training_hash
        self.json_change(self.test,'metadata.json',metadata)
        self.reseal(self.test)
        with self.assertRaisesRegex(ValueError,'Dataset identity'):
            audit.audit_heldout(self.test,self.train,spec,self.context['source_manifest_sha256'])

    def test_independent_invalid_proportion_domain(self):
        self.json_change(self.train,'summary.json',lambda d:d['rows'][0].update(accuracy=1+5e-10))
        self.reject_train('accuracy|bound|domain')

    def test_independent_negative_equation_domain(self):
        self.json_change(self.train,'summary.json',lambda d:d['fit_rows'][0].update(normal_equation_relative_max_abs=-1e-12))
        self.reject_train('equation|bound|domain')


if __name__=='__main__':
    names=[n for n in dir(IndependentAuditTests) if n.startswith('test_independent_')]
    suite=unittest.TestSuite(IndependentAuditTests(n) for n in names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
