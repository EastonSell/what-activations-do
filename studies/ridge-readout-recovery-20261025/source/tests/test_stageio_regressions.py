"""Safety regressions use synthetic arrays only, never production artifacts."""
import copy
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import custody
import pipeline
import stageio
from fixtures import fixture_archives, synthetic_context, seal_fixture


class StageSafetyTests(unittest.TestCase):
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
        cls.context['clearance'] = {
            'accepted': True, 'stage': 'heldout',
            'source_manifest_sha256': cls.context['source_manifest_sha256'],
            'protocol_manifest_sha256': cls.spec['protocol_manifest_sha256'],
            'training_manifest_sha256': custody.sha256(cls.training / 'MANIFEST.json'),
            'review_reference': 'Synthetic regression only, not scientific acceptance'}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))
        self.train = self.case / 'training'
        shutil.copytree(self.training, self.train)
        self.out = self.case / 'heldout'
        self.out.mkdir()
        self.context = copy.deepcopy(type(self).context)

    def tearDown(self):
        shutil.rmtree(self.case)

    def heldout(self):
        return pipeline.heldout_stage(self.ha, self.train, self.out, self.spec, self.context)

    def assert_no_test_access(self):
        self.assertFalse(any(row['path'] == str(self.ha.resolve()) for row in custody.access_log()))
        self.assertFalse((self.out / 'historical_heldout').exists())

    def test_missing_clearance_hash_rejected_before_heldout_access(self):
        del self.context['clearance']['training_manifest_sha256']
        custody.begin_access_log()
        with self.assertRaises(ValueError):
            self.heldout()
        self.assert_no_test_access()

    def test_resealed_training_requires_exact_clearance(self):
        (self.train / 'changed.txt').write_text('different artifact\n')
        for name in ('MANIFEST.json', 'COMPLETE.json', 'STAGE_RESOURCES.json'):
            (self.train / name).unlink()
        stageio.finalize(self.train, 'training', self.spec,
                         self.context['source_manifest_sha256'], time.monotonic())
        custody.begin_access_log()
        with self.assertRaises(ValueError):
            self.heldout()
        self.assert_no_test_access()

    def test_head_mutation_during_load_rejected_before_heldout_access(self):
        original = pipeline.load_npz
        def mutate_after_read(path, *args, **kwargs):
            data = original(path, *args, **kwargs)
            if Path(path).name == 'fitted-101.npz':
                with open(path, 'ab') as stream:
                    stream.write(b'changed after read')
            return data
        custody.begin_access_log()
        with patch.object(pipeline, 'load_npz', mutate_after_read):
            with self.assertRaises(ValueError):
                self.heldout()
        self.assert_no_test_access()

    def test_training_mutation_after_test_access_cannot_complete(self):
        original = pipeline.readout.primary
        def mutate_after_computation(*args, **kwargs):
            result = original(*args, **kwargs)
            with open(self.train / 'fitted-101.npz', 'ab') as stream:
                stream.write(b'changed while heldout was computing')
            return result
        with patch.object(pipeline.readout, 'primary', mutate_after_computation):
            with self.assertRaises(ValueError):
                self.heldout()
        self.assertFalse((self.out / 'WORKER_RESULT.json').exists())

    def receipt(self):
        return {'status': 'success', 'success': True, 'returncode': 0,
                'violations': [], 'errors': [], 'elapsed_seconds': .1,
                'peak_sampled_rss_bytes': 1000, 'output_bytes': 1000,
                'peak_sampled_output_bytes': 1000,
                'limits': {'wall_seconds': 600, 'rss_bytes': 1073741824,
                           'output_bytes': 134217728}}

    def test_watchdog_nonempty_violations_rejected(self):
        receipt = self.receipt()
        receipt['violations'] = ['output_limit']
        with self.assertRaises(ValueError):
            stageio.verify_watchdog(receipt, self.spec)

    def test_watchdog_nonempty_errors_rejected(self):
        receipt = self.receipt()
        receipt['errors'] = ['accounting failed']
        with self.assertRaises(ValueError):
            stageio.verify_watchdog(receipt, self.spec)

    def test_watchdog_peak_output_over_limit_rejected(self):
        receipt = self.receipt()
        receipt['peak_sampled_output_bytes'] = self.spec['config']['max_output_bytes_per_stage'] + 1
        with self.assertRaises(ValueError):
            stageio.verify_watchdog(receipt, self.spec)

    def test_watchdog_missing_peak_or_violations_rejected(self):
        for key in ('peak_sampled_output_bytes', 'violations', 'errors'):
            with self.subTest(key=key):
                receipt = self.receipt()
                del receipt[key]
                with self.assertRaises(ValueError):
                    stageio.verify_watchdog(receipt, self.spec)

    def test_terminal_verification_time_invalidates_completion(self):
        for name in ('MANIFEST.json', 'COMPLETE.json', 'STAGE_RESOURCES.json'):
            (self.train / name).unlink()
        now = [100.]
        original = custody.verify_manifest
        def delayed(*args, **kwargs):
            result = original(*args, **kwargs)
            now[0] += 601.
            return result
        with patch.object(stageio.time, 'monotonic', lambda: now[0]), \
             patch.object(custody, 'verify_manifest', delayed):
            with self.assertRaises(ValueError):
                stageio.finalize(self.train, 'training', self.spec,
                                 self.context['source_manifest_sha256'], 100.)
        self.assertTrue((self.train / 'FAILED_FINALIZATION.json').is_file())
        with self.assertRaises(ValueError):
            stageio.verify_completed(self.train, 'training', self.spec,
                                     self.context['source_manifest_sha256'])

    def test_invalid_elapsed_resource_evidence_rejected(self):
        for elapsed in ('-1', '-1e309', '1e309', 'true'):
            with self.subTest(elapsed=elapsed):
                path = self.train / 'STAGE_RESOURCES.json'
                path.write_text('{"elapsed_before_sealing_seconds":' + elapsed + '}')
                for name in ('MANIFEST.json', 'COMPLETE.json'):
                    (self.train / name).unlink()
                custody.write_json(self.train / 'MANIFEST.json',
                                   custody.manifest(self.train, custody.files_under(self.train)))
                custody.write_json(self.train / 'COMPLETE.json', {
                    'manifest_sha256': custody.sha256(self.train / 'MANIFEST.json'),
                    'status': 'execution_complete_pending_independent_audit'})
                with self.assertRaises(ValueError):
                    stageio.verify_completed(self.train, 'training', self.spec,
                                             self.context['source_manifest_sha256'])


if __name__ == '__main__':
    unittest.main()
