"""Manufactured metadata-only receipts; no real artifact, checkpoint or dataset."""
from pathlib import Path
import unittest
from unittest.mock import patch
import audit
import settings

class AuditClearanceTests(unittest.TestCase):
    def setUp(self):
        self.spec = settings.production_spec()
        self.source = 'a'*64
    def check(self, which, clearance):
        metadata = {'runtime': {'numpy': '2.3.5'}, 'clearance': clearance,
                    'training_manifest_sha256': 'b'*64}
        receipts = {'STAGE_RESOURCES.json': {'elapsed_before_sealing_seconds': .1,
                    'bytes_before_sealing': 10, 'peak_sampled_rss_bytes': 1000},
                    'WATCHDOG.json': {'peak_sampled_rss_bytes': 1000},
                    'ACCESS_LOG.json': {'operations': [{'operation': 'synthetic_test_only'}]}}
        with patch('settings.production_spec', return_value=self.spec), \
             patch('stageio.verify_completed', return_value=metadata), \
             patch('custody.read_json', side_effect=lambda path: receipts[path.name]):
            return audit.stage(Path('/manufactured_metadata_only'), which, self.spec, self.source)
    def good(self, which):
        return {'accepted': True, 'stage': which, 'source_manifest_sha256': self.source,
                'protocol_manifest_sha256': settings.PROTOCOL_HASH,
                'review_reference': 'Manufactured receipt, no permission',
                **({'training_manifest_sha256': 'b'*64} if which == 'heldout' else {})}
    def test_correct_clearance_bindings(self):
        for which in ('training', 'heldout'):
            self.check(which, self.good(which))
    def test_contradictory_execution_clearance_rejected(self):
        for which in ('training', 'heldout'):
            good = self.good(which)
            changes = [{'accepted': True}, dict(good, stage='wrong'),
                       dict(good, source_manifest_sha256='c'*64),
                       dict(good, protocol_manifest_sha256='d'*64),
                       dict(good, review_reference='')]
            if which == 'heldout': changes.append(dict(good, training_manifest_sha256='e'*64))
            for bad in changes:
                with self.subTest(which=which, clearance=bad), self.assertRaises(ValueError):
                    self.check(which, bad)

if __name__=='__main__': unittest.main()
