"""Public-package provenance tests; no study arrays, fitting, or data downloads."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
import custody
import settings

ORIGINAL_SOURCE = 'b169b28b144a511770743782e85f05ac008e09553e3845dab654a88e2604f528'
ORIGINAL_PROTOCOL = '061e722fb16c8ac0962307094b713c974658f9a049ac070f9c8f9cdfe709595a'
REPORT_HASHES = {
    'RESULTS.md': '5ae4420110292b65962febecc422bb221511c5131c09b42258adfc610be46e5e',
    'RESULTS.json': '29ecf13e124180fd32ce9ac73efe091e40ee51da0a4890b2aa6d9193185bac70',
    'ALL_CONDITION_METRICS.csv': '9a005a4043500d65826c511f1897f77d9030bfdcf0cc55fc946813b36f733569',
    'ALL_FIT_DIAGNOSTICS.csv': 'b1174cbb5205fcf820b9e145d283acf7ae0093b901002475bc7a42278988b007',
}


class PublicationTests(unittest.TestCase):
    def test_report_bytes_are_exact_reviewed_originals(self):
        for name, expected in REPORT_HASHES.items():
            self.assertEqual(custody.sha256(ROOT / 'report' / name), expected, name)

    def test_original_manifest_bytes_are_preserved(self):
        self.assertEqual(custody.sha256(ROOT / 'provenance/ORIGINAL_SOURCE_MANIFEST.json'), ORIGINAL_SOURCE)
        self.assertEqual(custody.sha256(ROOT / 'provenance/ORIGINAL_PROTOCOL_MANIFEST.json'), ORIGINAL_PROTOCOL)

    def test_public_source_freeze_has_a_new_identity(self):
        self.assertNotEqual(settings.verify_source(), ORIGINAL_SOURCE)
        self.assertNotEqual(settings.PROTOCOL_HASH, ORIGINAL_PROTOCOL)

    def reject_old_clearance(self, source_hash, protocol_hash):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'synthetic-clearance.json'
            custody.write_json(path, {
                'accepted': True, 'stage': 'training',
                'source_manifest_sha256': source_hash,
                'protocol_manifest_sha256': protocol_hash,
                'review_reference': 'Invented metadata only; never execution permission',
            })
            with self.assertRaises(ValueError):
                custody.clearance(path, 'training', settings.verify_source(), settings.PROTOCOL_HASH)

    def test_original_source_clearance_does_not_validate_derivative(self):
        self.reject_old_clearance(ORIGINAL_SOURCE, settings.PROTOCOL_HASH)

    def test_original_protocol_clearance_does_not_validate_derivative(self):
        self.reject_old_clearance(settings.verify_source(), ORIGINAL_PROTOCOL)

    def test_scientific_code_parameters_and_guards_unchanged(self):
        original = json.loads((ROOT / 'provenance/ORIGINAL_SOURCE_MANIFEST.json').read_text())
        for name, meta in original['files'].items():
            if name.endswith('.py') or name in ('historical-config.json', 'protocol_v2/config.json'):
                raw = (ROOT / 'source' / name).read_bytes()
                if name == 'settings.py':
                    raw = raw.replace(settings.PROTOCOL_HASH.encode(), ORIGINAL_PROTOCOL.encode())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta['sha256'], name)

    def test_no_private_paths_or_owner_repository_urls_in_source(self):
        pattern = re.compile(r'/(?:workspace/(?:scratch|shared)|Users|home/agent)/|https?://github[.]com/[^/\s]+/')
        for path in (ROOT / 'source').rglob('*'):
            if path.is_file() and path.suffix != '.pyc':
                self.assertIsNone(pattern.search(path.read_text()), str(path.relative_to(ROOT)))

    def test_no_array_archive_or_bytecode_payloads_shipped(self):
        for path in ROOT.rglob('*'):
            self.assertFalse(path.is_symlink(), str(path.relative_to(ROOT)))
            if path.is_file():
                self.assertNotIn(path.suffix, ('.npz', '.npy', '.tar', '.gz', '.zip', '.pyc'))


if __name__ == '__main__':
    unittest.main()
