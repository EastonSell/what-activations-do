"""Read-only public-package checks; no fitting, real arrays, or downloads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'verify_ridge_publication.py'
PACKAGE = ROOT / 'studies/ridge-readout-recovery-20261025'


class RidgePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.package = self.work / 'relocated study'
        shutil.copytree(PACKAGE, self.package)

    def run_check(self, package=None, *extra):
        command = [sys.executable, '-I', '-S', '-B', str(SCRIPT)]
        if package is not None:
            command.extend(['--package', str(package)])
        return subprocess.run(command + list(extra), cwd=self.work,
                              capture_output=True, text=True, timeout=10)

    def rejected(self, package=None):
        run = self.run_check(self.package if package is None else package)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertEqual(run.stderr, '')
        self.assertEqual(json.loads(run.stdout)['status'], 'failed')
        self.assertNotIn('verified_files', json.loads(run.stdout))
        return json.loads(run.stdout)

    def test_relocated_package_passes_without_site_packages(self):
        run = self.run_check(self.package)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report['status'], 'public_package_integrity_verified')
        self.assertEqual(report['verified_files'], 63)
        self.assertEqual(report['manifest_sha256'],
                         '81d0d1616f57142705b5bffa5713302feab6433a98cee53e538d83f6992ae0fc')
        self.assertEqual(report['checks_performed'], ['pinned_manifest', 'exact_file_coverage', 'file_sizes_and_sha256'])
        self.assertFalse(report['scientific_reproduction_performed'])
        self.assertFalse(report['synthetic_tests_run'])
        self.assertFalse(report['full_array_audit_performed'])

    def test_default_package_is_relative_to_script_not_working_directory(self):
        run = self.run_check()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(json.loads(run.stdout)['verified_files'], 63)

    def test_success_does_not_change_files_or_create_bytecode(self):
        before = {p.relative_to(self.package): p.read_bytes()
                  for p in self.package.rglob('*') if p.is_file()}
        self.assertEqual(self.run_check(self.package).returncode, 0)
        after = {p.relative_to(self.package): p.read_bytes()
                 for p in self.package.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(list(self.package.rglob('__pycache__')), [])

    def test_every_frozen_file_is_checked(self):
        for path in sorted(self.package.rglob('*')):
            if not path.is_file():
                continue
            with self.subTest(path=path.relative_to(self.package)):
                original = path.read_bytes()
                self.assertTrue(original, 'Frozen payload files are nonempty')
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                try:
                    self.rejected()
                finally:
                    path.write_bytes(original)

    def test_larger_and_truncated_files_are_rejected(self):
        path = self.package / 'README.md'
        original = path.read_bytes()
        for content in (original + b'\n', original[:-1]):
            path.write_bytes(content)
            self.rejected()

    def test_optimized_python_does_not_disable_verification(self):
        command = [sys.executable, '-I', '-S', '-B', '-O', str(SCRIPT),
                   '--package', str(self.package)]
        passed = subprocess.run(command, cwd=self.work, capture_output=True, text=True, timeout=10)
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        path = self.package / 'report/RESULTS.md'
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        failed = subprocess.run(command, cwd=self.work, capture_output=True, text=True, timeout=10)
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        self.assertEqual(json.loads(failed.stdout)['status'], 'failed')

    def test_missing_file_is_rejected(self):
        (self.package / 'report/RESULTS.json').unlink()
        self.rejected()

    def test_unlisted_nested_file_is_rejected(self):
        (self.package / 'source/.unexpected').write_text('not frozen')
        self.rejected()

    def test_case_changed_filename_is_not_exact_coverage(self):
        path = self.package / 'README.md'
        path.rename(self.package / 'readme.md')
        self.rejected()

    def test_resealing_changed_payload_cannot_replace_pinned_identity(self):
        changed = self.package / 'README.md'
        changed.write_text('Different claims.\n')
        path = self.package / 'PACKAGE_MANIFEST.json'
        manifest = json.loads(path.read_text())
        manifest['files']['README.md'] = {
            'bytes': changed.stat().st_size,
            'sha256': hashlib.sha256(changed.read_bytes()).hexdigest(),
        }
        path.write_text(json.dumps(manifest))
        self.assertIn('manifest identity', self.rejected()['error'].lower())

    def test_tampered_package_code_is_never_imported(self):
        marker = self.work / 'executed.txt'
        (self.package / 'source/custody.py').write_text(
            f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n')
        self.rejected()
        self.assertFalse(marker.exists())

    def test_missing_or_file_root_reports_failure(self):
        self.rejected(self.work / 'absent')
        self.rejected(self.package / 'README.md')

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_symlink_root_is_rejected(self):
        link = self.work / 'alias'
        link.symlink_to(self.package, target_is_directory=True)
        self.rejected(link)

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_matching_symlink_file_is_rejected(self):
        path = self.package / 'README.md'
        external = self.work / 'original-readme'
        path.rename(external)
        path.symlink_to(external)
        self.rejected()

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_symlink_directory_is_rejected(self):
        external = self.work / 'original-report'
        (self.package / 'report').rename(external)
        (self.package / 'report').symlink_to(external, target_is_directory=True)
        self.rejected()

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symlinks unavailable')
    def test_unlisted_dangling_symlink_is_rejected(self):
        (self.package / 'dangling').symlink_to(self.work / 'absent')
        self.rejected()

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'FIFO unavailable')
    def test_nonregular_file_is_rejected_without_blocking(self):
        path = self.package / 'report/RESULTS.json'
        path.unlink()
        os.mkfifo(path)
        self.rejected()

    def test_no_cli_hash_override_can_bless_another_package(self):
        run = self.run_check(self.package, '--expected-hash', '0' * 64)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(run.stdout, '')


if __name__ == '__main__':
    unittest.main()
