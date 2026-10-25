"""Real Linux subprocess tests; no study data, downloads, or external services."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if importlib.util.find_spec("watchdog") is not None:
    import watchdog
else:
    watchdog = None


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc")
class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(watchdog, "watchdog implementation is not present")
        self.temp = tempfile.TemporaryDirectory(prefix="watchdog-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run"
        self.out.mkdir()
        self.limits = dict(wall_seconds=3, rss_bytes=128 * 1024 * 1024,
                           output_bytes=1024 * 1024, poll_seconds=0.01)

    def run_child(self, code, **limits):
        return watchdog.supervise([sys.executable, "-c", code], self.out,
                                  dict(self.limits, **limits))

    def receipt(self):
        return json.loads((self.out / "WATCHDOG.json").read_text())

    def assert_terminal_failure(self, result, status):
        self.assertEqual(result["status"], status)
        self.assertFalse(result["success"])
        self.assertEqual(result, self.receipt())
        self.assertFalse((self.out / "ACCEPTED.json").exists())
        self.assertFalse((self.out / "MANIFEST.json").exists())

    def test_success_captures_logs_and_counts_receipt(self):
        # Break caught: zero-exit success without terminal metrics or output accounting.
        result = self.run_child("import sys,time; print('hello'); print('error log', file=sys.stderr); time.sleep(0.05)")
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result, self.receipt())
        self.assertGreater(result["elapsed_seconds"], 0)
        self.assertGreater(result["peak_sampled_rss_bytes"], 0)
        self.assertEqual((self.out / "stdout.log").read_text(), "hello\n")
        self.assertEqual((self.out / "stderr.log").read_text(), "error log\n")
        self.assertEqual(result["output_bytes"], sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()))
        self.assertIn("sampled", result["accounting_limitation"])

    def test_nonzero_exit_is_retained_failure(self):
        # Break caught: accepting a process that exits unsuccessfully.
        result = self.run_child("import sys; print('partial'); sys.exit(7)")
        self.assert_terminal_failure(result, "exit_failure")
        self.assertEqual(result["returncode"], 7)
        self.assertEqual((self.out / "stdout.log").read_text(), "partial\n")

    def test_wall_timeout_kills_child(self):
        # Break caught: trusting the child to police its own wall limit.
        result = self.run_child("import time; time.sleep(30)", wall_seconds=0.15)
        self.assert_terminal_failure(result, "wall_limit")
        self.assertLess(result["elapsed_seconds"], 2)
        self.assertEqual(result["returncode"], -signal.SIGKILL)

    def test_rss_overrun_kills_child(self):
        # Break caught: failing to sample resident allocations.
        result = self.run_child("import time; x=bytearray(64*1024*1024); time.sleep(30)",
                                rss_bytes=32 * 1024 * 1024)
        self.assert_terminal_failure(result, "rss_limit")
        self.assertGreater(result["peak_sampled_rss_bytes"], 32 * 1024 * 1024)
        self.assertEqual(result["returncode"], -signal.SIGKILL)

    def test_output_file_overrun_kills_child(self):
        # Break caught: checking logs but not other files in the output tree.
        result = self.run_child("from pathlib import Path; import time; Path('artifact').write_bytes(b'x'*70000); time.sleep(30)",
                                output_bytes=32768)
        self.assert_terminal_failure(result, "output_limit")
        self.assertGreater(result["output_bytes"], 32768)

    def test_fast_exit_cannot_hide_output_overrun(self):
        # Break caught: checking returncode before a final output scan.
        result = self.run_child("from pathlib import Path; Path('artifact').write_bytes(b'x'*70000)",
                                output_bytes=32768)
        self.assert_terminal_failure(result, "output_limit")

    def test_stdout_and_stderr_are_counted_without_pipes(self):
        # Break caught: unaccounted output streams or a blocked pipe on large output.
        result = self.run_child("import os,time; os.write(1,b'o'*100000); os.write(2,b'e'*100000); time.sleep(30)",
                                output_bytes=32768)
        self.assert_terminal_failure(result, "output_limit")
        self.assertGreater(result["output_bytes"], 32768)
        self.assertGreater((self.out / "stdout.log").stat().st_size, 0)

    def test_grandchild_is_killed_on_wall_timeout(self):
        # Break caught: killing only the direct child leaves descendants alive.
        code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); Path('grandchild.pid').write_text(str(p.pid)); time.sleep(30)"
        result = self.run_child(code, wall_seconds=0.3)
        self.assert_terminal_failure(result, "wall_limit")
        pid = int((self.out / "grandchild.pid").read_text())
        self.assert_process_terminated(pid)

    def test_grandchild_rss_counts_toward_tree_budget(self):
        # Break caught: RSS accounting only includes the direct child.
        code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; x=bytearray(64*1024*1024); time.sleep(30)']); Path('grandchild.pid').write_text(str(p.pid)); time.sleep(30)"
        result = self.run_child(code, rss_bytes=48 * 1024 * 1024)
        self.assert_terminal_failure(result, "rss_limit")
        self.assertGreater(result["peak_sampled_rss_bytes"], 48 * 1024 * 1024)
        self.assert_process_terminated(int((self.out / "grandchild.pid").read_text()))

    def test_unfinished_descendants_block_success(self):
        # Break caught: a successful parent leaks a still-running child.
        code = "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); Path('grandchild.pid').write_text(str(p.pid))"
        result = self.run_child(code)
        self.assert_terminal_failure(result, "watchdog_error")
        self.assert_process_terminated(int((self.out / "grandchild.pid").read_text()))

    def assert_process_terminated(self, pid):
        path = Path('/proc') / str(pid) / 'stat'
        for _ in range(100):
            try:
                state = path.read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                return
            if state in ('Z', 'X'):
                return
            time.sleep(0.01)
        self.fail(f"process {pid} still live")

    def test_existing_receipt_is_not_overwritten_or_relaunched(self):
        # Break caught: retrying a run or overwriting evidence.
        marker = self.out / "WATCHDOG.json"
        marker.write_text('historical receipt')
        with self.assertRaises(FileExistsError):
            self.run_child("from pathlib import Path; Path('restarted').write_text('bad')")
        self.assertEqual(marker.read_text(), 'historical receipt')
        self.assertFalse((self.out / 'restarted').exists())

    def test_nonfresh_output_directory_is_rejected(self):
        # Break caught: contaminating fresh-output accounting with an earlier run.
        (self.out / 'earlier').write_text('data')
        with self.assertRaises(FileExistsError):
            self.run_child("pass")
        self.assertFalse((self.out / 'WATCHDOG.json').exists())

    def test_missing_command_is_terminal_launch_failure(self):
        # Break caught: treating failed execution as success or losing the receipt.
        result = watchdog.supervise([str(self.out / "no-such-command")], self.out, self.limits)
        self.assert_terminal_failure(result, "launch_failure")
        self.assertIsNone(result["returncode"])

    def test_receipt_bytes_cannot_turn_over_budget_run_into_success(self):
        # Break caught: counting child artifacts but forgetting watchdog metadata.
        result = self.run_child("pass", output_bytes=128)
        self.assert_terminal_failure(result, "output_limit")
        self.assertEqual(result['returncode'], 0)
        self.assertGreater(result['output_bytes'], 128)
        self.assertEqual(result['output_bytes'], sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()))

    def test_custom_logs_and_environment(self):
        # Break caught: losing explicit environment or allowing untracked custom logs.
        result = watchdog.supervise([sys.executable, '-c', "import os,sys; print(os.environ['WD_TEST']); print('err',file=sys.stderr)"], self.out,
                                    dict(self.limits, stdout_name='logs/out.txt', stderr_name='logs/err.txt'),
                                    env=dict(os.environ, WD_TEST='seen'))
        self.assertEqual(result['status'], 'success')
        self.assertEqual((self.out / 'logs/out.txt').read_text(), 'seen\n')
        self.assertEqual((self.out / 'logs/err.txt').read_text(), 'err\n')

    def test_protocol_maxima_cannot_be_relaxed(self):
        # Break caught: callers silently raising frozen protocol ceilings.
        for changes in ({'wall_seconds': 601}, {'rss_bytes': 1024**3+1},
                        {'output_bytes': 128*1024**2+1}, {'wall_seconds': float('nan')},
                        {'poll_seconds': 0}, {'rss_bytes': -1}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                watchdog.supervise([sys.executable, '-c', 'pass'], self.out, dict(self.limits, **changes))
        self.assertEqual(list(self.out.iterdir()), [])

    def test_log_paths_cannot_escape_or_replace_receipt(self):
        # Break caught: writing logs outside the accounting boundary or over evidence.
        for name in ('../escape.log', '/tmp/escape.log', 'WATCHDOG.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                watchdog.supervise([sys.executable, '-c', 'pass'], self.out, dict(self.limits, stdout_name=name))
        self.assertEqual(list(self.out.iterdir()), [])

    def test_symlink_output_fails_closed(self):
        # Break caught: following output symlinks outside the output tree or ignoring them.
        result = self.run_child("import os,time; os.symlink('/dev/null','escape'); time.sleep(30)")
        self.assert_terminal_failure(result, "watchdog_error")


if __name__ == '__main__':
    unittest.main()
