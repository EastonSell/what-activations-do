import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import storage
import audit

class RunnerTests(unittest.TestCase):
    def test_real_data_cli_refuses_without_stage_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'never-created'
            p=subprocess.run([sys.executable,'study.py','--stage','training','--out',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertFalse(out.exists())
            self.assertIn('--clearance',p.stderr)
    def test_independent_rescorer_matches_hand_computed_losses(self):
        import numpy as np
        z=np.array([[0.,0.],[0.,np.log(3.)]])
        a=audit.rescore(z,np.array([0,1]))
        np.testing.assert_allclose(a['ce'],[np.log(2.),-np.log(.75)],atol=1e-12)
        np.testing.assert_allclose(a['margin'],[0.,np.log(3.)],atol=1e-12)
    def test_completion_requires_watchdog_and_matching_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); storage.write_json(root/'x.json',{'value':1})
            storage.write_json(root/'MANIFEST.json',storage.manifest(root,['x.json']))
            storage.write_json(root/'COMPLETE.json',{'manifest_sha256':storage.sha256(root/'MANIFEST.json')})
            with self.assertRaises(ValueError): audit.accepted(root)
            storage.write_json(root/'WATCHDOG.json',{'status':'completed','returncode':0,'elapsed_seconds':1.,'limit_seconds':600})
            audit.accepted(root)
            (root/'x.json').write_text('tampered')
            with self.assertRaises(ValueError): audit.accepted(root)

if __name__=='__main__': unittest.main()

class WatchdogTests(unittest.TestCase):
    def test_watchdog_stops_its_own_overdue_synthetic_child(self):
        import run_stage
        report=run_stage.supervise([sys.executable,'-c','import time; time.sleep(5)'],.05)
        self.assertEqual(report['status'],'timeout')
        self.assertLess(report['elapsed_seconds'],3.)
        self.assertNotEqual(report['returncode'],0)
