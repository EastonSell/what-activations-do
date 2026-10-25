from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import sys
import custody
import settings
import runner
from fixtures import fixture_archives,synthetic_context,authorize_heldout_fixture


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_actual_subprocess_synthetic_training_and_heldout(self):
        ta,ha,spec=fixture_archives(self.root); context=synthetic_context()
        payload=self.root/'fixture.json'; custody.write_json(payload,{'spec':spec,'context':context})
        helper=self.root/'synthetic_child.py'
        helper.write_text('import sys,json,pathlib,pipeline\n'
          'p=json.loads(pathlib.Path(sys.argv[1]).read_text()); stage=sys.argv[2]; out=pathlib.Path(sys.argv[3])\n'
          'r=pipeline.train_stage(sys.argv[4],out,p["spec"],p["context"]) if stage=="training" else pipeline.heldout_stage(sys.argv[4],sys.argv[5],out,p["spec"],p["context"])\n'
          'sys.exit(0 if stage=="heldout" or r["gates"]["eligible"] else 2)\n')
        training=self.root/'training'
        command=[sys.executable,'-B',str(helper),str(payload),'training',str(training),str(ta)]
        result=runner.run_command(command,training,'training',spec,context['source_manifest_sha256'])
        self.assertEqual(result['status'],'execution_complete_pending_independent_audit')
        self.assertTrue((training/'COMPLETE.json').is_file())
        authorize_heldout_fixture(training,spec,context)
        payload.unlink()
        custody.write_json(payload,{'spec':spec,'context':context})
        test=self.root/'heldout'
        command=[sys.executable,'-B',str(helper),str(payload),'heldout',str(test),str(ha),str(training)]
        result=runner.run_command(command,test,'heldout',spec,context['source_manifest_sha256'])
        self.assertEqual(result['status'],'execution_complete_pending_independent_audit')
        self.assertEqual(len(custody.read_json(test/'summary.json')['rows']),100)
    def test_nonzero_child_has_no_completion(self):
        spec=settings.production_spec(); out=self.root/'out'
        result=runner.run_command([sys.executable,'-c','raise SystemExit(7)'],out,'training',spec,'a'*64)
        self.assertNotEqual(result['status'],'execution_complete_pending_independent_audit')
        self.assertFalse((out/'COMPLETE.json').exists())
    def test_output_reuse_refused(self):
        out=self.root/'out'; out.mkdir()
        with self.assertRaises(FileExistsError): runner.run_command([sys.executable,'-c','pass'],out,'training',settings.production_spec(),'a'*64)
    def test_source_file_change_rejected(self):
        copied=self.root/'source'
        shutil.copytree(settings.ROOT,copied,ignore=shutil.ignore_patterns('__pycache__','SOURCE_MANIFEST.json'))
        # copytree ignore also excluded the reviewed nested manifest; restore it.
        shutil.copy2(settings.ROOT/'protocol_v2/SOURCE_MANIFEST.json',copied/'protocol_v2/SOURCE_MANIFEST.json')
        with patch.object(settings,'ROOT',copied):
            custody.write_json(copied/'SOURCE_MANIFEST.json',custody.manifest(copied,settings.source_paths()))
            settings.verify_source()
            with open(copied/'pipeline.py','a') as f: f.write('\n# changed after freeze\n')
            with self.assertRaises(ValueError): settings.verify_source()


if __name__=='__main__': unittest.main()
