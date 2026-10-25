"""Synthetic reviewer probes; never loads real archives, models, or datasets."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

RUNNER=Path(__file__).resolve().parent.parent/'source'
sys.path[:0]=[str(RUNNER),str(RUNNER/'tests')]
import custody, pipeline, settings, stageio
from fixtures import fixture_archives, synthetic_context, seal_fixture

class CustodyReview(unittest.TestCase):
    def test_changed_training_manifest_rejected_before_test_read(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ta,ha,spec=fixture_archives(root)
            context=synthetic_context(); train=root/'train'; train.mkdir()
            pipeline.train_stage(ta,train,spec,context); seal_fixture(train,'training',spec,context)
            approved=custody.sha256(train/'MANIFEST.json')
            context['clearance']={'accepted':True,'stage':'heldout','training_manifest_sha256':approved,
                'source_manifest_sha256':context['source_manifest_sha256'],
                'protocol_manifest_sha256':settings.PROTOCOL_HASH,'review_reference':'synthetic review only'}
            # Simulate a different independently complete artifact replacing the one
            # whose manifest was admitted by the worker; no scientific values change.
            (train/'reviewer-change.txt').write_text('different artifact, not cleared\n')
            for name in ['MANIFEST.json','COMPLETE.json','STAGE_RESOURCES.json']:
                (train/name).unlink()
            stageio.finalize(train,'training',spec,context['source_manifest_sha256'],time.monotonic())
            assert custody.sha256(train/'MANIFEST.json')!=approved
            out=root/'heldout'; out.mkdir(); custody.begin_access_log()
            with self.assertRaises(ValueError):
                pipeline.heldout_stage(ha,train,out,spec,context)
            self.assertFalse(any(r['path']==str(ha.resolve()) for r in custody.access_log()))

    def test_midread_training_file_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ta,ha,spec=fixture_archives(root)
            context=synthetic_context(); train=root/'train'; train.mkdir()
            pipeline.train_stage(ta,train,spec,context); seal_fixture(train,'training',spec,context)
            approved=custody.sha256(train/'MANIFEST.json')
            context['clearance']={'accepted':True,'stage':'heldout','training_manifest_sha256':approved,
                'source_manifest_sha256':context['source_manifest_sha256'],
                'protocol_manifest_sha256':settings.PROTOCOL_HASH,'review_reference':'synthetic review only'}
            target=train/'fitted-101.npz'; original=pipeline.load_npz; changed=[False]
            def mutate_before_load(path,keys=None):
                if Path(path)==target and not changed[0]:
                    arrays=original(path,keys)
                    arrays['intercepts'][0,0]+=0.5
                    target.unlink(); custody.write_npz(target,**arrays); changed[0]=True
                return original(path,keys)
            out=root/'heldout'; out.mkdir()
            with patch.object(pipeline,'load_npz',mutate_before_load):
                with self.assertRaises(ValueError): pipeline.heldout_stage(ha,train,out,spec,context)
            self.assertTrue(changed[0])
            self.assertEqual(custody.sha256(train/'MANIFEST.json'),approved)

    def test_inconsistent_watchdog_failure_evidence_rejected(self):
        spec=settings.production_spec()
        receipt={'status':'success','success':True,'returncode':0,'violations':['output_limit'],
          'elapsed_seconds':1.,'peak_sampled_rss_bytes':1000,'output_bytes':1000,
          'peak_sampled_output_bytes':spec['config']['max_output_bytes_per_stage']+1,
          'limits':{'wall_seconds':600,'rss_bytes':1073741824,'output_bytes':134217728}}
        with self.assertRaises(ValueError): stageio.verify_watchdog(receipt,spec)

    def test_inconsistent_watchdog_error_evidence_rejected(self):
        spec=settings.production_spec()
        receipt={'status':'success','success':True,'returncode':0,'violations':[],
          'errors':['synthetic accounting failure'],'elapsed_seconds':1.,
          'peak_sampled_rss_bytes':1000,'output_bytes':1000,'peak_sampled_output_bytes':1000,
          'limits':{'wall_seconds':600,'rss_bytes':1073741824,'output_bytes':134217728}}
        with self.assertRaises(ValueError): stageio.verify_watchdog(receipt,spec)

    def test_negative_sealed_elapsed_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ta,ha,spec=fixture_archives(root)
            context=synthetic_context(); out=root/'train'; out.mkdir()
            pipeline.train_stage(ta,out,spec,context); seal_fixture(out,'training',spec,context)
            resources=custody.read_json(out/'STAGE_RESOURCES.json')
            resources['elapsed_before_sealing_seconds']=-1.
            (out/'STAGE_RESOURCES.json').write_text(json.dumps(resources))
            for name in ['MANIFEST.json','COMPLETE.json']: (out/name).unlink()
            custody.write_json(out/'MANIFEST.json',custody.manifest(out,custody.files_under(out)))
            custody.write_json(out/'COMPLETE.json',{'manifest_sha256':custody.sha256(out/'MANIFEST.json'),
              'status':'execution_complete_pending_independent_audit'})
            with self.assertRaises(ValueError):
                stageio.verify_completed(out,'training',spec,context['source_manifest_sha256'])

    def test_terminal_verification_included_in_wall_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); ta,ha,spec=fixture_archives(root)
            context=synthetic_context(); out=root/'train'; out.mkdir()
            pipeline.train_stage(ta,out,spec,context)
            custody.write_json(out/'WATCHDOG.json',{'status':'success','success':True,'returncode':0,
              'elapsed_seconds':.1,'peak_sampled_rss_bytes':1000,'output_bytes':1000,
              'peak_sampled_output_bytes':1000,'violations':[],'errors':[],
              'limits':{'wall_seconds':600,'rss_bytes':1073741824,'output_bytes':134217728}})
            now=[100.0]; original=custody.verify_manifest
            def delayed_verification(*args,**kwargs):
                result=original(*args,**kwargs)
                now[0]+=601.0
                return result
            with patch.object(stageio.time,'monotonic',lambda:now[0]), patch.object(custody,'verify_manifest',delayed_verification):
                with self.assertRaises(ValueError):
                    stageio.finalize(out,'training',spec,context['source_manifest_sha256'],100.)

if __name__=='__main__': unittest.main(verbosity=2)
