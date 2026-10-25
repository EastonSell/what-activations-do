"""Reviewer-only audit supervisor probes using the author's invented fixtures."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

RUNNER=Path(__file__).resolve().parent.parent/'source'
sys.path[:0]=[str(RUNNER),str(RUNNER/'tests')]
import custody
import audit_runner
from test_audit_runner import AuditRunnerTests

class IndependentAuditOperations(AuditRunnerTests):
    def test_budget_exhausted_before_child_does_not_launch(self):
        args,_=self.options(); real=audit_runner.verify_inputs; now=[100.]
        def delayed(*args,**kwargs):
            result=real(*args,**kwargs); now[0]+=601.; return result
        with patch.object(audit_runner.time,'monotonic',lambda:now[0]), \
             patch.object(audit_runner,'verify_inputs',delayed), \
             patch.object(audit_runner,'supervise',side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError,'resource ceiling'):
                audit_runner.run_command([sys.executable,'-c','pass'],args,self.spec,self.source_hash,
                                         source_recheck=lambda:self.source_hash)
        self.assertFalse((args.out/'AUDIT_COMPLETE.json').exists())

    def test_terminal_manifest_verification_in_budget(self):
        real=custody.verify_manifest; now=[100.]
        def delayed(root,*args,**kwargs):
            result=real(root,*args,**kwargs)
            if Path(root)==self.case/'audit': now[0]+=601.
            return result
        with patch.object(audit_runner.time,'monotonic',lambda:now[0]), \
             patch.object(custody,'verify_manifest',delayed):
            with self.assertRaisesRegex(ValueError,'resource ceiling'): self.run_fixture()
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())
        self.assertTrue((self.case/'audit/FAILED_AUDIT.json').exists())

    def test_real_input_mutation_after_child_rejected(self):
        target=self.training/'fitted-101.npz'; saved=target.read_bytes()
        def mutate():
            target.write_bytes(saved+b'reviewer-only synthetic mutation')
            return self.source_hash
        try:
            with self.assertRaisesRegex(ValueError,'hash/size mismatch'):
                self.run_fixture(recheck=mutate)
            self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())
        finally: target.write_bytes(saved)

    def test_bound_clearance_mutation_after_child_rejected(self):
        def mutate():
            p=self.case/'clearance-training.json'; value=custody.read_json(p)
            value['review_reference']='different clearance after execution'
            import json
            p.write_text(json.dumps(value))
            return self.source_hash
        with self.assertRaisesRegex(ValueError,'clearance changed'): self.run_fixture(recheck=mutate)
        self.assertFalse((self.case/'audit/AUDIT_COMPLETE.json').exists())

    def test_audit_failure_evidence_cannot_claim_success(self):
        base={'status':'success','success':True,'returncode':0,'violations':[],'errors':[],
              'limits':{'wall_seconds':600,'rss_bytes':1073741824,'output_bytes':134217728},
              'elapsed_seconds':1.,'peak_sampled_rss_bytes':1000,'output_bytes':1000,
              'peak_sampled_output_bytes':1000}
        for key,bad in [('violations',['rss_limit']),('errors',['accounting failure']),
                        ('peak_sampled_output_bytes',134217729),('elapsed_seconds',-1.),
                        ('peak_sampled_rss_bytes',float('inf'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit_runner._verify_watchdog(dict(base,**{key:bad}))

if __name__=='__main__':
    suite=unittest.TestSuite(IndependentAuditOperations(name) for name in IndependentAuditOperations.__dict__ if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
