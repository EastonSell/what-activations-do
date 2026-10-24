import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import audit_cached

SOURCE=Path(__file__).resolve().parents[1]/'source'
sys.path[:0]=[str(SOURCE),str(SOURCE/'tests')]
import audit,pipeline,storage,test_pipeline

class AdapterTests(unittest.TestCase):
    def test_cached_member_is_exact_and_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.npz'; np.savez_compressed(path,x=np.arange(100,dtype=np.float64))
            with audit_cached.cached_load():
                with np.load(path,allow_pickle=False) as f:
                    a=f['x']; b=f['x']
                    self.assertIs(a,b)
                    np.testing.assert_array_equal(a,np.arange(100,dtype=np.float64))
                    with self.assertRaises(ValueError): a[0]=999
            with np.load(path,allow_pickle=False) as f: self.assertEqual(f['x'][0],0)
    def test_exact_frozen_audit_agreement_on_fabricated_archive(self):
        fixture=test_pipeline.IndependentAuditTests(); x,y,s,cfg=fixture.data()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); tr=root/'train'; te=root/'test'; tr.mkdir(); te.mkdir()
            pipeline.train_stage(x,y,s,cfg,tr,storage.Budget(tr,100,10**12,10**8)); fixture.seal(tr,cfg,'training')
            pipeline.test_stage(x,y,s,cfg,tr,te,storage.Budget(te,100,10**12,10**8)); fixture.seal(te,cfg,'heldout')
            plain=audit.audit_test(te,tr,root/'source')
            with audit_cached.cached_load(): cached=audit.audit_test(te,tr,root/'source')
            self.assertEqual(plain,cached)

if __name__=='__main__': unittest.main()
