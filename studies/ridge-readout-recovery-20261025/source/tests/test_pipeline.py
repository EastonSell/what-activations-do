from pathlib import Path
import tempfile
import unittest
import numpy as np
import custody
import pipeline
from fixtures import fixture_archives,synthetic_context,seal_fixture


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.ta,self.ha,self.spec=fixture_archives(self.root)
        self.context=synthetic_context()
    def tearDown(self): self.tmp.cleanup()
    def train(self):
        out=self.root/'training'; out.mkdir()
        result=pipeline.train_stage(self.ta,out,self.spec,self.context)
        return out,result
    def test_synthetic_train_and_heldout_full_denominator(self):
        out,result=self.train()
        self.assertTrue(result['gates']['eligible'])
        self.assertEqual(result['readout_count'],90)
        self.assertFalse((out/'historical_training/test-identity.npz').exists())
        seal_fixture(out,'training',self.spec,self.context)
        test=self.root/'heldout'; test.mkdir()
        got=pipeline.heldout_stage(self.ha,out,test,self.spec,self.context)
        self.assertEqual(len(got['rows']),100)
        self.assertEqual(got['cases'],360)
        self.assertIn('case_resampling_interval95_pp',got['primary'])
    def test_unequal_clean_readout_gate_kept_as_failure(self):
        other=self.root/'bad'; other.mkdir(); ta,ha,spec=fixture_archives(other,True)
        out=self.root/'training'; out.mkdir()
        result=pipeline.train_stage(ta,out,spec,self.context)
        self.assertFalse(result['gates']['eligible'])
        self.assertEqual(result['readout_count'],90)
        test=self.root/'heldout'; test.mkdir()
        with self.assertRaises(ValueError): pipeline.heldout_stage(ha,out,test,spec,self.context)
        self.assertFalse((test/'historical_heldout').exists())
    def test_source_identity_rejected_before_heldout_access(self):
        out,_=self.train(); seal_fixture(out,'training',self.spec,self.context)
        ctx=dict(self.context,source_manifest_sha256='b'*64)
        test=self.root/'heldout'; test.mkdir()
        with self.assertRaises(ValueError): pipeline.heldout_stage(self.root/'DO_NOT_READ',out,test,self.spec,ctx)
        self.assertFalse((test/'historical_heldout').exists())
    def test_checkpoint_mutation_rejected(self):
        out,_=self.train(); seal_fixture(out,'training',self.spec,self.context)
        with open(out/'fitted-101.npz','ab') as f: f.write(b'changed')
        test=self.root/'heldout'; test.mkdir()
        with self.assertRaises(ValueError): pipeline.heldout_stage(self.ha,out,test,self.spec,self.context)
    def test_bad_archive_hash_does_not_import(self):
        out=self.root/'training'; out.mkdir(); spec=dict(self.spec,training_archive_sha256='0'*64)
        with self.assertRaises(ValueError): pipeline.train_stage(self.ta,out,spec,self.context)
        self.assertFalse((out/'historical_training').exists())
    def test_diagnostics_keep_integer_counts_and_rank(self):
        out,result=self.train()
        for row in result['rows']:
            d=row['support']; self.assertIs(type(d['strict_negative_coordinate_count']),int)
            self.assertLessEqual(d['strict_any_negative_case_count'],d['case_denominator'])
        for row in result['fit_rows']:
            self.assertLessEqual(row['actual_train_rank'],row['maximum_feature_subspace_dimension'])


if __name__=='__main__': unittest.main()
