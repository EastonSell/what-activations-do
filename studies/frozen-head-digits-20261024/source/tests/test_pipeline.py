import tempfile
import hashlib
import shutil
import unittest
from pathlib import Path
import numpy as np
import pipeline
import storage

class PipelineTests(unittest.TestCase):
    def data(self):
        x=np.array([[i%3/2,(i//3)%3/2,(i%2)/2] for i in range(30)],dtype=float)
        y=np.array([i%3 for i in range(30)])
        splits={'train':np.arange(18),'validation':np.arange(18,24),'test':np.arange(24,30)}
        cfg={'input_dim':3,'hidden_dim':4,'classes':3,'seeds':[101,211],'epochs':2,'batch_size':5,'learning_rate':.01,'alpha':.0001,'beta1':.9,'beta2':.999,'epsilon':1e-8,'competence_floor':0.,'competence_gap':1.,'random_draws':2,'ranks':[0,1,2,3,4],'primary_ranks':[1,2],'bootstrap_draws':100,'bootstrap_seed':7,'numerical_atol':1e-10}
        cfg.update({'data_sha256':hashlib.sha256((x*16.).tobytes()+y.astype(np.int64).tobytes()).hexdigest(),'wall_seconds':100,'rss_bytes':10**12,'disk_bytes':10**8,'threads':2})
        return x,y,splits,cfg
    def test_training_artifact_has_no_test_logits_and_fixed_hidden(self):
        x,y,s,cfg=self.data()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); b=storage.Budget(root,100,10**12,10**8)
            summary=pipeline.train_stage(x,y,s,cfg,root,b)
            self.assertTrue(summary['gates']['primary_pass'])
            self.assertEqual(summary['fit_count'],4)
            self.assertFalse(any('test' in p.name for p in root.glob('*.npz')))
            with np.load(root/'model-101-B.npz',allow_pickle=False) as p:
                np.testing.assert_array_equal(p['W1'],p['initial_W1'])
                self.assertEqual(p['validation_logits'].shape,(6,3))
    def test_synthetic_two_stage_outputs_all_cells(self):
        x,y,s,cfg=self.data()
        with tempfile.TemporaryDirectory() as d:
            train=Path(d)/'train'; test=Path(d)/'test'; train.mkdir(); test.mkdir()
            pipeline.train_stage(x,y,s,cfg,train,storage.Budget(train,100,10**12,10**8))
            summary=pipeline.test_stage(x,y,s,cfg,train,test,storage.Budget(test,100,10**12,10**8))
            self.assertEqual(summary['model_arm_count'],4)
            with np.load(test/'test-101-A.npz',allow_pickle=False) as p:
                self.assertEqual(p['logits'].shape,(5,5,6,3))
                for c in range(5): np.testing.assert_allclose(p['logits'][0,c],p['baseline_logits'],atol=1e-12)
                self.assertEqual(p['negative_fraction'].shape,(5,5,6))
    def test_split_overlap_and_missing_example_rejected(self):
        _,_,s,_=self.data()
        pipeline.validate_splits(s,30)
        bad={**s,'test':np.arange(23,30)}
        with self.assertRaises(ValueError): pipeline.validate_splits(bad,30)
        bad={**s,'test':np.arange(24,29)}
        with self.assertRaises(ValueError): pipeline.validate_splits(bad,30)

if __name__=='__main__': unittest.main()

class IndependentAuditTests(unittest.TestCase):
    data=PipelineTests.data
    def seal(self,root,config,stage):
        import audit
        source=root.parent/'source'
        if not source.exists():
            source.mkdir(); (source/'tests').mkdir()
            production=Path(__file__).resolve().parents[1]
            names=audit.SOURCE_BASE+sorted(str(p.relative_to(production)) for p in (production/'tests').glob('*.py'))
            for name in names: shutil.copyfile(production/name,source/name)
            (source/'config.json').write_text(__import__('json').dumps(config))
            storage.write_json(source/'SOURCE_MANIFEST.json',storage.manifest(source,names))
        source_hash=storage.sha256(source/'SOURCE_MANIFEST.json')
        training_hash=storage.sha256(root.parent/'train'/'MANIFEST.json') if stage=='heldout' else None
        clearance={'accepted':True,'stage':stage,'source_manifest_sha256':source_hash,'review_reference':'Fabricated fixture only, not real-data permission'}
        if training_hash: clearance['training_manifest_sha256']=training_hash
        storage.write_json(root/'metadata.json',{'config':config,'stage':stage,'source_manifest_sha256':source_hash,'training_manifest_sha256':training_hash,'dataset_sha256':config['data_sha256'],'clearance':clearance})
        storage.write_json(root/'resources.json',{'wall_seconds':10.,'rss_peak_bytes':10000,'retained_bytes':10000})
        storage.write_json(root/'MANIFEST.json',storage.manifest(root,[p.name for p in root.iterdir()]))
        storage.write_json(root/'COMPLETE.json',{'manifest_sha256':storage.sha256(root/'MANIFEST.json')})
        storage.write_json(root/'WATCHDOG.json',{'status':'completed','returncode':0,'elapsed_seconds':10.,'limit_seconds':config['wall_seconds']})
    def test_independent_audit_reconstructs_all_synthetic_cells(self):
        import audit
        x,y,s,cfg=self.data()
        with tempfile.TemporaryDirectory() as d:
            train=Path(d)/'train'; test=Path(d)/'test'; train.mkdir(); test.mkdir()
            pipeline.train_stage(x,y,s,cfg,train,storage.Budget(train,100,10**12,10**8))
            self.seal(train,cfg,'training')
            a=audit.audit_training(train,Path(d)/'source')
            self.assertEqual(a['models'],4)
            pipeline.test_stage(x,y,s,cfg,train,test,storage.Budget(test,100,10**12,10**8))
            self.seal(test,cfg,'heldout')
            b=audit.audit_test(test,train,Path(d)/'source')
            self.assertTrue(b['accepted'])
            self.assertEqual(b['reconstructed_cells'],100)
