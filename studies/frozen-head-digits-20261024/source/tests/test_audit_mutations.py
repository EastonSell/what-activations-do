"""All mutations are fabricated data; re-sealing prevents hash checks masking gaps."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit
import pipeline
import storage
import test_pipeline as fixtures


def reseal(root):
    names=[p.name for p in root.iterdir() if p.name not in ('MANIFEST.json','COMPLETE.json','WATCHDOG.json')]
    (root/'MANIFEST.json').write_text(json.dumps(storage.manifest(root,names)))
    (root/'COMPLETE.json').write_text(json.dumps({'manifest_sha256':storage.sha256(root/'MANIFEST.json')}))

def mutate_npz(path,change):
    with np.load(path,allow_pickle=False) as f: data={k:f[k].copy() for k in f.files}
    change(data)
    with path.open('wb') as f: np.savez_compressed(f,**data)

class AuditMutationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name); self.train=root/'train'; self.test=root/'test'
        self.train.mkdir(); self.test.mkdir()
        x,y,s,self.cfg=fixtures.PipelineTests().data()
        pipeline.train_stage(x,y,s,self.cfg,self.train,storage.Budget(self.train,100,10**12,10**8))
        fixtures.IndependentAuditTests().seal(self.train,self.cfg,'training')
        pipeline.test_stage(x,y,s,self.cfg,self.train,self.test,storage.Budget(self.test,100,10**12,10**8))
        fixtures.IndependentAuditTests().seal(self.test,self.cfg,'heldout')
    def test_rejects_resealed_false_primary_and_secondary_statistics(self):
        path=self.test/'summary.json'; original=json.loads(path.read_text())
        changes=[('primary','ci95',[1000.,2000.]),('primary','per_seed',[999.,999.]),
                 ('primary','positive_seeds',999),('primary','interpretation','invented'),
                 ('secondary','mean',999.),('secondary','per_seed',[999.,999.]),('secondary','ci95',[1000.,2000.]),
                 ('secondary','eligible',False),('secondary','interpretation','invented')]
        for section,key,value in changes:
            with self.subTest(section=section,key=key):
                changed=copy.deepcopy(original); changed[section][key]=value
                path.write_text(json.dumps(changed)); reseal(self.test)
                with self.assertRaises(ValueError): audit.audit_test(self.test,self.train,self.train.parent/'source')
    def test_rejects_self_consistent_unlinked_final_head(self):
        def change(d):
            d['b2']+=3.; d['logits']+=3.; d['baseline_logits']+=3.
        mutate_npz(self.test/'test-101-A.npz',change); reseal(self.test)
        with self.assertRaises(ValueError): audit.audit_test(self.test,self.train,self.train.parent/'source')
    def test_rejects_seeded_basis_change_even_when_projection_unchanged(self):
        def change(d): d['rank_bases'][0,:,0]*=-1
        mutate_npz(self.test/'test-101-A.npz',change); reseal(self.test)
        with self.assertRaises(ValueError): audit.audit_test(self.test,self.train,self.train.parent/'source')
    def test_rejects_effect_array_disagreement(self):
        mutate_npz(self.test/'effects.npz',lambda d:d['A'].__iadd__(1.)); reseal(self.test)
        with self.assertRaises(ValueError): audit.audit_test(self.test,self.train,self.train.parent/'source')
    def test_rejects_unprescribed_initial_snapshot(self):
        mutate_npz(self.train/'model-101-A.npz',lambda d:d['initial_W2'].__iadd__(1.)); reseal(self.train)
        with self.assertRaises(ValueError): audit.audit_training(self.train,self.train.parent/'source')
    def test_rejects_shared_but_wrong_batch_order_hashes(self):
        for arm in ('A','B'):
            path=self.train/f'history-101-{arm}.json'; value=json.loads(path.read_text())
            for row in value: row['order_sha256']='0'*64
            path.write_text(json.dumps(value))
        reseal(self.train)
        with self.assertRaises(ValueError): audit.audit_training(self.train,self.train.parent/'source')
    def test_rejects_false_final_training_loss(self):
        path=self.train/'history-101-A.json'; value=json.loads(path.read_text()); value[-1]['loss']=999.
        path.write_text(json.dumps(value)); reseal(self.train)
        with self.assertRaises(ValueError): audit.audit_training(self.train,self.train.parent/'source')
    def test_rejects_unexpected_manifest_coverage(self):
        (self.test/'unrequested.txt').write_text('extra'); reseal(self.test)
        with self.assertRaises(ValueError): audit.audit_test(self.test,self.train,self.train.parent/'source')
    def test_rejects_over_budget_resource_record(self):
        path=self.train/'resources.json'; value=json.loads(path.read_text()); value['rss_peak_bytes']=10**15
        path.write_text(json.dumps(value)); reseal(self.train)
        with self.assertRaises(ValueError): audit.audit_training(self.train,self.train.parent/'source')
    def test_rejects_missing_training_archive(self):
        with self.assertRaises(ValueError): audit.audit_test(self.test)
    def test_rejects_config_source_and_training_binding_changes(self):
        path=self.test/'metadata.json'; original=json.loads(path.read_text())
        for change in ('config','source','training'):
            with self.subTest(change=change):
                value=copy.deepcopy(original)
                if change=='config': value['config']['hidden_dim']+=1
                elif change=='source': value['source_manifest_sha256']='0'*64
                else: value['training_manifest_sha256']='0'*64
                path.write_text(json.dumps(value)); reseal(self.test)
                with self.assertRaises(ValueError): audit.audit_test(self.test,self.train,self.train.parent/'source')
    def test_rejects_invalid_epoch_time_and_nonfinite_loss(self):
        path=self.train/'history-101-A.json'; original=json.loads(path.read_text())
        for field,value in [('elapsed_seconds',-1.),('epoch_seconds',-1.),('loss',float('nan'))]:
            with self.subTest(field=field):
                changed=copy.deepcopy(original); changed[-1][field]=value
                path.write_text(json.dumps(changed)); reseal(self.train)
                with self.assertRaises(ValueError): audit.audit_training(self.train,self.train.parent/'source')
    def test_rejects_expanded_watchdog_limit(self):
        path=self.train/'WATCHDOG.json'; value=json.loads(path.read_text()); value['limit_seconds']+=1
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): audit.audit_training(self.train,self.train.parent/'source')
