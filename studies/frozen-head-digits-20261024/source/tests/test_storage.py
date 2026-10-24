import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import storage

class StorageTests(unittest.TestCase):
    def test_existing_output_and_dangling_symlink_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'new'; storage.fresh_directory(p)
            with self.assertRaises(FileExistsError): storage.fresh_directory(p)
            q=Path(d)/'link'; q.symlink_to(Path(d)/'absent')
            with self.assertRaises(FileExistsError): storage.fresh_directory(q)
    def test_json_and_npz_writes_are_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'; storage.write_json(p,{'ok':1})
            with self.assertRaises(FileExistsError): storage.write_json(p,{'ok':2})
            q=Path(d)/'x.npz'; storage.write_npz(q,a=np.array([3.]))
            with self.assertRaises(FileExistsError): storage.write_npz(q,a=np.array([4.]))
            with np.load(q,allow_pickle=False) as a: self.assertEqual(a['a'][0],3.)
    def test_manifest_rejects_tampering_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); storage.write_json(root/'x.json',{'ok':1})
            m=storage.manifest(root,['x.json']); storage.verify_manifest(root,m)
            (root/'x.json').write_text('changed')
            with self.assertRaises(ValueError): storage.verify_manifest(root,m)
            with self.assertRaises(ValueError): storage.manifest(root,['../outside'])
    def test_clearance_matches_stage_and_both_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval.json'
            storage.write_json(p,{'accepted':True,'stage':'heldout','source_manifest_sha256':'a'*64,'training_manifest_sha256':'b'*64,'review_reference':'Synthetic test only'})
            storage.check_clearance(p,'heldout','a'*64,'b'*64)
            for args in [('training','a'*64,None),('heldout','c'*64,'b'*64),('heldout','a'*64,'c'*64)]:
                with self.assertRaises(ValueError): storage.check_clearance(p,*args)
    def test_budget_rejects_oversize_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'big').write_bytes(b'12345')
            b=storage.Budget(root,wall_seconds=100,rss_bytes=10**12,disk_bytes=4)
            with self.assertRaises(RuntimeError): b.check()

if __name__=='__main__': unittest.main()
