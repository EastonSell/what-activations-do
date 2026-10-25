import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import custody


class CustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def make_bundle(self):
        root=self.root/'bundle'; root.mkdir()
        custody.write_json(root/'value.json', {'value':3})
        custody.write_json(root/'MANIFEST.json', custody.manifest(root, ['value.json']))
        return root
    def test_hash_verified_and_exact_coverage(self):
        root=self.make_bundle()
        custody.verify_manifest(root, 'MANIFEST.json', custody.sha256(root/'MANIFEST.json'))
        (root/'extra').write_text('not declared')
        with self.assertRaises(ValueError): custody.verify_manifest(root, 'MANIFEST.json')
    def test_tamper_and_bad_manifest_identity_rejected(self):
        root=self.make_bundle()
        with self.assertRaises(ValueError): custody.verify_manifest(root,'MANIFEST.json','0'*64)
        (root/'value.json').write_text('changed')
        with self.assertRaises(ValueError): custody.verify_manifest(root,'MANIFEST.json')
    def test_symlink_and_parent_escape_rejected(self):
        root=self.make_bundle()
        for path in ['../escape', '/etc/hosts']:
            with self.assertRaises(ValueError): custody.safe_file(root,path)
        (root/'link').symlink_to(root/'value.json')
        with self.assertRaises(ValueError): custody.safe_file(root,'link')
    def test_nonfinite_json_and_overwrite_rejected(self):
        path=self.root/'a.json'; custody.write_json(path, {'a':1})
        with self.assertRaises(FileExistsError): custody.write_json(path, {'a':2})
        path.write_text('{"a":NaN}')
        with self.assertRaises(ValueError): custody.read_json(path)
    def test_clearance_requires_all_bindings(self):
        path=self.root/'clearance.json'
        base={'accepted':True,'stage':'training','source_manifest_sha256':'a'*64,
              'protocol_manifest_sha256':'b'*64,'review_reference':'synthetic review'}
        custody.write_json(path,base)
        custody.clearance(path,'training','a'*64,'b'*64)
        for key in ['accepted','stage','source_manifest_sha256','protocol_manifest_sha256','review_reference']:
            bad=base.copy(); bad[key]=None; path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): custody.clearance(path,'training','a'*64,'b'*64)
    def test_heldout_clearance_binds_training(self):
        path=self.root/'clearance.json'
        custody.write_json(path,{'accepted':True,'stage':'heldout','source_manifest_sha256':'a'*64,
          'protocol_manifest_sha256':'b'*64,'review_reference':'synthetic','training_manifest_sha256':'c'*64})
        with self.assertRaises(ValueError): custody.clearance(path,'heldout','a'*64,'b'*64,'d'*64)
    def tar(self, members):
        path=self.root/'fixture.tar'
        with tarfile.open(path,'w') as archive:
            for name,raw in members:
                info=tarfile.TarInfo(name); info.size=len(raw); archive.addfile(info,io.BytesIO(raw))
        return path
    def test_archive_hash_checked_before_output_creation(self):
        archive=self.tar([('a',b'value')]); out=self.root/'out'
        with self.assertRaises(ValueError): custody.import_archive(archive,out,'0'*64,'1'*64,{'value.json'},10000)
        self.assertFalse(out.exists())
    def test_archive_selective_import(self):
        payload=b'{"value":3}'
        m={'schema':1,'files':{'value.json':{'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}}}
        raw=json.dumps(m).encode(); mh=hashlib.sha256(raw).hexdigest()
        archive=self.tar([('prefix/MANIFEST.json',raw),('prefix/value.json',payload)])
        out=self.root/'out'
        custody.import_archive(archive,out,custody.sha256(archive),mh,{'value.json'},10000)
        self.assertEqual((out/'value.json').read_bytes(),payload)
    def test_archive_escape_and_duplicate_rejected(self):
        for names in [['../escape'],['dup','dup']]:
            archive=self.tar([(n,b'a') for n in names]); out=self.root/'out'
            with self.assertRaises(ValueError): custody.import_archive(archive,out,custody.sha256(archive),'1'*64,{'x'},1000)
            archive.unlink()
    def test_archive_expansion_limit_rejected(self):
        archive=self.tar([('large',b'a'*2000)])
        with self.assertRaises(ValueError): custody.import_archive(archive,self.root/'out',custody.sha256(archive),'1'*64,{'large'},100)


if __name__=='__main__': unittest.main()
