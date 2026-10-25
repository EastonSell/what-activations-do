"""Frozen production identities; the command line exposes no scientific overrides."""
from pathlib import Path
import custody

ROOT=Path(__file__).resolve().parent
PROTOCOL_HASH='b11706ad6e1fd24224447ae509bf464fb4550d5e69249b820f83421b8b7ec76e'
HISTORICAL_SOURCE_HASH='10a4ea2c6083e5f248cd4cd211b83ec6530f71e9e11591df9ee2f1d4e7ebf117'


def production_spec():
    return {'kind':'historical_digits_production',
      'protocol_manifest_sha256':PROTOCOL_HASH,
      'historical_source_sha256':HISTORICAL_SOURCE_HASH,
      'training_archive_sha256':'79f4f5e467e65bc8e0571e9f68414b03f9deb54e2067fc2988537422050cd548',
      'training_manifest_sha256':'eee5fe23598de00c5960e2cae9557b7655401eeb0411a6553e3d80e96b771399',
      'heldout_archive_sha256':'74843384a96f893b7c91d86f438a38cd02e37f26c18c5813c3a762a2692a2358',
      'heldout_manifest_sha256':'0f4096e4ce89c6190f21a2c67ddb98f0233e0f735a7d9adbd820eec211fcc8f8',
      'data_sha256':'f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443',
      'historical_config':custody.read_json(ROOT/'historical-config.json'),
      'config':custody.read_json(ROOT/'protocol_v2/config.json'),
      'numeric':{'geometry_atol':1e-10,'projector_atol':1e-8,'eigensystem_atol':1e-8,
                 'eigensystem_rtol':1e-10,'score_atol':1e-9,'readout_atol':1e-9}}


def source_paths():
    return [n for n in custody.files_under(ROOT)
            if '__pycache__' not in Path(n).parts and not n.endswith('.pyc') and n!='SOURCE_MANIFEST.json']


def verify_source():
    source=ROOT/'SOURCE_MANIFEST.json'
    value=custody.verify_manifest(ROOT,'SOURCE_MANIFEST.json',exact=False)
    if set(value['files'])!=set(source_paths()): raise ValueError('Runner source coverage differs from freeze')
    protocol=ROOT/'protocol_v2'
    if custody.sha256(protocol/'SOURCE_MANIFEST.json')!=PROTOCOL_HASH:
        raise ValueError('Reviewed numerical protocol manifest identity differs')
    frozen=custody.read_json(protocol/'SOURCE_MANIFEST.json')
    for name,meta in frozen['files'].items():
        p=custody.safe_file(protocol,name)
        if p.stat().st_size!=meta['bytes'] or custody.sha256(p)!=meta['sha256']:
            raise ValueError('Reviewed numerical protocol file changed: '+name)
    return custody.sha256(source)
