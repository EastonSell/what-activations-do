"""Verify the frozen public ridge package without importing its code.

Python standard library only. Read-only, offline, and deliberately limited to
byte integrity: this neither reruns an experiment nor attests scientific truth.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT / 'studies/ridge-readout-recovery-20261025'
MANIFEST_NAME = 'PACKAGE_MANIFEST.json'
MANIFEST_SHA256 = '81d0d1616f57142705b5bffa5713302feab6433a98cee53e538d83f6992ae0fc'
PUBLICATION_COMMIT = '6e360115341bd0f132999c5cd1f6b37bb264bf53'


def read_regular(path, expected_size=None, limit=1024 * 1024):
    """Read bounded regular files, not links, devices, or blocking FIFOs."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Expected a regular file: ' + path.name)
    flags = (os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0)
             | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Expected a regular file: ' + path.name)
        if info.st_size > limit or (expected_size is not None and info.st_size != expected_size):
            raise ValueError('File size mismatch: ' + path.name)
        raw = stream.read(limit + 1)
        if len(raw) > limit or (expected_size is not None and len(raw) != expected_size):
            raise ValueError('File size mismatch: ' + path.name)
        return raw


def file_inventory(root):
    """Reject symlinks and special files, including unlisted hidden entries."""
    found = set()
    pending = [(root, 0)]
    entries = 0
    while pending:
        directory, depth = pending.pop()
        if depth > 64:
            raise ValueError('Package directory depth exceeds bound')
        with os.scandir(directory) as children:
            for entry in children:
                entries += 1
                if entries > 4096:
                    raise ValueError('Package entry count exceeds bound')
                relative = Path(entry.path).relative_to(root).as_posix()
                if entry.is_symlink():
                    raise ValueError('Symlink in package: ' + relative)
                if entry.is_dir(follow_symlinks=False):
                    pending.append((Path(entry.path), depth + 1))
                elif entry.is_file(follow_symlinks=False):
                    found.add(relative)
                else:
                    raise ValueError('Nonregular entry in package: ' + relative)
    return found


def validate_manifest(raw):
    # Bind identity before trusting paths, metadata, or claims from this file.
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Pinned package manifest identity mismatch')
    value = json.loads(raw)
    if type(value.get('schema')) is not int or value['schema'] != 1:
        raise ValueError('Invalid package manifest schema')
    files = value.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Invalid package manifest files')
    for name, metadata in files.items():
        path = PurePosixPath(name)
        if (not name or '\\' in name or path.is_absolute() or '..' in path.parts
                or path.as_posix() != name or name == MANIFEST_NAME):
            raise ValueError('Unsafe package path')
        if (not isinstance(metadata, dict) or set(metadata) != {'bytes', 'sha256'}
                or type(metadata['bytes']) is not int or not 0 <= metadata['bytes'] <= 1024 * 1024
                or not isinstance(metadata['sha256'], str) or len(metadata['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in metadata['sha256'])):
            raise ValueError('Invalid package file metadata')
    return files


def verify_package(package=DEFAULT_PACKAGE):
    root = Path(package).absolute()
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError('Package root must be a real directory')
    raw = read_regular(root / MANIFEST_NAME, limit=64 * 1024)
    files = validate_manifest(raw)
    if file_inventory(root) != set(files) | {MANIFEST_NAME}:
        raise ValueError('Package file coverage mismatch: missing or unlisted file')
    for name, metadata in sorted(files.items()):
        content = read_regular(root / name, expected_size=metadata['bytes'])
        if hashlib.sha256(content).hexdigest() != metadata['sha256']:
            raise ValueError('Package file hash mismatch: ' + name)
    return {
        'status': 'public_package_integrity_verified',
        'publication_commit': PUBLICATION_COMMIT,
        'manifest_sha256': MANIFEST_SHA256,
        'verified_files': len(files) + 1,
        'checks_performed': ['pinned_manifest', 'exact_file_coverage', 'file_sizes_and_sha256'],
        'scientific_reproduction_performed': False,
        'synthetic_tests_run': False,
        'full_array_audit_performed': False,
        'limitations': [
            'Byte identity is not an independent attestation of scientific truth.',
            'No aggregate arithmetic, fitting, resampling, or model inference is performed.',
            'Original arrays and checkpoints are not in this public package.',
            'The public source is a derivative, not the original executed source.',
            'Passing this check grants no permission for a new run or data access.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE,
                        help='Path to a complete, unmodified copy of the frozen public package')
    args = parser.parse_args(argv)
    try:
        result = verify_package(args.package)
    except (OSError, ValueError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
