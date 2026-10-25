# Verify the frozen ridge publication before running its tests

From the repository root, run:

```sh
python -I -S -B verify_ridge_publication.py
```

This is a Python 3.10+ standard-library-only, offline, read-only check. It works
without NumPy or any other installed package. `-I` isolates Python from the
working directory and user environment; `-S` disables site-package loading;
`-B` suppresses bytecode. The script does not import or execute any study code.

To check a relocated copy, use its complete package directory:

```sh
python -I -S -B /path/to/repo/verify_ridge_publication.py --package '/path/to/ridge package'
```

The default package path is relative to the script, not the current directory.
The command prints JSON to standard output and creates no report file. Exit 0
means the byte-integrity checks passed, exit 1 means verification failed, and
exit 2 means the command-line arguments were invalid. Save logs outside the
frozen package if needed. Added logs or bytecode inside it correctly fail exact
file coverage.

## What is checked

The verifier pins the package manifest to SHA-256
`81d0d1616f57142705b5bffa5713302feab6433a98cee53e538d83f6992ae0fc`,
published at commit `6e360115341bd0f132999c5cd1f6b37bb264bf53`.
It verifies all 63 files: that manifest and its 62 payload entries. It rejects
missing or additional files, changed sizes or SHA-256 hashes, symlinks, and
nonregular entries such as FIFOs. Recomputing a modified package's manifest
does not bless it: there is intentionally no command-line hash override.

The implementation lives outside
`studies/ridge-readout-recovery-20261025/`. No frozen study bytes, reports,
numerical kernels, source/protocol manifests, historical tests, or clearances
were changed. The separate checker avoids asking an unverified package's own
Python code to verify itself. It does not replace a trusted checkout of this
checker or provide a signature, remote attestation, or protection against a
hostile process concurrently changing the filesystem. Check a stable local copy.

## What passing does not establish

Byte integrity is not a new scientific result or independent attestation of
scientific truth. This command does not check aggregate arithmetic, run synthetic
tests, fit a readout, bootstrap cases, perform model inference, access original
arrays/checkpoints, or grant permission for a new run.

After integrity passes, follow the six suites and environment requirements in
[the frozen reproduction guide](../studies/ridge-readout-recovery-20261025/REPRODUCTION.md).
Those separately check public summary arithmetic and synthetic behavior; the
operational suites require Linux. Run this integrity command again afterward.
Full reproduction and full-array audit still require the exact owner-retained
inputs and original provenance described in that guide. The public source is an
explicit derivative, not the source executed for the reported result.

The reported 42.11% versus 95.61% fixed-ridge accuracy and its limitations remain
unchanged: historical test exposure, the same 360 cases shared by five fixed
representations, unmatched random-control energy/rank, and limited readout class.
Neither a hash match nor passing software tests repairs those limitations.

## Regression checks

```sh
python -B -m unittest discover -s tests -p test_ridge_publication.py -v
```

The tests run the real CLI in subprocesses with site packages disabled. They
cover relocation, default-path independence, same-size corruption of each of
the 63 frozen files, size changes, missing/extra files, attempted resealing,
symlinks and FIFOs where supported, unexecuted tampered package code, and
verification under optimized Python (`-O`). They also verify that the checker
does not change the package. These are integrity regressions, not experiments.
The existing root test discovery automatically includes this test file.
