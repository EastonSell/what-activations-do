# Portable public source derivative

This is a sanitized derivative of the source used for the completed ridge
readout-recovery study, not the exact executed source tree. The exact report is
in `../report/`. This directory contains no real arrays, archives or operative
study clearances. It does not authorize an experiment, audit or publication.

## Identity and unchanged computation

The original source-manifest SHA-256 is
`b169b28b144a511770743782e85f05ac008e09553e3845dab654a88e2604f528`.
The original protocol-manifest SHA-256 is
`061e722fb16c8ac0962307094b713c974658f9a049ac070f9c8f9cdfe709595a`.
Exact original manifest bytes are retained in `../provenance/`.

Ancillary documentation and historical synthetic logs were sanitized. The
protocol and source manifests were regenerated. Only `settings.py`'s baked
protocol-manifest hash changed in executable source. Every other Python source
file and test, both numerical configurations, all solver operations, scientific
parameters, gates and integrity checks are byte-identical to the original.
See `../provenance/DERIVATIVE_LEDGER.json` for exact per-file hashes and reasons.

`settings.verify_source()` still requires complete source coverage and validates
the baked protocol identity and all frozen payload hashes. Original clearances
cannot validate this derivative. Old sealed scientific outputs bind the original
source, so auditing those outputs requires that exact immutable original source,
not relabeling artifacts or loosening guards to accept this copy.

The protocol's preparation-status text is historical, not the present study
status. Its nominal 32-versus-24 rank shorthand is explicitly qualified by the
completed report's measured ranks. Nothing in packaging changes the experiment.

## Components and verification

- `pipeline.py` and `protocol_v2/readout.py`: unchanged train-only geometry,
  fixed 90 affine ridge heads, 100 held-out conditions and paired analysis.
- `runner.py`, `worker.py`, `stageio.py`, `custody.py`, `watchdog.py`: unchanged
  stage separation, exclusive outputs, identity checks and sampled supervision.
- `audit.py`, `audit_runner.py`, `audit_worker.py`: unchanged independent SVD
  verification, complete rescoring and separately bounded audit.
- `tests/` and `protocol_v2/tests/`: unchanged synthetic tests, no real inputs.

Run the commands and runtime requirements in `../REPRODUCTION.md`. All generated arrays and synthetic archives
are temporary. No data download or historical artifact access is needed.

Clearance JSON is a workflow guard, not proof of human permission. Mechanical
completion is not scientific acceptance. Sampled child-tree RSS/output/wall
accounting is not kernel containment; supervisor memory and unseen spikes or
outside-output writes are not comprehensively covered. Strict margin comparisons
retain frozen floating-point operation order, including boundary sensitivity.

This result concerns one fixed ridge readout, historical features and split. It
cannot establish irreversible information loss, calibrated population inference,
matched-energy controls or natural activation support. Read the exact report for
all scientific and resource limitations, and `../REPRODUCTION.md` for what is
and is not reproducible from the public files alone.
