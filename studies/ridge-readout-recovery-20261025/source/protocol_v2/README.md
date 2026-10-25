> Public sanitized derivative of historical preparation text. Scientific methods, numerical rules and limitations are retained. Historical status statements below describe preparation, not the completed result. Original and derivative hashes are mapped in the package provenance ledger.

# Next empirical experiment: readout recovery after PCA deletion

## Status

**Prepared for independent review, not executed.** The package includes a
fully specified prospective protocol and NumPy numerical kernels with 26
passing synthetic tests. No real checkpoints, train/validation arrays or test
arrays were accessed. No empirical result is claimed.

The question: does an equally specified newly fitted linear readout recover
digit accuracy after eight leading hidden PCs are removed? This tests residual
linear decodability, not whether the original frozen head relied on the PCs.
Recovery would not negate that reliance; failure would not prove that all
information was destroyed.

## Read and run

- PROTOCOL.md: fixed estimand, 90 readouts, baselines, primary endpoint,
  margin, descriptive case-resampling interval, gates, resources, adverse outcomes and scope.
- EVIDENCE_REVIEW.json: exact inspected publication/source references.
- config.json: frozen study constants.
- readout.py: train-only geometry, projection, ridge fitting and paired analysis.
- tests/test_readout.py: pure synthetic numerical and claim-boundary checks.
- GREEN-tests.txt: complete final synthetic test output.
- SOURCE_MANIFEST.json: hashes of review inputs.

With an already available Python/NumPy runtime:

    OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s tests -v

This command has no network activity, no dataset loader and no model import.
It uses synthetic arrays only. The full suite takes well under one second in
the preparation environment; this does not predict a real study's runtime.

## Intentional missing components and stop condition

There is deliberately no real-data CLI or automatic run. Before execution,
an independent reviewer must assess the protocol and numerical kernels;
then a separately reviewed runner needs archive materialization, provenance
binding, staged train/test guards, output integrity and resource supervision.
Training/validation clearance and a later held-out clearance are separate.
Source preparation is complete; experimental execution is not.

This historical preparation document alone does not authorize real-data execution.
