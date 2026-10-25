# Ridge readout recovery after PCA deletion

This package preserves the exact reviewed result and provides a sanitized,
portable derivative of its source, protocol and synthetic tests. It contains
no real study arrays, checkpoints, data archives or operative clearances.

## Result

Deleting eight leading hidden PCs left the fixed refitted ridge head at **42.11%**
accuracy versus **95.61%** for the matched untouched ridge head. The primary gap
was **−53.50 percentage points**, with descriptive paired-case sensitivity
interval **[−56.94, −50.11]**. Refitting improved the edited original head's
15.11% accuracy by **27.00 points**, but did not restore near-baseline accuracy.

This is a fixed-readout result, not irreversible information loss or failure of
all possible readouts. The five fixed representations share the same 360 test
cases. Historical test exposure, unknown writer dependence, unequal removed
energy/retained rank, artificial edits and sampled resource-accounting limits
remain important. Read the full [exact report](report/RESULTS.md).

## Contents

- `report/`: byte-identical final RESULTS.md, RESULTS.json and both complete CSVs.
- `source/`: explicitly labeled public derivative; all numerical kernels,
  configurations, original tests and runtime guards are unchanged. Only the
  baked protocol-manifest identity changes in original executable source.
- `provenance/`: exact original manifest bytes and a per-file derivative ledger.
- `review/`: public receipt of independent computational/source/report checks,
  two original report-review summary records and portable synthetic review probes.
- `tests/`: public report-byte/provenance and aggregate CSV consistency checks.
- `validation/`: package-preparation synthetic test evidence, not experiments.
- `PACKAGE_MANIFEST.json`: final complete public payload freeze, excluding itself.

The checks recorded here are not external peer review. Original full-array
verification relied on separately owner-retained evidence; the public package
supports summary-level checking and synthetic tests, not public re-audit of
arrays it does not contain. The independent review receipt is not permission
for a new run, data access or publication.

## Start here

See [REPRODUCTION.md](REPRODUCTION.md) for public commands, exact retained-input
requirements and the boundary between source derivatives and original-source
audit. No installation, network request or real-data run is required by the
public checks. The full operational test suite requires Linux.

The original frozen-head study is referenced by repository-relative paths at
commit `d5278434f34f8b0a9b7a3a0051875963737a6363`, including
`studies/frozen-head-digits-20261024/report/RESULTS.md`. No claim of a new
literature search or scientific novelty is made.

This directory is a prepared publication artifact. Its presence does not
establish that it has been published. Any later external privacy/package review
should cite the exact PACKAGE_MANIFEST.json hash and remain outside that frozen
payload to avoid circular self-binding.
