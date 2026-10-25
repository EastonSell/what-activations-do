> Public sanitized derivative of historical preparation text. Scientific methods, numerical rules and limitations are retained. Historical status statements below describe preparation, not the completed result. Original and derivative hashes are mapped in the package provenance ledger.

# Bounded saved-evidence audit CLI

This audit is a separately cleared verification stage, not an experiment stage,
permission to access data, or independent scientific acceptance. No real archive,
dataset, checkpoint, training artifact, or held-out artifact was accessed while
implementing or testing it. Tests use invented one-hot arrays and synthetic
archives. The mathematical content of `protocol_v2/` remains unchanged; this public derivative has new source/protocol manifest identities.

## Preconditions and clearance

The entire runner source tree must first be frozen under its reviewed
`SOURCE_MANIFEST.json`. The production CLI verifies that source manifest and the
frozen protocol before any scientific input read. Its source identity will change
when these new modules and tests are included in the freeze. An older clearance
cannot authorize the updated source. There are no command-line scientific,
synthetic-data, adaptive-method, resource-budget, or arbitrary-command overrides.

A reviewer must separately authorize `audit-training` or `audit-heldout`. Supply
expected artifact hashes from that review, not from an unauthorized local read.
The separate clearance JSON has these fields:

- `accepted`: true
- `stage`: `audit-training` or `audit-heldout`, not `training` or `heldout`
- `source_manifest_sha256`: the exact newly reviewed source-manifest SHA256
- `protocol_manifest_sha256`: the exact matching protocol-manifest SHA256
- `artifact_manifest_sha256`: the exact audited stage's `MANIFEST.json` SHA256
- `training_manifest_sha256`: null for training; exact new training-stage manifest
  SHA256 for held-out audits
- `review_reference`: a nonblank reference to the separate audit authorization
- `audit_budget`: exactly the object below

```json
{
  "cpu_only": true,
  "blas_threads_max": 2,
  "wall_seconds": 600,
  "rss_bytes": 1073741824,
  "output_bytes": 134217728,
  "deterministic_head_verifications": 90
}
```

These are independent audit limits; they are not additional experimental refits,
not charged to the prior fit stage, and cannot be increased through this CLI.
The training audit independently SVD-recomputes the existing 90 fixed heads.
The held-out audit does the same 90 deterministic verifications once as part of
checking its training dependency, then independently reconstructs held-out scores,
metrics, fixed case bootstrap, and the descriptive numerical decision. It does
not create new experiment arms, change any head, or adapt a scientific parameter.

## Command form (not authorization to execute)

```text
python -B audit_runner.py --stage training \
  --artifact /reviewed/training \
  --artifact-manifest-sha256 EXPECTED_TRAINING_MANIFEST_SHA256 \
  --clearance /reviewed/audit-training-clearance.json \
  --out /separate/fresh-audit-training

python -B audit_runner.py --stage heldout \
  --artifact /reviewed/heldout \
  --artifact-manifest-sha256 EXPECTED_HELDOUT_MANIFEST_SHA256 \
  --training /reviewed/training \
  --training-manifest-sha256 EXPECTED_TRAINING_MANIFEST_SHA256 \
  --clearance /reviewed/audit-heldout-clearance.json \
  --out /separate/fresh-audit-heldout
```

The output parent must exist. Outputs may neither overlap the source, clearance,
or any input, nor contain them. Existing outputs cannot be retried, resumed,
overwritten, or reused. The auditor never writes into inputs or changes their
permissions. Before and after numerical work, both child and parent independently
verify the exact sealed inputs; the parent records complete byte-identity
snapshots, including completion files. Source and clearance are rechecked too.

## Outputs and interpretation

The child writes `AUDIT_REPORT.json` and its explicit operation log. The parent
writes `WATCHDOG.json`, `AUDIT_INPUTS.json`, `AUDIT_RESOURCES.json`, a separate
`MANIFEST.json`, and only then `AUDIT_COMPLETE.json`. The completion status is
`mechanical_audit_passed_pending_reviewer_decision`, with `scientific_acceptance`
and `authorizes_execution` both false. It is only valid with its manifest, bound
inputs and clearance, successful watchdog, and absence of failure evidence.
Failure, mutation, missing report, source drift, numerical disagreement, or a
resource overrun produces no retained audit-completion marker. Failed outputs
are evidence, not permission to retry or a partial acceptance.

An independent reviewer still decides scientific acceptance. Neither mechanical
success nor a descriptive sensitivity-rule result establishes calibrated
noninferiority, population uncertainty, writer generalization, or a broad claim
about neural-network representations.

## Resource and isolation limitations

The fixed audit child uses CPU NumPy 2.3.5 and at most two BLAS threads. The same
reviewed Linux watchdog externally samples the entire observed child process
tree, its output directory, stdout, and stderr. Its numeric phase has a 600-second
wall ceiling, 1-GiB child-tree RSS ceiling, and 128-MiB output ceiling. A separate
whole-audit timer spans input verification through terminal sealing; source and
clearance preflight are outside this timer. Final acceptance checks the entire
retained output, including supervisor metadata and manifests, against 128 MiB.

This is not kernel containment. Short resource spikes, writes outside the output,
very short-lived/escaped descendants, and concurrent mutate-then-restore activity
can evade sampled accounting or before/after hashes. Shared pages may be counted
more than once. Parent memory is excluded from child-tree RSS. File hashes and
rechecked read-only code establish observed custody, not OS-enforced immutability
or protection against hostile concurrent mutation. Review assumes non-adversarial
local execution and exclusive control of the frozen source and sealed inputs.
The full parent process is not independently OS-killed at its whole-audit timer;
checks fail closed if the timer is exceeded. No receipt is a signature or external
attestation, and a manually fabricated clearance cannot substitute for a real
reviewer's authorization.

## Synthetic verification

Run from the runner directory with pinned BLAS threads:

```text
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest discover -s tests -v
```

The test module `tests/test_audit_runner.py` executes invented-array audits in
actual supervised subprocesses and checks gating before artifact reads,
wrong identities, output overlap/reuse, incomplete or nonzero workers, source
mutation, input mutation, read-only input snapshots, and suppressed success
markers on failures. Production data access remains separately blocked pending
source review and explicit matching per-audit clearance.
