> Public sanitized derivative of historical preparation text. Scientific methods, numerical rules and limitations are retained. Historical status statements below describe preparation, not the completed result. Original and derivative hashes are mapped in the package provenance ledger.

# PCA deletion and recoverable linear readout: candidate protocol v2

Prepared 2026-10-24. **Source preparation and synthetic checks only.** No real
training arrays, checkpoints, validation arrays, or held-out arrays have been
loaded in this task. The protocol has not received independent review. It is
not an externally registered preregistration. No scientific result exists.

## The question and why this is next

After deletion of eight leading hidden principal components, can an equally
specified, newly fitted linear classifier retain almost all of the accuracy
available to that same classifier on the unchanged representation?

The preceding frozen-head experiment established strong sensitivity of the
original classifier to artificial edits. It explicitly did not establish
irreversible information loss. A different head may access surviving signal.
This test estimates that decodability, not the original head's causal use.
Recovery cannot negate the original causal reliance. Non-recovery cannot
establish information destruction beyond the tested readout class and budget.

This follows the published severe rank-eight result. The rank and question are
therefore informed by prior evidence, not a retrospectively undiscovered
confirmatory hypothesis. The new readout output is unmeasured; the benchmark,
split and original test outputs are already exposed. Describe this as a
prospective, fixed-analysis follow-up on historical representations and data.

## Prior evidence inspected and alternatives considered

At the original frozen-head repository commit
`d5278434f34f8b0a9b7a3a0051875963737a6363`, the frozen-head study reports
97.22% clean accuracy and 15.11% rank-eight PCA accuracy. Its mean primary
ranks 1–8 loss excess over norm-matched perturbations was 1.325640 nats.
The trained-versus-random comparison failed its competence gate. Artificial
edits left natural ReLU support. The older digit study evaluated a nearest-
centroid probe, which is a different, more restrictive readout than this one.
No ridge-refit comparison was found in the inspected tree/report/source or
repository searches; this is a repository coverage statement, not a novelty
claim about the scientific literature.

- Matched-support random perturbations: important, but a nonnegative coordinate
  check is not a natural-support guarantee, and matching several support proxies
  risks an unvalidated rejection sampler. This proposal does not solve that.
- Unrelated project-selection background is omitted from this public derivative.

Sources and precise inspected commits are in EVIDENCE_REVIEW.json.

## Fixed inputs, provenance and reuse

Reuse only the five fully trained A representations with seeds
101, 211, 307, 401, 503 from the original study. Do not use the failed random-
feature B comparison. Do not retrain W1 or b1, tune representations, alter
split indices, replace a model, or replace a failed seed.

Required training archive identity:
- `frozen-head-study-training-20261024-1813.tar.gz`
- SHA-256 `79f4f5e467e65bc8e0571e9f68414b03f9deb54e2067fc2988537422050cd548`
- Training manifest SHA-256
  `eee5fe23598de00c5960e2cae9557b7655401eeb0411a6553e3d80e96b771399`

The training archive supplies 1,078 training and 359 validation cases and
original model/geometry arrays. There are 360 fixed test cases. Preserve
the original exact case indices, train labels, pixel scaling and dataset hash.
The source dataset identity is
`f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443`.
Input custody must be verified against the historical manifest, not accepted
from these names alone. The archive is owner-retained; a public hash is not
a public download link or access guarantee.

The separate held-out archive is not needed during head fitting. A later
test stage needs either authenticated original held-out inputs or a separately
reviewed exact dataset reconstruction. Do not fetch or load it during the
preparation/training stages. Hash the archive before extraction; enforce safe
relative paths and fresh destinations. All stage reads/writes must be recorded.

## Fixed transformations and heads

For each model, use its original train-only hidden mean mu and 32-dimensional
orthonormal PCA basis V. Independently recompute training geometry for an
integrity check; use the archived projector after verification. Let z=h-mu.

Exactly 18 refitted arms per seed, 90 readouts total:

1. Untouched refit: X=z, deletion rank 0.
2. PCA-8 refit: X=z-(z V_8)V_8^T.
3. Sixteen random-rank-8 refits: X=z-(z Q_8)Q_8^T, one per draw 0–15.

Each Q is exactly the original study's Haar-QR rank-control recipe:
PCG64(SeedSequence([model_seed, 2, draw, 0])), a 32×32 standard-normal
matrix, NumPy QR, and Q columns multiplied by sign(diag(R)), with a zero
sign treated as +1. These are rank-matched controls, not energy-matched
controls. They intentionally remove eight dimensions while differing in
variance removed. Do not claim that their comparison isolates every
geometric feature or generic perturbation magnitude.

Every head has the same nominal 32→10 affine parameterization, identical
float64 solver, training cases, one-hot targets, fixed lambda=0.01, and one
linear solve. The effective data rank is 32 versus 24, the intended deletion
contrast; nominal parameter equality does not equate effective capacity.
There is no hidden layer, per-coordinate rescaling, hyperparameter grid,
validation selection, warm start, new ReLU, clipping, or augmentation.

Minimize exactly:

    sum_i ||X_i W + b - onehot(y_i)||² / (2 n) + lambda ||W||² / 2

The bias is not penalized. Center X and targets on their training means,
solve (Xc^T Xc + n lambda I) W = Xc^T Yc once, then set
b = mean(Y) - mean(X) W. The solve is the fixed optimization budget;
there is no early stopping, adaptive extension or convergence race. Record
the normal-equation residual, stationarity residual, objective, condition
number and timings for every head. Ties use lowest class index (argmax).
Ridge scores are discriminants, not calibrated probabilities: accuracy is
the endpoint; do not invent cross-entropy by treating these as probabilities.

Training and test use the same transformation and refitted head. This addresses
readout/edited-feature distribution mismatch within this model family, but
does not make the edited vectors natural activations or establish the natural
support of the representation. Some negative coordinates may remain.

## Stages and failure gates

0. Independent review of this protocol, kernels, tests and source hash. Build
   and independently review the still-missing staged runner, custody adapter,
   audit and resource watchdog before any real-data access. This package is
   reviewable numerical preparation, not an execution-ready experiment.
1. Training-only stage, after clearance tied to the exact final source manifest:
   verify input identities; reconstruct all five training activations from
   W1,b1 and original train_x; verify mu/projectors and original frozen heads;
   fit all 90 readouts once; evaluate validation; seal all arrays and hashes.
   Never read test inputs/labels in this stage, including for convenience.
2. All five original and untouched-refit validation accuracies must be >=90%.
   Every untouched-refit validation accuracy must be no more than 2 percentage
   points below its original frozen-head accuracy. Failure blocks the primary
   test, is retained and reported, and does not authorize lambda changes or
   seed removal. PCA and random validation scores are descriptive, never
   used to choose an arm, rank, penalty, endpoint or exclusion.
3. Every head must have maximum absolute objective-gradient coordinate <=1e-9,
   normalized normal-equation residual <=1e-10, and penalized Gram condition
   number <=1e12. Define A=Xc^T Xc+n lambda I, B=Xc^T Yc and that equation
   residual as max(abs(A W-B))/max(1,max(abs(B))), using the elementwise
   maximum, not a matrix row-sum norm. Any nonfinite intermediate, parameter,
   diagnostic or prediction score is an invalid-run failure before argmax.
   Every source/model/geometry
   identity and normal-equation check must pass. Failure blocks the stage;
   never silently substitute a pseudoinverse, alter regularization or retry.
4. Independent acceptance of the sealed training artifact and a separately
   recorded held-out clearance naming its manifest precede any test read.
5. One held-out evaluation, all 90 fitted readouts on all 360 fixed cases.
   Also evaluate unchanged original heads at clean and PCA-8 features (10
   original-head conditions) solely for the prespecified descriptive rescue
   contrast. Use original restored-mean h' for these frozen heads. This
   recomputation is context, not a new replication claim. No fitting here.
6. Independent full rescore and source/input/output binding before any result
   is accepted or shared as evidence. An execution failure remains a failure;
   partial runs are retained and never mixed into a complete denominator.

The present kernels cannot enforce custody or stage order by themselves. A
clearance file is a workflow guard, not proof of human approval or authenticity.

## One frozen primary endpoint and decision rule

For each test case i and fixed representation seed s, let
d_si = correct(PCA-8 refit) - correct(untouched refit). Primary D is the
mean of all 5×360 paired differences, reported in percentage points.
This is residual accuracy after deletion relative to the matched refit,
not the gain relative to the original damaged head.

Create exactly 10,000 paired case-bootstrap samples using PCG64(20261027).
Each sample resamples 360 case indices with replacement, preserving every
case's complete five-seed vector. Average seeds within cases, then average
sampled cases; use NumPy linear 2.5th/97.5th percentile quantiles. The interval
is conditional on these fixed learned representations, this historical split,
this readout class and fixed penalty. It does not estimate new writer,
dataset, training-seed-population or feature-learning variability. Cases may
share writers whose IDs are unavailable, so even case-level independence is
not established. It is a descriptive conditional sensitivity interval, not a
calibrated confidence interval: no nominal 95% coverage or calibrated
noninferiority test is claimed, including because rank eight was chosen after
inspection of historical held-out results. The frozen decision rule below is
an explicitly descriptive criterion, not a hypothesis-test error guarantee.

The 2-point margin is a prospective researcher-chosen practical tolerance,
not an established domain standard or a post-result choice:

- Mark the near-baseline decodability sensitivity rule met only if the lower
  2.5th percentile bound is
  strictly above -2 points AND every one of the five seed gaps is >=-5 points.
- If the upper 97.5th percentile bound is strictly below -2 points, mark the
  two-point margin not met by the sensitivity rule for this ridge readout.
  This is not a calibrated exclusion, nor a claim that no other readout could
  recover the signal.
- Otherwise the primary result is inconclusive. A failed seed-consistency
  gate remains inconclusive even if the pooled interval is favorable.

All validity and competence gates are prerequisites. The numerical decision
function does not itself accept a study or establish that these gates passed.

## Descriptive secondary outputs, not extra winners

Report every seed and arm, not just a favorable average:
- Original clean/PCA-8 frozen-head accuracy and refitted PCA-8 accuracy.
- Refit-minus-frozen PCA-8 accuracy gain, distinct from primary D.
- Mean and all 16 individual random-rank-8 refit accuracies and the PCA-minus-
  random contrast. No control draw is discarded or used for selection.
- Training/validation/test accuracy, confusion matrices, per-case scores,
  training objectives and numerical diagnostics, rank/variance removed,
  negative-coordinate rates, full cost and memory observations. Negative
  activation diagnostics MUST use restored-mean h'=mu+X, never centered
  ridge features X. For every seed/arm/split cell, separately record strict
  h'<0 and roundoff-aware h'<-1e-10 counts and fractions: negative coordinates
  divided by n_cases*32, and cases with any negative divided by n_cases.
  Report the integer denominators and both thresholds. These diagnose only
  ReLU-orthant violations; nonnegative vectors need not be naturally supported.

No significance labels, new rank sweep, nonlinear probes, fitted calibrators
or tuning are added after test inspection. A future alternative is a new
protocol, not a repaired primary result.

## Budget and operational boundary

CPU only, two BLAS threads maximum; zero model/dataset download, installations,
API charges, credentials, external posts or cross-host access. Per real-data
stage: <=600 seconds wall time, <=1 GiB RSS and <=128 MiB new outputs. Exactly
90 readout solves, at most 100 test conditions including original-head context.
Use an independent wall watchdog and periodic RSS checks; describe these as
process accounting, not kernel containment. No retry/resume or output overwrite.
Stop immediately on a limit, preserve partial evidence, do not issue an accepted
manifest. Existing retained archives remain read-only and outside output dirs.

The preparation task stops at independent-review submission. No real dataset,
archive import, training or held-out stage is authorized by this document alone.

## Required independent audit and adverse outcomes

Reconstruct all X from original inputs/weights; compare all retained means,
projectors, Q draws, heads and full score arrays. Refit the ridge solution via
an independently implemented augmented normal equation or SVD formula. Check
no leakage, exact identities/counts, class order/tie convention, paired case
order, every stationarity and budget gate, bootstrap, decision and all secondary
denominators. Deliberate corruption probes must reject a changed label, case
order, seed, projector, head, source hash, missing condition and failed gate.

Report failure even if the scientific narrative would be nicer without it.
Synthetic recovery and non-recovery fixtures below validate arithmetic only;
they are constructed outcomes and must never be described as evidence that
the historical digit models do or do not retain useful information.
