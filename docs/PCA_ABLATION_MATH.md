# Three mathematical limits on the PCA-ablation interpretation

**Correction and sharper question — 27 October 2026.** These are proofs and
constructed counterexamples about the specified methods, not new digit or
language-model experiments. They do not alter the historical results or frozen
study packages. The underlying linear algebra and positive-ReLU rescaling
symmetry are established mathematics, not claimed research novelties.

Separate [dated numerical errata](NUMERICAL_ERRATA_20261027.md) document
prospective code corrections without modifying frozen historical runners.

## 1. The PCA ridge refit cannot change retained coefficients

Let X be the n-by-d centered training activations, Yc the centered one-hot
targets, C = XᵀX/n, and D = XᵀYc/n. The archived ridge objective is

    ||XW + 1bᵀ − Y||²_F / (2n) + λ||W||²_F / 2,    λ > 0.

The intercept is unpenalized. In these centered coordinates b is the vector of
class frequencies, and the unique weight matrix is W = (C+λI)⁻¹D. Let P be an
orthogonal projector onto retained training principal components. Since PC=CP,

    W_P = (PCP+λI)⁻¹PD = PW.

**Proof.** Multiplying PW by PCP+λI gives PD using P²=P and PC=CP; λ>0 ensures
invertibility. Rank deficiency and tied eigenvalues do not invalidate the
argument. The intercept remains b. Therefore for every centered row vector x,

    xP W_P + bᵀ = xP W + bᵀ.

So applying the untouched ridge head to PCA-edited features gives exactly the
same scores as the PCA-refitted ridge, in real arithmetic. It does not give the
same scores as applying the untouched head to untouched features.

The assumptions match [`readout.py`](../studies/ridge-readout-recovery-20261025/source/protocol_v2/readout.py)
and [`pipeline.py`](../studies/ridge-readout-recovery-20261025/source/pipeline.py):
both arms use the same centered training cases, labels, archived training PCA,
isotropic penalty λ=0.01 and unpenalized intercept. Intercepts expressed in raw
rather than centered coordinates must be adjusted for the mean.

**Interpretation.** The reported 42.11% versus 95.61% remains a valid comparison
of two specified ridge decision rules. Refitting is redundant because the
retained coefficients are already optimal for this same projected squared-loss
objective—not because every linear classifier must fail. The accuracy gap is
not algebraically predetermined. The 15.11% to 42.11% gain compares the original
cross-entropy-trained frozen head with a different ridge-trained head; it is
not evidence of PCA-specific adaptation of the untouched ridge's retained
coefficients.

For a general orthogonal projector, subtracting the normal equations instead
gives the exact compensation identity

    W_P − PW = (PCP+λI)⁻¹ PC(I−P)W.

Retained/deleted training cross-covariances PC(I−P) vanish for training PCA, but
generally not for random deletion. Thus random-deletion refits can change
retained coefficients. This structural difference does not invalidate their
measured accuracies.

The minimized penalized training objective also has an exact decomposition.
For Cq_j=ν_jq_j and deleted PC indices R,

    J_P − J = ½ ∑_{j in R} ||q_jᵀD||² / (ν_j+λ).

Expanding the quadratic gives the minimum
`||Yc||²_F/(2n) − ½ tr[Dᵀ(C+λI)⁻¹D]`; deletion removes exactly the displayed
spectral terms. This makes target alignment, not variance alone, explicit.
It is not a theorem about accuracy or semantic importance.

The identity remains true for any *shared* positive λ and shared target matrix.
Different penalties/targets between arms, nonquadratic objectives,
non-isotropic penalties, or a projector not commuting with the actual training
covariance can break it. Rescaling both arms and recomputing their shared PCA
does not itself break it.

## 2. Equal-norm perturbations need not match information removal

For centered row state z, let Q project onto the k deleted PCs and let R be the
orthogonal rotation in the archived frozen-head energy control. Its code uses

    PCA:       z ↦ z(I−Q)
    control:   z ↦ z(I−QR).

The removed vectors zQ and zQR have identical norms. But I−Q has rank d−k,
whereas I−QR can have full rank. In two dimensions, take

    Q = [[1,0],[0,0]],   R = [[0,−1],[1,0]].

Then I−QR = [[1,1],[0,1]] has determinant one. The cases z=(±1,0) collapse to
the same point under PCA, but become (±1,±1) under the control. Both removals
have norm one; only the projection destroys their distinguishing sign.

For ideal Haar R and k<d, I−QR is invertible with probability one: its
determinant is analytic on each component of O(d), and is not identically zero
on either. To see the latter, choose R=−I on the deleted subspace and either
orientation on its nonempty orthogonal complement. The zero set of a
nonzero real-analytic function has measure zero. This does not certify any
specific finite-precision archived draw, and the k<d restriction matters.

**Interpretation.** Equal perturbation magnitude does not make this a comparison
of equally rank-deficient interventions. The frozen-head contrast still measures
different sensitivity to those two edits. Invertibility does not force a frozen
head to behave correctly: compensating the downstream decoder is a different
operation. Conversely, ambient singularity alone need not destroy information
on a restricted activation support or lose anything needed for the task.

## 3. The deletion effect is not intrinsic to the network function

Here is an exact counterexample, realizable by the repository's ReLU
architecture. Take the four equally weighted inputs (s,n)∈{−1,+1}², with class
zero for s=+1 and class one for s=−1. Define

    h = ReLU((s,n)+(2,2)) = (s+2,n+2),
    W = [[1,−1],[0,0]],    b = (−2,2).

Clean logits are (s,−s), with 100% accuracy. For any positive diagonal S, scale
the incoming hidden weights/bias by S and the output weights by S⁻¹. Positive
homogeneity `ReLU(aS)=ReLU(a)S` preserves every clean logit for **every input**.

Compare S_A=diag(α,β) and S_B=diag(β,α), with α>β>0. Their centered activation
covariances on the four cases are diag(α²,β²) and diag(β²,α²). They have the
same distinct spectrum. In both parameterizations, leading-PC deletion removes
one direction, leaves design rank one, removes variance fraction
α²/(α²+β²), and removes squared norm α² on every case.

Apply the repository's mean-restored edit `h′=μ+(h−μ)(I−vvᵀ)`:

- Under A it deletes the signal coordinate. The edited logits are (0,0).
  Lowest-index argmax gives 50% accuracy. For each n, opposite labels have
  identical edited states, so no decoder of those states alone can exceed 50%
  on this specified uniform distribution.
- Under B it deletes only the nuisance coordinate. The logits remain (s,−s)
  and accuracy remains 100%.

**What is refuted:** a universal claim that leading-PC ablation measures an
intrinsic property of the network function determined by variance/rank alone.
Even matching the spectrum, retained rank and per-case perturbation norm does
not restore that invariance. The actual fixed-coordinate digit result is not
contradicted: equivalent functions were subjected to different interventions.
PCA is equivariant under orthogonal coordinate changes; this counterexample
uses anisotropic positive scaling, not rotation. Positive ReLU rescaling is
established prior art, including [Dinh et al. (2017)](https://proceedings.mlr.press/v70/dinh17b.html).

## Recoverability requires a target and a decoder class

Projection is not automatically information destruction on a restricted
support. For example, with t uniform on {−2,−1,1,2}, let x=(t,2t²−5). Its
covariance is diag(2.5,9). Leading-PC deletion retains t and deletes 2t²−5.
The deleted coordinate cannot be recovered by an affine function of t: it is
centered and uncorrelated with t, so the best affine squared error is nine.
Nevertheless the nonlinear function `2t²−5` recovers it exactly. This is a
logical counterexample to inferring all-decoder information loss from affine
failure, not evidence that the digit representation has this structure.

Logit preservation, preservation of softmax probabilities, preservation of a
classifier's decision, and retention of the true label are different targets.
For instance, adding the same scalar to all logits changes their values but
not their softmax probabilities or argmax. A mathematical recovery claim must
name its target, support/distribution and decoder family.

## A sharper, unanswered empirical question

**How much of leading-PC ablation damage is stable under function-preserving
hidden-unit rescaling, and how much is specific to the chosen decoder?**

A useful future test would predeclare label-blind positive rescalings, verify
unchanged clean logits, then measure whether the frozen-head ablation curve
survives. Readout comparisons must name their objective and penalty metric:
keeping an isotropic ridge penalty after rescaling generally changes the
estimator. Rank, removed energy and actual training-design rank should be
reported separately; none alone establishes a causal explanation. This is a
proposed question, not a newly executed study or permission to resume one.

## Verification and remaining evidence boundary

The three `tests/test_pca_*.py` / `tests/test_activation_energy_control_rank.py`
files check the derivations on tiny deterministic constructed arrays; the ridge
and rescaling files additionally exercise the archived kernels. They check the
identities and their failure modes, not the historical digits.
The rescaling test uses α=4, β=1 and an exact known eigenvector for the zero-logit
tie; it separately checks the SVD projection within floating tolerance.

With authorized access to the original sealed training artifact, the ridge
identity can be checked by comparing saved PCA weights with `P @ clean_weights`
and the saved centered intercepts. No refit, model inference or test labels are
needed. That check is **not performed here**: the public package omits those
arrays. Floating-point tolerances and projector/normal-equation residuals must
be checked; bitwise equality is not promised. Published objective differences
alone do not independently certify the original arrays or spectral terms.

This review also clarifies earlier prose. The freeze-bound
[`docs/design.md`](design.md) calls the digit splits “independent stratified
splits/seeds”; the less ambiguous description is **three seeded, overlapping
stratified splits, not independent datasets**. Independent randomized split
draws can overlap; overlap alone does not prove dependence of the split
assignment mechanism. The issue is treating reused cases as independent new
datasets. That historical file remains
byte-identical rather than silently rewriting its manifest. The nonfrozen
[research overview](RESEARCH_OVERVIEW.md) now qualifies its earlier claim that
direction removal necessarily discards information. Neither correction changes
saved splits or scores. The root `MANIFEST.json` is a historical October 21
snapshot, not an assertion that the evolving README or new files match it.
