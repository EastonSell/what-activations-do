# Projection rank, point separation, and decoder limits

This is a mathematical note, not a new empirical study or a reinterpretation of the frozen results. It explains one reason nominally rank-matched deletion controls need not have the same effective training-design rank. Nothing here establishes why a particular model succeeds or fails.

## Setup and result

Let **X** be a fixed real n-by-d matrix of rank r. It may be a centered training activation matrix, in which case all statements concern that particular centered matrix, not uncentered held-out data. Let 0 ≤ k ≤ d and m = d − k. Let Q be the orthogonal projector onto an m-dimensional subspace sampled uniformly (Haar) from the Grassmannian, independently of X. For m = 0 use Q = 0; for m = d use Q = I. Then, in exact real arithmetic,

**rank(XQ) = min(r, m), almost surely.**

For comparison, take a full singular value decomposition of X and remove the first k right singular directions, ordering positive singular values first. If k > r, complete these with null right singular directions. The resulting projector Q_PCA satisfies

**rank(XQ_PCA) = r − min(k, r).**

Thus the two deletions have the same ambient deleted dimension k but generally retain different design ranks. For example, with r = d and 0 ≤ k ≤ d both retain d − k; when r < d and k > 0 they can differ. These identities do **not** imply matched removed energy, conditioning, readout accuracy, or generalization. The [frozen ridge report](../studies/ridge-readout-recovery-20261025/report/RESULTS.md) reports numerical training-design ranks and removed variance separately; the ideal result does not certify its individual finite-precision draws. In a fitted pipeline X must be the matrix whose singular directions were actually used; a projector fit on one sample need not produce this PCA identity on another.

### Proof

For 0 < m < d, let Z be d-by-m with independent standard real Gaussian entries. Z has full column rank almost surely, and its column space is Haar distributed; write U for an orthonormal basis of that space. Q = UUᵀ, so rank(XQ) = rank(XU) = rank(XZ), since U and Z span the same space and right multiplication by Uᵀ preserves rank of XU. The upper bound min(r,m) is immediate. For the lower bound, consider minors of size s = min(r,m) of XZ. At least one such minor is a nonzero polynomial in entries of Z: choosing s columns of Z from right singular vectors corresponding to positive singular values makes XZ have rank s. A nonzero real polynomial vanishes on a Lebesgue-null set, and Z has a density. Hence rank(XZ) ≥ s almost surely. The endpoint cases are immediate.

For PCA, write X = LΣVᵀ. Orthogonal deletion of the first k columns of V zeroes precisely the first min(k,r) positive singular terms; all other positive singular terms remain. Their left and right vectors are orthogonal, so the resulting rank is r − min(k,r). Equal singular values permit different choices of basis inside a tied singular subspace but do not alter the count. This assumes exact positive versus zero singular values; deleting arbitrary directions is not covered by this PCA formula.

## Finite point separation is weaker than useful recovery

For any fixed finite set of pairwise distinct activation vectors x₁,…,x_N in Rᵈ, a Haar-random projector Q of rank m ≥ 1 preserves their pairwise distinctness almost surely. For each nonzero difference v = xᵢ − xⱼ, the event vQ = 0 has probability zero: the first column of a Haar orthonormal basis is uniform on the unit sphere and has nonzero inner product with fixed v almost surely. A finite union over pairs is still null. At m = 0 all points coincide. This statement is about a fixed finite set independent of the random projector; an adaptively chosen set, an infinite set, or finite-precision computations requires separate analysis.

Consequently, for arbitrary labels on those N fixed points, an unconstrained decoder can memorize the projected labels by a lookup table (or any function interpolating these distinct inputs). This existence statement is not a guarantee for a fixed original head, affine/ridge decoder, trained decoder, limited sample, robustness, or unseen examples. Distinct points can be extremely close after projection; separation does not give a useful margin, stable decoding, or generalization. The observed restricted-ridge recovery and frozen-head damage remain exactly the bounded empirical outcomes reported in the frozen study.

## Numerical interpretation and synthetic checks

Exact algebraic rank is not numerical rank. A numerical SVD typically counts σᵢ > τ, where one common default is τ = σ_max · max(n,d) · machine epsilon; the absolute threshold must scale with units and noise, and different thresholds may produce different ranks. The frozen ridge [pipeline](../studies/ridge-readout-recovery-20261025/source/pipeline.py) uses this scale-dependent rank threshold; the historical configuration’s `numerical_atol=1e-10` is a separate setting and is not that rank threshold. Near-zero projected singular values and nearly coincident projected points can be rounded away. A finite pseudorandom generator and floating-point QR produce a discrete approximation, not a continuous Haar draw, so the exact almost-sure statement is not a literal guarantee for any implementation or seed. QR sign conventions do not change the projected subspace; finite precision, poor conditioning, and rank thresholds still matter. Synthetic regression tests exercise well-separated cases and a deliberately aligned exceptional subspace; they illustrate the identities but cannot prove probability-one claims or reproduce a model result.
