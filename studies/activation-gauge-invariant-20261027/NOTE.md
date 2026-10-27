# A gauge-stable control for the activation-ablation question

## Answer
Yes: standardize training activations per coordinate, remove a complete spectral subspace of their correlation matrix, and transport the edit back to the original coordinates. This eliminates positive per-unit scaling as a cause of differing edited logits. It deliberately asks a different question from ordinary covariance-PCA ablation.

## Exact statement and proof
Use row-vector states h, training mean μ, and diagonal matrix D of positive training standard deviations on nonconstant columns. Let Z=(H−1μ)D⁻¹. For an orthogonal retained spectral projector P of ZᵀZ/n, define

    T(h) = μ + (h−μ)D⁻¹ P D.

Under a positive diagonal hidden scaling S, H′=HS, μ′=μS and D′=DS. Since D and S commute, Z′=Z. When the deletion boundary does not cut an eigenvalue multiplicity, the retained spectral subspace and its projector are unique, even if there are ties entirely inside a retained or deleted block. Thus P′=P. Direct substitution gives T′(hS)=T(h)S.

For downstream affine logits hW+b, compensate W′=S⁻¹W, leaving b unchanged. Then T′(hS)W′+b=T(h)W+b for every h, including held-out states. The same intertwining argument applies to a downstream function transformed as g′(u)=g(uS⁻¹). Positive hidden rescaling is compatible with ReLU homogeneity when incoming weights and hidden biases are scaled consistently. This tool does not alter a trained network itself.

Constant training columns have no valid standard deviation. This implementation leaves them unchanged on every input; it does NOT replace their held-out values by the training mean. Their indices are unchanged by positive diagonal scaling, so the proof extends blockwise with identity on those coordinates. All-constant training inputs admit deletion rank zero only.

In original coordinates the variable-block operator A=D⁻¹PD satisfies A²=A but generally Aᵀ≠A. It is an oblique projection in the ordinary Euclidean metric, orthogonal only in the standardized metric. Original-coordinate removed energy, per-case perturbation norms and covariance-PC deletion are not matched. Standardization can amplify tiny-variance coordinates. This is a gauge-stable comparison arm, not proof that its directions are semantically meaningful or superior.

## Deterministic checks
Protocol SHA-256: `ddd672f7e375e8ff5e17949027dacdabbea59af0dffc4b0020a5a954454f7183`.
First execution: 7/7 unittest methods passed. The gauge test includes all 3 prescribed scale vectors × 3 deletion ranks = 9 cells and checks clean logits, edited logits, projectors and transported edited states (absolute tolerance 1e-10). Two constructed query rows deliberately vary the training-constant column. The ordinary-PCA negative control reproduces 50% versus 100% on all four counterexample inputs. Tie handling rejects the equal-eigenvalue rank-one cut; zero and full deletion are accepted. Every first-run outcome is retained in FIRST_RUN.txt. This is a source/control check, not a new learned-digit evaluation.

Finite precision: the theorem is exact arithmetic. The implementation rejects a relative eigengap ≤1e-10 times max(1, largest eigenvalue), nonfinite inputs, invalid ranks and unrepresentable standard deviations/output. The threshold is numerical ambiguity handling, not an exact symbolic multiplicity test. Near-constant columns and extreme dynamic ranges can lose precision or change apparent rank under floating-point scaling. Positive scaling does not confer a blanket numerical guarantee. Tests cover the frozen moderate scales, not all floating-point values. A production experimental runner must log scales, spectra, gaps, projector residuals and clean-logit deviations and reject bad numerical cases rather than silently dropping them.

## Prior art and research value
Correlation-PCA scale invariance is established, not a discovery here. Penn State's primary course explanation describes covariance versus correlation PCA and standardization: https://online.stat.psu.edu/stat505/Lesson11 . Positive-ReLU reparameterization is likewise established: Dinh et al., https://proceedings.mlr.press/v70/dinh17b.html . The broader distinction between representation invariance groups and usefulness is discussed by Kornblith et al., https://proceedings.mlr.press/v97/kornblith19a.html . The contribution here is a concrete, testable prospective control for the specific gauge failure already identified in this repository. It is not invariant to arbitrary invertible hidden transformations, neuron duplication, or nonlinear reparameterization.

## Sharper prospective experiment
For an admitted fixed checkpoint and new evaluation panel: predeclare label-blind diagonal scalings, verify identical clean logits, and compare ordinary covariance-PCA versus this correlation-PCA control across the same predeclared ranks. Gauge variation in the correlation arm is a failed implementation/numerics gate, not a scientific effect. Report ordinary-PCA variation separately from the difference between ordinary and standardized metrics. Retain every seed/rank/scaling, tie rejection and failed gate in the denominator. Select neither scalings nor tolerances using evaluation labels. Decoder comparisons require consistent transported regularization; isotropic ridge after standardization changes the original estimator.

No checkpoint or new evaluation panel has been used here. Existing paused/uncertain publications and empirical-run gates remain unchanged. Independent review is required before publication.

Run: `python -B -m unittest -v` from this directory (NumPy already installed). No historical repository file was modified.

### Review-driven correction
Independent review identified that the initial `rescale()` helper could return infinity for finite inputs/scales (e.g. 1e308 × 2, or 1 / 1e-320). The helper now rejects nonfinite scaled states/weights; a regression method covers both. Final local suites pass 8/8 methods in normal and optimized Python. The original 7-test logs remain unchanged. This repair prevents that observed overflow escape; it does not extend the exact-arithmetic theorem to all floating-point scales.

### The original counterexample becomes an inadmissible rank-one cut
In the earlier independent signal/nuisance four-point construction, both standardized coordinates have variance one and zero covariance. Correlation-PCA therefore has a two-dimensional tied eigenspace. This control rejects rank-one deletion for that construction rather than manufacturing a preferred direction. It accepts zero or whole-block deletion. Thus it removes the rescaling ambiguity by a declared invariant intervention and admission rule; it does not recover a uniquely meaningful rank-one direction or preserve every requested rank.
