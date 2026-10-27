# Independent review — 2026-10-27

## Verdict
Mathematical claim accepted under exact arithmetic and stated spectral-boundary rule. Reusable helper release is **pending one numerical input/output safety correction** and review of forthcoming NOTE.md. No historical-model evaluation or scientific novelty claim is warranted by these tests.

## Proof check
For variable training columns, positive diagonal S gives mu'=mu S and D'=DS, so standardized training and new states are identical. Their correlation matrix and the projector onto a union of complete eigenspaces are identical. Mapping back yields T'(hS)=T(h)S. The compensating affine decoder S^-1 W therefore preserves edited and clean logits. The unchanged output bias is essential. Zero-variance training columns pass through unchanged even if held-out states vary there, so they satisfy the same equivariance. Mean restoration is included, not optional.

The raw-coordinate linear part is A=D^-1 P D: A²=A but generally A is not symmetric. This is an oblique projection in the original Euclidean metric, so the method does not preserve or repair the original Euclidean-PCA estimand; it introduces a different, gauge-stable intervention. Its scaling invariance is not an arbitrary-basis invariance, semantic identification, or guarantee of predictive benefit. Cutoff eigenspace ties must be rejected or handled as whole eigenspaces. Internal basis choices within fully retained/deleted eigenspaces do not matter.

## Reproduction
- Original 7 unit tests passed normal Python and python -O (14 test executions); logs in REVIEW_TESTS.txt.
- Six supplementary cases: scales (.3,2.7,1.1) and (1e-8,1e8,3.14159), each deletion k=0,1,2, on the existing deterministic train/state/decoder arrays. Maximum absolute edited-logit discrepancy 3.552713678800501e-15. These supplement, not replace, the frozen nine scale/rank cases.
- The ordinary-PCA negative-control accuracy contrast is 0.5 versus 1.0, as specified. Correlation PCA of that particular symmetric four-point example has tied eigenvalues; k=1 is deliberately rejected, not an available rank-one answer for that example.

## Numerical finding (retained)
`rescale()` validates finite positive input scales but does not validate its output. Finite inputs states=[[1e308]], weights=[[1.]], scales=[2.] return an infinite scaled state. states=[[1.]], weights=[[1.]], scales=[1e-320] return infinite compensated weights. Add explicit finiteness checks of both results and regression tests before reuse. This is a helper defect, not a counterexample to the exact-arithmetic theorem or a failed frozen gauge test.

Even with that correction, exact floating-point rescaling equivariance cannot be universal: overflow, underflow, rounding that merges distinct values, and near-eigenspace boundaries can change numerical admissibility. Tiny/large representable numbers may be rejected by current squared-norm standard deviations. Document this rather than promising arbitrary finite scales. The test tolerance is absolute and the tested scales/data are bounded.

Protocol hash observed: ddd672f7e375e8ff5e17949027dacdabbea59af0dffc4b0020a5a954454f7183. The protocol file was not modified. No implementation files changed by reviewer.

## Resolution and final verdict
The author corrected `rescale()` to reject nonfinite scaled states or compensated weights and added a regression test for both cases. Independent rerun passes **8/8 tests in normal Python and 8/8 under -O**; `REVIEW_FINAL_TESTS.txt` retains logs. Earlier seven-test logs and the review finding remain retained. No frozen hypothesis or first-run result was replaced.

`NOTE.md` now reviewed: exact algebra, downstream affine/general-function transport, constant-column rule, full/zero-deletion semantics, original-coordinate obliqueness, boundary rejection, finite precision, prior-art attribution and distinct prospective estimand are correctly described. No remaining blocker for this bounded mathematical/control-code artifact. This acceptance does not endorse novelty, scientific performance, universal numerical robustness, or a new empirical run.

Final reviewed SHA-256 identities:
- control.py: `39e04e1e2b3dbe6004103e72070f61c5baecdc719dc34810fc9d323f4934ce36`
- test_control.py: `274476dc56b45206a7e18e3d3feb40f4820b5ca6c1e36acafc2e11933d9acfbe`
- NOTE.md: `dd99c65aeabc4618b88d8a681189cb9af8e31e088eced4ac711ea9d9be2be345`
- PROTOCOL.md: `ddd672f7e375e8ff5e17949027dacdabbea59af0dffc4b0020a5a954454f7183`

Review scope excludes independent web verification of cited prior-art pages. No novel-mathematics claim is endorsed, and the directly verified algebra suffices for the stated bounded identity.

## Final prose addendum accepted
Reviewed NOTE.md's appended review-correction history and explicit four-point tie paragraph. Both accurately preserve the observed numerical defect/fix and clarify that correlation PCA rejects, rather than resolves, the original example's rank-one tied cut. Final accepted NOTE.md SHA-256 is `ce6d3ac7e73d245156e5ef9f263c54016b9f7c76d6a73e6d1a7c1b34c50c5fd9`, superseding the earlier prose hash only. Rechecked control.py and test_control.py hashes remain unchanged from the final test run. No additional blocker.
