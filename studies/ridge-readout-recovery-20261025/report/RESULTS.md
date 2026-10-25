# Ridge readout after leading-PCA deletion: partial rescue, not near-baseline recovery

Completed 2026-10-25 UTC. Mechanically audited; independent review decisions are recorded separately.

## Bottom line

After removing the eight leading hidden principal components, the fixed newly fitted ridge head reached **42.11% accuracy**, versus **95.61% for the matched ridge head on untouched representations**. The prespecified primary difference was **−53.50 percentage points**. Its descriptive paired case-resampling interval was **[−56.94, −50.11] points**. The frozen two-point sensitivity rule was not met; all five seed gaps were worse than −5 points.

Refitting nevertheless improved accuracy over the original frozen head on PCA-edited features: **15.11% to 42.11%, a 27.00-point gain**. This is a separate descriptive rescue contrast, not the primary comparison. The original frozen head's clean accuracy was **97.22%**, not the matched ridge baseline.

This result establishes failure of near-baseline recovery for this specific ridge readout, fixed regularizer, representations, data and intervention. It does **not** establish irreversible information loss or failure of every possible readout, and cannot undo the earlier original-head causal-sensitivity result.

## Fixed design and gates

- Reused the five original trained A representations, seeds 101, 211, 307, 401 and 503. No feature-learning retraining, substituted checkpoints, replacement seeds, new split, or hyperparameter selection.
- Original split: 1,078 training, 359 validation and 360 historical test cases. The same 360 test cases were evaluated for all five representations. Their 1,800 seed-case predictions are not 1,800 independent observations.
- Exactly 90 new affine ridge heads: untouched, PCA-8 and 16 fixed Haar-QR random rank-8 deletion conditions per seed. Identical nominal 32→10 parameterization, float64 solver, λ=0.01, unpenalized intercept and one solve each. No nonlinear rescue, clipping, rescaling or new ReLU.
- All five original and untouched-refit validation accuracies exceeded 90%. Every untouched refit was within two points of its original head on validation. Worst deficit: 1.1142 points. All numerical gates passed.
- One training execution and one held-out evaluation; no retry, tuning, arm exclusion, alternate endpoint or scientific-protocol change. All 100 test conditions were retained: 90 refitted conditions and ten original-head context conditions.
- Separate bounded audits verified all 90 heads using SVD and reconstructed every full score array and metric; the held-out audit also independently verified the complete fixed bootstrap. Input/source hashes remained unchanged.

## Primary results by fixed seed

All accuracies below use the same 360 cases. Random means retain all 16 draws for each seed.

| Seed | Untouched ridge | PCA-8 ridge | Primary gap, pp | Original PCA-8 head | Random-8 ridge mean |
|---|---:|---:|---:|---:|---:|
| 101 | 96.11% | 43.06% | -53.06 | 13.89% | 95.26% |
| 211 | 95.83% | 42.50% | -53.33 | 14.17% | 94.91% |
| 307 | 94.72% | 35.56% | -59.17 | 15.00% | 94.43% |
| 401 | 95.83% | 46.11% | -49.72 | 14.72% | 95.82% |
| 503 | 95.56% | 43.33% | -52.22 | 17.78% | 95.16% |
| Mean | 95.61% | 42.11% | -53.50 | 15.11% | 95.11% |

The primary endpoint is PCA-8 refit minus untouched refit. The exact frozen decision label is `two_point_margin_not_met_by_sensitivity_rule`. This is not a calibrated hypothesis-test rejection or a general noninferiority conclusion.

The interval uses 10,000 paired case resamples with PCG64 seed 20261027, preserving each case's entire five-seed vector, then linear 2.5th/97.5th percentiles. It is a descriptive conditional sensitivity interval, not a confidence interval with claimed nominal 95% coverage. The strict frozen comparisons were preserved without rounding or changing the rule.

## Important control and rank limitations

The random-deletion ridge heads averaged **95.11%**, compared with 42.11% for PCA deletion. But these controls match the **nominal ambient deletion rank**, not removed energy or actual retained training-design rank.

The source review explicitly anticipated checking actual ranks rather than assuming full rank. Measured numerical training-design ranks were **28–30 untouched, 20–22 after PCA-8, and 24 after random-8** under the frozen rank threshold. The protocol's “32 versus 24” dimensions are ambient upper bounds, not the observed design ranks. These finite-training-matrix ranks are not estimates of intrinsic information dimension.

| Seed | Untouched train rank | PCA-8 train rank | Random-8 train ranks | PCA removed train variance | Random mean removed train variance |
|---|---:|---:|---:|---:|---:|
| 101 | 30 | 22 | 24 | 93.23% | 23.97% |
| 211 | 30 | 22 | 24 | 93.63% | 24.41% |
| 307 | 28 | 20 | 24 | 93.07% | 25.59% |
| 401 | 29 | 21 | 24 | 92.74% | 25.82% |
| 503 | 30 | 22 | 24 | 92.48% | 25.57% |

PCA-8 removed roughly 92.48–93.63% of centered training variance; per-seed random-control means removed 23.97–25.82%. The PCA-versus-random contrast therefore does not isolate “variance alone,” a universal geometry mechanism, matched energy, or equal effective capacity. No post-result attempt was made to repair that control by selecting draws or changing rank.

## Support and interpretation limits

- Edits are artificial. Restoring the training mean does not make them natural activations. For PCA-8 on test cases, the fraction of coordinates below −1e−10 ranged from 0.30% to 1.48%; the fraction of cases containing any such coordinate ranged from 8.06% to 45.28%. Strict-zero and roundoff-aware integer counts and denominators are retained for every arm/split. Nonnegative coordinates also do not guarantee natural support.
- This is a prospective fixed-analysis follow-up on historical models and data, not a fresh confirmatory holdout. Rank eight and the question were motivated by already viewed historical test results. The benchmark and split were already exposed.
- Writer IDs are unavailable; case independence is unestablished. Resampling does not estimate new-writer, new-dataset, training-seed-population or feature-learning uncertainty.
- Ridge scores are discriminants, not calibrated probabilities. Accuracy is the endpoint; no invented cross-entropy or calibrated-probability claim is made.
- Failure here concerns λ=0.01 ridge with the fixed features and budget, not all penalties or nonlinear readout classes. The 27-point rescue is evidence of some accessible predictive signal in this tested setting, not an information-theoretic measurement of signal preserved or destroyed.
- The original frozen-head result measured that original classifier's sensitivity to edits. Refit decoding measures a different question. Neither replaces the other.

## Resource and evidence audit

Each execution and each audit had a separate fixed ceiling: 600 seconds, 1 GiB sampled child-tree RSS and 128 MiB new output; CPU only, two OpenBLAS threads. All watchdogs reported zero errors and violations.

| Stage | Child wall, s | Elapsed before seal, s | Sampled peak RSS, bytes | Sealed bytes |
|---|---:|---:|---:|---:|
| Training | 0.927939 | 0.940189 | 42569728 | 14830953 |
| Training audit | 0.674110 | 0.773246 | 42500096 | 67846 |
| Held-out evaluation | 0.924927 | 0.933877 | 50126848 | 3097130 |
| Held-out audit | 0.961887 | 1.081626 | 45006848 | 94128 |

These are sampled process-accounting observations, not kernel containment. Supervisor memory is excluded; brief spikes, escaped processes and outside-output writes are not conclusively excluded. Source/clearance-only CLI preflight is outside the whole-stage timer. Recorded elapsed values are before sealing; terminal checks enforce the accepted-run ceiling. Explicit file-operation logs are not OS tracing. Audit SVD reconstructions are deterministic verification of the same 90 heads, not additional experimental arms.

## Reproducibility identities and retained outputs

- source_manifest_sha256: `b169b28b144a511770743782e85f05ac008e09553e3845dab654a88e2604f528`
- protocol_manifest_sha256: `061e722fb16c8ac0962307094b713c974658f9a049ac070f9c8f9cdfe709595a`
- training_manifest_sha256: `bf3d3ba9d24e4ab293d3a69e33421c79cf6c4f50dbc54e97f6b53e85b2481fe5`
- training_audit_manifest_sha256: `00f2f6660dc7ae31e529ce2dd544275b00b828346d51f13ef0d529e82728c99b`
- heldout_manifest_sha256: `b87755cc1b1a1ad1b32a2741979554122488446052a00bfac375334539f80a1f`
- heldout_audit_manifest_sha256: `6f30e1924d377b9deb133e06f244d8d373682c9ea56265084533332fba73986a`

`RESULTS.json` holds exact aggregate/per-seed results and identities. `ALL_CONDITION_METRICS.csv` contains all 180 training/validation cells and 100 test cells, with support integer counts. `ALL_FIT_DIAGNOSTICS.csv` contains all 90 fits' numerical/rank/variance diagnostics. Sealed execution artifacts retain full per-case scores, correctness, labels/case order, confusion matrices, manifests, source/input bindings, clearances and resource receipts. Separate independent inspection records document unchanged inputs and outputs.

The numerical protocol and frozen runner source were not changed after data access. Mechanical audit alone does not establish broader scientific acceptance; independent review and interpretation remain separate decisions.
