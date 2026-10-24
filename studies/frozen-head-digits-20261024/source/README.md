# Frozen-head digit study: source-only review package

This package contains an executable prospective experiment, not a scientific
result. Only fabricated-data unit/integration tests have run. Real digits have
not been loaded, trained on, or evaluated during source preparation.

Read PROTOCOL.md, config.json, core.py, pipeline.py, study.py, run_stage.py,
storage.py, audit.py, and tests/. The baseline public repository commit is
baa500a5567b3845513edb52d9b241edbeb45bab. This package is separate from all
historical copying experiments. Nothing was published or changed in that repo.

## Safe verification

    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v

Tests use fabricated arrays only, including short synthetic optimizer checks.
The data-dependent runner imports load_digits only after source/clearance guards.
Numerical environment observed: Linux, Python 3.12.14, NumPy 2.3.5, sklearn 1.8.0.
No dependencies were installed. The runner requires NumPy 2.3.5/sklearn 1.8.0;
threadpoolctl is already available in that environment.

## Review and execution gates

The source freeze is SOURCE_MANIFEST.json. Recompute and compare every listed
SHA256 before approving execution. Its own file SHA256 binds clearance to this
exact protocol/config/source/test bundle. Any source correction requires a new
manifest and a new review clearance. Existing artifact directories are refused.

No real-data clearance is included. The parent/reviewer must independently review
both method and implementation before supplying a training clearance JSON:

    {"accepted":true,"stage":"training","source_manifest_sha256":"REVIEWED_MANIFEST_HASH","review_reference":"ACTUAL_REVIEW_REFERENCE"}

After a verified execution route is available, the authorized operator can run:

    python run_stage.py --stage training --clearance APPROVED.json --out NEW_TRAIN_DIR
    python audit.py NEW_TRAIN_DIR

Stop there for independent training-artifact review. The held-out stage requires
primary competence success and a separate clearance with stage "heldout" plus
training_manifest_sha256 equal to NEW_TRAIN_DIR/MANIFEST.json's SHA256:

    python run_stage.py --stage heldout --clearance APPROVED_TEST.json --training NEW_TRAIN_DIR --out NEW_TEST_DIR
    python audit.py NEW_TEST_DIR --training NEW_TRAIN_DIR

A clearance is a workflow guard, not cryptographic proof of authorization. This
package contains no authorization to create one or to begin actual training.
No failed run is resumed. Preserve failures and choose a new directory only after
review. The independent watchdog writes WATCHDOG.json. COMPLETE.json alone is
insufficient; a failure marker or absent/failed watchdog makes the stage invalid.

## Interpretation boundaries

Main comparison uses the actual frozen classifier trained end-to-end. Random
hidden features get a separately trained head, not a chance-level random head.
The secondary comparison is only a descriptive training-regime association
and is forbidden if its competence matching gate fails. Even matched accuracies
do not match calibration, logit scales, margins, or head geometry; a larger A
effect is not proof of representation-learning specificity. All ranks/seed results remain visible. Random rank projection
and norm-matched rotated perturbation are different controls. Mean restoration
prevents accidental origin shifts. Reused local digits do not become a virgin
benchmark or a writer-held-out test by changing the split.

The independent audit requires the accepted training directory for held-out
results, verifies original head/geometry bytes and the complete dataset hash,
regenerates all seeded initialization/batch/control randomness, reconstructs
intervention logits, and independently recomputes per-case/seed effects,
bootstrap intervals, signs, decisions and the descriptive secondary contrast.
It checks exact file coverage, source/config/clearance bindings and resources.
It does not independently retrain models. Bootstrap intervals remain conditional
on the chosen fitted models and sampled directions, and do not cover writer,
training-seed or random-control Monte Carlo variability.

Version 2 addresses the independent review's source-only audit blockers.
See verification/REVIEW_FIXES.md and synthetic re-sealed mutation tests.
No previous source freeze is cleared for actual execution by these revisions.
