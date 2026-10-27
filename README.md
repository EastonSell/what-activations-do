# What Activations Do

## Abstract

Removing high-variance activation directions can make a representation look simpler while damaging a model's predictions. This repository studies that distinction through controlled geometry examples, digit classifiers and a pinned small language model. Its latest two published digit studies separate the original classifier's sensitivity to an edit from a new readout's ability to recover performance. Leading-PC deletion harmed the tested frozen heads more than equal-norm orientation perturbations; after eight-PC deletion, a fixed ridge refit recovered only part of the lost accuracy. These results answer a bounded intervention-and-recovery question, not whether information is irreversibly lost or a unique semantic circuit has been found.

## Research question

**For the tested digit representations, is removing high-variance activation directions harmless, and can a fixed refitted ridge readout restore performance after eight-PC deletion?**

The broader motivation is to understand what centering, projection and activation replacement change in a model's answers. The finite question above is answered by the two completed studies below; the geometry and language-model examples provide separate context, not extra replications of that answer.

## Answer and results

**No: the specified leading-PC edits damaged the original frozen heads, and the tested ridge refit provided partial rescue, not near-baseline recovery.** The comparisons answer different questions and must keep their own baselines:

| Question and published evidence | Matched comparison | Result |
|---|---|---|
| [Frozen-head sensitivity](studies/frozen-head-digits-20261024/report/RESULTS.md) | PCA deletion minus equal-norm orientation perturbation, averaged over ranks 1–8 and five trained models | **+1.325640 nats** cross-entropy damage; conditional case-bootstrap interval **[1.246820, 1.404320]**; all five seed effects positive |
| [Fixed-ridge recovery](studies/ridge-readout-recovery-20261025/report/RESULTS.md) | PCA-8 ridge versus untouched ridge | **42.11% versus 95.61%** accuracy; primary gap **−53.50 percentage points**, descriptive conditional interval **[−56.94, −50.11]** |
| [Descriptive rescue](studies/ridge-readout-recovery-20261025/report/RESULTS.md) | PCA-8 ridge versus the original head on the same PCA-edited features | **42.11% versus 15.11%**, a **+27.00-point** gain; not the primary comparison |

The original frozen head's clean accuracy was **97.22%**, not the **95.61%** matched ridge baseline. The ridge study did not meet its frozen two-point recovery sensitivity rule. That failure applies to the specified readout and intervention, not every possible decoder.

### What was tested

The frozen-head study used five fully trained 64→32 ReLU→10-logit digit classifiers and five paired frozen-random-hidden controls. The ridge follow-up reused the five trained representations, fitting 90 affine ridge heads at fixed λ=0.01: untouched, PCA-8, and 16 random rank-8 deletion conditions per seed. Both studies used the same historical split of **1,078 training, 359 validation and 360 test cases**. The 360 test cases were shared across models; 1,800 seed-case predictions are not 1,800 independent observations. See the [frozen-head design](studies/frozen-head-digits-20261024/README.md) and [ridge design](studies/ridge-readout-recovery-20261025/README.md).

### What the answer does not establish

- **Learning-specificity was not established.** The frozen-random-hidden controls failed the frozen-head study's competence-matching gate; the trained-versus-random contrast is ineligible for that conclusion.
- **These were artificial interventions on a reused benchmark.** They sometimes moved activations outside natural ReLU support. The intervals are conditional on the fixed models and shared cases, not new-writer or training-population uncertainty. The ridge interval is a descriptive sensitivity interval, not a confidence interval with claimed nominal coverage.
- **The controls are not interchangeable.** Frozen-head equal-norm controls are rotations of the removed vector, not projections. Ridge random deletions match nominal ambient rank, but not removed energy or actual retained training-design rank. They do not isolate variance alone or equal effective capacity.
- **Partial recovery is not an information-theoretic verdict.** Failure of this fixed ridge readout does not demonstrate irreversible loss, failure of all readouts, unique semantic localization, or transfer to language models or brains.

## Check the published evidence first

From a trusted checkout, verify the frozen 63-file ridge package without importing its code, then check the verifier and published summary tables:

```sh
python -I -S -B verify_ridge_publication.py
python -B -m unittest discover -s tests -p test_ridge_publication.py -v
(cd studies/ridge-readout-recovery-20261025 && python -B -m unittest discover -s tests -v)
python -I -S -B verify_ridge_publication.py
```

These checks use Python's standard library, read saved evidence and test integrity rejection; they do not train a model, run checkpoint inference, reconstruct full score arrays or rerun the real bootstrap. The table checks recompute reported accuracy and contrast arithmetic from all published condition rows, including integer denominators. Passing hashes or software tests is not independent confirmation of scientific truth.

The [verification guide](docs/RIDGE_PUBLICATION_CHECKS.md) explains the pinned manifest and exact file coverage. The [frozen reproduction guide](studies/ridge-readout-recovery-20261025/REPRODUCTION.md) separates these saved-summary checks from five additional synthetic suites (NumPy/threadpoolctl; operational tests require Linux), numerical reproduction and a full-array audit. The public derivative preserves report bytes but is not the original executed source; original checkpoints and complete arrays are not included. A fresh fit or a newly downloaded dataset would not reproduce the exact retained result.

## Start here

- [Explore a small geometry example](docs/START_HERE.md), including why moving the origin differs from deleting directions.
- [Read the research overview](docs/RESEARCH_OVERVIEW.md) for hypotheses, methods, related work and next tests.
- [Inspect the copy-target interchange study](studies/copy-activation-interchange/README.md) and its [full report](studies/copy-activation-interchange/report/REPORT.md).
- [Inspect the frozen-head digit ablation](studies/frozen-head-digits-20261024/README.md): completed norm-matched perturbation study, with its [result and limitations](studies/frozen-head-digits-20261024/report/RESULTS.md). The trained-versus-random comparison failed its competence gate.
- [Inspect the ridge readout recovery study](studies/ridge-readout-recovery-20261025/README.md): refitting after eight-PC deletion reached 42.11% versus 95.61% untouched accuracy, a partial rescue rather than near-baseline recovery; [full results and limitations](studies/ridge-readout-recovery-20261025/report/RESULTS.md).
- [Verify the frozen ridge package](docs/RIDGE_PUBLICATION_CHECKS.md) with an offline, standard-library-only checker before executing its code; integrity verification is not a new experiment or full-array audit.
- [Use SETUP.md](SETUP.md) for the original experiments; later studies have separate setup and evidence instructions.

## Watch the narrated explanation

[![Narrated concept illustration](docs/animations/narrated/poster.png)](docs/animations/narrated/concept.mp4)

[49.4-second narrated MP4](docs/animations/narrated/concept.mp4) · [Complete transcript](docs/animations/narrated/transcript.txt) · [Captions](docs/animations/narrated/captions.vtt) · [Editable source and reproduction](docs/animations/narrated/RENDERING.md)

Speech-led Manim animation with a synthetic stock voice. Exact geometry illustration · not a model experiment. This is an explanatory upgrade, not a new experimental result. Captions have approximate sentence timing. The [static-first viewer](docs/animations/narrated/index.html), poster and complete transcript provide a no-motion alternative. The original silent version and its historical QA remain unchanged below.

## Watch the concept in 24 seconds

[![Two points with opposite y labels move onto the x axis and coincide at (1,0), showing information loss under a hand-chosen projection.](docs/animations/poster.png)](docs/animations/concept.mp4)

[24-second Manim MP4](docs/animations/concept.mp4) · [Optional GIF preview](docs/animations/concept.gif) · [Text transcript](docs/animations/transcript.txt) · [Captions](docs/animations/captions.vtt)

Rendered with Manim Community 0.22.0. Exact geometry illustration · not a model experiment. No audio. The poster and transcript provide a no-motion alternative. The downloaded [HTML viewer](docs/animations/index.html) is static by default and never autoplays.

## Earlier experiments and separate language-model evidence

**High variance can carry useful information.** In [constructed high-signal-variance data](results/experiment.json), mean pair-ranking accuracy fell from 1.0000 to 0.5318 after removing one principal component, across nine seed×dimension conditions with 384 test pairs each. On learned digit features, a nearest-centroid probe fell from 95.49% to 91.05% after one-PC removal and 38.02% after eight: means across three overlapping seeded test splits of 540 cases each, distinct from the later 360-case split. Centering alone left those digit predictions unchanged. [Original digit results and attribution](docs/real-data-results.md) explain the scope.

**Small projection effects can be hard to distinguish.** A validation-selected four-PC removal preserved 20/20 held-out word-copying predictions in ten paired groups, as did every random-control condition. This was removal of four directions from 576, with an accuracy ceiling in all conditions. [The report](milestones/copy-subspace-test-results/README.md) documents the rank/energy controls and missing full intervention logits.

**Interchange sensitivity changed across sampled sites.** In a fixed 64-prompt SmolLM2-135M-Instruct panel comprising 32 lexical pairs, final-position replacement at block 24 produced 60/63 genuine donor switches among baseline-correct prompts, versus 0/63 for source-word replacement. Identity, causal-prefix and whole-state controls were exact. The [report and transfer map](studies/copy-activation-interchange/report/REPORT.md) cover all sampled cells. Seven sampled blocks cannot identify a unique circuit or an exact transition point.

**Arithmetic capability gates failed.** The [milestone history](MILESTONES.md) retains those outcomes. The [frozen evaluation](milestones/frozen-evaluation-results/README.md) stopped before validation/test inference or projection fitting. Later copying results do not repair those failures.

## Optional original experiments

These commands run the earlier experiments or audits; they are **not** the saved-evidence closeout checks above and do not reproduce the later frozen-head/ridge results. Python 3.10+, NumPy and scikit-learn are sufficient for the original experiments; the synthetic experiment needs NumPy alone. From the repository root:

```sh
python run_checks.py
OPENBLAS_NUM_THREADS=1 python run_experiment.py --out /tmp/representation-synthetic-new.json
OPENBLAS_NUM_THREADS=1 python run_digits.py --out /tmp/representation-digits-new.json
python split_audit.py --output /tmp/representation-audit-fresh.json
```

Use fresh output paths; the experiment CLIs refuse overwrite. The digit study uses locally bundled UCI data and performs no download. [Setup and release-safety notes](SETUP.md) describe dependency locks and aggregate checks. [Interchange evidence reconstruction](studies/copy-activation-interchange/GITHUB_DOWNLOAD.md) is separate from new model inference.

## Open questions beyond this release

Can another readout recover what this fixed ridge head could not, and do the digit effects survive genuinely new data and better-matched controls? Separately, does the copying-site pattern survive a new lexical panel, prompt family or model checkpoint? These remain untested questions here, not prerequisites for accepting the bounded results above. The [research overview](docs/RESEARCH_OVERVIEW.md) gives the wider research agenda. Biological correspondence would require actual brain data and a separate study.

Original code, configurations and synthetic outputs are [MIT licensed](LICENSE). UCI digits use CC BY 4.0, with [creator attribution](docs/real-data-results.md). Dependencies, model weights and papers retain their own licenses; their content is not relicensed by this repository.
