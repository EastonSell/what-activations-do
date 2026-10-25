# What Activations Do

A representation can look cleaner after an edit and still lose information the model needs. This project asks what centering, projection and activation replacement actually change in a model's answers.

The work moves from controlled vector examples to learned digit features and a pinned small language model. Capability gates come before interpretation: failed tasks and failed gates stay in the record.

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

## What the experiments found

**High variance can carry useful information.** In constructed high-signal-variance data, pair-ranking accuracy fell from 1.0000 to 0.5318 after removing one principal component. On learned digit features, a nearest-centroid probe fell from 95.49% to 91.05% after one-PC removal and 38.02% after eight. Centering alone left those digit predictions unchanged. [Original results and attribution](docs/real-data-results.md) explain the scope.

**Small projection effects can be hard to distinguish.** A validation-selected four-PC removal preserved 20/20 held-out word-copying predictions, as did every random-control condition. This was removal of four directions from 576, with an accuracy ceiling in all conditions. [The report](milestones/copy-subspace-test-results/README.md) documents the rank/energy controls and missing full intervention logits.

**Interchange sensitivity changed across sampled sites.** In a fixed 64-prompt SmolLM2-135M-Instruct panel, final-position replacement at block 24 produced 60/63 genuine donor switches among baseline-correct prompts, versus 0/63 for source-word replacement. Identity, causal-prefix and whole-state controls were exact. The [report and transfer map](studies/copy-activation-interchange/report/REPORT.md) cover all sampled cells. Seven sampled blocks cannot identify a unique circuit or an exact transition point.

**Arithmetic capability gates failed.** The [milestone history](MILESTONES.md) retains those outcomes. The [frozen evaluation](milestones/frozen-evaluation-results/README.md) stopped before validation/test inference or projection fitting. Later copying results do not repair those failures.

## Run

Python 3.10+, NumPy and scikit-learn are sufficient; the synthetic experiment needs NumPy alone. From the repository root:

```sh
python run_checks.py
OPENBLAS_NUM_THREADS=1 python run_experiment.py --out /tmp/representation-synthetic-new.json
OPENBLAS_NUM_THREADS=1 python run_digits.py --out /tmp/representation-digits-new.json
python split_audit.py --output /tmp/representation-audit-fresh.json
```

Use fresh output paths; the experiment CLIs refuse overwrite. The digit study uses locally bundled UCI data and performs no download. [Setup and release-safety notes](SETUP.md) describe dependency locks and aggregate checks. [Interchange evidence reconstruction](studies/copy-activation-interchange/GITHUB_DOWNLOAD.md) is separate from new model inference.

## Next questions

Does the observed site pattern survive a new lexical panel and prompt family? Can matched-energy perturbations isolate a specific causal feature rather than broadly alter the state? What survives across model checkpoints? The [research overview](docs/RESEARCH_OVERVIEW.md) outlines prospective tests. Biological correspondence would require actual brain data and a separate study.

Original code, configurations and synthetic outputs are [MIT licensed](LICENSE). UCI digits use CC BY 4.0, with [creator attribution](docs/real-data-results.md). Dependencies, model weights and papers retain their own licenses; their content is not relicensed by this repository.
