# What Activations Do
## Geometry controls, capability gates and causal interchange in neural representations

Evidence snapshot: 22 October 2026. The studies below form a sequence of distinct tests; a later result does not replace an earlier failed gate.

**27 October mathematical correction:** this historical overview now qualifies projection versus information loss and links the [three structural limits on PCA-ablation interpretation](PCA_ABLATION_MATH.md). Its older empirical snapshot is not a new experiment or a substitute for the later frozen-head/ridge reports.

## Abstract

This project investigates how representation edits affect task behavior. Synthetic controls distinguish recentering from projections that lose information in specified examples; learned digit features show that leading principal components can carry useful class information. Pretrained arithmetic studies fail prespecified capability gates, preventing downstream interpretation. A later word-copying study finds preserved exact predictions after validation-selected removal of four training PCs, with an accuracy ceiling in random controls as well. A separate 64-prompt activation-interchange study shows a coarse change from source-word to final-position sensitivity across sampled decoder blocks. The evidence supports task- and intervention-specific observations. It does not identify an exclusive circuit, a semantic truth direction or brain correspondence.

## Question and motivation

A geometric change is easy to visualize but harder to interpret. Moving an origin can change cosine relationships while preserving Euclidean differences. Removing a direction is non-invertible on the full ambient space, but may remain injective on the actual activation support or preserve everything needed for a task. Information loss requires a specified support/distribution and recovery target; failure of one decoder does not establish it. A probe can decode something that a model does not rely on, while an intervention can disrupt many features at once.

The research asks:

1. Which transformations change only the readout geometry, and which remove task-relevant information?
2. Can a model perform the target task before its activations are interpreted?
3. What changes in behavior when a selected activation is removed or replaced, relative to appropriate identity, random and positional controls?

## Related primary work

- [All-but-the-Top](https://arxiv.org/abs/1702.01417v2) studies mean and dominant-direction removal for word representations. Its lexical benchmarks motivate testing the operation, while the constructed high-signal control asks when it can fail.
- [Kornblith et al., Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html) examines invariance and CKA-based comparison. Similar representations under a statistic need not be causally interchangeable.
- [Geiger et al., Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997v2) provides a basis for testing proposed internal variables through interchange interventions. This project's copying experiment uses direct residual replacement rather than establishing a full causal abstraction.

The studies borrow these experimental ideas while using different tasks and evaluation designs. A biological correspondence study would require actual neural measurements and a separate data/provenance design. This repository has not accessed a brain dataset.

## Hypotheses and decision structure

**Geometry test:** centering and PCA removal should be evaluated separately. An adversarial high-signal-variance regime tests the claim that dominant directions are expendable. The learned-digit experiment then checks behavior beyond generated vectors.

**Capability gate:** the model must meet a frozen task criterion before projection fitting or held-out interpretation proceeds. A failed gate stops downstream analysis and remains part of the record.

**Subspace test:** a validation-selected removal can preserve copying accuracy. Comparison with rank-matched and realized-energy-matched random perturbations asks whether any observed effect is specific to the selected directions. Preserved accuracy on a ceiling-limited task is weak evidence about necessity.

**Interchange test:** donor replacements at sampled positions and blocks may change the recipient's next-token answer. Identity, causal-prefix and whole-state donor conditions test implementation and positional specificity. Endpoint identities must be distinguished from empirical discoveries.

## Methods and controls

The original [synthetic experiment](../run_experiment.py) varies dimensions, signal-variance regimes and component counts across three seeds. Means and PCs are fitted on training data. [The digit study](real-data-results.md) uses trained MLP hidden states and held-out nearest-centroid probes. The configurations share data and design; the 54 configurations are summarized across only three seeds.

The [split-confound audit](../milestones/split-confound-audit/README.md) checks source overlap and answer-only controls without new inference. The [frozen unordered-arithmetic benchmark](../milestones/unordered-arithmetic-benchmark/README.md) removes unordered-problem overlap. Its [model gate](../milestones/frozen-evaluation-results/README.md) controls whether later fitting and held-out evaluation are permitted.

The [copy-subspace protocol](../milestones/copy-subspace/PROTOCOL.md) separates training, validation selection and held-out lexical groups. The [interchange protocol](../studies/copy-activation-interchange/source/PROTOCOL.md) uses 64 fixed one-token words in 32 disjoint lexical pair groups, both orientations, seven sampled decoder blocks and six fixed conditions. The model revision is `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Hooks patch raw block outputs before final normalization.

## Measured findings

- **Geometry:** high-signal synthetic pair ranking fell from 1.0000 to 0.5318 after one-PC removal. Digit probe accuracy fell from 95.49% to 91.05% after one PC and 38.02% after eight; centering alone left those predictions unchanged. [Digit report](real-data-results.md), [synthetic outputs](../results/experiment.json).
- **Arithmetic gate:** 30/60 supplied-answer preferences and 16/33 entirely correct groups were below the frozen thresholds. Both initial execution and repeat stopped before validation/test inference or projections. [Gate outcome](../milestones/frozen-evaluation-results/README.md).
- **Copy-subspace preservation:** the selected four-PC removal and every random-control condition preserved 20/20 exact held-out predictions. The intervention removed four directions from a 576-dimensional state, leaving a high-dimensional complement. Full intervention logits were not retained; the report distinguishes scalar reconstruction from independent full-logit rescoring. [Test report](../milestones/copy-subspace-test-results/README.md).
- **Copy interchange:** all 128 baseline and 2,688 patch forwards completed. At block 24, final-position replacement caused 60/63 genuine donor switches among baseline-correct prompts; source-word replacement caused none. At earlier sampled blocks, source-word replacement was strongly effective. Full-vocabulary control errors were zero. The final-block identities follow from the architecture. [Full report and all cells](../studies/copy-activation-interchange/report/REPORT.md).

## Reproducibility and limitations

Begin with [the learning guide](START_HERE.md), [setup](../SETUP.md) and [root commands](../README.md#run). The [interchange download guide](../studies/copy-activation-interchange/GITHUB_DOWNLOAD.md) separates evidence reconstruction from fresh inference. Historical source snapshots, failures and manifests remain preserved.

The copying results cover one checkpoint, prompt family and investigator-selected lexical panel. Seven sampled blocks do not localize an exact transition. Replacing a full state changes multiple features. Identifying an exclusive, necessary or sufficient feature requires further intervention designs. Bootstrap uncertainty is conditional on the investigator-selected lexical panel. Projection selection for preservation and task ceilings constrain the negative subspace finding.

Original code/synthetic outputs are MIT licensed. UCI digits attribution and CC BY 4.0 terms are recorded in [the data report](real-data-results.md); model and dependency licenses remain separate.

## Falsifiable next steps

1. Freeze a new lexical panel and prompt family before inference. Failure to reproduce the source/final-position contrast would narrow the observed pattern to the original panel.
2. Compare interventions matched in realized energy, position and baseline competence. A putative feature-specific effect should exceed those controls without broad off-target disruption.
3. Test a model/task pair that passes the original capability gate before fitting projections. Keep failed gates and all exclusions visible. Claims of generality require independently specified models and tasks; prior failures remain archived alongside later results.

These are proposed experiments. A source-only runner, corpus or harness is preparation until its declared evaluation has actually completed.
