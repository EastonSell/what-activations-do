# Frozen-head digit ablation: prospective protocol v2 (pre-execution review corrections)

Status: source preparation and synthetic-only verification. No digit training,
validation evaluation, held-out inference, or scientific outcome exists yet.
Independent source/protocol review and a matching stage clearance are required.

## Scope

This separate prospective study extends the public EastonSell/what-activations-do
snapshot baa500a5567b3845513edb52d9b241edbeb45bab. It does not resume any other
study. The question is whether leading hidden PCs causally affect the model's
own frozen classifier beyond generic perturbation magnitude. This is neither
a new dataset nor a writer-held-out benchmark, language-model circuit study,
claim of semantic localization, or evidence of irreversible information loss.
The local sklearn digits data were used in historical work; held-out means
excluded from fitting/selection in this run, not previously unseen by research.

## Data and split

Use sklearn 1.8.0 load_digits, float64 pixels divided by 16, all 1,797 samples.
Require the published data/target concatenation SHA256
f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443.
Use train_test_split with random_state=20261024, stratify=y, test_size=0.4,
then split that remainder with random_state=20261025, stratify=remainder labels,
test_size=0.5. This gives 1078 train, 359 validation, 360 test examples.
Save exact indices. Split identity remains identical across model seeds/arms.
No test prediction or label-based tuning occurs during the training stage.

## Model and exact training

Five seeds: 101, 211, 307, 401, 503. A 64 -> 32 ReLU -> 10 linear-logit MLP.
Float64 everywhere, no dropout, augmentation, clipping, normalization, or
additional ReLU after intervention. PCG64(SeedSequence([seed,0])) initializes
W1,b1 then W2,b2 sequentially, uniform +/-sqrt(6/(fan_in+fan_out)); both arms
start with copied identical arrays. Arm A trains everything. Arm B freezes
W1,b1 byte-for-byte and trains W2,b2 only. Retain the wholly untrained snapshot
for descriptive diagnostics; chance accuracy is not evidence of robustness.

Each arm uses exactly 250 epochs, batch_size=200 including the short final
batch. For epoch e, PCG64(SeedSequence([seed,1,e])) permutes all train indices.
The same exact order is used in both arms. Adam uses lr=.001, beta1=.9,
beta2=.999, epsilon=1e-8, bias-corrected first/second moments, one timestep per
batch starting at one; separate zero moment arrays per arm. Batch gradients
average cross entropy over the actual batch, plus alpha/n_train * W with
alpha=.0001 on weights only. The loss definition is mean CE plus
alpha/(2*n_train)*(||W1||^2+||W2||^2). Frozen W1 contributes a constant in B.
No early stopping or adaptive epoch extension. Record every epoch's full
training loss, elapsed time, and update count. A terminal loss is not proof of
optimizer convergence. This explicit implementation differs from sklearn's
legacy optimizer details and is not a numerical reproduction of that run.

## Stages, gates, and no cherry-picking

Training clearance is bound to the exact source manifest SHA256 and protocol.
Training stage completes all ten fixed fits, validates clean baselines, fits
mean/PCA only on training hidden states, and seals a hash manifest. Every A
validation accuracy must be >= .90 for downstream task interpretation.
For the secondary A-versus-B learning-specific comparison, every pair must
also have both arms >= .90 and absolute accuracy difference <= .05. Failed
seeds remain. No replacement, retuning, or favorable subset aggregation.
Failure of the secondary gate does not suppress the A-only experiment, but
forbids a learning-specific conclusion. A failed primary gate blocks the
held-out stage. Held-out execution requires independent acceptance of the
training artifact and a separate clearance naming its exact manifest hash.
Clearance files are workflow guards, not cryptographic proof of human approval.

## Interventions

For each arm, training hidden states yield mean mu and full SVD basis V.
Let z=h-mu. Test k=0..32, with k=1..8 the fixed primary low-rank range.
Every classifier stays frozen; no refitted probe or centroids.
1. PCA: h' = h - (z V_k) V_k^T.
2. Rank random: h' = h - (z Q_k) Q_k^T.
3. Equal-energy orientation random: h' = h - ((z V_k) V_k^T) R.
The last is a norm-matched perturbation, NOT an orthogonal random projection.
Use 16 draws. For each seed/draw, Q and R are independent Gaussian-QR Haar
orthogonal matrices from SeedSequence([seed,2,draw,0/1]), fixing QR diagonal
signs. Draws are shared across paired arms and nested k. Training mean is
restored implicitly by subtracting only the centered component from raw h.
Save requested/realized energy and negative-activation rates. Off-manifold
states remain a limitation; do not post-hoc clip them. k=0 must reproduce
baseline. k=32 for PCA/rank must yield constant mu within numerical tolerance.

## Outcomes and falsifiers

Save full 10-class float64 logits for all examples, seeds, arms, k, controls,
plus baseline and fully untrained logits. Metrics: accuracy, stable CE,
correct-class margin against the strongest other class, perturbation norm,
negative activation proportion. Dense curves are descriptive, not k selection.
Primary statistic: mean over test examples, five A seeds and k=1..8 of PCA CE
minus the mean of the 16 equal-energy CE controls. Baseline loss cancels.
Report all seed effects. Secondary: analogous A-minus-B contrast, only when
the secondary gate passes. It is a descriptive training-regime association,
not an isolated representation-learning effect: accuracy matching does not
match calibration, logit scale, margins, or output-weight geometry. A larger
A effect alone is not proof of learned-representation specificity. Primary versus rank controls is descriptive.
A paired case bootstrap (10,000 draws, PCG64 seed 20261026) preserves every
example's complete seed/k/control pairing. Interval coverage is conditional
on this fixed reused sample, trained models, and sampled random directions;
no claim that the repeated runs are independent datasets. The interval does not
model writer variation, training-seed variation, or control Monte Carlo variation. For the primary
claim, positive conditional 95% lower bound plus >=4/5 positive seed effects
is supporting evidence; nonpositive upper bound is evidence against the
predicted excess; otherwise inconclusive. No other outcome gets a confirmatory
significance label. Similar or larger B excess, if competence-comparable,
undercuts learning-specific interpretation. Floor-level untrained accuracy
never counts as robustness. Rank-only harm without equal-energy excess is
consistent with perturbation size, not demonstrated directional specificity.

## Required checks, resources, and audit

Finite-difference gradients and scalar Adam checks on synthetic fixtures;
independent forward equality; fixed-hidden-byte equality; deterministic paired
batch schedule; disjoint split; train-only PCA; k=0 identity; k=32 endpoint;
orthogonality; per-example equal-energy tolerance; original frozen parameter
hashes; full coverage; independent full-logit re-score, per-case/seed effects, bootstrap
quantiles, signs, decision and secondary checks. The held-out audit requires
the sealed training directory, compares original head/geometry bytes, regenerates
all seeded controls, and reconstructs the complete dataset hash from the archived
train/validation/test inputs and identities. Training audit regenerates seeded
initializations and every batch order, checks epoch timing and final train loss.
Both audits enforce exact file coverage, source/config/clearance binding, and
resource limits. Auditing is not independent retraining. Invalid checks reject
the run rather than quietly removing outcomes.

Per stage: <=600 seconds wall, <=1 GiB RSS, <=512 MiB retained output, two BLAS
threads. Use an independent wall watchdog plus internal budget checks; report
that this is single-process/thread resource accounting, not kernel containment.
All outputs use fresh directories and exclusive file creation. No overwrite,
no resume, no silent retry. A failed stage retains its status and partial files
without generating an accepted manifest. Execution route must be rechecked
before any real-data stage. No publication is part of this source-only task.

## Primary sources

- Public runner: https://github.com/EastonSell/what-activations-do/blob/baa500a5567b3845513edb52d9b241edbeb45bab/run_digits.py
- Prior limitations: https://github.com/EastonSell/what-activations-do/blob/baa500a5567b3845513edb52d9b241edbeb45bab/docs/real-data-results.md
- Dataset: https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html
- UCI attribution and CC BY 4.0: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Adam: https://arxiv.org/abs/1412.6980
