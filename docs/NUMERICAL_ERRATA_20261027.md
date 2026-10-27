# Numerical errata — 27 October 2026

This is an additive correction, not a revision of historical evidence. No
archived result, study source, source freeze or manifest was changed, and no
model, empirical dataset or training run was used for these checks. There is
no evidence here that the boundary defects below changed published metrics.
The separate [PCA mathematics note](PCA_ABLATION_MATH.md) addresses interpretation.

## Prospective geometry API

The original `geometry.py` is retained for compatibility and frozen
reproduction; its SHA-256 remains
`170c40648d7abc5df559209d234d6167c5e88cea7c55e7e98409d1f4a820eb00`.
It is referenced by the original root manifest and the frozen evaluation
runner's source freeze. Existing runners still import it: **they have not
silently acquired the corrections below.** New callers should explicitly use:

```python
from geometry_v2 import ranking, cosine, cka
```

The new module corrects these reproducible API/numerical defects:

- **Pair alignment:** historical `ranking` can broadcast three positive rows
  against one negative row and report accuracy rather than rejecting the
  unmatched pairs. V2 requires equal matrix shapes and a finite one-dimensional
  axis of the correct width. Exact score ties still receive half credit.
- **Undefined cosine:** the original array implementation returns zero when
  either row is the zero vector, silently treating undefined values as valid
  similarities. V2 raises `ValueError`, as the existing educational demo does.
  It also rejects unmatched/nonfinite arrays. Existing callers were inspected;
  no deliberate use of the zero-vector convention was found.
- **Scale stability:** forming squared norms and Gram products directly can
  overflow or underflow. For example, original self-CKA is `nan` after scaling
  a nonconstant matrix by `1e100`, and can incorrectly report a constant
  representation after scaling it by `1e-100`. V2 normalizes each cosine row,
  or each already-centered CKA matrix, before these products. These positive
  scale changes leave the mathematical quantities unchanged.

The CKA implementation translates by a reference row before subtracting the
remaining mean, avoiding false variance from mean roundoff in constant columns.
It centers before normalizing so a large constant coordinate does not suppress
a small varying coordinate. Unrepresentable intermediate centering raises an arithmetic error instead of silently
returning a nonfinite score. This is a targeted float64 correction, not a
promise of arbitrary-precision behavior. Projection, direction and synthetic
pair-generation entrypoints are reexported unchanged; they are not newly
audited or redesigned by this patch.

## Frozen copy-study log-softmax

`studies/copy-activation-interchange/source/patchstudy/metrics.py` computes
log probabilities by subtracting `max(z) + log(sum(exp(z-max(z))))` from `z`.
In exact arithmetic that is correct. In floating point, adding the maximum
back can erase the much smaller log-normalizer. For 49,152 equal finite
float32 logits of `1e20`, the frozen function reports probability **1** per
token instead of **1/49,152**. Probability-dependent metrics therefore have
a numerical failure mode at sufficiently large common offsets.

The prospective correction keeps the whole calculation in shifted coordinates:

```python
shifted = z - z.max()
logp = shifted - np.log(np.exp(shifted).sum(dtype=np.float64))
probability = np.exp(logp)
```

[The small reference implementation](../examples/learning/log_softmax_reference.py)
validates finite real vector inputs and demonstrates this correction. It is
not wired into any historical runner. It cannot recover logit differences
already rounded away in the supplied inputs. Frozen copy-study code and
reported scores remain untouched; testing the original saved logit ranges
would be required before asserting any historical numerical impact.

## Bounded verification

Run from the repository root (NumPy already installed):

```sh
python -B -m unittest discover -s tests -p test_geometry_v2.py -v
python -O -B -m unittest discover -s tests -p test_geometry_v2.py -v
python -B -m unittest discover -s tests -p test_log_softmax_reference.py -v
```

These use deterministic constructed arrays only. Regression tests failed
against the previous formulas/behavior before the prospective corrections
passed. This is not a full repository-suite run: that suite includes empirical
digit training, which was outside this mathematical correction's authorized
scope. No new benchmark estimate, historical-score recomputation or inference
was performed.
