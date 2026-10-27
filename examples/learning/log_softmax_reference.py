"""Prospective stable log-softmax, separate from frozen study code.

No model, dataset, random sampling, file access or execution on import.
See docs/NUMERICAL_ERRATA_20261027.md. This does not replace archived metrics.
"""
import numpy as np


def distribution(logits):
    """Return log probabilities and probabilities for a finite real vector.

Keep the normalizer in shifted coordinates: adding the original maximum
back can discard the small log-normalizer when a common shift is huge.
Float64 cannot recover differences already rounded out of input logits.
"""
    source = np.asarray(logits)
    if source.dtype.kind not in 'fiu':
        raise ValueError('Expected real numerical logits')
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError('Expected a nonempty finite logits vector')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        shifted = z-np.max(z)
        logp = shifted-np.log(np.exp(shifted).sum(dtype=np.float64))
        return logp, np.exp(logp)
