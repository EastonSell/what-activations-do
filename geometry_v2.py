"""Prospective numerical fixes; historical geometry.py remains frozen.

All arrays are samples by features. Ranking rejects unintended broadcasting;
cosine rejects zero rows rather than assigning an arbitrary value; cosine and
linear CKA normalize before quadratic products to avoid scale overflow and
underflow. Other geometry entrypoints are reexported unchanged. This version
does not migrate historical runners or reinterpret their recorded results.
"""
import numpy as np

from geometry import direction, fit_transform, matrix, paired, transform

__all__ = ['matrix', 'fit_transform', 'transform', 'direction', 'ranking',
           'cosine', 'cka', 'paired']


def ranking(positive, negative, axis):
    """Matched-pair ordering accuracy, with half credit for exact ties."""
    positive, negative = matrix(positive), matrix(negative)
    if positive.shape != negative.shape:
        raise ValueError('Matched shapes required')
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (positive.shape[1],) or not np.isfinite(axis).all():
        raise ValueError('Expected a finite feature-axis vector')
    with np.errstate(over='raise', invalid='raise'):
        scores = (positive-negative)@axis
    return float(np.mean((scores > 0)+.5*(scores == 0)))


def cosine(a, b):
    """Row-wise cosine; undefined zero-vector pairs raise ValueError."""
    a, b = matrix(a), matrix(b)
    if a.shape != b.shape:
        raise ValueError('Matched shapes required')
    scale_a = np.max(np.abs(a), axis=1, keepdims=True)
    scale_b = np.max(np.abs(b), axis=1, keepdims=True)
    if np.any(scale_a == 0) or np.any(scale_b == 0):
        raise ValueError('Cosine undefined for a zero vector')
    a, b = a/scale_a, b/scale_b
    denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    return np.clip(np.sum(a*b, axis=1)/denominator, -1., 1.)


def cka(a, b):
    """Centered linear CKA, invariant to independent nonzero global scales.

    Translate by a reference row before subtracting the residual mean. This
    prevents mean roundoff from inventing variation in constant coordinates.
    Center before scaling so constant high-magnitude coordinates do not erase
    small varying ones. Unrepresentable intermediate centering fails closed;
    this is not arbitrary-precision arithmetic for every finite float64 array.
    """
    a, b = matrix(a), matrix(b)
    if len(a) != len(b):
        raise ValueError('Aligned samples required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        a, b = a-a[0], b-b[0]
        a, b = a-a.mean(0), b-b.mean(0)
        scale_a, scale_b = np.max(np.abs(a)), np.max(np.abs(b))
        if scale_a == 0 or scale_b == 0:
            raise ValueError('CKA undefined for constant representation')
        a, b = a/scale_a, b/scale_b
        denominator = np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)
        return float(np.clip(np.linalg.norm(a.T@b)**2/denominator, 0., 1.))
