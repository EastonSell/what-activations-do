"""Prospective gauge-stable correlation-PCA control. No historical runner changed."""
import numpy as np


def _matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('expected nonempty finite matrix')
    return x


class CorrelationDeletion:
    def __init__(self, train, delete, gap_tolerance=1e-10):
        x = _matrix(train)
        if not np.isfinite(gap_tolerance) or gap_tolerance < 0:
            raise ValueError('invalid gap tolerance')
        self.mean = x.mean(axis=0)
        # Exact constant detection avoids mean-rounding giving constants fake variance.
        self.variable = np.any(x != x[0], axis=0)
        m = int(self.variable.sum())
        if isinstance(delete, bool) or not isinstance(delete, (int, np.integer)) or not 0 <= delete <= m:
            raise ValueError('invalid deletion rank')
        c = x[:, self.variable] - self.mean[self.variable]
        self.scale = np.sqrt(np.mean(c*c, axis=0))
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('unrepresentable standard deviation')
        z = c / self.scale
        if m:
            eig, vec = np.linalg.eigh(z.T @ z / len(z))
            if 0 < delete < m and eig[-delete] - eig[-delete-1] <= gap_tolerance * max(1., abs(eig[-1])):
                raise ValueError('deletion boundary cuts a tied/near-tied eigenspace')
            q = vec[:, m-delete:] if delete else vec[:, :0]
            self.projector = np.eye(m) - q @ q.T
        else:
            self.projector = np.empty((0, 0))

    def transform(self, states):
        x = _matrix(states)
        if x.shape[1] != len(self.mean):
            raise ValueError('feature count mismatch')
        out = x.copy()
        v = self.variable
        out[:, v] = self.mean[v] + ((x[:, v]-self.mean[v])/self.scale) @ self.projector * self.scale
        if not np.isfinite(out).all():
            raise ValueError('unrepresentable transformed values')
        return out


def rescale(states, weights, scales):
    x, w = _matrix(states), _matrix(weights)
    s = np.asarray(scales, dtype=float)
    if s.shape != (x.shape[1],) or w.shape[0] != x.shape[1] or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError('expected compatible positive finite diagonal scaling')
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        xs, ws = x*s, w/s[:, None]
    if not np.isfinite(xs).all() or not np.isfinite(ws).all():
        raise ValueError("unrepresentable rescaled states or weights")
    return xs, ws
