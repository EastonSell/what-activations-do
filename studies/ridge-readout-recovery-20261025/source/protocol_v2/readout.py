"""Pure numerical kernels: no dataset, model, archive, network or file access.

The prospective study's real-data runner is intentionally not supplied. Kernel
output alone is NOT an accepted experiment: protocol review, input custody,
training gate, separate held-out clearance and independent audit remain needed.
"""
import numpy as np


def matrix(value):
    source = np.asarray(value)
    if source.dtype.kind not in 'fiu':
        raise ValueError('Expected real numerical values')
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError('Expected a nonempty finite matrix')
    return value


def labels(y, rows, classes):
    y = np.asarray(y)
    if (not isinstance(classes, (int, np.integer)) or classes < 2 or
        y.shape != (rows,) or y.dtype.kind not in 'iu' or
        np.any(y < 0) or np.any(y >= classes)):
        raise ValueError('Expected in-range integer class labels')
    return y


@np.errstate(over='raise', invalid='raise', divide='raise')
def geometry(train_hidden):
    h = matrix(train_hidden)
    if len(h) < h.shape[1]:
        raise ValueError('Training rows must cover hidden width')
    mu = h.mean(axis=0)
    _, _, vt = np.linalg.svd(h-mu, full_matrices=False)
    return mu, vt.T


@np.errstate(over='raise', invalid='raise', divide='raise')
def project(hidden, training_mean, basis, removed_rank):
    h = matrix(hidden)
    mu = np.asarray(training_mean, dtype=np.float64)
    q = matrix(basis)
    width = h.shape[1]
    if mu.shape != (width,) or not np.isfinite(mu).all():
        raise ValueError('Invalid training mean')
    if q.shape != (width, width) or not np.allclose(q.T@q, np.eye(width), atol=1e-10, rtol=0):
        raise ValueError('Basis must be square and orthonormal')
    if not isinstance(removed_rank, (int, np.integer)) or not 0 <= removed_rank <= width:
        raise ValueError('Invalid removed rank')
    z = h-mu
    transformed = z-(z@q[:, :removed_rank])@q[:, :removed_rank].T
    if not np.isfinite(transformed).all():
        raise FloatingPointError('Nonfinite transformed feature')
    return transformed


@np.errstate(over='raise', invalid='raise', divide='raise')
def ridge(train_features, train_labels, classes, penalty):
    """Minimize ||XW+b-Y||_F^2/(2n) + penalty*||W||_F^2/2.

    The intercept is unpenalized. The unique penalized weight solution uses one
    float64 linear solve. Rank-deficient projections need no adaptive fallback.
    """
    x = matrix(train_features)
    y = labels(train_labels, len(x), classes)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Penalty must be positive and finite')
    target = np.eye(classes)[y]
    xmean = x.mean(axis=0)
    prior = target.mean(axis=0)
    xc = x-xmean
    gram = xc.T@xc + len(x)*penalty*np.eye(x.shape[1])
    rhs = xc.T@(target-prior)
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise FloatingPointError('Nonfinite normal equation')
    weights = np.linalg.solve(gram, rhs)
    intercept = prior-xmean@weights
    if not np.isfinite(weights).all() or not np.isfinite(intercept).all():
        raise FloatingPointError('Nonfinite ridge solution')
    residual = x@weights+intercept-target
    gradient = x.T@residual/len(x)+penalty*weights
    stationarity = max(float(np.max(np.abs(gradient))),
                       float(np.max(np.abs(residual.mean(axis=0)))))
    equation_residual = float(np.max(np.abs(gram@weights-rhs)) /
                              max(1., float(np.max(np.abs(rhs)))))
    condition = float(np.linalg.cond(gram))
    objective = float(np.sum(residual**2)/(2*len(x))+penalty*np.sum(weights**2)/2)
    if not np.isfinite([stationarity, equation_residual, condition, objective]).all():
        raise FloatingPointError('Nonfinite ridge diagnostic')
    return {'weights': weights, 'intercept': intercept,
            'stationarity_max_abs': stationarity,
            'normal_equation_relative_max_abs': equation_residual,
            'gram_condition_number': condition, 'objective': objective}


@np.errstate(over='raise', invalid='raise', divide='raise')
def predict_scores(fit, features):
    x = matrix(features)
    w = matrix(fit['weights'])
    b = np.asarray(fit['intercept'], dtype=np.float64)
    if x.shape[1] != w.shape[0] or b.shape != (w.shape[1],) or not np.isfinite(b).all():
        raise ValueError('Readout shape or finiteness mismatch')
    scores = x@w+b
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite prediction score; classification forbidden')
    return scores


def correct(fit, features, y):
    score = predict_scores(fit, features)
    y = labels(y, len(score), score.shape[1])
    return score.argmax(axis=1) == y


@np.errstate(over='raise', invalid='raise', divide='raise')
def support_diagnostics(centered_features, training_mean):
    """Nonnegativity proxy on restored h'=mu+X, NEVER on centered X alone."""
    x = matrix(centered_features)
    mu = np.asarray(training_mean, dtype=np.float64)
    if mu.shape != (x.shape[1],) or not np.isfinite(mu).all():
        raise ValueError('Invalid training mean')
    restored = x+mu
    if not np.isfinite(restored).all():
        raise FloatingPointError('Nonfinite restored activation')
    strict, material = restored < 0., restored < -1e-10
    return {'strict_negative_coordinate_fraction': float(strict.mean()),
            'strict_any_negative_case_fraction': float(strict.any(axis=1).mean()),
            'material_negative_coordinate_fraction': float(material.mean()),
            'material_any_negative_case_fraction': float(material.any(axis=1).mean()),
            'coordinate_denominator': int(restored.size),
            'case_denominator': int(len(restored)),
            'material_threshold': -1e-10,
            'scope': 'ReLU nonnegativity proxy only, not a guarantee of natural activation support'}


def random_basis(seed, draw, width):
    """Same Haar-QR recipe as the historical rank-random projection controls."""
    if not all(isinstance(v, (int, np.integer)) and v >= 0 for v in (seed, draw, width)) or width == 0:
        raise ValueError('Invalid random basis identity')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2, draw, 0])))
    q, r = np.linalg.qr(rng.normal(size=(width, width)))
    signs = np.sign(np.diag(r)); signs[signs == 0] = 1
    return q*signs


def eligibility(original_validation, clean_refit_validation):
    a = np.asarray(original_validation, dtype=np.float64)
    b = np.asarray(clean_refit_validation, dtype=np.float64)
    if (a.shape != (5,) or b.shape != (5,) or not np.isfinite(a).all() or
        not np.isfinite(b).all() or np.any(a < 0) or np.any(a > 1) or
        np.any(b < 0) or np.any(b > 1)):
        raise ValueError('Expected five finite validation accuracies per baseline')
    floors = (a >= .90) & (b >= .90)
    deficits = b-a
    comparable = deficits >= -.02
    return {'eligible': bool(np.all(floors & comparable)),
            'per_seed_floor_pass': floors.tolist(),
            'per_seed_baseline_gap_pp': (100*deficits).tolist(),
            'per_seed_comparability_pass': comparable.tolist()}


def primary(pca_correct, clean_refit_correct):
    a, b = np.asarray(pca_correct), np.asarray(clean_refit_correct)
    if a.shape != (5, 360) or b.shape != a.shape or a.dtype != bool or b.dtype != bool:
        raise ValueError('Expected complete paired boolean arrays: 5 seeds x 360 fixed cases')
    effects = a.astype(float)-b.astype(float)
    by_case = effects.mean(axis=0)
    by_seed = effects.mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(20261027))
    bootstrap = np.empty(10000)
    for start in range(0, 10000, 100):
        indices = rng.integers(0, 360, size=(100, 360))
        bootstrap[start:start+100] = by_case[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [.025, .975], method='linear')
    seed_gate = bool(np.all(by_seed >= -.05))
    decision = ('near_baseline_decodability_sensitivity_rule_met'
                if lower > -.02 and seed_gate else
                'two_point_margin_not_met_by_sensitivity_rule'
                if upper < -.02 else 'inconclusive')
    return {'mean_pp': float(100*effects.mean()),
            'per_seed_pp': (100*by_seed).tolist(),
            'case_resampling_interval95_pp': [float(100*lower), float(100*upper)],
            'all_seed_deficits_within_5pp': seed_gate, 'decision': decision,
            'bootstrap_seed': 20261027, 'bootstrap_draws': 10000,
            'scope': 'Descriptive case-resampling sensitivity only; no nominal coverage or calibrated noninferiority claim. Fixed historical models, split, rank and tested readout class; no writer or seed-population uncertainty.',
            'status': 'numerical_decision_only_requires_all_protocol_gates_and_independent_audit'}
