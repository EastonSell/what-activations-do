"""Read-only independent evidence audit. No production access on import.

The numerical verifier uses an SVD ridge solution, independently reconstructs
projections/metrics, and uses integer case totals for the bootstrap. It does
not import pipeline or protocol kernels. Recomputations are verification, not
additional experimental arms. audit_runner.py supplies production supervision.
"""
from pathlib import Path
import math
import numpy as np
import custody
import settings
import stageio

NAMES = ['untouched', 'pca8'] + [f'random8-{i:02}' for i in range(16)]
SUPPORT_SCOPE = 'ReLU nonnegativity proxy only, not a guarantee of natural activation support'
PRIMARY_SCOPE = ('Descriptive case-resampling sensitivity only; no nominal coverage or calibrated '
                 'noninferiority claim. Fixed historical models, split, rank and tested readout '
                 'class; no writer or seed-population uncertainty.')


def require(condition, message):
    if not condition: raise ValueError(message)


def finite(value, name, low=0, high=None):
    require(type(value) in (int, float) and math.isfinite(value), name + ' is not finite')
    require(low is None or value >= low, name + ' below bound')
    require(high is None or value <= high, name + ' exceeds bound')


def compare(actual, expected, name, atol=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ' keys differ')
        for key in expected: compare(actual[key], expected[key], name + '.' + key, atol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + ' length differs')
        for i, value in enumerate(expected): compare(actual[i], value, f'{name}[{i}]', atol)
    elif type(expected) in (int, bool, str):
        require(type(actual) is type(expected) and actual == expected, name + ' differs')
    elif type(expected) is float:
        finite(actual, name, None)
        require(abs(actual-expected) <= atol, name + ' differs')
    else: raise ValueError('Unsupported expected audit schema: ' + name)


def close(actual, expected, name, atol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and
            np.isfinite(expected).all() and np.allclose(actual, expected, rtol=0, atol=atol), name + ' differs')


def arrays(path, keys=None):
    path = custody.safe_file(Path(path).parent, Path(path).name)
    custody.record_access('audit_read_npz_without_pickle', path)
    with np.load(path, allow_pickle=False) as opened:
        require(keys is None or set(opened.files) == set(keys), 'Array key coverage differs')
        result = {key: opened[key].copy() for key in opened.files}
    for key, value in result.items():
        require(value.dtype.kind in 'fiuUb', 'Unsupported array dtype: ' + key)
        if value.dtype.kind in 'fiu': require(np.isfinite(value).all(), 'Nonfinite array: ' + key)
    return result


def real_array(value, shape, name):
    require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), name + ' shape/dtype/finite failure')


def labels(value, n):
    require(value.shape == (n,) and value.dtype.kind in 'iu' and np.all((value >= 0) & (value < 10)), 'Invalid labels')


def validate_spec(spec):
    frozen = settings.production_spec()
    require(spec.get('kind') in {'historical_digits_production', 'synthetic_fixture_not_scientific'}, 'Unreviewed specification kind')
    for key in ('config', 'numeric', 'protocol_manifest_sha256', 'historical_source_sha256'):
        require(spec.get(key) == frozen[key], 'Frozen specification differs: ' + key)
    if spec['kind'] == 'historical_digits_production': require(spec == frozen, 'Production identities differ')
    else: require(set(spec) == set(frozen), 'Synthetic specification coverage differs')
    for key in ('training_archive_sha256', 'training_manifest_sha256', 'heldout_archive_sha256', 'heldout_manifest_sha256', 'data_sha256'):
        value = spec.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Invalid digest: ' + key)


def stage(root, which, spec, source_hash):
    validate_spec(spec)
    require(isinstance(source_hash, str) and len(source_hash) == 64 and all(c in '0123456789abcdef' for c in source_hash), 'Invalid source identity')
    meta = stageio.verify_completed(root, which, spec, source_hash)
    resources = custody.read_json(root / 'STAGE_RESOURCES.json')
    cfg = spec['config']
    finite(resources['elapsed_before_sealing_seconds'], 'stage wall', 0, cfg['max_seconds_per_stage'])
    finite(resources['bytes_before_sealing'], 'stage bytes', 0, cfg['max_output_bytes_per_stage'])
    finite(resources['peak_sampled_rss_bytes'], 'stage RSS', 0, cfg['max_rss_bytes'])
    watchdog = custody.read_json(root / 'WATCHDOG.json')
    require(resources['peak_sampled_rss_bytes'] == watchdog['peak_sampled_rss_bytes'], 'Resource RSS receipts differ')
    if spec['kind'] == 'historical_digits_production':
        require(meta.get('runtime', {}).get('numpy') == '2.3.5', 'Unreviewed numerical runtime')
        cleared = meta.get('clearance', {})
        require(isinstance(cleared, dict) and cleared.get('accepted') is True and
                cleared.get('stage') == which and cleared.get('source_manifest_sha256') == source_hash and
                cleared.get('protocol_manifest_sha256') == spec['protocol_manifest_sha256'] and
                isinstance(cleared.get('review_reference'), str) and bool(cleared['review_reference'].strip()),
                'Missing or contradictory execution clearance evidence')
        if which == 'heldout':
            training_hash = meta.get('training_manifest_sha256')
            require(isinstance(training_hash, str) and len(training_hash) == 64 and
                    all(c in '0123456789abcdef' for c in training_hash) and
                    cleared.get('training_manifest_sha256') == training_hash,
                    'Execution clearance training identity differs')
        access = custody.read_json(root / 'ACCESS_LOG.json')
        require(isinstance(access.get('operations'), list) and access['operations'], 'Missing custody operations')
    return meta


def old_watchdog(root):
    value = custody.read_json(root / 'WATCHDOG.json')
    require(value['status'] == 'completed' and value['returncode'] == 0 and value['limit_seconds'] == 600, 'Historical watchdog failed')
    finite(value['elapsed_seconds'], 'Historical watchdog wall', 0, 600)


def historical(root, spec):
    cfg = spec['config']
    manifest = custody.verify_manifest(root, expected_hash=spec['training_manifest_sha256'], extra_allowed={'COMPLETE.json', 'WATCHDOG.json'})
    expected = {'splits.npz', 'metadata.json', 'summary.json', 'resources.json'}
    expected |= {f'{kind}-{seed}-{arm}.{ext}' for seed in cfg['seeds'] for arm in ('A', 'B') for kind, ext in [('model', 'npz'), ('history', 'json')]}
    require(set(manifest['files']) == expected, 'Historical training coverage differs')
    require(custody.read_json(root / 'COMPLETE.json')['manifest_sha256'] == spec['training_manifest_sha256'], 'Historical completion differs')
    old_watchdog(root)
    meta = custody.read_json(root / 'metadata.json')
    require(meta['stage'] == 'training' and meta['source_manifest_sha256'] == spec['historical_source_sha256'] and
            meta['config'] == spec['historical_config'] and meta['dataset_sha256'] == spec['data_sha256'], 'Historical metadata differs')
    summary = custody.read_json(root / 'summary.json')
    require(summary['gates']['primary_pass'] is True, 'Historical competence gate failed')
    data = arrays(root / 'splits.npz', {'train', 'validation', 'test', 'train_x', 'train_y', 'validation_x', 'validation_y'})
    ids = np.concatenate([data[name] for name in ('train', 'validation', 'test')])
    require(ids.dtype.kind in 'iu' and np.array_equal(np.sort(ids), np.arange(1797)), 'Split partition differs')
    for split, n in [('train', 1078), ('validation', 359), ('test', 360)]:
        require(data[split].shape == (n,) and data[split].dtype.kind in 'iu', 'Split count/order dtype differs')
        if split != 'test':
            real_array(data[split + '_x'], (n, 64), split + ' input'); labels(data[split + '_y'], n)
    models, original = {}, []
    for seed in cfg['seeds']:
        model = arrays(root / f'model-{seed}-A.npz')
        for key, shape in [('W1', (64, 32)), ('b1', (32,)), ('W2', (32, 10)), ('b2', (10,)), ('mean', (32,)), ('pcs', (32, 32)), ('singular_values', (32,))]:
            real_array(model[key], shape, key)
        train = np.maximum(data['train_x'] @ model['W1'] + model['b1'], 0.)
        validation = np.maximum(data['validation_x'] @ model['W1'] + model['b1'], 0.)
        logits = validation @ model['W2'] + model['b2']
        for key, value in [('train_hidden', train), ('validation_hidden', validation), ('validation_logits', logits), ('mean', train.mean(0))]:
            close(model[key], value, 'Historical ' + key, spec['numeric']['geometry_atol'])
        z = train - train.mean(0); _, singular, vt = np.linalg.svd(z, full_matrices=False); basis = model['pcs']
        close(basis.T @ basis, np.eye(32), 'PCA orthogonality', 1e-10)
        require(np.all(model['singular_values'] >= 0) and np.all(np.diff(model['singular_values']) <= 1e-10), 'PCA singular ordering differs')
        require(np.allclose(basis.T @ (z.T @ z) @ basis, np.diag(model['singular_values']**2),
                            atol=spec['numeric']['eigensystem_atol'], rtol=spec['numeric']['eigensystem_rtol']), 'PCA eigensystem differs')
        close(basis[:, :8] @ basis[:, :8].T, vt[:8].T @ vt[:8], 'PCA top-eight projector', spec['numeric']['projector_atol'])
        accuracy = float(np.mean(np.argmax(logits, 1) == data['validation_y']))
        rows = [row for row in summary['rows'] if row['seed'] == seed and row['arm'] == 'A']
        require(len(rows) == 1 and rows[0]['validation_accuracy'] == accuracy, 'Historical validation accuracy differs')
        original.append(accuracy); models[seed] = dict(model, train_hidden=train, validation_hidden=validation)
    return data, models, original


def bases(seed, model):
    result = [(model['pcs'], 0), (model['pcs'], 8)]
    for draw in range(16):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 2, draw, 0])))
        q, r = np.linalg.qr(generator.standard_normal((32, 32)))
        result.append((q * np.where(np.diag(r) < 0, -1., 1.), 8))
    return result


def features(hidden, mean, basis, k):
    z = hidden - mean
    # Preserve arithmetic order for strict zero-threshold support counts. A
    # separately formed projector verifies the same geometry within tolerance.
    x = z - (z @ basis[:, :k]) @ basis[:, :k].T
    close(x, z @ (np.eye(32) - basis[:, :k] @ basis[:, :k].T), 'Independent projector', 1e-10)
    return x


def support(x, mu):
    restored = x + mu; require(np.isfinite(restored).all(), 'Nonfinite restored support')
    result = {'coordinate_denominator': int(restored.size), 'case_denominator': len(restored), 'material_threshold': -1e-10, 'scope': SUPPORT_SCOPE}
    for name, threshold in [('strict', 0.), ('material', -1e-10)]:
        negative = restored < threshold
        count, cases = int(np.count_nonzero(negative)), int(np.count_nonzero(negative.any(1)))
        result.update({name + '_negative_coordinate_count': count, name + '_any_negative_case_count': cases,
                       name + '_negative_coordinate_fraction': count / restored.size, name + '_any_negative_case_fraction': cases / len(restored)})
    return result


def row(seed, condition, split, scores, y, x, mu):
    real_array(scores, (len(y), 10), 'Prediction scores')
    prediction = np.argmax(scores, axis=1)
    confusion = np.bincount(y * 10 + prediction, minlength=100).reshape(10, 10)
    count = int(np.count_nonzero(prediction == y))
    return {'seed': seed, 'condition': condition, 'split': split, 'correct_count': count,
            'cases': len(y), 'accuracy': count / len(y), 'confusion_matrix': confusion.tolist(), 'support': support(x, mu)}


def metric_domains(rows):
    for value in rows:
        finite(value['accuracy'], 'stored accuracy', 0, 1)
        for key, proportion in value['support'].items():
            if key.endswith('_fraction'):
                finite(proportion, 'stored support proportion', 0, 1)


def ridge_check(x, y, w, b, cfg):
    target = np.eye(10)[y]; xc, yc = x - x.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    independent_w = vt.T @ ((singular / (singular**2 + len(x)*.01))[:, None] * (u.T @ yc))
    independent_b = target.mean(0) - x.mean(0) @ independent_w
    close(w, independent_w, 'Independent SVD ridge weights/head'); close(b, independent_b, 'Independent SVD ridge intercept/head')
    residual = x @ w + b - target; gradient = x.T @ residual / len(x) + .01*w
    stationarity = max(float(np.abs(gradient).max()), float(np.abs(residual.mean(0)).max()))
    gram, rhs = xc.T @ xc + len(x)*.01*np.eye(32), xc.T @ yc
    equation = float(np.abs(gram @ w - rhs).max() / max(1., float(np.abs(rhs).max())))
    condition = float(np.linalg.cond(gram)); objective = float((residual**2).sum()/(2*len(x)) + .01*(w**2).sum()/2)
    for value, limit, name in [(stationarity, cfg['normal_equation_gradient_atol'], 'stationarity'),
                               (equation, cfg['normal_equation_relative_max_abs_tolerance'], 'normal equation'),
                               (condition, cfg['gram_condition_number_max'], 'condition')]: finite(value, name, 0, limit)
    finite(objective, 'objective')
    threshold = max(x.shape)*np.finfo(np.float64).eps*singular[0]
    return {'stationarity_max_abs': stationarity, 'normal_equation_relative_max_abs': equation,
            'gram_condition_number': condition, 'objective': objective,
            'actual_train_rank': int(np.count_nonzero(singular > threshold)), 'rank_threshold': float(threshold)}


def _audit_training(root, spec, source_hash):
    root = Path(root); stage(root, 'training', spec, source_hash)
    data, models, original = historical(root / 'historical_training', spec)
    summary = custody.read_json(root / 'summary.json')
    require(summary['stage'] == 'training' and summary['status'] == 'completed', 'Training state differs')
    compare(summary['readout_count'], 90, 'readout_count')
    require(len(summary['rows']) == 180 and len(summary['fit_rows']) == 90, 'Training condition coverage differs')
    metric_domains(summary['rows'])
    for key in ('original_validation_accuracy', 'untouched_refit_validation_accuracy'):
        for probability in summary[key]: finite(probability, 'stored validation accuracy', 0, 1)
    expected_rows, clean = [], []
    for si, seed in enumerate(spec['config']['seeds']):
        model = models[seed]
        fitted = arrays(root / f'fitted-{seed}.npz', {'weights', 'intercepts', 'condition_names', 'train_scores', 'validation_scores'})
        require(fitted['condition_names'].tolist() == NAMES, 'Readout condition order differs')
        for key, shape in [('weights', (18, 32, 10)), ('intercepts', (18, 10)), ('train_scores', (18, 1078, 10)), ('validation_scores', (18, 359, 10))]: real_array(fitted[key], shape, key)
        for ci, (basis, k) in enumerate(bases(seed, model)):
            xs = {split: features(model[split + '_hidden'], model['mean'], basis, k) for split in ('train', 'validation')}
            x, w, b = xs['train'], fitted['weights'][ci], fitted['intercepts'][ci]
            diag = ridge_check(x, data['train_y'], w, b, spec['config'])
            centered = model['train_hidden'] - model['mean']; removed = centered - x
            expected_fit = {'seed': seed, 'condition': NAMES[ci], **diag, 'maximum_feature_subspace_dimension': 32-k, 'removed_rank': k,
                            'mean_removed_squared_norm': float(np.mean(np.sum(removed**2, axis=1))),
                            'removed_variance_fraction': float(np.sum(removed**2)/max(np.sum(centered**2), np.finfo(float).tiny))}
            fit_row = summary['fit_rows'][si*18 + ci]; finite(fit_row.get('elapsed_seconds'), 'head elapsed', 0, 600)
            for key, low, high in [('stationarity_max_abs', 0, 1e-9),
                                   ('normal_equation_relative_max_abs', 0, 1e-10),
                                   ('gram_condition_number', 1, 1e12), ('objective', 0, None),
                                   ('rank_threshold', 0, None), ('mean_removed_squared_norm', 0, None),
                                   ('removed_variance_fraction', 0, 1)]:
                finite(fit_row[key], 'stored ' + key, low, high)
            compare({key: value for key, value in fit_row.items() if key != 'elapsed_seconds'}, expected_fit, 'fit diagnostics')
            require(fit_row['stationarity_max_abs'] <= 1e-9 and fit_row['normal_equation_relative_max_abs'] <= 1e-10 and fit_row['gram_condition_number'] <= 1e12, 'Reported numerical gate failed')
            for split in ('train', 'validation'):
                scores = xs[split] @ w + b; close(fitted[split + '_scores'][ci], scores, split + ' full score')
                expected_rows.append(row(seed, NAMES[ci], split, scores, data[split + '_y'], xs[split], model['mean']))
            if ci == 0: clean.append(expected_rows[-1]['accuracy'])
    compare(summary['rows'], expected_rows, 'Training metrics'); compare(summary['original_validation_accuracy'], original, 'Original validation')
    compare(summary['untouched_refit_validation_accuracy'], clean, 'Refit validation')
    original_a, clean_a = np.array(original), np.array(clean)
    floors, gaps = (original_a >= .9) & (clean_a >= .9), clean_a-original_a
    gates = {'eligible': bool(np.all(floors & (gaps >= -.02))), 'per_seed_floor_pass': floors.tolist(),
             'per_seed_baseline_gap_pp': (100*gaps).tolist(), 'per_seed_comparability_pass': (gaps >= -.02).tolist()}
    compare(summary['gates'], gates, 'Competence gates'); require(gates['eligible'], 'Training competence is not eligible')
    finite(summary['elapsed_compute_seconds'], 'Training compute wall', 0, 600)
    compare(custody.read_json(root / 'WORKER_RESULT.json'), {'stage': 'training', 'status': 'completed', 'readout_count': 90}, 'Training worker result')
    return data, models, {'status': 'passed', 'stage': 'training', 'readout_count': 90,
                         'manifest_sha256': custody.sha256(root / 'MANIFEST.json'),
                         'scope': 'Saved-evidence audit, not authorization or independent scientific acceptance'}


@np.errstate(over='raise', invalid='raise', divide='raise')
def audit_training(root, spec, source_hash):
    return _audit_training(Path(root), spec, source_hash)[2]


def primary(a, b):
    require(a.shape == (5, 360) and b.shape == (5, 360) and a.dtype == bool and b.dtype == bool, 'Incomplete primary denominator')
    differences = a.astype(np.int64) - b.astype(np.int64); totals = differences.sum(axis=0)
    by_case = differences.astype(np.float64).mean(axis=0)
    rng = np.random.Generator(np.random.PCG64(20261027)); resampled = np.empty(10000)
    for i in range(10000):
        selected = rng.integers(360, size=360)
        # The frozen protocol first averages seeds per case, then sampled cases.
        # Keep that operation order at strict margin boundaries, rather than
        # silently substituting exact-rational division or rounding the rule.
        resampled[i] = by_case[selected].mean()
        integer_check = totals[selected].sum()/1800
        require(abs(resampled[i]-integer_check) <= 1e-12, 'Case pairing/count bootstrap differs')
    lo, hi = np.quantile(resampled, [.025, .975], method='linear'); per_seed = differences.mean(axis=1)
    seed_pass = bool(np.all(per_seed >= -.05))
    decision = ('near_baseline_decodability_sensitivity_rule_met' if lo > -.02 and seed_pass else
                'two_point_margin_not_met_by_sensitivity_rule' if hi < -.02 else 'inconclusive')
    return {'mean_pp': float(100*differences.mean()), 'per_seed_pp': (100*per_seed).tolist(),
            'case_resampling_interval95_pp': [float(100*lo), float(100*hi)], 'all_seed_deficits_within_5pp': seed_pass, 'decision': decision,
            'bootstrap_seed': 20261027, 'bootstrap_draws': 10000, 'scope': PRIMARY_SCOPE,
            'status': 'numerical_decision_only_requires_all_protocol_gates_and_independent_audit'}


@np.errstate(over='raise', invalid='raise', divide='raise')
def audit_heldout(root, training, spec, source_hash):
    root, training = Path(root), Path(training); meta = stage(root, 'heldout', spec, source_hash)
    data, models, train_report = _audit_training(training, spec, source_hash)
    require(meta['training_manifest_sha256'] == train_report['manifest_sha256'], 'Held-out training binding differs')
    historical_root = root / 'historical_heldout'; manifest_path = historical_root / 'MANIFEST.json'
    require(custody.sha256(manifest_path) == spec['heldout_manifest_sha256'], 'Historical held-out manifest differs')
    manifest = custody.read_json(manifest_path)
    selected = {'test-identity.npz', 'metadata.json', 'COMPLETE.json', 'WATCHDOG.json', 'MANIFEST.json'}
    require(set(custody.files_under(historical_root)) == selected, 'Historical held-out selected coverage differs')
    for name in ('test-identity.npz', 'metadata.json'):
        require(custody.manifest(historical_root, [name])['files'][name] == manifest['files'][name], 'Historical held-out payload differs')
    old_watchdog(historical_root)
    require(custody.read_json(historical_root / 'COMPLETE.json')['manifest_sha256'] == spec['heldout_manifest_sha256'], 'Historical held-out completion differs')
    old_meta = custody.read_json(historical_root / 'metadata.json')
    require(old_meta['stage'] == 'heldout' and old_meta['source_manifest_sha256'] == spec['historical_source_sha256'] and
            old_meta['training_manifest_sha256'] == spec['training_manifest_sha256'] and old_meta['config'] == spec['historical_config'], 'Historical held-out metadata differs')
    identity = arrays(historical_root / 'test-identity.npz', {'indices', 'labels', 'test_x'})
    require(identity['indices'].dtype.kind in 'iu' and np.array_equal(identity['indices'], data['test']), 'Test case order differs')
    real_array(identity['test_x'], (360, 64), 'Test input'); labels(identity['labels'], 360)
    x_all, y_all = np.empty((1797, 64)), np.empty(1797, dtype=np.int64)
    for split in ('train', 'validation'): x_all[data[split]], y_all[data[split]] = data[split + '_x'], data[split + '_y']
    x_all[data['test']], y_all[data['test']] = identity['test_x'], identity['labels']
    require(custody.digest_bytes((16*x_all).tobytes()+y_all.tobytes()) == spec['data_sha256'], 'Dataset identity differs')
    expected_rows, pca, clean, gains, contrasts = [], [], [], [], []; y = identity['labels']
    for seed in spec['config']['seeds']:
        model = models[seed]; fitted = arrays(training / f'fitted-{seed}.npz')
        evidence = arrays(root / f'test-{seed}.npz', {'scores', 'original_scores', 'correctness', 'case_indices', 'labels'})
        require(evidence['case_indices'].dtype.kind in 'iu' and np.array_equal(evidence['case_indices'], identity['indices']) and
                evidence['labels'].dtype.kind in 'iu' and np.array_equal(evidence['labels'], y), 'Case order/labels differ')
        real_array(evidence['scores'], (18, 360, 10), 'Test scores'); real_array(evidence['original_scores'], (2, 360, 10), 'Original scores')
        require(evidence['correctness'].shape == (18, 360) and evidence['correctness'].dtype == bool, 'Correctness denominator differs')
        hidden = np.maximum(identity['test_x'] @ model['W1'] + model['b1'], 0.); correct, xs = [], []
        for ci, (basis, k) in enumerate(bases(seed, model)):
            x = features(hidden, model['mean'], basis, k); xs.append(x)
            scores = x @ fitted['weights'][ci] + fitted['intercepts'][ci]
            close(evidence['scores'][ci], scores, 'Test full score'); correct.append(scores.argmax(1) == y)
            expected_rows.append(row(seed, NAMES[ci], 'test', scores, y, x, model['mean']))
        require(np.array_equal(evidence['correctness'], correct), 'Saved correctness differs'); frozen = []
        for ci, name in enumerate(('original-clean', 'original-pca8')):
            scores = (xs[ci] + model['mean']) @ model['W2'] + model['b2']
            close(evidence['original_scores'][ci], scores, 'Original frozen full score'); frozen.append(scores.argmax(1) == y)
            expected_rows.append(row(seed, name, 'test', scores, y, xs[ci], model['mean']))
        pca.append(correct[1]); clean.append(correct[0]); gains.append(float(100*(np.mean(correct[1])-np.mean(frozen[1]))))
        contrasts.append(float(100*(np.mean(correct[1])-np.mean(correct[2:]))))
    expected = {'stage': 'heldout', 'cases': 360, 'rows': expected_rows, 'primary': primary(np.array(pca), np.array(clean)),
                'pca_refit_minus_original_pca_accuracy_pp_per_seed': gains, 'pca_refit_minus_mean_random_refit_accuracy_pp_per_seed': contrasts,
                'status': 'completed_pending_independent_audit', 'scientific_acceptance': False}
    actual = custody.read_json(root / 'summary.json')
    metric_domains(actual['rows'])
    for value in [actual['primary']['mean_pp'], *actual['primary']['per_seed_pp'],
                  *actual['primary']['case_resampling_interval95_pp'],
                  *actual['pca_refit_minus_original_pca_accuracy_pp_per_seed'],
                  *actual['pca_refit_minus_mean_random_refit_accuracy_pp_per_seed']]:
        finite(value, 'stored accuracy contrast', -100, 100)
    compare(actual, expected, 'Held-out summary')
    compare(custody.read_json(root / 'WORKER_RESULT.json'), {'stage': 'heldout', 'status': 'completed', 'test_conditions': 100}, 'Held-out worker result')
    return {'status': 'passed', 'stage': 'heldout', 'test_conditions': 100, 'cases': 360, 'verification_refits': 90,
            'training_manifest_sha256': train_report['manifest_sha256'], 'manifest_sha256': custody.sha256(root / 'MANIFEST.json'),
            'scope': 'Saved-evidence audit, not authorization or independent scientific acceptance'}
