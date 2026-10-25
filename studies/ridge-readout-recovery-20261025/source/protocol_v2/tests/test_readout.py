import unittest
import numpy as np
import readout


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(712)
        self.h = self.rng.normal(size=(60, 8))
        self.y = np.arange(60) % 3

    def test_identity_projection(self):
        mu, basis = readout.geometry(self.h)
        np.testing.assert_allclose(readout.project(self.h, mu, basis, 0), self.h-mu)

    def test_full_deletion_zero(self):
        mu, basis = readout.geometry(self.h)
        np.testing.assert_allclose(readout.project(self.h, mu, basis, 8), 0, atol=1e-12)

    def test_projection_orthogonality_and_rank(self):
        mu, basis = readout.geometry(self.h)
        z = readout.project(self.h, mu, basis, 2)
        np.testing.assert_allclose(z @ basis[:, :2], 0, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(z), 6)

    def test_training_mean_only(self):
        mu, basis = readout.geometry(self.h)
        new = self.rng.normal(size=(4, 8)) + 100
        np.testing.assert_allclose(readout.project(new, mu, basis, 0), new-self.h.mean(0))

    def test_fit_matches_augmented_normal_equations(self):
        x = self.h + 3
        fit = readout.ridge(x, self.y, 3, .01)
        augmented = np.column_stack([x, np.ones(len(x))])
        penalty = np.diag([len(x)*.01]*8+[0])
        expected = np.linalg.solve(augmented.T@augmented+penalty,
                                   augmented.T@np.eye(3)[self.y])
        np.testing.assert_allclose(fit['weights'], expected[:-1], atol=1e-11)
        np.testing.assert_allclose(fit['intercept'], expected[-1], atol=1e-11)

    def test_ridge_stationarity(self):
        fit = readout.ridge(self.h, self.y, 3, .01)
        residual = readout.predict_scores(fit, self.h)-np.eye(3)[self.y]
        grad = self.h.T@residual/len(self.h)+.01*fit['weights']
        np.testing.assert_allclose(grad, 0, atol=1e-12)
        np.testing.assert_allclose(residual.mean(0), 0, atol=1e-12)

    def test_normal_equation_residual_defined_and_small(self):
        fit = readout.ridge(self.h, self.y, 3, .01)
        xc = self.h-self.h.mean(0)
        target = np.eye(3)[self.y]
        rhs = xc.T@(target-target.mean(0))
        gram = xc.T@xc+len(xc)*.01*np.eye(8)
        expected = np.max(np.abs(gram@fit['weights']-rhs))/max(1., np.max(np.abs(rhs)))
        self.assertEqual(fit['normal_equation_relative_max_abs'], expected)
        self.assertLessEqual(expected, 1e-10)

    def test_prediction_overflow_rejected_before_argmax(self):
        fit = {'weights': np.array([[1e308, 1e308]]), 'intercept': np.zeros(2)}
        with self.assertRaises(FloatingPointError):
            readout.predict_scores(fit, np.array([[2.]]))
        with self.assertRaises(FloatingPointError):
            readout.correct(fit, np.array([[2.]]), np.array([0]))

    def test_ridge_intermediate_overflow_rejected(self):
        with self.assertRaises(FloatingPointError):
            readout.ridge(np.array([[1e200], [-1e200]]), np.array([0, 1]), 2, .01)

    def test_geometry_and_projection_overflow_rejected(self):
        with self.assertRaises(FloatingPointError):
            readout.geometry(np.full((4, 2), 1e308))
        with self.assertRaises(FloatingPointError):
            readout.project(np.full((2, 2), 1e308), np.full(2, -1e308), np.eye(2), 1)

    def test_support_diagnostic_restores_training_mean(self):
        restored = np.array([[0., 2.], [2., 0.]])
        mean = np.array([1., 1.])
        r = readout.support_diagnostics(restored-mean, mean)
        self.assertEqual(r['strict_negative_coordinate_fraction'], 0.)
        self.assertEqual(r['strict_any_negative_case_fraction'], 0.)
        self.assertEqual(r['coordinate_denominator'], 4)
        self.assertEqual(r['case_denominator'], 2)

    def test_support_diagnostic_distinguishes_roundoff(self):
        restored = np.array([[-1e-12, 2.], [-1e-2, 0.]])
        r = readout.support_diagnostics(restored, np.zeros(2))
        self.assertEqual(r['strict_negative_coordinate_fraction'], .5)
        self.assertEqual(r['strict_any_negative_case_fraction'], 1.)
        self.assertEqual(r['material_negative_coordinate_fraction'], .25)
        self.assertEqual(r['material_any_negative_case_fraction'], .5)

    def test_translation_equivariance(self):
        base = readout.ridge(self.h, self.y, 3, .01)
        shifted = readout.ridge(self.h+11, self.y, 3, .01)
        np.testing.assert_allclose(readout.predict_scores(base, self.h),
                                   readout.predict_scores(shifted, self.h+11), atol=1e-11)

    def test_rotation_equivariance(self):
        q = readout.random_basis(101, 0, 8)
        a = readout.ridge(self.h, self.y, 3, .01)
        b = readout.ridge(self.h@q, self.y, 3, .01)
        np.testing.assert_allclose(readout.predict_scores(a, self.h),
                                   readout.predict_scores(b, self.h@q), atol=1e-11)

    def test_zero_features_give_class_priors(self):
        y = np.array([0, 0, 0, 1])
        fit = readout.ridge(np.zeros((4, 8)), y, 3, .01)
        np.testing.assert_allclose(readout.predict_scores(fit, np.zeros((2, 8))),
                                   [[.75, .25, 0], [.75, .25, 0]])

    def test_input_arrays_not_mutated(self):
        h, y = self.h.copy(), self.y.copy()
        mu, basis = readout.geometry(h)
        readout.ridge(readout.project(h, mu, basis, 2), y, 3, .01)
        np.testing.assert_array_equal(self.h, h)
        np.testing.assert_array_equal(self.y, y)

    def test_identical_seed_basis(self):
        a, b = readout.random_basis(101, 4, 8), readout.random_basis(101, 4, 8)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a.T@a, np.eye(8), atol=1e-12)
        self.assertFalse(np.array_equal(a, readout.random_basis(101, 5, 8)))

    def test_invalid_matrix_and_labels_rejected(self):
        for bad in [np.array([[np.nan]]), np.empty((0, 8)), np.ones(4)]:
            with self.assertRaises(ValueError): readout.geometry(bad)
        for bad in [self.y.astype(float), self.y[:-1], self.y+3]:
            with self.assertRaises(ValueError): readout.ridge(self.h, bad, 3, .01)
        with self.assertRaises(ValueError): readout.ridge(self.h, self.y, 3, 0)

    def test_nonorthogonal_basis_rejected(self):
        mu, basis = readout.geometry(self.h)
        with self.assertRaises(ValueError): readout.project(self.h, mu, basis*2, 2)


class ScientificChecks(unittest.TestCase):
    def test_redundant_signal_recovers_after_changed_readout(self):
        # Label is redundantly present in two coordinates. Frozen readout uses
        # only the removed coordinate; a new linear readout can use the other.
        y = np.array([0, 1]*30)
        signed = 2*y-1
        h = np.column_stack([10*signed, signed]).astype(float)
        z = readout.project(h, np.zeros(2), np.eye(2), 1)
        frozen = z@np.array([1., 0.])
        self.assertEqual(float(np.mean((frozen>0)==y)), .5)
        fit = readout.ridge(z, y, 2, .01)
        self.assertEqual(float(readout.correct(fit, z, y).mean()), 1.)

    def test_unique_signal_is_not_recoverable_by_readout(self):
        y = np.array([0, 1]*30)
        h = np.column_stack([2*y-1, np.zeros(60)]).astype(float)
        z = readout.project(h, np.zeros(2), np.eye(2), 1)
        fit = readout.ridge(z, y, 2, .01)
        self.assertEqual(float(readout.correct(fit, z, y).mean()), .5)

    def test_gates_reject_one_failed_seed_without_dropping(self):
        original = np.full(5, .96)
        self.assertTrue(readout.eligibility(original, np.full(5, .95))['eligible'])
        self.assertFalse(readout.eligibility(original, [.95, .95, .93, .95, .95])['eligible'])

    def test_primary_null_has_zero_interval(self):
        a = np.ones((5, 360), dtype=bool)
        r = readout.primary(a, a)
        self.assertEqual(r['mean_pp'], 0)
        self.assertEqual(r['case_resampling_interval95_pp'], [0., 0.])
        self.assertEqual(r['decision'], 'near_baseline_decodability_sensitivity_rule_met')

    def test_primary_adverse_direction_and_denominator(self):
        base = np.ones((5, 360), dtype=bool)
        pca = base.copy(); pca[:, :72] = False
        r = readout.primary(pca, base)
        self.assertAlmostEqual(r['mean_pp'], -20.)
        self.assertEqual(r['decision'], 'two_point_margin_not_met_by_sensitivity_rule')
        with self.assertRaises(ValueError): readout.primary(pca[:, :-1], base[:, :-1])
        with self.assertRaises(ValueError): readout.primary(pca.astype(float), base)

    def test_seed_gate_blocks_averaged_success(self):
        base = np.ones((5, 360), dtype=bool)
        pca = base.copy(); pca[0, :22] = False
        r = readout.primary(pca, base)
        self.assertGreater(r['case_resampling_interval95_pp'][0], -2)
        self.assertEqual(r['decision'], 'inconclusive')

    def test_exact_case_pairing_not_independent_arm_bootstrap(self):
        a = np.zeros((5, 360), dtype=bool); a[:, ::2] = True
        result = readout.primary(a, a)
        self.assertEqual(result['case_resampling_interval95_pp'], [0., 0.])


if __name__ == '__main__': unittest.main()
