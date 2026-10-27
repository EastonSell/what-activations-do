"""Prospective geometry regression tests on deterministic finite toy arrays.

No model, empirical dataset, training run, random sampling or artifact rewrite.
The historical geometry.py and its freeze files are deliberately unchanged.
"""
import unittest

import numpy as np

import geometry_v2 as geometry


class ProspectiveGeometryTests(unittest.TestCase):
    def test_ranking_rejects_broadcast_pair_counts(self):
        with self.assertRaises(ValueError):
            geometry.ranking(np.ones((3, 2)), np.zeros((1, 2)), np.ones(2))

    def test_ranking_rejects_broadcast_feature_counts(self):
        with self.assertRaises(ValueError):
            geometry.ranking(np.ones((3, 2)), np.zeros((3, 1)), np.ones(2))

    def test_ranking_rejects_nonvector_or_nonfinite_axis(self):
        positive = np.ones((3, 2)); negative = np.zeros_like(positive)
        for axis in (np.ones((2, 1)), np.array([1., np.nan]), np.ones(3)):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    geometry.ranking(positive, negative, axis)

    def test_ranking_preserves_half_credit_for_ties(self):
        positive = np.array([[1., 0.], [0., 0.], [-1., 0.]])
        self.assertEqual(geometry.ranking(positive, np.zeros_like(positive), [1., 0.]), .5)

    def test_cosine_rejects_undefined_zero_vectors(self):
        for a, b in (([[0., 0.]], [[1., 0.]]), ([[1., 0.]], [[0., 0.]])):
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError):
                    geometry.cosine(np.array(a), np.array(b))

    def test_cosine_rejects_broadcast_shapes(self):
        with self.assertRaises(ValueError):
            geometry.cosine(np.ones((3, 2)), np.ones((1, 2)))

    def test_cosine_rejects_nonfinite_coordinates(self):
        with self.assertRaises(ValueError):
            geometry.cosine(np.array([[np.nan, 0.]]), np.array([[1., 0.]]))

    def test_cosine_preserves_positive_row_scaling(self):
        a = np.array([[1., 2.], [1., -2.], [-1., 0.]])
        b = np.array([[1., 0.], [1., 2.], [1., 0.]])
        expected = [1/np.sqrt(5.), -3/5, -1.]
        scales = np.array([[1e200], [1e-200], [1e100]])
        with np.errstate(all='raise'):
            np.testing.assert_allclose(geometry.cosine(a*scales, b/scales), expected, atol=1e-15)

    def test_cka_self_similarity_at_large_and_small_scales(self):
        x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        for scale in (1e100, 1e-100):
            with self.subTest(scale=scale), np.errstate(all='raise'):
                self.assertAlmostEqual(geometry.cka(x*scale, x*scale), 1.)

    def test_cka_rectangular_hand_calculated_value(self):
        a = np.array([[1.], [0.], [-1.]])
        b = np.array([[1., 0.], [0., 1.], [-1., -1.]])
        with np.errstate(all='raise'):
            self.assertAlmostEqual(geometry.cka(a*1e-100, b*1e100), np.sqrt(5/8))

    def test_cka_rejects_constant_representations(self):
        with self.assertRaises(ValueError):
            geometry.cka(np.ones((3, 2)), np.eye(3))

    def test_cka_rejects_nonbinary_constant_despite_mean_roundoff(self):
        constant = np.full((3, 2), .1)
        with self.assertRaises(ValueError):
            geometry.cka(constant, constant)

    def test_cka_discards_exact_constant_column_before_mean_roundoff(self):
        varying = np.arange(-3., 4.)[:, None]
        with_offset = np.column_stack([np.full(7, 1e300), varying[:, 0]])
        self.assertAlmostEqual(geometry.cka(with_offset, varying), 1.)

    def test_cka_translation_and_orthogonal_rotation(self):
        a = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        q = np.array([[0., -1.], [1., 0.]])
        self.assertAlmostEqual(geometry.cka(a+7, (a@q)-11), 1.)

    def test_unchanged_training_geometry_api_is_available(self):
        x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        state = geometry.fit_transform(x)
        np.testing.assert_array_equal(geometry.transform(x, state), x)
        np.testing.assert_array_equal(geometry.direction(x, -x), np.zeros(2))


if __name__ == '__main__':
    unittest.main()
