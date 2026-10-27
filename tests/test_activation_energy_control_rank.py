"""Deterministic 2D algebra checks, not empirical activation/model evidence.

Assumptions: exact specified finite matrices evaluated in float64, row-vector
convention, fixed mean and fixed head. No model, dataset, fit, randomness, file
output, or historical-array access. These checks illustrate, not prove, the
almost-sure Haar statement in the companion note.
"""
import unittest

import numpy as np


class EnergyControlRankTests(unittest.TestCase):
    def setUp(self):
        self.p = np.diag([1., 0.])
        self.r = np.array([[0., -1.], [1., 0.]])
        self.z = np.array([[-1., 0.], [1., 0.]])
        self.a = np.eye(2)-self.p@self.r

    def test_matched_removed_norm_but_different_map_rank(self):
        np.testing.assert_array_equal(self.r.T@self.r, np.eye(2))
        np.testing.assert_array_equal(
            np.linalg.norm(self.z@self.p, axis=1),
            np.linalg.norm(self.z@self.p@self.r, axis=1))
        self.assertEqual(np.linalg.matrix_rank(np.eye(2)-self.p), 1)
        self.assertEqual(np.linalg.matrix_rank(self.a), 2)
        self.assertEqual(np.linalg.det(self.a), 1.)

    def test_pair_collapse_versus_invertible_edit(self):
        np.testing.assert_array_equal(self.z@(np.eye(2)-self.p), np.zeros((2, 2)))
        np.testing.assert_array_equal(self.z@self.a, [[-1., -1.], [1., 1.]])
        np.testing.assert_array_equal((self.z@self.a)@np.linalg.inv(self.a), self.z)

    def test_affine_mean_and_compensated_head(self):
        mean = np.array([3., 4.])
        h = mean+self.z
        edited = h-self.z@self.p@self.r
        np.testing.assert_array_equal(edited, mean+self.z@self.a)
        w = np.array([2., 3.]); bias = 7.
        w_new = np.linalg.solve(self.a, w)
        bias_new = bias+mean@w-mean@w_new
        np.testing.assert_array_equal(edited@w_new+bias_new, h@w+bias)

    def test_frozen_head_can_favor_invertible_control(self):
        y = np.array([False, True])
        w = np.array([1., 0.])
        self.assertEqual(float(np.mean((self.z@self.a@w > 0) == y)), 1.)
        self.assertEqual(float(np.mean((self.z@(np.eye(2)-self.p)@w > 0) == y)), .5)

    def test_almost_sure_does_not_mean_every_rotation(self):
        exceptional = np.eye(2)-self.p@np.eye(2)
        self.assertEqual(np.linalg.det(exceptional), 0.)
        self.assertEqual(np.linalg.matrix_rank(exceptional), 1)


if __name__ == '__main__':
    unittest.main()
