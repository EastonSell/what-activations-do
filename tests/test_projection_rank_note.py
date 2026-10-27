"""Synthetic checks of the exact-rank note, not a model experiment."""
import unittest
import numpy as np


class ProjectionRankNoteTests(unittest.TestCase):
    def test_random_and_pca_deletions_have_distinct_design_ranks(self):
        # Exactly rank 3 in R^6, with distinct positive singular values.
        x = np.diag([4., 3., 2., 0., 0., 0.])
        _, _, vt = np.linalg.svd(x, full_matrices=True)
        for k in (0, 1, 3, 5, 6):
            m = 6 - k
            pca = np.eye(6) - vt[:k].T @ vt[:k]
            self.assertEqual(np.linalg.matrix_rank(x @ pca, tol=1e-10), 3 - min(k, 3))
            if m == 0:
                q = np.zeros((6, 6))
            else:
                u, _ = np.linalg.qr(np.random.default_rng(42 + k).normal(size=(6, m)))
                q = u @ u.T
            self.assertEqual(np.linalg.matrix_rank(x @ q, tol=1e-10), min(3, m))

    def test_exceptional_aligned_projection_does_not_satisfy_generic_rank(self):
        x = np.diag([1., 1., 0., 0.])
        q = np.diag([0., 0., 1., 1.])
        self.assertEqual(np.linalg.matrix_rank(x @ q), 0)
        self.assertEqual(min(np.linalg.matrix_rank(x), 2), 2)

    def test_finite_points_separate_even_when_design_rank_is_one(self):
        x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        u, _ = np.linalg.qr(np.random.default_rng(71).normal(size=(3, 1)))
        y = x @ u
        self.assertEqual(np.linalg.matrix_rank(y, tol=1e-10), 1)
        self.assertGreater(min(abs(float(y[i, 0] - y[j, 0]))
                               for i in range(len(y)) for j in range(i)), 1e-10)
        labels = [0, 1, 1, 0]
        lookup = {tuple(row): label for row, label in zip(y, labels)}
        self.assertEqual([lookup[tuple(row)] for row in y], labels)

    def test_rank_depends_on_numerical_threshold(self):
        x = np.diag([1., 1e-12])
        self.assertEqual(np.linalg.matrix_rank(x, tol=1e-14), 2)
        self.assertEqual(np.linalg.matrix_rank(x, tol=1e-10), 1)


if __name__ == '__main__':
    unittest.main()
