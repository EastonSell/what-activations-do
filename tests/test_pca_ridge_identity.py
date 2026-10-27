"""Deterministic algebra checks, not digit/model experiments.

Normally run from the repository: python -B -m unittest discover -s tests
    -p test_pca_ridge_identity.py -v

For review-only staging outside the repository, set WAD_REPO_ROOT explicitly.
No source, data, model or package file is written by these tests.
"""
import importlib.util
import itertools
import os
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "These optional numerical algebra checks require NumPy already installed; "
        "the standard-library publication checks are separate."
    ) from exc

ROOT = Path(os.environ.get("WAD_REPO_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = ROOT / "studies/ridge-readout-recovery-20261025/source/protocol_v2/readout.py"
if not SOURCE.is_file():
    raise RuntimeError(f"Archived numerical kernel not found: {SOURCE}")
spec = importlib.util.spec_from_file_location("archived_pca_ridge_readout", SOURCE)
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)


class PCARidgeIdentity(unittest.TestCase):
    def setUp(self):
        factors = np.array(list(itertools.product((-1., 1.), repeat=3)))
        self.x = factors * np.array([3., 2., 1.])
        self.y = ((factors[:, 0] > 0).astype(int) * 2
                  + (factors[:, 1] > 0).astype(int))
        self.lam = .01

    def fit_pair(self, x=None, removed=1, shift=None):
        x = self.x if x is None else x
        shift = np.zeros(x.shape[1]) if shift is None else np.asarray(shift)
        h = x + shift
        mu, q = readout.geometry(h)
        clean_x = readout.project(h, mu, q, 0)
        edit_x = readout.project(h, mu, q, removed)
        clean = readout.ridge(clean_x, self.y, 4, self.lam)
        edited = readout.ridge(edit_x, self.y, 4, self.lam)
        p = np.eye(x.shape[1]) - q[:, :removed] @ q[:, :removed].T
        return clean_x, edit_x, q, p, clean, edited

    def assert_identity(self, result):
        x, xp, q, p, clean, edited = result
        np.testing.assert_allclose(edited["weights"], p @ clean["weights"],
                                   atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(edited["intercept"], clean["intercept"],
                                   atol=2e-12, rtol=2e-12)
        # These are invented new feature vectors, not rows used in fitting.
        new_x = np.array([[.3, -1.7, 2.1], [4.2, .1, -.7]])
        np.testing.assert_allclose(
            readout.predict_scores(edited, new_x @ p),
            readout.predict_scores(clean, new_x @ p), atol=2e-12, rtol=2e-12)

    def test_pca_deleted_weights_and_all_vector_prediction_identity(self):
        self.assert_identity(self.fit_pair(removed=1))

    def test_no_deletion(self):
        self.assert_identity(self.fit_pair(removed=0))

    def test_all_directions_deleted(self):
        result = self.fit_pair(removed=3)
        self.assert_identity(result)
        np.testing.assert_allclose(result[-1]["weights"], 0, atol=2e-12)
        np.testing.assert_allclose(result[-1]["intercept"], .25, atol=2e-12)

    def test_rank_deficient_training_matrix(self):
        x = self.x.copy()
        x[:, 2] = 0
        self.assert_identity(self.fit_pair(x=x, removed=1))

    def test_rotated_direction_inside_tied_positive_eigenspace(self):
        x = self.x.copy()
        x[:, 1] *= 1.5  # First two covariance eigenvalues now both equal nine.
        u = np.array([.6, .8, 0.])
        p = np.eye(3) - np.outer(u, u)
        c = x.T @ x / len(x)
        np.testing.assert_allclose(c @ p, p @ c, atol=2e-12, rtol=2e-12)
        clean = readout.ridge(x, self.y, 4, self.lam)
        edited = readout.ridge(x @ p, self.y, 4, self.lam)
        self.assert_identity((x, x @ p, None, p, clean, edited))

    def test_nonzero_raw_mean_and_intercept_coordinate_change(self):
        shift = np.array([20., 40., 60.])
        result = self.fit_pair(shift=shift, removed=1)
        self.assert_identity(result)
        centered_fit = result[-2]
        raw_fit = readout.ridge(self.x + shift, self.y, 4, self.lam)
        np.testing.assert_allclose(raw_fit["weights"], centered_fit["weights"])
        np.testing.assert_allclose(raw_fit["intercept"],
                                   centered_fit["intercept"] - shift @ raw_fit["weights"])
        np.testing.assert_allclose(readout.predict_scores(raw_fit, self.x + shift),
                                   readout.predict_scores(centered_fit, self.x))

    def test_minimized_objective_spectral_decomposition(self):
        x, xp, q, p, clean, edited = self.fit_pair(removed=2)
        yc = np.eye(4)[self.y] - .25
        c = x.T @ x / len(x)
        d = x.T @ yc / len(x)
        nu = np.diag(q.T @ c @ q)
        coordinates = q.T @ d
        expected = .5 * np.sum(np.sum(coordinates[:2] ** 2, axis=1)
                               / (nu[:2] + self.lam))
        self.assertAlmostEqual(edited["objective"] - clean["objective"],
                               float(expected), places=12)

    def test_noncommuting_projection_requires_cross_covariance_compensation(self):
        c = self.x.T @ self.x / len(self.x)
        u = np.array([1., 1., 0.]) / np.sqrt(2.)
        p = np.eye(3) - np.outer(u, u)
        clean = readout.ridge(self.x, self.y, 4, self.lam)
        edited = readout.ridge(self.x @ p, self.y, 4, self.lam)
        actual = edited["weights"] - p @ clean["weights"]
        expected = np.linalg.solve(p @ c @ p + self.lam * np.eye(3),
                                   p @ c @ (np.eye(3)-p) @ clean["weights"])
        self.assertGreater(float(np.linalg.norm(c @ p - p @ c)), 1.)
        self.assertGreater(float(np.linalg.norm(actual)), 1e-3)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_changed_penalty_breaks_same_weight_identity(self):
        x, xp, q, p, clean, edited = self.fit_pair(removed=1)
        different = readout.ridge(xp, self.y, 4, .5)
        self.assertGreater(float(np.linalg.norm(different["weights"]
                                                - p @ clean["weights"])), 1e-3)

    def test_imperfect_solve_residual_identity_and_bound(self):
        x = self.x
        c = x.T @ x / len(x)
        d = x.T @ (np.eye(4)[self.y] - .25) / len(x)
        u = np.array([1., 1., 0.]) / np.sqrt(2.)
        p = np.eye(3) - np.outer(u, u)
        a = c + self.lam * np.eye(3)
        ap = p @ c @ p + self.lam * np.eye(3)
        # Deliberately perturb both solves, so the residual terms are nonzero.
        w = readout.ridge(x, self.y, 4, self.lam)["weights"] + 1e-5
        wp = readout.ridge(x @ p, self.y, 4, self.lam)["weights"] - 2e-5
        r, rp = a @ w - d, ap @ wp - p @ d
        coupling = p @ c @ (np.eye(3)-p) @ w
        np.testing.assert_allclose(ap @ (wp-p @ w), coupling + rp-p @ r,
                                   atol=2e-12, rtol=2e-12)
        upper = (np.linalg.norm(coupling) + np.linalg.norm(rp)
                 + np.linalg.norm(p @ r)) / self.lam
        self.assertLessEqual(float(np.linalg.norm(wp-p @ w)), float(upper)+1e-12)

    def test_positive_affine_loss_does_not_imply_nonlinear_information_loss(self):
        t = np.array([-2., -1., 1., 2.])
        x = np.column_stack((t, 2*t*t-5))
        mu, q = readout.geometry(x)
        xp = readout.project(x, mu, q, 1)  # Removes variance-9 second coordinate.
        np.testing.assert_allclose(xp[:, 0], t, atol=2e-12)
        np.testing.assert_allclose(xp[:, 1], 0, atol=2e-12)
        np.testing.assert_allclose(np.mean(x[:, 0]*x[:, 1]), 0, atol=2e-12)
        # An arbitrary affine function of retained t cannot reproduce x2:
        # x2 is centered and orthogonal to both the constant and t.
        self.assertAlmostEqual(float(np.mean(x[:, 1]**2)), 9.)
        # Yet the nonlinear map reconstructs it exactly: no information lost.
        np.testing.assert_allclose(2*xp[:, 0]**2-5, x[:, 1], atol=2e-12)


if __name__ == "__main__":
    unittest.main()
