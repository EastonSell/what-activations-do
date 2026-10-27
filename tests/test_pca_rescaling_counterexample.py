"""Exact constructed counterexample; not an empirical study or model run.

Exercises the existing frozen-head intervention and ridge projection kernels.
No training, data download, checkpoint, random generator, or file write occurs.
Run with PYTHONDONTWRITEBYTECODE=1 and one BLAS thread. See accompanying proof.
For review-only staging outside the repository, set WAD_REPO_ROOT explicitly.
"""
import importlib.util
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


REPO = Path(os.environ.get("WAD_REPO_ROOT",
    Path(__file__).resolve().parents[1]))


def load_module(name, relative):
    source = REPO / relative
    if not source.is_file():
        raise RuntimeError(f"Archived numerical kernel not found: {source}")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module("frozen_core_for_rescaling_check",
    "studies/frozen-head-digits-20261024/source/core.py")
readout = load_module("ridge_readout_for_rescaling_check",
    "studies/ridge-readout-recovery-20261025/source/protocol_v2/readout.py")


class RescalingCounterexample(unittest.TestCase):
    def setUp(self):
        # Uniform Cartesian support: s is class signal, n is independent noise.
        self.inputs = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        self.labels = np.array([1, 1, 0, 0])
        self.hidden = np.maximum(0., self.inputs + 2.)
        self.base_weights = np.array([[1., -1.], [0., 0.]])
        self.bias = np.array([-2., 2.])
        self.scales = (np.array([4., 1.]), np.array([1., 4.]))
        self.expected_logits = np.array([[-1., 1.], [-1., 1.], [1., -1.], [1., -1.]])

    def test_relu_positive_rescaling_preserves_clean_logits(self):
        # Fails if hidden units are scaled without inversely scaling their head.
        for scale in self.scales:
            with self.subTest(scale=scale.tolist()):
                hidden = np.maximum(0., self.inputs * scale + 2. * scale)
                np.testing.assert_array_equal(hidden, self.hidden * scale)
                weights = self.base_weights / scale[:, None]
                np.testing.assert_array_equal(hidden @ weights + self.bias, self.expected_logits)
                self.assertEqual(float(np.mean((hidden @ weights + self.bias).argmax(1) == self.labels)), 1.)

    def test_same_distinct_covariance_eigenvalues_and_removed_energy(self):
        # Fails if the construction secretly changes the spectrum or tie-breaking PC.
        for scale in self.scales:
            hidden = self.hidden * scale
            mean, basis, singular = core.fit_geometry(hidden)
            np.testing.assert_allclose(singular ** 2 / 4, [16., 1.], atol=1e-13, rtol=0)
            self.assertGreater(singular[0], singular[1])
            edited = core.intervene(hidden, mean, basis, 1, "pca")
            np.testing.assert_allclose(np.sum((hidden-edited)**2, axis=1),
                                       [16., 16., 16., 16.], atol=1e-12, rtol=0)
            self.assertAlmostEqual(float(np.sum((hidden-edited)**2) / np.sum((hidden-mean)**2)), 16./17., places=14)
            self.assertEqual(np.linalg.matrix_rank(hidden-mean), 2)
            self.assertEqual(np.linalg.matrix_rank(edited-mean), 1)

    def test_signal_scaled_top_pc_deletion_loses_class(self):
        # Fails if projection removes the wrong direction or omits mean restoration.
        hidden = self.hidden * self.scales[0]
        mean, basis, _ = core.fit_geometry(hidden)
        edited = core.intervene(hidden, mean, basis, 1, "pca")
        np.testing.assert_allclose(edited, [[8., 1.], [8., 3.], [8., 1.], [8., 3.]], atol=1e-13, rtol=0)
        scores = edited @ (self.base_weights/self.scales[0][:, None]) + self.bias
        np.testing.assert_allclose(scores, np.zeros((4, 2)), atol=1e-13, rtol=0)
        # Analytic coordinates remove floating-point ambiguity at the exact tie.
        analytic_basis = np.eye(2)
        exact = core.intervene(hidden, mean, analytic_basis, 1, "pca")
        exact_scores = exact @ (self.base_weights/self.scales[0][:, None]) + self.bias
        np.testing.assert_array_equal(exact_scores, np.zeros((4, 2)))
        self.assertEqual(float(np.mean(exact_scores.argmax(1) == self.labels)), .5)
        np.testing.assert_array_equal(exact[[0, 1]], exact[[2, 3]])

    def test_nuisance_scaled_top_pc_deletion_preserves_class(self):
        hidden = self.hidden * self.scales[1]
        mean, basis, _ = core.fit_geometry(hidden)
        edited = core.intervene(hidden, mean, basis, 1, "pca")
        np.testing.assert_allclose(edited, [[1., 8.], [1., 8.], [3., 8.], [3., 8.]], atol=1e-13, rtol=0)
        scores = edited @ (self.base_weights/self.scales[1][:, None]) + self.bias
        np.testing.assert_allclose(scores, self.expected_logits, atol=1e-13, rtol=0)
        self.assertEqual(float(np.mean(scores.argmax(1) == self.labels)), 1.)

    def test_ridge_centered_projection_matches_restored_frozen_edit(self):
        # Fails if readout.project and frozen-head intervention have mismatched means.
        for scale in self.scales:
            hidden = self.hidden * scale
            mean, basis, _ = core.fit_geometry(hidden)
            centered = readout.project(hidden, mean, basis, 1)
            restored = core.intervene(hidden, mean, basis, 1, "pca")
            np.testing.assert_allclose(centered + mean, restored, atol=1e-13, rtol=0)

    def test_uncompensated_rescaling_is_not_function_preserving(self):
        # Negative control: the inverse output rescaling is an essential assumption.
        bad = (self.hidden * self.scales[0]) @ self.base_weights + self.bias
        self.assertGreater(float(np.max(np.abs(bad-self.expected_logits))), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
