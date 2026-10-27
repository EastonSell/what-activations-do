"""Deterministic numerical reference checks; never loads a language model."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1]/'examples/learning/log_softmax_reference.py'
SPEC = importlib.util.spec_from_file_location('log_softmax_reference', PATH)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class LogSoftmaxReferenceTests(unittest.TestCase):
    def test_huge_common_float32_shift_remains_uniform(self):
        logits = np.full(49152, np.float32(1e20))
        logp, probability = REFERENCE.distribution(logits)
        np.testing.assert_allclose(logp, -np.log(49152.), atol=1e-14)
        np.testing.assert_allclose(probability, 1/49152., rtol=1e-14)
        self.assertAlmostEqual(float(probability.sum()), 1.)

    def test_preserves_representable_common_shift(self):
        logits = np.array([0., 1., 2.])
        logp, probability = REFERENCE.distribution(logits)
        shifted_logp, shifted_probability = REFERENCE.distribution(logits+2**40)
        np.testing.assert_array_equal(shifted_logp, logp)
        np.testing.assert_array_equal(shifted_probability, probability)

    def test_normal_range_matches_hand_calculated_probabilities(self):
        logp, probability = REFERENCE.distribution(np.log([1., 2., 3.]))
        np.testing.assert_allclose(probability, [1/6, 2/6, 3/6], atol=1e-15)
        np.testing.assert_allclose(np.exp(logp), probability, atol=1e-15)

    def test_rejects_invalid_logits(self):
        for value in ([], [[1., 2.]], [np.inf, 0.], [np.nan, 0.], [1+2j, 0]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    REFERENCE.distribution(value)


if __name__ == '__main__':
    unittest.main()
