"""Checks only published aggregate tables; does not load or reconstruct arrays."""
import csv
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 211, 307, 401, 503)
CONDITIONS = ('untouched', 'pca8') + tuple(f'random8-{n:02}' for n in range(16))


class PublicSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'report/RESULTS.json').read_text())
        with (ROOT / 'report/ALL_CONDITION_METRICS.csv').open(newline='') as f:
            cls.metrics = list(csv.DictReader(f))
        with (ROOT / 'report/ALL_FIT_DIAGNOSTICS.csv').open(newline='') as f:
            cls.fits = list(csv.DictReader(f))

    def test_complete_unique_metric_and_fit_coverage(self):
        expected = {(s, c, split) for s in SEEDS for c in CONDITIONS for split in ('train', 'validation', 'test')}
        expected |= {(s, c, 'test') for s in SEEDS for c in ('original-clean', 'original-pca8')}
        keys = [(int(r['seed']), r['condition'], r['split']) for r in self.metrics]
        self.assertEqual(len(keys), 280)
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(set(keys), expected)
        fit_keys = [(int(r['seed']), r['condition']) for r in self.fits]
        self.assertEqual(len(fit_keys), 90)
        self.assertEqual(set(fit_keys), {(s, c) for s in SEEDS for c in CONDITIONS})

    def test_accuracy_and_support_integer_denominators(self):
        for r in self.metrics:
            n = int(r['cases'])
            self.assertEqual(n, {'train': 1078, 'validation': 359, 'test': 360}[r['split']])
            self.assertEqual(float(r['accuracy']), int(r['correct_count']) / n)
            self.assertEqual(int(r['coordinate_denominator']), n * 32)
            self.assertEqual(int(r['case_denominator']), n)
            for unit, denominator in (('coordinate', n * 32), ('any_negative_case', n)):
                suffix = 'negative_coordinate_count' if unit == 'coordinate' else 'any_negative_case_count'
                strict, material = int(r['strict_' + suffix]), int(r['material_' + suffix])
                self.assertTrue(0 <= material <= strict <= denominator)

    def test_report_means_and_distinct_primary_rescue_contrasts(self):
        counts = {}
        for condition in ('untouched', 'pca8', 'original-clean', 'original-pca8'):
            counts[condition] = sum(int(r['correct_count']) for r in self.metrics if r['split'] == 'test' and r['condition'] == condition)
            self.assertAlmostEqual(counts[condition] / 1800, self.result['mean_accuracy'][condition], places=14)
        self.assertEqual(counts, {'untouched': 1721, 'pca8': 758, 'original-clean': 1750, 'original-pca8': 272})
        self.assertEqual(100 * (counts['pca8'] - counts['untouched']) / 1800, self.result['primary']['mean_pp'])
        self.assertEqual(100 * (counts['pca8'] - counts['original-pca8']) / 1800, self.result['mean_pca8_refit_minus_original_pca8_pp'])
        random_count = sum(int(r['correct_count']) for r in self.metrics if r['split'] == 'test' and r['condition'].startswith('random8-'))
        self.assertEqual(random_count, 27393)
        self.assertAlmostEqual(random_count / 28800, self.result['mean_random8_ridge_accuracy'], places=14)

    def test_saved_fit_diagnostics_obey_fixed_gates_and_report_ranks(self):
        expected_ranks = {r['seed']: r for r in self.result['per_seed']}
        for r in self.fits:
            for key, upper in (('normal_equation_relative_max_abs', 1e-10), ('stationarity_max_abs', 1e-9), ('gram_condition_number', 1e12)):
                value = float(r[key])
                self.assertTrue(math.isfinite(value) and 0 <= value <= upper)
            seed = expected_ranks[int(r['seed'])]
            if r['condition'] == 'untouched':
                self.assertEqual(int(r['actual_train_rank']), seed['untouched_numerical_train_rank'])
            elif r['condition'] == 'pca8':
                self.assertEqual(int(r['actual_train_rank']), seed['pca8_numerical_train_rank'])
            else:
                self.assertIn(int(r['actual_train_rank']), seed['random8_numerical_train_ranks'])


if __name__ == '__main__':
    unittest.main()
