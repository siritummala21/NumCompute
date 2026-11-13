"""Regression tests for the issues raised in the Assignment 2.1 marking feedback.

Each test reproduces one probe from the feedback and asserts the *exact*
expected answer:

* StandardScaler must standardise a tiny but non-constant feature
  ([1e-16, 2e-16, 3e-16]) to unit variance.

Written with ``unittest`` so it runs under both pytest and
``python -m unittest``.
"""

import unittest
import warnings

import numpy as np

from numcompute.preprocessing import MinMaxScaler, StandardScaler


class TestScalerTinyAndHugeValues(unittest.TestCase):

    def test_tiny_nonconstant_feature_has_unit_variance(self):
        X = np.array([[1e-16], [2e-16], [3e-16]])
        Z = StandardScaler().fit(X).transform(X)
        np.testing.assert_allclose(Z.ravel(), [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], rtol=1e-12)
        self.assertAlmostEqual(float(Z.std()), 1.0, places=12)

    def test_extreme_magnitudes(self):
        for X in (np.array([[1e-170], [2e-170], [3e-170]]), np.array([[-1e308], [0.0], [1e308]])):
            with self.subTest(magnitude=float(np.abs(X).max())):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    Z = StandardScaler().fit(X).transform(X)
                np.testing.assert_allclose(Z.ravel(), [-np.sqrt(1.5), 0.0, np.sqrt(1.5)],
                                           rtol=1e-12)

    def test_truly_constant_column_still_maps_to_zero(self):
        X = np.column_stack([np.full(5, 0.1), np.arange(5.0)])
        sc = StandardScaler().fit(X)
        self.assertEqual(sc.scale_[0], 1.0)
        np.testing.assert_array_equal(sc.transform(X)[:, 0], 0.0)

    def test_minmax_tiny_and_huge_ranges(self):
        tiny = np.array([[1e-16], [2e-16], [3e-16]])
        np.testing.assert_allclose(MinMaxScaler().fit(tiny).transform(tiny).ravel(), [0, 0.5, 1])
        huge = np.array([[-1e308], [0.0], [1e308]])
        np.testing.assert_allclose(MinMaxScaler().fit(huge).transform(huge).ravel(), [0, 0.5, 1])


if __name__ == "__main__":
    unittest.main()
