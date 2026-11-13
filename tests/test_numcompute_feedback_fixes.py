"""Regression tests for the issues raised in the Assignment 2.1 marking feedback.

Each test reproduces one probe from the feedback and asserts the *exact*
expected answer:

* StandardScaler must standardise a tiny but non-constant feature
  ([1e-16, 2e-16, 3e-16]) to unit variance.
* Euclidean distance between 1e9 and 1e9 + 1 must be 1, not 0.
* grad / jacobian must reject a NaN (or otherwise invalid) step size.

Written with ``unittest`` so it runs under both pytest and
``python -m unittest``.
"""

import unittest
import warnings

import numpy as np

from numcompute.optim import grad, gradient_descent, jacobian, line_search
from numcompute.preprocessing import MinMaxScaler, StandardScaler
from numcompute.utils import pairwise_distances

NAN, INF = float("nan"), float("inf")


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


class TestDistancesWithLargeCoordinates(unittest.TestCase):

    def test_feedback_probe_exact(self):
        self.assertEqual(pairwise_distances([[1e9]], [[1e9 + 1]])[0, 0], 1.0)
        self.assertEqual(pairwise_distances([[1e9, 5.0]], [[1e9 + 1, 5.0]],
                                            metric="sqeuclidean")[0, 0], 1.0)

    def test_known_distances_at_many_offsets(self):
        for offset in (0.0, 1e6, 1e9, 1e12, -1e15):
            with self.subTest(offset=offset):
                A = np.array([[offset, offset], [offset + 3.0, offset + 4.0]])
                np.testing.assert_array_equal(pairwise_distances(A), [[0, 5], [5, 0]])

    def test_tiny_coordinates_do_not_underflow(self):
        self.assertAlmostEqual(pairwise_distances([[0.0]], [[3e-200]])[0, 0] / 3e-200, 1.0,
                               places=12)

    def test_matches_direct_loop(self):
        rng = np.random.default_rng(0)
        A, B = rng.normal(size=(40, 5)), rng.normal(size=(25, 5))
        ref = np.array([[np.sqrt(np.sum((a - b) ** 2)) for b in B] for a in A])
        np.testing.assert_allclose(pairwise_distances(A, B), ref, rtol=1e-12)
        np.testing.assert_allclose(pairwise_distances(A, B, metric="sqeuclidean"), ref ** 2,
                                   rtol=1e-12)


class TestOptimiserStepValidation(unittest.TestCase):

    def quad(self, x):
        return float(np.sum(np.asarray(x) ** 2))

    def test_invalid_steps_rejected(self):
        for h in (NAN, INF, -INF, 0.0, -1e-5):
            for fn in (grad, jacobian):
                with self.subTest(fn=fn.__name__, h=h):
                    with self.assertRaisesRegex(ValueError, "h must be a finite number > 0"):
                        fn(self.quad, [1.0, 2.0], h=h)
        with self.assertRaises(TypeError):
            grad(self.quad, [1.0, 2.0], h=True)
        with self.assertRaises(TypeError):
            grad(self.quad, [1.0, 2.0], h="1e-5")

    def test_valid_step_still_exact(self):
        np.testing.assert_allclose(grad(self.quad, [1.0, -2.0], h=1e-5), [2.0, -4.0], rtol=1e-8)

    def test_line_search_and_descent_parameters(self):
        with self.assertRaises(ValueError):
            line_search(self.quad, [1.0], [-1.0], alpha0=NAN)
        with self.assertRaises(ValueError):
            line_search(self.quad, [1.0], [-1.0], max_iter=0)
        with self.assertRaises(ValueError):
            gradient_descent(self.quad, [1.0], tol=NAN)
        with self.assertRaises(ValueError):
            gradient_descent(self.quad, [1.0], alpha0=0.0)
        result = gradient_descent(self.quad, [3.0, -1.0])
        np.testing.assert_allclose(result["x"], [0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
