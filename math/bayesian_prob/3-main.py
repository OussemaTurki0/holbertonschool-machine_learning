#!/usr/bin/env python3
"""
Unit tests for likelihood, intersection, marginal, and posterior functions.
"""

import unittest
import numpy as np
posterior = __import__('3-posterior').posterior

class TestBayesianCalculations(unittest.TestCase):

    def test_likelihood_valid(self):
        x, n = 5, 10
        P = np.array([0.1, 0.5, 0.9])
        expected = np.array([0.00148803, 0.24609375, 0.00000001])
        result = likelihood(x, n, P)
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_intersection_valid(self):
        x, n = 5, 10
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.3, 0.4, 0.3])
        expected = likelihood(x, n, P) * Pr
        result = intersection(x, n, P, Pr)
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_marginal_valid(self):
        x, n = 5, 10
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.3, 0.4, 0.3])
        intersection_values = intersection(x, n, P, Pr)
        expected = np.sum(intersection_values)
        result = marginal(x, n, P, Pr)
        self.assertAlmostEqual(result, expected, places=8)

    def test_posterior_valid(self):
        x, n = 5, 10
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.3, 0.4, 0.3])
        expected = intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
        result = posterior(x, n, P, Pr)
        np.testing.assert_almost_equal(result, expected, decimal=8)

    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            likelihood(3, -5, np.array([0.5]))

    def test_invalid_x(self):
        with self.assertRaises(ValueError):
            likelihood(-1, 5, np.array([0.5]))

    def test_x_greater_than_n(self):
        with self.assertRaises(ValueError):
            likelihood(6, 5, np.array([0.5]))

    def test_invalid_P_type(self):
        with self.assertRaises(TypeError):
            likelihood(3, 5, [0.5, 0.7])

    def test_invalid_Pr_shape(self):
        with self.assertRaises(TypeError):
            intersection(3, 5, np.array([0.5]), np.array([[0.3, 0.7]]))

    def test_invalid_probability_values(self):
        with self.assertRaises(ValueError):
            likelihood(3, 5, np.array([1.2, -0.1]))

    def test_Pr_does_not_sum_to_1(self):
        with self.assertRaises(ValueError):
            posterior(3, 5, np.array([0.3, 0.7]), np.array([0.6, 0.6]))


if __name__ == "__main__":
    unittest.main()
