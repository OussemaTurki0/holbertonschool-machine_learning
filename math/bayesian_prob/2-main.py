#!/usr/bin/env python3
"""
Unit tests for the likelihood, intersection, and marginal functions.
"""

import unittest
import numpy as np
marginal = __import__('2-marginal').marginal

class TestLikelihoodFunctions(unittest.TestCase):

    def test_likelihood_valid(self):
        P = np.array([0.1, 0.5, 0.9])
        x = 3
        n = 5
        expected = np.array([0.0081, 0.3125, 0.0729])
        result = likelihood(x, n, P)
        np.testing.assert_almost_equal(result, expected, decimal=4)

    def test_likelihood_invalid_n(self):
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(3, -1, P)

    def test_likelihood_invalid_x(self):
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(-1, 5, P)

    def test_likelihood_x_greater_than_n(self):
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(6, 5, P)

    def test_likelihood_invalid_P_type(self):
        with self.assertRaises(TypeError):
            likelihood(3, 5, [0.1, 0.5])

    def test_intersection_valid(self):
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.2, 0.5, 0.3])
        x = 3
        n = 5
        expected = np.array([0.00162, 0.15625, 0.02187])
        result = intersection(x, n, P, Pr)
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_intersection_invalid_Pr(self):
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.3, 0.3, 0.3])
        with self.assertRaises(ValueError):
            intersection(3, 5, P, Pr)

    def test_marginal_valid(self):
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.2, 0.5, 0.3])
        x = 3
        n = 5
        result = marginal(x, n, P, Pr)
        expected = 0.17974
        self.assertAlmostEqual(result, expected, places=5)

    def test_marginal_invalid_Pr_sum(self):
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.2, 0.4, 0.3])
        with self.assertRaises(ValueError):
            marginal(3, 5, P, Pr)

    def test_marginal_invalid_Pr_type(self):
        P = np.array([0.1, 0.5, 0.9])
        Pr = [0.2, 0.5, 0.3]  # Not a numpy array
        with self.assertRaises(TypeError):
            marginal(3, 5, P, Pr)

if __name__ == '__main__':
    unittest.main()
