#!/usr/bin/env python3
"""
Unit tests for the likelihood and intersection functions.
"""

import unittest
import numpy as np
intersection = __import__('1-intersection').intersection
likelihood = __import__('0-likelihood').likelihood

class TestLikelihood(unittest.TestCase):
    def test_valid_likelihood(self):
        """Test the likelihood function with valid inputs."""
        P = np.array([0.1, 0.5, 0.9])
        x = 5
        n = 10
        result = likelihood(x, n, P)
        expected = np.array([0.00148803, 0.24609375, 0.00148803])
        np.testing.assert_almost_equal(result, expected, decimal=7)

    def test_invalid_n(self):
        """Test that the likelihood function raises an error for invalid n."""
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(5, -1, P)
        with self.assertRaises(ValueError):
            likelihood(5, 0, P)
        with self.assertRaises(ValueError):
            likelihood(5, "10", P)

    def test_invalid_x(self):
        """Test that the likelihood function raises an error for invalid x."""
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(-1, 10, P)
        with self.assertRaises(ValueError):
            likelihood("5", 10, P)
        with self.assertRaises(ValueError):
            likelihood(15, 10, P)

    def test_invalid_P(self):
        """Test that the likelihood function raises an error for invalid P."""
        with self.assertRaises(TypeError):
            likelihood(5, 10, [0.1, 0.5, 0.9])  # Not a numpy array
        with self.assertRaises(TypeError):
            likelihood(5, 10, 0.5)  # Not an array
        with self.assertRaises(ValueError):
            likelihood(5, 10, np.array([0.1, -0.5, 1.2]))  # Out of range


class TestIntersection(unittest.TestCase):
    def test_valid_intersection(self):
        """Test the intersection function with valid inputs."""
        P = np.array([0.1, 0.5, 0.9])
        Pr = np.array([0.2, 0.3, 0.5])
        x = 5
        n = 10
        result = intersection(x, n, P, Pr)
        expected = np.array([0.00029761, 0.07382813, 0.00074402])
        np.testing.assert_almost_equal(result, expected, decimal=7)

    def test_invalid_Pr(self):
        """Test that the intersection function raises an error for invalid Pr."""
        P = np.array([0.1, 0.5, 0.9])
        with self.assertRaises(TypeError):
            intersection(5, 10, P, [0.2, 0.3, 0.5])  # Not a numpy array
        with self.assertRaises(TypeError):
            intersection(5, 10, P, np.array([0.2, 0.3]))  # Mismatched shape
        with self.assertRaises(ValueError):
            intersection(5, 10, P, np.array([0.2, 0.3, -0.5]))  # Out of range
        with self.assertRaises(ValueError):
            intersection(5, 10, P, np.array([0.2, 0.3, 0.7]))  # Does not sum to 1

    def test_empty_P_and_Pr(self):
        """Test the intersection function with empty P and Pr arrays."""
        P = np.array([])
        Pr = np.array([])
        x = 5
        n = 10
        result = intersection(x, n, P, Pr)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
