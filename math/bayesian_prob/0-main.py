#!/usr/bin/env python3
"""
Unit tests for the likelihood function from 0-likelihood.py.
"""

import unittest
import numpy as np
likelihood = __import__('0-likelihood').likelihood

class TestLikelihood(unittest.TestCase):
    def test_valid_inputs(self):
        """Test the likelihood function with valid inputs."""
        P = np.array([0.1, 0.5, 0.9])
        x = 5
        n = 10
        result = likelihood(x, n, P)
        expected = np.array([0.00148803, 0.24609375, 0.00148803])
        np.testing.assert_almost_equal(result, expected, decimal=7)

    def test_invalid_n(self):
        """Test that the function raises an error for invalid n."""
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(5, -1, P)
        with self.assertRaises(ValueError):
            likelihood(5, 0, P)
        with self.assertRaises(ValueError):
            likelihood(5, "10", P)

    def test_invalid_x(self):
        """Test that the function raises an error for invalid x."""
        P = np.array([0.5])
        with self.assertRaises(ValueError):
            likelihood(-1, 10, P)
        with self.assertRaises(ValueError):
            likelihood("5", 10, P)
        with self.assertRaises(ValueError):
            likelihood(15, 10, P)

    def test_invalid_P(self):
        """Test that the function raises an error for invalid P."""
        with self.assertRaises(TypeError):
            likelihood(5, 10, [0.1, 0.5, 0.9])  # Not a numpy array
        with self.assertRaises(TypeError):
            likelihood(5, 10, 0.5)  # Not an array
        with self.assertRaises(ValueError):
            likelihood(5, 10, np.array([0.1, -0.5, 1.2]))  # Out of range

    def test_empty_P(self):
        """Test that the function works with an empty array."""
        P = np.array([])
        x = 5
        n = 10
        result = likelihood(x, n, P)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
