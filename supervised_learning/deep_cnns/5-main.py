#!/usr/bin/env python3
"""
Test suite for the dense_block module
"""

import unittest
import tensorflow as tf
from dense_block import dense_block


class TestDenseBlock(unittest.TestCase):
    def setUp(self):
        """
        Setup initial parameters for testing
        """
        self.input_shape = (1, 32, 32, 64)  # Batch size 1, 32x32 image with 64 filters
        self.X = tf.random.uniform(self.input_shape)
        self.nb_filters = 64
        self.growth_rate = 32
        self.layers = 4

    def test_output_shape(self):
        """
        Test that the dense block outputs the correct shape
        """
        output, filters = dense_block(self.X, self.nb_filters, self.growth_rate, self.layers)
        expected_filters = self.nb_filters + self.growth_rate * self.layers
        self.assertEqual(output.shape, (1, 32, 32, expected_filters))
        self.assertEqual(filters, expected_filters)

    def test_filter_increase(self):
        """
        Test that the number of filters increases by the growth rate each layer
        """
        _, filters = dense_block(self.X, self.nb_filters, self.growth_rate, 1)
        self.assertEqual(filters, self.nb_filters + self.growth_rate)

    def test_output_type(self):
        """
        Test that the output is a TensorFlow tensor
        """
        output, _ = dense_block(self.X, self.nb_filters, self.growth_rate, self.layers)
        self.assertIsInstance(output, tf.Tensor)

    def test_concatenation(self):
        """
        Test that the output contains concatenated features from all layers
        """
        output, _ = dense_block(self.X, self.nb_filters, self.growth_rate, self.layers)
        # Check that the output is formed by concatenation
        self.assertGreater(output.shape[-1], self.nb_filters)


if __name__ == "__main__":
    unittest.main()
