#!/usr/bin/env python3
"""
Test suite for the transition_layer module
"""

import unittest
import tensorflow as tf
transition_layer = __import__('6-transition_layer').transition_layer


class TestTransitionLayer(unittest.TestCase):
    def setUp(self):
        """
        Setup initial parameters for testing
        """
        self.input_shape = (1, 32, 32, 64)  # Batch size 1, 32x32 image with 64 filters
        self.X = tf.random.uniform(self.input_shape)
        self.nb_filters = 64
        self.compression = 0.5  # Compression factor

    def test_output_shape(self):
        """
        Test that the transition layer outputs the correct shape
        """
        output, filters = transition_layer(self.X, self.nb_filters, self.compression)
        expected_filters = int(self.nb_filters * self.compression)
        self.assertEqual(output.shape, (1, 16, 16, expected_filters))
        self.assertEqual(filters, expected_filters)

    def test_filter_reduction(self):
        """
        Test that the number of filters is reduced by the compression factor
        """
        _, filters = transition_layer(self.X, self.nb_filters, 0.25)
        self.assertEqual(filters, int(self.nb_filters * 0.25))

    def test_output_type(self):
        """
        Test that the output is a TensorFlow tensor
        """
        output, _ = transition_layer(self.X, self.nb_filters, self.compression)
        self.assertIsInstance(output, tf.Tensor)

    def test_pooling_effect(self):
        """
        Test that average pooling reduces the spatial dimensions by half
        """
        output, _ = transition_layer(self.X, self.nb_filters, self.compression)
        self.assertEqual(output.shape[1:3], (16, 16))


if __name__ == "__main__":
    unittest.main()
