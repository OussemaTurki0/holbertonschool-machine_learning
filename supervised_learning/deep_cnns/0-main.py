#!/usr/bin/env python3
"""
Test suite for the inception_block module
"""

import unittest
import tensorflow as tf
inception_block = __import__('0-inception_block').inception_block


class TestInceptionBlock(unittest.TestCase):
    def setUp(self):
        """
        Setup an input tensor for testing
        """
        self.input_shape = (8, 8, 3)  # Example input shape (height, width, channels)
        self.A_prev = tf.keras.Input(shape=self.input_shape)

    def test_inception_block_output_shape(self):
        """
        Test that the output shape of the inception block is correct
        """
        filters = (32, 32, 64, 16, 32, 16)
        output = inception_block(self.A_prev, filters)
        model = tf.keras.Model(inputs=self.A_prev, outputs=output)

        expected_channels = sum(filters)
        self.assertEqual(output.shape[-1], expected_channels)
        self.assertEqual(output.shape[1:3], self.input_shape[:2])

    def test_inception_block_output_type(self):
        """
        Test that the output type of the inception block is a Tensor
        """
        filters = (32, 32, 64, 16, 32, 16)
        output = inception_block(self.A_prev, filters)
        self.assertIsInstance(output, tf.Tensor)

    def test_invalid_filters_length(self):
        """
        Test that an error is raised if filters tuple is not of length 6
        """
        with self.assertRaises(ValueError):
            inception_block(self.A_prev, (32, 32, 64))  # Invalid length

    def test_activation_relu(self):
        """
        Test that ReLU activation is used in all layers
        """
        filters = (8, 8, 16, 4, 8, 4)
        output = inception_block(self.A_prev, filters)
        model = tf.keras.Model(inputs=self.A_prev, outputs=output)

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.assertEqual(layer.activation.__name__, 'relu')


if __name__ == "__main__":
    unittest.main()
