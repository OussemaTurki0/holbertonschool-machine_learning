#!/usr/bin/env python3
"""
Test suite for the identity_block module
"""

import unittest
import tensorflow as tf
from identity_block import identity_block


class TestIdentityBlock(unittest.TestCase):
    def setUp(self):
        """
        Setup a sample input tensor for testing
        """
        self.input_tensor = tf.random.uniform((1, 56, 56, 64))

    def test_output_shape(self):
        """
        Test that the output shape is the same as the input shape
        """
        output = identity_block(self.input_tensor, (64, 64, 64))
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_output_type(self):
        """
        Test that the output is a TensorFlow tensor
        """
        output = identity_block(self.input_tensor, (64, 64, 64))
        self.assertIsInstance(output, tf.Tensor)

    def test_no_dimension_change(self):
        """
        Test that dimensions remain unchanged in an identity block
        """
        output = identity_block(self.input_tensor, (64, 64, 64))
        self.assertEqual(output.shape[1:], self.input_tensor.shape[1:])

    def test_activation_presence(self):
        """
        Test that the output is activated with ReLU
        """
        output = identity_block(self.input_tensor, (64, 64, 64))
        self.assertTrue(tf.reduce_all(output >= 0))  # ReLU ensures no negative values

    def test_different_filter_sizes(self):
        """
        Test that the block works with different filter sizes
        """
        output = identity_block(self.input_tensor, (32, 64, 128))
        self.assertEqual(output.shape, self.input_tensor.shape)


if __name__ == "__main__":
    unittest.main()
