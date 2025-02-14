#!/usr/bin/env python3
"""
Test suite for the projection_block module
"""

import unittest
import tensorflow as tf
projection_block = __import__('3-projection_block').projection_block


class TestProjectionBlock(unittest.TestCase):
    def setUp(self):
        """
        Setup a sample input tensor for testing
        """
        self.input_tensor = tf.random.uniform((1, 56, 56, 64))

    def test_output_shape_stride_2(self):
        """
        Test that the output shape changes when stride s=2
        """
        output = projection_block(self.input_tensor, (64, 64, 256), s=2)
        expected_shape = (1, 28, 28, 256)  # Input size halved due to stride
        self.assertEqual(output.shape, expected_shape)

    def test_output_shape_stride_1(self):
        """
        Test that the output shape remains the same when stride s=1
        """
        output = projection_block(self.input_tensor, (64, 64, 256), s=1)
        expected_shape = (1, 56, 56, 256)
        self.assertEqual(output.shape, expected_shape)

    def test_output_type(self):
        """
        Test that the output is a TensorFlow tensor
        """
        output = projection_block(self.input_tensor, (64, 64, 256))
        self.assertIsInstance(output, tf.Tensor)

    def test_activation_presence(self):
        """
        Test that the output is activated with ReLU
        """
        output = projection_block(self.input_tensor, (64, 64, 256))
        self.assertTrue(tf.reduce_all(output >= 0))  # ReLU ensures no negative values

    def test_different_filter_sizes(self):
        """
        Test that the block works with different filter sizes and strides
        """
        output = projection_block(self.input_tensor, (32, 64, 128), s=2)
        self.assertEqual(output.shape, (1, 28, 28, 128))


if __name__ == "__main__":
    unittest.main()
