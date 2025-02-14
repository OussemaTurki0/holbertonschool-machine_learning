#!/usr/bin/env python3
"""
Test suite for the inception_network module
"""

import unittest
import tensorflow as tf
inception_network = __import__('1-inception_network').inception_network


class TestInceptionNetwork(unittest.TestCase):
    def setUp(self):
        """
        Setup the model for testing
        """
        self.model = inception_network()

    def test_model_output_shape(self):
        """
        Test that the output shape of the model is correct
        """
        self.assertEqual(self.model.output_shape, (None, 1000))

    def test_model_input_shape(self):
        """
        Test that the input shape of the model is correct
        """
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))

    def test_model_type(self):
        """
        Test that the model is an instance of Keras Model
        """
        self.assertIsInstance(self.model, tf.keras.Model)

    def test_model_layer_count(self):
        """
        Test that the model has the expected number of layers
        """
        # Expected layers count (may vary slightly based on your implementation)
        self.assertGreaterEqual(len(self.model.layers), 20)

    def test_model_compile(self):
        """
        Test that the model compiles successfully
        """
        try:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        except Exception as e:
            self.fail(f"Model failed to compile: {e}")


if __name__ == "__main__":
    unittest.main()
