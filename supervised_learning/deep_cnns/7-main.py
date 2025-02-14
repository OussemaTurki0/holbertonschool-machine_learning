#!/usr/bin/env python3
"""
Test suite for the densenet121 implementation
"""

import unittest
from tensorflow import keras as K
from densenet121 import densenet121


class TestDenseNet121(unittest.TestCase):
    def setUp(self):
        """
        Setup the DenseNet121 model for testing
        """
        self.model = densenet121(growth_rate=32, compression=0.5)

    def test_model_output_shape(self):
        """
        Test that the model output shape is correct
        """
        self.assertEqual(self.model.output_shape, (None, 1000))

    def test_model_input_shape(self):
        """
        Test that the model input shape is correct
        """
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))

    def test_model_type(self):
        """
        Test that the returned object is a Keras Model
        """
        self.assertIsInstance(self.model, K.Model)

    def test_layers_count(self):
        """
        Test that the model has the expected number of layers
        """
        # DenseNet-121 typically has 428 layers (including input/output)
        self.assertGreaterEqual(len(self.model.layers), 428 - 50)  # Allow some margin

    def test_model_compile(self):
        """
        Test that the model compiles successfully
        """
        try:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        except Exception as e:
            self.fail(f"Model failed to compile: {e}")

    def test_prediction(self):
        """
        Test that the model can perform a forward pass
        """
        import numpy as np
        sample_input = np.random.random((1, 224, 224, 3)).astype('float32')
        try:
            prediction = self.model.predict(sample_input)
            self.assertEqual(prediction.shape, (1, 1000))
        except Exception as e:
            self.fail(f"Model failed to predict: {e}")


if __name__ == "__main__":
    unittest.main()
