#!/usr/bin/env python3
"""
Test suite for the resnet50 module
"""

import unittest
import tensorflow as tf
resnet50 = __import__('4-resnet50').resnet50


class TestResNet50(unittest.TestCase):
    def setUp(self):
        """
        Setup ResNet-50 model for testing
        """
        self.model = resnet50()

    def test_output_shape(self):
        """
        Test that the model outputs the correct shape
        """
        dummy_input = tf.random.uniform((1, 224, 224, 3))
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 1000))

    def test_total_layers(self):
        """
        Test the total number of layers in the ResNet-50 model
        """
        # ResNet-50 typically has 175 layers in Keras
        self.assertEqual(len(self.model.layers), 175)

    def test_layer_names(self):
        """
        Test that key layers are present in the model
        """
        layer_names = [layer.name for layer in self.model.layers]
        self.assertIn('conv2d', layer_names[0])
        self.assertIn('batch_normalization', ' '.join(layer_names))
        self.assertIn('activation', ' '.join(layer_names))
        self.assertIn('dense', layer_names[-1])

    def test_output_type(self):
        """
        Test that the model output is a TensorFlow tensor
        """
        dummy_input = tf.random.uniform((1, 224, 224, 3))
        output = self.model(dummy_input)
        self.assertIsInstance(output, tf.Tensor)

    def test_softmax_activation(self):
        """
        Test that the output layer uses softmax activation
        """
        output_layer = self.model.layers[-1]
        self.assertEqual(output_layer.activation.__name__, 'softmax')


if __name__ == "__main__":
    unittest.main()
