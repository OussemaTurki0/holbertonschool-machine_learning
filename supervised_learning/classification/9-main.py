#!/usr/bin/env python3
import unittest
import numpy as np
NN = __import__('9-neural_network').NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def test_initialization(self):
        # Test if the NeuralNetwork initializes correctly
        nx = 3
        nodes = 2
        nn = NN(nx, nodes)

        # Check if W1 has correct shape
        self.assertEqual(nn.W1.shape, (nodes, nx))

        # Check if b1 has correct shape
        self.assertEqual(nn.b1.shape, (nodes, 1))

        # Check if W2 has correct shape
        self.assertEqual(nn.W2.shape, (1, nodes))

        # Check if b2 has correct shape
        self.assertEqual(nn.b2.shape, ())

        # Check if A1 and A2 are initialized to 0
        self.assertEqual(nn.A1, 0)
        self.assertEqual(nn.A2, 0)

    def test_invalid_nx(self):
        # Test if invalid nx raises an error
        with self.assertRaises(ValueError):
            nn = NN(0, 2)

    def test_invalid_nodes(self):
        # Test if invalid nodes raises an error
        with self.assertRaises(ValueError):
            nn = NN(3, 0)

    def test_invalid_type_nx(self):
        # Test if non-integer nx raises an error
        with self.assertRaises(TypeError):
            nn = NN("3", 2)

    def test_invalid_type_nodes(self):
        # Test if non-integer nodes raises an error
        with self.assertRaises(TypeError):
            nn = NN(3, "2")

if __name__ == '__main__':
    unittest.main()
