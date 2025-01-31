import unittest
import numpy as np
l2_reg_gradient_descent = __import__('1-l2_reg_gradient_descent').l2_reg_gradient_descent

class TestL2RegGradientDescent(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for testing
        """
        self.Y = np.array([[1, 0], [0, 1]])  # 2 samples, 2 labels
        self.weights = {
            'W1': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'b1': np.array([[0.1], [0.2]]),
            'W2': np.array([[0.5, 0.6], [0.7, 0.8]]),
            'b2': np.array([[0.5], [0.6]]),
        }
        self.cache = {
            'A0': np.array([[0.1, 0.2], [0.3, 0.4]]),  # Example input features
            'A1': np.array([[0.2, 0.4], [0.5, 0.6]]),  # Example activations from hidden layer
            'A2': np.array([[0.7, 0.8], [0.9, 1.0]]),  # Example activations from output layer
        }
        self.alpha = 0.01
        self.lambtha = 0.1
        self.L = 2  # Number of layers

    def test_l2_reg_gradient_descent(self):
        """
        Test the l2_reg_gradient_descent function
        """
        # Copy the weights before updating
        initial_weights = {key: value.copy() for key, value in self.weights.items()}
        
        # Run the gradient descent update
        l2_reg_gradient_descent(self.Y, self.weights, self.cache, self.alpha, self.lambtha, self.L)

        # Check if the weights and biases have been updated
        for key in self.weights:
            if key.startswith('W'):
                self.assertFalse(np.array_equal(self.weights[key], initial_weights[key]))
            elif key.startswith('b'):
                self.assertFalse(np.array_equal(self.weights[key], initial_weights[key]))
        
    def test_no_update_with_zero_alpha(self):
        """
        Test if no update happens when alpha is zero.
        """
        zero_alpha = 0.0
        initial_weights = {key: value.copy() for key, value in self.weights.items()}
        initial_biases = {key: value.copy() for key, value in self.weights.items() if key.startswith('b')}
        
        # Run gradient descent with zero alpha
        l2_reg_gradient_descent(self.Y, self.weights, self.cache, zero_alpha, self.lambtha, self.L)

        # Check that weights and biases are not updated
        for key in self.weights:
            if key.startswith('W') or key.startswith('b'):
                self.assertTrue(np.array_equal(self.weights[key], initial_weights[key]))
    
    def test_single_sample(self):
        """
        Test gradient descent with a single sample.
        """
        Y_single = np.array([[1], [0]])  # Single sample, 2 labels
        cache_single = {
            'A0': np.array([[0.1]]),  # Single input feature
            'A1': np.array([[0.2]]),  # Single hidden layer activation
            'A2': np.array([[0.7]]),  # Single output layer activation
        }
        
        initial_weights_single = {key: value.copy() for key, value in self.weights.items()}
        
        # Run gradient descent with a single sample
        l2_reg_gradient_descent(Y_single, self.weights, cache_single, self.alpha, self.lambtha, self.L)

        # Ensure weights and biases are updated
        for key in self.weights:
            if key.startswith('W'):
                self.assertFalse(np.array_equal(self.weights[key], initial_weights_single[key]))
            elif key.startswith('b'):
                self.assertFalse(np.array_equal(self.weights[key], initial_weights_single[key]))

if __name__ == '__main__':
    unittest.main()
