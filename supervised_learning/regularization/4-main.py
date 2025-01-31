import unittest
import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop

class TestDropoutForwardProp(unittest.TestCase):
    
    def test_dropout_forward_prop_shapes(self):
        # Test to check the shape of the cache returned by the function
        
        # Example input data (2 features, 3 examples)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Example weights for 3 layers
        weights = {
            'W1': np.array([[1, 1], [1, 1]]),
            'b1': np.array([[0], [0]]),
            'W2': np.array([[1, 1]]),
            'b2': np.array([[0]]),
            'W3': np.array([[1]]),
            'b3': np.array([[0]]),
        }
        
        # Number of layers
        L = 3
        keep_prob = 0.8
        
        # Call the dropout_forward_prop function
        cache = dropout_forward_prop(X, weights, L, keep_prob)
        
        # Test if the cache contains all the expected keys
        self.assertTrue('A0' in cache)
        self.assertTrue('A1' in cache)
        self.assertTrue('A2' in cache)
        self.assertTrue('A3' in cache)
        
        # Test the shapes of the returned A matrices
        self.assertEqual(cache['A0'].shape, X.shape)
        self.assertEqual(cache['A1'].shape, (2, 3))  # After W1 and b1
        self.assertEqual(cache['A2'].shape, (1, 3))  # After W2 and b2
        self.assertEqual(cache['A3'].shape, (1, 3))  # After W3 and b3
        
    def test_dropout_mask(self):
        # Test if dropout is applied during non-output layers
        
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        weights = {
            'W1': np.array([[1, 1], [1, 1]]),
            'b1': np.array([[0], [0]]),
            'W2': np.array([[1, 1]]),
            'b2': np.array([[0]]),
            'W3': np.array([[1]]),
            'b3': np.array([[0]]),
        }
        
        L = 3
        keep_prob = 0.8
        
        # Call the dropout_forward_prop function
        cache = dropout_forward_prop(X, weights, L, keep_prob)
        
        # Test if dropout masks are created in layers before output (A1, A2)
        self.assertTrue('D1' in cache)
        self.assertTrue('D2' in cache)
        
        # Ensure dropout masks are boolean (True/False values)
        self.assertTrue(np.all(np.logical_or(cache['D1'] == 0, cache['D1'] == 1)))
        self.assertTrue(np.all(np.logical_or(cache['D2'] == 0, cache['D2'] == 1)))
        
    def test_output_layer_activation(self):
        # Test that the output layer uses softmax
        
        X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 features, 2 examples
        
        weights = {
            'W1': np.array([[1, 1, 1], [1, 1, 1]]),
            'b1': np.array([[0], [0]]),
            'W2': np.array([[1, 1]]),
            'b2': np.array([[0]]),
            'W3': np.array([[1]]),
            'b3': np.array([[0]]),
        }
        
        L = 3
        keep_prob = 1  # No dropout for simplicity
        
        # Call the dropout_forward_prop function
        cache = dropout_forward_prop(X, weights, L, keep_prob)
        
        # Check if the output layer uses softmax
        A3 = cache['A3']
        
        # Test that A3 sums to 1 for each example (softmax property)
        for i in range(A3.shape[1]):
            self.assertAlmostEqual(np.sum(A3[:, i]), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
