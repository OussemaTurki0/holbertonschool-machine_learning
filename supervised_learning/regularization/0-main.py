import unittest
import numpy as np
l2_reg_cost = __import__('0-l2_reg_cost').l2_reg_cost

class TestL2RegCost(unittest.TestCase):
    
    def setUp(self):
        """
        Set up test data for testing
        """
        self.weights = {
            'W1': np.array([[1, 2], [3, 4]]),
            'W2': np.array([[5, 6], [7, 8]]),
        }
        self.cost = 5.0
        self.lambtha = 0.1
        self.L = 2  # Number of layers
        self.m = 3  # Number of samples
    
    def test_l2_reg_cost(self):
        """
        Test the l2_reg_cost function
        """
        result = l2_reg_cost(self.cost, self.lambtha, self.weights, self.L, self.m)
        
        # Calculate the expected l2 regularization term
        l2_reg_term = 0
        for i in range(1, self.L + 1):
            l2_reg_term += np.sum(np.square(self.weights['W' + str(i)]))
        l2_reg_term *= self.lambtha / (2 * self.m)
        
        expected_cost = self.cost + l2_reg_term
        
        self.assertAlmostEqual(result, expected_cost, places=6)
    
    def test_no_weights(self):
        """
        Test when weights are empty
        """
        empty_weights = {}
        result = l2_reg_cost(self.cost, self.lambtha, empty_weights, self.L, self.m)
        self.assertEqual(result, self.cost)

    def test_single_weight(self):
        """
        Test when there is only one weight matrix
        """
        single_weight = {'W1': np.array([[1, 1], [1, 1]])}
        result = l2_reg_cost(self.cost, self.lambtha, single_weight, 1, self.m)
        expected_l2_reg_term = np.sum(np.square(single_weight['W1'])) * self.lambtha / (2 * self.m)
        self.assertAlmostEqual(result, self.cost + expected_l2_reg_term, places=6)

if __name__ == '__main__':
    unittest.main()
