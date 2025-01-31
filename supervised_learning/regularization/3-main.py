import unittest
import tensorflow as tf
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost
l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

class TestL2RegCreateLayer(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for testing
        """
        # Create a tensor to simulate previous layer output
        self.prev = tf.random.normal([5, 3])  # 5 samples, 3 features
        
        # Define the parameters for the new layer
        self.n = 4  # Number of units in the new layer
        self.activation = tf.nn.relu  # ReLU activation function
        self.lambtha = 0.1  # L2 regularization parameter

    def test_l2_reg_create_layer(self):
        """
        Test the l2_reg_create_layer function
        """
        # Create the new layer
        output = l2_reg_create_layer(self.prev, self.n, self.activation, self.lambtha)

        # Check the output tensor shape
        self.assertEqual(output.shape, (5, self.n))  # 5 samples, 4 units in the new layer
        
        # Check if the activation function is applied
        self.assertTrue(tf.reduce_all(output >= 0), "ReLU activation not applied correctly")

    def test_l2_regularization(self):
        """
        Test if L2 regularization is correctly applied
        """
        # Create the model with the custom layer
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(3,)),
            l2_reg_create_layer(self.prev, self.n, self.activation, self.lambtha)
        ])

        # Check if the regularizer is applied
        regularizer = model.layers[1].kernel_regularizer
        self.assertIsInstance(regularizer, tf.keras.regularizers.L2, "L2 regularizer not applied correctly")
        self.assertEqual(regularizer.l2, self.lambtha, "L2 regularization parameter mismatch")

if __name__ == '__main__':
    unittest.main()
