#!/usr/bin/env python3
"""
Test for the create_batch_norm_layer function.
"""

import tensorflow as tf
import numpy as np

create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer  # Replace with your script's name

def test_create_batch_norm_layer():
    """
    Test the create_batch_norm_layer function.
    """
    # Set random seed for reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # Define previous layer output (for testing, we create random data)
    prev_output = tf.random.normal([5, 10])  # 5 samples, 10 features

    # Define parameters for the batch norm layer
    n = 5  # Number of nodes in the current layer
    activation = tf.nn.relu  # Use ReLU as the activation function

    # Create batch normalization layer
    output = create_batch_norm_layer(prev_output, n, activation)

    # Print results
    print("Testing create_batch_norm_layer function:\n")
    print("Previous output (prev_output):")
    print(prev_output.numpy())
    print("\nBatch normalized output:")
    print(output.numpy())

if __name__ == '__main__':
    test_create_batch_norm_layer()
