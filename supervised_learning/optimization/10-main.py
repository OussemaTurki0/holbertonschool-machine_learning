#!/usr/bin/env python3
"""
Test file for creating Adam optimizer using TensorFlow.
"""

import tensorflow as tf
create_Adam_op = __import__('10-Adam').create_Adam_op

if __name__ == '__main__':
    # Define example parameters
    alpha = 0.001  # Learning rate
    beta1 = 0.9    # Adam's first moment decay rate
    beta2 = 0.999  # Adam's second moment decay rate
    epsilon = 1e-7  # Small number to avoid division by zero

    print("Setting up Adam optimizer with the following parameters:")
    print(f"alpha: {alpha}")
    print(f"beta1: {beta1}")
    print(f"beta2: {beta2}")
    print(f"epsilon: {epsilon}\n")

    # Create the Adam optimizer using the function
    adam_optimizer = create_Adam_op(alpha, beta1, beta2, epsilon)

    # Check the optimizer type
    print(f"Created optimizer: {adam_optimizer}")
    print(f"Optimizer type: {type(adam_optimizer)}")
    
    # Verify the parameters (attributes) of the created Adam optimizer
    print("\nChecking optimizer parameters:")
    print(f"Learning rate: {adam_optimizer.learning_rate.numpy()}")
    print(f"Beta1: {adam_optimizer.beta_1.numpy()}")
    print(f"Beta2: {adam_optimizer.beta_2.numpy()}")
    print(f"Epsilon: {adam_optimizer.epsilon.numpy()}")
