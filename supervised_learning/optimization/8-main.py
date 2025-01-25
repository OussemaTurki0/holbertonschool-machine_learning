#!/usr/bin/env python3

import tensorflow as tf
create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op  # Replace 'your_script_name' with the actual file name

if __name__ == '__main__':
    # Define example parameters
    alpha = 0.01  # Learning rate
    beta2 = 0.9   # RMSProp weight (decay rate)
    epsilon = 1e-8  # Small number to avoid division by zero

    # Create the optimizer
    optimizer = create_RMSProp_op(alpha, beta2, epsilon)

    # Print details about the optimizer
    print("Optimizer details:")
    print(optimizer)

    # Create a simple example to test the optimizer
    # Define a variable and a simple quadratic loss function
    var = tf.Variable([5.0], dtype=tf.float32)
    loss_fn = lambda: (var - 2.0) ** 2  # Loss = (var - 2)^2

    # Perform one optimization step
    print("\nInitial variable value:", var.numpy())
    optimizer.minimize(loss_fn, var_list=[var])
    print("Variable value after one step:", var.numpy())

    # Expected behavior:
    # - The variable should move closer to 2.0 with each optimization step.
