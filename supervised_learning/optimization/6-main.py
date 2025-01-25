#!/usr/bin/env python3

import tensorflow as tf
create_momentum_op = __import__('6-momentum').create_momentum_op

if __name__ == '__main__':
    # Define example parameters
    alpha = 0.01  # Learning rate
    beta1 = 0.9   # Momentum parameter

    # Create the optimizer
    optimizer = create_momentum_op(alpha, beta1)

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
