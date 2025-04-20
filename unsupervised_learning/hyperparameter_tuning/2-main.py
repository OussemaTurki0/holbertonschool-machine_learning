#!/usr/bin/env python3

# Importing GaussianProcess from 2-gp
GP = __import__('2-gp').GaussianProcess
import numpy as np

def black_box_function(x):
    """Mystery function to simulate unknown behavior."""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)

if __name__ == '__main__':
    # Set random seed
    np.random.seed(42)

    # Initial samples
    initial_X = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    initial_Y = black_box_function(initial_X)

    # Create Gaussian Process model
    gp = GP(initial_X, initial_Y, l=0.6, sigma_f=2.0)

    # Generate a new data point and evaluate it
    new_X = np.random.uniform(-np.pi, 2 * np.pi, (1,))
    new_Y = black_box_function(new_X)

    print(f"New input X: {new_X}")
    print(f"New observed Y: {new_Y}")

    # Update GP model with new observation
    gp.update(new_X, new_Y)

    print("Updated X shape and data:", gp.X.shape, "\n", gp.X)
    print("Updated Y shape and data:", gp.Y.shape, "\n", gp.Y)
    print("Updated Kernel matrix shape and values:", gp.K.shape, "\n", gp.K)
