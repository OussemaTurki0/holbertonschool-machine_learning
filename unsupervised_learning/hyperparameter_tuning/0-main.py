#!/usr/bin/env python3

# Import the custom GaussianProcess class
GP = __import__('0-gp').GaussianProcess
import numpy as np

def black_box_function(x):
    """Simulates a black-box function for optimization."""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)

if __name__ == '__main__':
    """Initialize and test the Gaussian Process model."""
    np.random.seed(42)

    # Generate initial sample points
    initial_X = np.random.uniform(low=-np.pi, high=2 * np.pi, size=(2, 1))
    initial_Y = black_box_function(initial_X)

    # Create GP instance with custom length scale and signal variance
    gp = GP(initial_X, initial_Y, l=0.7, sigma_f=1.8)

    # Diagnostics
    print(gp.X is initial_X)
    print(gp.Y is initial_Y)
    print(f"Length scale: {gp.l}")
    print(f"Signal variance: {gp.sigma_f}")
    print(f"Covariance matrix shape: {gp.K.shape}")
    print("Covariance matrix:\n", gp.K)
    print("Covariance check passed:", np.allclose(gp.kernel(initial_X, initial_X), gp.K))
