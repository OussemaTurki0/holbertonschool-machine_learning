#!/usr/bin/env python3

# Import GaussianProcess class from '1-gp' file
GP = __import__('1-gp').GaussianProcess
import numpy as np

def simulate_function(x):
    """Simulated function mimicking a black-box behavior."""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)

if __name__ == '__main__':
    # Seed for reproducibility
    np.random.seed(21)

    # Initial training data
    train_X = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    train_Y = simulate_function(train_X)

    # Initialize Gaussian Process
    gp = GP(train_X, train_Y, l=0.65, sigma_f=2.1)

    # New input points for prediction
    test_X = np.random.uniform(-np.pi, 2 * np.pi, (10, 1))
    mean, std_dev = gp.predict(test_X)

    # Output results
    print("Mean predictions:", mean.shape, "\n", mean)
    print("Standard deviations:", std_dev.shape, "\n", std_dev)
