#!/usr/bin/env python3
"""
Test for the batch_norm function.
"""

import numpy as np
batch_norm = __import__('13-batch_norm').batch_norm  # Replace 'your_script_name' with the actual file name

if __name__ == '__main__':
    # Define test parameters
    np.random.seed(0)
    Z = np.random.randn(5, 3)  # Random input matrix (5 examples, 3 features)
    gamma = np.ones((1, 3))    # Scale parameter (same shape as number of features)
    beta = np.zeros((1, 3))    # Shift parameter (same shape as number of features)
    epsilon = 1e-8             # Small value to avoid division by zero

    print("Testing batch normalization with the following parameters:")
    print(f"Input Z: \n{Z}")
    print(f"Gamma: {gamma}")
    print(f"Beta: {beta}")
    print(f"Epsilon: {epsilon}\n")

    # Apply batch normalization
    Z_tilde = batch_norm(Z, gamma, beta, epsilon)

    # Output the normalized result
    print(f"Normalized output Z_tilde: \n{Z_tilde}")
