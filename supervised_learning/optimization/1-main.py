#!/usr/bin/env python3

import numpy as np
normalize = __import__('1-normalize').normalize

if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(5, 3)  # Generate a 5x3 random matrix
    m = np.mean(X, axis=0)  # Calculate the mean
    s = np.std(X, axis=0)  # Calculate the standard deviation
    Z = normalize(X, m, s)  # Normalize the matrix

    print("Original matrix (X):")
    print(X)
    print("\nMean (m):")
    print(m)
    print("\nStandard deviation (s):")
    print(s)
    print("\nNormalized matrix (Z):")
    print(Z)

    # Check if the normalization was successful (mean ~ 0, std ~ 1)
    print("\nNormalized matrix mean (should be ~0):")
    print(np.mean(Z, axis=0))
    print("\nNormalized matrix std (should be ~1):")
    print(np.std(Z, axis=0))
