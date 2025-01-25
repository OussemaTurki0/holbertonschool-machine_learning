#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants

if __name__ == '__main__':
    np.random.seed(42)  # Set seed for reproducibility
    X = np.random.randn(5, 3) * 2 + 3  # Generate random 5x3 matrix with mean=3, std=2
    mean, std_dev = normalization_constants(X)
    print("Mean:", mean)
    print("Standard Deviation:", std_dev)
