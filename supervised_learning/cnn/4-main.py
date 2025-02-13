#!/usr/bin/env python3
"""
Test file for pool_backward
"""

import numpy as np
from pool_backward import pool_backward

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # Example input (A_prev is the previous layer output)
    A_prev = np.random.randn(2, 5, 5, 3)  # 2 examples, 5x5 size, 3 channels

    # Gradient from the next layer (dA)
    dA = np.random.randn(2, 3, 3, 3)  # 2 examples, 3x3 output size, 3 channels

    # Kernel shape and stride for pooling
    kernel_shape = (2, 2)  # 2x2 pooling window
    stride = (1, 1)        # Stride of 1 for both height and width

    # Test max pooling backpropagation
    print("Testing Max Pooling Backpropagation...")
    dA_prev_max = pool_backward(dA, A_prev, kernel_shape, stride, mode='max')
    print("Max Pooling Backpropagation dA_prev shape:", dA_prev_max.shape)
    print(dA_prev_max)

    # Test average pooling backpropagation
    print("\nTesting Average Pooling Backpropagation...")
    dA_prev_avg = pool_backward(dA, A_prev, kernel_shape, stride, mode='avg')
    print("Average Pooling Backpropagation dA_prev shape:", dA_prev_avg.shape)
    print(dA_prev_avg)
