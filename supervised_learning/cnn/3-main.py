#!/usr/bin/env python3
"""
Test file for pool_backward
"""

import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

if __name__ == "__main__":
    np.random.seed(0)
    A_prev = np.random.randn(2, 5, 5, 3)  # 2 examples, 5x5 size, 3 channels
    dA = np.random.randn(2, 3, 3, 3)      # Gradient from the next layer
    kernel_shape = (2, 2)                 # 2x2 pool size
    stride = (1, 1)                       # Stride of 1 for both dimensions

    # Test max pooling
    dA_prev_max = pool_backward(dA, A_prev, kernel_shape, stride, mode='max')
    print("Max Pooling Backpropagation")
    print("dA_prev shape:", dA_prev_max.shape)
    print("dA_prev:", dA_prev_max)

    # Test average pooling
    dA_prev_avg = pool_backward(dA, A_prev, kernel_shape, stride, mode='avg')
    print("\nAverage Pooling Backpropagation")
    print("dA_prev shape:", dA_prev_avg.shape)
    print("dA_prev:", dA_prev_avg)
