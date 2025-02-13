#!/usr/bin/env python3
"""
Test file for pool_forward
"""

import numpy as np
pool_forward = __import__('1-pool_forward').pool_forward

if __name__ == "__main__":
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)  # 2 examples, 5x5 size, 3 channels
    kernel_shape = (2, 2)                  # Pooling kernel size
    stride = (2, 2)                        # Stride for height and width

    # Test Max Pooling
    A_max = pool_forward(A_prev, kernel_shape, stride, mode='max')
    print("Max Pooling Output shape:", A_max.shape)
    print("Max Pooling Output:", A_max)

    # Test Average Pooling
    A_avg = pool_forward(A_prev, kernel_shape, stride, mode='avg')
    print("\nAverage Pooling Output shape:", A_avg.shape)
    print("Average Pooling Output:", A_avg)
