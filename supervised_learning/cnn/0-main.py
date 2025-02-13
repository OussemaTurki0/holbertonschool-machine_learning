#!/usr/bin/env python3
"""
Test file for conv_forward
"""

import numpy as np
conv_forward = __import__('0-conv_forward').conv_forward

def relu(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)

if __name__ == "__main__":
    np.random.seed(0)
    A_prev = np.random.randn(10, 4, 4, 3)  # 10 images, 4x4 size, 3 channels
    W = np.random.randn(2, 2, 3, 8)        # 2x2 kernels, 3 channels, 8 filters
    b = np.random.randn(1, 1, 1, 8)        # Bias for each of the 8 filters
    stride = (1, 1)
    padding = 'same'

    A = conv_forward(A_prev, W, b, relu, padding, stride)
    print("Output shape:", A.shape)
    print("Output:", A)
