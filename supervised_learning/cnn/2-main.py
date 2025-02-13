#!/usr/bin/env python3
"""
Test file for conv_backward
"""

import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

if __name__ == "__main__":
    np.random.seed(0)
    A_prev = np.random.randn(2, 5, 5, 3)  # 2 examples, 5x5 size, 3 channels
    W = np.random.randn(3, 3, 3, 8)        # 3x3 kernels, 3 channels, 8 filters
    b = np.random.randn(1, 1, 1, 8)        # 8 biases
    dZ = np.random.randn(2, 5, 5, 8)       # Gradient from the next layer
    stride = (1, 1)

    dA_prev, dW, db = conv_backward(dZ, A_prev, W, b, padding="same", stride=stride)

    print("dA_prev shape:", dA_prev.shape)
    print("dA_prev:", dA_prev)
    print("\ndW shape:", dW.shape)
    print("dW:", dW)
    print("\ndb shape:", db.shape)
    print("db:", db)
