#!/usr/bin/env python3
"""
Test file for convolve function with multiple kernels
"""

import numpy as np
convolve = __import__('5-convolve').convolve
def test_convolve():
    np.random.seed(0)
    images = np.random.randn(4, 6, 6, 3)  # 4 images, 6x6 pixels, 3 channels
    kernels = np.random.randn(3, 3, 3, 2)  # 3x3 kernels, 3 channels, 2 kernels

    # Test with 'same' padding and default stride
    output_same = convolve(images, kernels, padding='same')
    print("Output with 'same' padding:")
    print(output_same)
    print("Output shape:", output_same.shape)

    # Test with 'valid' padding
    output_valid = convolve(images, kernels, padding='valid')
    print("\nOutput with 'valid' padding:")
    print(output_valid)
    print("Output shape:", output_valid.shape)

    # Test with custom padding
    output_custom = convolve(images, kernels, padding=(1, 1))
    print("\nOutput with custom padding (1, 1):")
    print(output_custom)
    print("Output shape:", output_custom.shape)

    # Test with stride (2, 2)
    output_stride = convolve(images, kernels, stride=(2, 2))
    print("\nOutput with stride (2, 2):")
    print(output_stride)
    print("Output shape:", output_stride.shape)

if __name__ == "__main__":
    test_convolve()
