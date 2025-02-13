#!/usr/bin/env python3
"""
Test file for convolve_channels function
"""

import numpy as np
convolve_channels = __import__('4-convolve_channels').convolve_channels
def test_convolve_channels():
    np.random.seed(0)
    images = np.random.randn(5, 8, 8, 3)  # 5 images, 8x8 pixels, 3 channels
    kernel = np.random.randn(3, 3, 3)  # 3x3 kernel with 3 channels

    # Test with 'same' padding and default stride
    output_same = convolve_channels(images, kernel, padding='same')
    print("Output with 'same' padding:")
    print(output_same)
    print("Output shape:", output_same.shape)

    # Test with 'valid' padding
    output_valid = convolve_channels(images, kernel, padding='valid')
    print("\nOutput with 'valid' padding:")
    print(output_valid)
    print("Output shape:", output_valid.shape)

    # Test with custom padding
    output_custom = convolve_channels(images, kernel, padding=(2, 2))
    print("\nOutput with custom padding (2, 2):")
    print(output_custom)
    print("Output shape:", output_custom.shape)

    # Test with stride (2, 2)
    output_stride = convolve_channels(images, kernel, stride=(2, 2))
    print("\nOutput with stride (2, 2):")
    print(output_stride)
    print("Output shape:", output_stride.shape)

if __name__ == "__main__":
    test_convolve_channels()
