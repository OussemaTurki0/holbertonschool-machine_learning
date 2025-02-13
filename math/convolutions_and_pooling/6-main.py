#!/usr/bin/env python3
"""
Test file for pool function
"""

import numpy as np
pool = __import__('6-pool').pool

def test_pool():
    np.random.seed(0)
    images = np.random.randn(3, 8, 8, 3)  # 3 images, 8x8 pixels, 3 channels

    # Test max pooling with kernel (2, 2) and stride (2, 2)
    output_max = pool(images, (2, 2), (2, 2), mode='max')
    print("Max Pooling Output (2x2 kernel, stride 2x2):")
    print(output_max)
    print("Output shape:", output_max.shape)

    # Test average pooling with kernel (2, 2) and stride (2, 2)
    output_avg = pool(images, (2, 2), (2, 2), mode='avg')
    print("\nAverage Pooling Output (2x2 kernel, stride 2x2):")
    print(output_avg)
    print("Output shape:", output_avg.shape)

    # Test with different kernel (3, 3) and stride (1, 1) for max pooling
    output_max_custom = pool(images, (3, 3), (1, 1), mode='max')
    print("\nMax Pooling Output (3x3 kernel, stride 1x1):")
    print(output_max_custom)
    print("Output shape:", output_max_custom.shape)

    # Test with different kernel (3, 3) and stride (1, 1) for average pooling
    output_avg_custom = pool(images, (3, 3), (1, 1), mode='avg')
    print("\nAverage Pooling Output (3x3 kernel, stride 1x1):")
    print(output_avg_custom)
    print("Output shape:", output_avg_custom.shape)

if __name__ == "__main__":
    test_pool()
