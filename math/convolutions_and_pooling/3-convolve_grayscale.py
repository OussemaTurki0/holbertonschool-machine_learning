#!/usr/bin/env python3
"""
Strided Convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.
    """
    # Setup matrixes and padding dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Pading indexes: (before, after), shortcut is (padding,)
    padded_imgs = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                         mode='constant',)

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded images, scaling indexes by stride
            region = padded_imgs[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            # Convolve each image (m) for this region (i, j)
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
