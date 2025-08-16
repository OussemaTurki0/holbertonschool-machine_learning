#!/usr/bin/env python3
"""
This module provides a function to flip an image horizontally.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image (tf.Tensor): 3D tensor representing the image

    Returns:
        tf.Tensor: horizontally flipped image
    """
    return tf.image.flip_left_right(image)
