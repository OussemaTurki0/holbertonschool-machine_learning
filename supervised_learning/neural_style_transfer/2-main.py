#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('2-neural_style').NST

def set_random_seed(seed=42):
    """Set seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == '__main__':
    # Load images
    style_img = mpimg.imread("starry_night.jpg")
    content_img = mpimg.imread("golden_gate.jpg")

    # Set random seed
    set_random_seed()

    # Initialize NST
    nst = NST(style_img, content_img)

    # Create a random input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3), dtype=tf.float32)

    # Compute Gram matrix
    gram_result = nst.gram_matrix(input_tensor)

    # Print results
    tf.print("Input Tensor:", input_tensor, summarize=5)
    tf.print("Gram Matrix:", gram_result, summarize=5)
