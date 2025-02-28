#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('8-neural_style').NST

def set_random_seeds(seed_value):
    """
    Set up the random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_image(image_path):
    """
    Helper function to load an image from the given path.
    """
    return mpimg.imread(image_path)

if __name__ == '__main__':
    # File paths for the style and content images
    style_image_path = "starry_night.jpg"
    content_image_path = "golden_gate.jpg"

    # Load images
    style_img = load_image(style_image_path)
    content_img = load_image(content_image_path)

    # Set the random seed for reproducibility
    SEED = 100
    set_random_seeds(SEED)

    # Initialize the NST model
    neural_style_transfer = NST(style_img, content_img)

    # Create a TensorFlow variable from the content image
    generated_img = tf.Variable(neural_style_transfer.content_image)

    # Compute gradients and costs
    grads, total_cost, content_cost, style_cost = neural_style_transfer.compute_grads(generated_img)

    # Output the computed costs and gradients
    print(f"Total Cost: {total_cost}")
    print(f"Content Cost: {content_cost}")
    print(f"Style Cost: {style_cost}")
    print(f"Gradients: {grads}")
