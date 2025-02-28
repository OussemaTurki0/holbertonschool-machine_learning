#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('7-neural_style').NST

def set_random_seeds(seed_value):
    """
    Configure random seeds for reproducibility across various libraries.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_image_from_path(image_path):
    """
    Load image from the specified path.
    """
    return mpimg.imread(image_path)

if __name__ == '__main__':
    # Define file paths for style and content images
    style_image_path = "starry_night.jpg"
    content_image_path = "golden_gate.jpg"

    # Load the style and content images
    style_img = load_image_from_path(style_image_path)
    content_img = load_image_from_path(content_image_path)

    # Set the seed for random number generation
    SEED = 42
    set_random_seeds(SEED)

    # Create an instance of the Neural Style Transfer model
    transfer_model = NST(style_img, content_img)

    # Generate a random image with the same shape as the content image
    generated_img = np.random.uniform(low=0.0, high=1.0, size=transfer_model.content_image.shape)
    generated_img = tf.cast(generated_img, tf.float32)

    # Compute the total cost, content cost, and style cost
    total_cost, content_cost, style_cost = transfer_model.total_cost(generated_img)

    # Print the calculated costs
    print(f"Total Cost: {total_cost}")
    print(f"Content Cost: {content_cost}")
    print(f"Style Cost: {style_cost}")
