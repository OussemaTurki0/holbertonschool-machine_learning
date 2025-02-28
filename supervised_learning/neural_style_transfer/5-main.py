#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('5-neural_style').NST

def initialize_seeds(seed_value):
    """
    Initialize random seeds for reproducibility across libraries
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_and_process_image(image_path):
    """
    Loads and processes an image to prepare for neural style transfer
    """
    image = mpimg.imread(image_path)
    return image

if __name__ == '__main__':
    # Define image paths
    style_img_path = "starry_night.jpg"
    content_img_path = "golden_gate.jpg"

    # Load images
    style_image = load_and_process_image(style_img_path)
    content_image = load_and_process_image(content_img_path)

    # Initialize random seeds for reproducibility
    RANDOM_SEED = 123
    initialize_seeds(RANDOM_SEED)

    # Initialize the NST class with images
    style_transfer = NST(style_image, content_image)

    # Preprocess the content image
    vgg19_model = tf.keras.applications.vgg19
    content_image_preprocessed = vgg19_model.preprocess_input(style_transfer.content_image * 255)

    # Get the style feature outputs (excluding the last layer)
    feature_outputs = style_transfer.model(content_image_preprocessed)[:-1]

    # Calculate style cost
    computed_style_cost = style_transfer.style_cost(feature_outputs)

    print("Calculated Style Cost: ", computed_style_cost)
