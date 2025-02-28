#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('4-neural_style').NST

def set_seeds(seed_value):
    """
    Sets the random seed for reproducibility across different libraries
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_image(image_path):
    """
    Loads and returns an image from the provided file path
    """
    return mpimg.imread(image_path)

if __name__ == '__main__':
    # Load images
    style_path = "starry_night.jpg"
    content_path = "golden_gate.jpg"

    style_img = load_image(style_path)
    content_img = load_image(content_path)

    # Create NST object
    neural_style = NST(style_img, content_img)

    # Set reproducibility seeds
    SEED = 42
    set_seeds(SEED)

    # Preprocess content image for VGG19 input
    vgg19_model = tf.keras.applications.vgg19
    preprocessed_content = vgg19_model.preprocess_input(neural_style.content_image * 255)

    # Get the feature map and compute style cost for the first layer
    feature_map = neural_style.model(preprocessed_content)
    style_cost = neural_style.layer_style_cost(feature_map[0], neural_style.gram_style_features[0])

    print("Computed Style Cost for First Layer: ", style_cost)
