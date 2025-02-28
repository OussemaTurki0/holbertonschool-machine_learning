#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('6-neural_style').NST

def set_reproducibility(seed_value):
    """
    Set up seeds for reproducibility across random libraries and TensorFlow.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_image(image_path):
    """
    Load image from file path.
    """
    return mpimg.imread(image_path)

if __name__ == '__main__':
    # Image file paths
    style_image_path = "starry_night.jpg"
    content_image_path = "golden_gate.jpg"

    # Load the style and content images
    style_img = load_image(style_image_path)
    content_img = load_image(content_image_path)

    # Set the random seed for reproducibility
    SEED_VALUE = 42
    set_reproducibility(SEED_VALUE)

    # Initialize the neural style transfer instance
    transfer_model = NST(style_img, content_img)

    # Create a random generated image to start the process
    generated_image = np.random.uniform(low=0.0, high=1.0, size=transfer_model.content_image.shape)
    generated_image = generated_image.astype('float32')

    # Preprocess the image for the VGG19 model
    vgg_model = tf.keras.applications.vgg19
    preprocessed_image = vgg_model.preprocess_input(generated_image * 255)

    # Get the output from the model
    model_outputs = transfer_model.model(preprocessed_image)

    # Extract the content output and calculate the content cost
    content_layer_output = model_outputs[-1]
    content_loss = transfer_model.content_cost(content_layer_output)

    print("Content loss:", content_loss)
