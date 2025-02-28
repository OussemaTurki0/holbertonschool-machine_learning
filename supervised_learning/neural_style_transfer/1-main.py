#!/usr/bin/env python3

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

NST = __import__('1-neural_style').NST

def load_image(image_path):
    """Load and normalize an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: '{image_path}' not found!")

    image = mpimg.imread(image_path)
    if image.max() > 1:
        image = image / 255.0  # Normalize if needed
    return image

if __name__ == '__main__':
    style_path = "starry_night.jpg"
    content_path = "golden_gate.jpg"

    try:
        style_image = load_image(style_path)
        content_image = load_image(content_path)
        
        nst = NST(style_image, content_image)
        nst.model.summary()
    except Exception as e:
        print(f"An error occurred: {e}")
