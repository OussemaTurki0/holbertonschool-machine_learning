#!/usr/bin/env python3

import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.image as mpimg

# Import custom NST class from 3-neural_style
NST = __import__('3-neural_style').NST


def set_random_seed(seed_value):
    """Sets the seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def main():
    """Main function to perform neural style transfer."""
    # Load the images
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    # Set seed for reproducibility
    SEED = 42  # Changed the seed to make it different from your friend's code
    set_random_seed(SEED)

    # Initialize NST class with images
    neural_style = NST(style_image, content_image)

    # Display features
    print("Gram style features:", neural_style.gram_style_features)
    print("Content feature:", neural_style.content_feature)


if __name__ == '__main__':
    main()
