#!/usr/bin/env python3
"""
Test script for NST class
"""

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
NST = __import__('0-neural_style').NST


def main():
    """
    Main function to test NST class functionality
    """
    style_img = mpimg.imread("starry_night.jpg")
    content_img = mpimg.imread("golden_gate.jpg")

    print("Style Layers:", NST.style_layers)
    print("Content Layer:", NST.content_layer)

    nst_model = NST(style_img, content_img)
    processed_style = nst_model.scale_image(style_img)
    processed_content = nst_model.scale_image(content_img)

    print("Style Image:", type(nst_model.style_image), nst_model.style_image.shape,
          "Min:", np.min(nst_model.style_image), "Max:", np.max(nst_model.style_image))
    print("Content Image:", type(nst_model.content_image), nst_model.content_image.shape,
          "Min:", np.min(nst_model.content_image), "Max:", np.max(nst_model.content_image))
    print("Alpha:", nst_model.alpha)
    print("Beta:", nst_model.beta)
    print("Eager Execution Enabled:", tf.executing_eagerly())

    assert np.array_equal(processed_style, nst_model.style_image), "Style image scaling mismatch"
    assert np.array_equal(processed_content, nst_model.content_image), "Content image scaling mismatch"

    plt.imshow(nst_model.style_image[0])
    plt.title("Processed Style Image")
    plt.show()
    
    plt.imshow(nst_model.content_image[0])
    plt.title("Processed Content Image")
    plt.show()

if __name__ == "__main__":
    main()
