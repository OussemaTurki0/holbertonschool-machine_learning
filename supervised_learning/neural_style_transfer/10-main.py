#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import tensorflow as tf

#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('10-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    # Reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    nst = NST(style_image, content_image)
    generated_image, cost = nst.generate_image(iterations=2000, step=100, lr=0.002)
    print("Best cost:", cost)
    plt.imshow(generated_image)
    plt.show()
    mpimg.imsave("starry_gate2.jpg", generated_image)
def initialize_random_seeds(seed_value):
    """
    Set the random seed for reproducibility of results.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def load_image(image_path):
    """
    Helper function to load an image from the provided path.
    """
    return mpimg.imread(image_path)

if __name__ == '__main__':
    # Image paths
    style_img_path = "starry_night.jpg"
    content_img_path = "golden_gate.jpg"

    # Load the style and content images
    style_image = load_image(style_img_path)
    content_image = load_image(content_img_path)

    # Seed for reproducibility
    SEED = 42
    initialize_random_seeds(SEED)

    # Initialize the Neural Style Transfer model
    style_transfer = NST(style_image, content_image)

    # Generate the stylized image
    result_image, final_cost = style_transfer.generate_image(iterations=2000, step=100, lr=0.002)

    # Display the results
    print(f"Final cost after 2000 iterations: {final_cost}")
    plt.imshow(result_image)
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()

    # Save the output image
    output_image_path = "starry_gate_result.jpg"
    mpimg.imsave(output_image_path, result_image)
