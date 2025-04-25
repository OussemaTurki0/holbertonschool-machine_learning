#!/usr/bin/env python3
"""
Test file for 2-convolutional.py (convolutional autoencoder function)
"""

import numpy as np
autoencoder = __import__('2-convolutional').autoencoder

# Parameters for the test
input_dims = (28, 28, 1)  # Example: grayscale 28x28 images (like MNIST)
filters = [32, 64]
latent_dims = (7, 7, 64)  # After two maxpool layers (divided by 2 each time)

# Build the models
encoder, decoder, auto = autoencoder(input_dims, filters, latent_dims)

# Print model summaries
print("Encoder Summary:")
encoder.summary()

print("\nDecoder Summary:")
decoder.summary()

print("\nConvolutional Autoencoder Summary:")
auto.summary()

# Create fake image data
X_test = np.random.rand(10, 28, 28, 1)  # 10 fake images
reconstructed = auto.predict(X_test)

print("\nOriginal shape:", X_test.shape)
print("Reconstructed shape:", reconstructed.shape)
