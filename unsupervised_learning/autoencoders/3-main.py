#!/usr/bin/env python3
"""
Test file for 3-variational.py (variational autoencoder function)
"""

import numpy as np
autoencoder = __import__('3-variational').autoencoder

# Parameters for the test
input_dims = 784  # Example: flattened 28x28 images
hidden_layers = [512, 256]
latent_dims = 2  # Small latent space for VAE visualization

# Build the models
encoder, decoder, auto = autoencoder(input_dims, hidden_layers, latent_dims)

# Print model summaries
print("Encoder Summary:")
encoder.summary()

print("\nDecoder Summary:")
decoder.summary()

print("\nVariational Autoencoder (VAE) Summary:")
auto.summary()

# Create fake input data
X_test = np.random.rand(10, input_dims)  # 10 random samples
reconstructed = auto.predict(X_test)

print("\nOriginal shape:", X_test.shape)
print("Reconstructed shape:", reconstructed.shape)
