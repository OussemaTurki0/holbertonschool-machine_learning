#!/usr/bin/env python3
"""
Test file for 0-vanilla.py (autoencoder function)
"""

import numpy as np
autoencoder = __import__('0-vanilla').autoencoder

# Parameters for the test
input_dims = 784  # Example: 28x28 images flattened
hidden_layers = [128, 64]
latent_dims = 32

# Build the models
encoder, decoder, auto = autoencoder(input_dims, hidden_layers, latent_dims)

# Print model summaries
print("Encoder Summary:")
encoder.summary()

print("\nDecoder Summary:")
decoder.summary()

print("\nAutoencoder Summary:")
auto.summary()

# Create fake data
X_test = np.random.rand(10, input_dims)  # 10 samples
reconstructed = auto.predict(X_test)

print("\nOriginal shape:", X_test.shape)
print("Reconstructed shape:", reconstructed.shape)
