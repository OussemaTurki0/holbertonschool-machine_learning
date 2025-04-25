#!/usr/bin/env python3
"""
Test file for 1-sparse.py (sparse autoencoder function)
"""

import numpy as np
autoencoder = __import__('1-sparse').autoencoder

# Parameters for the test
input_dims = 784  # Example: 28x28 images flattened
hidden_layers = [128, 64]
latent_dims = 32
lambtha = 1e-5  # small lambda value for L1 regularization

# Build the models
encoder, decoder, auto = autoencoder(input_dims, hidden_layers, latent_dims, lambtha)

# Print model summaries
print("Encoder Summary:")
encoder.summary()

print("\nDecoder Summary:")
decoder.summary()

print("\nSparse Autoencoder Summary:")
auto.summary()

# Create fake data
X_test = np.random.rand(10, input_dims)  # 10 samples
reconstructed = auto.predict(X_test)

print("\nOriginal shape:", X_test.shape)
print("Reconstructed shape:", reconstructed.shape)
