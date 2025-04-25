#!/usr/bin/env python3

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

autoencoder = __import__('2-convolutional').autoencoder

# Set random seeds for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load and preprocess MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

# Build and train the convolutional autoencoder
encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
auto.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Encode and decode test images
latent_features = encoder.predict(x_test[:10])
decoded_imgs = decoder.predict(latent_features)[:, :, :, 0]

print("Mean of encoded features:", np.mean(latent_features))

# Plotting original and reconstructed images
for idx in range(10):
    ax = plt.subplot(2, 10, idx + 1)
    ax.axis('off')
    plt.imshow(x_test[idx, :, :, 0], cmap='gray')

    ax = plt.subplot(2, 10, idx + 11)
    ax.axis('off')
    plt.imshow(decoded_imgs[idx], cmap='gray')

plt.show()
