#!/usr/bin/env python3

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

autoencoder = __import__('1-sparse').autoencoder

# Set seeds for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load and preprocess MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 784)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 784)).astype('float32') / 255.0

# Build and train the sparse autoencoder
encoder, decoder, auto = autoencoder(784, [128, 64], 32, 1e-5)
auto.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Encode and decode some test samples
encoded_imgs = encoder.predict(x_test[:10])
decoded_imgs = decoder.predict(encoded_imgs)

print(np.mean(encoded_imgs))

# Plot original and reconstructed images
for idx in range(10):
    ax = plt.subplot(2, 10, idx + 1)
    ax.axis('off')
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')

    ax = plt.subplot(2, 10, idx + 11)
    ax.axis('off')
    plt.imshow(decoded_imgs[idx].reshape(28, 28), cmap='gray')

plt.show()
