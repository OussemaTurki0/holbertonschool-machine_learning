#!/usr/bin/env python3

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

autoencoder = __import__('3-variational').autoencoder

# Set seeds for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 784)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 784)).astype('float32') / 255.0

# Build and train the variational autoencoder
encoder, decoder, auto = autoencoder(784, [512], 2)
auto.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Encode and reconstruct some test samples
z_samples, mu, log_var = encoder.predict(x_test[:10])
print("Latent means:\n", mu)
print("Latent std deviations:\n", np.exp(log_var / 2))

reconstructed_imgs = decoder.predict(z_samples).reshape((-1, 28, 28))
x_test_imgs = x_test.reshape((-1, 28, 28))

# Plot original and reconstructed digits
for idx in range(10):
    ax = plt.subplot(2, 10, idx + 1)
    ax.axis('off')
    plt.imshow(x_test_imgs[idx], cmap='gray')

    ax = plt.subplot(2, 10, idx + 11)
    ax.axis('off')
    plt.imshow(reconstructed_imgs[idx], cmap='gray')

plt.show()

# Sampling from latent space grid
l1 = np.linspace(-3, 3, 25)
l2 = np.linspace(-3, 3, 25)
grid = np.stack(np.meshgrid(l1, l2, indexing='ij'), axis=-1)
generated_imgs = decoder.predict(grid.reshape((-1, 2)), batch_size=125)

# Plotting the generated digits
for idx in range(25 * 25):
    ax = plt.subplot(25, 25, idx + 1)
    ax.axis('off')
    plt.imshow(generated_imgs[idx].reshape((28, 28)), cmap='gray')

plt.show()
