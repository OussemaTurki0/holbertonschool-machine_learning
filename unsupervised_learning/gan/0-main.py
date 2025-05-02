#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt

WGAN_clip = __import__('1-wgan_clip').WGAN_clip


# Set random seeds for reproducibility
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Custom tensor hashing for verification
def tensor_hash(tensor):
    flat = tensor.numpy().flatten()
    return sum([hash(x) % (1 << 30) for x in flat]) % (1 << 30)


# Latent vector generators
def spherical_latents(n, d):
    u = tf.random.normal((n, d))
    return u / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u), axis=1) + 1e-8), [-1, 1])

def normal_latents(n, d):
    return tf.random.normal((n, d))

def uniform_latents(n, d):
    return tf.random.uniform((n, d))


# Build generator and discriminator
def build_fc_gan_layers(structure, real_data=None, latent_kind="normal"):
    # Latent generator
    if latent_kind == "uniform":
        latent_fn = lambda k: uniform_latents(k, structure[0])
    elif latent_kind == "normal":
        latent_fn = lambda k: normal_latents(k, structure[0])
    else:
        latent_fn = lambda k: spherical_latents(k, structure[0])

    # Generator model
    g_in = keras.Input(shape=(structure[0],))
    h = keras.layers.Dense(structure[1], activation='tanh')(g_in)
    for i in range(2, len(structure) - 1):
        h = keras.layers.Dense(structure[i], activation='tanh')(h)
    g_out = keras.layers.Dense(structure[-1], activation='sigmoid')(h)
    generator = keras.Model(g_in, g_out)

    # Discriminator model
    d_in = keras.Input(shape=(structure[-1],))
    h = keras.layers.Dense(structure[-2], activation='tanh')(d_in)
    for i in range(2, len(structure) - 1):
        h = keras.layers.Dense(structure[-1 * i], activation='tanh')(h)
    d_out = keras.layers.Dense(1, activation='tanh')(h)
    discriminator = keras.Model(d_in, d_out)

    return generator, discriminator, latent_fn


# Train a GAN with the given class and structure
def train_gan(gan_type, data, model_structure, epochs,
              batch_size=200, steps=250, latent_kind="normal", lr=0.005):
    gen, disc, lat_fn = build_fc_gan_layers(model_structure, data, latent_kind)
    gan = gan_type(gen, disc, lat_fn, data, learning_rate=lr)
    gan.compile()
    gan.fit(data, epochs=epochs, steps_per_epoch=steps, verbose=1)
    return gan


# Sample dataset: 4 colorful clouds
def make_four_clouds(n):
    x = np.random.randn(n) * 0.05
    y = np.random.randn(n) * 0.05
    x[:n // 2] += 0.75
    x[n // 2:] += 0.25
    y[n // 4:n // 2] += 0.25
    y[:n // 4] += 0.75
    y[n // 2:3 * n // 4] += 0.75
    y[3 * n // 4:] += 0.25
    r = np.minimum(x ** 2, 1)
    g = np.minimum(y ** 2, 1)
    b = np.maximum(1 - r - g, 0)
    return tf.convert_to_tensor(np.stack([x, y, r, g, b], axis=-1))


# Visualization of 5D data (RGB-coded)
def plot_5d_gan_result(G, show=True, title=None, save_as=None, dpi=200):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flat:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    X = G.real_examples.numpy()
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=X[:, 2:], s=1)
    axes[0, 0].set_title("Real Samples")

    latents = G.latent_generator(10000)
    Y = G.generator(latents).numpy()
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], c=Y[:, 2:], s=1)
    axes[0, 1].set_title("Fake Samples")

    d_real = G.discriminator(G.real_examples).numpy()
    d_fake = G.discriminator(G.generator(latents)).numpy()
    vmin = min(d_real.min(), d_fake.min())
    vmax = max(d_real.max(), d_fake.max())

    axes[1, 0].scatter(X[:, 0], X[:, 1], c=d_real, s=1, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("D(real)")

    axes[1, 1].scatter(Y[:, 0], Y[:, 1], c=d_fake, s=1, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("D(fake)")

    for i in range(2):
        axes[i, 2].set_xlim(-3, 3)
        axes[i, 2].set_ylim(-3, 3)

    a = np.linspace(-3, 3, 150)
    b = np.linspace(-3, 3, 150)
    Xg, Yg = np.meshgrid(a, b)
    latent_grid = tf.convert_to_tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=-1))
    d_map = G.discriminator(G.generator(latent_grid)).numpy()
    axes[0, 2].pcolormesh(a, b, d_map[:, 0].reshape(150, 150), shading='gouraud')
    axes[0, 2].set_title("D∘G on latent grid")

    axes[1, 2].scatter(latents.numpy()[:, 0], latents.numpy()[:, 1], c=d_fake, s=1)
    axes[1, 2].set_title("D∘G on samples")

    if title:
        fig.suptitle(title)

    if save_as:
        plt.savefig(save_as, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    elif show:
        plt.show()


# Run training and visualize results
set_random_seeds(0)
gan_model = train_gan(WGAN_clip, make_four_clouds(1000), [2, 10, 10, 5], 16, steps=100, lr=0.001)
plot_5d_gan_result(gan_model, show=True, title="WGAN-Clip after 16 epochs")
