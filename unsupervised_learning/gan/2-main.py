#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random

# Import your WGAN_GP model
WGAN_GP = __import__('2-wgan_gp').WGAN_GP  # adjust if your file is named differently

## Set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

## Latent vector generators
def spheric_generator(n, d):
    u = tf.random.normal((n, d))
    return u / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u), axis=1) + 1e-8), [-1, 1])

## Create a 5D "4 clouds" dataset
def four_clouds_example(n):
    X = np.random.randn(n) * 0.05
    Y = np.random.randn(n) * 0.05
    X[:n//2] += 0.75
    X[n//2:] += 0.25
    Y[:n//4] += 0.75
    Y[n//4:n//2] += 0.25
    Y[n//2:3*n//4] += 0.75
    Y[3*n//4:] += 0.25
    R = np.minimum(X**2, 1)
    G = np.minimum(Y**2, 1)
    B = np.maximum(1 - R - G, 0)
    return tf.convert_to_tensor(np.stack([X, Y, R, G, B], axis=1), dtype=tf.float32)

## Build MLP generator and discriminator
def build_fc_gen_discr(shape, latent_type="spheric"):
    def latent_gen(n):
        return spheric_generator(n, shape[0])

    # Generator
    g_input = keras.Input(shape=(shape[0],))
    h = keras.layers.Dense(shape[1], activation='tanh')(g_input)
    for i in range(2, len(shape) - 1):
        h = keras.layers.Dense(shape[i], activation='tanh')(h)
    g_output = keras.layers.Dense(shape[-1], activation='sigmoid')(h)
    generator = keras.Model(g_input, g_output)

    # Discriminator
    d_input = keras.Input(shape=(shape[-1],))
    h = keras.layers.Dense(shape[-2], activation='tanh')(d_input)
    for i in range(2, len(shape) - 1):
        h = keras.layers.Dense(shape[-1 * i], activation='tanh')(h)
    d_output = keras.layers.Dense(1)(h)
    discriminator = keras.Model(d_input, d_output)

    return generator, discriminator, latent_gen

## Visualize 5D results
def visualize_5D(G, title=None, show=True):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flat:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    real = G.real_data.numpy()
    fake = G.get_fake_sample(10000).numpy()
    d_real = G.discriminator(G.real_data).numpy()
    d_fake = G.discriminator(tf.convert_to_tensor(fake)).numpy()

    axes[0, 0].scatter(real[:, 0], real[:, 1], c=real[:, 2:], s=1)
    axes[0, 0].set_title("Real Samples")

    axes[0, 1].scatter(fake[:, 0], fake[:, 1], c=fake[:, 2:], s=1)
    axes[0, 1].set_title("Fake Samples")

    vmin, vmax = d_real.min(), d_fake.max()
    axes[1, 0].scatter(real[:, 0], real[:, 1], c=d_real, s=1, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("D(real)")

    axes[1, 1].scatter(fake[:, 0], fake[:, 1], c=d_fake, s=1, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("D(fake)")

    latent = G.latent_generator(10000)
    latent_vals = tf.convert_to_tensor(latent)
    A = np.linspace(-3, 3, 150)
    B = np.linspace(-3, 3, 150)
    Xg, Yg = np.meshgrid(A, B)
    Z = tf.stack([Xg.ravel(), Yg.ravel()], axis=1)
    preds = G.discriminator(G.generator(Z)).numpy()
    axes[0, 2].pcolormesh(A, B, preds[:, 0].reshape(150, 150), shading='gouraud')
    axes[0, 2].set_title("D∘G on latent grid")

    axes[1, 2].scatter(latent[:, 0], latent[:, 1], c=d_fake, s=1)
    axes[1, 2].set_title("Latent points colored by D(fake)")

    if title:
        fig.suptitle(title)
    if show:
        plt.show()


## LET’S GO
set_seed()
real_data = four_clouds_example(1000)
generator, discriminator, latent_gen = build_fc_gen_discr([2, 10, 10, 5])
G = WGAN_GP(generator, discriminator, latent_gen, real_data,
            batch_size=200, disc_iter=5, learning_rate=0.0001)
G.compile()
G.fit(real_data, epochs=16, steps_per_epoch=100, verbose=1)
visualize_5D(G, title="WGAN-GP after 16 epochs")
