#!/usr/bin/env python3
"""
Simple GAN architecture implementation.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class Simple_GAN(keras.Model):
    """
    A simple Generative Adversarial Network (GAN) class.
    """

    def __init__(self, generator, discriminator, latent_gen_fn,
                 real_data, batch_size=200, disc_steps=2,
                 lr=0.005):
        """
        Constructor for the Simple_GAN model.
        """
        super().__init__()
        self.latent_gen_fn = latent_gen_fn
        self.real_data = real_data
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_steps = disc_steps

        self.lr = lr
        self.beta1 = 0.5
        self.beta2 = 0.9

        # Configure generator
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(x, tf.ones_like(x))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.lr,
            beta_1=self.beta1,
            beta_2=self.beta2
        )
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)

        # Configure discriminator
        self.discriminator.loss = lambda real, fake: (
            tf.keras.losses.MeanSquaredError()(real, tf.ones_like(real)) +
            tf.keras.losses.MeanSquaredError()(fake, -tf.ones_like(fake))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.lr,
            beta_1=self.beta1,
            beta_2=self.beta2
        )
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss)

    def get_real_sample(self, size=None):
        """
        Fetches a random batch of real samples.
        """
        if size is None:
            size = self.batch_size
        total = tf.shape(self.real_data)[0]
        indices = tf.random.shuffle(tf.range(total))[:size]
        return tf.gather(self.real_data, indices)

    def get_fake_sample(self, size=None, training=False):
        """
        Produces a batch of fake samples from latent vectors.
        """
        if size is None:
            size = self.batch_size
        latents = self.latent_gen_fn(size)
        return self.generator(latents, training=training)

    def train_step(self, _):
        """
        Executes a single training iteration.
        """
        for _ in range(self.disc_steps):
            with tf.GradientTape() as tape:
                real_data = self.get_real_sample()
                fake_data = self.get_fake_sample(training=True)

                pred_real = self.discriminator(real_data, training=True)
                pred_fake = self.discriminator(fake_data, training=True)

                d_loss = self.discriminator.loss(pred_real, pred_fake)

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            pred = self.discriminator(fake_batch, training=False)
            g_loss = self.generator.loss(pred)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        return {"discr_loss": d_loss, "gen_loss": g_loss}
