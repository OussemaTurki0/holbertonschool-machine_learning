#!/usr/bin/env python3
"""
WGAN_clip module
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    WGAN_clip class
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initializes the WGAN clip model.
        """
        super().__init__()  # Initialize Keras.Model
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)  # Opposite
        self.generator.optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # Discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
                tf.reduce_mean(y) - tf.reduce_mean(x))  # Difference
        self.discriminator.optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    # Generate real samples
    def get_real_sample(self, size=None):
        """
        Generates a batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Generate fake samples
    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # Perform one training step
    def train_step(self, useless_argument):
        """
        Performs one training step for the WGAN.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(real_output, fake_output)

            discr_grads = tape.gradient(discr_loss,
                                        self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                    zip(discr_grads,
                        self.discriminator.trainable_variables)
                    )

            # Clip the weights of the discriminator between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            gen_output = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(gen_output)

        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
                zip(gen_grads,
                    self.generator.trainable_variables)
                )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}