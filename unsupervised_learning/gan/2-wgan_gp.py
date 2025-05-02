#!/usr/bin/env python3
"""
WGAN-GP (Wasserstein GAN with Gradient Penalty)
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class WGAN_GP(keras.Model):
    """
    Implementation of WGAN with Gradient Penalty.
    """

    def __init__(self, generator, discriminator, latent_fn,
                 real_data, batch_size=200, disc_steps=2,
                 lr=0.005, lambda_gp=10):
        """
        Initialize WGAN_GP model.
        """
        super().__init__()
        self.latent_fn = latent_fn
        self.real_data = real_data
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_steps = disc_steps

        self.lr = lr
        self.lambda_gp = lambda_gp
        self.beta1 = 0.3
        self.beta2 = 0.9

        # Generator: maximize D(G(z)) → minimize -D(G(z))
        self.generator.loss = lambda logits: -tf.reduce_mean(logits)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2
        )
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # Discriminator: maximize D(real) - D(fake)
        self.discriminator.loss = lambda real, fake: tf.reduce_mean(fake) - tf.reduce_mean(real)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2
        )
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_real_sample(self, size=None):
        """
        Sample real data batch.
        """
        if size is None:
            size = self.batch_size
        indices = tf.random.shuffle(tf.range(tf.shape(self.real_data)[0]))[:size]
        return tf.gather(self.real_data, indices)

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake data from latent vectors.
        """
        if size is None:
            size = self.batch_size
        z = self.latent_fn(size)
        return self.generator(z, training=training)

    def interpolate(self, real, fake):
        """
        Generate random interpolation between real and fake.
        """
        alpha = tf.random.uniform(shape=[self.batch_size] + [1]*(len(real.shape)-1))
        return real * alpha + fake * (1 - alpha)

    def compute_gp(self, interpolated):
        """
        Compute gradient penalty from interpolated samples.
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = self.discriminator(interpolated, training=True)
        grads = tape.gradient(preds, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=list(range(1, len(grads.shape)))))
        return tf.reduce_mean(tf.square(norm - 1.0))

    def train_step(self, _):
        """
        Performs one training step.
        """
        for _ in range(self.disc_steps):
            with tf.GradientTape() as tape:
                real_batch = self.get_real_sample()
                fake_batch = self.get_fake_sample(training=True)
                inter_samples = self.interpolate(real_batch, fake_batch)

                real_logits = self.discriminator(real_batch, training=True)
                fake_logits = self.discriminator(fake_batch, training=True)

                disc_loss = self.discriminator.loss(real_logits, fake_logits)
                gp_term = self.compute_gp(inter_samples)
                total_disc_loss = disc_loss + self.lambda_gp * gp_term

            disc_grads = tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(disc_grads, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_batch = self.get_fake_sample(training=True)
            gen_logits = self.discriminator(fake_batch, training=False)
            gen_loss = self.generator.loss(gen_logits)

        gen_grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {
            "discr_loss": disc_loss,
            "gen_loss": gen_loss,
            "gp": gp_term
        }
