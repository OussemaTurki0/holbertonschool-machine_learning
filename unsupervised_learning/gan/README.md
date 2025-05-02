# GANs - Generative Adversarial Networks

This project introduces the fundamental concepts behind Generative Adversarial Networks (GANs) and walks through several key variants: Simple GANs, Wasserstein GANs (WGANs), and Wasserstein GANs with Gradient Penalty (WGAN-GP). The goal is to build an intuitive and practical understanding of how GANs work by progressively implementing and testing different architectures.

## Objectives

- Understand the adversarial learning framework of GANs.
- Implement a basic GAN (Simple_GAN) where two neural networks – generator and discriminator – are trained in opposition.
- Learn about Wasserstein GANs, which improve training stability and convergence through a better loss function and weight clipping.
- Extend to WGAN-GP, which introduces gradient penalty to enforce Lipschitz continuity without clipping weights.
- Train a GAN to generate realistic human faces from random noise.
- Compare the capabilities of GANs to traditional methods like PCA in modeling image data.

## Project Structure

### Task 0: Simple GAN
A basic adversarial training model using Mean Squared Error losses for both generator and discriminator.

### Task 1: WGAN with Weight Clipping
A modified GAN using the Wasserstein loss, replacing sigmoid activations and using weight clipping for the discriminator.

### Task 2: WGAN-GP (Gradient Penalty)
Improves upon WGAN by using a gradient penalty instead of weight clipping for more stable training.

### Task 3: Convolutional Networks
Extends previous models to work with image data by using CNN-based generators and discriminators.

### Task 4: Face Generator
Trains a WGAN-GP model on a dataset of face images to generate synthetic faces that look realistic.

### Appendix: PCA vs GANs
Includes a brief comparison between PCA-based image generation and GANs, showing how GANs outperform PCA in generating realistic samples.
