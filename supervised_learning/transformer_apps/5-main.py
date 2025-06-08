#!/usr/bin/env python3

import tensorflow as tf
train_model = __import__('5-train').train_transformer

# Ensure reproducibility
tf.random.set_seed(0)

# Train a transformer model with specified hyperparameters
model = train_model(num_layers=4, d_model=128, num_heads=8,
                    dff=512, batch_size=32, max_len=40, epochs=2)

# Display the model type
print(type(model))
