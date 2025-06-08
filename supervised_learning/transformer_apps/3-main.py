#!/usr/bin/env python3

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset

# Set a fixed seed for reproducibility
tf.random.set_seed(0)

# Initialize dataset with batch size 32 and max tokens 40
corpus = Dataset(batch_size=32, max_len=40)

# Display a training batch sample
for src, tgt in corpus.data_train.take(1):
    print(src, tgt)

# Display a validation batch sample
for src, tgt in corpus.data_valid.take(1):
    print(src, tgt)
