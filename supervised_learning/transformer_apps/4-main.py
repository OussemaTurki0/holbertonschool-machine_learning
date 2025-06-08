#!/usr/bin/env python3

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks

# Set seed for consistent results
tf.random.set_seed(0)

# Initialize the dataset with specific batch size and max length
dataset = Dataset(batch_size=32, max_len=40)

# Generate and display masks for a sample batch
for source_seq, target_seq in dataset.data_train.take(1):
    masks = create_masks(source_seq, target_seq)
    print(masks)
