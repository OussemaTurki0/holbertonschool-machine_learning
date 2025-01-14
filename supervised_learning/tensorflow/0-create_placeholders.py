#!/usr/bin/env python3
"""Defines a function to create placeholders"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates and returns two placeholders for the input data and labels.
    Args:
        nx (int): Number of features in the input data.
        classes (int): Number of classes for classification.
    Returns:
        x (tf.placeholder): Placeholder for input data.
        y (tf.placeholder): Placeholder for one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
