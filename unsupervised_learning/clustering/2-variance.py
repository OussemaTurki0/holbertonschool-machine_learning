#!/usr/bin/env python3
"""
Cluster Variance Calculation
"""

import numpy as np


def variance(X, C):
    """
    Computes the total intra-cluster variance for a given dataset.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the dataset.
                           - n: Number of data points
                           - d: Number of dimensions per data point
        C (numpy.ndarray): 2D array of shape (k, d) containing the cluster centroids.
                           - k: Number of clusters

    Returns:
        float: The calculated total variance, or None if inputs are invalid.
    """
    # Validate input data types and dimensions
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate the distance between each data point and each centroid
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    # Find the closest centroid for each point and compute variance
    min_distances = np.min(distances, axis=1)
    variance = np.sum(min_distances ** 2)

    return variance
