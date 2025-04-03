#!/usr/bin/env python3
"""
Initialize GMM
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans

def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model (GMM).

    Parameters:
    X (numpy.ndarray): 2D array of shape (n, d) representing the dataset.
    k (int): Number of clusters (positive integer).

    Returns:
    tuple: (pi, m, S), or (None, None, None) on failure.
        - pi: Prior probabilities for each cluster, evenly distributed.
        - m: Centroid means for each cluster, initialized using K-means.
        - S: Covariance matrices for each cluster, as identity matrices.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    # Initialize prior probabilities equally among clusters
    pi = np.ones((k,)) / k

    # Initialize centroids using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Initialize covariance matrices as identity matrices
    S = np.repeat(np.eye(X.shape[1])[np.newaxis, :, :], k, axis=0)

    return pi, m, S
