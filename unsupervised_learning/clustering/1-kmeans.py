#!/usr/bin/env python3

"""
K-means Clustering Implementation
"""

import numpy as np


def initialize(X, k):
    """
    Initializes centroids for K-means clustering.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the dataset.
                           - n: Number of data points
                           - d: Number of dimensions per data point
        k (int): Number of clusters to form.

    Returns:
        numpy.ndarray: 2D array of shape (k, d) with the initialized centroids,
                       or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))
    return centroids


def kmeans(X, k, max_iters=1000):
    """
    Applies K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) representing the dataset.
                           - n: Number of data points
                           - d: Number of dimensions per data point
        k (int): Number of clusters to form.
        max_iters (int): Maximum number of iterations to run the algorithm.

    Returns:
        tuple: (centroids, labels), where:
            - centroids (numpy.ndarray): Shape (k, d), cluster centroid coordinates.
            - labels (numpy.ndarray): Shape (n,), cluster assignment for each point.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(max_iters, int) or max_iters <= 0:
        return None, None

    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    for iteration in range(max_iters):
        prev_centroids = np.copy(centroids)

        # Calculate distances from each point to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids by averaging the points in each cluster
        for i in range(k):
            cluster_points = X[labels == i]
            if cluster_points.size == 0:
                centroids[i] = initialize(X, 1)
            else:
                centroids[i] = np.mean(cluster_points, axis=0)

        # Check for convergence
        if np.allclose(centroids, prev_centroids):
            break

    return centroids, labels
