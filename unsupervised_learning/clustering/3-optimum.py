#!/usr/bin/env python3
"""
Optimizing k - Kmeans
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Determines the optimal number of clusters using variance reduction.

    Parameters:
    - X (numpy.ndarray): 2D array of shape (n, d) representing the dataset.
    - kmin (int): Minimum number of clusters (inclusive).
    - kmax (int): Maximum number of clusters (inclusive).
    - iterations (int): Maximum number of iterations for K-means.

    Returns:
    - tuple: (results, d_vars), or (None, None) on failure.
        - results: List of tuples with centroids and classifications for each cluster size.
        - d_vars: List of variance differences from the smallest cluster size.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    max_clusters = X.shape[0] if kmax is None else kmax
    results, d_vars = [], []

    # Calculate variance for the smallest number of clusters
    C, clss = kmeans(X, kmin, iterations)
    base_variance = variance(X, C)
    results.append((C, clss))
    d_vars.append(0.0)

    # Iterate through the remaining cluster sizes
    for k in range(kmin + 1, max_clusters + 1):
        C, clss = kmeans(X, k, iterations)
        current_variance = variance(X, C)
        results.append((C, clss))
        d_vars.append(base_variance - current_variance)

    return results, d_vars
