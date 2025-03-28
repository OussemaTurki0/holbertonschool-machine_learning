#!/usr/bin/env python3

import numpy as np
pca = __import__('0-pca').pca

# Seed for reproducibility
np.random.seed(0)

# Generate random data for the demonstration
x_vals = np.random.normal(size=50)
y_vals = np.random.normal(size=50)
z_vals = np.random.normal(size=50)

# Create correlated features
feature_1 = 2 * x_vals
feature_2 = -5 * y_vals
feature_3 = 10 * z_vals

# Combine all features into a data matrix
data_matrix = np.array([x_vals, y_vals, z_vals, feature_1, feature_2, feature_3]).T

# Standardize the data (mean centering)
mean_centered_data = data_matrix - np.mean(data_matrix, axis=0)

# Perform PCA to get the principal components
principal_components = pca(mean_centered_data)

# Project the data onto the principal components
transformed_data = np.matmul(mean_centered_data, principal_components)

# Output the transformed data
print(transformed_data)

# Reconstruct the data using the inverse transformation
reconstructed_data = np.matmul(transformed_data, principal_components.T)

# Calculate and print the reconstruction error (mean squared error)
mse = np.sum(np.square(mean_centered_data - reconstructed_data)) / data_matrix.shape[0]
print(f"Mean Squared Error: {mse}")
