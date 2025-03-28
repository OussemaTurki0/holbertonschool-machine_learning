#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca

# Load dataset (MNIST example)
dataset = np.loadtxt("mnist2500_X.txt")

# Print the shape and contents of the dataset
print('Dataset Shape:', dataset.shape)
print('Dataset Preview:', dataset)

# Perform PCA to reduce the dataset to 50 principal components
reduced_data = pca(dataset, 50)

# Output the shape and contents of the transformed data
print('Transformed Data Shape:', reduced_data.shape)
print('Transformed Data Preview:', reduced_data)
