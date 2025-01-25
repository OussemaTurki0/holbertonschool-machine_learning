#!/usr/bin/env python3

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data

if __name__ == '__main__':
    np.random.seed(42)  # Set seed for reproducibility

    # Create example data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.array([0, 1, 0, 1])

    print("Original X:")
    print(X)
    print("\nOriginal Y:")
    print(Y)

    # Shuffle data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    print("\nShuffled X:")
    print(X_shuffled)
    print("\nShuffled Y:")
    print(Y_shuffled)

    # Verify that the shuffle maintains correspondence between X and Y
    for i in range(X_shuffled.shape[0]):
        original_index = np.where((X == X_shuffled[i]).all(axis=1))[0][0]
        assert Y_shuffled[i] == Y[original_index], "Mismatch in correspondence between X and Y!"
    print("\nShuffle verified: Correspondence between X and Y is maintained.")
