#!/usr/bin/env python3

import numpy as np
create_mini_batches = __import__('3-mini_batch').create_mini_batches

if __name__ == '__main__':
    np.random.seed(42)  # Set seed for reproducibility

    # Example input data and labels
    X = np.array([[i, i + 1] for i in range(10)])  # 10 samples, 2 features each
    Y = np.array([i % 2 for i in range(10)])  # Binary labels for 10 samples
    batch_size = 3

    print("Input data (X):")
    print(X)
    print("\nInput labels (Y):")
    print(Y)

    # Create mini-batches
    mini_batches = create_mini_batches(X, Y, batch_size)

    print(f"\nMini-batches (batch size = {batch_size}):")
    for idx, (X_batch, Y_batch) in enumerate(mini_batches):
        print(f"\nBatch {idx + 1}:")
        print("X_batch:")
        print(X_batch)
        print("Y_batch:")
        print(Y_batch)

    # Verify all samples are included and correspondence is maintained
    total_samples = sum(batch[0].shape[0] for batch in mini_batches)
    assert total_samples == X.shape[0], "Not all samples are included in the mini-batches!"

    print("\nAll samples are included, and mini-batches created successfully.")
