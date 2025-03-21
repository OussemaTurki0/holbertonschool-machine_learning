#!/usr/bin/env python3

if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    # Define test matrices
    matrices = [
        (np.array([[5, 1], [1, 1]]), "Matrix 2x2 (positive definite)"),
        (np.array([[2, 4], [4, 8]]), "Matrix 2x2 (singular)"),
        (np.array([[-1, 1], [1, -1]]), "Matrix 2x2 (indefinite)"),
        (np.array([[-2, 4], [4, -9]]), "Matrix 2x2 (negative definite)"),
        (np.array([[1, 2], [2, 1]]), "Matrix 2x2 (positive semi-definite)"),
        (np.array([]), "Empty matrix"),
        (np.array([[1, 2, 3], [4, 5, 6]]), "Matrix 2x3 (non-square)"),
        ([[1, 2], [1, 2]], "List input instead of numpy array"),
    ]

    # Test each matrix and handle errors
    for mat, description in matrices:
        print(f"Testing {description}:")
        try:
            print(definiteness(mat))
        except Exception as e:
            print(f"Error for {description}: {e}")
        print("-" * 40)
