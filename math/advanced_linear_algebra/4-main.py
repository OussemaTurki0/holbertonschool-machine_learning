#!/usr/bin/env python3

if __name__ == '__main__':
    inverse = __import__('4-inverse').inverse

    # Define test matrices
    test_matrices = [
        ([[5]], "Matrix 1x1"),
        ([[1, 2], [3, 4]], "Matrix 2x2"),
        ([[1, 1], [1, 1]], "Matrix 2x2 (singular)"),
        ([[5, 7, 9], [3, 1, 8], [6, 2, 4]], "Matrix 3x3"),
        ([], "Empty matrix"),
        ([[1, 2, 3], [4, 5, 6]], "Matrix 2x3 (non-square)"),
    ]

    # Iterate over test cases
    for mat, description in test_matrices:
        print(f"Testing {description}:")
        try:
            print(inverse(mat))
        except Exception as e:
            print(f"Error for {description}: {e}")
        print("-" * 40)
