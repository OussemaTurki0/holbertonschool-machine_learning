#!/usr/bin/env python3

if __name__ == '__main__':
    minor = __import__('1-minor').minor
    # Test matrices
    mat0 = [[]]  # Empty matrix
    mat1 = [[5]]  # 1x1 matrix
    mat2 = [[1, 2], [3, 4]]  # 2x2 matrix
    mat3 = [[1, 2, 3], [0, 4, 5], [6, 7, 8]]  # 3x3 matrix
    mat4 = [[2, 3, 4, 5], [1, 0, 2, 3], [3, 5, 6, 7], [4, 5, 1, 2]]  # 4x4 matrix

    print("Testing minor function:")

    # Handle empty matrix case separately
    try:
        print(f"minor(mat0): {minor(mat0)}")  # Should raise an error or return something specific
    except Exception as e:
        print(f"Error with mat0 (empty matrix): {e}")
    
    print(f"minor(mat1): {minor(mat1)}")  # Should return [[1]]
    print(f"minor(mat2): {minor(mat2)}")  # Should return the minor matrix of 2x2
    print(f"minor(mat3): {minor(mat3)}")  # Should return the minor matrix of 3x3
    print(f"minor(mat4): {minor(mat4)}")  # Should return the minor matrix of 4x4

    try:
        print(f"minor([]): {minor([])}")  # Empty matrix, should raise an exception
    except Exception as e:
        print(f"Error with empty matrix: {e}")
        
    try:
        print(f"minor([[1, 2], [3]])")  # Non-square matrix, should raise an exception
        minor([[1, 2], [3]])
    except Exception as e:
        print(f"Error with non-square matrix: {e}")
