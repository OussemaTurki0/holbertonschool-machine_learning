#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    # Test matrices
    mat1 = [[5]]  # 1x1 matrix
    mat2 = [[1, 2], [3, 4]]  # 2x2 matrix
    mat3 = [[1, 2, 3], [0, 4, 5], [6, 7, 8]]  # 3x3 matrix
    mat4 = [[2, 3, 4, 5], [1, 0, 2, 3], [3, 5, 6, 7], [4, 5, 1, 2]]  # 4x4 matrix

    print("Testing cofactor function:")

    print(f"cofactor(mat1): {cofactor(mat1)}")  # Should return the cofactor of a 1x1 matrix
    print(f"cofactor(mat2): {cofactor(mat2)}")  # Should return the cofactor matrix of 2x2
    print(f"cofactor(mat3): {cofactor(mat3)}")  # Should return the cofactor matrix of 3x3
    print(f"cofactor(mat4): {cofactor(mat4)}")  # Should return the cofactor matrix of 4x4

    try:
        print(f"cofactor([]): {cofactor([])}")  # Empty matrix, should raise an exception
    except Exception as e:
        print(f"Error with empty matrix: {e}")
        
    try:
        print(f"cofactor([[1, 2], [3]])")  # Non-square matrix, should raise an exception
        cofactor([[1, 2], [3]])
    except Exception as e:
        print(f"Error with non-square matrix: {e}")
