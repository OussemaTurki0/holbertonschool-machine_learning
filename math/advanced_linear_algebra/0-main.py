#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    # Test matrices
    mat0 = [[]]  # Empty matrix
    mat1 = [[5]]  # 1x1 matrix
    mat2 = [[1, 2], [3, 4]]  # 2x2 matrix
    mat3 = [[1, 1], [1, 1]]  # 2x2 matrix with identical rows
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]  # 3x3 matrix
    mat5 = []  # Empty matrix (should raise an error)
    mat6 = [[1, 2, 3], [4, 5, 6]]  # Non-square matrix (should raise an error)

    # Test each matrix and print the result or exception
    print("Testing determinant function:")
    
    print(f"determinant(mat0): {determinant(mat0)}")
    print(f"determinant(mat1): {determinant(mat1)}")
    print(f"determinant(mat2): {determinant(mat2)}")
    print(f"determinant(mat3): {determinant(mat3)}")
    print(f"determinant(mat4): {determinant(mat4)}")
    
    try:
        print(f"determinant(mat5): {determinant(mat5)}")
    except Exception as e:
        print(f"Error with mat5: {e}")
        
    try:
        print(f"determinant(mat6): {determinant(mat6)}")
    except Exception as e:
        print(f"Error with mat6: {e}")
