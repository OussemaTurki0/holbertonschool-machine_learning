#!/usr/bin/env python3
"""
Test file for absorbing function.
"""

import numpy as np
absorbing = __import__('2-absorbing').absorbing

# Test Case 1: Absorbing matrix
P1 = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.5, 0.5, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.5, 0.5]])

print("P1 is absorbing?", absorbing(P1))  # Expected: True

# Test Case 2: Not absorbing
P2 = np.array([[0.5, 0.5],
               [0.2, 0.8]])

print("P2 is absorbing?", absorbing(P2))  # Expected: False

# Test Case 3: All states absorbing
P3 = np.eye(3)
print("P3 is absorbing?", absorbing(P3))  # Expected: True

# Test Case 4: Invalid input
P4 = "not a matrix"
print("P4 is absorbing?", absorbing(P4))  # Expected: False

# Test Case 5: Single state absorbing
P5 = np.array([[1]])
print("P5 is absorbing?", absorbing(P5))  # Expected: True
