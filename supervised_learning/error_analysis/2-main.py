#!/usr/bin/env python3
"""
Test file for precision function
"""

import numpy as np
precision = __import__('2-precision').precision

# Test case: 3-class confusion matrix
confusion_matrix = np.array([
    [5, 1, 0],  # Class 0: 5 TP, 1 FP
    [2, 3, 1],  # Class 1: 3 TP, 2+1 FP
    [0, 2, 4]   # Class 2: 4 TP, 2 FP
])

expected_output = np.array([
    5 / (5 + 2),  # Precision for Class 0 = TP / (TP + FP) = 5 / 7
    3 / (3 + 4),  # Precision for Class 1 = 3 / 7
    4 / (4 + 1)   # Precision for Class 2 = 4 / 5
])

# Run function
precision_scores = precision(confusion_matrix)

# Print results
print("Precision Scores:", precision_scores)
print("Expected Output:", expected_output)

# Check correctness
assert np.allclose(precision_scores, expected_output), "Test failed!"

print("Test passed successfully!")
