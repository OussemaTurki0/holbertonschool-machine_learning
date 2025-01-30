#!/usr/bin/env python3
"""
Test file for sensitivity function
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

# Test case 1: Simple 3-class confusion matrix
confusion_matrix = np.array([
    [5, 1, 0],  # Class 0: 5 TP, 1 FN
    [2, 3, 1],  # Class 1: 3 TP, 2+1 FN
    [0, 2, 4]   # Class 2: 4 TP, 2 FN
])

expected_output = np.array([
    5 / (5 + 1),  # Sensitivity for Class 0 = TP / (TP + FN) = 5 / 6
    3 / (3 + 3),  # Sensitivity for Class 1 = 3 / 6
    4 / (4 + 2)   # Sensitivity for Class 2 = 4 / 6
])

# Run function
sensitivity_scores = sensitivity(confusion_matrix)

# Print results
print("Sensitivity Scores:", sensitivity_scores)
print("Expected Output:", expected_output)

# Check correctness
assert np.allclose(sensitivity_scores, expected_output), "Test failed!"

print("Test passed successfully!")
