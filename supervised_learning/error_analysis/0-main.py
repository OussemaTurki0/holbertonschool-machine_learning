#!/usr/bin/env python3
"""
Test file for create_confusion_matrix
"""

import numpy as np
create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

# Test case 1: Simple 3-class example
labels = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
])

logits = np.array([
    [1, 0, 0],  # Predicted Class 0
    [0, 1, 0],  # Predicted Class 1
    [0, 1, 0],  # Predicted Class 1 (Incorrect)
    [1, 0, 0],  # Predicted Class 0
    [0, 0, 1],  # Predicted Class 2 (Incorrect)
])

expected_output = np.array([
    [2, 0, 0],  # 2 correct for Class 0
    [0, 1, 1],  # 1 correct for Class 1, 1 misclassified as Class 2
    [0, 1, 0],  # 1 misclassified from Class 2 to Class 1
])

# Run function
conf_matrix = create_confusion_matrix(labels, logits)

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("Expected Output:\n", expected_output)

# Check correctness
assert np.array_equal(conf_matrix, expected_output), "Test failed!"

print("Test passed successfully!")
