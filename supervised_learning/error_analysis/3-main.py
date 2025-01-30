#!/usr/bin/env python3
"""
Test file for specificity function
"""

import numpy as np
specificity = __import__('3-specificity').specificity

# Test case: 3-class confusion matrix
confusion_matrix = np.array([
    [50, 10, 5],  # Class 0
    [8, 30, 7],   # Class 1
    [3, 6, 40]    # Class 2
])

# Manually computing expected specificity values
total = np.sum(confusion_matrix)

true_positives = np.diag(confusion_matrix)
false_positives = np.sum(confusion_matrix, axis=0) - true_positives
false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
true_negatives = total - (true_positives + false_positives + false_negatives)

expected_output = true_negatives / (true_negatives + false_positives)

# Run function
specificity_scores = specificity(confusion_matrix)

# Print results
print("Specificity Scores:", specificity_scores)
print("Expected Output:", expected_output)

# Check correctness
assert np.allclose(specificity_scores, expected_output), "Test failed!"

print("Test passed successfully!")
