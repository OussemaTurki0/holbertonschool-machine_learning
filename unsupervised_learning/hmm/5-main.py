#!/usr/bin/env python3
"""
Test file for the backward algorithm function in HMM.
"""

import numpy as np
backward = __import__('5-backward').backward

# Define an example HMM
Observation = np.array([0, 1, 2])  # Observation sequence
Emission = np.array([[0.5, 0.4, 0.1],  # Emission matrix
                     [0.1, 0.3, 0.6]])
Transition = np.array([[0.7, 0.3],     # Transition matrix
                       [0.4, 0.6]])
Initial = np.array([[0.6],             # Initial probabilities
                    [0.4]])

# Run backward algorithm
P, B = backward(Observation, Emission, Transition, Initial)

# Print results
print("Probability of the observation sequence:", P)
print("Backward path probability matrix:\n", B)
