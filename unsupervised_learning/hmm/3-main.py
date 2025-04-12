#!/usr/bin/env python3
"""
Test file for forward function in HMM.
"""

import numpy as np
forward = __import__('3-forward').forward

# Example data for testing
Observation = np.array([0, 1, 2])
Emission = np.array([[0.5, 0.4, 0.1],
                     [0.1, 0.3, 0.6]])
Transition = np.array([[0.7, 0.3],
                       [0.4, 0.6]])
Initial = np.array([[0.6],
                    [0.4]])

# Run forward algorithm
P, F = forward(Observation, Emission, Transition, Initial)

print("Probability of the observation sequence:", P)
print("Forward path probability matrix:\n", F)
