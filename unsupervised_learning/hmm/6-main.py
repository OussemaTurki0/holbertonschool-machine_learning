#!/usr/bin/env python3
"""
Test for baum_welch function
"""

import numpy as np
baum_welch = __import__('6-baum_welch').baum_welch

# Sample HMM parameters
np.random.seed(0)

Observation = np.array([0, 1, 2, 1, 0])  # Observation sequence (indices)

# Hidden states: 2, Observation symbols: 3
Transition = np.array([[0.6, 0.4],
                       [0.3, 0.7]])

Emission = np.array([[0.5, 0.4, 0.1],
                     [0.1, 0.3, 0.6]])

Initial = np.array([[0.6],
                    [0.4]])

# Run the Baum-Welch algorithm
Transition_updated, Emission_updated = baum_welch(
    Observation, Transition, Emission, Initial, iterations=10)

# Print updated parameters
print("Updated Transition matrix:")
print(Transition_updated)

print("\nUpdated Emission matrix:")
print(Emission_updated)
