#!/usr/bin/env python3
"""
Test file for viterbi function in HMM.
"""

import numpy as np
viterbi = __import__('4-viterbi').viterbi

# Example HMM setup
Observation = np.array([0, 1, 2])  # Observations
Emission = np.array([[0.5, 0.4, 0.1],  # Emission matrix
                     [0.1, 0.3, 0.6]])
Transition = np.array([[0.7, 0.3],     # Transition matrix
                       [0.4, 0.6]])
Initial = np.array([[0.6],             # Initial state probabilities
                    [0.4]])

# Run Viterbi algorithm
path, P = viterbi(Observation, Emission, Transition, Initial)

# Output
print("Most likely hidden state sequence:", path)
print("Probability of the path:", P)
