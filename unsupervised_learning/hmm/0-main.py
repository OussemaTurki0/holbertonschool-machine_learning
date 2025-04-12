#!/usr/bin/env python3
"""
Test file for markov_chain function.
"""

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

# Define a transition matrix (P)
P = np.array([[0.25, 0.25, 0.25, 0.25],
              [0.1, 0.3, 0.4, 0.2],
              [0.2, 0.2, 0.5, 0.1],
              [0.3, 0.3, 0.2, 0.2]])

# Define an initial state vector (s)
s = np.array([[1.0, 0.0, 0.0, 0.0]])

# Number of transitions
t = 10

# Compute the state distribution after t transitions
result = markov_chain(P, s, t)

# Display result
print(f"State distribution after {t} transitions:")
print(result)
