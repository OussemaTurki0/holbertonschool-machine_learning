#!/usr/bin/env python3
"""
Test file for regular function.
"""

import numpy as np
regular = __import__('1-regular').regular

# Example of a regular transition matrix
P = np.array([[0.5, 0.5],
              [0.2, 0.8]])

# Call the function to compute the steady state
steady_state = regular(P)

# Display the result
if steady_state is not None:
    print("Steady state probabilities:")
    print(steady_state)
else:
    print("The matrix is not regular or something went wrong.")
