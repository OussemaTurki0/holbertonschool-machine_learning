#!/usr/bin/env python3
"""
Test file for Neuron class in 2-neuron.py
"""

import numpy as np
Neuron = __import__('2-neuron').Neuron

def main():
    try:
        print("Test 1: Invalid nx (not an integer)")
        neuron = Neuron("10")  # Should raise a TypeError
    except Exception as e:
        print(e)

    try:
        print("\nTest 2: Invalid nx (negative integer)")
        neuron = Neuron(-10)  # Should raise a ValueError
    except Exception as e:
        print(e)

    try:
        print("\nTest 3: Invalid nx (zero)")
        neuron = Neuron(0)  # Should raise a ValueError
    except Exception as e:
        print(e)

    try:
        print("\nTest 4: Valid nx (positive integer)")
        nx = 3  # Let's test with 3 features
        neuron = Neuron(nx)
        print("Weights (W):", neuron.W)
        print("Bias (b):", neuron.b)
        print("Activation output (A):", neuron.A)
    except Exception as e:
        print(e)

    try:
        print("\nTest 5: Forward Propagation")
        # Generate random input data for 3 features and 5 examples (3x5)
        X = np.random.randn(nx, 5)
        A = neuron.forward_prop(X)
        print("Output of forward propagation (A):", A)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
