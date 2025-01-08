#!/usr/bin/env python3
"""
Test file for Neuron class in 1-neuron.py
"""

import numpy as np
Neuron = __import__('1-neuron').Neuron

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
        nx = 10
        neuron = Neuron(nx)
        print("Weights (W):", neuron.W)
        print("Bias (b):", neuron.b)
        print("Activation output (A):", neuron.A)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
