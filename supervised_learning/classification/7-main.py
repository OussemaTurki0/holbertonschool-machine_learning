#!/usr/bin/env python3
"""
Test file for Neuron class defined in 7-neuron.py.
"""

import numpy as np
Neuron = __import__('7-neuron').Neuron

def test_neuron():
    # Test case 1: Initializing with valid number of input features (nx)
    try:
        neuron = Neuron(3)  # 3 input features
        print("Test 1 Passed: Neuron initialized successfully with 3 input features")
    except Exception as e:
        print(f"Test 1 Failed: {e}")

    # Test case 2: Attempt to initialize with invalid nx (non-integer)
    try:
        neuron = Neuron("3")  # Invalid nx type (string instead of integer)
        print("Test 2 Failed: Should have raised TypeError for non-integer nx")
    except TypeError:
        print("Test 2 Passed: Correctly raised TypeError for non-integer nx")
    
    # Test case 3: Attempt to initialize with nx < 1
    try:
        neuron = Neuron(0)  # Invalid nx value (should be a positive integer)
        print("Test 3 Failed: Should have raised ValueError for nx < 1")
    except ValueError:
        print("Test 3 Passed: Correctly raised ValueError for nx < 1")

    # Test case 4: Forward propagation
    try:
        X = np.array([[1, 2, 3], [4, 5, 6]])  # 2 features, 3 examples
        Y = np.array([[1, 0, 1]])  # 1 output, 3 examples
        neuron = Neuron(2)
        A = neuron.forward_prop(X)  # Forward propagation
        print("Test 4 Passed: Forward propagation successful")
    except Exception as e:
        print(f"Test 4 Failed: {e}")

    # Test case 5: Cost function
    try:
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 0, 1]])
        neuron = Neuron(2)
        A = neuron.forward_prop(X)
        cost = neuron.cost(Y, A)  # Calculate cost
        print("Test 5 Passed: Cost function computed successfully")
    except Exception as e:
        print(f"Test 5 Failed: {e}")

    # Test case 6: Train the neuron
    try:
        X = np.array([[1, 2, 3], [4, 5, 6]])  # 2 features, 3 examples
        Y = np.array([[1, 0, 1]])  # 1 output, 3 examples
        neuron = Neuron(2)
        prediction, cost = neuron.train(X, Y, iterations=1000, alpha=0.05, verbose=False, graph=False)
        print("Test 6 Passed: Neuron trained successfully")
    except Exception as e:
        print(f"Test 6 Failed: {e}")

if __name__ == "__main__":
    test_neuron()
