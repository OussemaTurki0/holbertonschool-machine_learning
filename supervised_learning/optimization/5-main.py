#!/usr/bin/env python3

import numpy as np
update_variables_momentum = __import__('5-momentum').update_variables_momentum

if __name__ == '__main__':
    # Example values for the parameters
    alpha = 0.1  # Learning rate
    beta1 = 0.9  # Momentum hyperparameter
    var = np.array([1.0, 2.0, 3.0])  # Variable to update
    grad = np.array([0.1, 0.2, 0.3])  # Gradient
    v = np.array([0.0, 0.0, 0.0])  # Initial velocity

    print("Initial values:")
    print(f"var: {var}")
    print(f"grad: {grad}")
    print(f"v: {v}\n")

    # Perform one update
    var, v = update_variables_momentum(alpha, beta1, var, grad, v)

    print("After one update:")
    print(f"var: {var}")
    print(f"v: {v}")

    # Manually verify the results for a simple test
    # Expected v: [0.01, 0.02, 0.03] (momentum calculation)
    # Expected var: [0.999, 1.998, 2.997] (update step)
    expected_v = np.array([0.01, 0.02, 0.03])
    expected_var = np.array([0.999, 1.998, 2.997])

    print("\nExpected values:")
    print(f"Expected v: {expected_v}")
    print(f"Expected var: {expected_var}")

    # Validate the results
    assert np.allclose(v, expected_v, atol=1e-6), "Mismatch in velocity!"
    assert np.allclose(var, expected_var, atol=1e-6), "Mismatch in variable update!"

    print("\nValidation passed: Calculated values match expected values.")
