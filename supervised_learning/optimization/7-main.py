#!/usr/bin/env python3

import numpy as np
update_variables_RMSProp = __import__('7-RMSProp').update_variables_RMSProp  # Replace 'your_script_name' with your actual file name

if __name__ == '__main__':
    # Example parameters
    alpha = 0.01  # Learning rate
    beta2 = 0.9  # RMSProp decay rate
    epsilon = 1e-8  # Small value to prevent division by zero
    var = np.array([1.0, 2.0, 3.0])  # Variable to update
    grad = np.array([0.1, 0.2, 0.3])  # Gradient of the variable
    s = np.array([0.0, 0.0, 0.0])  # Initial squared gradient

    print("Initial values:")
    print(f"var: {var}")
    print(f"grad: {grad}")
    print(f"s: {s}\n")

    # Perform one RMSProp update
    var, s = update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s)

    print("After one update:")
    print(f"var: {var}")
    print(f"s: {s}")

    # Manually verify the results for a simple test
    # Expected values for `s` and `var`:
    expected_s = beta2 * s + (1 - beta2) * np.square(grad)  # Updated s
    expected_var = var - alpha * grad / (np.sqrt(expected_s) + epsilon)  # Updated var

    print("\nExpected values:")
    print(f"Expected s: {expected_s}")
    print(f"Expected var: {expected_var}")

    # Validate the results
    assert np.allclose(s, expected_s, atol=1e-6), "Mismatch in updated squared gradient (s)!"
    assert np.allclose(var, expected_var, atol=1e-6), "Mismatch in updated variable (var)!"

    print("\nValidation passed: Calculated values match expected values.")
