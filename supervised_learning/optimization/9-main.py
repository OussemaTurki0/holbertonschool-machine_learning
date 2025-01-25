#!/usr/bin/env python3

import numpy as np
update_variables_Adam = __import__('9-Adam').update_variables_Adam

if __name__ == '__main__':
    # Define example parameters
    alpha = 0.01  # Learning rate
    beta1 = 0.9   # Adam's first moment decay rate
    beta2 = 0.999  # Adam's second moment decay rate
    epsilon = 1e-8  # Small number to avoid division by zero
    var = np.array([1.0, 2.0, 3.0])  # Variable to update
    grad = np.array([0.1, 0.2, 0.3])  # Gradient of the variable
    v = np.array([0.0, 0.0, 0.0])  # Initial first moment (velocity)
    s = np.array([0.0, 0.0, 0.0])  # Initial second moment (squared gradients)
    t = 1  # Time step (typically starts at 1)

    print("Initial values:")
    print(f"var: {var}")
    print(f"grad: {grad}")
    print(f"v: {v}")
    print(f"s: {s}")
    print(f"t: {t}\n")

    # Perform one Adam update
    var, v, s = update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t)

    print("After one update:")
    print(f"var: {var}")
    print(f"v: {v}")
    print(f"s: {s}")

    # Manually verify the results for a simple test
    # Expected values for `v`, `s`, `v_corrected`, `s_corrected`, and `var`:
    expected_v = beta1 * v + (1 - beta1) * grad  # Updated v
    expected_s = beta2 * s + (1 - beta2) * np.square(grad)  # Updated s
    expected_v_corrected = expected_v / (1 - beta1 ** t)  # Bias-corrected v
    expected_s_corrected = expected_s / (1 - beta2 ** t)  # Bias-corrected s
    expected_var = var - alpha * expected_v_corrected / (np.sqrt(expected_s_corrected) + epsilon)  # Updated var

    print("\nExpected values:")
    print(f"Expected v: {expected_v}")
    print(f"Expected s: {expected_s}")
    print(f"Expected v_corrected: {expected_v_corrected}")
    print(f"Expected s_corrected: {expected_s_corrected}")
    print(f"Expected var: {expected_var}")

    # Validate the results
    assert np.allclose(v, expected_v, atol=1e-6), "Mismatch in updated first moment (v)!"
    assert np.allclose(s, expected_s, atol=1e-6), "Mismatch in updated second moment (s)!"
    assert np.allclose(var, expected_var, atol=1e-6), "Mismatch in updated variable (var)!"

    print("\nValidation passed: Calculated values match expected values.")
