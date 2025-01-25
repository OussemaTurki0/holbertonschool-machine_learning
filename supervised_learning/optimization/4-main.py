#!/usr/bin/env python3

moving_average = __import__('4-moving_average').moving_average

if __name__ == '__main__':
    # Example data and beta value
    data = [10, 20, 30, 40, 50]
    beta = 0.9

    print("Input data:")
    print(data)
    print(f"\nBeta value: {beta}")

    # Calculate the moving average
    moving_averages = moving_average(data, beta)

    print("\nCalculated moving averages:")
    print(moving_averages)

    # Manually verify first few values (for simple data and beta)
    # Expected values are derived based on the weighted moving average formula
    expected_moving_averages = [10.0, 19.047619047619047, 27.2562358276644, 34.73083119794356, 41.56984780569059]

    print("\nExpected moving averages:")
    print(expected_moving_averages)

    # Validate results
    for i in range(len(data)):
        assert abs(moving_averages[i] - expected_moving_averages[i]) < 1e-6, f"Mismatch at index {i}!"

    print("\nValidation passed: Calculated values match expected values.")
