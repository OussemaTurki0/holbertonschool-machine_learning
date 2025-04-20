#!/usr/bin/env python3

# Importing the Bayesian Optimization class
BO = __import__('4-bayes_opt').BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt


def black_box_func(x):
    """The function we're aiming to optimize."""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)


if __name__ == '__main__':
    np.random.seed(42)

    # Initialize two random input samples
    init_X = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    init_Y = black_box_func(init_X)

    # Create Bayesian Optimization instance with defined parameters
    optimizer = BO(
        f=black_box_func,
        X_init=init_X,
        Y_init=init_Y,
        bounds=(-np.pi, 2 * np.pi),
        ac_samples=50,
        l=0.6,
        sigma_f=2,
        xsi=0.05
    )

    # Calculate Expected Improvement and get the next candidate point
    next_X, expected_improvement = optimizer.acquisition()

    print("Expected Improvement values:\n", expected_improvement)
    print("Suggested next input to evaluate:", next_X)

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.scatter(init_X.ravel(), init_Y.ravel(), color='green', label='Initial Samples')
    plt.plot(optimizer.X_s.ravel(), expected_improvement.ravel(), color='red', label='Expected Improvement')
    plt.axvline(x=next_X, linestyle='--', color='blue', label='Next Suggested Point')
    plt.title("Bayesian Optimization - Acquisition Function")
    plt.xlabel("Input values")
    plt.ylabel("Expected Improvement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
