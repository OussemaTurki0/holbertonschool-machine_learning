#!/usr/bin/env python3

# Importing the custom BayesianOptimization class from another file
BO = __import__('5-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def black_box_function(x):
    """Mystery function we're trying to optimize"""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)


if __name__ == '__main__':
    np.random.seed(42)

    # Generate 2 initial random input values
    initial_X = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    initial_Y = black_box_function(initial_X)

    # Create Bayesian Optimizer instance
    optimizer = BO(
        f=black_box_function,
        X_init=initial_X,
        Y_init=initial_Y,
        bounds=(-np.pi, 2 * np.pi),
        ac_samples=50,
        l=0.6,
        sigma_f=2
    )

    # Perform optimization for 50 iterations
    best_X, best_Y = optimizer.optimize(iterations=50)

    # Print results
    print("Best input found (X):", best_X)
    print("Function value at best input (Y):", best_Y)
    print("All points evaluated during optimization:\n", optimizer.gp.X)
