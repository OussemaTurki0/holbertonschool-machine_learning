#!/usr/bin/env python3

# Import custom Gaussian Process and Bayesian Optimization modules
GP = __import__('2-gp').GaussianProcess
BO = __import__('3-bayes_opt').BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt


def black_box_func(x):
    """Simulates the unknown function we're trying to optimize."""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)


if __name__ == '__main__':
    np.random.seed(42)

    # Generate initial sample inputs and corresponding outputs
    initial_X = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    initial_Y = black_box_func(initial_X)

    # Define bounds and create Bayesian Optimization instance
    bounds = (-np.pi, 2 * np.pi)
    optimizer = BO(
        f=black_box_func,
        X_init=initial_X,
        Y_init=initial_Y,
        bounds=bounds,
        ac_samples=50,
        l=2,
        sigma_f=3,
        xsi=0.05
    )

    # Verification prints
    print("Function match:", optimizer.f is black_box_func)
    print("GP instance:", isinstance(optimizer.gp, GP))
    print("Initial inputs match:", optimizer.gp.X is initial_X)
    print("Initial outputs match:", optimizer.gp.Y is initial_Y)
    print("Length-scale (l):", optimizer.gp.l)
    print("Signal variance (sigma_f):", optimizer.gp.sigma_f)
    print("Sample grid shape:", optimizer.X_s.shape)
    print("Exploration factor (xsi):", optimizer.xsi)
    print("Optimization mode (minimize):", optimizer.minimize)
