#!/usr/bin/env python3

import numpy as np
init_params = __import__('4-initialize').initialize
compute_expectation = __import__('6-expectation').expectation

def generate_data():
    np.random.seed(11)
    cluster1 = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], 10000)
    cluster2 = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], 750)
    cluster3 = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], 750)
    cluster4 = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], 1000)
    full_dataset = np.vstack((cluster1, cluster2, cluster3, cluster4))
    np.random.shuffle(full_dataset)
    return full_dataset

def main():
    data = generate_data()
    priors, means, covariances = init_params(data, 4)
    responsibilities, log_likelihood = compute_expectation(data, priors, means, covariances)

    print(responsibilities)
    print(np.sum(responsibilities, axis=0))
    print(log_likelihood)

if __name__ == '__main__':
    main()
