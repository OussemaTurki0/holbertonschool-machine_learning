#!/usr/bin/env python3

import numpy as np
init = __import__('4-initialize').initialize
e_step = __import__('6-expectation').expectation
m_step = __import__('7-maximization').maximization

def create_samples():
    np.random.seed(11)
    group1 = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], 10000)
    group2 = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], 750)
    group3 = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], 750)
    group4 = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], 1000)
    dataset = np.vstack((group1, group2, group3, group4))
    np.random.shuffle(dataset)
    return dataset

def main():
    data = create_samples()
    weights, means, covs = init(data, 4)
    responsibilities, _ = e_step(data, weights, means, covs)
    new_weights, new_means, new_covs = m_step(data, responsibilities)

    print(new_weights)
    print(new_means)
    print(new_covs)

if __name__ == '__main__':
    main()
