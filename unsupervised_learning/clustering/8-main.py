#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

run_em = __import__('8-EM').expectation_maximization

def generate_dataset():
    np.random.seed(11)
    cluster_a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], 10000)
    cluster_b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], 750)
    cluster_c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], 750)
    cluster_d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], 1000)
    data = np.vstack((cluster_a, cluster_b, cluster_c, cluster_d))
    np.random.shuffle(data)
    return data

def visualize_results(points, means, responsibilities, components, log_likelihood):
    labels = np.sum(responsibilities * np.arange(components).reshape(components, 1), axis=0)
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=20)
    plt.scatter(means[:, 0], means[:, 1], c=np.arange(components), s=60, marker='*')
    plt.title("EM Clustering Result")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()
    print(points.shape[0] * weights)
    print(means)
    print(covariances)
    print(log_likelihood)

if __name__ == '__main__':
    dataset = generate_dataset()
    num_clusters = 4
    weights, means, covariances, resp, log_likelihood = run_em(dataset, num_clusters, iterations=150, verbose=True)
    visualize_results(dataset, means, resp, num_clusters, log_likelihood)
