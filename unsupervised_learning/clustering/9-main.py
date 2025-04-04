#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

compute_bic = __import__('9-BIC').BIC

def generate_data(seed=11):
    np.random.seed(seed)
    sample1 = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], 10000)
    sample2 = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], 750)
    sample3 = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], 750)
    sample4 = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], 1000)
    all_data = np.vstack((sample1, sample2, sample3, sample4))
    np.random.shuffle(all_data)
    return all_data

def plot_results(log_likelihoods, bic_scores):
    clusters_range = np.arange(1, len(log_likelihoods) + 1)
    
    plt.plot(clusters_range, log_likelihoods, 'r')
    plt.title("Log Likelihood vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Log Likelihood")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(clusters_range, bic_scores, 'b')
    plt.title("BIC Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data = generate_data()
    optimal_k, model_info, log_likelihoods, bic_scores = compute_bic(data, kmin=1, kmax=10)
    
    print(optimal_k)
    print(model_info)
    print(log_likelihoods)
    print(bic_scores)

    plot_results(log_likelihoods, bic_scores)
