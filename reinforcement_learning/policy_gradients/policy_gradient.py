#!/usr/bin/env python3
"""
Module for computing action probabilities and gradients using policy gradients.
"""

import numpy as np


def policy(state_matrix, weight_matrix):
    """
    Applies a softmax policy over the dot product of state and weight.
    """
    logits = np.dot(state_matrix, weight_matrix)
    shifted = np.exp(logits - np.max(logits))  # Stabilized softmax
    return shifted / np.sum(shifted, axis=1, keepdims=True)


def policy_gradient(observation, weights):
    """
    Calculates action and its gradient based on a single state.
    """
    observation = observation.reshape(1, -1)  # Ensure 2D input
    probabilities = policy(observation, weights).flatten()

    selected_action = np.random.choice(len(probabilities), p=probabilities)

    # Compute gradient: subtract 1 from selected action's probability
    dlog = probabilities.copy()
    dlog[selected_action] -= 1

    grad = np.outer(observation, -dlog)

    return selected_action, grad
