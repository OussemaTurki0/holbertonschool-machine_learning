#!/usr/bin/env python3
"""
TD(λ) algorithm
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for estimating the value function.

    Parameters:
    - env: the environment instance
    - V: a numpy.ndarray of shape (s,) containing the value estimate
    - policy: a function that takes in a state and returns the next action
    - lambtha: the eligibility trace decay rate (λ)
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount factor

    Returns:
    - V: the updated value estimate
    """
    for episode in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state
            if terminated or truncated:
                break

    return V
