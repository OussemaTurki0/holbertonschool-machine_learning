#!/usr/bin/env python3
"""
Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for estimating the value function.
    """
    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        visited = set()

        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            if state not in visited:
                V[state] = V[state] + alpha * (G - V[state])
                visited.add(state)

    return V
