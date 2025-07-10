#!/usr/bin/env python3
"""
SARSA(λ) algorithm with eligibility traces
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects an action using the epsilon-greedy policy.

    Parameters:
    - Q: numpy.ndarray containing the Q-table
    - state: current state
    - epsilon: exploration rate

    Returns:
    - action: the selected action
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    else:
        return np.random.randint(Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm with eligibility traces to update Q-values.

    Parameters:
    - env: the OpenAI Gym environment
    - Q: np.ndarray (s, a), initial Q-table
    - lambtha: eligibility trace decay rate (λ)
    - episodes: total number of episodes to train
    - max_steps: max steps per episode
    - alpha: learning rate
    - gamma: discount rate
    - epsilon: initial exploration rate
    - min_epsilon: minimum epsilon value
    - epsilon_decay: decay rate for epsilon

    Returns:
    - Q: the updated Q-table
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            new_state, reward, done, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)

            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
            eligibility[state, action] += 1

            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = new_state
            action = new_action

            if done or truncated:
                break

        # Decay epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q
