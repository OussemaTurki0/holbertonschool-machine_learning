#!/usr/bin/env python3
"""
Module for training an agent using REINFORCE with policy gradients.
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Trains an agent over multiple episodes using the REINFORCE algorithm.
    """
    episode_scores = []
    theta = np.random.rand(4, 2)  # Random initialization of policy weights

    for ep in range(nb_episodes):
        obs, _ = env.reset()
        finished = False
        rewards = []
        grad_logps = []

        while not finished:
            if show_result and ep % 1000 == 0:
                env.render()

            act, grad = policy_gradient(obs, theta)
            obs_next, reward, finished, _, _ = env.step(int(act))

            rewards.append(reward)
            grad_logps.append(grad)
            obs = obs_next

        total = sum(rewards)
        episode_scores.append(total)
        print(f"Episode: {ep} Score: {total}")

        for i in range(len(rewards)):
            Gt = sum([gamma ** k * rewards[k + i] for k in range(len(rewards) - i)])
            theta += alpha * Gt * grad_logps[i]

    return episode_scores
