#!/usr/bin/env python3
"""
Load a trained DQN agent and play Breakout with rendering.
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Permute
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

def main():
    env = gym.make('Breakout-v5', render_mode='human')
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)

    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape

    model = build_model(input_shape, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
                   target_model_update=10000, policy=policy, gamma=0.99, train_interval=4, delta_clip=1.0)
    dqn.compile(tf.keras.optimizers.Adam(learning_rate=0.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')

    # Play one episode
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = dqn.forward(obs[0])
        obs, reward, done, truncated, info = env.step(action)
    env.close()

if __name__ == '__main__':
    main()
