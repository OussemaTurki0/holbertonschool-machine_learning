#!/usr/bin/env python3
"""
Train a DQN agent on the Breakout environment using keras-rl2 and gymnasium.
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def build_model(input_shape, nb_actions):
    """
    Build the DQN model using CNN layers appropriate for Atari frames.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))  # From (channels, height, width) to (height, width, channels)
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def main():
    # Create and wrap the Breakout environment
    env = gym.make('BreakoutNoFrameskip-v4', render_mode=None)  # Important: NoFrameskip
    env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True)  # Skip frames here, not in base env
    env = FrameStack(env, 4)

    # Get action space and input shape
    nb_actions = env.action_space.n
    input_shape = (4, 84, 84)  # (frames, height, width)

    # Build and show model
    model = build_model(input_shape, nb_actions)
    print(model.summary())

    # Configure memory and exploration policy
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    # Initialize and compile the DQN agent
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=10000,
                   policy=policy,
                   gamma=0.99,
                   train_interval=4,
                   delta_clip=1.0)

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train the agent
    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)

    # Save the final policy weights
    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    main()
