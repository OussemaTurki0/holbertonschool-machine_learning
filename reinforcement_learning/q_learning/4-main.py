#!/usr/bin/env python3

import numpy as np
import importlib

# Dynamically import required functions
load_env_fn = importlib.import_module('0-load_env').load_frozen_lake
init_q_fn = importlib.import_module('1-q_init').q_init
train_fn = importlib.import_module('3-q_learning').train
play_fn = importlib.import_module('4-play').play

# Set fixed random seed for consistent results
np.random.seed(0)

# Define a custom map layout
custom_map = [
    ['S', 'F', 'F'],
    ['F', 'H', 'H'],
    ['F', 'F', 'G']
]

# Initialize environment and Q-table
environment = load_env_fn(desc=custom_map)
Q_table = init_q_fn(environment)

# Train the agent on the environment
Q_table, _ = train_fn(environment, Q_table)

# Reset environment before playing
environment.reset()

# Run the trained policy and collect results
total_reward, frames = play_fn(environment, Q_table)

# Output total reward and display frames
print(f'Total Rewards: {total_reward}')
for frame in frames:
    print(frame)
