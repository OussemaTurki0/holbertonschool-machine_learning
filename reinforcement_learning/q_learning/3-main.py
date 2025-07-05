#!/usr/bin/env python3

import numpy as np
import importlib

# Import necessary functions
load_environment = importlib.import_module('0-load_env').load_frozen_lake
initialize_q = importlib.import_module('1-q_init').q_init
q_learning_train = importlib.import_module('3-q_learning').train

# Fix random seed for reproducibility
np.random.seed(0)

# Custom FrozenLake map layout
lake_map = [
    ['S', 'F', 'F'],
    ['F', 'H', 'H'],
    ['F', 'F', 'G']
]

# Create environment and initialize Q-table
environment = load_environment(desc=lake_map)
Q_table = initialize_q(environment)

# Train Q-learning agent
Q_table, rewards_log = q_learning_train(environment, Q_table)

# Print final Q-table
print(Q_table)

# Divide total rewards into 10 equal chunks and print average rewards per chunk
chunks = np.array_split(np.array(rewards_log), 10)
for idx, chunk in enumerate(chunks, start=1):
    print(f"Episode {(idx * 500)} : {np.mean(chunk)}")
