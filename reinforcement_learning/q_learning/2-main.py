#!/usr/bin/env python3

import numpy as np
import importlib

# Load required modules dynamically
load_env = importlib.import_module('0-load_env').load_frozen_lake
initialize_Q = importlib.import_module('1-q_init').q_init
select_action = importlib.import_module('2-epsilon_greedy').epsilon_greedy

# Custom lake description
custom_map = [
    ['S', 'F', 'F'],
    ['F', 'H', 'H'],
    ['F', 'F', 'G']
]

# Initialize environment and Q-table
environment = load_env(desc=custom_map)
Q_table = initialize_Q(environment)

# Manually set Q-values for state 7
Q_table[7] = np.array([0.5, 0.7, 1.0, -1.0])

# Set random seed and pick actions using epsilon-greedy policy
np.random.seed(0)
print(select_action(Q_table, 7, 0.5))

np.random.seed(1)
print(select_action(Q_table, 7, 0.5))
