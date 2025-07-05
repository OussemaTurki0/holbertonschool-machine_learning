#!/usr/bin/env python3

import importlib

# Dynamically import the necessary functions
lake_loader = importlib.import_module('0-load_env').load_frozen_lake
initialize_q_table = importlib.import_module('1-q_init').q_init

# Create the FrozenLake environment (default)
frozen_env = lake_loader()
Q_table = initialize_q_table(frozen_env)
print(Q_table.shape)

# Create environment with slippery surface enabled
frozen_env_slippery = lake_loader(is_slippery=True)
Q_table_slippery = initialize_q_table(frozen_env_slippery)
print(Q_table_slippery.shape)

# Custom lake configuration
custom_lake_layout = [
    ['S', 'F', 'F'],
    ['F', 'H', 'H'],
    ['F', 'F', 'G']
]

# Load environment using the custom map
frozen_env_custom = lake_loader(desc=custom_lake_layout)
Q_table_custom = initialize_q_table(frozen_env_custom)
print(Q_table_custom.shape)

# Load the 4x4 map version of the environment
frozen_env_4x4 = lake_loader(map_name='4x4')
Q_table_4x4 = initialize_q_table(frozen_env_4x4)
print(Q_table_4x4.shape)
