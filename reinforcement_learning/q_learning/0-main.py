#!/usr/bin/env python3

from importlib import import_module

# Import the environment loader from '0-load_env'
env_loader = import_module('0-load_env').load_frozen_lake

# Load the default FrozenLake environment
environment = env_loader()
print(environment.unwrapped.desc)
print(len(environment.unwrapped.P[0][0]))
print(environment.unwrapped.P[0][0])

# Load the FrozenLake environment with slippery surface enabled
environment_slippery = env_loader(is_slippery=True)
print(environment_slippery.unwrapped.desc)
print(len(environment_slippery.unwrapped.P[0][0]))
print(environment_slippery.unwrapped.P[0][0])

# Define a custom lake layout
custom_map = [
    ['S', 'F', 'F'],
    ['F', 'H', 'H'],
    ['F', 'F', 'G']
]

# Load environment with the custom map
environment_custom = env_loader(desc=custom_map)
print(environment_custom.unwrapped.desc)

# Load the 4x4 map variant of FrozenLake
environment_4x4 = env_loader(map_name='4x4')
print(environment_4x4.unwrapped.desc)
