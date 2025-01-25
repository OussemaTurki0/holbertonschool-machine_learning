#!/usr/bin/env python3
"""
Test file for the learning_rate_decay function.
"""

import numpy as np
learning_rate_decay = __import__('11-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    # Define example parameters
    alpha = 0.1           # Initial learning rate
    decay_rate = 0.05     # Decay rate
    global_step = 50      # Current step in training
    decay_step = 10       # Step interval after which to apply decay

    print("Setting up learning rate decay with the following parameters:")
    print(f"Initial alpha: {alpha}")
    print(f"Decay rate: {decay_rate}")
    print(f"Global step: {global_step}")
    print(f"Decay step: {decay_step}\n")

    # Apply learning rate decay
    new_alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_step)

    # Print the updated learning rate
    print(f"Updated learning rate after decay: {new_alpha}")
