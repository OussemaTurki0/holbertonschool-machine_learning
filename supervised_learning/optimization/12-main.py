#!/usr/bin/env python3
"""
Test for the learning_rate_decay function.
"""

learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay

if __name__ == '__main__':
    # Define test parameters
    alpha = 0.1           # Initial learning rate
    decay_rate = 0.05     # Decay rate
    global_step = 100     # Global step (current step in training)
    decay_step = 20       # Step interval after which decay is applied

    print("Testing learning rate decay with the following parameters:")
    print(f"Initial alpha: {alpha}")
    print(f"Decay rate: {decay_rate}")
    print(f"Global step: {global_step}")
    print(f"Decay step: {decay_step}\n")

    # Call the learning_rate_decay function
    new_alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_step)

    # Output the updated learning rate
    print(f"Updated learning rate: {new_alpha}")
