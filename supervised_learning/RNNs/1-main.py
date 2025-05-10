#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
rnn_forward = __import__('1-rnn').rnn

# For reproducibility
np.random.seed(1)

# Initialize RNN cell with input dim = 10, hidden dim = 15, output dim = 5
cell = RNNCell(i=10, h=15, o=5)

# Set random biases
cell.bh = np.random.randn(1, 15)
cell.by = np.random.randn(1, 5)

# Simulate input sequence: 6 time steps, batch size 8, input dim 10
sequence = np.random.randn(6, 8, 10)

# Initial hidden state: batch size 8, hidden dim 15
initial_state = np.zeros((8, 15))

# Perform forward pass over time steps
hidden_states, outputs = rnn_forward(cell, sequence, initial_state)

# Print shapes and content
print("Shape of hidden states:", hidden_states.shape)
print("Hidden states:\n", hidden_states)

print("Shape of outputs:", outputs.shape)
print("Outputs:\n", outputs)
