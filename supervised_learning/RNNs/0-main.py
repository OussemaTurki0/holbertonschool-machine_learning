#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell

# Set seed for reproducibility
np.random.seed(0)

# Create an RNN cell with input size 10, hidden size 15, and output size 5
cell = RNNCell(i=10, h=15, o=5)

# Display initial weights and biases
print("Initial hidden weights (Wh):", cell.Wh)
print("Initial output weights (Wy):", cell.Wy)
print("Initial hidden bias (bh):", cell.bh)
print("Initial output bias (by):", cell.by)

# Manually update biases with new random values
cell.bh = np.random.randn(1, 15)
cell.by = np.random.randn(1, 5)

# Generate random input and previous hidden state
input_data = np.random.randn(8, 10)
prev_hidden = np.random.randn(8, 15)

# Run one forward step
next_hidden, output = cell.forward(prev_hidden, input_data)

# Print the results
print("Next hidden state shape:", next_hidden.shape)
print("Next hidden state:\n", next_hidden)

print("Output shape:", output.shape)
print("Output:\n", output)
