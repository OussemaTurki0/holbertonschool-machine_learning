#!/usr/bin/env python3

import numpy as np
LSTMCell = __import__('3-lstm_cell').LSTMCell

# Set a random seed for reproducibility
np.random.seed(3)

# Initialize LSTMCell with input size 10, hidden size 15, output size 5
cell = LSTMCell(i=10, h=15, o=5)

# Show initial weights and biases
print("Forget gate weights (Wf):", cell.Wf)
print("Update gate weights (Wu):", cell.Wu)
print("Candidate cell weights (Wc):", cell.Wc)
print("Output gate weights (Wo):", cell.Wo)
print("Output layer weights (Wy):", cell.Wy)
print("Forget gate bias (bf):", cell.bf)
print("Update gate bias (bu):", cell.bu)
print("Candidate cell bias (bc):", cell.bc)
print("Output gate bias (bo):", cell.bo)
print("Output bias (by):", cell.by)

# Simulate learned biases
cell.bf = np.random.randn(1, 15)
cell.bu = np.random.randn(1, 15)
cell.bc = np.random.randn(1, 15)
cell.bo = np.random.randn(1, 15)
cell.by = np.random.randn(1, 5)

# Dummy inputs and previous states
input_t = np.random.randn(8, 10)
prev_hidden = np.random.randn(8, 15)
prev_cell = np.random.randn(8, 15)

# Perform one forward step
next_hidden, next_cell, output = cell.forward(prev_hidden, prev_cell, input_t)

# Print shapes and results
print("Hidden state shape:", next_hidden.shape)
print("Hidden state:\n", next_hidden)

print("Cell state shape:", next_cell.shape)
print("Cell state:\n", next_cell)

print("Output shape:", output.shape)
print("Output:\n", output)
