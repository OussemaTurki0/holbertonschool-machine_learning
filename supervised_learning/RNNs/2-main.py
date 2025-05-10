#!/usr/bin/env python3

import numpy as np
GRUCell = __import__('2-gru_cell').GRUCell

# Set seed for consistent random values
np.random.seed(2)

# Instantiate a GRU cell with input dim=10, hidden dim=15, output dim=5
cell = GRUCell(i=10, h=15, o=5)

# Display initial weights and biases
print("Update gate weights (Wz):", cell.Wz)
print("Reset gate weights (Wr):", cell.Wr)
print("Candidate hidden weights (Wh):", cell.Wh)
print("Output weights (Wy):", cell.Wy)
print("Update gate bias (bz):", cell.bz)
print("Reset gate bias (br):", cell.br)
print("Candidate hidden bias (bh):", cell.bh)
print("Output bias (by):", cell.by)

# Randomize biases to simulate training updates
cell.bz = np.random.randn(1, 15)
cell.br = np.random.randn(1, 15)
cell.bh = np.random.randn(1, 15)
cell.by = np.random.randn(1, 5)

# Prepare dummy input and previous hidden state
input_example = np.random.randn(8, 10)
prev_state = np.random.randn(8, 15)

# Perform one forward pass
next_state, output = cell.forward(prev_state, input_example)

# Output the results
print("Next hidden state shape:", next_state.shape)
print("Next hidden state:\n", next_state)
print("Output shape:", output.shape)
print("Output:\n", output)
