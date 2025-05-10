#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('5-bi_forward').BidirectionalCell

# Set seed for consistent results
np.random.seed(5)

# Initialize the bidirectional RNN cell
bi_rnn = BidirectionalCell(i=10, h=15, o=5)

# Display initialized parameters
print("Forward hidden weights (Whf):", bi_rnn.Whf)
print("Backward hidden weights (Whb):", bi_rnn.Whb)
print("Output weights (Wy):", bi_rnn.Wy)
print("Forward bias (bhf):", bi_rnn.bhf)
print("Backward bias (bhb):", bi_rnn.bhb)
print("Output bias (by):", bi_rnn.by)

# Assign custom bias for forward direction to simulate training
bi_rnn.bhf = np.random.randn(1, 15)

# Create dummy inputs
prev_h_forward = np.random.randn(8, 15)
current_input = np.random.randn(8, 10)

# Perform forward step
forward_output = bi_rnn.forward(prev_h_forward, current_input)

# Output the shape and values of the new hidden state
print("Shape of forward hidden state:", forward_output.shape)
print(forward_output)
