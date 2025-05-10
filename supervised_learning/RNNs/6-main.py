#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('6-bi_backward').BidirectionalCell

# Ensure reproducible results
np.random.seed(6)

# Instantiate the BidirectionalCell with input size 10, hidden size 15, and output size 5
bi_rnn = BidirectionalCell(i=10, h=15, o=5)

# Set a custom bias for the backward pass
bi_rnn.bhb = np.random.randn(1, 15)

# Generate sample hidden state and input vector
next_hidden_state = np.random.randn(8, 15)
input_at_timestep = np.random.randn(8, 10)

# Perform the backward computation
backward_hidden = bi_rnn.backward(next_hidden_state, input_at_timestep)

# Print the shape and content of the hidden state
print("Shape of backward hidden state:", backward_hidden.shape)
print(backward_hidden)
