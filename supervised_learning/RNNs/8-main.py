#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output').BidirectionalCell
bi_rnn = __import__('8-bi_rnn').bi_rnn

# Set a consistent seed for reproducible results
np.random.seed(8)

# Initialize a bidirectional RNN cell with input dim 10, hidden dim 15, and output dim 5
bidir_cell = BidirectionalCell(i=10, h=15, o=5)

# Simulate a batch of input sequences: (time_steps=6, batch_size=8, input_size=10)
sequence_batch = np.random.randn(6, 8, 10)

# Set the initial hidden states for both directions to zeros
initial_forward_state = np.zeros((8, 15))
initial_backward_state = np.zeros((8, 15))

# Run the bidirectional RNN over the input sequence
hidden_states, predictions = bi_rnn(
    bidir_cell, sequence_batch, initial_forward_state, initial_backward_state
)

# Print the results: combined hidden states and output predictions
print("Shape of hidden states:", hidden_states.shape)
print(hidden_states)
print("Shape of predictions:", predictions.shape)
print(predictions)
