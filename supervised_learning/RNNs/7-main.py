#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output').BidirectionalCell

# Embark on a kaleidoscopic journey through a bidirectional RNN’s output realm
np.random.seed(7)

# Instantiate this intricate cell, ready to orchestrate transformations across time
bi_rnn_cell = BidirectionalCell(i=10, h=15, o=5)

# Infuse the output layer with some randomly conjured victuals
bi_rnn_cell.by = np.random.randn(1, 5)

# A verdant mosaic of hidden states, from past and future, intertwined into a 3D tapestry
hidden_states_combined = np.random.randn(6, 8, 30)  # (time_steps, batch_size, hidden_fwd + hidden_bwd)

# Let the cell transcend the raw data into meaningful outputs
output_predictions = bi_rnn_cell.output(hidden_states_combined)

# Reveal the shape and intricate content of this transformed layer
print("Output shape beckons:", output_predictions.shape)
print(output_predictions)
