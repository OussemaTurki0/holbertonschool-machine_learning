#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
deep_rnn = __import__('4-deep_rnn').deep_rnn

# Ensure reproducibility
np.random.seed(1)

# Create a stacked RNN with 3 layers
layer1 = RNNCell(i=10, h=15, o=1)
layer2 = RNNCell(i=15, h=15, o=1)
layer3 = RNNCell(i=15, h=15, o=5)

layers = [layer1, layer2, layer3]

# Assign random biases to hidden layers
for layer in layers:
    layer.bh = np.random.randn(1, 15)
# Assign random output bias to the last layer
layer3.by = np.random.randn(1, 5)

# Generate input data (t=6 time steps, m=8 batch size, i=10 input features)
input_data = np.random.randn(6, 8, 10)

# Initialize hidden states for each layer (l=3 layers, m=8, h=15)
initial_hidden_states = np.zeros((3, 8, 15))

# Run forward propagation through deep RNN
hidden_states, output = deep_rnn(layers, input_data, initial_hidden_states)

# Output results
print("All hidden states shape:", hidden_states.shape)
print(hidden_states)
print("Final output shape:", output.shape)
print(output)
