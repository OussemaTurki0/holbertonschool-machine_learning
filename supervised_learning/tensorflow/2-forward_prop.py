create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes, activations):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.Tensor): Input data placeholder.
        layer_sizes (list): List of number of nodes in each layer.
        activations (list): List of activation functions for each layer.

    Returns:
        tf.Tensor: Prediction of the network.
    """
    for i in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[i], activations[i])
    return x
