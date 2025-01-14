def create_layer(prev, n, activation):
    """
    Creates a layer for the neural network.

    Args:
        prev (tf.Tensor): Output of the previous layer.
        n (int): Number of nodes in the layer.
        activation (callable): Activation function to use.

    Returns:
        tf.Tensor: The layer output.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation, 
                            kernel_initializer=initializer, name="layer")
    return layer(prev)
