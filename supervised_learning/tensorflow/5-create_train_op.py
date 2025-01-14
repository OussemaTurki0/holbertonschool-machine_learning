def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Args:
        loss (tf.Tensor): Loss of the network's prediction.
        alpha (float): Learning rate.

    Returns:
        tf.Operation: Training operation.
    """
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train_op
