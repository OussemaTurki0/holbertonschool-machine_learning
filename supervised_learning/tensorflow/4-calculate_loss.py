def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss.

    Args:
        y (tf.Tensor): Placeholder for the labels.
        y_pred (tf.Tensor): Tensor containing network predictions.

    Returns:
        tf.Tensor: Loss tensor.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
