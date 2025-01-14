def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (tf.Tensor): Placeholder for the labels.
        y_pred (tf.Tensor): Tensor containing network predictions.

    Returns:
        tf.Tensor: Accuracy tensor.
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
