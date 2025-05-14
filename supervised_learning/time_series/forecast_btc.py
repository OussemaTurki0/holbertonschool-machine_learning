#!/usr/bin/env python3
"""Bitcoin Price Forecasting using SimpleRNN"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_forecasting_model(input_shape):
    """
    Builds and compiles a SimpleRNN-based model for BTC price prediction.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.SimpleRNN(64, activation='tanh'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def display_loss_curves(history):
    """
    Displays and saves the loss curve for training and validation.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

def load_and_split_data():
    """
    Loads and splits data into training and validation sets.
    """
    X = np.load("X.npy")
    y = np.load("y.npy")
    split = int(len(X) * 0.75)
    return (X[:split], y[:split], X[split:], y[split:])

def prepare_datasets(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Converts data into TensorFlow datasets.
    """
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    return train_ds, val_ds

def main():
    # Load data
    X_train, y_train, X_val, y_val = load_and_split_data()
    input_shape = (X_train.shape[1], 1)

    # Create datasets
    train_data, val_data = prepare_datasets(X_train, y_train, X_val, y_val)

    # Build and train model
    model = build_forecasting_model(input_shape)
    history = model.fit(train_data, validation_data=val_data, epochs=10)

    # Save results
    model.save("btc_rnn_forecast_model.h5")
    display_loss_curves(history)

if __name__ == "__main__":
    main()
