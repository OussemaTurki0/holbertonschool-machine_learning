#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def preprocess_data(X, Y):
    """Normalize and preprocess CIFAR-10 data."""
    X_processed = tf.keras.applications.densenet.preprocess_input(X.astype('float32'))
    Y_processed = tf.keras.utils.to_categorical(Y, 10)
    return X_processed, Y_processed

def build_cnn():
    """Construct the transfer learning model using DenseNet121."""
    base_cnn = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_cnn.trainable = False 

    inputs = tf.keras.Input(shape=(32, 32, 3))
    resized = tf.keras.layers.Resizing(224, 224)(inputs)
    features = base_cnn(resized, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    dense = tf.keras.layers.Dense(512, activation='relu')(pooled)
    dropout = tf.keras.layers.Dropout(0.4)(dense)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(dropout)

    model = tf.keras.Model(inputs, outputs)
    return model

def run_training():
    """Load data, train the model, and save it."""
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

    # ✅ Preprocess data using the correct function
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Build and compile model
    model = build_cnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train_p, Y_train_p, epochs=12, batch_size=128, validation_data=(X_test_p, Y_test_p), verbose=1)

    # Save trained model
    model.save('cifar10.h5')

if __name__ == '__main__':
    run_training()
