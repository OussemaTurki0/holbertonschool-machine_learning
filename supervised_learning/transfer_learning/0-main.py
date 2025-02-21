import unittest
import numpy as np
import tensorflow as tf
preprocess_data = __import__('0-transfer').preprocess_data
class TestModel(unittest.TestCase):

    def test_preprocess_data(self):
        """Test the preprocess_data function."""
        # Create dummy CIFAR-10 data (32x32x3 images, labels)
        X_dummy = np.random.randint(0, 255, (10, 32, 32, 3))
        Y_dummy = np.random.randint(0, 10, (10, 1))

        X_processed, Y_processed = preprocess_data(X_dummy, Y_dummy)

        # Check shapes of processed data
        self.assertEqual(X_processed.shape, (10, 32, 32, 3))  # Same as input since resizing happens in the model
        self.assertEqual(Y_processed.shape, (10, 10))  # One-hot encoded labels for 10 classes

    def test_build_cnn(self):
        """Test the build_cnn function."""
        model = build_cnn()

        # Check that the model has the expected layers
        self.assertEqual(len(model.layers), 10)  # Number of layers in the model should be 10
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Dense) for layer in model.layers))  # Dense layer should exist
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Conv2D) for layer in model.layers))  # Conv2D layer should exist
        self.assertTrue(any(isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) for layer in model.layers))  # Pooling layer should exist

    def test_model_compile(self):
        """Test that the model compiles without errors."""
        model = build_cnn()
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Check if the model is compiled successfully
        self.assertTrue(model._is_compiled)

    def test_run_training(self):
        """Test that the training process can run without errors (without actually training)."""
        # Mock data
        X_train = np.random.rand(10, 32, 32, 3)
        Y_train = np.random.randint(0, 10, (10, 1))
        X_test = np.random.rand(10, 32, 32, 3)
        Y_test = np.random.randint(0, 10, (10, 1))

        # Preprocess the data
        X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
        X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

        # Build the model
        model = build_cnn()
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Run the training step (but don't actually train to avoid long waits)
        model.fit(X_train_p, Y_train_p, epochs=1, batch_size=2, validation_data=(X_test_p, Y_test_p), verbose=0)

        # Check that the model trained without errors
        self.assertTrue(model.history.history['accuracy'])

if __name__ == '__main__':
    unittest.main()
