#!/usr/bin/env python3
"""
Test file for 0-yolo.py
"""

import unittest
import keras as K
import numpy as np
from 0-yolo import Yolo


class TestYolo(unittest.TestCase):
    """
    Test case for the Yolo class.
    """

    def setUp(self):
        """
        Set up test parameters.
        """
        self.model_path = "test_model.h5"
        self.classes_path = "test_classes.txt"
        self.class_t = 0.5
        self.nms_t = 0.3
        self.anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                                 [[30, 61], [62, 45], [59, 119]],
                                 [[10, 13], [16, 30], [33, 23]]])

        # Create a dummy model for testing
        model = K.models.Sequential()
        model.add(K.layers.Dense(1, input_shape=(1,)))
        model.save(self.model_path)

        # Create a dummy classes file
        with open(self.classes_path, "w") as f:
            f.write("person\ncar\ndog")

    def test_initialization(self):
        """
        Test if the Yolo class initializes correctly.
        """
        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)

        self.assertIsInstance(yolo.model, K.models.Model)
        self.assertEqual(yolo.class_names, ["person", "car", "dog"])
        self.assertEqual(yolo.class_t, self.class_t)
        self.assertEqual(yolo.nms_t, self.nms_t)
        np.testing.assert_array_equal(yolo.anchors, self.anchors)

    def tearDown(self):
        """
        Clean up test files.
        """
        import os
        os.remove(self.model_path)
        os.remove(self.classes_path)


if __name__ == "__main__":
    unittest.main()
