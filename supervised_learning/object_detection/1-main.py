import unittest
import numpy as np
import tensorflow.keras as K
from 1-yolo import Yolo

class TestYolo(unittest.TestCase):
    """
    Test case for the Yolo class.
    """

    def setUp(self):
        """
        Set up test parameters.
        """
        # Dummy paths for the test
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

    def test_load_class_names(self):
        """
        Test the loading of class names from file.
        """
        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)
        self.assertEqual(yolo.class_names, ["person", "car", "dog"])

    def test_process_outputs(self):
        """
        Test the process_outputs method of the Yolo class.
        """
        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)

        # Dummy outputs (this should be in the same shape as the model's output)
        outputs = [np.random.random((13, 13, 3, 85)), np.random.random((26, 26, 3, 85))]
        image_size = (416, 416)

        boxes, box_confidences, box_class_probs = yolo.process_outputs(outputs, image_size)

        # Check if boxes, confidences, and class_probs are returned as lists
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(box_confidences, list)
        self.assertIsInstance(box_class_probs, list)

        # Check if each element in the lists has the correct shape
        self.assertEqual(boxes[0].shape, (13, 13, 3, 4))  # Assuming box coordinates are (x_min, y_min, x_max, y_max)
        self.assertEqual(box_confidences[0].shape, (13, 13, 3, 1))
        self.assertEqual(box_class_probs[0].shape, (13, 13, 3, 80))  # Assuming 80 classes in total

    def tearDown(self):
        """
        Clean up test files.
        """
        import os
        os.remove(self.model_path)
        os.remove(self.classes_path)

if __name__ == "__main__":
    unittest.main()
