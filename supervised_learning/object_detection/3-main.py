import unittest
import keras as K
import numpy as np
import importlib.util
import sys

# Specify the path to your module file
module_name = "yolo_0"
file_path = "./3-yolo.py"

# Load the module dynamically
spec = importlib.util.spec_from_file_location(module_name, file_path)
yolo_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = yolo_module
spec.loader.exec_module(yolo_module)

# Now you can access the Yolo class
Yolo = yolo_module.Yolo


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

    def test_process_outputs(self):
        """
        Test the processing of outputs from the Darknet model.
        """
        dummy_output = np.random.random((13, 13, 3, 85))  # Example output shape [grid_height, grid_width, anchor_boxes, output_size]
        outputs = [dummy_output]
        image_size = (416, 416)  # Example image size

        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)
        boxes, confidences, class_probs = yolo.process_outputs(outputs, image_size)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(confidences), 1)
        self.assertEqual(len(class_probs), 1)

    def test_filter_boxes(self):
        """
        Test filtering of boxes based on confidence scores and class probabilities.
        """
        dummy_boxes = np.random.random((10, 4))  # 10 random boxes
        dummy_confidences = np.random.random((10, 1))  # 10 random confidences
        dummy_class_probs = np.random.random((10, 3))  # 10 random class probabilities (3 classes)

        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)
        filtered_boxes, box_classes, box_scores = yolo.filter_boxes(dummy_boxes, dummy_confidences, dummy_class_probs)

        self.assertTrue(filtered_boxes.shape[0] <= 10)
        self.assertTrue(box_classes.shape[0] <= 10)
        self.assertTrue(box_scores.shape[0] <= 10)

    def test_non_max_suppression(self):
        """
        Test non-max suppression.
        """
        dummy_boxes = np.random.random((10, 4))  # 10 random boxes
        dummy_classes = np.random.randint(0, 3, 10)  # Random class labels
        dummy_scores = np.random.random(10)  # Random scores

        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)
        nms_boxes, nms_classes, nms_scores = yolo.non_max_suppression(dummy_boxes, dummy_classes, dummy_scores)

        self.assertEqual(len(nms_boxes), len(nms_classes))
        self.assertEqual(len(nms_classes), len(nms_scores))

    def tearDown(self):
        """
        Clean up test files.
        """
        import os
        os.remove(self.model_path)
        os.remove(self.classes_path)


if __name__ == "__main__":
    unittest.main()
