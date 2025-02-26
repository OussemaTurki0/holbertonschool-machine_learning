import unittest
import keras as K
import numpy as np
import importlib.util
import sys

# Specify the path to your module file
module_name = "yolo_2"
file_path = "./2-yolo.py"

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
        Test if the process_outputs method returns expected values.
        """
        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)
        
        # Dummy output from the model
        dummy_output = np.random.rand(1, 13, 13, 3, 85)  # Example dimensions: (grid_height, grid_width, anchor_boxes, values)
        
        # Expected output dimensions
        image_size = (416, 416)
        
        boxes, confidences, class_probs = yolo.process_outputs([dummy_output], image_size)

        # Check if the processed outputs are of the expected shapes
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(confidences), 1)
        self.assertEqual(len(class_probs), 1)

    def test_filter_boxes(self):
        """
        Test if the filter_boxes method returns expected filtered boxes.
        """
        yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)
        
        # Dummy outputs from process_outputs method
        boxes = [np.random.rand(13, 13, 3, 4)]
        box_confidences = [np.random.rand(13, 13, 3, 1)]
        box_class_probs = [np.random.rand(13, 13, 3, 80)]  # Assuming 80 classes

        filtered_boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)

        # Check if the filtered boxes are non-empty
        self.assertGreater(len(filtered_boxes), 0)
        self.assertGreater(len(box_classes), 0)
        self.assertGreater(len(box_scores), 0)

    def tearDown(self):
        """
        Clean up test files.
        """
        import os
        os.remove(self.model_path)
        os.remove(self.classes_path)


if __name__ == "__main__":
    unittest.main()
