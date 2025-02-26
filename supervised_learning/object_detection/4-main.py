import unittest
import numpy as np
from 4-yolo import Yolo

class TestYolo(unittest.TestCase):
    def setUp(self):
        """
        Set up test variables before each test
        """
        # Sample file paths and parameters
        self.model_path = 'path_to_model.h5'  # Update with the actual model path
        self.classes_path = 'path_to_classes.txt'  # Update with the actual class file path
        self.class_t = 0.5
        self.nms_t = 0.4
        self.anchors = np.array([
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]]
        ])

        self.yolo = Yolo(self.model_path, self.classes_path, self.class_t, self.nms_t, self.anchors)

    def test_load_class_names(self):
        """
        Test the class names loading from the file
        """
        class_names = self.yolo._load_class_names(self.classes_path)
        self.assertIsInstance(class_names, list)
        self.assertGreater(len(class_names), 0)

    def test_process_outputs(self):
        """
        Test the processing of outputs from the model
        """
        # Fake outputs for testing purposes
        fake_output = np.random.rand(13, 13, 3, 85)  # Example shape (grid_size, grid_size, num_anchors, output_values)
        outputs = [fake_output]  # A list of outputs
        image_size = (416, 416)  # Example image size

        boxes, box_confidences, box_class_probs = self.yolo.process_outputs(outputs, image_size)

        self.assertIsInstance(boxes, list)
        self.assertIsInstance(box_confidences, list)
        self.assertIsInstance(box_class_probs, list)

    def test_filter_boxes(self):
        """
        Test filtering of boxes based on class score
        """
        fake_boxes = np.random.rand(10, 4)
        fake_confidences = [np.random.rand(10, 1)]
        fake_class_probs = [np.random.rand(10, 80)]  # 80 classes

        filtered_boxes, box_classes, box_scores = self.yolo.filter_boxes(fake_boxes, fake_confidences, fake_class_probs)

        self.assertIsInstance(filtered_boxes, np.ndarray)
        self.assertIsInstance(box_classes, np.ndarray)
        self.assertIsInstance(box_scores, np.ndarray)

    def test_non_max_suppression(self):
        """
        Test non-max suppression for filtering out overlapping boxes
        """
        fake_filtered_boxes = np.random.rand(10, 4)
        fake_box_classes = np.random.randint(0, 2, size=10)  # Two classes
        fake_box_scores = np.random.rand(10)

        nms_boxes, nms_classes, nms_scores = self.yolo.non_max_suppression(fake_filtered_boxes, fake_box_classes, fake_box_scores)

        self.assertIsInstance(nms_boxes, np.ndarray)
        self.assertIsInstance(nms_classes, np.ndarray)
        self.assertIsInstance(nms_scores, np.ndarray)

    def test_load_images(self):
        """
        Test loading images from a folder
        """
        folder_path = 'path_to_images'  # Update with an actual folder path
        images, image_paths = self.yolo.load_images(folder_path)

        self.assertIsInstance(images, list)
        self.assertIsInstance(image_paths, list)
        self.assertGreater(len(images), 0)

if __name__ == '__main__':
    unittest.main()
