import unittest
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
class TestConvolveGrayscaleValid(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.images = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        ])  # Shape (1, 3, 3)

        self.kernel = np.array([
            [1, 0],
            [0, -1]
        ])  # Shape (2, 2)

    def test_convolution_output_shape(self):
        """Test if the output shape is correct."""
        output = convolve_grayscale_valid(self.images, self.kernel)
        expected_shape = (1, 2, 2)  # (m, h-kh+1, w-kw+1)
        self.assertEqual(output.shape, expected_shape)

    def test_convolution_correctness(self):
        """Test if convolution is computed correctly."""
        output = convolve_grayscale_valid(self.images, self.kernel)
        expected_output = np.array([
            [[-3, -3],
             [-3, -3]]
        ])  # Manually computed result
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_multiple_images(self):
        """Test with multiple images."""
        images = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[9, 8, 7],
             [6, 5, 4],
             [3, 2, 1]]
        ])  # Shape (2, 3, 3)
        output = convolve_grayscale_valid(images, self.kernel)
        expected_output = np.array([
            [[-3, -3],
             [-3, -3]],
            [[3, 3],
             [3, 3]]
        ])
        np.testing.assert_array_almost_equal(output, expected_output)

if __name__ == "__main__":
    unittest.main()
