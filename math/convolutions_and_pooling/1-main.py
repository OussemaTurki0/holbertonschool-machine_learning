import unittest
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same

class TestConvolveGrayscaleSame(unittest.TestCase):
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

    def test_output_shape(self):
        """Test if the output shape matches the input shape."""
        output = convolve_grayscale_same(self.images, self.kernel)
        self.assertEqual(output.shape, self.images.shape)

    def test_correctness(self):
        """Test if the convolution produces the expected values."""
        output = convolve_grayscale_same(self.images, self.kernel)
        expected_output = np.array([
            [[1,  2,  3],
             [4,  0,  6],
             [7,  8, -9]]
        ])  # Manually computed result
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_multiple_images(self):
        """Test convolution with multiple grayscale images."""
        images = np.array([
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[9, 8, 7],
             [6, 5, 4],
             [3, 2, 1]]
        ])  # Shape (2, 3, 3)
        output = convolve_grayscale_same(images, self.kernel)
        self.assertEqual(output.shape, images.shape)

if __name__ == "__main__":
    unittest.main()
