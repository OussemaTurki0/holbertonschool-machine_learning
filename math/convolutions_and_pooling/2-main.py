import unittest
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding

class TestConvolveGrayscalePadding(unittest.TestCase):
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

        self.padding = (1, 1)  # Custom padding (ph, pw)

    def test_output_shape(self):
        """Test if the output shape is correct with padding."""
        output = convolve_grayscale_padding(self.images, self.kernel, self.padding)
        expected_shape = (1, 4, 4)  # (m, h+2ph-kh+1, w+2pw-kw+1)
        self.assertEqual(output.shape, expected_shape)

    def test_correctness(self):
        """Test if the convolution produces the expected values."""
        output = convolve_grayscale_padding(self.images, self.kernel, self.padding)
        expected_output = np.array([
            [[  0,   1,   2,   0],
             [  1,   2,   3,  -2],
             [  4,   0,   6,  -5],
             [  0,  -7,  -8,   0]]
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
        output = convolve_grayscale_padding(images, self.kernel, self.padding)
        self.assertEqual(output.shape, (2, 4, 4))

if __name__ == "__main__":
    unittest.main()
