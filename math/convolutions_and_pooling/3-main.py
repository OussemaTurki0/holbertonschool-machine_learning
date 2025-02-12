import unittest
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale

class TestConvolveGrayscale(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.images = np.array([
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]]
        ])  # Shape (1, 4, 4)

        self.kernel = np.array([
            [1, 0],
            [0, -1]
        ])  # Shape (2, 2)

    def test_valid_padding(self):
        """Test convolution with 'valid' padding."""
        output = convolve_grayscale(self.images, self.kernel, padding='valid')
        expected_shape = (1, 3, 3)  # (m, h-kh+1, w-kw+1)
        self.assertEqual(output.shape, expected_shape)
    
    def test_same_padding(self):
        """Test convolution with 'same' padding."""
        output = convolve_grayscale(self.images, self.kernel, padding='same')
        expected_shape = (1, 4, 4)  # (m, h, w) since 'same' keeps dimensions
        self.assertEqual(output.shape, expected_shape)
    
    def test_custom_padding(self):
        """Test convolution with custom padding."""
        output = convolve_grayscale(self.images, self.kernel, padding=(1, 1))
        expected_shape = (1, 5, 5)  # (m, h+2ph-kh+1, w+2pw-kw+1)
        self.assertEqual(output.shape, expected_shape)
    
    def test_stride(self):
        """Test convolution with stride (2,2)."""
        output = convolve_grayscale(self.images, self.kernel, padding='valid', stride=(2, 2))
        expected_shape = (1, 2, 2)  # (m, (h-kh)//sh+1, (w-kw)//sw+1)
        self.assertEqual(output.shape, expected_shape)

if __name__ == "__main__":
    unittest.main()
