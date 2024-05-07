import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from math_utils import kl_divergence

class TestKLDivergence(unittest.TestCase):
    def test_kl_divergence(self):
        # Target 2D MATLAB matrix 1 = [1, 2, 3; 4, 5, 6; 7, 8, 9];
        # Target 2D MATLAB matrix 2 = [9, 8, 7; 6, 5, 4; 3, 2, 1];
        x = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        y = np.array([[9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]])
        
        # Expected kldr from MATLAB, having normalised and unraveled the above matrices
        expected_kldir = 0.6688

        # Calculated kldir
        calculated_kldir = kl_divergence(x, y)
        
        np.testing.assert_almost_equal(calculated_kldir, expected_kldir, decimal=4)
        
if __name__ == '__main__':
    unittest.main()