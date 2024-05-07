import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from math_utils import normalise_matrix_columns, normalise_vector

class TestMatrixNormalisation(unittest.TestCase):
    def test_normalise_matrix_columns_2D(self):
        
        # Test logic of normalise_matrix_columns, which is normalise_matrix() in MATLAB

        # Target 2D MATLAB matrix = [1, 2, 3; 4, 5, 6; 7, 8, 9];

        m = np.array([  [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        
        # Expected matrix from MATLAB
        expected_normalised_m = np.array([  [0.0833, 0.1333, 0.1667],
                                            [0.3333, 0.3333, 0.3333],
                                            [0.5833, 0.5333, 0.5000]])

        # Calculated normalised matrix
        calculated_normalised_m = normalise_matrix_columns(m)
        
        np.testing.assert_array_almost_equal(calculated_normalised_m, expected_normalised_m, decimal=4)
        
    def test_normalise_matrix_columns_3D(self):
        
        # Test logic of normalise_matrix_columns, which is normalise_matrix() in MATLAB

        # Target 3D MATLAB matrix = cat(3, [1, 2, 3; 4, 5, 6; 7, 8, 9], [10, 11, 12; 13, 14, 15; 16, 17, 18]);

        m = np.array([[[1, 10], [2, 11], [3, 12]],
                      [[4, 13], [5, 14], [6, 15]],
                      [[7, 16], [8, 17], [9, 18]]])
        # Expected matrix from MATLAB
        expected_normalised_m = np.array([  [[0.0833, 0.2564], [0.1333, 0.2619], [0.1667, 0.2667]],
                                            [[0.3333, 0.3333], [0.3333, 0.3333], [0.3333, 0.3333]],
                                            [[0.5833, 0.4103], [0.5333, 0.4048], [0.5000, 0.4000]]])
        # Calculated normalised matrix
        calculated_normalised_m = normalise_matrix_columns(m)
        # Check whether entire arrays are equal
        np.testing.assert_array_almost_equal(calculated_normalised_m, expected_normalised_m, decimal=4)
        
        # Expected sub array from m(:,2,1) in MATLAB
        expected_sub_array = np.array((.1333, .3333, .5333))
        # Actual value of y 
        calculated_sub_array = calculated_normalised_m[:, 1, 0]
        # Check whether sub arrays are equal
        np.testing.assert_array_almost_equal(expected_sub_array, calculated_sub_array, decimal=4)

    def test_normalise_vector(self):
    
        # Test logic of normalise_and_sum_rows, which is normalise() in MATLAB

        # Target MATLAB vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Expected matrix from MATLAB
        expected_normalised_v = np.array([0.0182, 0.0364, 0.0545, 0.0727, 0.0909, 
                                          0.1091, 0.1273, 0.1455, 0.1636, 0.1818])

        # Calculated normalised matrix
        calculated_normalised_v = normalise_vector(v)
        
        np.testing.assert_array_almost_equal(calculated_normalised_v, expected_normalised_v, decimal=4)
        
        # Test vector of zeros
        # Target MATLAB vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        v = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        expected_normalised_v = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        calculated_normalised_v = normalise_vector(v)
        
        np.testing.assert_array_almost_equal(calculated_normalised_v, expected_normalised_v, decimal=4)
            
if __name__ == '__main__':
    unittest.main()