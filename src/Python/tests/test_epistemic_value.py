# test_epistemic_value.py

import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from active_inference_utils import G_epistemic_value
from math_utils import normalise_matrix_columns

class TestEpistemicValue(unittest.TestCase):
    
    def test_small_arrays(self):
        # Define the likelihood array A and the probability density s
        A = [np.array([[0.2, 0.5, 0.3], 
                       [0.5, 0.3, 0.2], 
                       [0.3, 0.2, 0.5]]),  # likelihoods for the first outcome
             np.array([[0.3, 0.2, 0.5], 
                       [0.2, 0.5, 0.3], 
                       [0.5, 0.3, 0.2]]),  # likelihoods for the second outcome
             np.array([[0.5, 0.3, 0.2], 
                       [0.3, 0.2, 0.5], 
                       [0.2, 0.5, 0.3]])]  # likelihoods for the third outcome
        s = np.array([0.3, 0.3, 0.4])  # probability density of causes

        # Value calculated from MATLAB code
        expected_G = 0.1909  

        # Call the G_epistemic_value function
        calculated_G = G_epistemic_value(A, s)

        # Assert that the expected and calculated values are close enough
        self.assertAlmostEqual(expected_G, calculated_G, places=4)
    
    def test_large_arrays(self):
        # Define the likelihood array A and the probability density s
        A = [
            np.zeros((100, 100, 4)),  # 100x100x4
            np.zeros((4, 100, 4)),    # 4x100x4
            np.zeros((5, 100, 4))     # 5x100x4
        ]

        for i in range(100):
            A[0][i,i,:] = 1

        # Directly set the values for A[1]
        A[1] += .1
        A[1] = normalise_matrix_columns(A[1])

        # Directly set the values for A[2]
        A[2][4, :, :] = 1
        for i in range(4):
            A[2][i, 54, i] = 1
            A[2][4, 54, i] = 0
        
        # Define s
        s = [np.zeros(100), 
             np.array([0.25, 0.25, 0.25, 0.25])]  # probability density of causes
        s[0][51] = 1 # agent having moved once since start

        # Value calculated from MATLAB code
        expected_G = 0 

        # Call the G_epistemic_value function
        calculated_G = G_epistemic_value(A, s)

        # Assert that the expected and calculated values are close enough
        self.assertAlmostEqual(expected_G, calculated_G, places=4)


# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()
