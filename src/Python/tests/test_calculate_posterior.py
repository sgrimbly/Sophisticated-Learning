# test_calculate_posterior.py

import unittest
import numpy as np
import sys
import os

# Add directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from active_inference_utils import calculate_posterior
from math_utils import normalise_matrix_columns

class TestCalculatePosterior(unittest.TestCase):
    
    def test_start(self):
        # Define the posterior Q
        Q = [np.zeros(100), 
             np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q[0][50] = 1 # starting position
        
        # Define the observations O
        O = [np.zeros(100), 
             np.array([1, 0, 0, 0]), # no resource observed
             np.array([0, 0, 0, 0, 1])] # no context observed
        O[0][50] = 1 # starting position
        
        # Define the concentration parameters y
        y = [np.zeros((100, 100, 4)),  # 100x100x4
             np.zeros((4, 100, 4)),    # 4x100x4
             np.zeros((5, 100, 4))]     # 5x100x4

        # Set hill locations for y[2]
        y[2][4, :, :] = 1
        for i in range(4):
            y[2][i, 54, i] = 1
            y[2][4, 54, i] = 0
        
        y[1] += .1
        y[1] = normalise_matrix_columns(y[1])
        
        # Value calculated from MATLAB code
        expected_P = [np.zeros(100), 
                      np.array([[0.25, 0.25, 0.25, 0.25]])]  # uniform belief about context
        expected_P[0][50] = 1 # starting position

        # Call the calculate_posterior function
        calculated_P = calculate_posterior(Q,y,O)

        # Assert that the expected and calculated values are close enough
        for array1, array2 in zip(expected_P, calculated_P):
            self.assertTrue(np.array_equal(array1, array2),
                            "Arrays in the lists are not equal")

    def test_hill(self):
        # Define the posterior Q
        Q = [np.zeros(100), 
             np.array([[0.25, 0.25, 0.25, 0.25]])]  # probability density of causes
        Q[0][54] = 1 # agent on hill
        
        # Define the observations O
        O = [np.zeros(100), 
             np.array([1, 0, 0, 0]), # no resource observed
             np.array([0, 0, 1, 0, 0])] # third context observed
        O[0][54] = 1 # agent on hill
        
        # Define the concentration parameters y
        y = [np.zeros((100, 100, 4)),  # 100x100x4
             np.zeros((4, 100, 4)),    # 4x100x4
             np.zeros((5, 100, 4))]     # 5x100x4

        # Set hill locations for y[2]
        y[2][4, :, :] = 1
        for i in range(4):
            y[2][i, 54, i] = 1
            y[2][4, 54, i] = 0
        
        # Set updated concentration parameters y[2], values from MATLAB when agent is at hill for first time
        y[1] += .1
        y[1] = normalise_matrix_columns(y[1])
        y[1][:,54,0] = [.4763, .1739, .1739, .1739]
        y[1][:,60,0] = [.4763, .1739, .1739, .1739]
        y[1][:,64,0] = [.4763, .1739, .1739, .1739]
        y[1][:,70,0] = [.6000, .1333, .1333, .1333]
        y[1][:,74,0] = [.4763, .1739, .1739, .1739]
        y[1][:,80,0] = [.4763, .1739, .1739, .1739]
        y[1][:,81,0] = [.4763, .1739, .1739, .1739]
        y[1][:,82,0] = [.4763, .1739, .1739, .1739]
        y[1][:,84,0] = [.4763, .1739, .1739, .1739]
        y[1][:,90,0] = [.4763, .1739, .1739, .1739]
        y[1][:,91,0] = [.4763, .1739, .1739, .1739]
        y[1][:,92,0] = [.4763, .1739, .1739, .1739]
        y[1][:,93,0] = [.4915, .1695, .1695, .1695]
        y[1][:,94,0] = [.4867, .1711, .1711, .1711]
        
        y[1][:,54,1] = [.4763, .1739, .1739, .1739]
        y[1][:,60,1] = [.4763, .1739, .1739, .1739]
        y[1][:,64,1] = [.5082, .1639, .1639, .1639]
        y[1][:,70,1] = [.6000, .1333, .1333, .1333]
        y[1][:,74,1] = [.5323, .1559, .1559, .1559]
        y[1][:,80,1] = [.4763, .1739, .1739, .1739]
        y[1][:,81,1] = [.4763, .1739, .1739, .1739]
        y[1][:,82,1] = [.4763, .1739, .1739, .1739]
        y[1][:,84,1] = [.5521, .1493, .1493, .1493]
        y[1][:,90,1] = [.4763, .1739, .1739, .1739]
        y[1][:,91,1] = [.4763, .1739, .1739, .1739]
        y[1][:,92,1] = [.4763, .1739, .1739, .1739]
        y[1][:,93,1] = [.5819, .1394, .1394, .1394]
        y[1][:,94,1] = [.5684, .1439, .1439, .1439]
        
        y[1][:,54,2] = [.4763, .1739, .1739, .1739]
        y[1][:,60,2] = [.4763, .1739, .1739, .1739]
        y[1][:,64,2] = [.7581, .0806, .0806, .0806]
        y[1][:,70,2] = [.6000, .1333, .1333, .1333]
        y[1][:,74,2] = [.7514, .0829, .0829, .0829]
        y[1][:,80,2] = [.4763, .1739, .1739, .1739]
        y[1][:,81,2] = [.4763, .1739, .1739, .1739]
        y[1][:,82,2] = [.4763, .1739, .1739, .1739]
        y[1][:,84,2] = [.7447, .0851, .0851, .0851]
        y[1][:,90,2] = [.4763, .1739, .1739, .1739]
        y[1][:,91,2] = [.4763, .1739, .1739, .1739]
        y[1][:,92,2] = [.4763, .1739, .1739, .1739]
        y[1][:,93,2] = [.7313, .0896, .0896, .0896]
        y[1][:,94,2] = [.7380, .0873, .0873, .0873]
        
        y[1][:,54,3] = [.4763, .1739, .1739, .1739]
        y[1][:,60,3] = [.4763, .1739, .1739, .1739]
        y[1][:,64,3] = [.4763, .1739, .1739, .1739]
        y[1][:,70,3] = [.6000, .1333, .1333, .1333]
        y[1][:,74,3] = [.4763, .1739, .1739, .1739]
        y[1][:,80,3] = [.4763, .1739, .1739, .1739]
        y[1][:,81,3] = [.4763, .1739, .1739, .1739]
        y[1][:,82,3] = [.4763, .1739, .1739, .1739]
        y[1][:,84,3] = [.4763, .1739, .1739, .1739]
        y[1][:,90,3] = [.4763, .1739, .1739, .1739]
        y[1][:,91,3] = [.4763, .1739, .1739, .1739]
        y[1][:,92,3] = [.4763, .1739, .1739, .1739]
        y[1][:,93,3] = [.4790, .1737, .1737, .1737]
        y[1][:,94,3] = [.4786, .1738, .1738, .1738]
                
        # Value calculated from MATLAB code
        expected_P = [np.zeros(100), 
                      np.array([[0, 0, 1, 0]])]  # agent believes its in 3rd context
        expected_P[0][54] = 1 # agent on hill

        # Call the calculate_posterior function
        calculated_P = calculate_posterior(Q,y,O)

        # Assert that the expected and calculated values are close enough
        for array1, array2 in zip(expected_P, calculated_P):
            self.assertTrue(np.array_equal(array1, array2),
                            "Arrays in the lists are not equal")

# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()